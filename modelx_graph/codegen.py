from __future__ import annotations

import ast
import math
from pathlib import Path

from .frontend import CanonicalModel
from .template_ir import TemplateError
from .codegen_common import CodegenError
from .cython_expression import emit_cython_expr
from .optimized_program import OptimizedProgram
from .program_semantics import guard_code_from_stmt


def _offset(node: ast.AST, var: str) -> int | None:
    if isinstance(node, ast.Name) and node.id == var:
        return 0
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.left, ast.Name)
        and node.left.id == var
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, int)
    ):
        if isinstance(node.op, ast.Sub):
            return -int(node.right.value)
        if isinstance(node.op, ast.Add):
            return int(node.right.value)
    return None


class CythonGenerator:
    """Lower the executable graph-template IR to Cython.

    Python and Cython consume the same :class:`ExecutableGraph`; optimization
    levels are storage/IR passes over
    that schedule rather than separate compiler frontends.
    """

    def __init__(
        self,
        model: CanonicalModel | OptimizedProgram,
        *,
        optimization_level: str | None = None,
        full_array: bool | None = None,
        emission_mode: str = "array_abi",
    ):
        prepared = model if isinstance(model, OptimizedProgram) else None
        if prepared is not None:
            if optimization_level is not None and optimization_level.upper() != prepared.optimization_level:
                raise CodegenError("optimized program and requested optimization level differ")
            if full_array is not None and bool(full_array) != prepared.full_array:
                raise CodegenError("optimized program and requested full_array policy differ")
            self.m = prepared.canonical
            self.v = prepared.variants
            self.schedule = prepared.schedule
            self.layout = prepared.layout
            self.optimization_level = prepared.optimization_level
            self.full_array = prepared.full_array
            self.build_fingerprint = prepared.build_fingerprint
            self.input_order = list(prepared.input_order)
            self.input_specs = {spec.key: spec for spec in prepared.inputs}
            self.pre_static_order = list(prepared.pre_static_order)
            self.post_static_order = {i: list(rows) for i, rows in prepared.post_static_order}
            self.coordinate_step = prepared.coordinate_step
        else:
            self.m = model
            self.v = model.variants
            self.schedule = model.executable
            if optimization_level is None:
                # O2 is the measured default: it keeps O1 liveness/ring storage and
                # fuses only provably order-preserving reductions.
                optimization_level = "O0" if full_array else "O2"
            level = optimization_level.upper()
            if full_array is not None and bool(full_array) != (level == "O0"):
                raise CodegenError("full_array and optimization_level request conflicting storage policies")
            try:
                self.layout = self.schedule.storage_plan(level)
            except TemplateError as exc:
                raise CodegenError(str(exc)) from exc
            self.optimization_level = level
            self.full_array = level == "O0"
            self.build_fingerprint = self.schedule.build_fingerprint(level)
            self.input_order = list(self.m.inputs)
            self.input_specs = self.m.inputs
            self.pre_static_order = list(self.schedule.pre_scalars)
            self.post_static_order = {b.index: list(b.post_scalars) for b in self.schedule.loops}
            steps = {int(b.coordinate_step) for b in self.schedule.loops}
            if len(steps) != 1:
                raise CodegenError("current native backend requires executable loops to share one signed coordinate step")
            self.coordinate_step = next(iter(steps))
        self.guards = tuple(prepared.guards) if prepared is not None else tuple(
            sorted((g for cv in self.v.values() for g in cv.guards), key=lambda g: g.code)
        )
        self.validated_static_facts = tuple(prepared.validated_static_facts) if prepared is not None else tuple(
            getattr(self.m, "validated_static_facts", ())
        )
        self.has_missing_intrinsic = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__is_missing__"
            for cv in self.v.values()
            for tree in ((cv.function,) if cv.function is not None else ())
            for node in ast.walk(tree)
        ) or any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__is_missing__"
            for cv in self.v.values() if cv.reduction is not None
            for tree in (cv.reduction.init, cv.reduction.body_expr, *cv.reduction.range_args, *cv.reduction.filters)
            for node in ast.walk(tree)
        )
        self.has_erf_intrinsic = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__erf__"
            for cv in self.v.values()
            for tree in ((cv.function,) if cv.function is not None else ())
            for node in ast.walk(tree)
        )
        self.has_isinf_intrinsic = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__is_inf__"
            for cv in self.v.values()
            for tree in ((cv.function,) if cv.function is not None else ())
            for node in ast.walk(tree)
        )
        self.has_infinity_constant = any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, float)
            and math.isinf(node.value)
            for cv in self.v.values()
            for tree in ((cv.function,) if cv.function is not None else ())
            for node in ast.walk(tree)
        ) or any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, float)
            and math.isinf(node.value)
            for cv in self.v.values() if cv.reduction is not None
            for tree in (cv.reduction.init, cv.reduction.body_expr, *cv.reduction.range_args, *cv.reduction.filters)
            for node in ast.walk(tree)
        )
        self.has_step_lookup_intrinsic = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__step_lookup_1d__"
            for cv in self.v.values()
            for tree in ((cv.function,) if cv.function is not None else ())
            for node in ast.walk(tree)
        ) or any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__step_lookup_1d__"
            for cv in self.v.values() if cv.reduction is not None
            for tree in (cv.reduction.init, cv.reduction.body_expr, *cv.reduction.range_args, *cv.reduction.filters)
            for node in ast.walk(tree)
        )
        self.has_interval_lookup_intrinsic = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__interval_lookup_1d__"
            for cv in self.v.values()
            for tree in ((cv.function,) if cv.function is not None else ())
            for node in ast.walk(tree)
        ) or any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__interval_lookup_1d__"
            for cv in self.v.values() if cv.reduction is not None
            for tree in (cv.reduction.init, cv.reduction.body_expr, *cv.reduction.range_args, *cv.reduction.filters)
            for node in ast.walk(tree)
        )
        self.has_interp_lookup_intrinsic = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__interp_lookup_1d__"
            for cv in self.v.values()
            for tree in ((cv.function,) if cv.function is not None else ())
            for node in ast.walk(tree)
        ) or any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "__interp_lookup_1d__"
            for cv in self.v.values() if cv.reduction is not None
            for tree in (cv.reduction.init, cv.reduction.body_expr, *cv.reduction.range_args, *cv.reduction.filters)
            for node in ast.walk(tree)
        )
        self.guard_map = {g.code: g for g in self.guards}
        self.has_guards = bool(self.guards)
        mode = str(emission_mode).lower()
        if mode not in {"array_abi", "direct_locals"}:
            raise CodegenError(f"unsupported Cython emission_mode {emission_mode!r}")
        self.emission_mode = mode
        self.direct_prev_uids: frozenset[str] = frozenset()
        if mode == "direct_locals":
            prev: set[str] = set()
            for uid, depth in self.layout.ring_depths.items():
                if depth != 2:
                    continue
                phase = self.schedule.coordinate_phase[uid]
                block = self.schedule.loops[phase]
                prior_shift = -int(block.scan_direction)
                uses = [u for u in self.schedule.uses if u.source == uid and u.context == "coordinate"]
                nonzero = [u for u in uses if u.offset not in (None, 0)]
                if nonzero and all(self._position_shift(int(u.offset)) == prior_shift for u in nonzero):
                    prev.add(uid)
            self.direct_prev_uids = frozenset(prev)

    def _math_cimport_line(self) -> str:
        names = "pow, fabs, rint, exp, log, sqrt, sin, cos, floor, ceil"
        if self.has_missing_intrinsic:
            names += ", isnan"
        if self.has_erf_intrinsic:
            names += ", erf"
        if self.has_isinf_intrinsic:
            names += ", isinf"
        if self.has_infinity_constant:
            names += ", INFINITY"
        return f"from libc.math cimport {names}, NAN"

    def _step_lookup_helper_lines(self) -> list[str]:
        if not self.has_step_lookup_intrinsic:
            return []
        return [
            "cdef inline double _step_lookup_1d(double[::1] axis, double[::1] values, double q, double default, long below_mode) noexcept nogil:",
            "    cdef Py_ssize_t lo = 0",
            "    cdef Py_ssize_t hi = axis.shape[0]",
            "    cdef Py_ssize_t mid",
            "    while lo < hi:",
            "        mid = (lo + hi) // 2",
            "        if axis[mid] <= q:",
            "            lo = mid + 1",
            "        else:",
            "            hi = mid",
            "    if lo == 0:",
            "        return values[0] if below_mode == 1 else default",
            "    return values[lo - 1]",
            "",
        ]

    def _interval_lookup_helper_lines(self) -> list[str]:
        if not self.has_interval_lookup_intrinsic:
            return []
        return [
            "cdef inline double _interval_lookup_1d(double[::1] lower, double[::1] upper, double[::1] values, double q, long closure_mode, long fallback_mode, double default) noexcept nogil:",
            "    cdef Py_ssize_t j",
            "    for j in range(values.shape[0]):",
            "        if (closure_mode == 1 and lower[j] <= q and q <= upper[j]) or (closure_mode == 0 and lower[j] < q and q <= upper[j]):",
            "            return values[j]",
            "    if fallback_mode == 0:",
            "        return values[values.shape[0] - 1]",
            "    if q > upper[upper.shape[0] - 1]:",
            "        return values[values.shape[0] - 1]",
            "    return default",
            "",
        ]

    def _interp_lookup_helper_lines(self) -> list[str]:
        if not self.has_interp_lookup_intrinsic:
            return []
        return [
            "cdef inline double _interp_lookup_1d(double[::1] axis, double[::1] values, double q, long mode) noexcept nogil:",
            "    cdef Py_ssize_t n = axis.shape[0]",
            "    cdef Py_ssize_t lo",
            "    cdef Py_ssize_t hi",
            "    cdef Py_ssize_t left",
            "    cdef Py_ssize_t right",
            "    cdef Py_ssize_t mid",
            "    cdef double frac",
            "    if q <= axis[0]:",
            "        lo = 0",
            "    elif q >= axis[n - 1]:",
            "        lo = n - 2",
            "    else:",
            "        left = 0",
            "        right = n",
            "        while left < right:",
            "            mid = (left + right) // 2",
            "            if axis[mid] <= q:",
            "                left = mid + 1",
            "            else:",
            "                right = mid",
            "        lo = left - 1",
            "    hi = lo + 1",
            "    frac = (q - axis[lo]) / (axis[hi] - axis[lo])",
            "    if mode == 1:",
            "        return values[lo] * pow(values[hi] / values[lo], frac)",
            "    return values[lo] + (values[hi] - values[lo]) * frac",
            "",
        ]

    def _direct_current_name(self, uid: str) -> str:
        return f"c_{uid}"

    def _direct_scalar_name(self, uid: str) -> str:
        return f"s_{uid}"

    def _direct_history_name(self, uid: str) -> str:
        return f"h_{uid}"

    def _direct_ring_name(self, uid: str) -> str:
        return f"r_{uid}"

    def _direct_prev_name(self, uid: str) -> str:
        return f"p_{uid}"

    def _direct_derived_name(self, uid: str) -> str:
        return f"d_{uid}"

    def _direct_region_name(self, uid: str) -> str:
        return f"g_{uid}"

    def _direct_boundary_name(self, uid: str) -> str:
        return f"b_{uid}"

    def _scalar_expr(self, uid: str) -> str:
        if self.emission_mode == "direct_locals":
            return self._direct_scalar_name(uid)
        return f"svals[{self.layout.scalar_slots[uid]}]"

    def _current_expr(self, uid: str) -> str:
        if self.emission_mode == "direct_locals":
            return self._direct_current_name(uid)
        return f"cur[{self.layout.current_slots[uid]}]"

    def _derived_state_expr(self, uid: str) -> str:
        if self.emission_mode == "direct_locals":
            return self._direct_derived_name(uid)
        return f"state[{self.layout.derived_state_slots[uid]}]"

    def _derived_region_expr(self, uid: str) -> str:
        if uid in self.layout.derived_region_state_slots:
            if self.emission_mode == "direct_locals":
                return self._direct_region_name(uid)
            return f"state[{self.layout.derived_region_state_slots[uid]}]"
        if uid in self.layout.current_slots:
            return self._current_expr(uid)
        raise CodegenError(f"derived-region value {uid} has no storage slot")

    def _boundary_seed_expr(self, uid: str) -> str:
        if uid not in self.layout.boundary_seed_slots:
            raise CodegenError(f"boundary seed for {self.v[uid].source_fullname} was not allocated")
        if self.emission_mode == "direct_locals":
            return self._direct_boundary_name(uid)
        return f"state[{self.layout.boundary_seed_slots[uid]}]"

    def ctype(self, dtype: str) -> str:
        if dtype == "int64":
            return "long"
        if dtype == "bool":
            return "bint"
        return "double"

    def _cast_read(self, expr: str, dtype: str) -> str:
        if dtype == "int64":
            return f"(<long>({expr}))"
        if dtype == "bool":
            return f"(<bint>({expr}))"
        return expr

    def input_decl(self, key: str) -> str:
        s = self.input_specs[key]
        if s.ndim == 0:
            return f"{self.ctype(s.dtype)} {key}"
        if s.dtype == "int64" and s.ndim == 1:
            return f"long[::1] {key}"
        if s.dtype == "bool" and s.ndim == 1:
            return f"cnp.npy_bool[::1] {key}"
        if s.dtype == "float64" and s.ndim == 1:
            return f"double[::1] {key}"
        if s.ndim >= 2 and s.dtype in {"float64", "int64", "bool"}:
            base = "double" if s.dtype == "float64" else "long" if s.dtype == "int64" else "cnp.npy_bool"
            axes = [":"] * int(s.ndim)
            axes[-1] = "::1"
            return f"{base}[{', '.join(axes)}] {key}"
        raise CodegenError(f"unsupported input layout {key}: {s}")

    def _input_expr(self, name: str, args: list[ast.AST], ctx) -> str:
        if not args or not isinstance(args[0], ast.Constant):
            raise CodegenError(f"{name} requires internal literal input key")
        key = args[0].value
        if key not in self.input_specs:
            raise CodegenError(f"unknown input key {key}")
        spec = self.input_specs[key]
        if name == "point_input":
            if spec.ndim != 1:
                raise CodegenError(f"point input {key} must be 1-D")
            return self._cast_read(f"{key}[i]", spec.dtype)
        if name == "global_input":
            if spec.ndim != 0:
                raise CodegenError(f"global scalar {key} must be 0-D")
            return self._cast_read(key, spec.dtype)
        if name == "array_input":
            if len(args) != 2 or spec.ndim != 1:
                raise CodegenError(f"array input {key} requires one index")
            idx = self.expr(args[1], ctx)
            return self._cast_read(f"{key}[<Py_ssize_t>({idx})]", spec.dtype)
        if name == "table_input":
            if spec.ndim < 1 or len(args) != int(spec.ndim) + 1:
                raise CodegenError(f"table input {key} requires {int(spec.ndim)} indices")
            indices = [self.expr(arg, ctx) for arg in args[1:]]
            index_src = ", ".join(f"<Py_ssize_t>({idx})" for idx in indices)
            return self._cast_read(f"{key}[{index_src}]", spec.dtype)
        raise CodegenError(name)

    def _history_read(self, uid: str, index: str) -> str:
        if uid not in self.layout.history_slots:
            raise CodegenError(f"history for {self.v[uid].source_fullname} was not materialized")
        if self.emission_mode == "direct_locals":
            raw = f"{self._direct_history_name(uid)}[<Py_ssize_t>({index})]"
        else:
            slot = self.layout.history_slots[uid]
            raw = f"work[{slot} * T + <Py_ssize_t>({index})]"
        return self._cast_read(raw, self.v[uid].dtype)

    def _ring_read(self, uid: str, index: str) -> str:
        if uid not in self.layout.ring_bases:
            raise CodegenError(f"rolling state for {self.v[uid].source_fullname} was not allocated")
        depth = self.layout.ring_depths[uid]
        if self.emission_mode == "direct_locals":
            if uid in self.direct_prev_uids:
                return self._cast_read(self._direct_prev_name(uid), self.v[uid].dtype)
            raw = f"{self._direct_ring_name(uid)}[<Py_ssize_t>({index}) % {depth}]"
        else:
            base = self.layout.ring_bases[uid]
            raw = f"state[{base} + (<Py_ssize_t>({index}) % {depth})]"
        return self._cast_read(raw, self.v[uid].dtype)

    def _position_shift(self, coordinate_offset: int) -> int:
        step = int(self.coordinate_step)
        if coordinate_offset % step != 0:
            raise CodegenError(
                f"coordinate offset {coordinate_offset} is not aligned to executable range step {step}"
            )
        return coordinate_offset // step

    def _pure_scalar_arg_name(self, uid: str) -> str:
        return f"ps_{uid}"

    def _pure_map_call(self, uid: str, args: list[ast.AST], ctx) -> str:
        if len(args) != 1 or uid not in self.schedule.pure_maps:
            raise CodegenError(f"invalid PureMap call {uid}")
        qexpr = self.expr(args[0], ctx)
        arglist = [f"<long>({qexpr})", "i"]
        if self.has_guards:
            arglist.append("error_code")
        scalar_args = ctx.get("pure_scalar_args", {})
        for dep in self.schedule.pure_maps[uid].scalar_dependencies:
            if ctx.get("mode") == "pure_map":
                if dep not in scalar_args:
                    raise CodegenError(
                        f"PureMap {ctx.get('caller_uid')} lacks scalar dependency {dep} required by {uid}"
                    )
                arglist.append(scalar_args[dep])
            else:
                arglist.append(self._scalar_expr(dep))
        arglist += self.input_order
        return f"eval_{uid}({', '.join(arglist)})"

    def _cell_call(self, uid: str, args: list[ast.AST], ctx) -> str:
        callee = self.v[uid]
        role = self.schedule.roles.get(uid, callee.role)
        mode = ctx.get("mode", "formula")
        if role == "scalar":
            if args:
                raise CodegenError(f"scalar Cell {uid} unexpectedly has arguments")
            if mode == "pure_map":
                scalar_args = ctx.get("pure_scalar_args", {})
                if uid not in scalar_args:
                    raise CodegenError(
                        f"PureMap {ctx.get('caller_uid')} attempted undeclared scalar dependency {uid}"
                    )
                return self._cast_read(scalar_args[uid], callee.dtype)
            return self._cast_read(self._scalar_expr(uid), callee.dtype)
        if role == "reduction":
            if mode == "pure_map":
                raise CodegenError(
                    f"PureMap {ctx.get('caller_uid')} attempted to read reduction state {uid}"
                )
            if args:
                raise CodegenError(f"reduction Cell {uid} unexpectedly has arguments")
            return self._cast_read(self._scalar_expr(uid), callee.dtype)
        if role == "pure_map":
            return self._pure_map_call(uid, args, ctx)
        if mode == "pure_map":
            raise CodegenError(
                f"PureMap {ctx.get('caller_uid')} attempted to read scheduled state {callee.source_fullname}"
            )
        if role == "vector":
            raise CodegenError(f"vector-return Cell {callee.source_fullname} was not scalarized")
        if role == "derived_state":
            if len(args) != 1 or uid not in self.layout.derived_state_slots:
                raise CodegenError(f"invalid derived-state Cell call {uid}")
            caller_uid = ctx.get("caller_uid")
            if caller_uid == uid:
                off = _offset(args[0], ctx.get("loop_var", "t"))
                expected = self.schedule.derived_clock_states[uid].self_offset
                if off != expected:
                    raise CodegenError(
                        f"derived state {callee.source_fullname} may read only its proven previous bucket"
                    )
            else:
                rel = (
                    self.schedule.derived_clock_relation(uid, caller_uid, args[0])
                    if caller_uid is not None
                    else None
                )
                if rel is None:
                    raise CodegenError(
                        f"derived state {callee.source_fullname} read has no proven current-bucket relation"
                    )
            return self._cast_read(self._derived_state_expr(uid), callee.dtype)
        if role == "derived_region":
            if len(args) != 1:
                raise CodegenError(f"derived-region Cell {uid} requires one derived coordinate")
            region = self.schedule.derived_region_for_member(uid)
            if region is None:
                raise CodegenError(f"derived-region member {uid} has no executable region")
            caller_uid = ctx.get("caller_uid")
            if caller_uid in region.member_uids:
                off = _offset(args[0], ctx.get("loop_var", "t"))
                if off != 0:
                    raise CodegenError(
                        f"derived-region member {callee.source_fullname} may be read internally only at the current bucket"
                    )
                return self._cast_read(self._derived_region_expr(uid), callee.dtype)
            rel = (
                self.schedule.derived_clock_relation(uid, caller_uid, args[0])
                if caller_uid is not None
                else None
            )
            if rel is None or uid not in self.layout.derived_region_state_slots:
                raise CodegenError(
                    f"derived-region member {callee.source_fullname} has no persistent current-bucket read"
                )
            return self._cast_read(self._derived_region_expr(uid), callee.dtype)
        if role != "coordinate" or len(args) != 1:
            raise CodegenError(f"invalid coordinate Cell call {uid}")

        loop_var = ctx.get("loop_var", "t")
        caller_uid = ctx.get("caller_uid")
        transition = (
            self.schedule.transition_lag_relation(uid, caller_uid, args[0])
            if caller_uid is not None
            else None
        )
        if transition is not None:
            shift = self._position_shift(transition.proof.primary_offset)
            if shift != -1:
                raise CodegenError("first executable derived region supports only exact primary lag -1")
            pos = ctx.get("loop_pos_c", "p")
            idx = f"({pos}-1)"
            if uid in self.layout.history_slots:
                return self._history_read(uid, idx)
            return self._ring_read(uid, idx)
        off = _offset(args[0], loop_var)
        if off is None and caller_uid is not None:
            alias = self.schedule.affine_alias_relation(uid, caller_uid, args[0])
            if alias is not None:
                off = int(alias.offset)
        if off is None:
            relation = (
                self.schedule.derived_history_relation(uid, caller_uid, args[0])
                if caller_uid is not None
                else None
            )
            if relation is None or relation.proof is None:
                raise CodegenError(
                    f"dynamic coordinate read for {callee.source_fullname} has no typed causal-history proof"
                )
            if uid not in self.layout.history_slots:
                raise CodegenError(
                    f"proven derived-coordinate read for {callee.source_fullname} has no materialized history"
                )
            pos = ctx.get("loop_pos_c", "p")
            texpr = ctx.get("loop_var_c", loop_var)
            qexpr = self.expr(args[0], ctx)
            step = int(self.coordinate_step)
            idx = (
                f"({pos} + ((<Py_ssize_t>({qexpr}) - <Py_ssize_t>({texpr})) / {step}))"
            )
            return self._history_read(uid, idx)
        shift = self._position_shift(int(off))
        pos = ctx.get("loop_pos_c", "p")
        src_phase = self.schedule.coordinate_phase[uid]
        cur_phase = ctx.get("phase", src_phase)
        mode = ctx.get("mode", "formula")

        def pos_index(delta: int) -> str:
            return f"({pos}{delta:+d})" if delta else pos

        def seeded(read_expr: str) -> str:
            if shift == -1 and uid in self.layout.boundary_seed_slots:
                seed = self._cast_read(self._boundary_seed_expr(uid), callee.dtype)
                return f"({seed} if {pos} == 0 else {read_expr})"
            return read_expr

        if mode == "reduction":
            return seeded(self._history_read(uid, pos_index(shift)))
        if mode == "fused_reduction":
            if shift != 0 or src_phase > cur_phase:
                raise CodegenError(
                    f"fused reduction requested unavailable coordinate value {callee.source_fullname} offset {off}"
                )
            if src_phase == cur_phase:
                return self._cast_read(self._current_expr(uid), callee.dtype)
            return self._history_read(uid, pos)

        if src_phase < cur_phase:
            return seeded(self._history_read(uid, pos_index(shift)))
        if src_phase > cur_phase:
            raise CodegenError(f"executable loop {cur_phase} depends on future loop {src_phase}")
        if shift == 0:
            return self._cast_read(self._current_expr(uid), callee.dtype)
        idx = pos_index(shift)
        if uid in self.layout.history_slots:
            return seeded(self._history_read(uid, idx))
        return seeded(self._ring_read(uid, idx))

    def expr(self, n: ast.AST, ctx) -> str:
        # Scalar syntax/provider intrinsics are shared with the recursive
        # backend.  Execution topology remains in ``_cell_call``.
        return emit_cython_expr(self, n, ctx)

    def _expr_dtype(self, node: ast.AST, locals_: dict[str, str]) -> str | None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, int):
                return "int64"
            if isinstance(node.value, float):
                return "float64"
            return None
        if isinstance(node, ast.Name):
            return locals_.get(node.id)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return "bool"
            return self._expr_dtype(node.operand, locals_)
        if isinstance(node, ast.Compare) or isinstance(node, ast.BoolOp):
            return "bool"
        if isinstance(node, ast.IfExp):
            a = self._expr_dtype(node.body, locals_)
            b = self._expr_dtype(node.orelse, locals_)
            if a == b:
                return a
            if {a, b} <= {"bool", "int64"}:
                return "int64"
            if a is not None and b is not None:
                return "float64"
            return a or b
        if isinstance(node, ast.BinOp):
            a = self._expr_dtype(node.left, locals_)
            b = self._expr_dtype(node.right, locals_)
            if isinstance(node.op, ast.Div):
                return "float64"
            if a in ("bool", "int64") and b in ("bool", "int64") and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
            ):
                return "int64"
            if a is not None and b is not None:
                return "float64"
            return a or b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.v:
                return self.v[name].dtype
            if name in ("point_input", "global_input", "array_input", "table_input"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    spec = self.input_specs.get(node.args[0].value)
                    return None if spec is None else spec.dtype
            if name in ("__exact_axis_code__", "__exact_axis_code_valid__"):
                return "int64"
            if name == "int":
                return "int64"
            if name == "bool":
                return "bool"
            if name == "float":
                return "float64"
            if name in ("is_missing", "isinf"):
                return "bool"
            if name in ("min", "max", "abs") and node.args:
                kinds = [self._expr_dtype(arg, locals_) for arg in node.args]
                if all(k in ("bool", "int64") for k in kinds):
                    return "int64"
                if all(k is not None for k in kinds):
                    return "float64"
        return None

    @staticmethod
    def _merge_local_dtype(old: str | None, new: str | None) -> str | None:
        if new is None:
            return old
        if old is None or old == new:
            return new
        if {old, new} <= {"bool", "int64"}:
            return "int64"
        return "float64"

    def _local_types(self, fn: ast.FunctionDef) -> dict[str, str]:
        assignments: list[tuple[str, ast.AST]] = []
        names: set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                        assignments.append((target.id, n.value))
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                names.add(n.target.id)
                assignments.append((n.target.id, n.value))
        inferred: dict[str, str] = {}
        # A few passes are enough for normalized straight-line locals. Unknown
        # expressions retain the historical safe double default.
        for _ in range(max(1, len(names) + 1)):
            changed = False
            for name, expr in assignments:
                merged = self._merge_local_dtype(inferred.get(name), self._expr_dtype(expr, inferred))
                if merged is not None and inferred.get(name) != merged:
                    inferred[name] = merged
                    changed = True
            if not changed:
                break
        return {name: self.ctype(inferred.get(name, "float64")) for name in names}

    def _helper_phase(self, uid: str) -> int:
        role = self.schedule.roles[uid]
        if role == "pure_map":
            return 0
        if role == "coordinate":
            return self.schedule.coordinate_phase[uid]
        if role == "derived_state":
            return self.schedule.derived_clock_states[uid].phase
        if role == "derived_region":
            region = self.schedule.derived_region_for_member(uid)
            if region is None:
                raise CodegenError(f"derived-region member {uid} has no schedule")
            return region.phase
        if role == "scalar":
            return max(0, self.schedule.scalar_barrier.get(uid, 0) - 1)
        return 0

    def _pure_map_helper_order(self) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(uid: str) -> None:
            if uid in seen:
                return
            seen.add(uid)
            spec = self.schedule.pure_maps[uid]
            for dep in spec.pure_map_dependencies:
                if dep in self.schedule.pure_maps:
                    visit(dep)
            order.append(uid)

        for uid in sorted(self.schedule.pure_maps):
            visit(uid)
        return order

    def _helper_order(self) -> list[str]:
        order: list[str] = self._pure_map_helper_order()
        order.extend(uid for uid in self.schedule.pre_scalars if uid not in order)
        for block in self.schedule.loops:
            order.extend(uid for uid in block.families if uid not in order)
            order.extend(uid for uid in block.post_scalars if uid not in order)
        order.extend(uid for uid in sorted(self.v) if uid not in order)
        return order

    def helper(self, uid: str) -> list[str]:
        cv = self.v[uid]
        fn = cv.function
        role = self.schedule.roles.get(uid, cv.role)
        if fn is None or role in ("reduction", "vector"):
            return []
        local_types = self._local_types(fn)
        if role == "pure_map":
            spec = self.schedule.pure_maps[uid]
            args = ["long t", "Py_ssize_t i"]
            if self.has_guards:
                args.append("cnp.int64_t* error_code")
            args += [
                f"{self.ctype(self.v[dep].dtype)} {self._pure_scalar_arg_name(dep)}"
                for dep in spec.scalar_dependencies
            ]
            args += [self.input_decl(k) for k in self.input_order]
            mode = "pure_map"
            loop_pos = "-1"
            pure_scalar_args = {
                dep: self._pure_scalar_arg_name(dep) for dep in spec.scalar_dependencies
            }
        else:
            args = ["long t", "Py_ssize_t p", "Py_ssize_t i"]
            if self.has_guards:
                args.append("cnp.int64_t* error_code")
            args += [
                "double* cur",
                "double* state",
                "double* svals",
                "double* work",
                "Py_ssize_t T",
            ] + [self.input_decl(k) for k in self.input_order]
            mode = "formula"
            loop_pos = "p"
            pure_scalar_args = {}
        helper_ctype = self.ctype(cv.dtype)
        helper_sentinel = "NAN" if helper_ctype == "double" else "0"
        lines = [f"cdef inline {helper_ctype} eval_{uid}(", "    " + ",\n    ".join(args), ") noexcept nogil:"]
        for name, typ in local_types.items():
            init = "0.0" if typ == "double" else "0"
            lines.append(f"    cdef {typ} l_{name} = {init}")
        ctx = {
            "mode": mode,
            "phase": self._helper_phase(uid),
            "caller_uid": uid,
            "loop_var": "t",
            "loop_var_c": "t",
            "loop_pos_c": loop_pos,
            "pure_scalar_args": pure_scalar_args,
            "locals": {name: f"l_{name}" for name in local_types},
        }

        def emit_stmt(st: ast.stmt, indent: str) -> None:
            guard_code = guard_code_from_stmt(st)
            if guard_code is not None:
                lines.append(f"{indent}error_code[0] = {guard_code}")
                lines.append(f"{indent}return {helper_sentinel}")
                return
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                return
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                lines.append(f"{indent}l_{st.targets[0].id} = {self.expr(st.value, ctx)}")
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                lines.append(f"{indent}l_{st.target.id} = {self.expr(st.value, ctx)}")
            elif isinstance(st, ast.Return):
                if st.value is None:
                    raise CodegenError(f"{cv.source_fullname}: bare return is non-numeric")
                lines.append(f"{indent}return {self.expr(st.value, ctx)}")
            elif isinstance(st, ast.If):
                lines.append(f"{indent}if {self.expr(st.test, ctx)}:")
                if not st.body:
                    lines.append(indent + "    pass")
                for child in st.body:
                    emit_stmt(child, indent + "    ")
                if st.orelse:
                    lines.append(f"{indent}else:")
                    for child in st.orelse:
                        emit_stmt(child, indent + "    ")
            elif isinstance(st, ast.Raise):
                raise CodegenError(f"{cv.source_fullname}: reachable raise requires fallback")
            else:
                raise CodegenError(
                    f"{cv.source_fullname}: unsupported statement {ast.dump(st, include_attributes=False)}"
                )

        for st in fn.body:
            emit_stmt(st, "    ")
        if not any(isinstance(n, ast.Return) for n in ast.walk(fn)):
            raise CodegenError(f"{cv.source_fullname}: formula has no return")
        lines.append(f"    return {helper_sentinel}  # unreachable for validated total-return formulas")
        lines.append("")
        return lines

    def _direct_formula_locals(self, uid: str) -> dict[str, str]:
        fn = self.v[uid].function
        if fn is None:
            return {}
        return {name: f"l_{uid}_{name}" for name in self._local_types(fn)}

    def _emit_formula_inline(
        self,
        lines: list[str],
        uid: str,
        target: str,
        indent: str,
        *,
        t_expr: str = "t",
        p_expr: str = "p",
    ) -> None:
        """Inline one already-normalized formula into the current native kernel.

        The frontend/safety passes remain authoritative.  This routine only changes
        representation: function returns become assignments to ``target`` and the
        remaining statements are structurally threaded through each conditional so
        early-return semantics are preserved without a runtime ``done`` flag.
        """
        cv = self.v[uid]
        fn = cv.function
        if fn is None:
            raise CodegenError(f"{cv.source_fullname}: direct-local formula has no function")
        locals_map = self._direct_formula_locals(uid)
        ctx = {
            "mode": "formula",
            "phase": self._helper_phase(uid),
            "caller_uid": uid,
            "loop_var": "t",
            "loop_var_c": t_expr,
            "loop_pos_c": p_expr,
            "locals": locals_map,
        }

        def emit_seq(stmts: list[ast.stmt], ind: str) -> None:
            if not stmts:
                raise CodegenError(f"{cv.source_fullname}: formula path has no return")
            st, rest = stmts[0], stmts[1:]
            guard_code = guard_code_from_stmt(st)
            if guard_code is not None:
                lines.append(f"{ind}error_code[0] = {guard_code}")
                lines.append(f"{ind}if work != NULL:")
                lines.append(f"{ind}    free(work)")
                lines.append(f"{ind}return NAN")
                return
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                emit_seq(rest, ind)
                return
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                lines.append(f"{ind}{locals_map[st.targets[0].id]} = {self.expr(st.value, ctx)}")
                emit_seq(rest, ind)
                return
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                if st.value is None:
                    raise CodegenError(f"{cv.source_fullname}: annotation without value requires fallback")
                lines.append(f"{ind}{locals_map[st.target.id]} = {self.expr(st.value, ctx)}")
                emit_seq(rest, ind)
                return
            if isinstance(st, ast.Return):
                if st.value is None:
                    raise CodegenError(f"{cv.source_fullname}: bare return is non-numeric")
                lines.append(f"{ind}{target} = {self.expr(st.value, ctx)}")
                return
            if isinstance(st, ast.If):
                lines.append(f"{ind}if {self.expr(st.test, ctx)}:")
                emit_seq(list(st.body) + list(rest), ind + "    ")
                lines.append(f"{ind}else:")
                emit_seq(list(st.orelse) + list(rest), ind + "    ")
                return
            if isinstance(st, ast.Raise):
                raise CodegenError(f"{cv.source_fullname}: reachable raise requires fallback")
            raise CodegenError(
                f"{cv.source_fullname}: unsupported inline statement {ast.dump(st, include_attributes=False)}"
            )

        emit_seq(list(fn.body), indent)

    def _call_helper(self, uid: str, t_expr: str = "t", p_expr: str = "p") -> str:
        arglist = [t_expr, p_expr, "i"]
        if self.has_guards:
            arglist.append("error_code")
        arglist += ["cur", "state", "svals", "work", "T"] + self.input_order
        return f"eval_{uid}({', '.join(arglist)})"

    def _range_parts(self, args: tuple[ast.AST, ...]) -> tuple[ast.AST, ast.AST, int]:
        if len(args) == 1:
            return ast.Constant(0), args[0], 1
        if len(args) == 2:
            return args[0], args[1], 1
        if len(args) == 3:
            try:
                step = ast.literal_eval(args[2])
            except Exception:
                step = None
            if isinstance(step, int) and not isinstance(step, bool) and int(step) != 0:
                return args[0], args[1], int(step)
        raise CodegenError("executable range requires 1-3 arguments with a constant nonzero integer step")

    def _shared_fused_reduction_factors(self, block) -> list[tuple[str, ast.AST]]:
        """Find repeated multiplicative factors across fused reductions.

        This is deliberately a tiny structural optimization rather than a general
        common-subexpression engine.  Present-value style reductions frequently
        multiply different produced cashflows by the same expensive per-coordinate
        factor.  Materializing that factor once per loop iteration preserves source
        order within each reduction while avoiding duplicate pure numeric work.
        Filtered reductions are excluded so hoisting cannot make a conditionally
        unreachable expression execute eagerly.
        """
        counts: dict[str, tuple[int, ast.AST]] = {}
        for uid in block.reductions:
            if uid not in self.layout.fused_reductions:
                continue
            r = self.v[uid].reduction
            if r is None or r.filters:
                continue
            body = r.body_expr
            if not isinstance(body, ast.BinOp) or not isinstance(body.op, ast.Mult):
                continue
            for candidate in (body.left, body.right):
                # Ignore trivial factors; a named temporary is useful only when it
                # removes genuine arithmetic/table work.
                if sum(1 for _ in ast.walk(candidate)) < 5:
                    continue
                key = ast.dump(candidate, include_attributes=False)
                count, _ = counts.get(key, (0, candidate))
                counts[key] = (count + 1, candidate)
        repeated = [(key, node) for key, (count, node) in counts.items() if count >= 2]
        # Stable ordering keeps generated artifacts reproducible.
        repeated.sort(key=lambda item: item[0])
        return [(f"rf_{block.index}_{idx}", node) for idx, (_key, node) in enumerate(repeated)]

    def _emit_reduction_init(self, lines: list[str], uid: str, phase: int, indent: str = "    ") -> None:
        cv = self.v[uid]
        r = cv.reduction
        if r is None:
            raise CodegenError(f"reduction {uid} has no normalized specification")
        ctx = {
            "mode": "fused_reduction",
            "phase": phase,
            "loop_var": r.loop_var,
            "loop_var_c": "t",
            "loop_pos_c": "p",
            "locals": {r.target: "acc"},
        }
        lines.append(f"{indent}{self._scalar_expr(uid)} = {self.expr(r.init, ctx)}")

    def _emit_fused_reduction_update(self, lines: list[str], uid: str, phase: int, indent: str, expr_overrides: dict[str, str] | None = None) -> None:
        cv = self.v[uid]
        r = cv.reduction
        if r is None:
            raise CodegenError(f"reduction {uid} has no normalized specification")
        ctx = {
            "mode": "fused_reduction",
            "phase": phase,
            "loop_var": r.loop_var,
            "loop_var_c": "t",
            "loop_pos_c": "p",
            "locals": {r.target: self._scalar_expr(uid)},
            "expr_overrides": expr_overrides or {},
        }
        cur_indent = indent
        for filt in r.filters:
            lines.append(f"{cur_indent}if {self.expr(filt, ctx)}:")
            cur_indent += "    "
        lines.append(
            f"{cur_indent}{self._scalar_expr(uid)} += {self.expr(r.body_expr, ctx)}"
        )

    def _emit_standalone_reduction(self, lines: list[str], uid: str, phase: int) -> None:
        cv = self.v[uid]
        r = cv.reduction
        if r is None:
            raise CodegenError(f"reduction {uid} has no normalized specification")
        ctx = {
            "mode": "reduction",
            "phase": phase,
            "loop_var": r.loop_var,
            "loop_var_c": "q",
            "loop_pos_c": "k",
            "locals": {r.target: "acc"},
        }
        lines.append(f"    acc = {self.expr(r.init, ctx)}")
        # k is the physical storage position; q is the exact Python-range value.
        lines.append("    for k in range(T):")
        lines.append(f"        q = coord_start + k * {self.coordinate_step}")
        cur_indent = "        "
        for filt in r.filters:
            lines.append(f"{cur_indent}if {self.expr(filt, ctx)}:")
            cur_indent += "    "
        lines.append(f"{cur_indent}acc += {self.expr(r.body_expr, ctx)}")
        lines.append(f"    {self._scalar_expr(uid)} = acc")

    def _emit_direct_locals(self, module_name: str = "graph_template") -> str:
        lines = [
            "# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False",
            "from cython.parallel cimport prange",
            self._math_cimport_line(),
            "from libc.stdlib cimport malloc, free",
            "cimport numpy as cnp",
            "import numpy as np",
            "",
            f"BUILD_FINGERPRINT = {self.build_fingerprint!r}",
            f"OPTIMIZATION_LEVEL = {self.optimization_level!r}",
            "EMISSION_MODE = 'direct_locals'",
            f"FORMULA_HASH = {self.schedule.formula_hash!r}",
            "",
            f"# executable-template formula hash: {self.schedule.formula_hash}",
            f"# optimization level: {self.optimization_level}",
            "# direct-local emission from the shared executable graph/storage plan",
            "",
        ]
        lines += self._step_lookup_helper_lines()
        lines += self._interval_lookup_helper_lines()
        lines += self._interp_lookup_helper_lines()
        for uid in self._pure_map_helper_order():
            lines += self.helper(uid)
        input_decls = [self.input_decl(k) for k in self.input_order]
        run_one_decls = ["Py_ssize_t i"]
        if self.has_guards:
            run_one_decls.append("cnp.int64_t* error_code")
        run_one_decls += input_decls
        lines += [
            "cdef inline double run_one(",
            "    " + ",\n    ".join(run_one_decls),
            ") noexcept nogil:",
        ]
        # Stable logical locals.  Genericity stays in planning; the generated hot
        # kernel contains ordinary typed variables and direct history/ring names.
        for uid in self.schedule.scalar_uids:
            lines.append(f"    cdef {self.ctype(self.v[uid].dtype)} {self._direct_scalar_name(uid)} = 0")
        for uid in self.schedule.family_uids:
            lines.append(f"    cdef {self.ctype(self.v[uid].dtype)} {self._direct_current_name(uid)} = 0")
        for uid in self.schedule.derived_region_transient_uids:
            lines.append(f"    cdef {self.ctype(self.v[uid].dtype)} {self._direct_current_name(uid)} = 0")
        for uid in sorted(self.schedule.derived_clock_states):
            lines.append(f"    cdef {self.ctype(self.v[uid].dtype)} {self._direct_derived_name(uid)} = 0")
        for uid in sorted(self.layout.derived_region_state_slots):
            lines.append(f"    cdef {self.ctype(self.v[uid].dtype)} {self._direct_region_name(uid)} = 0")
        for uid in sorted(self.schedule.boundary_seeds):
            lines.append(f"    cdef {self.ctype(self.v[uid].dtype)} {self._direct_boundary_name(uid)} = 0")
        for uid in self._helper_order():
            fn = self.v[uid].function
            if fn is None or self.schedule.roles.get(uid, self.v[uid].role) in ("reduction", "vector", "pure_map"):
                continue
            for name, typ in self._local_types(fn).items():
                init = "0.0" if typ == "double" else "0"
                lines.append(f"    cdef {typ} l_{uid}_{name} = {init}")
        shared_factors_by_block = {
            block.index: self._shared_fused_reduction_factors(block)
            for block in self.schedule.loops
        }
        for block in self.schedule.loops:
            for name, _node in shared_factors_by_block[block.index]:
                lines.append(f"    cdef double {name} = 0.0")
        for uid in sorted(self.layout.history_slots):
            lines.append(f"    cdef double* {self._direct_history_name(uid)} = NULL")
        for uid in sorted(self.layout.ring_bases):
            depth = self.layout.ring_depths[uid]
            if uid in self.direct_prev_uids:
                lines.append(f"    cdef {self.ctype(self.v[uid].dtype)} {self._direct_prev_name(uid)} = 0")
            else:
                lines.append(f"    cdef double {self._direct_ring_name(uid)}[{depth}]")
        lines += [
            "    cdef double* work = NULL",
            "    cdef Py_ssize_t T = 0",
            "    cdef Py_ssize_t p, k, j",
            "    cdef long t, q, coord_start, coord_stop",
            "    cdef double acc",
            "    cdef double result = NAN",
        ]
        for uid in sorted(self.layout.ring_bases):
            if uid in self.direct_prev_uids:
                continue
            depth = self.layout.ring_depths[uid]
            lines.append(f"    for j in range({depth}):")
            lines.append(f"        {self._direct_ring_name(uid)}[j] = 0.0")

        for uid in self.schedule.pre_scalars:
            if self.v[uid].function is not None:
                self._emit_formula_inline(lines, uid, self._direct_scalar_name(uid), "    ", t_expr="-1", p_expr="-1")

        start, stop, step = self._range_parts(self.schedule.common_range_args)
        range_ctx = {"mode": "formula", "phase": 0, "loop_var": "t", "loop_var_c": "t", "loop_pos_c": "p", "locals": {}}
        lines.append(f"    coord_start = <long>({self.expr(start, range_ctx)})")
        lines.append(f"    coord_stop = <long>({self.expr(stop, range_ctx)})")
        if step > 0:
            lines += [
                "    if coord_stop <= coord_start:",
                "        T = 0",
                "    else:",
                f"        T = <Py_ssize_t>((coord_stop - coord_start + {step - 1}) // {step})",
            ]
        else:
            mag = -step
            lines += [
                "    if coord_stop >= coord_start:",
                "        T = 0",
                "    else:",
                f"        T = <Py_ssize_t>((coord_start - coord_stop + {mag - 1}) // {mag})",
            ]
        narr = len(self.layout.history_slots)
        if narr:
            lines += [
                f"    work = <double*>malloc(sizeof(double) * {narr} * (T if T > 0 else 1))",
                "    if work == NULL:",
                "        return NAN",
            ]
            for uid, slot in sorted(self.layout.history_slots.items(), key=lambda x: x[1]):
                lines.append(f"    {self._direct_history_name(uid)} = work + {slot} * (T if T > 0 else 1)")

        for block in self.schedule.loops:
            fused = [u for u in block.reductions if u in self.layout.fused_reductions]
            standalone = [u for u in block.reductions if u not in self.layout.fused_reductions]
            lines.append(f"    # executable loop block {block.index}: {block.uid}")
            derived = [
                state for state in self.schedule.derived_clock_states.values()
                if state.phase == block.index
            ]
            regions = [
                region for region in self.schedule.derived_clock_regions.values()
                if region.phase == block.index
            ]
            seeds = [
                seed for seed in self.schedule.boundary_seeds.values()
                if seed.phase == block.index
            ]
            for seed in sorted(seeds, key=lambda row: row.uid):
                lines.append("    if T > 0:")
                self._emit_formula_inline(
                    lines,
                    seed.uid,
                    self._direct_boundary_name(seed.uid),
                    "        ",
                    t_expr=str(seed.coordinate),
                    p_expr="-1",
                )
            for dstate in sorted(derived, key=lambda row: row.uid):
                lines.append("    if T > 0:")
                self._emit_formula_inline(
                    lines,
                    dstate.uid,
                    self._direct_derived_name(dstate.uid),
                    "        ",
                    t_expr=str(dstate.proof.initial_bucket),
                    p_expr="-1",
                )
            for region in sorted(regions, key=lambda row: row.root_uid):
                lines.append("    if T > 0:")
                for uid in region.member_uids:
                    target = (
                        self._direct_region_name(uid)
                        if uid in region.persistent_uids
                        else self._direct_current_name(uid)
                    )
                    self._emit_formula_inline(
                        lines,
                        uid,
                        target,
                        "        ",
                        t_expr=str(region.clock_proof.initial_bucket),
                        p_expr="-1",
                    )
            for uid in fused:
                self._emit_reduction_init(lines, uid, block.index)
            lines.append("    for p in range(T):" if block.scan_direction > 0 else "    for p in range(T - 1, -1, -1):")
            lines.append(f"        t = coord_start + p * {block.coordinate_step}")
            dctx = {
                "mode": "formula",
                "phase": block.index,
                "loop_var": "t",
                "loop_var_c": "t",
                "loop_pos_c": "p",
                "locals": {},
            }
            for dstate in sorted(derived, key=lambda row: row.uid):
                current_bucket = self.expr(dstate.mapping_expr, dctx)
                prev_ctx = dict(dctx)
                prev_ctx["loop_var_c"] = f"(t - {block.coordinate_step})"
                previous_bucket = self.expr(dstate.mapping_expr, prev_ctx)
                lines.append(
                    f"        if p > 0 and <long>({current_bucket}) != <long>({previous_bucket}):"
                )
                self._emit_formula_inline(
                    lines,
                    dstate.uid,
                    self._direct_derived_name(dstate.uid),
                    "            ",
                    t_expr=f"<long>({current_bucket})",
                    p_expr="p",
                )
            for region in sorted(regions, key=lambda row: row.root_uid):
                current_bucket = self.expr(region.mapping_expr, dctx)
                prev_ctx = dict(dctx)
                prev_ctx["loop_var_c"] = f"(t - {block.coordinate_step})"
                previous_bucket = self.expr(region.mapping_expr, prev_ctx)
                lines.append(
                    f"        if p > 0 and <long>({current_bucket}) != <long>({previous_bucket}):"
                )
                for uid in region.member_uids:
                    target = (
                        self._direct_region_name(uid)
                        if uid in region.persistent_uids
                        else self._direct_current_name(uid)
                    )
                    self._emit_formula_inline(
                        lines,
                        uid,
                        target,
                        "            ",
                        t_expr=f"<long>({current_bucket})",
                        p_expr="p",
                    )
            if not block.families and not fused:
                lines.append("        pass")
            for uid in block.families:
                target = self._direct_current_name(uid)
                self._emit_formula_inline(lines, uid, target, "        ")
                if uid in self.layout.history_slots:
                    lines.append(f"        {self._direct_history_name(uid)}[p] = {target}")
                if uid in self.layout.ring_bases and uid not in self.direct_prev_uids:
                    depth = self.layout.ring_depths[uid]
                    lines.append(f"        {self._direct_ring_name(uid)}[p % {depth}] = {target}")
            factor_overrides: dict[str, str] = {}
            if fused:
                factor_ctx = {
                    "mode": "fused_reduction",
                    "phase": block.index,
                    "loop_var": "t",
                    "loop_var_c": "t",
                    "loop_pos_c": "p",
                    "locals": {},
                }
                for factor_name, factor_node in shared_factors_by_block[block.index]:
                    lines.append(f"        {factor_name} = {self.expr(factor_node, factor_ctx)}")
                    factor_overrides[ast.dump(factor_node, include_attributes=False)] = factor_name
            for uid in fused:
                self._emit_fused_reduction_update(
                    lines, uid, block.index, "        ", expr_overrides=factor_overrides
                )
            for uid in block.families:
                if uid in self.direct_prev_uids:
                    lines.append(f"        {self._direct_prev_name(uid)} = {self._direct_current_name(uid)}")
            for uid in standalone:
                self._emit_standalone_reduction(lines, uid, block.index)
            for uid in block.post_scalars:
                if self.v[uid].function is not None:
                    self._emit_formula_inline(lines, uid, self._direct_scalar_name(uid), "    ", t_expr="-1", p_expr="-1")

        out = self.schedule.output_uid
        if self.schedule.roles.get(out) not in ("scalar", "reduction"):
            raise CodegenError("current portfolio API expects a scalar/reduction output")
        lines.append(f"    result = {self._direct_scalar_name(out)}")
        if narr:
            lines.append("    free(work)")
        lines += ["    return result", ""]
        lines += self._portfolio_wrapper_lines()
        return "\n".join(lines)

    def _portfolio_wrapper_lines(self) -> list[str]:
        sig: list[str] = []
        prep: list[str] = []
        call: list[str] = []
        n_source = None
        for key in self.input_order:
            spec = self.input_specs[key]
            if spec.ndim == 0:
                sig.append(f"{self.ctype(spec.dtype)} {key}")
                call.append(key)
            elif spec.dtype == "int64" and spec.ndim == 1:
                sig.append(f"cnp.ndarray[cnp.int64_t, ndim=1] np_{key}")
                prep.append(f"    cdef long[::1] {key} = np_{key}")
                call.append(key)
                if spec.scope == "point" and n_source is None:
                    n_source = f"np_{key}.shape[0]"
            elif spec.dtype == "bool" and spec.ndim == 1:
                sig.append(f"cnp.ndarray[cnp.npy_bool, ndim=1] np_{key}")
                prep.append(f"    cdef cnp.npy_bool[::1] {key} = np_{key}")
                call.append(key)
                if spec.scope == "point" and n_source is None:
                    n_source = f"np_{key}.shape[0]"
            elif spec.dtype == "float64" and spec.ndim == 1:
                sig.append(f"cnp.ndarray[cnp.float64_t, ndim=1] np_{key}")
                prep.append(f"    cdef double[::1] {key} = np_{key}")
                call.append(key)
                if spec.scope == "point" and n_source is None:
                    n_source = f"np_{key}.shape[0]"
            elif spec.ndim >= 2 and spec.dtype in {"float64", "int64", "bool"}:
                np_type = "cnp.float64_t" if spec.dtype == "float64" else "cnp.int64_t" if spec.dtype == "int64" else "cnp.npy_bool"
                base = "double" if spec.dtype == "float64" else "long" if spec.dtype == "int64" else "cnp.npy_bool"
                axes = [":"] * int(spec.ndim)
                axes[-1] = "::1"
                sig.append(f"cnp.ndarray[{np_type}, ndim={int(spec.ndim)}] np_{key}")
                prep.append(f"    cdef {base}[{','.join(axes)}] {key} = np_{key}")
                call.append(key)
            else:
                raise CodegenError(f"unsupported wrapper input {spec}")
        if n_source is None:
            raise CodegenError("at least one point-scoped input is required for the portfolio backend")
        lines = ["cpdef cnp.ndarray run(", "    " + ",\n    ".join(sig + ["int threads=1"]), "):"]
        lines += prep
        lines += [
            f"    cdef Py_ssize_t n = {n_source}",
        ]
        for key in self.input_order:
            spec = self.input_specs[key]
            if spec.enum_labels is not None:
                lines += [
                    f"    if np.any(np_{key} < -1) or np.any(np_{key} >= {len(spec.enum_labels)}):",
                    f"        raise ValueError({('enum input ' + key + ' contains an invalid code')!r})",
                ]
        for fact in self.validated_static_facts:
            spec = self.input_specs[fact.validation_input_key]
            if spec.enum_labels is not None:
                expected = spec.enum_labels.index(fact.value)
                kind = "enum"
            else:
                expected = int(fact.value) if spec.dtype == "int64" else bool(fact.value) if spec.dtype == "bool" else float(fact.value)
                kind = "scalar"
            lines += [
                f"    if np.any(np_{fact.validation_input_key} != {expected!r}):",
                f"        raise ValueError({('validated static ' + kind + ' ' + fact.source_name + ' changed from proven value ' + repr(fact.value))!r})",
            ]
        lines += [
            "    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_arr = np.empty(n, dtype=np.float64)",
            "    cdef double[::1] out = out_arr",
            "    cdef Py_ssize_t i",
        ]
        if self.has_guards:
            lines += [
                "    cdef cnp.ndarray[cnp.int64_t, ndim=1] guard_error_arr = np.zeros(n, dtype=np.int64)",
                "    cdef cnp.int64_t[::1] guard_errors = guard_error_arr",
            ]
        args = ", ".join(call)
        run_args = f"&guard_errors[i], {args}" if self.has_guards else args
        lines += [
            "    if threads <= 1 or n <= 1:",
            "        with nogil:",
            "            for i in range(n):",
            f"                out[i] = run_one(i, {run_args})",
            "    else:",
            "        for i in prange(n, nogil=True, schedule='static', num_threads=threads):",
            f"            out[i] = run_one(i, {run_args})",
        ]
        if self.has_guards:
            lines.append("    for i in range(n):")
            lines.append("        if guard_errors[i] != 0:")
            first = True
            for guard in self.guards:
                keyword = "if" if first else "elif"
                lines.append(f"            {keyword} guard_errors[i] == {guard.code}:")
                if guard.message is None:
                    lines.append(f"                raise {guard.exception_type}")
                else:
                    lines.append(f"                raise {guard.exception_type}({guard.message!r})")
                first = False
            lines.append("            else:")
            lines.append("                raise RuntimeError('unknown optimized guard code')")
        lines += ["    return out_arr", ""]
        return lines

    def emit(self, module_name: str = "graph_template") -> str:
        if self.emission_mode == "direct_locals":
            return self._emit_direct_locals(module_name)
        lines = [
            "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False",
            "from cython.parallel cimport prange",
            self._math_cimport_line(),
            "from libc.stdlib cimport malloc, free",
            "cimport numpy as cnp",
            "import numpy as np",
            "",
            f"BUILD_FINGERPRINT = {self.build_fingerprint!r}",
            f"OPTIMIZATION_LEVEL = {self.optimization_level!r}",
            f"FORMULA_HASH = {self.schedule.formula_hash!r}",
            "",
            f"# executable-template formula hash: {self.schedule.formula_hash}",
            f"# optimization level: {self.optimization_level}",
            "",
        ]
        lines += self._step_lookup_helper_lines()
        lines += self._interval_lookup_helper_lines()
        lines += self._interp_lookup_helper_lines()
        for uid in self._helper_order():
            lines += self.helper(uid)

        input_decls = [self.input_decl(k) for k in self.input_order]
        run_one_decls = ["Py_ssize_t i"]
        if self.has_guards:
            run_one_decls.append("cnp.int64_t* error_code")
        run_one_decls += input_decls
        lines += [
            "cdef inline double run_one(",
            "    " + ",\n    ".join(run_one_decls),
            ") noexcept nogil:",
        ]
        nc = max(1, self.layout.current_slot_count)
        ns = max(1, len(self.layout.scalar_slots))
        nr = max(1, self.layout.state_slot_count)
        lines += [
            f"    cdef double cur[{nc}]",
            f"    cdef double state[{nr}]",
            f"    cdef double svals[{ns}]",
            "    cdef double* work = NULL",
            "    cdef Py_ssize_t T = 0",
            "    cdef Py_ssize_t p, k, j",
            "    cdef long t, q, coord_start, coord_stop",
            "    cdef double acc",
            "    cdef double result = NAN",
            f"    for j in range({nc}):",
            "        cur[j] = 0.0",
            f"    for j in range({nr}):",
            "        state[j] = 0.0",
            f"    for j in range({ns}):",
            "        svals[j] = 0.0",
        ]

        for uid in self.schedule.pre_scalars:
            if self.v[uid].function is not None:
                lines.append(f"    svals[{self.layout.scalar_slots[uid]}] = {self._call_helper(uid, '-1', '-1')}")
                if self.has_guards:
                    lines += [
                        "    if error_code[0] != 0:",
                        "        if work != NULL:",
                        "            free(work)",
                        "        return NAN",
                    ]

        start, stop, step = self._range_parts(self.schedule.common_range_args)
        range_ctx = {"mode": "formula", "phase": 0, "loop_var": "t", "loop_var_c": "t", "loop_pos_c": "p", "locals": {}}
        lines.append(f"    coord_start = <long>({self.expr(start, range_ctx)})")
        lines.append(f"    coord_stop = <long>({self.expr(stop, range_ctx)})")
        if step > 0:
            lines += [
                "    if coord_stop <= coord_start:",
                "        T = 0",
                "    else:",
                f"        T = <Py_ssize_t>((coord_stop - coord_start + {step - 1}) // {step})",
            ]
        else:
            mag = -step
            lines += [
                "    if coord_stop >= coord_start:",
                "        T = 0",
                "    else:",
                f"        T = <Py_ssize_t>((coord_start - coord_stop + {mag - 1}) // {mag})",
            ]
        narr = len(self.layout.history_slots)
        if narr:
            lines += [
                f"    work = <double*>malloc(sizeof(double) * {narr} * (T if T > 0 else 1))",
                "    if work == NULL:",
                "        return NAN",
            ]

        for block in self.schedule.loops:
            fused = [u for u in block.reductions if u in self.layout.fused_reductions]
            standalone = [u for u in block.reductions if u not in self.layout.fused_reductions]
            lines.append(f"    # executable loop block {block.index}: {block.uid}")
            derived = [
                state_row for state_row in self.schedule.derived_clock_states.values()
                if state_row.phase == block.index
            ]
            regions = [
                region for region in self.schedule.derived_clock_regions.values()
                if region.phase == block.index
            ]
            seeds = [
                seed for seed in self.schedule.boundary_seeds.values()
                if seed.phase == block.index
            ]
            for seed in sorted(seeds, key=lambda row: row.uid):
                slot = self.layout.boundary_seed_slots[seed.uid]
                lines.append("    if T > 0:")
                lines.append(
                    f"        state[{slot}] = {self._call_helper(seed.uid, str(seed.coordinate), '-1')}"
                )
                if self.has_guards:
                    lines += [
                        "        if error_code[0] != 0:",
                        "            if work != NULL:",
                        "                free(work)",
                        "            return NAN",
                    ]
            for dstate in sorted(derived, key=lambda row: row.uid):
                slot = self.layout.derived_state_slots[dstate.uid]
                lines.append("    if T > 0:")
                lines.append(
                    f"        state[{slot}] = {self._call_helper(dstate.uid, str(dstate.proof.initial_bucket), '-1')}"
                )
                if self.has_guards:
                    lines += [
                        "        if error_code[0] != 0:",
                        "            if work != NULL:",
                        "                free(work)",
                        "            return NAN",
                    ]
            for region in sorted(regions, key=lambda row: row.root_uid):
                lines.append("    if T > 0:")
                for uid in region.member_uids:
                    target = (
                        f"state[{self.layout.derived_region_state_slots[uid]}]"
                        if uid in region.persistent_uids
                        else f"cur[{self.layout.current_slots[uid]}]"
                    )
                    lines.append(
                        f"        {target} = {self._call_helper(uid, str(region.clock_proof.initial_bucket), '-1')}"
                    )
                    if self.has_guards:
                        lines += [
                            "        if error_code[0] != 0:",
                            "            if work != NULL:",
                            "                free(work)",
                            "            return NAN",
                        ]
            for uid in fused:
                self._emit_reduction_init(lines, uid, block.index)
            lines.append("    for p in range(T):" if block.scan_direction > 0 else "    for p in range(T - 1, -1, -1):")
            lines.append(f"        t = coord_start + p * {block.coordinate_step}")
            dctx = {
                "mode": "formula",
                "phase": block.index,
                "loop_var": "t",
                "loop_var_c": "t",
                "loop_pos_c": "p",
                "locals": {},
            }
            for dstate in sorted(derived, key=lambda row: row.uid):
                slot = self.layout.derived_state_slots[dstate.uid]
                current_bucket = self.expr(dstate.mapping_expr, dctx)
                prev_ctx = dict(dctx)
                prev_ctx["loop_var_c"] = f"(t - {block.coordinate_step})"
                previous_bucket = self.expr(dstate.mapping_expr, prev_ctx)
                lines.append(
                    f"        if p > 0 and <long>({current_bucket}) != <long>({previous_bucket}):"
                )
                lines.append(
                    f"            state[{slot}] = {self._call_helper(dstate.uid, f'<long>({current_bucket})', 'p')}"
                )
                if self.has_guards:
                    lines += [
                        "            if error_code[0] != 0:",
                        "                if work != NULL:",
                        "                    free(work)",
                        "                return NAN",
                    ]
            for region in sorted(regions, key=lambda row: row.root_uid):
                current_bucket = self.expr(region.mapping_expr, dctx)
                prev_ctx = dict(dctx)
                prev_ctx["loop_var_c"] = f"(t - {block.coordinate_step})"
                previous_bucket = self.expr(region.mapping_expr, prev_ctx)
                lines.append(
                    f"        if p > 0 and <long>({current_bucket}) != <long>({previous_bucket}):"
                )
                for uid in region.member_uids:
                    target = (
                        f"state[{self.layout.derived_region_state_slots[uid]}]"
                        if uid in region.persistent_uids
                        else f"cur[{self.layout.current_slots[uid]}]"
                    )
                    lines.append(
                        f"            {target} = {self._call_helper(uid, f'<long>({current_bucket})', 'p')}"
                    )
                    if self.has_guards:
                        lines += [
                            "            if error_code[0] != 0:",
                            "                if work != NULL:",
                            "                    free(work)",
                            "                return NAN",
                        ]
            if not block.families and not fused:
                lines.append("        pass")
            for uid in block.families:
                slot = self.layout.current_slots[uid]
                lines.append(f"        cur[{slot}] = {self._call_helper(uid)}")
                if self.has_guards:
                    lines += [
                        "        if error_code[0] != 0:",
                        "            if work != NULL:",
                        "                free(work)",
                        "            return NAN",
                    ]
                if uid in self.layout.history_slots:
                    hslot = self.layout.history_slots[uid]
                    lines.append(f"        work[{hslot} * T + p] = cur[{slot}]")
                if uid in self.layout.ring_bases:
                    base = self.layout.ring_bases[uid]
                    depth = self.layout.ring_depths[uid]
                    lines.append(f"        state[{base} + (p % {depth})] = cur[{slot}]")
            for uid in fused:
                self._emit_fused_reduction_update(lines, uid, block.index, "        ")
            for uid in standalone:
                self._emit_standalone_reduction(lines, uid, block.index)
            for uid in block.post_scalars:
                if self.v[uid].function is not None:
                    lines.append(f"    svals[{self.layout.scalar_slots[uid]}] = {self._call_helper(uid, '-1', '-1')}")
                    if self.has_guards:
                        lines += [
                            "    if error_code[0] != 0:",
                            "        if work != NULL:",
                            "            free(work)",
                            "        return NAN",
                        ]

        out = self.schedule.output_uid
        if self.schedule.roles.get(out) not in ("scalar", "reduction"):
            raise CodegenError("current portfolio API expects a scalar/reduction output")
        lines.append(f"    result = svals[{self.layout.scalar_slots[out]}]")
        if narr:
            lines.append("    free(work)")
        lines += ["    return result", ""]

        lines += self._portfolio_wrapper_lines()

        return "\n".join(lines)

    def write(self, path: str | Path, module_name: str = "graph_template") -> str:
        text = self.emit(module_name)
        Path(path).write_text(text)
        return text
