from __future__ import annotations

import ast
import math
from pathlib import Path

from .codegen_common import CodegenError
from .frontend import CanonicalModel
from .optimized_program import OptimizedProgram, build_optimized_program
from .program_semantics import guard_code_from_stmt


class PythonLoopGenerator:
    """Emit readable pure Python from the same executable graph template as Cython."""

    def __init__(
        self,
        model: CanonicalModel | OptimizedProgram,
        *,
        optimization_level: str | None = None,
        full_array: bool | None = None,
    ):
        if isinstance(model, OptimizedProgram):
            if optimization_level is not None and optimization_level.upper() != model.optimization_level:
                raise CodegenError("optimized program and requested optimization level differ")
            if full_array is not None and bool(full_array) != model.full_array:
                raise CodegenError("optimized program and requested full_array policy differ")
            program = model
        else:
            program = build_optimized_program(
                model, optimization_level=optimization_level, full_array=full_array
            )
        self.program = program
        self.m = program.canonical
        self.v = program.variants
        self.schedule = program.schedule
        self.layout = program.layout
        self.optimization_level = program.optimization_level
        self.full_array = program.full_array
        self.input_order = list(program.input_order)
        self.input_specs = {spec.key: spec for spec in program.inputs}
        self.pre_static_order = list(program.pre_static_order)
        self.post_static_order = {i: list(rows) for i, rows in program.post_static_order}
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

    def _input_expr(self, name: str, args: list[ast.AST], ctx) -> str:
        if not args or not isinstance(args[0], ast.Constant):
            raise CodegenError(f"{name} requires an internal literal input key")
        key = args[0].value
        if key not in self.input_specs:
            raise CodegenError(f"unknown input key {key}")
        spec = self.input_specs[key]
        if name == "point_input":
            if spec.scope != "point" or spec.ndim != 1:
                raise CodegenError(f"point input {key} must be point-scoped 1-D")
            return f"inputs[{key!r}][i]"
        if name == "global_input":
            if spec.ndim != 0:
                raise CodegenError(f"global input {key} must be scalar")
            return f"inputs[{key!r}]"
        if name == "array_input":
            if len(args) != 2 or spec.ndim != 1:
                raise CodegenError(f"array input {key} requires one index")
            return f"inputs[{key!r}][int({self.expr(args[1], ctx)})]"
        if name == "table_input":
            if len(args) != 3 or spec.ndim != 2:
                raise CodegenError(f"table input {key} requires two indices")
            return (
                f"inputs[{key!r}][int({self.expr(args[1], ctx)}), "
                f"int({self.expr(args[2], ctx)})]"
            )
        raise CodegenError(name)

    def _history_read(self, uid: str, index: str) -> str:
        if uid not in self.layout.history_slots:
            raise CodegenError(f"history for {self.v[uid].source_fullname} was not materialized")
        return f"work[{self.layout.history_slots[uid]}, int({index})]"

    def _ring_read(self, uid: str, index: str) -> str:
        if uid not in self.layout.ring_bases:
            raise CodegenError(f"rolling state for {self.v[uid].source_fullname} was not allocated")
        base = self.layout.ring_bases[uid]
        depth = self.layout.ring_depths[uid]
        return f"state[{base} + (int({index}) % {depth})]"

    def _boundary_seed_read(self, uid: str) -> str:
        if uid not in self.layout.boundary_seed_slots:
            raise CodegenError(f"boundary seed for {self.v[uid].source_fullname} was not allocated")
        return f"state[{self.layout.boundary_seed_slots[uid]}]"

    def _derived_region_value(self, uid: str) -> str:
        if uid in self.layout.derived_region_state_slots:
            return f"state[{self.layout.derived_region_state_slots[uid]}]"
        if uid in self.layout.current_slots:
            return f"cur[{self.layout.current_slots[uid]}]"
        raise CodegenError(f"derived-region value {uid} has no storage slot")

    @staticmethod
    def _pure_scalar_arg_name(uid: str) -> str:
        return f"ps_{uid}"

    def _pure_map_call(self, uid: str, args: list[ast.AST], ctx) -> str:
        if len(args) != 1 or uid not in self.schedule.pure_maps:
            raise CodegenError(f"invalid PureMap call {uid}")
        q = self.expr(args[0], ctx)
        scalar_args: list[str] = []
        for dep in self.schedule.pure_maps[uid].scalar_dependencies:
            if ctx.get("mode") == "pure_map":
                available = set(ctx.get("pure_scalar_args", ()))
                if dep not in available:
                    raise CodegenError(
                        f"PureMap {ctx.get('caller_uid')} lacks proven scalar dependency {dep}"
                    )
                scalar_args.append(self._pure_scalar_arg_name(dep))
            else:
                scalar_args.append(f"svals[{self.layout.scalar_slots[dep]}]")
        call_args = [q, "i", *scalar_args, "inputs"]
        return f"eval_{uid}({', '.join(call_args)})"

    def _cell_call(self, uid: str, args: list[ast.AST], ctx) -> str:
        callee = self.v[uid]
        role = self.schedule.roles.get(uid, callee.role)
        mode = ctx.get("mode", "formula")
        if role == "scalar":
            if args:
                raise CodegenError(f"scalar Cell {uid} unexpectedly has arguments")
            if mode == "pure_map":
                available = set(ctx.get("pure_scalar_args", ()))
                if uid not in available:
                    raise CodegenError(
                        f"PureMap {ctx.get('caller_uid')} attempted undeclared scalar read {uid}"
                    )
                return self._pure_scalar_arg_name(uid)
            return f"svals[{self.layout.scalar_slots[uid]}]"
        if role == "reduction":
            if mode == "pure_map":
                raise CodegenError(
                    f"PureMap {ctx.get('caller_uid')} attempted to read reduction state {uid}"
                )
            if args:
                raise CodegenError(f"reduction Cell {uid} unexpectedly has arguments")
            return f"svals[{self.layout.scalar_slots[uid]}]"
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
                off = self.program.coordinate_offset(args[0], ctx.get("loop_var", "t"))
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
            return f"state[{self.layout.derived_state_slots[uid]}]"
        if role == "derived_region":
            if len(args) != 1:
                raise CodegenError(f"derived-region Cell {uid} requires one derived coordinate")
            region = self.schedule.derived_region_for_member(uid)
            if region is None:
                raise CodegenError(f"derived-region member {uid} has no executable region")
            caller_uid = ctx.get("caller_uid")
            if caller_uid in region.member_uids:
                off = self.program.coordinate_offset(args[0], ctx.get("loop_var", "t"))
                if off != 0:
                    raise CodegenError(
                        f"derived-region member {callee.source_fullname} may be read internally only at the current bucket"
                    )
                return self._derived_region_value(uid)
            rel = (
                self.schedule.derived_clock_relation(uid, caller_uid, args[0])
                if caller_uid is not None
                else None
            )
            if rel is None or uid not in self.layout.derived_region_state_slots:
                raise CodegenError(
                    f"derived-region member {callee.source_fullname} has no persistent current-bucket read"
                )
            return self._derived_region_value(uid)
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
            shift = self.program.position_shift(transition.proof.primary_offset)
            if shift != -1:
                raise CodegenError("first executable derived region supports only exact primary lag -1")
            pos = ctx.get("loop_pos_py", "p")
            idx = f"({pos}-1)"
            if uid in self.layout.history_slots:
                return self._history_read(uid, idx)
            return self._ring_read(uid, idx)
        off = self.program.coordinate_offset(args[0], loop_var)
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
            pos = ctx.get("loop_pos_py", "p")
            texpr = ctx.get("loop_var_py", loop_var)
            qexpr = self.expr(args[0], ctx)
            step = int(self.program.coordinate_step)
            idx = f"({pos} + ((int({qexpr}) - int({texpr})) // {step}))"
            return self._history_read(uid, idx)
        shift = self.program.position_shift(int(off))
        pos = ctx.get("loop_pos_py", "p")
        src_phase = self.schedule.coordinate_phase[uid]
        cur_phase = ctx.get("phase", src_phase)
        mode = ctx.get("mode", "formula")

        def pos_index(delta):
            return f"({pos}{delta:+d})" if delta else pos

        def seeded(read_expr: str) -> str:
            if shift == -1 and uid in self.layout.boundary_seed_slots:
                return f"({self._boundary_seed_read(uid)} if int({pos}) == 0 else {read_expr})"
            return read_expr

        if mode == "reduction":
            return seeded(self._history_read(uid, pos_index(shift)))
        if mode == "fused_reduction":
            if shift != 0 or src_phase > cur_phase:
                raise CodegenError(f"fused reduction requested unavailable value {callee.source_fullname}")
            if src_phase == cur_phase:
                return f"cur[{self.layout.current_slots[uid]}]"
            return self._history_read(uid, pos)
        if src_phase < cur_phase:
            return seeded(self._history_read(uid, pos_index(shift)))
        if src_phase > cur_phase:
            raise CodegenError("future executable-loop dependency")
        if shift == 0:
            return f"cur[{self.layout.current_slots[uid]}]"
        idx = pos_index(shift)
        if uid in self.layout.history_slots:
            return seeded(self._history_read(uid, idx))
        return seeded(self._ring_read(uid, idx))

    def expr(self, n: ast.AST, ctx) -> str:
        if isinstance(n, ast.Constant):
            if n.value is None:
                raise CodegenError("None is outside the numeric loop subset")
            if isinstance(n.value, float) and math.isinf(n.value):
                return "math.inf" if n.value > 0 else "-math.inf"
            return repr(n.value)
        if isinstance(n, ast.Name):
            if n.id in ctx.get("locals", {}):
                return ctx["locals"][n.id]
            if n.id == ctx.get("loop_var"):
                return ctx.get("loop_var_py", n.id)
            if n.id in ("True", "False"):
                return n.id
            raise CodegenError(f"unresolved name {n.id}")
        if isinstance(n, ast.UnaryOp):
            op = {ast.USub: "-", ast.UAdd: "+", ast.Not: "not "}.get(type(n.op))
            if op is None:
                raise CodegenError(type(n.op).__name__)
            return f"({op}{self.expr(n.operand, ctx)})"
        if isinstance(n, ast.BinOp):
            a, b = self.expr(n.left, ctx), self.expr(n.right, ctx)
            op = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.FloorDiv: "//",
                ast.Mod: "%",
                ast.Pow: "**",
            }.get(type(n.op))
            if op is None:
                raise CodegenError(type(n.op).__name__)
            return f"(({a}) {op} ({b}))"
        if isinstance(n, ast.BoolOp):
            op = " and " if isinstance(n.op, ast.And) else " or " if isinstance(n.op, ast.Or) else None
            if op is None:
                raise CodegenError("boolean op")
            return "(" + op.join(self.expr(x, ctx) for x in n.values) + ")"
        if isinstance(n, ast.Compare):
            ops = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
            parts = [self.expr(n.left, ctx)]
            for op, comp in zip(n.ops, n.comparators):
                if type(op) not in ops:
                    raise CodegenError(f"comparison {type(op).__name__} requires fallback")
                parts.extend([ops[type(op)], self.expr(comp, ctx)])
            return "(" + " ".join(parts) + ")"
        if isinstance(n, ast.IfExp):
            return f"({self.expr(n.body, ctx)} if {self.expr(n.test, ctx)} else {self.expr(n.orelse, ctx)})"
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                name = n.func.id
                if name in self.v:
                    return self._cell_call(name, n.args, ctx)
                if name in ("point_input", "global_input", "array_input", "table_input"):
                    return self._input_expr(name, n.args, ctx)
                if name in ("max", "min", "abs", "round", "int", "float", "bool"):
                    return f"{name}({', '.join(self.expr(x, ctx) for x in n.args)})"
                if name == "__is_missing__" and len(n.args) == 1:
                    return f"math.isnan(float({self.expr(n.args[0], ctx)}))"
                if name == "__is_inf__" and len(n.args) == 1:
                    return f"math.isinf(float({self.expr(n.args[0], ctx)}))"
                if name == "__erf__" and len(n.args) == 1:
                    return f"math.erf(float({self.expr(n.args[0], ctx)}))"
                if name == "__step_lookup_1d__" and len(n.args) == 5:
                    if not (isinstance(n.args[0], ast.Constant) and isinstance(n.args[1], ast.Constant)):
                        raise CodegenError("step lookup requires literal internal input keys")
                    xkey, ykey = n.args[0].value, n.args[1].value
                    if xkey not in self.input_specs or ykey not in self.input_specs:
                        raise CodegenError("step lookup references unknown input")
                    return (
                        f"_step_lookup_1d(inputs[{xkey!r}], inputs[{ykey!r}], "
                        f"{self.expr(n.args[2], ctx)}, {self.expr(n.args[3], ctx)}, "
                        f"{self.expr(n.args[4], ctx)})"
                    )
                if name == "__interval_lookup_1d__" and len(n.args) in {4, 7}:
                    if not all(isinstance(x, ast.Constant) for x in n.args[:3]):
                        raise CodegenError("interval lookup requires literal internal input keys")
                    lokey, hikey, ykey = (x.value for x in n.args[:3])
                    if any(k not in self.input_specs for k in (lokey, hikey, ykey)):
                        raise CodegenError("interval lookup references unknown input")
                    modes = ""
                    if len(n.args) == 7:
                        modes = ", " + ", ".join(self.expr(x, ctx) for x in n.args[4:7])
                    return (
                        f"_interval_lookup_1d(inputs[{lokey!r}], inputs[{hikey!r}], "
                        f"inputs[{ykey!r}], {self.expr(n.args[3], ctx)}{modes})"
                    )
                if name == "__interp_lookup_1d__" and len(n.args) == 4:
                    if not (
                        isinstance(n.args[0], ast.Constant) and isinstance(n.args[1], ast.Constant)
                        and isinstance(n.args[3], ast.Constant)
                    ):
                        raise CodegenError("interpolation lookup requires literal internal input keys and mode")
                    xkey, ykey = n.args[0].value, n.args[1].value
                    if xkey not in self.input_specs or ykey not in self.input_specs:
                        raise CodegenError("interpolation lookup references unknown input")
                    return (
                        f"_interp_lookup_1d(inputs[{xkey!r}], inputs[{ykey!r}], "
                        f"{self.expr(n.args[2], ctx)}, {int(n.args[3].value)})"
                    )
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
                receiver = n.func.value.id
                obj = self.m.space.refs.get(receiver)
                module_name = getattr(obj, "__name__", None)
                if module_name in ("math", "numpy") and n.func.attr in (
                    "exp", "log", "sqrt", "sin", "cos", "floor", "ceil"
                ) and len(n.args) == 1:
                    return f"math.{n.func.attr}({self.expr(n.args[0], ctx)})"
            raise CodegenError(f"unsupported call {ast.unparse(n)}")
        raise CodegenError(f"unsupported expression {ast.dump(n, include_attributes=False)}")

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
        locals_ = self.program.local_names(uid)
        if role == "pure_map":
            spec = self.schedule.pure_maps[uid]
            scalar_params = [self._pure_scalar_arg_name(dep) for dep in spec.scalar_dependencies]
            params = ["t", "i", *scalar_params, "inputs"]
            lines = [f"def eval_{uid}({', '.join(params)}):"]
            mode = "pure_map"
            loop_pos = "-1"
        else:
            lines = [f"def eval_{uid}(t, p, i, cur, state, svals, work, T, inputs):"]
            mode = "formula"
            loop_pos = "p"
        for name in locals_:
            lines.append(f"    l_{name} = None")
        ctx = {
            "mode": mode,
            "phase": self._helper_phase(uid),
            "caller_uid": uid,
            "loop_var": "t",
            "loop_var_py": "t",
            "loop_pos_py": loop_pos,
            "locals": {name: f"l_{name}" for name in locals_},
            "pure_scalar_args": (
                self.schedule.pure_maps[uid].scalar_dependencies
                if role == "pure_map" else ()
            ),
        }

        def emit_stmt(st: ast.stmt, indent: str) -> None:
            guard_code = guard_code_from_stmt(st)
            if guard_code is not None:
                guard = self.program.guard(guard_code)
                if guard.message is None:
                    lines.append(f"{indent}raise {guard.exception_type}")
                else:
                    lines.append(f"{indent}raise {guard.exception_type}({guard.message!r})")
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
            else:
                raise CodegenError(f"unsupported statement in Python loop backend: {type(st).__name__}")

        for st in fn.body:
            emit_stmt(st, "    ")
        lines.append("")
        return lines

    def _call(self, uid: str, t: str = "t", p: str = "p") -> str:
        return f"eval_{uid}({t}, {p}, i, cur, state, svals, work, T, inputs)"

    def _range_parts(self, args: tuple[ast.AST, ...]):
        return self.program.range_parts(args)

    def emit(self, module_name: str = "graph_python") -> str:
        lines = [
            "import math",
            "import numpy as np",
            "",
            f"BUILD_FINGERPRINT = {self.program.build_fingerprint!r}",
            f"OPTIMIZATION_LEVEL = {self.optimization_level!r}",
            f"FORMULA_HASH = {self.schedule.formula_hash!r}",
            "",
            f"# executable-template formula hash: {self.schedule.formula_hash}",
            f"# optimization level: {self.optimization_level}",
            "",
        ]
        if self.has_step_lookup_intrinsic:
            lines += [
                "def _step_lookup_1d(axis, values, q, default, below_mode):",
                "    j = int(np.searchsorted(axis, q, side='right')) - 1",
                "    if j < 0:",
                "        return float(values[0]) if int(below_mode) == 1 else float(default)",
                "    return float(values[j])",
                "",
            ]
        if self.has_interval_lookup_intrinsic:
            lines += [
                "def _interval_lookup_1d(lower, upper, values, q, closure_mode=0, fallback_mode=0, default=0.0):",
                "    for j in range(len(values)):",
                "        if (int(closure_mode) == 1 and lower[j] <= q <= upper[j]) or (int(closure_mode) == 0 and lower[j] < q <= upper[j]):",
                "            return float(values[j])",
                "    if int(fallback_mode) == 0 or q > upper[-1]:",
                "        return float(values[-1])",
                "    return float(default)",
                "",
            ]
        if self.has_interp_lookup_intrinsic:
            lines += [
                "def _interp_lookup_1d(axis, values, q, mode):",
                "    n = len(axis)",
                "    if q <= axis[0]:",
                "        lo = 0",
                "    elif q >= axis[n - 1]:",
                "        lo = n - 2",
                "    else:",
                "        lo = int(np.searchsorted(axis, q, side='right')) - 1",
                "    hi = lo + 1",
                "    frac = (q - axis[lo]) / (axis[hi] - axis[lo])",
                "    if int(mode) == 1:",
                "        return float(values[lo] * (values[hi] / values[lo]) ** frac)",
                "    return float(values[lo] + (values[hi] - values[lo]) * frac)",
                "",
            ]
        for uid in self._helper_order():
            lines += self.helper(uid)

        nc = max(1, self.layout.current_slot_count)
        ns = max(1, len(self.layout.scalar_slots))
        nr = max(1, self.layout.state_slot_count)
        narr = len(self.layout.history_slots)
        lines += [
            "def run_one(i, inputs):",
            f"    cur = [0.0] * {nc}",
            f"    state = [0.0] * {nr}",
            f"    svals = [0.0] * {ns}",
            "    work = None",
            "    T = 0",
        ]
        for uid in self.schedule.pre_scalars:
            if self.v[uid].function is not None:
                lines.append(f"    svals[{self.layout.scalar_slots[uid]}] = {self._call(uid, '-1', '-1')}")

        start, stop, step = self._range_parts(self.schedule.common_range_args)
        range_ctx = {"mode": "formula", "phase": 0, "loop_var": "t", "loop_var_py": "t", "loop_pos_py": "p", "locals": {}}
        lines.append(f"    coord_start = int({self.expr(start, range_ctx)})")
        lines.append(f"    coord_stop = int({self.expr(stop, range_ctx)})")
        lines.append(f"    coord_step = {step}")
        lines.append("    T = len(range(coord_start, coord_stop, coord_step))")
        lines.append(
            f"    work = np.empty(({narr}, max(T, 1)), dtype=np.float64)"
            if narr
            else "    work = None"
        )

        for block in self.schedule.loops:
            fused = [uid for uid in block.reductions if uid in self.layout.fused_reductions]
            standalone = [uid for uid in block.reductions if uid not in self.layout.fused_reductions]
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
                slot = self.layout.boundary_seed_slots[seed.uid]
                lines.append("    if T > 0:")
                lines.append(
                    f"        state[{slot}] = {self._call(seed.uid, str(seed.coordinate), '-1')}"
                )
            for dstate in sorted(derived, key=lambda row: row.uid):
                slot = self.layout.derived_state_slots[dstate.uid]
                lines.append(f"    if T > 0:")
                lines.append(
                    f"        state[{slot}] = {self._call(dstate.uid, str(dstate.proof.initial_bucket), '-1')}"
                )
            for region in sorted(regions, key=lambda row: row.root_uid):
                lines.append("    if T > 0:")
                for uid in region.member_uids:
                    lines.append(
                        f"        {self._derived_region_value(uid)} = "
                        f"{self._call(uid, str(region.clock_proof.initial_bucket), '-1')}"
                    )
            for uid in fused:
                r = self.v[uid].reduction
                if r is None:
                    raise CodegenError(f"reduction {uid} has no normalized specification")
                ctx = {
                    "mode": "fused_reduction",
                    "phase": block.index,
                    "loop_var": r.loop_var,
                    "loop_var_py": "t",
                    "loop_pos_py": "p",
                    "locals": {r.target: f"svals[{self.layout.scalar_slots[uid]}]"},
                }
                lines.append(f"    svals[{self.layout.scalar_slots[uid]}] = {self.expr(r.init, ctx)}")

            lines.append("    for p in range(T):" if block.scan_direction > 0 else "    for p in range(T - 1, -1, -1):")
            lines.append(f"        t = coord_start + p * {block.coordinate_step}")
            dctx = {
                "mode": "formula",
                "phase": block.index,
                "loop_var": "t",
                "loop_var_py": "t",
                "loop_pos_py": "p",
                "locals": {},
            }
            for dstate in sorted(derived, key=lambda row: row.uid):
                slot = self.layout.derived_state_slots[dstate.uid]
                current_bucket = self.expr(dstate.mapping_expr, dctx)
                prev_ctx = dict(dctx)
                prev_ctx["loop_var_py"] = f"(t - {block.coordinate_step})"
                previous_bucket = self.expr(dstate.mapping_expr, prev_ctx)
                lines.append(
                    f"        if p > 0 and int({current_bucket}) != int({previous_bucket}):"
                )
                lines.append(
                    f"            state[{slot}] = {self._call(dstate.uid, f'int({current_bucket})', 'p')}"
                )
            for region in sorted(regions, key=lambda row: row.root_uid):
                current_bucket = self.expr(region.mapping_expr, dctx)
                prev_ctx = dict(dctx)
                prev_ctx["loop_var_py"] = f"(t - {block.coordinate_step})"
                previous_bucket = self.expr(region.mapping_expr, prev_ctx)
                lines.append(
                    f"        if p > 0 and int({current_bucket}) != int({previous_bucket}):"
                )
                for uid in region.member_uids:
                    lines.append(
                        f"            {self._derived_region_value(uid)} = "
                        f"{self._call(uid, f'int({current_bucket})', 'p')}"
                    )
            if not block.families and not fused:
                lines.append("        pass")
            for uid in block.families:
                slot = self.layout.current_slots[uid]
                lines.append(f"        cur[{slot}] = {self._call(uid)}")
                if uid in self.layout.history_slots:
                    lines.append(f"        work[{self.layout.history_slots[uid]}, p] = cur[{slot}]")
                if uid in self.layout.ring_bases:
                    base = self.layout.ring_bases[uid]
                    depth = self.layout.ring_depths[uid]
                    lines.append(f"        state[{base} + (p % {depth})] = cur[{slot}]")
            for uid in fused:
                r = self.v[uid].reduction
                ctx = {
                    "mode": "fused_reduction",
                    "phase": block.index,
                    "loop_var": r.loop_var,
                    "loop_var_py": "t",
                    "loop_pos_py": "p",
                    "locals": {r.target: f"svals[{self.layout.scalar_slots[uid]}]"},
                }
                indent = "        "
                for filt in r.filters:
                    lines.append(f"{indent}if {self.expr(filt, ctx)}:")
                    indent += "    "
                lines.append(
                    f"{indent}svals[{self.layout.scalar_slots[uid]}] += {self.expr(r.body_expr, ctx)}"
                )

            for uid in standalone:
                r = self.v[uid].reduction
                if r is None:
                    raise CodegenError(f"reduction {uid} has no normalized specification")
                ctx = {
                    "mode": "reduction",
                    "phase": block.index,
                    "loop_var": r.loop_var,
                    "loop_var_py": r.loop_var,
                    "loop_pos_py": "k",
                    "locals": {r.target: "acc"},
                }
                lines.append(f"    acc = {self.expr(r.init, ctx)}")
                lines.append("    for k in range(T):")
                lines.append(f"        {r.loop_var} = coord_start + k * {block.coordinate_step}")
                indent = "        "
                for filt in r.filters:
                    lines.append(f"{indent}if {self.expr(filt, ctx)}:")
                    indent += "    "
                lines.append(f"{indent}acc += {self.expr(r.body_expr, ctx)}")
                lines.append(f"    svals[{self.layout.scalar_slots[uid]}] = acc")

            for uid in block.post_scalars:
                if self.v[uid].function is not None:
                    lines.append(f"    svals[{self.layout.scalar_slots[uid]}] = {self._call(uid, '-1', '-1')}")

        out = self.schedule.output_uid
        if self.schedule.roles.get(out) not in ("scalar", "reduction"):
            raise CodegenError("Python portfolio API expects scalar/reduction output")
        lines += [f"    return float(svals[{self.layout.scalar_slots[out]}])", ""]
        point_keys = [k for k, s in self.input_specs.items() if s.scope == "point" and s.ndim == 1]
        if not point_keys:
            raise CodegenError("at least one point-scoped input is required")
        lines += [
            "def run(inputs):",
            f"    n = len(inputs[{point_keys[0]!r}])",
        ]
        for key, spec in self.input_specs.items():
            if spec.enum_labels is not None:
                lines += [
                    f"    _enum_{key} = np.asarray(inputs[{key!r}])",
                    f"    if np.any(_enum_{key} < -1) or np.any(_enum_{key} >= {len(spec.enum_labels)}):",
                    f"        raise ValueError({('enum input ' + key + ' contains an invalid code')!r})",
                ]
        for fact in self.program.validated_static_facts:
            spec = self.input_specs[fact.validation_input_key]
            if spec.enum_labels is not None:
                expected = spec.enum_labels.index(fact.value)
                input_expr = f"_enum_{fact.validation_input_key}"
                kind = "enum"
            else:
                expected = int(fact.value) if spec.dtype == "int64" else bool(fact.value) if spec.dtype == "bool" else float(fact.value)
                input_expr = f"np.asarray(inputs[{fact.validation_input_key!r}])"
                kind = "scalar"
            lines += [
                f"    if np.any({input_expr} != {expected!r}):",
                f"        raise ValueError({('validated static ' + kind + ' ' + fact.source_name + ' changed from proven value ' + repr(fact.value))!r})",
            ]
        lines += [
            "    return np.fromiter((run_one(i, inputs) for i in range(n)), dtype=np.float64, count=n)",
            "",
        ]
        return "\n".join(lines)

    def write(self, path: str | Path, module_name: str = "graph_python") -> str:
        text = self.emit(module_name)
        Path(path).write_text(text)
        return text
