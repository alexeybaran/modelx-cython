from __future__ import annotations

"""Typed native recursive backend for :mod:`modelx_graph.native_formula`.

The backend evaluates Cells by direct recursive demand and memoizes the complete
argument tuple.  It has no Stage scheduler, scan direction or StageStoragePlan.
The first implementation uses generation-tagged open-addressed memo tables so
arbitrary integer Cell arity does not require a hard-coded time axis.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .codegen_common import CodegenError
from .cython_expression import emit_cython_expr, infer_local_types
from .native_formula import NativeFormulaCell, NativeFormulaProgram
from .local_control_semantics import (
    bounded_add_range_specs_by_for_id,
    bounded_monotone_first_hit_while_specs_by_while_id,
    bounded_filtered_min_range_specs_by_for_id,
    bounded_filtered_add_range_specs_by_for_id,
    continue_stmt_ids_for_supported_bounded_filtered_add,
    fixed_count_state_update_specs_by_for_id,
    recognize_first_hit_range_stmt,
)
from .program_semantics import guard_code_from_stmt
from .int64_range_proof import intervals_from_rows


class RecursiveNativeError(RuntimeError):
    pass


def _pow2(n: int) -> int:
    x = 1
    while x < n:
        x <<= 1
    return x


def _prepared_exact_int_bounds(formula: NativeFormulaProgram) -> dict[str, int]:
    """Certified exact-int scalar input values available to local control."""

    return {
        spec.key: int(spec.prepared_exact_int_value)
        for spec in formula.inputs
        if type(spec.prepared_exact_int_value) is int
        and isinstance(spec.prepared_exact_int_authority, str)
        and bool(spec.prepared_exact_int_authority)
    }


@dataclass(frozen=True)
class RecursiveMemoCell:
    uid: str
    arity: int
    capacity: int
    slot_base: int
    key_base: int

    @property
    def key_longs(self) -> int:
        return self.capacity * self.arity

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "arity": self.arity,
            "capacity": self.capacity,
            "slot_base": self.slot_base,
            "key_base": self.key_base,
            "key_longs": self.key_longs,
        }


@dataclass(frozen=True)
class RecursiveMemoPlan:
    schema: str
    cells: tuple[RecursiveMemoCell, ...]
    total_slots: int
    total_key_longs: int
    default_capacity: int

    def cell(self, uid: str) -> RecursiveMemoCell:
        for row in self.cells:
            if row.uid == uid:
                return row
        raise RecursiveNativeError(f"memo cell {uid!r} not found")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "default_capacity": self.default_capacity,
            "total_slots": self.total_slots,
            "total_key_longs": self.total_key_longs,
            "cells": [x.manifest() for x in self.cells],
        }


@dataclass(frozen=True)
class RecursiveNativeProgram:
    schema: str
    formula_program: NativeFormulaProgram
    memo_plan: RecursiveMemoPlan
    supported: bool
    blockers: tuple[str, ...] = ()

    @property
    def output_uid(self) -> str:
        return self.formula_program.output_uid

    @property
    def cells(self) -> Mapping[str, NativeFormulaCell]:
        return self.formula_program.cells

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "formula_program_uid": self.formula_program.uid,
            "output_uid": self.output_uid,
            "supported": self.supported,
            "blockers": list(self.blockers),
            "memo_plan": self.memo_plan.manifest(),
            "cell_count": len(self.cells),
            "executable_cell_count": sum(1 for x in self.cells.values() if x.execution_role == "formula"),
            "materialized_helper_count": sum(1 for x in self.cells.values() if x.execution_role != "formula"),
            "max_arity": max((x.arity for x in self.cells.values()), default=0),
            "stage_runtime_dependency": False,
        }


def build_recursive_memo_plan(
    formula_program: NativeFormulaProgram,
    *,
    default_capacity: int = 4096,
    capacity_by_uid: Mapping[str, int] | None = None,
) -> RecursiveMemoPlan:
    if default_capacity < 2:
        raise RecursiveNativeError("default recursive memo capacity must be >= 2")
    capacity_by_uid = capacity_by_uid or {}
    slot = 0
    key = 0
    rows: list[RecursiveMemoCell] = []
    for uid in sorted(formula_program.cells):
        if formula_program.cells[uid].execution_role != "formula":
            continue
        arity = formula_program.cells[uid].arity
        cap = 1 if arity == 0 else _pow2(int(capacity_by_uid.get(uid, default_capacity)))
        rows.append(RecursiveMemoCell(uid, arity, cap, slot, key))
        slot += cap
        key += cap * arity
    return RecursiveMemoPlan(
        schema="modelx_graph.recursive_memo.v1",
        cells=tuple(rows),
        total_slots=slot,
        total_key_longs=key,
        default_capacity=_pow2(default_capacity),
    )


def build_recursive_native_program(
    formula_program: NativeFormulaProgram,
    *,
    default_capacity: int = 4096,
    capacity_by_uid: Mapping[str, int] | None = None,
) -> RecursiveNativeProgram:
    blockers: list[str] = []
    active = {uid: cell for uid, cell in formula_program.cells.items() if cell.execution_role == "formula"}
    inactive = set(formula_program.cells) - set(active)
    for uid, cell in active.items():
        if cell.dtype not in {"float64", "int64", "bool"}:
            blockers.append(f"unsupported_result_dtype:{uid}:{cell.dtype}")
        if any(x not in {"int64", "bool"} for x in cell.parameter_dtypes):
            blockers.append(f"memo_key_dtype_not_exact_integer:{uid}:{cell.parameter_dtypes}")
        if cell.reduction is None and cell.function is None:
            blockers.append(f"missing_formula:{uid}")
        trees = []
        if cell.function is not None:
            trees.append(cell.function)
        if cell.reduction is not None:
            trees.extend((cell.reduction.init, cell.reduction.body_expr, *cell.reduction.range_args, *cell.reduction.filters))
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in inactive:
                    blockers.append(f"formula_calls_materialized_helper:{uid}:{node.func.id}")
    out = formula_program.cells.get(formula_program.output_uid)
    if out is None:
        blockers.append("output_missing")
    elif out.execution_role != "formula":
        blockers.append("output_is_not_executable_formula")
    elif out.arity != 0:
        blockers.append("foundation_run_all_requires_zero_arity_output")
    memo = build_recursive_memo_plan(
        formula_program,
        default_capacity=default_capacity,
        capacity_by_uid=capacity_by_uid,
    )
    return RecursiveNativeProgram(
        schema="modelx_graph.recursive_native_program.v1",
        formula_program=formula_program,
        memo_plan=memo,
        supported=not blockers,
        blockers=tuple(sorted(set(blockers))),
    )


class RecursiveCythonGenerator:
    def __init__(self, program: RecursiveNativeProgram):
        if not program.supported:
            raise RecursiveNativeError("recursive native program is blocked: " + "; ".join(program.blockers))
        self.program = program
        self.formula = program.formula_program
        self.all_cells = self.formula.cells
        self.v = {uid: cell for uid, cell in self.formula.cells.items() if cell.execution_role == "formula"}
        self.input_specs = {x.key: x for x in self.formula.inputs}
        self.input_order = list(self.formula.input_order)
        self.module_refs = dict(self.formula.module_refs)
        self._detect_intrinsics()

    @staticmethod
    def ctype(dtype: str) -> str:
        if dtype == "int64":
            return "long long"
        if dtype == "bool":
            return "bint"
        if dtype == "float64":
            return "double"
        raise RecursiveNativeError(f"unsupported Cython dtype {dtype!r}")

    @staticmethod
    def _cast_read(expr: str, dtype: str) -> str:
        if dtype == "int64":
            return f"(<long long>({expr}))"
        if dtype == "bool":
            return f"(<bint>({expr}))"
        return expr

    def _detect_intrinsics(self) -> None:
        nodes: list[ast.AST] = []
        for cell in self.v.values():
            if cell.reduction is not None:
                r = cell.reduction
                trees = (r.init, r.body_expr, *r.range_args, *r.filters)
            elif cell.function is not None:
                trees = (cell.function,)
            else:
                trees = ()
            for tree in trees:
                nodes.extend(ast.walk(tree))
        calls = {
            n.func.id
            for n in nodes
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.has_step_lookup_intrinsic = "__step_lookup_1d__" in calls
        self.sparse_lookup_shapes = tuple(sorted({
            (
                len(n.args) - 2,
                self.input_specs[n.args[1].value].dtype,
            )
            for n in nodes
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "__sparse_table_lookup__"
            and len(n.args) >= 3
            and isinstance(n.args[1], ast.Constant)
            and n.args[1].value in self.input_specs
        }))
        self.has_interval_lookup_intrinsic = "__interval_lookup_1d__" in calls
        self.has_interp_lookup_intrinsic = "__interp_lookup_1d__" in calls
        self.has_missing_intrinsic = "__is_missing__" in calls
        self.has_isinf_intrinsic = "__is_inf__" in calls
        self.has_erf_intrinsic = "__erf__" in calls
        self.has_valid_exact_axis_intrinsic = bool(
            {"__exact_axis_code_valid__", "__prepared_point_lookup__"} & calls
        )
        self.has_infinity_constant = any(
            isinstance(n, ast.Constant) and isinstance(n.value, float) and n.value in {float("inf"), float("-inf")}
            for n in nodes
        )

    def expr(self, n: ast.AST, ctx) -> str:
        return emit_cython_expr(self, n, ctx)

    def _cell_call(self, uid: str, args: list[ast.AST], ctx) -> str:
        cell = self.v[uid]
        if len(args) != cell.arity:
            raise CodegenError(f"recursive call {uid} expected {cell.arity} args, got {len(args)}")
        rendered = []
        for node, dtype in zip(args, cell.parameter_dtypes):
            rendered.append(f"<{self.ctype(dtype)}>({self.expr(node, ctx)})")
        suffix = "" if not rendered else ", " + ", ".join(rendered)
        return f"r_{uid}({ctx.get('policy_index_name', 'i')}{suffix})"

    def _input_expr(self, name: str, args: list[ast.AST], ctx) -> str:
        if not args or not isinstance(args[0], ast.Constant):
            raise CodegenError(f"{name} requires internal literal input key")
        key = args[0].value
        if key not in self.input_specs:
            raise CodegenError(f"unknown input key {key}")
        spec = self.input_specs[key]
        if name == "point_input":
            if spec.scope != "point" or spec.ndim != 1:
                raise CodegenError(f"point input {key} must be point-scoped 1-D")
            return self._cast_read(f"{key}[{ctx.get('policy_index_name', 'i')}]", spec.dtype)
        if name == "global_input":
            if spec.ndim != 0:
                raise CodegenError(f"global scalar {key} must be 0-D")
            return self._cast_read(key, spec.dtype)
        if name in {"array_input", "table_input"}:
            indices = args[1:]
            if len(indices) != spec.ndim:
                raise CodegenError(
                    f"{name} {key} expected {spec.ndim} indices, got {len(indices)}"
                )
            ix = ", ".join(f"<Py_ssize_t>({self.expr(x, ctx)})" for x in indices)
            return self._cast_read(f"{key}[{ix}]", spec.dtype)
        raise CodegenError(name)

    def _memoryview_type(self, spec) -> str:
        ctype = self.ctype(spec.dtype)
        if spec.dtype == "bool":
            ctype = "cnp.npy_bool"
        if spec.ndim == 0:
            return ctype
        axes = [":"] * spec.ndim
        axes[-1] = "::1"
        return f"{ctype}[{', '.join(axes)}]"

    def _scalar_input_init_lines(self, spec, *, indent: str = "    ") -> list[str]:
        """Emit scalar input binding, including I4a-2 exact-int validation."""

        key = spec.key
        raw = f"np.asarray(d[{key!r}]).reshape(())"
        exact = getattr(spec, "prepared_exact_int_value", None)
        if exact is not None:
            if spec.dtype != "int64" or type(exact) is not int:
                raise RecursiveNativeError(
                    f"prepared exact-int input {key!r} lacks int64 scalar ABI"
                )
            return [
                f"{indent}if {raw}.dtype.kind not in 'iu':",
                f"{indent}    raise ValueError('prepared exact-int input {key} requires integral scalar')",
                f"{indent}if int({raw}) != {int(exact)}:",
                f"{indent}    raise ValueError('prepared exact-int input {key} changed after compilation')",
                f"{indent}{key} = int({raw})",
            ]
        cast = "float" if spec.dtype == "float64" else (
            "bool" if spec.dtype == "bool" else "int"
        )
        return [f"{indent}{key} = {cast}({raw})"]

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

    def _lookup_helpers(self) -> list[str]:
        lines: list[str] = []
        for ndim, dtype in self.sparse_lookup_shapes:
            args = ", ".join(f"long long i{j}" for j in range(ndim))
            checks = " and ".join(f"keys[row, {j}] == i{j}" for j in range(ndim))
            value_ctype = "cnp.npy_bool" if dtype == "bool" else self.ctype(dtype)
            return_ctype = self.ctype(dtype)
            lines += [
                f"cdef inline {return_ctype} _sparse_table_lookup_{ndim}_{dtype}(long long[:, ::1] keys, {value_ctype}[::1] values, {args}) noexcept nogil:",
                "    cdef Py_ssize_t row",
                "    for row in range(values.shape[0]):",
                f"        if {checks}:",
                "            return values[row]",
                "    return <" + return_ctype + ">NAN",
                "",
            ]
        if self.has_step_lookup_intrinsic:
            lines += [
                "cdef inline double _step_lookup_1d(double[::1] axis, double[::1] values, double q, double default, long long below_mode) noexcept nogil:",
                "    cdef Py_ssize_t lo = 0", "    cdef Py_ssize_t hi = axis.shape[0]", "    cdef Py_ssize_t mid",
                "    while lo < hi:", "        mid = (lo + hi) // 2", "        if axis[mid] <= q:", "            lo = mid + 1", "        else:", "            hi = mid",
                "    if lo == 0:", "        return values[0] if below_mode == 1 else default", "    return values[lo - 1]", "",
            ]
        if self.has_interval_lookup_intrinsic:
            lines += [
                "cdef inline double _interval_lookup_1d(double[::1] lower, double[::1] upper, double[::1] values, double q, long long closure_mode, long long fallback_mode, double default) noexcept nogil:",
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
        if self.has_interp_lookup_intrinsic:
            lines += [
                "cdef inline double _interp_lookup_1d(double[::1] axis, double[::1] values, double q, long long mode) noexcept nogil:",
                "    cdef Py_ssize_t n = axis.shape[0]", "    cdef Py_ssize_t lo", "    cdef Py_ssize_t hi", "    cdef Py_ssize_t left", "    cdef Py_ssize_t right", "    cdef Py_ssize_t mid", "    cdef double frac",
                "    if q <= axis[0]:", "        lo = 0", "    elif q >= axis[n - 1]:", "        lo = n - 2", "    else:", "        left = 0", "        right = n", "        while left < right:", "            mid = (left + right) // 2", "            if axis[mid] <= q:", "                left = mid + 1", "            else:", "                right = mid", "        lo = left - 1",
                "    hi = lo + 1", "    frac = (q - axis[lo]) / (axis[hi] - axis[lo])", "    if mode == 1:", "        return values[lo] * pow(values[hi] / values[lo], frac)", "    return values[lo] + (values[hi] - values[lo]) * frac", "",
            ]
        return lines

    def _policy_index_name(self, cell: NativeFormulaCell) -> str:
        """Return a compiler-owned policy-index identifier that cannot shadow a Cell argument."""

        used = set(cell.parameters)
        name = "__mx_policy_i"
        while name in used:
            name += "_"
        return name

    def _function_signature(self, cell: NativeFormulaCell, prefix: str) -> str:
        args = [f"Py_ssize_t {self._policy_index_name(cell)}"]
        for name, dtype in zip(cell.parameters, cell.parameter_dtypes):
            args.append(f"{self.ctype(dtype)} {name}")
        return f"cdef inline {self.ctype(cell.dtype)} {prefix}_{cell.uid}({', '.join(args)}) noexcept nogil"

    def _emit_formula(self, cell: NativeFormulaCell) -> list[str]:
        fn = cell.function
        if fn is None:
            raise RecursiveNativeError(f"{cell.uid}: missing formula")
        parameter_types = dict(zip(cell.parameters, cell.parameter_dtypes))
        local_dtypes = infer_local_types(self, fn, parameter_types)
        fixed_count_specs = fixed_count_state_update_specs_by_for_id(
            fn,
            prepared_exact_int_bounds=_prepared_exact_int_bounds(self.formula),
        )
        bounded_add_specs = bounded_add_range_specs_by_for_id(
            fn,
            prepared_exact_int_bounds=_prepared_exact_int_bounds(self.formula),
        )
        integer_names = {
            name for name, dtype in zip(cell.parameters, cell.parameter_dtypes)
            if dtype == "int64"
        } | set(cell.local_integer_names)
        proof_name_intervals = intervals_from_rows(
            (*cell.parameter_int64_intervals, *cell.local_int64_intervals)
        )
        proof_call_intervals = intervals_from_rows(cell.callee_int64_intervals)
        proof_input_intervals = intervals_from_rows(cell.input_int64_intervals)
        scalar_dtypes = dict(parameter_types)
        scalar_dtypes.update(dict(cell.local_scalar_dtypes))
        bounded_while_specs = bounded_monotone_first_hit_while_specs_by_while_id(
            fn,
            integer_names=integer_names,
            positive_int_names=cell.local_positive_int_names,
        )
        filtered_min_specs = bounded_filtered_min_range_specs_by_for_id(
            fn, integer_names=integer_names
        )
        filtered_add_specs = bounded_filtered_add_range_specs_by_for_id(
            fn,
            integer_names=integer_names,
            name_intervals=proof_name_intervals,
            call_intervals=proof_call_intervals,
            input_intervals=proof_input_intervals,
            scalar_dtypes=scalar_dtypes,
        )
        filtered_add_continue_ids = continue_stmt_ids_for_supported_bounded_filtered_add(
            fn,
            integer_names=integer_names,
            name_intervals=proof_name_intervals,
            call_intervals=proof_call_intervals,
            input_intervals=proof_input_intervals,
            scalar_dtypes=scalar_dtypes,
        )
        lines = [self._function_signature(cell, "calc") + ":", "    global _guard_error"]
        for name, dtype in sorted(local_dtypes.items()):
            ctype = self.ctype(dtype)
            lines.append(f"    cdef {ctype} l_{name} = {'0.0' if ctype == 'double' else '0'}")
        ctx = {
            "locals": {**{p: p for p in cell.parameters}, **{n: f"l_{n}" for n in local_dtypes}},
            "local_dtypes": {**parameter_types, **local_dtypes},
            "caller_uid": cell.uid,
            "policy_index_name": self._policy_index_name(cell),
        }

        def emit_stmt(st: ast.stmt, indent: str) -> None:
            guard_code = guard_code_from_stmt(st)
            if guard_code is not None:
                lines.append(f"{indent}_guard_error = {int(guard_code)}")
                lines.append(f"{indent}return <{self.ctype(cell.dtype)}>({'NAN' if cell.dtype == 'float64' else '0'})")
                return
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                return
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                lines.append(f"{indent}l_{st.targets[0].id} = {self.expr(st.value, ctx)}")
                return
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                lines.append(f"{indent}l_{st.target.id} = {self.expr(st.value, ctx)}")
                return
            if isinstance(st, ast.Return):
                if st.value is None:
                    raise CodegenError(f"{cell.source_fullname}: bare return is non-numeric")
                lines.append(f"{indent}return <{self.ctype(cell.dtype)}>({self.expr(st.value, ctx)})")
                return
            if isinstance(st, ast.If):
                lines.append(f"{indent}if {self.expr(st.test, ctx)}:")
                if not st.body:
                    lines.append(indent + "    pass")
                for child in st.body:
                    emit_stmt(child, indent + "    ")
                if st.orelse:
                    lines.append(f"{indent}else:")
                    for child in st.orelse:
                        emit_stmt(child, indent + "    ")
                return
            if isinstance(st, ast.Continue):
                if id(st) not in filtered_add_continue_ids:
                    raise CodegenError(
                        f"recursive native continue unsupported in {cell.source_fullname}"
                    )
                lines.append(f"{indent}continue")
                return
            if isinstance(st, ast.While):
                spec = bounded_while_specs.get(id(st))
                if spec is None:
                    raise CodegenError(
                        f"recursive native local while unsupported in {cell.source_fullname}: "
                        f"{ast.dump(st, include_attributes=False)}"
                    )
                bound_expr = self.expr(spec.bound, ctx)
                lines.append(
                    f"{indent}while l_{spec.search_name} {spec.comparison} <long long>({bound_expr}):"
                )
                # Exact I4b-1A body: first-hit if/return followed by the
                # proven positive update.  Guard int64 overflow before the
                # update and fall through to the search-independent fallback.
                emit_stmt(st.body[0], indent + "    ")
                step_expr = self.expr(spec.step, ctx)
                lines.append(
                    f"{indent}    if l_{spec.search_name} > <long long>9223372036854775807 - <long long>({step_expr}):"
                )
                lines.append(f"{indent}        break")
                lines.append(
                    f"{indent}    l_{spec.search_name} += <long long>({step_expr})"
                )
                return
            if isinstance(st, ast.For):
                prepared_mapping = bool(
                    getattr(st, "_mx_prepared_mapping_range", False)
                    and isinstance(st.target, ast.Name)
                    and isinstance(st.iter, ast.Call)
                    and isinstance(st.iter.func, ast.Name)
                    and st.iter.func.id == "range"
                    and len(st.iter.args) == 2
                )
                spec = recognize_first_hit_range_stmt(st)
                fixed_spec = fixed_count_specs.get(id(st))
                add_spec = bounded_add_specs.get(id(st))
                filtered_min_spec = filtered_min_specs.get(id(st))
                filtered_add_spec = filtered_add_specs.get(id(st))
                if (
                    not prepared_mapping
                    and spec is None
                    and fixed_spec is None
                    and add_spec is None
                    and filtered_min_spec is None
                    and filtered_add_spec is None
                ):
                    raise CodegenError(
                        f"recursive native local for unsupported in {cell.source_fullname}: "
                        f"{ast.dump(st, include_attributes=False)}"
                    )
                if prepared_mapping:
                    range_args = tuple(st.iter.args)
                elif spec is not None:
                    range_args = spec.range_args
                elif fixed_spec is not None:
                    range_args = fixed_spec.range_args
                elif add_spec is not None:
                    range_args = add_spec.range_args
                elif filtered_min_spec is not None:
                    range_args = filtered_min_spec.range_args
                else:
                    range_args = filtered_add_spec.range_args  # type: ignore[union-attr]
                args = [self.expr(x, ctx) for x in range_args]
                if len(args) == 1:
                    range_expr = f"range(<long long>({args[0]}))"
                elif len(args) == 2:
                    range_expr = f"range(<long long>({args[0]}), <long long>({args[1]}))"
                else:
                    # Preserve genuine dev112/dev113 legacy first-hit semantics:
                    # that recognizer admits only an ast.Constant source step.
                    if spec is not None:
                        step = int(range_args[2].value)  # type: ignore[union-attr]
                    elif fixed_spec is not None:
                        step = int(fixed_spec.range_values[2])
                    elif add_spec is not None:
                        step = int(add_spec.range_values[2])
                    else:
                        # New dev115/dev124 filtered loops use their own
                        # recognizers, which admit exact signed literal range
                        # steps via _literal_int_value.  Dev124 deliberately
                        # rejects runtime steps before admission because Cython
                        # otherwise promotes range to Python iteration in nogil.
                        range_spec = (
                            filtered_min_spec
                            if filtered_min_spec is not None
                            else filtered_add_spec
                        )
                        step_node = range_spec.range_args[2]  # type: ignore[union-attr]
                        sign = (
                            -1
                            if isinstance(step_node, ast.UnaryOp)
                            and isinstance(step_node.op, ast.USub)
                            else 1
                        )
                        value_node = (
                            step_node.operand
                            if isinstance(step_node, ast.UnaryOp)
                            else step_node
                        )
                        step = sign * int(value_node.value)  # type: ignore[union-attr]
                    range_expr = (
                        f"range(<long long>({args[0]}), <long long>({args[1]}), {step})"
                    )
                if prepared_mapping:
                    iterator = st.target.id
                elif spec is not None:
                    iterator = spec.iterator
                elif fixed_spec is not None:
                    iterator = fixed_spec.iterator
                elif add_spec is not None:
                    iterator = add_spec.iterator
                elif filtered_min_spec is not None:
                    iterator = filtered_min_spec.iterator
                else:
                    iterator = filtered_add_spec.iterator  # type: ignore[union-attr]
                lines.append(f"{indent}for l_{iterator} in {range_expr}:")
                for child in st.body:
                    emit_stmt(child, indent + "    ")
                return
            if isinstance(st, ast.Pass):
                lines.append(indent + "pass")
                return
            raise CodegenError(
                f"recursive native statement unsupported in {cell.source_fullname}: "
                f"{ast.dump(st, include_attributes=False)}"
            )

        for st in fn.body:
            emit_stmt(st, "    ")
        lines.append(f"    return <{self.ctype(cell.dtype)}>0")
        lines.append("")
        return lines

    def _emit_reduction(self, cell: NativeFormulaCell) -> list[str]:
        r = cell.reduction
        if r is None:
            raise RecursiveNativeError(f"{cell.uid}: reduction missing")
        if len(r.range_args) not in {1, 2, 3}:
            raise CodegenError(f"{cell.uid}: unsupported reduction range arity")
        ctx = {
            "locals": {
                **{name: name for name in cell.parameters},
                r.target: "acc", r.loop_var: "q",
            },
            "local_dtypes": {
                **dict(zip(cell.parameters, cell.parameter_dtypes)),
                r.target: cell.dtype, r.loop_var: "int64",
            },
            "caller_uid": cell.uid,
            "policy_index_name": self._policy_index_name(cell),
        }
        args = [self.expr(x, ctx) for x in r.range_args]
        if len(args) == 1:
            start, stop, step = "0", args[0], "1"
        elif len(args) == 2:
            start, stop, step = args[0], args[1], "1"
        else:
            start, stop, step = args
        ct = self.ctype(cell.dtype)
        lines = [
            self._function_signature(cell, "calc") + ":",
            "    global _memo_error",
            f"    cdef {ct} acc = <{ct}>({self.expr(r.init, ctx)})",
            f"    cdef long long q = <long long>({start})",
            f"    cdef long long stop = <long long>({stop})",
            f"    cdef long long step = <long long>({step})",
            "    if step == 0:",
            "        _memo_error = 3",
            f"        return <{ct}>({'NAN' if cell.dtype == 'float64' else '0'})",
            "    while (q < stop if step > 0 else q > stop):",
        ]
        indent = "        "
        if r.filters:
            lines.append(indent + "if " + " and ".join(self.expr(x, ctx) for x in r.filters) + ":")
            indent += "    "
        lines.append(f"{indent}acc = <{ct}>(acc + ({self.expr(r.body_expr, ctx)}))")
        lines += ["        q += step", "    return acc", ""]
        return lines

    def _emit_wrapper(self, cell: NativeFormulaCell) -> list[str]:
        memo = self.program.memo_plan.cell(cell.uid)
        sig = self._function_signature(cell, "r") + ":"
        policy_i = self._policy_index_name(cell)
        ct = self.ctype(cell.dtype)
        zero = "NAN" if cell.dtype == "float64" else "0"
        if cell.arity == 0:
            return [
                sig,
                f"    cdef Py_ssize_t idx = {memo.slot_base}",
                f"    cdef {ct} val",
                "    global _memo_error",
                "    if _memo_gen[idx] == _epoch:",
                "        if _memo_busy[idx]:",
                "            _memo_error = 2",
                f"            return <{ct}>{zero}",
                f"        return <{ct}>_memo_values[idx]",
                "    _memo_gen[idx] = _epoch",
                "    _memo_busy[idx] = 1",
                f"    val = calc_{cell.uid}({policy_i})",
                "    _memo_values[idx] = <double>val",
                "    _memo_busy[idx] = 0",
                "    return val",
                "",
            ]

        mask = memo.capacity - 1
        lines = [
            sig,
            "    cdef unsigned long long h = 1469598103934665603ULL",
            "    cdef Py_ssize_t pos",
            "    cdef Py_ssize_t idx",
            "    cdef Py_ssize_t probe",
            "    cdef Py_ssize_t kb",
            f"    cdef {ct} val",
            "    cdef bint match",
            "    global _memo_error",
        ]
        for p in cell.parameters:
            lines.append(f"    h = (h ^ <unsigned long long>{p}) * 1099511628211ULL")
        lines += [
            f"    pos = <Py_ssize_t>(h & {mask})",
            f"    for probe in range({memo.capacity}):",
            f"        idx = {memo.slot_base} + pos",
            f"        kb = {memo.key_base} + pos * {cell.arity}",
            "        if _memo_gen[idx] != _epoch:",
            "            _memo_gen[idx] = _epoch",
            "            _memo_busy[idx] = 1",
        ]
        for j, p in enumerate(cell.parameters):
            lines.append(f"            _memo_keys[kb + {j}] = <long long>{p}")
        callargs = ", ".join(cell.parameters)
        suffix = "" if not callargs else ", " + callargs
        lines += [
            f"            val = calc_{cell.uid}({policy_i}{suffix})",
            "            _memo_values[idx] = <double>val",
            "            _memo_busy[idx] = 0",
            "            return val",
            "        match = True",
        ]
        for j, p in enumerate(cell.parameters):
            lines.append(f"        match = match and (_memo_keys[kb + {j}] == <long long>{p})")
        lines += [
            "        if match:",
            "            if _memo_busy[idx]:",
            "                _memo_error = 2",
            f"                return <{ct}>{zero}",
            f"            return <{ct}>_memo_values[idx]",
            f"        pos = (pos + 1) & {mask}",
            "    _memo_error = 1",
            f"    return <{ct}>{zero}",
            "",
        ]
        return lines

    def source(self, module_name: str = "recursive_native") -> str:
        p = self.program
        total_slots = max(1, p.memo_plan.total_slots)
        total_keys = max(1, p.memo_plan.total_key_longs)
        lines = [
            "# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False",
            self._math_cimport_line(),
            "from libc.stdlib cimport malloc, calloc",
            "from libc.string cimport memset",
            "cimport numpy as cnp",
            "import numpy as np",
            "",
            f"cdef Py_ssize_t _MEMO_SLOTS = {total_slots}",
            f"cdef Py_ssize_t _MEMO_KEY_LONGS = {total_keys}",
            "cdef double* _memo_values = NULL",
            "cdef unsigned int* _memo_gen = NULL",
            "cdef unsigned char* _memo_busy = NULL",
            "cdef long long* _memo_keys = NULL",
            "cdef unsigned int _epoch = 1",
            "cdef long long _memo_error = 0",
            "cdef long long _guard_error = 0",
            "cdef Py_ssize_t _n = 0",
            "",
        ]
        if self.has_valid_exact_axis_intrinsic:
            lines += [
                "cdef inline long long __exact_axis_invalid__(long long code) noexcept nogil:",
                "    global _guard_error",
                "    _guard_error = code",
                "    return 0",
                "",
            ]
        for spec in self.formula.inputs:
            lines.append(f"cdef {self._memoryview_type(spec)} {spec.key}")
        lines.append("")
        lines += self._lookup_helpers()

        # SCC forward declarations live in the companion .pxd written by
        # ``write``.  Cython does not accept ordinary forward cdef declarations
        # in a .pyx module.

        for uid in sorted(self.v):
            cell = self.v[uid]
            lines += self._emit_reduction(cell) if cell.reduction is not None else self._emit_formula(cell)
        for uid in sorted(self.v):
            lines += self._emit_wrapper(self.v[uid])

        point_specs = [x for x in self.formula.inputs if x.scope == "point"]
        if not point_specs:
            raise RecursiveNativeError("foundation recursive runner requires at least one point input")
        globals_line = ", ".join(["_memo_values", "_memo_gen", "_memo_busy", "_memo_keys", "_epoch", "_memo_error", "_guard_error", "_n"] + [x.key for x in self.formula.inputs])
        lines += ["cpdef init_inputs(dict d):", f"    global {globals_line}"]
        for spec in self.formula.inputs:
            key = spec.key
            if spec.ndim == 0:
                lines += self._scalar_input_init_lines(spec)
            else:
                npdt = "np.float64" if spec.dtype == "float64" else ("np.bool_" if spec.dtype == "bool" else "np.int64")
                lines.append(f"    {key} = np.ascontiguousarray(d[{key!r}], dtype={npdt})")
                if spec.expected_shape is not None:
                    shape = tuple(int(x) for x in spec.expected_shape)
                    checks = " or ".join(f"{key}.shape[{j}] != {dim}" for j, dim in enumerate(shape))
                    if checks:
                        lines.append(f"    if {checks}:")
                        lines.append(f"        raise ValueError('input {key} shape mismatch')")
        first = point_specs[0].key
        lines.append(f"    _n = {first}.shape[0]")
        for spec in point_specs[1:]:
            lines += [f"    if {spec.key}.shape[0] != _n:", f"        raise ValueError('point input length mismatch: {spec.key}')"]
        lines += [
            "    if _memo_values == NULL:",
            "        _memo_values = <double*>malloc(_MEMO_SLOTS * sizeof(double))",
            "        _memo_gen = <unsigned int*>calloc(_MEMO_SLOTS, sizeof(unsigned int))",
            "        _memo_busy = <unsigned char*>calloc(_MEMO_SLOTS, sizeof(unsigned char))",
            "        _memo_keys = <long long*>malloc(_MEMO_KEY_LONGS * sizeof(long long))",
            "        if _memo_values == NULL or _memo_gen == NULL or _memo_busy == NULL or _memo_keys == NULL:",
            "            raise MemoryError()",
            "    _epoch = 1",
            "    memset(_memo_gen, 0, _MEMO_SLOTS * sizeof(unsigned int))",
            "    memset(_memo_busy, 0, _MEMO_SLOTS * sizeof(unsigned char))",
            "    _memo_error = 0",
            "    _guard_error = 0",
            "    return None",
            "",
            "cdef inline void _next_epoch() noexcept nogil:",
            "    global _epoch",
            "    _epoch += 1",
            "    if _epoch == 0:",
            "        memset(_memo_gen, 0, _MEMO_SLOTS * sizeof(unsigned int))",
            "        _epoch = 1",
            "",
        ]
        out = self.v[self.formula.output_uid]
        lines += [
            "cdef inline double _eval_one(Py_ssize_t i) noexcept nogil:",
            "    _next_epoch()",
            f"    return <double>r_{out.uid}(i)",
            "",
            "cpdef cnp.ndarray run_all():",
            "    cdef cnp.ndarray arr = np.empty(_n, dtype=np.float64)",
            "    cdef double[::1] out = arr",
            "    cdef Py_ssize_t i",
            "    global _memo_error, _guard_error",
            "    _memo_error = 0",
            "    _guard_error = 0",
            "    with nogil:",
            "        for i in range(_n):",
            "            out[i] = _eval_one(i)",
            "            if _memo_error != 0 or _guard_error != 0:",
            "                break",
            "    if _memo_error != 0:",
            "        raise RuntimeError(f'recursive memo failure code {_memo_error}')",
            "    if _guard_error != 0:",
            "        raise ValueError(f'native formula guard failed with code {_guard_error}')",
            "    return arr",
            "",
            "cpdef double bench(long long reps):",
            "    cdef long long j",
            "    cdef Py_ssize_t i",
            "    cdef double acc = 0.0",
            "    global _memo_error, _guard_error",
            "    _memo_error = 0",
            "    _guard_error = 0",
            "    with nogil:",
            "        for j in range(reps):",
            "            for i in range(_n):",
            "                acc += _eval_one(i)",
            "    if _memo_error != 0:",
            "        raise RuntimeError(f'recursive memo failure code {_memo_error}')",
            "    if _guard_error != 0:",
            "        raise ValueError(f'native formula guard failed with code {_guard_error}')",
            "    return acc",
            "",
        ]
        return "\n".join(lines) + "\n"

    def declaration_source(self) -> str:
        lines = ["# cython: language_level=3"]
        for uid in sorted(self.v):
            cell = self.v[uid]
            # ``inline`` belongs on the implementation only; the declaration
            # supplies Cython with the SCC ABI before any body is compiled.
            for prefix in ("r", "calc"):
                sig = self._function_signature(cell, prefix).replace("cdef inline ", "cdef ", 1)
                lines.append(sig)
        return "\n".join(lines) + "\n"

    def write(self, path: str | Path, module_name: str = "recursive_native") -> str:
        source = self.source(module_name)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(source, encoding="utf-8")
        out.with_suffix(".pxd").write_text(self.declaration_source(), encoding="utf-8")
        return source


__all__ = [
    "RecursiveNativeError", "RecursiveMemoCell", "RecursiveMemoPlan",
    "RecursiveNativeProgram", "build_recursive_memo_plan",
    "build_recursive_native_program", "RecursiveCythonGenerator",
]
