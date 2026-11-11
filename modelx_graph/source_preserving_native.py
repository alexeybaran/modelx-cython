from __future__ import annotations

"""Source-preserving native formula backend for Architecture B.

This module deliberately keeps the production recursive runtime/storage model
separate from formula-language lowering.  It reuses ``DenseRecursiveProgram``
for memoization, prepared inputs and batch execution, while ordinary Python
statements/expressions are preserved for Cython itself to compile.

The transformer understands only compiler-owned leaves:

* normalized native Cell calls;
* prepared point/global/array/table inputs;
* normalized lookup/guard intrinsics;
* supported ``math``/``numpy`` scalar intrinsics that must become libc calls
  for ``nogil`` execution.

It does *not* lower normal ``if``/``for``/``while``/arithmetic syntax.
"""

import ast
import copy
import math
import os
import textwrap
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .cython_expression import expression_dtype
from .dense_recursive_native import (
    DenseRecursiveCythonGenerator,
    DenseRecursiveProgram,
    build_dense_recursive_program,
)
from .int64_range_proof import INT64_MAX, INT64_MIN, intervals_from_rows
from .native_formula import NativeFormulaCell, NativeFormulaProgram
from .program_semantics import guard_code_from_stmt
from .recursive_native import RecursiveNativeError


_NATIVE_DTYPES = {"bool", "int64", "float64"}
_RUNTIME_INPUT_CALLS = {"point_input", "global_input", "array_input", "table_input"}
_LIBC_MATH_INTRINSICS = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "sin": "sin",
    "cos": "cos",
    "floor": "floor",
    "ceil": "ceil",
}


class SourcePreservingNativeError(RecursiveNativeError):
    pass


def _join_dtype(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    if a == b:
        return a
    if a not in _NATIVE_DTYPES or b not in _NATIVE_DTYPES:
        return None
    if "float64" in {a, b}:
        return "float64"
    if "int64" in {a, b}:
        return "int64"
    return "bool"


@dataclass(frozen=True)
class CompleteNumericBound:
    """Finite complete numeric bound attached to a prepared runtime input.

    The bound is computed from the complete prepared input used to compile the
    fast plan, never from the representative trace sample.  Generated
    ``init_inputs`` code revalidates every bound that is relied on by a stronger
    Cython semantic directive.  An artifact therefore fails closed if later
    input preparation escapes the proof domain.
    """

    lower: float
    upper: float
    authority: str
    input_key: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise SourcePreservingNativeError("complete numeric bounds must be finite")
        if self.lower > self.upper:
            raise SourcePreservingNativeError("complete numeric bound is inverted")

    def manifest(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "authority": self.authority,
            "input_key": self.input_key,
        }


@dataclass(frozen=True)
class NumericInterval:
    lower: float
    upper: float
    authorities: tuple[str, ...] = ()
    input_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise SourcePreservingNativeError("numeric interval is inverted")

    @classmethod
    def exact(cls, value: float, authority: str) -> "NumericInterval":
        value = float(value)
        return cls(value, value, (authority,), ())

    def combine(self, other: "NumericInterval", lower: float, upper: float, authority: str) -> "NumericInterval":
        return NumericInterval(
            float(lower),
            float(upper),
            tuple(sorted(set(self.authorities + other.authorities + (authority,)))),
            tuple(sorted(set(self.input_keys + other.input_keys))),
        )

    def join(self, other: "NumericInterval", authority: str) -> "NumericInterval":
        return NumericInterval(
            min(self.lower, other.lower),
            max(self.upper, other.upper),
            tuple(sorted(set(self.authorities + other.authorities + (authority,)))),
            tuple(sorted(set(self.input_keys + other.input_keys))),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "authorities": list(self.authorities),
            "input_keys": list(self.input_keys),
        }


@dataclass(frozen=True)
class CythonDirectiveEvidence:
    cpow_safe: bool = False
    cpow_authorities: tuple[str, ...] = ()
    required_input_bounds: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "cpow_safe": self.cpow_safe,
            "cpow_authorities": list(self.cpow_authorities),
            "required_input_bounds": list(self.required_input_bounds),
        }


@dataclass(frozen=True)
class CellCompilePlan:
    uid: str
    source_name: str
    source_fullname: str
    declared_result_dtype: str
    effective_result_dtype: str
    return_dtypes: tuple[str, ...]
    local_dtypes: tuple[tuple[str, str], ...]
    result_widened: bool
    formula_source: str
    directives: CythonDirectiveEvidence
    blockers: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.blockers

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_name": self.source_name,
            "source_fullname": self.source_fullname,
            "declared_result_dtype": self.declared_result_dtype,
            "effective_result_dtype": self.effective_result_dtype,
            "return_dtypes": list(self.return_dtypes),
            "local_dtypes": dict(self.local_dtypes),
            "result_widened": self.result_widened,
            "formula_source": self.formula_source,
            "directives": self.directives.manifest(),
            "blockers": list(self.blockers),
            "supported": self.supported,
        }


@dataclass(frozen=True)
class SourcePreservingCompilePlan:
    schema: str
    formula_program: NativeFormulaProgram
    cells: Mapping[str, CellCompilePlan]
    input_bounds: Mapping[str, CompleteNumericBound]
    supported: bool
    blockers: tuple[str, ...] = ()
    source_representation: str = "normalized_ast_structural_source_v1"

    @property
    def required_input_bound_keys(self) -> tuple[str, ...]:
        out: set[str] = set()
        for cell in self.cells.values():
            out.update(cell.directives.required_input_bounds)
        return tuple(sorted(out))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_representation": self.source_representation,
            "supported": self.supported,
            "blockers": list(self.blockers),
            "required_input_bound_keys": list(self.required_input_bound_keys),
            "input_bounds": {
                key: bound.manifest() for key, bound in sorted(self.input_bounds.items())
            },
            "cells": {uid: row.manifest() for uid, row in sorted(self.cells.items())},
            "formula_program": self.formula_program.manifest(),
        }


@dataclass(frozen=True)
class SourcePreservingDenseProgram:
    schema: str
    compile_plan: SourcePreservingCompilePlan
    dense_program: DenseRecursiveProgram

    @property
    def formula_program(self) -> NativeFormulaProgram:
        return self.dense_program.formula_program

    @property
    def memo_plan(self):
        return self.dense_program.memo_plan

    @property
    def supported(self) -> bool:
        return self.compile_plan.supported and self.dense_program.supported

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.compile_plan.blockers + self.dense_program.blockers)))

    @property
    def output_uid(self) -> str:
        return self.dense_program.output_uid

    @property
    def memoization_policy(self) -> str:
        return self.dense_program.memoization_policy

    @property
    def overflow_policy(self) -> str:
        return self.dense_program.overflow_policy

    @property
    def native_hash_debug(self) -> bool:
        return self.dense_program.native_hash_debug

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "supported": self.supported,
            "blockers": list(self.blockers),
            "compile_plan": self.compile_plan.manifest(),
            "dense_runtime": self.dense_program.manifest(),
        }


class _DTypeEmitterView:
    def __init__(self, cells: Mapping[str, str], input_specs: Mapping[str, Any], module_refs: Mapping[str, str]):
        self.v = {uid: type("_Cell", (), {"dtype": dtype})() for uid, dtype in cells.items()}
        self.input_specs = input_specs
        self.module_refs = module_refs


def _strict_expression_dtype(emitter: _DTypeEmitterView, node: ast.AST, locals_: Mapping[str, str]) -> str | None:
    result = expression_dtype(emitter, node, dict(locals_))
    if result is not None:
        return result
    # Architecture-B strict typing knows additional compiler-owned scalar leaves
    # that the shared expression typer deliberately does not. Preserve that
    # stronger evidence through conditional expressions instead of losing a
    # proven ABI merely because a typed leaf appears below an IfExp.
    if isinstance(node, ast.IfExp):
        body = _strict_expression_dtype(emitter, node.body, locals_)
        orelse = _strict_expression_dtype(emitter, node.orelse, locals_)
        if body is None or orelse is None:
            return None
        return _join_dtype(body, orelse)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "round" and node.args:
                base = _strict_expression_dtype(emitter, node.args[0], locals_)
                if base == "float64":
                    return "float64"
                if base in {"bool", "int64"} and len(node.args) == 1:
                    return "int64"
            if name in {"__erf__"}:
                return "float64"
            if name in {"__is_missing__", "__is_inf__"}:
                return "bool"
            if name in {"__step_lookup_1d__", "__interval_lookup_1d__", "__interp_lookup_1d__"}:
                return "float64"
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            receiver = node.func.value.id
            module_name = emitter.module_refs.get(receiver, receiver)
            if module_name in {"math", "numpy"} and node.func.attr in _LIBC_MATH_INTRINSICS:
                return "float64"
    if isinstance(node, ast.Subscript):
        # Any surviving ordinary subscript is deliberately not guessed. Native
        # provider subscripts should already be represented by normalized input
        # calls and are handled above.
        return None
    return None


def _collect_local_dtypes(
    fn: ast.FunctionDef,
    *,
    cell_dtypes: Mapping[str, str],
    input_specs: Mapping[str, Any],
    module_refs: Mapping[str, str],
    parameter_dtypes: Mapping[str, str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    emitter = _DTypeEmitterView(cell_dtypes, input_specs, module_refs)
    assignments: list[tuple[str, ast.AST]] = []
    augmented: list[tuple[str, ast.AST, ast.operator]] = []
    names: set[str] = set()
    loop_names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.append((target.id, node.value))
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            assignments.append((node.target.id, node.value))
            names.add(node.target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            augmented.append((node.target.id, node.value, node.op))
            names.add(node.target.id)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            loop_names.add(node.target.id)
            names.add(node.target.id)

    inferred: dict[str, str] = dict(parameter_dtypes)
    for name in loop_names:
        inferred[name] = "int64"
    blockers: set[str] = set()
    for _ in range(max(2, len(names) + 2)):
        changed = False
        for name, expr in assignments:
            new = _strict_expression_dtype(emitter, expr, inferred)
            if new is None:
                continue
            merged = _join_dtype(inferred.get(name), new)
            if merged is None:
                blockers.add(f"local_dtype_conflict:{name}")
                continue
            if inferred.get(name) != merged:
                inferred[name] = merged
                changed = True
        for name, expr, op in augmented:
            rhs = _strict_expression_dtype(emitter, expr, inferred)
            lhs = inferred.get(name)
            if lhs is None or rhs is None:
                continue
            if isinstance(op, ast.Div):
                merged = "float64"
            else:
                merged = _join_dtype(lhs, rhs)
            if merged is None:
                blockers.add(f"local_augassign_dtype_conflict:{name}")
                continue
            if inferred.get(name) != merged:
                inferred[name] = merged
                changed = True
        if not changed:
            break
    locals_only = {name: inferred[name] for name in names if name in inferred and name not in parameter_dtypes}
    return locals_only, tuple(sorted(blockers))


class _ReturnCollector(ast.NodeVisitor):
    def __init__(self, root: ast.FunctionDef):
        self.root = root
        self.returns: list[ast.Return] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        if node is self.root:
            for stmt in node.body:
                self.visit(stmt)
        # Nested functions are independent Python scopes and are not part of
        # this Cell ABI. They are outside N0 Stage 1A in any event.

    def visit_Lambda(self, node: ast.Lambda):  # noqa: N802
        return

    def visit_Return(self, node: ast.Return):  # noqa: N802
        self.returns.append(node)


def _whole_body_abi(
    cell: NativeFormulaCell,
    *,
    cell_dtypes: Mapping[str, str],
    input_specs: Mapping[str, Any],
    module_refs: Mapping[str, str],
) -> tuple[str | None, tuple[str, ...], dict[str, str], tuple[str, ...]]:
    if cell.function is None:
        return cell.dtype, (), {}, ()
    parameter_types = dict(zip(cell.parameters, cell.parameter_dtypes))
    local_dtypes, local_blockers = _collect_local_dtypes(
        cell.function,
        cell_dtypes=cell_dtypes,
        input_specs=input_specs,
        module_refs=module_refs,
        parameter_dtypes=parameter_types,
    )
    emitter = _DTypeEmitterView(cell_dtypes, input_specs, module_refs)
    env = {**parameter_types, **local_dtypes}
    collector = _ReturnCollector(cell.function)
    collector.visit(cell.function)
    blockers = set(local_blockers)
    if not collector.returns:
        blockers.add("no_explicit_return")
        return None, (), local_dtypes, tuple(sorted(blockers))
    return_dtypes: list[str] = []
    effective: str | None = None
    for node in collector.returns:
        if node.value is None:
            blockers.add("bare_return")
            continue
        dtype = _strict_expression_dtype(emitter, node.value, env)
        if dtype is None:
            blockers.add("unproven_return_dtype:" + ast.unparse(node.value))
            continue
        return_dtypes.append(dtype)
        effective = _join_dtype(effective, dtype)
    if effective is None:
        return None, tuple(return_dtypes), local_dtypes, tuple(sorted(blockers))
    effective = _join_dtype(cell.dtype, effective)
    if effective is None:
        blockers.add(f"result_dtype_conflict:{cell.dtype}:{','.join(return_dtypes)}")
    return effective, tuple(return_dtypes), local_dtypes, tuple(sorted(blockers))


def derive_complete_input_numeric_bounds(
    formula: NativeFormulaProgram,
    bound_inputs: Mapping[str, Any],
    *,
    run_population: int | None = None,
) -> dict[str, CompleteNumericBound]:
    """Derive finite bounds from complete prepared inputs, never trace samples."""

    specs = {spec.key: spec for spec in formula.inputs}
    out: dict[str, CompleteNumericBound] = {}
    for key, spec in specs.items():
        if spec.dtype not in _NATIVE_DTYPES or key not in bound_inputs:
            continue
        arr = np.asarray(bound_inputs[key])
        if arr.size == 0:
            continue
        if arr.dtype.kind not in "biuf":
            continue
        # Cython's stronger real-power semantics are only enabled from finite
        # proof domains. NaN/inf therefore make the bound unavailable rather
        # than being silently ignored by nanmin/nanmax.
        values = arr.astype(np.float64, copy=False)
        if not np.all(np.isfinite(values)):
            continue
        lower = float(np.min(values))
        upper = float(np.max(values))
        scope = "complete_prepared_global_input"
        if spec.scope == "point":
            scope = "complete_prepared_frozen_run_domain"
        authority = scope + f":{key}"
        if run_population is not None and spec.scope == "point":
            authority += f":population={int(run_population)}"
        out[key] = CompleteNumericBound(lower, upper, authority, key)
    return out


def _interval_from_input(bound: CompleteNumericBound) -> NumericInterval:
    return NumericInterval(
        bound.lower,
        bound.upper,
        (bound.authority,),
        (bound.input_key,),
    )


def _safe_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mul_bounds(a: NumericInterval, b: NumericInterval) -> tuple[float, float]:
    vals = (a.lower * b.lower, a.lower * b.upper, a.upper * b.lower, a.upper * b.upper)
    return min(vals), max(vals)


def _div_bounds(a: NumericInterval, b: NumericInterval) -> tuple[float, float] | None:
    if b.lower <= 0.0 <= b.upper:
        return None
    vals = (a.lower / b.lower, a.lower / b.upper, a.upper / b.lower, a.upper / b.upper)
    if not all(math.isfinite(x) for x in vals):
        return None
    return min(vals), max(vals)


class _CellIntervalAnalyzer:
    def __init__(self, formula: NativeFormulaProgram, input_bounds: Mapping[str, CompleteNumericBound]):
        self.formula = formula
        self.cells = formula.cells
        self.input_bounds = dict(input_bounds)
        self.cell_cache: dict[str, NumericInterval | None] = {}
        self.active: set[str] = set()

    def parameter_env(self, cell: NativeFormulaCell) -> dict[str, NumericInterval]:
        rows = intervals_from_rows(cell.parameter_int64_intervals)
        out: dict[str, NumericInterval] = {}
        for name, dtype in zip(cell.parameters, cell.parameter_dtypes):
            if dtype == "bool":
                out[name] = NumericInterval(0.0, 1.0, (f"native_bool_parameter:{cell.uid}:{name}",), ())
            elif dtype == "int64" and name in rows:
                r = rows[name]
                # Full signed-int64 bounds are exact as a representation domain,
                # even when not useful for tight arithmetic.
                out[name] = NumericInterval(float(r.lower), float(r.upper), tuple(r.authorities), ())
        return out

    def expr(self, node: ast.AST, env: Mapping[str, NumericInterval]) -> NumericInterval | None:
        if isinstance(node, ast.Constant):
            value = _safe_numeric(node.value)
            return None if value is None else NumericInterval.exact(value, "literal")
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.UnaryOp):
            x = self.expr(node.operand, env)
            if x is None:
                return None
            if isinstance(node.op, ast.USub):
                return NumericInterval(-x.upper, -x.lower, x.authorities + ("unary_minus",), x.input_keys)
            if isinstance(node.op, ast.UAdd):
                return x
            if isinstance(node.op, ast.Not):
                return NumericInterval(0.0, 1.0, x.authorities + ("boolean_not",), x.input_keys)
            return None
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            deps: set[str] = set()
            auth: set[str] = {"boolean_result"}
            for child in ast.iter_child_nodes(node):
                part = self.expr(child, env)
                if part is not None:
                    deps.update(part.input_keys); auth.update(part.authorities)
            return NumericInterval(0.0, 1.0, tuple(sorted(auth)), tuple(sorted(deps)))
        if isinstance(node, ast.IfExp):
            a = self.expr(node.body, env); b = self.expr(node.orelse, env)
            if a is None or b is None:
                return None
            return a.join(b, "if_expression_union")
        if isinstance(node, ast.BinOp):
            a = self.expr(node.left, env); b = self.expr(node.right, env)
            if a is None or b is None:
                return None
            if isinstance(node.op, ast.Add):
                return a.combine(b, a.lower + b.lower, a.upper + b.upper, "add")
            if isinstance(node.op, ast.Sub):
                return a.combine(b, a.lower - b.upper, a.upper - b.lower, "sub")
            if isinstance(node.op, ast.Mult):
                lo, hi = _mul_bounds(a, b)
                return a.combine(b, lo, hi, "mul")
            if isinstance(node.op, ast.Div):
                bounds = _div_bounds(a, b)
                return None if bounds is None else a.combine(b, bounds[0], bounds[1], "true_div")
            if isinstance(node.op, ast.FloorDiv):
                bounds = _div_bounds(a, b)
                if bounds is None:
                    return None
                return a.combine(b, math.floor(bounds[0]), math.floor(bounds[1]), "floor_div")
            if isinstance(node.op, ast.Mod):
                if b.lower == b.upper and b.lower != 0:
                    m = abs(b.lower)
                    return a.combine(b, 0.0, max(0.0, m - 1.0), "mod")
                return None
            if isinstance(node.op, ast.Pow):
                # Conservative real-valued monotone subset. This is enough to
                # propagate common actuarial rate transforms while failing
                # closed for negative bases or sign-changing exponents.
                if a.lower < 0 or b.lower < 0:
                    return None
                try:
                    vals = [
                        math.pow(a.lower, b.lower), math.pow(a.lower, b.upper),
                        math.pow(a.upper, b.lower), math.pow(a.upper, b.upper),
                    ]
                except (ValueError, OverflowError, ZeroDivisionError):
                    return None
                if not all(math.isfinite(x) for x in vals):
                    return None
                return a.combine(b, min(vals), max(vals), "nonnegative_real_pow")
            return None
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in self.cells:
                    return self.cell(name)
                if name in _RUNTIME_INPUT_CALLS:
                    if not node.args or not isinstance(node.args[0], ast.Constant):
                        return None
                    key = node.args[0].value
                    bound = self.input_bounds.get(key)
                    return None if bound is None else _interval_from_input(bound)
                if name in {"max", "min"} and node.args:
                    parts = [self.expr(x, env) for x in node.args]
                    vals = [x for x in parts if x is not None]
                    if not vals:
                        return None
                    partial = len(vals) != len(parts)
                    if partial:
                        # Python min/max start with the first argument and only
                        # replace it when a later comparison succeeds.  A known
                        # first argument therefore gives a safe one-sided bound
                        # even if a later unknown is NaN: min(1.0, NaN) remains
                        # 1.0 and max(0.0, NaN) remains 0.0.  The reverse order
                        # is not safe because min(NaN, 1.0) can remain NaN.
                        # Keep this order-sensitive so incomplete interval
                        # evidence never silently assumes mathematical min/max
                        # semantics where Python comparison semantics differ.
                        first = parts[0]
                        if first is None:
                            return None
                        if name == "max":
                            lo, hi = first.lower, math.inf
                        else:
                            lo, hi = -math.inf, first.upper
                        auth_vals = [first]
                    else:
                        if name == "max":
                            lo = max(x.lower for x in vals)
                            hi = max(x.upper for x in vals)
                        else:
                            lo = min(x.lower for x in vals)
                            hi = min(x.upper for x in vals)
                        auth_vals = vals
                    auth = tuple(sorted(
                        {a for x in auth_vals for a in x.authorities}
                        | {f"builtin_{name}" + ("_one_sided" if partial else "")}
                    ))
                    deps = tuple(sorted({k for x in auth_vals for k in x.input_keys}))
                    return NumericInterval(lo, hi, auth, deps)
                if name == "abs" and len(node.args) == 1:
                    x = self.expr(node.args[0], env)
                    if x is None:
                        return None
                    hi = max(abs(x.lower), abs(x.upper))
                    lo = 0.0 if x.lower <= 0 <= x.upper else min(abs(x.lower), abs(x.upper))
                    return NumericInterval(lo, hi, x.authorities + ("builtin_abs",), x.input_keys)
                if name in {"int", "float"} and len(node.args) == 1:
                    return self.expr(node.args[0], env)
                if name == "bool" and len(node.args) == 1:
                    x = self.expr(node.args[0], env)
                    return NumericInterval(0.0, 1.0, (() if x is None else x.authorities) + ("builtin_bool",), () if x is None else x.input_keys)
                if name in {"__is_missing__", "__is_inf__"}:
                    return NumericInterval(0.0, 1.0, (name,), ())
                if name == "__erf__":
                    x = self.expr(node.args[0], env) if node.args else None
                    return NumericInterval(-1.0, 1.0, (() if x is None else x.authorities) + ("erf_range",), () if x is None else x.input_keys)
                if name == "__sparse_table_lookup__" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    bound = self.input_bounds.get(node.args[1].value)
                    return None if bound is None else _interval_from_input(bound)
                if name == "__step_lookup_1d__" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    bound = self.input_bounds.get(node.args[1].value)
                    return None if bound is None else _interval_from_input(bound)
                if name == "__interval_lookup_1d__" and len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
                    bound = self.input_bounds.get(node.args[2].value)
                    return None if bound is None else _interval_from_input(bound)
                if name == "__interp_lookup_1d__" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    bound = self.input_bounds.get(node.args[1].value)
                    return None if bound is None else _interval_from_input(bound)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and len(node.args) == 1:
                receiver = node.func.value.id
                module_name = dict(self.formula.module_refs).get(receiver, receiver)
                x = self.expr(node.args[0], env)
                if module_name in {"math", "numpy"} and x is not None:
                    try:
                        if node.func.attr == "exp":
                            return NumericInterval(math.exp(x.lower), math.exp(x.upper), x.authorities + ("math_exp",), x.input_keys)
                        if node.func.attr == "sqrt" and x.lower >= 0:
                            return NumericInterval(math.sqrt(x.lower), math.sqrt(x.upper), x.authorities + ("math_sqrt",), x.input_keys)
                        if node.func.attr in {"sin", "cos"}:
                            return NumericInterval(-1.0, 1.0, x.authorities + ("bounded_trig",), x.input_keys)
                        if node.func.attr in {"floor", "ceil"}:
                            func = math.floor if node.func.attr == "floor" else math.ceil
                            return NumericInterval(float(func(x.lower)), float(func(x.upper)), x.authorities + ("math_rounding",), x.input_keys)
                    except (ValueError, OverflowError):
                        return None
            return None
        return None

    def block(self, statements: Sequence[ast.stmt], env: Mapping[str, NumericInterval]) -> tuple[dict[str, NumericInterval], list[NumericInterval], bool]:
        current = dict(env)
        returns: list[NumericInterval] = []
        complete = True
        for stmt in statements:
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                # Function docstrings are source metadata, not executable
                # numeric semantics.  Treating them as unknown statements would
                # unnecessarily destroy an otherwise complete interval proof.
                continue
            if isinstance(stmt, ast.Expr) and guard_code_from_stmt(stmt) is not None:
                # terminal guard branch
                return current, returns, complete
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                value = self.expr(stmt.value, current)
                if value is None:
                    # Losing a local interval does not by itself invalidate a
                    # later one-sided proof such as ``min(1.0, unknown)``.  The
                    # return expression remains responsible for proving the
                    # bound it actually needs.
                    current.pop(stmt.targets[0].id, None)
                else:
                    current[stmt.targets[0].id] = value
                continue
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                value = self.expr(stmt.value, current)
                if value is None:
                    current.pop(stmt.target.id, None)
                else:
                    current[stmt.target.id] = value
                continue
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                left = current.get(stmt.target.id); right = self.expr(stmt.value, current)
                if left is None or right is None:
                    current.pop(stmt.target.id, None); continue
                fake = ast.BinOp(ast.Constant(0), stmt.op, ast.Constant(0))
                # Reuse interval arithmetic with explicit temporary nodes would
                # lose dependencies; handle common ops directly.
                if isinstance(stmt.op, ast.Add):
                    current[stmt.target.id] = left.combine(right, left.lower + right.lower, left.upper + right.upper, "aug_add")
                elif isinstance(stmt.op, ast.Sub):
                    current[stmt.target.id] = left.combine(right, left.lower - right.upper, left.upper - right.lower, "aug_sub")
                elif isinstance(stmt.op, ast.Mult):
                    lo, hi = _mul_bounds(left, right); current[stmt.target.id] = left.combine(right, lo, hi, "aug_mul")
                else:
                    current.pop(stmt.target.id, None); complete = False
                continue
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    complete = False
                else:
                    value = self.expr(stmt.value, current)
                    if value is None:
                        complete = False
                    else:
                        returns.append(value)
                continue
            if isinstance(stmt, ast.If):
                env_a, ret_a, ok_a = self.block(stmt.body, current)
                env_b, ret_b, ok_b = self.block(stmt.orelse, current) if stmt.orelse else (dict(current), [], True)
                returns.extend(ret_a); returns.extend(ret_b)
                merged: dict[str, NumericInterval] = {}
                for name in set(env_a) & set(env_b):
                    merged[name] = env_a[name].join(env_b[name], "if_local_union")
                current = merged
                complete = complete and ok_a and ok_b
                continue
            if isinstance(stmt, (ast.For, ast.While, ast.Try, ast.With, ast.Match)):
                # Stage 1A interval evidence deliberately does not interpret
                # arbitrary control flow. Cython may still compile the formula;
                # the only lost feature here is stronger numeric directives.
                complete = False
                continue
            if isinstance(stmt, (ast.Pass, ast.Continue, ast.Break)):
                continue
            # Unknown statement: interval proof becomes partial, not formula
            # rejection. Native eligibility remains Cython's job.
            complete = False
        return current, returns, complete

    def cell(self, uid: str) -> NumericInterval | None:
        if uid in self.cell_cache:
            return self.cell_cache[uid]
        if uid in self.active:
            return None
        cell = self.cells.get(uid)
        if cell is None or cell.function is None or cell.reduction is not None:
            self.cell_cache[uid] = None
            return None
        self.active.add(uid)
        try:
            _env, returns, complete = self.block(cell.function.body, self.parameter_env(cell))
            if not returns or not complete:
                result = None
            else:
                result = returns[0]
                for other in returns[1:]:
                    result = result.join(other, f"cell_return_union:{uid}")
                result = NumericInterval(
                    result.lower,
                    result.upper,
                    tuple(sorted(set(result.authorities + (f"complete_formula_interval:{uid}",)))),
                    result.input_keys,
                )
            self.cell_cache[uid] = result
            return result
        finally:
            self.active.remove(uid)

    def cpow_evidence(self, cell: NativeFormulaCell) -> CythonDirectiveEvidence:
        if cell.function is None:
            return CythonDirectiveEvidence()
        pow_nodes = [node for node in ast.walk(cell.function) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)]
        if not pow_nodes:
            return CythonDirectiveEvidence()
        # Build a conservative local environment once. The common actuarial
        # power transforms are direct return expressions; local assignments are
        # resolved by the block interpreter when it can prove them.
        env = self.parameter_env(cell)
        local_env, _returns, _complete = self.block(cell.function.body, env)
        env = {**env, **local_env}
        authorities: set[str] = set()
        keys: set[str] = set()
        emitter = _DTypeEmitterView(
            {uid: c.dtype for uid, c in self.cells.items()},
            {x.key: x for x in self.formula.inputs},
            dict(self.formula.module_refs),
        )
        param_types = dict(zip(cell.parameters, cell.parameter_dtypes))
        local_types, _ = _collect_local_dtypes(
            cell.function,
            cell_dtypes={uid: c.dtype for uid, c in self.cells.items()},
            input_specs={x.key: x for x in self.formula.inputs},
            module_refs=dict(self.formula.module_refs),
            parameter_dtypes=param_types,
        )
        dtype_env = {**param_types, **local_types}
        for node in pow_nodes:
            base = self.expr(node.left, env)
            exponent = self.expr(node.right, env)
            base_dtype = _strict_expression_dtype(emitter, node.left, dtype_env)
            exponent_dtype = _strict_expression_dtype(emitter, node.right, dtype_env)
            # cpow is useful/safe here only for real floating power. Leave
            # integral exponentiation to Cython's Python-compatible semantics.
            if "float64" not in {base_dtype, exponent_dtype}:
                return CythonDirectiveEvidence()
            if base is None or exponent is None:
                return CythonDirectiveEvidence()
            if base.lower < 0.0:
                return CythonDirectiveEvidence()
            if base.lower == 0.0 and exponent.lower < 0.0:
                return CythonDirectiveEvidence()
            authorities.update(base.authorities)
            authorities.update(exponent.authorities)
            authorities.add("cpow_equivalence:nonnegative_real_base")
            keys.update(base.input_keys)
            keys.update(exponent.input_keys)
        return CythonDirectiveEvidence(
            cpow_safe=True,
            cpow_authorities=tuple(sorted(authorities)),
            required_input_bounds=tuple(sorted(keys)),
        )


def build_source_preserving_compile_plan(
    formula: NativeFormulaProgram,
    *,
    input_bounds: Mapping[str, CompleteNumericBound] | None = None,
    lexical_sources: Mapping[str, str] | None = None,
) -> SourcePreservingCompilePlan:
    """Build source-preserving formula metadata and widen unsafe trace ABIs.

    Stage 1B can feed this pass a source-transparent candidate program that has
    not crossed the historical statement whitelist. The old emitter remains a
    differential oracle.
    """

    input_bounds = dict(input_bounds or {})
    lexical_sources = dict(lexical_sources or {})
    input_specs = {x.key: x for x in formula.inputs}
    module_refs = dict(formula.module_refs)
    effective = {uid: cell.dtype for uid, cell in formula.cells.items()}
    analyses: dict[str, tuple[str | None, tuple[str, ...], dict[str, str], tuple[str, ...]]] = {}

    # Result widening can propagate through native Cell calls. Iterate only over
    # the selected formula program; this is an ABI join, not a value-domain
    # fixed point and does not infer execution topology.
    for _ in range(max(2, len(formula.cells) + 1)):
        changed = False
        analyses.clear()
        for uid, cell in formula.cells.items():
            if (
                cell.execution_role != "formula"
                or cell.reduction is not None
                or (cell.logical_result_ndim == 1 and cell.ranked_function is not None)
            ):
                # Rank-1 formulas use the dedicated Cython/NumPy ABI. Their
                # traced element dtype and fixed geometry are already the ABI
                # authority; scalar whole-body typing would incorrectly reject
                # intentional array construction/slicing/matmul expressions.
                analyses[uid] = (cell.dtype, (), dict(cell.local_scalar_dtypes), ())
                continue
            row = _whole_body_abi(
                cell,
                cell_dtypes=effective,
                input_specs=input_specs,
                module_refs=module_refs,
            )
            analyses[uid] = row
            inferred = row[0]
            if inferred is not None:
                widened = _join_dtype(effective[uid], inferred)
                if widened is not None and widened != effective[uid]:
                    effective[uid] = widened
                    changed = True
        if not changed:
            break

    widened_cells: dict[str, NativeFormulaCell] = {}
    cell_plans: dict[str, CellCompilePlan] = {}
    global_blockers: set[str] = set()
    for uid, cell in formula.cells.items():
        inferred, returns, locals_, blockers = analyses.get(uid, (cell.dtype, (), {}, ()))
        eff = effective.get(uid, cell.dtype)
        if cell.execution_role == "formula":
            if inferred is None:
                blockers = tuple(sorted(set(blockers + ("native_result_abi_unproven",))))
            if eff not in _NATIVE_DTYPES:
                blockers = tuple(sorted(set(blockers + (f"unsupported_effective_result_dtype:{eff}",))))
        else:
            # Materialized helpers are preparation/provider metadata. They are
            # carried through the NativeFormulaProgram for provenance but are
            # not emitted as recursive calc/r functions and therefore do not
            # participate in the N0 formula ABI gate.
            blockers = ()
        widened_cell = replace(
            cell,
            dtype=eff,
            local_scalar_dtypes=tuple(sorted(locals_.items())) if locals_ else cell.local_scalar_dtypes,
        )
        widened_cells[uid] = widened_cell
        for blocker in blockers:
            global_blockers.add(f"{uid}:{blocker}")
        formula_source = lexical_sources.get(uid) or (
            "<reduction>" if cell.function is None else ast.unparse(cell.function)
        )
        cell_plans[uid] = CellCompilePlan(
            uid=uid,
            source_name=cell.source_name,
            source_fullname=cell.source_fullname,
            declared_result_dtype=cell.dtype,
            effective_result_dtype=eff,
            return_dtypes=returns,
            local_dtypes=tuple(sorted(locals_.items())),
            result_widened=eff != cell.dtype,
            formula_source=formula_source,
            directives=CythonDirectiveEvidence(),
            blockers=blockers,
        )

    widened_formula = replace(
        formula,
        schema="modelx_graph.native_formula.source_preserving.v1",
        uid=formula.uid + "_sp1",
        cells=widened_cells,
        provenance=formula.provenance + "+source_preserving_abi_v1",
    )

    # Derive Cython directives only after ABI widening, so interval/callee type
    # reasoning sees the actual ABI that will be generated.
    analyzer = _CellIntervalAnalyzer(widened_formula, input_bounds)
    final_plans: dict[str, CellCompilePlan] = {}
    for uid, row in cell_plans.items():
        ranked = (
            widened_cells[uid].logical_result_ndim == 1
            and widened_cells[uid].ranked_function is not None
        )
        directives = (
            CythonDirectiveEvidence()
            if ranked
            else analyzer.cpow_evidence(widened_cells[uid])
            if row.supported
            else CythonDirectiveEvidence()
        )
        final_plans[uid] = replace(row, directives=directives)

    return SourcePreservingCompilePlan(
        schema="modelx_graph.source_preserving_compile_plan.v1",
        formula_program=widened_formula,
        cells=final_plans,
        input_bounds=input_bounds,
        supported=not global_blockers,
        blockers=tuple(sorted(global_blockers)),
        source_representation=(
            "modelx_cython_libcst_lexical_source+normalized_ast_native_bindings_v2"
            if lexical_sources
            else "normalized_ast_structural_source_v1"
        ),
    )


def build_source_preserving_dense_program(
    formula: NativeFormulaProgram,
    *,
    parameter_ranges: Mapping[str, Sequence[tuple[int, int] | None]] | None = None,
    input_bounds: Mapping[str, CompleteNumericBound] | None = None,
    lexical_sources: Mapping[str, str] | None = None,
    max_dense_slots_per_cell: int = 1_000_000,
    native_hash_capacity_by_uid: Mapping[str, int] | None = None,
    native_hash_debug: bool = False,
) -> SourcePreservingDenseProgram:
    compile_plan = build_source_preserving_compile_plan(
        formula,
        input_bounds=input_bounds,
        lexical_sources=lexical_sources,
    )
    if not compile_plan.supported:
        # Still construct a diagnostic wrapper without trying to physicalize an
        # ABI that was proven inconsistent.
        raise SourcePreservingNativeError(
            "source-preserving compile plan blocked: " + "; ".join(compile_plan.blockers)
        )
    dense = build_dense_recursive_program(
        compile_plan.formula_program,
        parameter_ranges=parameter_ranges,
        max_dense_slots_per_cell=max_dense_slots_per_cell,
        native_hash_capacity_by_uid=native_hash_capacity_by_uid,
        native_hash_debug=native_hash_debug,
    )
    return SourcePreservingDenseProgram(
        schema="modelx_graph.source_preserving_dense_program.v1",
        compile_plan=compile_plan,
        dense_program=dense,
    )


_ROUND_COMPAT_MIN_DIGITS = -22
_ROUND_COMPAT_MAX_DIGITS = 22

# CPython's short-float-repr ``round(float, ndigits)`` uses correctly rounded
# decimal conversion.  Cython 3.2 cannot lower that operation in nogil code.
# The compatibility helper below implements the same mathematical operation for
# the deliberately narrow integer precision interval [-22, 22] using exact
# IEEE-754 decomposition and unsigned 128-bit integer arithmetic.  It contains
# no model/product semantics and is injected only when an admitted native Cell
# actually contains round().
_ROUND_COMPAT_C = r"""
#include <stdint.h>
#include <math.h>

typedef __uint128_t mx_u128;

static const uint64_t mx_round_pow5[23] = {
    1ULL, 5ULL, 25ULL, 125ULL, 625ULL, 3125ULL, 15625ULL, 78125ULL,
    390625ULL, 1953125ULL, 9765625ULL, 48828125ULL, 244140625ULL,
    1220703125ULL, 6103515625ULL, 30517578125ULL, 152587890625ULL,
    762939453125ULL, 3814697265625ULL, 19073486328125ULL,
    95367431640625ULL, 476837158203125ULL, 2384185791015625ULL
};

static int mx_round_bitlen128(mx_u128 x) {
    uint64_t hi = (uint64_t)(x >> 64);
    uint64_t lo = (uint64_t)x;
    if (hi) return 64 + (64 - __builtin_clzll(hi));
    if (lo) return 64 - __builtin_clzll(lo);
    return 0;
}

static mx_u128 mx_round_pow10u(int n) {
    return ((mx_u128)mx_round_pow5[n]) << n;
}

static mx_u128 mx_round_div_u128(mx_u128 num, mx_u128 den) {
    mx_u128 q = num / den;
    mx_u128 r = num - q * den;
    mx_u128 other = den - r;
    if (r > other || (r == other && (q & 1))) q++;
    return q;
}

static mx_u128 mx_round_div_pow2(mx_u128 value, int shift) {
    mx_u128 q, rem, half;
    if (shift <= 0) return value << (-shift);
    if (shift >= 128) return 0;
    q = value >> shift;
    rem = value - (q << shift);
    half = ((mx_u128)1) << (shift - 1);
    if (rem > half || (rem == half && (q & 1))) q++;
    return q;
}

/* Correctly round positive num/den to binary64, then restore the sign.  The
   admitted decimal precision bound keeps all intermediate shifts <= 127. */
static double mx_round_rational_to_double(mx_u128 num, mx_u128 den, int neg) {
    int bn, bd, exponent, shift;
    mx_u128 scaled_num, scaled_den, mant, rem, other;
    double out;
    if (!num) return copysign(0.0, neg ? -1.0 : 1.0);
    bn = mx_round_bitlen128(num);
    bd = mx_round_bitlen128(den);
    exponent = bn - bd;
    if (exponent >= 0) {
        if (num < (den << exponent)) exponent--;
    }
    else {
        if ((num << (-exponent)) < den) exponent--;
    }
    shift = 52 - exponent;
    if (shift >= 0) {
        scaled_num = num << shift;
        scaled_den = den;
    }
    else {
        scaled_num = num;
        scaled_den = den << (-shift);
    }
    mant = scaled_num / scaled_den;
    rem = scaled_num - mant * scaled_den;
    other = scaled_den - rem;
    if (rem > other || (rem == other && (mant & 1))) mant++;
    if (mant == (((mx_u128)1) << 53)) {
        mant >>= 1;
        exponent++;
    }
    out = ldexp((double)(uint64_t)mant, exponent - 52);
    return neg ? -out : out;
}

/* Correctly round the exact integer value a * 2**scale2 to binary64. */
static double mx_round_scaled_integer_to_double(
    mx_u128 a, int scale2, int neg, int *err
) {
    int bits, exponent, shift;
    mx_u128 mant, rem, half;
    double out;
    if (!a) return copysign(0.0, neg ? -1.0 : 1.0);
    bits = mx_round_bitlen128(a);
    exponent = (bits - 1) + scale2;
    if (exponent > 1023) {
        *err = 1;
        return 0.0;
    }
    if (bits <= 53) {
        mant = a << (53 - bits);
    }
    else {
        shift = bits - 53;
        mant = a >> shift;
        rem = a - (mant << shift);
        half = ((mx_u128)1) << (shift - 1);
        if (rem > half || (rem == half && (mant & 1))) mant++;
        if (mant == (((mx_u128)1) << 53)) {
            mant >>= 1;
            exponent++;
            if (exponent > 1023) {
                *err = 1;
                return 0.0;
            }
        }
    }
    out = ldexp((double)(uint64_t)mant, exponent - 52);
    return neg ? -out : out;
}

static double mx_python_float_round_compat(double x, long long digits, int *err) {
    union { double d; uint64_t u; } bits;
    uint64_t frac, mantissa;
    unsigned expraw;
    int neg, exponent2, n, k, shift;
    mx_u128 exact, rounded, den, scaled;

    *err = 0;
    if (!isfinite(x) || x == 0.0) return x;
    if (digits < -22 || digits > 22) {
        *err = 2;
        return 0.0;
    }

    bits.d = x;
    neg = (int)(bits.u >> 63);
    expraw = (unsigned)((bits.u >> 52) & 0x7ffU);
    frac = bits.u & 0xfffffffffffffULL;
    if (expraw == 0) {
        mantissa = frac;
        exponent2 = -1074;
    }
    else {
        mantissa = (1ULL << 52) | frac;
        exponent2 = (int)expraw - 1023 - 52;
    }

    if (digits >= 0) {
        n = (int)digits;
        exact = (mx_u128)mantissa * (mx_u128)mx_round_pow5[n];
        shift = -(exponent2 + n);
        /* x * 10**n is already integral, so decimal rounding is identity. */
        if (shift <= 0) return x;
        rounded = mx_round_div_pow2(exact, shift);
        if (!rounded) return copysign(0.0, x);
        return mx_round_rational_to_double(
            rounded, mx_round_pow10u(n), neg
        );
    }

    k = (int)(-digits);
    /* If one binary ulp is wider than the decimal rounding quantum, the
       correctly rounded decimal target converts straight back to x. */
    if (exponent2 >= 0) {
        if (exponent2 >= 128 || (((mx_u128)1) << exponent2) > mx_round_pow10u(k))
            return x;
    }

    shift = exponent2 - k;
    if (shift >= 0) {
        if (mx_round_bitlen128((mx_u128)mantissa) + shift > 128) {
            *err = 3;
            return 0.0;
        }
        exact = ((mx_u128)mantissa) << shift;
        den = (mx_u128)mx_round_pow5[k];
    }
    else {
        int dshift = -shift;
        if (mx_round_bitlen128((mx_u128)mx_round_pow5[k]) + dshift > 128)
            return copysign(0.0, x);
        exact = (mx_u128)mantissa;
        den = ((mx_u128)mx_round_pow5[k]) << dshift;
    }
    rounded = mx_round_div_u128(exact, den);
    if (!rounded) return copysign(0.0, x);
    if (
        mx_round_bitlen128(rounded)
        + mx_round_bitlen128((mx_u128)mx_round_pow5[k]) > 128
    ) {
        *err = 4;
        return 0.0;
    }
    scaled = rounded * (mx_u128)mx_round_pow5[k];
    return mx_round_scaled_integer_to_double(scaled, k, neg, err);
}
"""


def _round_compat_helper_lines() -> list[str]:
    code = _ROUND_COMPAT_C
    if os.name == "nt":
        from .portable_rounding import PORTABLE_U128_CXX
        code = code.replace("typedef __uint128_t mx_u128;", PORTABLE_U128_CXX)
        code = code.replace("__builtin_clzll", "mx_clz64")
    lines = ["cdef extern from *:", '    """']
    lines.extend("    " + line for line in code.strip("\\n").splitlines())
    lines += [
        '    """',
        "    double mx_python_float_round_compat(double x, long long digits, int *err) noexcept nogil",
        "",
        "cdef inline double _native_round(double x, long long digits) noexcept nogil:",
        "    cdef int err = 0",
        "    cdef double out",
        "    global _guard_error",
        f"    if digits < {_ROUND_COMPAT_MIN_DIGITS} or digits > {_ROUND_COMPAT_MAX_DIGITS}:",
        "        _guard_error = -93501",
        "        return 0.0",
        "    out = mx_python_float_round_compat(x, digits, &err)",
        "    if err != 0:",
        "        _guard_error = -93502",
        "        return 0.0",
        "    return out",
        "",
    ]
    return lines


def _real_power_operand_dtype(emitter, node, env):
    """Prove scalar numeric operands; never infer through an unknown operand."""
    if isinstance(node, ast.Constant):
        return "float64" if type(node.value) is float else "int64" if type(node.value) is int else None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _real_power_operand_dtype(emitter, node.operand, env)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
        a = _real_power_operand_dtype(emitter, node.left, env)
        b = _real_power_operand_dtype(emitter, node.right, env)
        if a not in {"int64", "float64"} or b not in {"int64", "float64"}:
            return None
        return "float64" if "float64" in (a, b) or isinstance(node.op, (ast.Div, ast.Pow)) else "int64"
    if isinstance(node, ast.IfExp):
        a = _real_power_operand_dtype(emitter, node.body, env)
        b = _real_power_operand_dtype(emitter, node.orelse, env)
        return ("float64" if "float64" in (a, b) else "int64") if a in {"int64", "float64"} and b in {"int64", "float64"} else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        cell = emitter.v.get(name)
        if cell is not None and getattr(cell, "logical_result_ndim", 0) == 0:
            return cell.dtype if cell.dtype in {"int64", "float64"} else None
        if name in _RUNTIME_INPUT_CALLS and node.args and isinstance(node.args[0], ast.Constant):
            spec = emitter.input_specs.get(node.args[0].value)
            return spec.dtype if spec is not None and spec.dtype in {"int64", "float64"} else None
        if name in {"min", "max", "abs", "float", "int"} and node.args and not node.keywords:
            kinds = [_real_power_operand_dtype(emitter, arg, env) for arg in node.args]
            if all(kind in {"int64", "float64"} for kind in kinds):
                return "float64" if name == "float" or (name != "int" and "float64" in kinds) else "int64"
    return None


def _checked_real_power_helper_lines() -> list[str]:
    return [
        "from libc.math cimport pow as _checked_c_pow, floor as _checked_c_floor, isfinite as _checked_c_isfinite",
        "cdef inline double _native_checked_real_pow(double b, double e) noexcept nogil:",
        "    global _guard_error",
        "    if _checked_c_isfinite(b) and _checked_c_isfinite(e):",
        "        if b < 0.0 and e != _checked_c_floor(e):",
        "            _guard_error = -93503",
        "            return 0.0",
        "        if b == 0.0 and e < 0.0:",
        "            _guard_error = -93504",
        "            return 0.0",
        "    return _checked_c_pow(b, e)",
        "",
    ]


class _ModelxLeafRewriter(ast.NodeTransformer):
    """Rewrite only compiler-owned modelx/runtime leaves, preserving Python syntax."""

    def __init__(
        self,
        generator: "SourcePreservingDenseCythonGenerator",
        cell: NativeFormulaCell,
        policy_index_name: str,
    ):
        self.g = generator
        self.cell = cell
        self.policy_index_name = policy_index_name
        mixed_program = getattr(generator, "mixed_program", None)
        self.python_semantics = (
            cell.uid in getattr(mixed_program, "core_cache_cython_p_uids", ())
            or cell.uid in getattr(mixed_program, "python_key_backed_uids", ())
        )
        self._opaque_power_scope = 0
        self._function_depth = 0
        self._python_power_env = {}

    def visit_FunctionDef(self, node):  # noqa: N802
        self._function_depth += 1
        try:
            if self._function_depth > 1:
                return self._visit_power_scope(node)
            if self.python_semantics:
                from collections import Counter
                writes = Counter(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
                categories = {name for name, domain in zip(self.cell.parameters, getattr(self.cell, "parameter_category_domains", ())) if domain is not None}
                env = {name: dtype for name, dtype in zip(self.cell.parameters, self.cell.parameter_dtypes) if not writes[name] and name not in categories}
                # Only single assignments in the outer straight-line body.
                # Do not borrow native local types for arbitrary Python objects.
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        name = stmt.targets[0].id
                        if writes[name] == 1:
                            kind = _real_power_operand_dtype(self.g, stmt.value, env)
                            if kind in {"int64", "float64"}:
                                env[name] = kind
                self._python_power_env = env
            return self.generic_visit(node)
        finally:
            self._function_depth -= 1

    def _visit_power_scope(self, node):
        # Comprehension/lambda bindings can shadow a typed formula local.
        self._opaque_power_scope += 1
        try:
            return self.generic_visit(node)
        finally:
            self._opaque_power_scope -= 1

    visit_GeneratorExp = _visit_power_scope
    visit_ListComp = _visit_power_scope
    visit_SetComp = _visit_power_scope
    visit_DictComp = _visit_power_scope
    visit_Lambda = _visit_power_scope

    def visit_For(self, node):  # noqa: N802
        # Materialize a short numeric tuple once, in source order, then use a
        # C range loop. Keep the loop itself so break/continue/else retain their
        # Python meaning, including changes to the target inside the body.
        if (self.python_semantics or not isinstance(node.iter, ast.Tuple)
                or not isinstance(node.target, ast.Name) or not 1 <= len(node.iter.elts) <= 16):
            return self.generic_visit(node)
        row = self.g.compile_plan.cells.get(self.cell.uid)
        env = dict(zip(self.cell.parameters, self.cell.parameter_dtypes))
        if row is not None:
            env.update(dict(row.local_dtypes))
        kinds = [_real_power_operand_dtype(self.g, elt, env) for elt in node.iter.elts]
        if (any(kind not in {"int64", "float64"} for kind in kinds)
                or env.get(node.target.id) not in {"int64", "float64"}
                or ("float64" in kinds and env[node.target.id] != "float64")):
            return self.generic_visit(node)
        names = getattr(self, "_tuple_names", None)
        if names is None:
            fn = getattr(self.cell, "function", None)
            names = {n.id for n in ast.walk(fn or node) if isinstance(n, ast.Name)}
            self._tuple_names = names
        def fresh():
            index = len(names)
            name = f"__mx_fixed_tuple_{index}"
            while name in names:
                index += 1
                name = f"__mx_fixed_tuple_{index}"
            names.add(name)
            return name
        temps = [fresh() for _ in node.iter.elts]
        counter = fresh()
        node = self.generic_visit(node)
        prepared = [ast.Assign([ast.Name(name, ast.Store())], value) for name, value in zip(temps, node.iter.elts)]
        select = ast.Name(temps[-1], ast.Load())
        for i in reversed(range(len(temps) - 1)):
            select = ast.IfExp(ast.Compare(ast.Name(counter, ast.Load()), [ast.Eq()], [ast.Constant(i)]), ast.Name(temps[i], ast.Load()), select)
        loop = ast.For(ast.Name(counter, ast.Store()), ast.Call(ast.Name("range", ast.Load()), [ast.Constant(len(temps))], []),
                       [ast.Assign([node.target], select), *node.body], node.orelse)
        return [ast.copy_location(stmt, node) for stmt in [*prepared, loop]]

    def visit_BinOp(self, node):  # noqa: N802
        checked = False
        if isinstance(node.op, ast.Pow) and not self._opaque_power_scope:
            row = self.g.compile_plan.cells.get(self.cell.uid)
            env = dict(zip(self.cell.parameters, self.cell.parameter_dtypes))
            if row is not None:
                env.update(dict(row.local_dtypes))
            if self.python_semantics:
                env = self._python_power_env
            # Integer exponents keep Cython's existing semantics (2 ** -t).
            checked = (
                _real_power_operand_dtype(self.g, node.left, env) in {"int64", "float64"}
                and _real_power_operand_dtype(self.g, node.right, env) == "float64"
            )
        node = self.generic_visit(node)
        if checked:
            return ast.copy_location(ast.Call(
                ast.Name("_native_checked_real_pow", ast.Load()), [node.left, node.right], []
            ), node)
        return node

    def visit_Expr(self, node: ast.Expr):  # noqa: N802
        code = guard_code_from_stmt(node)
        if code is not None:
            zero: int | float = 0.0 if self.cell.dtype == "float64" else 0
            return [
                ast.copy_location(ast.Assign([ast.Name("_guard_error", ast.Store())], ast.Constant(code)), node),
                ast.copy_location(ast.Return(ast.Constant(zero)), node),
            ]
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):  # noqa: N802
        conversion_dtype = None
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"int", "float", "bool"}
            and len(node.args) == 1
        ):
            row = self.g.compile_plan.cells.get(self.cell.uid)
            env = dict(zip(self.cell.parameters, self.cell.parameter_dtypes))
            if row is not None:
                env.update(dict(row.local_dtypes))
            conversion_dtype = expression_dtype(self.g, node.args[0], env)

        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.g.v:
                target = self.g.v[name]
                if len(node.args) != target.arity or node.keywords:
                    raise SourcePreservingNativeError(
                        f"{self.cell.source_fullname}: native call {name} expected {target.arity} positional args"
                    )
                return ast.copy_location(
                    ast.Call(
                        ast.Name(f"r_{name}", ast.Load()),
                        [ast.Name(self.policy_index_name, ast.Load()), *node.args],
                        [],
                    ),
                    node,
                )
            if name in _RUNTIME_INPUT_CALLS:
                return self._input_call(name, node)
            if name == "__is_missing__" and len(node.args) == 1:
                return ast.copy_location(ast.Call(ast.Name("isnan", ast.Load()), node.args, []), node)
            if name == "__is_inf__" and len(node.args) == 1:
                return ast.copy_location(ast.Call(ast.Name("isinf", ast.Load()), node.args, []), node)
            if name == "__erf__" and len(node.args) == 1:
                return ast.copy_location(ast.Call(ast.Name("erf", ast.Load()), node.args, []), node)
            if name == "int" and len(node.args) == 1:
                # ``generic_visit`` may have rewritten round(x[, const]) into
                # our compiler-owned double helper.  The pre-rewrite dtype
                # inference does not know that private helper name, so recover
                # the type here instead of leaving Python int() in nogil code.
                effective_conversion_dtype = conversion_dtype
                if (
                    effective_conversion_dtype is None
                    and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name)
                    and node.args[0].func.id == "_native_round"
                ):
                    effective_conversion_dtype = "float64"
                if effective_conversion_dtype in _NATIVE_DTYPES:
                    if effective_conversion_dtype in {"bool", "int64"}:
                        return node.args[0]
                    return ast.copy_location(
                        ast.Call(ast.Name("_native_int_from_double", ast.Load()), node.args, []),
                        node,
                    )
            if name == "float" and len(node.args) == 1 and conversion_dtype in _NATIVE_DTYPES:
                if conversion_dtype == "float64":
                    return node.args[0]
                return ast.copy_location(ast.Call(ast.Name("_native_float_from_long", ast.Load()), node.args, []), node)
            if name == "bool" and len(node.args) == 1 and conversion_dtype in _NATIVE_DTYPES:
                if conversion_dtype == "bool":
                    return node.args[0]
                return ast.copy_location(ast.Compare(node.args[0], [ast.NotEq()], [ast.Constant(0)]), node)
            if name == "round" and 1 <= len(node.args) <= 2 and not self.python_semantics:
                # Cython 3.2 cannot lower Python float round(x, ndigits) in
                # native nogil code after the operands are resolved.  Planner
                # admission has already proved the two-argument compatibility
                # domain, so this is the final narrow compatibility boundary;
                # the surrounding Python formula remains Cython-owned.
                digits = ast.Constant(0) if len(node.args) == 1 else node.args[1]
                return ast.copy_location(
                    ast.Call(ast.Name("_native_round", ast.Load()), [node.args[0], digits], []),
                    node,
                )
            if name == "__exact_axis_code__" and len(node.args) == 4:
                value, start, step, _size = node.args
                return ast.copy_location(ast.BinOp(ast.BinOp(value, ast.Sub(), start), ast.FloorDiv(), step), node)
            if name == "__exact_axis_code_valid__" and len(node.args) == 5:
                value, valid, start, step, size = node.args
                code = ast.BinOp(ast.BinOp(value, ast.Sub(), start), ast.FloorDiv(), step)
                test = ast.BoolOp(ast.And(), [
                    valid,
                    ast.Compare(copy.deepcopy(code), [ast.GtE()], [ast.Constant(0)]),
                    ast.Compare(copy.deepcopy(code), [ast.Lt()], [size]),
                ])
                return ast.copy_location(
                    ast.IfExp(test, code, ast.Call(ast.Name("__exact_axis_invalid__", ast.Load()), [ast.Constant(-93002)], [])),
                    node,
                )
            if name == "__prepared_point_lookup__" and len(node.args) == 2:
                if not all(isinstance(arg, ast.Constant) for arg in node.args):
                    raise SourcePreservingNativeError(
                        "prepared point lookup requires literal prepared-input keys"
                    )
                value_key, valid_key = (arg.value for arg in node.args)
                value_spec = self.g.input_specs.get(value_key)
                valid_spec = self.g.input_specs.get(valid_key)
                if value_spec is None or valid_spec is None:
                    raise SourcePreservingNativeError("unknown prepared point lookup input")
                value_expr = self._input_call(
                    "point_input", ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(value_key)], [])
                )
                valid_expr = self._input_call(
                    "point_input", ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(valid_key)], [])
                )
                return ast.copy_location(
                    ast.IfExp(
                        valid_expr,
                        value_expr,
                        ast.Call(
                            ast.Name("__exact_axis_invalid__", ast.Load()),
                            [ast.Constant(-93003)],
                            [],
                        ),
                    ),
                    node,
                )
            if name == "__sparse_table_lookup__" and len(node.args) >= 3:
                if not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[1], ast.Constant)):
                    raise SourcePreservingNativeError("sparse lookup requires literal prepared-input keys")
                keys_key, values_key = node.args[0].value, node.args[1].value
                spec = self.g.input_specs.get(values_key)
                if spec is None:
                    raise SourcePreservingNativeError(f"unknown sparse value input {values_key!r}")
                helper = f"_sparse_table_lookup_{len(node.args)-2}_{spec.dtype}"
                return ast.copy_location(
                    ast.Call(ast.Name(helper, ast.Load()), [ast.Name(keys_key, ast.Load()), ast.Name(values_key, ast.Load()), *node.args[2:]], []),
                    node,
                )
            if name == "__step_lookup_1d__" and len(node.args) == 5:
                if not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[1], ast.Constant)):
                    raise SourcePreservingNativeError("step lookup requires literal prepared-input keys")
                return ast.copy_location(ast.Call(ast.Name("_step_lookup_1d", ast.Load()), [ast.Name(node.args[0].value, ast.Load()), ast.Name(node.args[1].value, ast.Load()), *node.args[2:]], []), node)
            if name == "__interval_lookup_1d__" and len(node.args) in {4, 7}:
                if not all(isinstance(x, ast.Constant) for x in node.args[:3]):
                    raise SourcePreservingNativeError("interval lookup requires literal prepared-input keys")
                tail = list(node.args[3:])
                if len(node.args) == 4:
                    tail += [ast.Constant(0), ast.Constant(0), ast.Constant(0.0)]
                return ast.copy_location(
                    ast.Call(
                        ast.Name("_interval_lookup_1d", ast.Load()),
                        [ast.Name(x.value, ast.Load()) for x in node.args[:3]] + tail,
                        [],
                    ),
                    node,
                )
            if name == "__interp_lookup_1d__" and len(node.args) == 4:
                if not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[1], ast.Constant)):
                    raise SourcePreservingNativeError("interpolation lookup requires literal prepared-input keys")
                return ast.copy_location(ast.Call(ast.Name("_interp_lookup_1d", ast.Load()), [ast.Name(node.args[0].value, ast.Load()), ast.Name(node.args[1].value, ast.Load()), *node.args[2:]], []), node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            receiver = node.func.value.id
            module_name = self.g.module_refs.get(receiver, receiver)
            target = _LIBC_MATH_INTRINSICS.get(node.func.attr)
            if module_name in {"math", "numpy"} and target is not None and len(node.args) == 1:
                return ast.copy_location(ast.Call(ast.Name(target, ast.Load()), node.args, []), node)
        return node

    def _input_call(self, name: str, node: ast.Call) -> ast.AST:
        if not node.args or not isinstance(node.args[0], ast.Constant):
            raise SourcePreservingNativeError(f"{name} requires literal prepared-input key")
        key = node.args[0].value
        spec = self.g.input_specs.get(key)
        if spec is None:
            raise SourcePreservingNativeError(f"unknown prepared-input key {key!r}")
        if name == "point_input":
            if spec.scope != "point" or spec.ndim != 1 or len(node.args) != 1:
                raise SourcePreservingNativeError(f"invalid point_input contract for {key!r}")
            return ast.copy_location(
                ast.Subscript(ast.Name(key, ast.Load()), ast.Name(self.policy_index_name, ast.Load()), ast.Load()),
                node,
            )
        if name == "global_input":
            if spec.ndim != 0 or len(node.args) != 1:
                raise SourcePreservingNativeError(f"invalid global_input contract for {key!r}")
            return ast.copy_location(ast.Name(key, ast.Load()), node)
        indices = node.args[1:]
        if len(indices) != spec.ndim:
            raise SourcePreservingNativeError(f"{name} {key!r} expected {spec.ndim} indices")
        sl: ast.expr = indices[0] if len(indices) == 1 else ast.Tuple(list(indices), ast.Load())
        return ast.copy_location(ast.Subscript(ast.Name(key, ast.Load()), sl, ast.Load()), node)


class SourcePreservingDenseCythonGenerator(DenseRecursiveCythonGenerator):
    """Dense recursive generator whose formula bodies are left to Cython."""

    def __init__(self, program: SourcePreservingDenseProgram):
        if not program.supported:
            raise SourcePreservingNativeError("source-preserving program blocked: " + "; ".join(program.blockers))
        self.source_program = program
        self.compile_plan = program.compile_plan
        super().__init__(program.dense_program)
        self.has_native_round = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "round"
            and 1 <= len(node.args) <= 2
            for cell in self.v.values()
            if cell.function is not None
            for node in ast.walk(cell.function)
        )

    def _lookup_helpers(self) -> list[str]:
        lines = super()._lookup_helpers()
        lines += _checked_real_power_helper_lines()
        if self.has_native_round:
            lines += _round_compat_helper_lines()
        lines += [
            "cdef inline long long _native_int_from_double(double x) noexcept nogil:",
            "    return <long long>x",
            "",
            "cdef inline double _native_float_from_long(long long x) noexcept nogil:",
            "    return <double>x",
            "",
        ]
        return lines

    def _policy_index_name(self, cell: NativeFormulaCell) -> str:
        used = set(cell.parameters)
        if cell.function is not None:
            used.update(node.id for node in ast.walk(cell.function) if isinstance(node, ast.Name))
        name = "__mx_policy_i"
        while name in used:
            name += "_"
        return name

    def _emit_formula(self, cell: NativeFormulaCell) -> list[str]:
        fn = cell.function
        if fn is None:
            raise SourcePreservingNativeError(f"{cell.uid}: missing formula")
        row = self.compile_plan.cells[cell.uid]
        policy_index_name = self._policy_index_name(cell)
        transformed = copy.deepcopy(fn)
        transformed = _ModelxLeafRewriter(self, cell, policy_index_name).visit(transformed)
        assert isinstance(transformed, ast.FunctionDef)
        ast.fix_missing_locations(transformed)
        body = list(transformed.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if not body:
            raise SourcePreservingNativeError(f"{cell.uid}: empty formula body")

        lines: list[str] = []
        if row.directives.cpow_safe:
            lines.append("@cython.cpow(True)")
        signature = self._function_signature(cell, "calc")
        lines += [signature + ":", "    global _guard_error"]
        for name, dtype in row.local_dtypes:
            if name in cell.parameters:
                continue
            ctype = self.ctype(dtype)
            initial = "0.0" if dtype == "float64" else "0"
            lines.append(f"    cdef {ctype} {name} = {initial}")
        for statement in body:
            lines.extend(textwrap.indent(ast.unparse(statement), "    ").splitlines())
        lines.append("")
        return lines

    def source(self, module_name: str = "source_preserving_dense_native") -> str:
        # Rank-1 outputs reuse DenseRecursiveCythonGenerator's established
        # ranked Cython/NumPy path. Scalar source-preserving rewriting remains
        # unchanged; the ranked generator consumes ranked_function directly.
        source = super().source(module_name)
        source = source.replace("cdivision=True", "cdivision=False", 1)
        if any(row.directives.cpow_safe for row in self.compile_plan.cells.values()):
            source = source.replace("cimport numpy as cnp\n", "cimport numpy as cnp\ncimport cython\n", 1)

        # Bind-time validation makes complete prepared-input bounds part of the
        # compiled artifact contract only when a stronger Cython semantic relies
        # on them. This is outside the nogil hot path and scales with input
        # preparation, not recursive evaluation.
        for key in self.compile_plan.required_input_bound_keys:
            bound = self.compile_plan.input_bounds[key]
            anchor = f"    {key} = np.ascontiguousarray(d[{key!r}]"
            position = source.find(anchor)
            if position < 0:
                # scalar inputs use a different assignment shape
                scalar_anchor = f"    {key} = "
                position = source.find(scalar_anchor, source.find("cpdef init_inputs(dict d):"))
                if position < 0:
                    raise SourcePreservingNativeError(f"cannot locate init binding for evidenced input {key}")
                line_end = source.find("\n", position)
            else:
                line_end = source.find("\n", position)
            validation = (
                f"    if not np.all(np.isfinite(np.asarray({key}, dtype=np.float64))):\n"
                f"        raise ValueError('source-preserving proof input {key} became non-finite')\n"
                f"    if np.any(np.asarray({key}) < {bound.lower!r}) or np.any(np.asarray({key}) > {bound.upper!r}):\n"
                f"        raise ValueError('source-preserving proof input {key} escaped compiled numeric bounds')\n"
            )
            source = source[: line_end + 1] + validation + source[line_end + 1 :]
        return source


def build_source_preserving_program_from_fast_plan(
    plan,
    *,
    max_dense_slots_per_cell: int = 1_000_000,
    native_hash_capacity_by_uid: Mapping[str, int] | None = None,
    native_hash_debug: bool = False,
) -> SourcePreservingDenseProgram:
    """Build Stage-1A backend from a ``ModelxCythonFastPlan``.

    Complete numeric input bounds are derived from ``plan.run_keys`` rather than
    the representative trace sample. They are used only for optional semantic
    directives and are revalidated by generated ``init_inputs``.
    """

    formula = getattr(plan, "source_preserving_native_program", None)
    if formula is None:
        formula = plan.native_program
    if formula is None:
        raise SourcePreservingNativeError(
            "fast plan has no source-transparent native candidate program"
        )
    bind_source = getattr(plan, "bind_source_preserving_inputs", None)
    bound_inputs = (
        bind_source()
        if callable(bind_source)
        else plan.bind_inputs()
    )
    input_bounds = derive_complete_input_numeric_bounds(
        formula,
        bound_inputs,
        run_population=len(plan.run_keys),
    )

    # Native exact-hash storage remains evidence-backed. Stage 1B may admit a
    # formula that the historical syntax whitelist never reached; if that also
    # creates a new hash memo Cell, fail closed rather than silently using the
    # dense runtime's placeholder capacity. A later storage-evidence pass can
    # profile that key domain independently of syntax admission.
    from .dense_recursive_native import build_dense_memo_plan
    memo_probe = build_dense_memo_plan(
        formula,
        parameter_ranges=plan.parameter_ranges,
        memoization_policy="native_cache",
        native_hash_capacity=16,
    )
    required_hash_uids = {
        row.uid for row in memo_probe.cells if row.kind == "hash"
    }
    capacities = (
        dict(plan.native_hash_capacity_by_uid)
        if native_hash_capacity_by_uid is None
        else dict(native_hash_capacity_by_uid)
    )
    missing_hash = sorted(required_hash_uids - set(capacities))
    if missing_hash:
        raise SourcePreservingNativeError(
            "source-transparent candidate needs complete-run native hash capacity "
            "evidence for: " + ",".join(missing_hash)
        )
    return build_source_preserving_dense_program(
        formula,
        parameter_ranges=plan.parameter_ranges,
        input_bounds=input_bounds,
        lexical_sources=getattr(plan, "formula_lexical_sources", {}),
        max_dense_slots_per_cell=max_dense_slots_per_cell,
        native_hash_capacity_by_uid=capacities,
        native_hash_debug=native_hash_debug,
    )


__all__ = [
    "SourcePreservingNativeError",
    "CompleteNumericBound",
    "NumericInterval",
    "CythonDirectiveEvidence",
    "CellCompilePlan",
    "SourcePreservingCompilePlan",
    "SourcePreservingDenseProgram",
    "derive_complete_input_numeric_bounds",
    "build_source_preserving_compile_plan",
    "build_source_preserving_dense_program",
    "build_source_preserving_program_from_fast_plan",
    "SourcePreservingDenseCythonGenerator",
]
