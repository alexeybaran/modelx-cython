from __future__ import annotations

"""Static recursive frontend foundation.

This module gives the recursive compiler an execution-independent ownership layer
between the exported model and ``NativeFormulaProgram``.  It deliberately stops
before provider/table lowering: a static seed owns Cell identity, signatures,
call edges, defaults, guards and formula ASTs; a later binding context supplies
only types and normalized runtime inputs.

The important invariant is that no realized trace, StageExecutionPlan or
StageStoragePlan participates in Cell discovery.
"""

import ast
import copy
import hashlib
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Mapping

from .fast_recursive_graph import (
    FastRecursiveCallSite,
    FastRecursiveCell,
    FastRecursiveProgram,
)
from .modelx_cython_formula_engine import normalize_formula_function
from .native_formula import NativeFormulaCell, NativeFormulaError, NativeFormulaProgram
from .normalized_operator import NormalizedOperator
from .program_inputs import OptimizedInputSpec
from .program_semantics import GuardSpec, normalize_formula_guards


class StaticRecursiveFrontendError(RuntimeError):
    pass


_IDENT = re.compile(r"[^0-9A-Za-z_]+")


def _safe_ident(value: str) -> str:
    out = _IDENT.sub("_", value).strip("_") or "cell"
    if out[0].isdigit():
        out = "_" + out
    return out


@dataclass(frozen=True, order=True)
class QualifiedCellKey:
    """Stable Cell identity independent of trace variants and runtime policy."""

    module_key: str
    class_name: str
    space_name: str
    cell_name: str

    @property
    def qualified_name(self) -> str:
        return f"{self.module_key}::{self.class_name}::{self.space_name}::{self.cell_name}"

    @property
    def uid(self) -> str:
        # Keep the readable Cell name while making collisions across spaces/modules
        # impossible in generated C identifiers.
        digest = hashlib.sha256(self.qualified_name.encode("utf-8")).hexdigest()[:12]
        return f"q_{digest}_{_safe_ident(self.cell_name)}"

    def manifest(self) -> dict[str, str]:
        return {
            "module_key": self.module_key,
            "class_name": self.class_name,
            "space_name": self.space_name,
            "cell_name": self.cell_name,
            "qualified_name": self.qualified_name,
            "uid": self.uid,
        }


@dataclass(frozen=True)
class StaticRecursiveFormulaCall:
    caller: QualifiedCellKey
    callee: QualifiedCellKey
    source: str
    lineno: int
    bound_args: tuple[tuple[str, str, str, str], ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "caller_uid": self.caller.uid,
            "callee_uid": self.callee.uid,
            "caller": self.caller.qualified_name,
            "callee": self.callee.qualified_name,
            "source": self.source,
            "lineno": self.lineno,
            "bound_args": [
                {
                    "parameter": p,
                    "source": src,
                    "origin": origin,
                    "relation": relation,
                }
                for p, src, origin, relation in self.bound_args
            ],
        }


@dataclass(frozen=True)
class StaticRecursiveFormulaCell:
    key: QualifiedCellKey
    parameters: tuple[str, ...]
    defaults: tuple[str | None, ...]
    function: ast.FunctionDef
    guards: tuple[GuardSpec, ...]
    dependency_uids: tuple[str, ...]
    external_calls: tuple[str, ...]
    syntax_features: tuple[str, ...]

    @property
    def uid(self) -> str:
        return self.key.uid

    def manifest(self) -> dict[str, Any]:
        return {
            "key": self.key.manifest(),
            "parameters": list(self.parameters),
            "defaults": list(self.defaults),
            "dependencies": list(self.dependency_uids),
            "external_calls": list(self.external_calls),
            "syntax_features": list(self.syntax_features),
            "guards": [g.manifest() for g in self.guards],
            "formula": ast.unparse(self.function),
        }


@dataclass(frozen=True)
class StaticRecursiveFormulaSeed:
    schema: str
    source_path: str
    module_key: str
    class_name: str
    space_name: str
    target_key: QualifiedCellKey
    cells: Mapping[str, StaticRecursiveFormulaCell]
    calls: tuple[StaticRecursiveFormulaCall, ...]
    graph_blockers: tuple[str, ...]
    provider_boundary: tuple[str, ...]

    @property
    def target_uid(self) -> str:
        return self.target_key.uid

    @property
    def graph_supported(self) -> bool:
        return not self.graph_blockers

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_path": self.source_path,
            "module_key": self.module_key,
            "class_name": self.class_name,
            "space_name": self.space_name,
            "target": self.target_key.manifest(),
            "graph_supported": self.graph_supported,
            "graph_blockers": list(self.graph_blockers),
            "cell_count": len(self.cells),
            "call_count": len(self.calls),
            "provider_boundary_count": len(self.provider_boundary),
            "provider_boundary": list(self.provider_boundary),
            "cells": {uid: cell.manifest() for uid, cell in sorted(self.cells.items())},
            "calls": [call.manifest() for call in self.calls],
            "trace_variant_dependency": False,
            "stage_dependency": False,
        }


class _RewriteStaticCellCalls(ast.NodeTransformer):
    def __init__(
        self,
        uid_by_name: Mapping[str, str],
        cells_by_name: Mapping[str, FastRecursiveCell],
    ):
        self.uid_by_name = uid_by_name
        self.cells_by_name = cells_by_name

    @staticmethod
    def _default_expr(source: str) -> ast.expr:
        try:
            return ast.parse(source, mode="eval").body
        except SyntaxError as exc:
            raise StaticRecursiveFrontendError(
                f"cannot parse static default argument {source!r}: {exc}"
            ) from exc

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        fun = node.func
        if not (
            isinstance(fun, ast.Attribute)
            and isinstance(fun.value, ast.Name)
            and fun.value.id == "self"
            and fun.attr in self.uid_by_name
        ):
            return node

        callee = self.cells_by_name[fun.attr]
        params = list(callee.parameters)
        if len(node.args) > len(params) or any(kw.arg is None for kw in node.keywords):
            raise StaticRecursiveFrontendError(
                f"static call to {fun.attr} is not canonically bindable"
            )
        values: dict[str, ast.expr] = {}
        for param, arg in zip(params, node.args):
            values[param.name] = arg
        known = {p.name: p for p in params}
        for kw in node.keywords:
            assert kw.arg is not None
            if kw.arg not in known or kw.arg in values:
                raise StaticRecursiveFrontendError(
                    f"static call to {fun.attr} has invalid keyword {kw.arg!r}"
                )
            values[kw.arg] = kw.value
        args: list[ast.expr] = []
        for param in params:
            if param.name in values:
                args.append(values[param.name])
            elif param.default_source is not None:
                args.append(self._default_expr(param.default_source))
            else:
                raise StaticRecursiveFrontendError(
                    f"static call to {fun.attr} misses required argument {param.name!r}"
                )
        node.func = ast.copy_location(
            ast.Name(id=self.uid_by_name[fun.attr], ctx=ast.Load()), fun
        )
        node.args = args
        node.keywords = []
        return node


def _parse_exported_cell(cell: FastRecursiveCell) -> ast.FunctionDef:
    try:
        mod = ast.parse(textwrap.dedent(cell.source))
    except SyntaxError as exc:
        raise StaticRecursiveFrontendError(
            f"cannot parse exported static Cell {cell.name}: {exc}"
        ) from exc
    fns = [x for x in mod.body if isinstance(x, ast.FunctionDef)]
    if len(fns) != 1:
        raise StaticRecursiveFrontendError(
            f"exported static Cell {cell.name} did not contain one function"
        )
    return fns[0]


def _key(program: FastRecursiveProgram, module_key: str, cell_name: str) -> QualifiedCellKey:
    return QualifiedCellKey(
        module_key=module_key,
        class_name=program.class_name,
        space_name=program.space_name,
        cell_name=cell_name,
    )


def build_static_recursive_formula_seed(
    program: FastRecursiveProgram,
    *,
    module_key: str | None = None,
) -> StaticRecursiveFormulaSeed:
    """Freeze the static target closure into qualified formula IR.

    This function does *not* claim provider/table support.  Every external call is
    retained as an explicit provider-boundary item.  That makes the next layer's
    responsibilities visible instead of silently borrowing trace-specialized
    variants.
    """

    if not program.graph_supported:
        raise StaticRecursiveFrontendError(
            "static recursive graph is blocked: " + "; ".join(program.graph_blockers)
        )
    module_key = module_key or program.module_key
    keys = {name: _key(program, module_key, name) for name in program.cells}
    uid_by_name = {name: key.uid for name, key in keys.items()}

    next_guard_code = 1
    cells: dict[str, StaticRecursiveFormulaCell] = {}
    for name in sorted(program.cells):
        source_cell = program.cells[name]
        fn = _parse_exported_cell(source_cell)
        fn = _RewriteStaticCellCalls(uid_by_name, program.cells).visit(copy.deepcopy(fn))
        assert isinstance(fn, ast.FunctionDef)
        # Reuse the mature modelx-cython normalization primitive: it strips the
        # generated ``self`` ABI and unrolls the same finite literal generators.
        fn = normalize_formula_function(fn, cell_name=keys[name].uid)
        # All internal static calls have had defaults/keywords materialized into
        # the call sites above, so native Cell ABIs are explicit and default-free.
        fn.args.defaults = []
        fn.args.kw_defaults = []
        fn, guards, next_guard_code = normalize_formula_guards(
            fn, uid=keys[name].uid, next_code=next_guard_code
        )
        dep_uids = tuple(sorted({uid_by_name[c.callee] for c in source_cell.dependency_calls}))
        cells[keys[name].uid] = StaticRecursiveFormulaCell(
            key=keys[name],
            parameters=tuple(p.name for p in source_cell.parameters),
            defaults=tuple(p.default_source for p in source_cell.parameters),
            function=fn,
            guards=guards,
            dependency_uids=dep_uids,
            external_calls=tuple(x.source for x in source_cell.external_calls),
            syntax_features=source_cell.syntax_features,
        )

    calls = tuple(
        StaticRecursiveFormulaCall(
            caller=keys[site.caller],
            callee=keys[site.callee],
            source=site.source,
            lineno=site.lineno,
            bound_args=tuple(
                (arg.parameter, arg.source, arg.origin, arg.relation)
                for arg in site.bound_args
            ),
        )
        for site in program.call_sites
    )
    provider_boundary = tuple(
        sorted(
            f"{keys[x.caller].qualified_name}: {x.source} [{x.kind}]"
            for x in program.external_calls
        )
    )
    return StaticRecursiveFormulaSeed(
        schema="modelx_graph.static_recursive_formula_seed.v1",
        source_path=program.source_path,
        module_key=module_key,
        class_name=program.class_name,
        space_name=program.space_name,
        target_key=keys[program.target],
        cells=cells,
        calls=calls,
        graph_blockers=program.graph_blockers,
        provider_boundary=provider_boundary,
    )


@dataclass(frozen=True)
class StaticRecursiveBindingContext:
    """Minimal typed/provider contract required to enter NativeFormulaProgram.

    The foundation intentionally requires exact qualified UIDs.  It does not
    accept source-name fallbacks, because those become ambiguous as soon as more
    than one Space contains a Cell with the same name.
    """

    result_dtypes: Mapping[str, str]
    parameter_dtypes: Mapping[str, Mapping[str, str]]
    input_order: tuple[str, ...] = ()
    inputs: tuple[OptimizedInputSpec, ...] = ()
    normalized_operators: tuple[NormalizedOperator, ...] = ()
    module_refs: tuple[tuple[str, str], ...] = ()
    allow_external_calls: bool = False


def build_native_formula_program_from_static(
    seed: StaticRecursiveFormulaSeed,
    context: StaticRecursiveBindingContext,
) -> NativeFormulaProgram:
    """Enter the existing native formula IR from a static graph-owned seed.

    In this foundation stage external provider syntax is rejected unless the
    caller explicitly marks it already normalized.  Future provider lowering can
    satisfy this contract without changing Cell discovery/identity.
    """

    if seed.provider_boundary and not context.allow_external_calls:
        raise NativeFormulaError(
            "static provider boundary is unresolved: " + "; ".join(seed.provider_boundary[:8])
        )
    if set(context.input_order) != {x.key for x in context.inputs}:
        raise NativeFormulaError("static binding input_order/input specs disagree")

    cells: dict[str, NativeFormulaCell] = {}
    for uid, cell in seed.cells.items():
        dtype = context.result_dtypes.get(uid)
        if dtype not in {"float64", "int64", "bool"}:
            raise NativeFormulaError(f"{uid}: result dtype is not proven")
        pd = context.parameter_dtypes.get(uid, {})
        parameter_dtypes: list[str] = []
        for name in cell.parameters:
            kind = pd.get(name)
            if kind not in {"float64", "int64", "bool"}:
                raise NativeFormulaError(
                    f"{uid}: parameter dtype for {name!r} is not proven"
                )
            parameter_dtypes.append(kind)
        cells[uid] = NativeFormulaCell(
            uid=uid,
            source_name=cell.key.cell_name,
            source_fullname=cell.key.qualified_name,
            dtype=dtype,
            parameters=cell.parameters,
            parameter_dtypes=tuple(parameter_dtypes),
            function=cell.function,
            reduction=None,
            guards=cell.guards,
        )

    identity = [seed.schema, seed.target_uid, *context.input_order]
    identity.extend(f"{uid}:{ast.unparse(c.function)}" for uid, c in sorted(seed.cells.items()))
    program_uid = "nfs_" + hashlib.sha256("\0".join(identity).encode("utf-8")).hexdigest()[:16]
    return NativeFormulaProgram(
        schema="modelx_graph.native_formula.v1",
        uid=program_uid,
        output_uid=seed.target_uid,
        cells=cells,
        input_order=context.input_order,
        inputs=context.inputs,
        normalized_operators=context.normalized_operators,
        module_refs=context.module_refs,
        provenance="static_recursive_formula_seed_v1",
    )


__all__ = [
    "StaticRecursiveFrontendError",
    "QualifiedCellKey",
    "StaticRecursiveFormulaCall",
    "StaticRecursiveFormulaCell",
    "StaticRecursiveFormulaSeed",
    "StaticRecursiveBindingContext",
    "build_static_recursive_formula_seed",
    "build_native_formula_program_from_static",
]
