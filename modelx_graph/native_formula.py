from __future__ import annotations

"""Schedule-independent native formula IR.

This layer freezes the output of the mature syntax/provider frontend before an
execution backend chooses Stage scans or recursive demand.  It intentionally
contains no execution stages, storage slots, scan directions, SCC scheduling or
trace dispatch machinery.
"""

import ast
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from .frontend import ReductionSpec
from .normalized_operator import NormalizedOperator
from .program_inputs import OptimizedInputSpec
from .program_semantics import GuardSpec


class NativeFormulaError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeFormulaCell:
    uid: str
    source_name: str
    source_fullname: str
    dtype: str
    parameters: tuple[str, ...]
    parameter_dtypes: tuple[str, ...]
    function: ast.FunctionDef | None
    reduction: ReductionSpec | None
    guards: tuple[GuardSpec, ...] = ()
    execution_role: str = "formula"  # formula | materialized_helper
    result_category_domain: str | None = None
    parameter_category_domains: tuple[str | None, ...] = ()
    logical_result_ndim: int = 0
    logical_result_shape: tuple[int, ...] | None = None
    ranked_function: ast.FunctionDef | None = None
    local_scalar_dtypes: tuple[tuple[str, str], ...] = ()
    local_integer_names: tuple[str, ...] = ()
    local_positive_int_names: tuple[str, ...] = ()
    local_positive_int_domains: tuple[tuple[str, tuple[int, ...]], ...] = ()
    parameter_int64_intervals: tuple[tuple[str, int, int, str], ...] = ()
    local_int64_intervals: tuple[tuple[str, int, int, str], ...] = ()
    callee_int64_intervals: tuple[tuple[str, int, int, str], ...] = ()
    input_int64_intervals: tuple[tuple[str, int, int, str], ...] = ()
    local_control_evidence_source: str = ""

    @property
    def arity(self) -> int:
        return len(self.parameters)

    def source(self) -> str:
        if self.reduction is not None:
            r = self.reduction
            return "reduction(" + ",".join((
                ast.unparse(r.init), r.target, r.loop_var,
                *(ast.unparse(x) for x in r.range_args),
                ast.unparse(r.body_expr),
                *(ast.unparse(x) for x in r.filters),
            )) + ")"
        if self.function is None:
            return "<none>"
        return ast.unparse(self.function)

    def ranked_source(self) -> str | None:
        return None if self.ranked_function is None else ast.unparse(self.ranked_function)

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_name": self.source_name,
            "source_fullname": self.source_fullname,
            "dtype": self.dtype,
            "parameters": list(self.parameters),
            "parameter_dtypes": list(self.parameter_dtypes),
            "kind": "reduction" if self.reduction is not None else "formula",
            "execution_role": self.execution_role,
            "result_category_domain": self.result_category_domain,
            "parameter_category_domains": list(self.parameter_category_domains),
            "logical_result_ndim": self.logical_result_ndim,
            "logical_result_shape": (
                None if self.logical_result_shape is None
                else list(self.logical_result_shape)
            ),
            "source": self.source(),
            "ranked_source": self.ranked_source(),
            "local_control_evidence": {
                "local_scalar_dtypes": dict(self.local_scalar_dtypes),
                "integer_names": list(self.local_integer_names),
                "positive_int_names": list(self.local_positive_int_names),
                "positive_int_domains": {
                    name: list(values) for name, values in self.local_positive_int_domains
                },
                "parameter_int64_intervals": {
                    name: {"lower": lower, "upper": upper, "authority": authority}
                    for name, lower, upper, authority in self.parameter_int64_intervals
                },
                "local_int64_intervals": {
                    name: {"lower": lower, "upper": upper, "authority": authority}
                    for name, lower, upper, authority in self.local_int64_intervals
                },
                "callee_int64_intervals": {
                    name: {"lower": lower, "upper": upper, "authority": authority}
                    for name, lower, upper, authority in self.callee_int64_intervals
                },
                "input_int64_intervals": {
                    name: {"lower": lower, "upper": upper, "authority": authority}
                    for name, lower, upper, authority in self.input_int64_intervals
                },
                "source": self.local_control_evidence_source,
            },
            "guards": [g.manifest() for g in self.guards],
        }


@dataclass(frozen=True)
class NativeFormulaProgram:
    schema: str
    uid: str
    output_uid: str
    cells: Mapping[str, NativeFormulaCell]
    input_order: tuple[str, ...]
    inputs: tuple[OptimizedInputSpec, ...]
    normalized_operators: tuple[NormalizedOperator, ...] = ()
    module_refs: tuple[tuple[str, str], ...] = ()
    provenance: str = "normalized_frontend"
    category_domains: tuple[tuple[str, tuple[object, ...]], ...] = ()

    @property
    def variants(self) -> Mapping[str, NativeFormulaCell]:
        return self.cells

    def input_spec(self, key: str) -> OptimizedInputSpec:
        for spec in self.inputs:
            if spec.key == key:
                return spec
        raise NativeFormulaError(f"unknown native input key {key!r}")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "uid": self.uid,
            "output_uid": self.output_uid,
            "provenance": self.provenance,
            "cell_count": len(self.cells),
            "executable_cell_count": sum(1 for c in self.cells.values() if c.execution_role == "formula"),
            "materialized_helper_count": sum(1 for c in self.cells.values() if c.execution_role != "formula"),
            "max_arity": max((c.arity for c in self.cells.values()), default=0),
            "input_order": list(self.input_order),
            "inputs": {x.key: x.manifest() for x in self.inputs},
            "normalized_operators": [x.manifest() for x in self.normalized_operators],
            "module_refs": dict(self.module_refs),
            "category_domains": {domain_id: list(labels) for domain_id, labels in self.category_domains},
            "cells": {uid: cell.manifest() for uid, cell in sorted(self.cells.items())},
        }


def _parameter_dtypes(
    fn: ast.FunctionDef,
    explicit: Mapping[str, str] | None = None,
    *,
    proven_integer_parameters: tuple[str, ...] = (),
) -> tuple[str, ...]:
    explicit = explicit or {}
    proven_integer = set(proven_integer_parameters)
    result = []
    for arg in fn.args.args:
        dtype = explicit.get(arg.arg)
        if dtype is None and arg.arg in proven_integer:
            dtype = "int64"
        if dtype is None:
            raise NativeFormulaError(
                f"native parameter dtype for {arg.arg!r} is not proven; "
                "supply parameter_dtypes_by_uid or static argument-domain evidence"
            )
        if dtype not in {"int64", "float64", "bool"}:
            raise NativeFormulaError(f"unsupported native parameter dtype {dtype!r} for {arg.arg}")
        result.append(dtype)
    return tuple(result)


def build_native_formula_program(
    program,
    *,
    parameter_dtypes_by_uid: Mapping[str, Mapping[str, str]] | None = None,
) -> NativeFormulaProgram:
    """Freeze normalized formulas/inputs from a mature optimized frontend result.

    ``program`` may be a StageOptimizedProgram or OptimizedProgram.  Only
    normalized formula/provider evidence is copied.  Stage/storage decisions are
    deliberately not represented in the returned IR.
    """
    variants = getattr(program, "variants", None)
    inputs = tuple(getattr(program, "inputs", ()))
    input_order = tuple(getattr(program, "input_order", ()))
    output_uid = getattr(program, "output_uid", None)
    if variants is None or output_uid is None:
        raise NativeFormulaError("source program does not expose normalized variants/output_uid")
    if set(input_order) != {x.key for x in inputs}:
        raise NativeFormulaError("normalized input_order/input specs disagree")

    pdtypes = parameter_dtypes_by_uid or {}
    cells: dict[str, NativeFormulaCell] = {}
    for uid, row in variants.items():
        fn = row.function
        reduction = row.reduction
        if reduction is not None:
            parameters: tuple[str, ...] = ()
            parameter_dtypes: tuple[str, ...] = ()
        elif fn is not None:
            if fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
                raise NativeFormulaError(f"{uid}: non-simple normalized signature is not native-recursive yet")
            if fn.args.defaults or fn.args.kw_defaults:
                raise NativeFormulaError(f"{uid}: defaults must be bound before normalized native IR")
            parameters = tuple(x.arg for x in fn.args.args)
            time_param = getattr(row, "time_param_source", None)
            proven_integer = (str(time_param),) if time_param in parameters else ()
            parameter_dtypes = _parameter_dtypes(
                fn, pdtypes.get(uid), proven_integer_parameters=proven_integer
            )
        else:
            raise NativeFormulaError(f"{uid}: normalized variant has neither function nor reduction")
        cells[uid] = NativeFormulaCell(
            uid=uid,
            source_name=row.source_name,
            source_fullname=row.source_fullname,
            dtype=row.dtype,
            parameters=parameters,
            parameter_dtypes=parameter_dtypes,
            function=fn,
            reduction=reduction,
            guards=tuple(getattr(row, "guards", ())),
        )

    if output_uid not in cells:
        raise NativeFormulaError(f"normalized output uid {output_uid!r} is absent")

    module_refs: dict[str, str] = {}
    legacy = getattr(program, "legacy_program", None)
    canonical = getattr(program, "canonical", None)
    if canonical is None and legacy is not None:
        canonical = getattr(legacy, "canonical", None)
    space = getattr(canonical, "space", None)
    refs = getattr(space, "refs", {}) if space is not None else {}
    if hasattr(refs, "items"):
        for name, obj in refs.items():
            module_name = getattr(obj, "__name__", None)
            if module_name in {"math", "numpy"}:
                module_refs[str(name)] = str(module_name)

    normalized_operators = tuple(getattr(program, "normalized_operators", ()))
    identity_parts = [
        "native_formula_v1", str(output_uid), *input_order,
        *(f"{x.key}:{x.dtype}:{x.ndim}:{x.scope}:{x.expected_shape}:{x.domain_token}" for x in inputs),
        *(f"{uid}:{cells[uid].dtype}:{cells[uid].parameters}:{cells[uid].source()}" for uid in sorted(cells)),
    ]
    uid = "nf_" + hashlib.sha256("\0".join(identity_parts).encode("utf-8", "backslashreplace")).hexdigest()[:16]
    return NativeFormulaProgram(
        schema="modelx_graph.native_formula.v1",
        uid=uid,
        output_uid=output_uid,
        cells=cells,
        input_order=input_order,
        inputs=inputs,
        normalized_operators=normalized_operators,
        module_refs=tuple(sorted(module_refs.items())),
        provenance="mature_frontend_normalized_formula_bridge_v1",
    )


__all__ = [
    "NativeFormulaError", "NativeFormulaCell", "NativeFormulaProgram",
    "build_native_formula_program",
]
