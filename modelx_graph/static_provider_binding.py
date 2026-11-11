from __future__ import annotations

"""Static graph-owned provider binding adapter.

This module connects :class:`StaticRecursiveFormulaSeed` to the mature
``GraphModelCompiler`` provider/input normalizers without borrowing its Cell set
or trace-derived variant UIDs.  A representative trace is used only as runtime
schema/type evidence.  Formula identity and closure remain the qualified static
seed.

The adapter deliberately reuses the mature frontend's private normalization
primitives in this bounded stage.  It does not copy pandas/modelx conversion
logic.  A later cleanup may promote those primitives into a public provider
service once this ownership boundary is proven on real products.
"""

import ast
import copy
import hashlib
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .frontend import (
    CanonicalModel,
    FrontendError,
    GraphModelCompiler,
    InputRegistry,
    InputSpec,
    ReductionSpec,
)
from .native_formula import NativeFormulaCell, NativeFormulaProgram
from .normalized_operator import NormalizedOperator
from .program_inputs import OptimizedInputSpec
from .static_recursive_frontend import StaticRecursiveFormulaSeed
from .static_category_abi import (
    StaticCategoryABIError, StaticCategoryDomain, StaticCategoryEncodingPlan,
    build_static_category_encoding_plan,
)
from .static_type_domain import (
    StaticTypeDomainProof,
    StaticValueDomain,
    build_static_type_domain_proof,
)


class StaticProviderBindingError(RuntimeError):
    pass


class _StaticProviderEvidenceCompiler(GraphModelCompiler):
    """Initialize mature provider/type evidence without building its canonical program.

    The static seed owns formula closure.  Running ``GraphModelCompiler._build_canonical``
    here would construct a second trace-owned executable graph that this adapter must
    never consume, and is unnecessarily expensive on larger products.
    """

    def _preflight_observed_dependencies(self):  # type: ignore[override]
        # StaticRecursiveFormulaSeed already proves the complete formula closure.
        # Missing representative observations are type/provider evidence gaps, not
        # permission to delete static Cells.
        return None

    def _build_canonical(self):  # type: ignore[override]
        return None


_ALLOWED_RESULT_DTYPES = {"float64", "int64", "bool"}
_RUNTIME_INPUT_CALLS = {"point_input", "global_input", "array_input", "table_input"}
_NATIVE_INTRINSICS = {
    "min", "max", "abs", "round", "int", "float", "bool",
    "__is_missing__", "__is_inf__", "__erf__", "__exact_axis_code__", "__exact_axis_code_valid__",
    "__step_lookup_1d__", "__interval_lookup_1d__", "__interp_lookup_1d__",
    "__sparse_table_lookup__", "__prepared_point_lookup__",
    "__guard_fail__",
}
_MATH_CALLS = {"exp", "log", "sqrt", "sin", "cos", "floor", "ceil"}


def _freeze_input(spec: InputSpec) -> OptimizedInputSpec:
    return OptimizedInputSpec(
        key=spec.key,
        dtype=spec.dtype,
        ndim=int(spec.ndim),
        scope=spec.scope,
        description=spec.description,
        expected_shape=spec.expected_shape,
        domain_token=spec.domain_token,
        normalized_table=spec.normalized_table,
        point_row_source=spec.point_row_source,
        point_field=spec.point_field,
        normalized_structured_column=spec.normalized_structured_column,
        enum_labels=spec.enum_labels,
        category_validity_key=spec.category_validity_key,
        normalized_lookup_1d=spec.normalized_lookup_1d,
        lookup_1d_role=spec.lookup_1d_role,
        prepared_exact_int_value=spec.prepared_exact_int_value,
        prepared_exact_int_authority=spec.prepared_exact_int_authority,
        prepared_exact_int_proof_domain_size=spec.prepared_exact_int_proof_domain_size,
        prepared_exact_int_source_name=spec.prepared_exact_int_source_name,
    )


def _tree_parts(cell: NativeFormulaCell) -> tuple[ast.AST, ...]:
    if cell.reduction is not None:
        r = cell.reduction
        return (r.init, r.body_expr, *r.range_args, *r.filters)
    if cell.function is not None:
        return (cell.function,)
    return ()


def _input_keys(cell: NativeFormulaCell) -> tuple[str, ...]:
    out: set[str] = set()
    for tree in _tree_parts(cell):
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _RUNTIME_INPUT_CALLS
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                continue
            out.add(str(node.args[0].value))
    return tuple(sorted(out))


def _dtype_of_scalar(value: Any) -> str | None:
    import numpy as np
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int64"
    if isinstance(value, (float, np.floating)):
        return "float64"
    return None


def _parameter_dtype_proof(frontend: GraphModelCompiler, source_name: str, parameters: tuple[str, ...]) -> tuple[tuple[str | None, ...], tuple[str, ...]]:
    rows = [tr for tr in frontend.variant_trace_by_uid.values() if tr.schema.name == source_name]
    result: list[str | None] = []
    proof: list[str] = []
    for pos, name in enumerate(parameters):
        observed: set[str] = set()
        sources: set[str] = set()
        for tr in rows:
            if pos >= len(tr.schema.parameters) or tr.schema.parameters[pos] != name:
                continue
            if tr.key.time_pos == pos:
                observed.add("int64")
                sources.add("representative_coordinate_axis")
                continue
            aux_pos = pos if tr.key.time_pos is None or pos < tr.key.time_pos else pos - 1
            if 0 <= aux_pos < len(tr.key.aux_values):
                kind = _dtype_of_scalar(tr.key.aux_values[aux_pos])
                if kind is not None:
                    observed.add(kind)
                    sources.add("representative_aux_argument")
        if len(observed) == 1:
            result.append(next(iter(observed)))
            proof.append("+".join(sorted(sources)) or "representative_argument")
        else:
            result.append(None)
            proof.append("unproven" if not observed else "conflicting:" + ",".join(sorted(observed)))
    return tuple(result), tuple(proof)


def _result_dtype_proof(frontend: GraphModelCompiler, source_name: str) -> tuple[str | None, str]:
    rows = [tr for tr in frontend.variant_trace_by_uid.values() if tr.schema.name == source_name]
    kinds = {str(tr.dtype) for tr in rows}
    if len(kinds) == 1:
        return next(iter(kinds)), "representative_trace_schema"
    if not kinds:
        return None, "unobserved_by_representative_trace"
    return None, "conflicting_representative_dtypes:" + ",".join(sorted(kinds))


def _residual_blockers(
    cell: NativeFormulaCell,
    *,
    all_uids: set[str],
    module_refs: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Return backend-readiness blockers after mature provider normalization."""

    blockers: set[tuple[str, str]] = set()
    if cell.execution_role != "formula":
        return ()
    if cell.function is not None:
        allowed_stmt = (ast.FunctionDef, ast.Expr, ast.Assign, ast.AnnAssign, ast.Return, ast.If, ast.Pass)
        for node in ast.walk(cell.function):
            if isinstance(node, ast.stmt) and not isinstance(node, allowed_stmt):
                blockers.add(("control_syntax", type(node).__name__))
    for tree in _tree_parts(cell):
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                blockers.add(("provider", "residual_subscript:" + ast.unparse(node)))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    if name in all_uids or name in _RUNTIME_INPUT_CALLS or name in _NATIVE_INTRINSICS:
                        continue
                    blockers.add(("provider", "residual_call:" + ast.unparse(node)))
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    receiver = node.func.value.id
                    module = module_refs.get(receiver)
                    if module in {"math", "numpy"} and node.func.attr in _MATH_CALLS:
                        continue
                    blockers.add(("context_space", "residual_attribute_call:" + ast.unparse(node)))
                else:
                    blockers.add(("provider", "residual_dynamic_call:" + ast.unparse(node)))
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                blockers.add(("control_syntax", type(node).__name__))
            elif isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn, ast.Is, ast.IsNot)) for op in node.ops):
                blockers.add(("control_syntax", "unsupported_compare:" + ast.unparse(node)))
    return tuple(sorted(blockers))


def _exception_category(exc: Exception) -> str:
    text = str(exc).lower()
    if "space" in text or "itemspace" in text or "context" in text:
        return "context_space"
    if "external table" in text or "dataframe" in text or "point field" in text or "table " in text:
        return "provider"
    if "dtype" in text or "type" in text or "non-numeric" in text:
        return "type"
    if "for " in text or "while" in text or "generator" in text or "comprehension" in text:
        return "control_syntax"
    return "provider"


@dataclass(frozen=True)
class _StaticNormalizedCandidate:
    function: ast.FunctionDef | None
    reduction: ReductionSpec | None
    blocker_items: tuple[tuple[str, str], ...] = ()


def _finite_domain_text(domain: StaticValueDomain) -> str:
    return "[" + ",".join(repr(x) for x in domain.exact_values) + "]"


def _resolve_type_evidence(
    static_domain: StaticValueDomain,
    *,
    representative_dtype: str | None,
    representative_proof: str,
    context: str,
    provider_blocked: bool = False,
) -> tuple[str | None, str, tuple[tuple[str, str], ...]]:
    """Resolve static > provider schema > representative fallback, fail closed.

    The finite-literal case is deliberately not assigned an integer code here.  It is
    semantically proven but awaits the later categorical ABI stage.
    """

    blockers: list[tuple[str, str]] = []
    if static_domain.proven and static_domain.numeric_dtype in _ALLOWED_RESULT_DTYPES:
        dtype = static_domain.numeric_dtype
        if representative_dtype in _ALLOWED_RESULT_DTYPES and representative_dtype != dtype:
            blockers.append((
                "type",
                f"contradiction:{context}:static={dtype}:representative={representative_dtype}",
            ))
        elif representative_dtype is not None and representative_dtype not in _ALLOWED_RESULT_DTYPES:
            blockers.append((
                "type",
                f"contradiction:{context}:static={dtype}:representative={representative_dtype}",
            ))
        proof = "static_type_domain"
        if representative_dtype == dtype:
            proof += "+representative_corroboration"
        return dtype, proof, tuple(blockers)

    if static_domain.proven and static_domain.finite_literal:
        if representative_dtype in _ALLOWED_RESULT_DTYPES:
            blockers.append((
                "type",
                f"contradiction:{context}:static_finite_literal={_finite_domain_text(static_domain)}:"
                f"representative={representative_dtype}",
            ))
        blockers.append((
            "categorical_abi",
            f"{context}:finite_literal_domain={_finite_domain_text(static_domain)}",
        ))
        return None, "static_finite_literal_domain", tuple(blockers)

    # A partial static numeric fact may be corroborated by representative evidence.
    # It is not admitted on its own because its static blockers remain unresolved.
    if static_domain.numeric_dtype in _ALLOWED_RESULT_DTYPES:
        if representative_dtype in _ALLOWED_RESULT_DTYPES:
            # A partial static candidate is not yet a proof; unresolved branches
            # may legitimately widen int->float. Representative evidence may be
            # used as fallback until the static blockers are discharged.
            note = "representative_fallback+static_partial_numeric"
            if representative_dtype != static_domain.numeric_dtype:
                note += f":candidate={static_domain.numeric_dtype}"
            return representative_dtype, note, ()

    if representative_dtype in _ALLOWED_RESULT_DTYPES:
        return representative_dtype, "representative_fallback:" + representative_proof, ()

    if static_domain.conflict:
        blockers.append((
            "type",
            f"{context}:static_conflict:{'|'.join(static_domain.blockers)}",
        ))
    elif representative_dtype == "object" and provider_blocked:
        # The provider conversion is the authority that must establish the eventual
        # finite/category domain. Do not duplicate the same deficiency as a numeric
        # type blocker; the static domain record remains visibly unknown/blocked.
        return None, "provider_owned_nonnumeric_domain:" + representative_proof, ()
    else:
        detail = "|".join(static_domain.blockers) if static_domain.blockers else "unknown"
        blockers.append(("type", f"{context}:unproven:{detail}"))
    return None, "unproven_static_and_representative", tuple(blockers)


@dataclass(frozen=True)
class StaticCellBindingRecord:
    uid: str
    source_name: str
    source_fullname: str
    status: str  # native_formula_ready | materialized_helper | blocked
    execution_role: str
    result_dtype: str | None
    result_dtype_proof: str
    parameter_dtypes: tuple[tuple[str, str | None, str], ...]
    provider_input_keys: tuple[str, ...] = ()
    blocker_items: tuple[tuple[str, str], ...] = ()
    static_result_domain: StaticValueDomain | None = None
    static_parameter_domains: tuple[tuple[str, StaticValueDomain], ...] = ()
    representative_result_dtype: str | None = None
    representative_result_proof: str = ""
    representative_parameter_dtypes: tuple[tuple[str, str | None, str], ...] = ()
    result_category_domain: str | None = None
    parameter_category_domains: tuple[tuple[str, str | None], ...] = ()

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(f"{kind}:{detail}" for kind, detail in self.blocker_items)

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_name": self.source_name,
            "source_fullname": self.source_fullname,
            "status": self.status,
            "execution_role": self.execution_role,
            "result_dtype": self.result_dtype,
            "result_dtype_proof": self.result_dtype_proof,
            "parameter_dtypes": [
                {"name": name, "dtype": dtype, "proof": proof}
                for name, dtype, proof in self.parameter_dtypes
            ],
            "provider_input_keys": list(self.provider_input_keys),
            "blockers": [
                {"kind": kind, "detail": detail}
                for kind, detail in self.blocker_items
            ],
            "static_type_domain": {
                "result": None if self.static_result_domain is None else self.static_result_domain.manifest(),
                "parameters": {
                    name: domain.manifest() for name, domain in self.static_parameter_domains
                },
            },
            "category_abi": {
                "result_domain": self.result_category_domain,
                "parameter_domains": {name: domain for name, domain in self.parameter_category_domains if domain is not None},
            },
            "representative_evidence": {
                "result_dtype": self.representative_result_dtype,
                "result_proof": self.representative_result_proof,
                "parameters": [
                    {"name": name, "dtype": dtype, "proof": proof}
                    for name, dtype, proof in self.representative_parameter_dtypes
                ],
            },
        }


@dataclass
class StaticProviderBindingPlan:
    schema: str
    seed: StaticRecursiveFormulaSeed
    run_keys: tuple[Any, ...]
    records: tuple[StaticCellBindingRecord, ...]
    native_program: NativeFormulaProgram | None
    runtime_inputs: Mapping[str, InputSpec] = field(repr=False)
    normalized_operators: tuple[NormalizedOperator, ...] = ()
    evidence_trace_variant_count: int = 0
    type_domain_proof: StaticTypeDomainProof | None = field(default=None, repr=False)
    category_encoding_plan: StaticCategoryEncodingPlan | None = field(default=None, repr=False)

    @property
    def admitted(self) -> bool:
        return self.native_program is not None and not any(r.status == "blocked" for r in self.records)

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(
            f"{r.uid}:{kind}:{detail}"
            for r in self.records
            for kind, detail in r.blocker_items
        )

    def bind_inputs(self, keys: Sequence[Any] | None = None) -> dict[str, Any]:
        holder = SimpleNamespace(
            run_keys=self.run_keys,
            inputs=dict(self.runtime_inputs),
            validated_static_facts=(),
        )
        raw = CanonicalModel.raw_inputs(holder, self.run_keys if keys is None else tuple(keys), None)
        return CanonicalModel.coerce_inputs(holder, raw)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed_schema": self.seed.schema,
            "target_uid": self.seed.target_uid,
            "static_cell_count": len(self.seed.cells),
            "native_cell_count": None if self.native_program is None else len(self.native_program.cells),
            "executable_cell_count": None if self.native_program is None else sum(
                1 for c in self.native_program.cells.values() if c.execution_role == "formula"
            ),
            "materialized_helper_count": sum(r.status == "materialized_helper" for r in self.records),
            "blocked_cell_count": sum(r.status == "blocked" for r in self.records),
            "admitted": self.admitted,
            "trace_variant_dependency": False,
            "stage_dependency": False,
            "representative_trace_used_for_schema_evidence_only": True,
            "evidence_trace_variant_count": self.evidence_trace_variant_count,
            "category_encoding": None if self.category_encoding_plan is None else self.category_encoding_plan.manifest(),
            "static_type_domain": None if self.type_domain_proof is None else {
                "schema": self.type_domain_proof.schema,
                "iterations": self.type_domain_proof.iterations,
                "numeric_result_count": sum(
                    c.result_domain.proven and c.result_domain.numeric_dtype in _ALLOWED_RESULT_DTYPES
                    for c in self.type_domain_proof.cells.values()
                ),
                "finite_literal_result_count": sum(
                    c.result_domain.proven and c.result_domain.finite_literal
                    for c in self.type_domain_proof.cells.values()
                ),
            },
            "record_set_equal_to_static_seed": {r.uid for r in self.records} == set(self.seed.cells),
            "cell_set_equal_to_static_seed": (
                None if self.native_program is None
                else set(self.native_program.cells) == set(self.seed.cells)
            ),
            "inputs": {
                key: _freeze_input(spec).manifest()
                for key, spec in self.runtime_inputs.items()
            },
            "normalized_operators": [x.manifest() for x in self.normalized_operators],
            "cells": [r.manifest() for r in self.records],
        }



def _point_field_source(fn: ast.FunctionDef, point_uid: str) -> str | None:
    """Recognize a direct/guarded point-row scalar selector structurally.

    This is only source-to-provider binding evidence.  Value encoding itself is
    delegated to the mature ``InputRegistry.point_field(..., enum_labels=...)``.
    """
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]

    def field_from_expr(expr: ast.AST) -> str | None:
        if not (
            isinstance(expr, ast.Subscript)
            and isinstance(expr.value, ast.Call)
            and isinstance(expr.value.func, ast.Name)
            and expr.value.func.id == point_uid
            and not expr.value.args and not expr.value.keywords
            and isinstance(expr.slice, ast.Constant)
            and isinstance(expr.slice.value, str)
        ):
            return None
        return str(expr.slice.value)

    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
        return field_from_expr(body[0].value)
    if len(body) >= 2 and isinstance(body[0], ast.Assign) and len(body[0].targets) == 1 and isinstance(body[0].targets[0], ast.Name):
        field = field_from_expr(body[0].value)
        local = body[0].targets[0].id
        if field is not None and isinstance(body[-1], ast.Return) and isinstance(body[-1].value, ast.Name) and body[-1].value.id == local:
            return field
    return None


def _discover_point_category_domains(
    evidence: GraphModelCompiler,
    seed: StaticRecursiveFormulaSeed,
    type_proof: StaticTypeDomainProof,
    *,
    point_uid: str | None,
) -> tuple[dict[str, StaticCategoryDomain], dict[str, str]]:
    if point_uid is None or evidence.model_point_row is None:
        return {}, {}
    frame = evidence.model_point_row.provider()
    result: dict[str, StaticCategoryDomain] = {}
    fields: dict[str, str] = {}
    for uid, cell in seed.cells.items():
        if cell.parameters:
            continue
        field = _point_field_source(cell.function, point_uid)
        if field is None or field not in frame.columns:
            continue
        values = list(frame[field].tolist())
        if not values or not all(isinstance(value, str) for value in values):
            continue
        observed = set(values)
        static = type_proof.cells[uid].result_domain
        if static.proven and static.finite_literal:
            static_domain = StaticCategoryDomain.from_labels(
                static.exact_values, f"static_point_result:{uid}", *static.provenance
            )
            if not observed <= set(static_domain.labels):
                raise StaticProviderBindingError(
                    f"point category provider for {uid} contains labels outside static domain: "
                    f"observed={sorted(observed)!r}, static={static_domain.labels!r}"
                )
            domain = static_domain
        else:
            domain = StaticCategoryDomain.from_labels(
                tuple(observed), f"provider_point_field:{field}", f"provider_result:{uid}"
            )
        result[uid] = domain
        fields[uid] = field
    return result, fields


class _BindPointCategorySubscript(ast.NodeTransformer):
    def __init__(self, point_uid: str, field: str, input_key: str):
        self.point_uid = point_uid
        self.field = field
        self.input_key = input_key

    def visit_Subscript(self, node: ast.Subscript):  # noqa: N802
        node = self.generic_visit(node)
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == self.point_uid
            and not node.value.args and not node.value.keywords
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == self.field
        ):
            return ast.copy_location(
                ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(self.input_key)], []), node
            )
        return node


def _normalize_static_candidates(
    evidence: GraphModelCompiler,
    seed: StaticRecursiveFormulaSeed,
    *,
    materialized_uids: set[str],
    all_uids: set[str],
    module_refs: Mapping[str, str],
    category_plan: StaticCategoryEncodingPlan | None = None,
    point_category_fields: Mapping[str, str] | None = None,
    point_uid: str | None = None,
) -> tuple[dict[str, _StaticNormalizedCandidate], dict[str, ast.FunctionDef]]:
    point_category_fields = point_category_fields or {}
    candidates: dict[str, _StaticNormalizedCandidate] = {}
    formula_overrides: dict[str, ast.FunctionDef] = {}

    for uid, source_cell in seed.cells.items():
        if uid in materialized_uids:
            continue
        blockers: list[tuple[str, str]] = []
        fn: ast.FunctionDef | None = None
        reduction: ReductionSpec | None = None
        try:
            candidate = copy.deepcopy(source_cell.function)
            field = point_category_fields.get(uid)
            if field is not None:
                if category_plan is None or point_uid is None or evidence.model_point_row is None:
                    raise StaticCategoryABIError(f"{uid}: point category binding lacks category plan")
                did = category_plan.result_domain_id(uid)
                if did is None:
                    raise StaticCategoryABIError(f"{uid}: point category result domain is missing")
                domain = category_plan.domain(did)
                if any(not isinstance(label, str) for label in domain.labels):
                    raise StaticCategoryABIError(
                        f"{uid}: point category provider currently requires string-only labels"
                    )
                key = evidence.registry.point_field(
                    evidence.model_point_row, field, enum_labels=tuple(domain.labels)
                )
                candidate = _BindPointCategorySubscript(point_uid, field, key).visit(candidate)
                candidate = ast.fix_missing_locations(candidate)

            if category_plan is not None:
                candidate = category_plan.rewrite_function(uid, candidate)

            candidate = evidence._inline_vector_subscripts(copy.deepcopy(candidate))
            candidate = evidence._normalize_external_tables(candidate, operator_uid=uid)
            lower = evidence._expr_lowerer(uid)
            lower.prepare_function(candidate)
            reduction = evidence._normalize_reduction(candidate)
            if reduction is not None:
                reduction = ReductionSpec(
                    lower.visit(copy.deepcopy(reduction.init)),
                    reduction.target,
                    reduction.loop_var,
                    tuple(lower.visit(copy.deepcopy(x)) for x in reduction.range_args),
                    lower.visit(copy.deepcopy(reduction.body_expr)),
                    tuple(lower.visit(copy.deepcopy(x)) for x in reduction.filters),
                )
                ast.fix_missing_locations(reduction.body_expr)
            else:
                candidate.body = [lower.visit(copy.deepcopy(st)) for st in candidate.body]
                candidate = ast.fix_missing_locations(candidate)
                candidate = lower.cleanup_function(candidate)
                candidate = evidence._normalize_scalar_intrinsics(candidate)
                candidate = evidence._normalize_simple_local_syntax(candidate)
                fn = candidate
                formula_overrides[uid] = candidate
        except Exception as exc:
            kind = "categorical_abi" if isinstance(exc, StaticCategoryABIError) else _exception_category(exc)
            blockers.append((kind, f"{type(exc).__name__}:{exc}"))

        if fn is not None or reduction is not None:
            provisional = NativeFormulaCell(
                uid=uid,
                source_name=source_cell.key.cell_name,
                source_fullname=source_cell.key.qualified_name,
                dtype="float64",
                parameters=source_cell.parameters if reduction is None else (),
                parameter_dtypes=tuple("int64" for _ in source_cell.parameters) if reduction is None else (),
                function=fn,
                reduction=reduction,
                guards=source_cell.guards,
            )
            blockers.extend(_residual_blockers(
                provisional, all_uids=all_uids, module_refs=module_refs
            ))
        candidates[uid] = _StaticNormalizedCandidate(
            function=fn,
            reduction=reduction,
            blocker_items=tuple(sorted(set(blockers))),
        )
    return candidates, formula_overrides

def build_static_provider_binding_plan(
    seed: StaticRecursiveFormulaSeed,
    *,
    model: Any,
    space: Any,
    output_name: str,
    sample_keys: Sequence[Any],
    run_keys: Sequence[Any],
) -> StaticProviderBindingPlan:
    """Bind providers/types while preserving the qualified static Cell set.

    V02397 adds a static type/domain fixed point *after* provider normalization and
    *before* representative trace type fallback. Representative traces still supply
    schema evidence, never formula/Cell authority.
    """

    if not seed.graph_supported:
        raise StaticProviderBindingError("static seed graph is blocked")
    run_keys = tuple(run_keys)
    sample_keys = tuple(sample_keys)
    evidence = _StaticProviderEvidenceCompiler(
        model,
        space=space,
        output=output_name,
        sample_keys=sample_keys,
        run_keys=run_keys,
        canonical_schedule_authority=False,
        legacy_comparison=False,
    )

    evidence.registry = InputRegistry(evidence)
    evidence.multiindex_axes = {}
    evidence._normalized_operators = {}

    source_to_uid = {cell.key.cell_name: uid for uid, cell in seed.cells.items()}
    if len(source_to_uid) != len(seed.cells):
        raise StaticProviderBindingError("static seed contains duplicate source Cell names in one Space")

    for uid, cell in seed.cells.items():
        evidence.base_funcs[uid] = copy.deepcopy(cell.function)
    evidence.vector_elements = evidence._discover_vector_helpers()
    if evidence.model_point_row is not None:
        point_uid = source_to_uid.get(evidence.model_point_row.cell_name)
        if point_uid is not None:
            evidence.model_point_row = replace(evidence.model_point_row, cell_name=point_uid)

    module_refs: dict[str, str] = {}
    for name, obj in evidence.refs.items():
        module_name = getattr(obj, "__name__", None)
        if module_name in {"math", "numpy"}:
            module_refs[str(name)] = str(module_name)

    all_uids = set(seed.cells)
    vector_uids = {uid for uid in evidence.vector_elements if uid in all_uids}
    point_uid = (
        evidence.model_point_row.cell_name
        if evidence.model_point_row is not None and evidence.model_point_row.cell_name in all_uids
        else None
    )
    materialized_uids = set(vector_uids)
    if point_uid is not None:
        materialized_uids.add(point_uid)

    # Pass 1: normalize the pre-category numeric/provider subset.  This is only
    # evidence for the static type/domain fixed point and is discarded before final
    # category-aware normalization.
    candidates, formula_overrides = _normalize_static_candidates(
        evidence, seed, materialized_uids=materialized_uids, all_uids=all_uids,
        module_refs=module_refs, point_uid=point_uid,
    )

    type_proof = build_static_type_domain_proof(
        seed,
        formula_overrides=formula_overrides,
        input_specs=evidence.registry.specs,
        module_refs=module_refs,
        materialized_uids=tuple(sorted(materialized_uids)),
    )

    # Stage V02398+V02399: freeze the semantic category ABI, augmenting purely
    # static finite domains with provider-proven point-field domains.  Then reset
    # runtime input/operator registration and perform one final normalization pass
    # using the exact category codes.  Static Cell identity remains authoritative.
    provider_category_domains, point_category_fields = _discover_point_category_domains(
        evidence, seed, type_proof, point_uid=point_uid
    )
    category_plan = build_static_category_encoding_plan(
        seed, type_proof, provider_result_domains=provider_category_domains
    )
    if not category_plan.supported:
        raise StaticProviderBindingError(
            "static category encoding plan is blocked: " + "; ".join(category_plan.blockers)
        )

    evidence.registry = InputRegistry(evidence)
    evidence.multiindex_axes = {}
    evidence._normalized_operators = {}
    evidence.static_category_result_domains = {
        uid: tuple(category_plan.domain(did).labels)
        for uid, did in category_plan.result_domains.items()
        if all(isinstance(label, str) for label in category_plan.domain(did).labels)
    }
    candidates, _encoded_formula_overrides = _normalize_static_candidates(
        evidence, seed, materialized_uids=materialized_uids, all_uids=all_uids,
        module_refs=module_refs, category_plan=category_plan,
        point_category_fields=point_category_fields, point_uid=point_uid,
    )

    native_cells: dict[str, NativeFormulaCell] = {}
    record_rows: list[StaticCellBindingRecord] = []

    for uid, source_cell in seed.cells.items():
        rep_result_dtype, rep_result_proof = _result_dtype_proof(
            evidence, source_cell.key.cell_name
        )
        rep_param_kinds, rep_param_proofs = _parameter_dtype_proof(
            evidence, source_cell.key.cell_name, source_cell.parameters
        )
        rep_param_manifest = tuple(
            (name, dtype, proof)
            for name, dtype, proof in zip(
                source_cell.parameters, rep_param_kinds, rep_param_proofs
            )
        )
        cell_proof = type_proof.cells[uid]
        static_param_map = dict(cell_proof.parameter_domains)

        if uid in materialized_uids:
            helper_kind = "vector_helper" if uid in vector_uids else "point_row_provider"
            # Materialized helpers remain semantic identities but are not forced into
            # the numeric ABI. Their static domain is still recorded for correspondence.
            native_cells[uid] = NativeFormulaCell(
                uid=uid,
                source_name=source_cell.key.cell_name,
                source_fullname=source_cell.key.qualified_name,
                dtype=rep_result_dtype or "object",
                parameters=source_cell.parameters,
                parameter_dtypes=tuple(x or "object" for x in rep_param_kinds),
                function=source_cell.function,
                reduction=None,
                guards=source_cell.guards,
                execution_role="materialized_helper",
            )
            record_rows.append(StaticCellBindingRecord(
                uid=uid,
                source_name=source_cell.key.cell_name,
                source_fullname=source_cell.key.qualified_name,
                status="materialized_helper",
                execution_role="materialized_helper",
                result_dtype=rep_result_dtype,
                result_dtype_proof=rep_result_proof + "+" + helper_kind,
                parameter_dtypes=rep_param_manifest,
                static_result_domain=cell_proof.result_domain,
                static_parameter_domains=cell_proof.parameter_domains,
                representative_result_dtype=rep_result_dtype,
                representative_result_proof=rep_result_proof,
                representative_parameter_dtypes=rep_param_manifest,
            ))
            continue

        candidate = candidates.get(uid, _StaticNormalizedCandidate(None, None, ()))
        blockers: list[tuple[str, str]] = list(candidate.blocker_items)
        provider_blocked = any(kind in {"provider", "context_space"} for kind, _ in blockers)

        result_category_domain = category_plan.result_domain_id(uid)
        if result_category_domain is not None:
            result_dtype = "int64"
            result_proof = "static_category_abi:" + result_category_domain
            result_blockers: tuple[tuple[str, str], ...] = ()
        else:
            result_dtype, result_proof, result_blockers = _resolve_type_evidence(
                cell_proof.result_domain,
                representative_dtype=rep_result_dtype,
                representative_proof=rep_result_proof,
                context="result",
                provider_blocked=provider_blocked,
            )
        blockers.extend(result_blockers)

        final_param_kinds: list[str | None] = []
        final_param_proofs: list[str] = []
        param_category_domains: list[str | None] = []
        for name, rep_dtype, rep_proof in rep_param_manifest:
            category_domain = category_plan.parameter_domain_id(uid, name)
            param_category_domains.append(category_domain)
            if category_domain is not None:
                dtype, proof, items = "int64", "static_category_abi:" + category_domain, ()
            else:
                static_domain = static_param_map[name]
                dtype, proof, items = _resolve_type_evidence(
                    static_domain,
                    representative_dtype=rep_dtype,
                    representative_proof=rep_proof,
                    context=f"parameter:{name}",
                    provider_blocked=provider_blocked,
                )
            final_param_kinds.append(dtype)
            final_param_proofs.append(proof)
            blockers.extend(items)
        param_manifest = tuple(
            (name, dtype, proof)
            for name, dtype, proof in zip(
                source_cell.parameters, final_param_kinds, final_param_proofs
            )
        )
        param_category_manifest = tuple(zip(source_cell.parameters, param_category_domains))

        fn = candidate.function
        reduction = candidate.reduction
        if result_dtype not in _ALLOWED_RESULT_DTYPES and not provider_blocked:
            blockers.append(("type", "result:native_numeric_dtype_unproven"))
        for name, dtype, _proof in param_manifest:
            if dtype not in _ALLOWED_RESULT_DTYPES:
                blockers.append(("type", f"parameter:{name}:native_numeric_dtype_unproven"))

        blockers = list(sorted(set(blockers)))
        if (fn is not None or reduction is not None) and not blockers:
            assert result_dtype is not None
            assert all(x is not None for x in final_param_kinds)
            provisional = NativeFormulaCell(
                uid=uid,
                source_name=source_cell.key.cell_name,
                source_fullname=source_cell.key.qualified_name,
                dtype=result_dtype,
                parameters=source_cell.parameters if reduction is None else (),
                parameter_dtypes=tuple(x for x in final_param_kinds if x is not None) if reduction is None else (),
                function=fn,
                reduction=reduction,
                guards=source_cell.guards,
                result_category_domain=result_category_domain,
                parameter_category_domains=tuple(param_category_domains) if reduction is None else (),
            )
            native_cells[uid] = provisional
            record_rows.append(StaticCellBindingRecord(
                uid=uid,
                source_name=source_cell.key.cell_name,
                source_fullname=source_cell.key.qualified_name,
                status="native_formula_ready",
                execution_role="formula",
                result_dtype=result_dtype,
                result_dtype_proof=result_proof,
                parameter_dtypes=param_manifest,
                provider_input_keys=_input_keys(provisional),
                static_result_domain=cell_proof.result_domain,
                static_parameter_domains=cell_proof.parameter_domains,
                representative_result_dtype=rep_result_dtype,
                representative_result_proof=rep_result_proof,
                representative_parameter_dtypes=rep_param_manifest,
                result_category_domain=result_category_domain,
                parameter_category_domains=param_category_manifest,
            ))
            continue

        record_rows.append(StaticCellBindingRecord(
            uid=uid,
            source_name=source_cell.key.cell_name,
            source_fullname=source_cell.key.qualified_name,
            status="blocked",
            execution_role="formula",
            result_dtype=result_dtype,
            result_dtype_proof=result_proof,
            parameter_dtypes=param_manifest,
            blocker_items=tuple(blockers),
            static_result_domain=cell_proof.result_domain,
            static_parameter_domains=cell_proof.parameter_domains,
            representative_result_dtype=rep_result_dtype,
            representative_result_proof=rep_result_proof,
            representative_parameter_dtypes=rep_param_manifest,
            result_category_domain=result_category_domain,
            parameter_category_domains=param_category_manifest,
        ))

    # Verify helper materialization removed all executable calls to inactive helpers.
    inactive = {uid for uid, cell in native_cells.items() if cell.execution_role != "formula"}
    for uid, cell in list(native_cells.items()):
        if cell.execution_role != "formula":
            continue
        dangling = sorted({
            node.func.id
            for tree in _tree_parts(cell)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in inactive
        })
        if dangling:
            native_cells.pop(uid, None)
            for idx, record in enumerate(record_rows):
                if record.uid == uid:
                    record_rows[idx] = replace(
                        record,
                        status="blocked",
                        blocker_items=record.blocker_items + tuple(
                            ("provider", f"call_to_materialized_helper:{target}")
                            for target in dangling
                        ),
                    )
                    break

    blocked = {r.uid for r in record_rows if r.status == "blocked"}
    native_program: NativeFormulaProgram | None = None
    if not blocked:
        if set(native_cells) != all_uids:
            missing = sorted(all_uids - set(native_cells))
            raise StaticProviderBindingError(
                "admitted static binding lost Cell identities: " + ", ".join(missing)
            )
        frozen_inputs = tuple(_freeze_input(spec) for spec in evidence.registry.specs.values())
        identity = [seed.schema, seed.target_uid, *evidence.registry.specs]
        identity.extend(
            f"{uid}:{native_cells[uid].execution_role}:{native_cells[uid].dtype}:{native_cells[uid].source()}"
            for uid in sorted(native_cells)
        )
        program_uid = "nfsp_" + hashlib.sha256(
            "\0".join(identity).encode("utf-8", "backslashreplace")
        ).hexdigest()[:16]
        native_program = NativeFormulaProgram(
            schema="modelx_graph.native_formula.v1",
            uid=program_uid,
            output_uid=seed.target_uid,
            cells=native_cells,
            input_order=tuple(evidence.registry.specs),
            inputs=frozen_inputs,
            normalized_operators=tuple(sorted(evidence._normalized_operators.values(), key=lambda x: x.uid)),
            module_refs=tuple(sorted(module_refs.items())),
            provenance="static_recursive_provider_binding_v4_category_provider_closure",
            category_domains=tuple(
                (domain_id, tuple(category_plan.domains[domain_id].labels))
                for domain_id in sorted(category_plan.domains)
            ),
        )
        if set(native_program.cells) != set(seed.cells):
            raise StaticProviderBindingError("NativeFormulaProgram Cell set differs from static seed")

    return StaticProviderBindingPlan(
        schema="modelx_graph.static_provider_binding.v4",
        seed=seed,
        run_keys=run_keys,
        records=tuple(sorted(record_rows, key=lambda r: r.uid)),
        native_program=native_program,
        runtime_inputs=dict(evidence.registry.specs),
        normalized_operators=tuple(sorted(evidence._normalized_operators.values(), key=lambda x: x.uid)),
        evidence_trace_variant_count=len(evidence.variant_trace_by_uid),
        type_domain_proof=type_proof,
        category_encoding_plan=category_plan,
    )


__all__ = [
    "StaticProviderBindingError",
    "StaticCellBindingRecord",
    "StaticProviderBindingPlan",
    "build_static_provider_binding_plan",
]
