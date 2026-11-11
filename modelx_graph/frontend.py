from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .trace import TraceCapture, VariantKey, VariantTrace, CellSchema
from .trace import capture_trace
from .ir import build_graph_ir
from .template_ir import (
    ExecutableGraph, ExecutablePureMapSpec, build_executable_graph,
    prove_executable_pure_maps, TemplateError,
)
from .domain_graph import DomainGraph, ValueSemanticsEvidence, build_shadow_domain_graph, classify_source_value_semantics
from .canonical_semantic_graph import (
    CanonicalLoopTopologyFact, RecoveredLoopTopologyFact, CanonicalSemanticGraph,
    freeze_graph_ir_loop_topology, freeze_recovered_loop_topology,
    build_canonical_semantic_graph,
)
from .canonical_schedule import (
    CanonicalComponentScheduleAnalysis, CanonicalExecutionSchedule,
    analyze_canonical_component_schedule, build_canonical_execution_schedule,
)
from .stage_execution_plan import StageExecutionPlan, build_stage_execution_plan
from .stage_storage_plan import StageStoragePlan, build_stage_storage_plan
from .safety import validate_formula_control_flow, SemanticSafetyError
from .program_inputs import (
    NormalizedTableAxis, NormalizedTableInput, NormalizedPointRowSource,
    NormalizedLookup1D, NormalizedStructuredColumn,
    NormalizedStructuredSourceDemand, NormalizedStructuredRowMap,
)
from .invocation import OutputInvocation, RunDomain, ResultDomain
from .normalized_operator import NormalizedOperator
from .semantic_ir import AsofLookupIR, ast_fingerprint
from .program_semantics import (
    GuardSpec, StaticScalarFact, ValidatedStaticFact, FiniteDomainFact, FixedCoordinateFact,
    CoordinateRecurrenceDomainFact, RuntimeScalarHelperFact, SourceBackedScheduledFact,
    StepLookupDomainFact, ProgramSemanticError,
    normalize_formula_guards,
)
from .modelx_cython_formula_engine import unroll_finite_literal_generators
from .category_validity import CategoryValidityError, point_category_exact_axis_call
from .affine_domain import (
    AffineIntForm, AffineRangeConstraint, affine_bounds_under_range,
)
from .structured_provider import (
    StructuredFramePlan, StructuredProviderError, encode_column,
)


class FrontendError(RuntimeError):
    pass


RUNTIME_SCALAR_HELPER_EXPANSION_LIMIT = 20000


def _emit_translation_context_diagnostic(phase: str, **extra: object) -> None:
    """Persist optional low-cost preparation checkpoints for OOM diagnosis."""

    target = os.environ.get("MODELX_GRAPH_PLAN_DIAGNOSTICS")
    if not target:
        return
    rss_kb = hwm_kb = None
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kb = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    row = {"phase": f"translation_context:{phase}", "rss_kb": rss_kb, "hwm_kb": hwm_kb}
    row.update(extra)
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        pass


def _resolve_proof_alias(
    node: ast.AST,
    aliases: dict[str, ast.AST],
    *,
    max_hops: int = 8,
) -> tuple[ast.AST, int, bool]:
    """Resolve preparation-only scalar aliases without cloning ASTs.

    The helper is deliberately tiny and side-effect free so cycle handling can
    be unit-tested independently of modelx.  A cycle or hop-budget exhaustion
    returns the last unresolved node; callers must treat that as unknown proof
    evidence rather than inventing a value.
    """
    current = node
    seen: set[str] = set()
    hops = 0
    for _ in range(max_hops + 1):
        if not (isinstance(current, ast.Name) and current.id in aliases):
            return current, hops, False
        name = current.id
        if name in seen:
            return current, hops, True
        seen.add(name)
        current = aliases[name]
        hops += 1
    return current, hops, True


def _proof_alias_rhs_is_snapshot_safe(name: str, value: ast.AST) -> bool:
    """Whether a local assignment can be used as a textual proof alias.

    ``x = min(x, cap)`` is a perfectly ordinary Python update but it is not a
    textual definition of the post-assignment ``x``: the RHS ``x`` denotes the
    pre-assignment value.  Treating it as ``x -> min(x, cap)`` creates a fake
    recursive expression.  Such updates stay in emitted code but are excluded
    from proof-alias substitution.
    """
    return not any(
        isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id == name
        for child in ast.walk(value)
    )


@dataclass(frozen=True)
class HierarchicalItemSpaceParameterBinding:
    """Structural proof that a child Space Cells value inherits owner coordinates.

    The binding is deliberately about modelx hierarchy identity, not identifier
    spelling.  ``parameter_names`` are declared by the selected RunDomain Space and
    become concrete references only on the corresponding ItemSpace descendant.
    """

    ref_name: str
    cell_name: str
    owner_space_fullname: str
    child_space_fullname: str
    parameter_names: tuple[str, ...]
    parameter_positions: tuple[int, ...]
    dependency_cells: tuple[str, ...] = ()


@dataclass(frozen=True)
class FiniteSymbolicPointFieldPlan:
    """Preparation-only descriptor for finite symbolic point-field dispatch."""

    selector_name: str
    selector_values: tuple[Any, ...]
    field_names: tuple[str, ...]
    code_expr: ast.AST = field(compare=False, repr=False)
    domain_source: str = ""


def _finite_symbolic_point_field_plan(
    frontend: Any, operator_uid: str | None, node: ast.AST
) -> FiniteSymbolicPointFieldPlan | None:
    """Resolve a tiny finite symbolic string language to native integer dispatch.

    The accepted language is literal string concatenation plus one proven finite
    selector: ``str(integer_parameter)`` or a finite category parameter.  String
    construction is used only while preparing field names; emitted formulas receive
    a numeric selector code.  Multiple/dynamic selectors and arbitrary formatting
    are deliberately rejected.
    """
    if operator_uid is None:
        return None
    int_domains = getattr(frontend, "finite_symbolic_parameter_domains", {})
    int_sources = getattr(frontend, "finite_symbolic_parameter_domain_sources", {})
    category_domains = getattr(frontend, "static_category_parameter_domains", {})
    nullable_categories = getattr(frontend, "static_category_nullable_parameters", set())

    def marker_for_name(name: str, *, stringified: bool):
        if (operator_uid, name) in int_domains:
            return ("int", name) if stringified else None
        if (operator_uid, name) in category_domains:
            if (operator_uid, name) in nullable_categories:
                return None
            return ("category", name)
        return None

    def parse(expr: ast.AST):
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):
                return [("text", expr.value)]
            return None
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            left = parse(expr.left)
            right = parse(expr.right)
            if left is None or right is None:
                return None
            return left + right
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "str"
            and len(expr.args) == 1
            and not expr.keywords
        ):
            arg = expr.args[0]
            if isinstance(arg, ast.Name):
                marker = marker_for_name(arg.id, stringified=True)
                if marker is not None:
                    return [("selector", marker)]
            ok, value = frontend._proven_static_expr(arg)
            if ok and isinstance(value, (str, int, float, bool)):
                return [("text", str(value))]
            return None
        if isinstance(expr, ast.Name):
            marker = marker_for_name(expr.id, stringified=False)
            if marker is not None:
                return [("selector", marker)]
        ok, value = frontend._proven_static_expr(expr)
        if ok and isinstance(value, str):
            return [("text", value)]
        return None

    pieces = parse(copy.deepcopy(node))
    if not pieces:
        return None
    markers = [piece[1] for piece in pieces if piece[0] == "selector"]
    if not markers or len(set(markers)) != 1:
        return None
    kind, name = markers[0]
    if kind == "int":
        values = tuple(int_domains.get((operator_uid, name), ()))
        source = int_sources.get((operator_uid, name), "finite_integer_source_domain")
    else:
        values = tuple(category_domains.get((operator_uid, name), ()))
        source = "selected_closure_CategoryABI_parameter_domain"
    if not values or len(values) > 32 or len(set(values)) != len(values):
        return None

    fields: list[str] = []
    for value in values:
        parts: list[str] = []
        for tag, payload in pieces:
            parts.append(str(payload) if tag == "text" else str(value))
        fields.append("".join(parts))

    selector = ast.Name(id=name, ctx=ast.Load())
    if kind == "category":
        code_expr: ast.AST = selector
    elif len(values) == 1:
        code_expr = ast.Constant(0)
    elif all(type(value) is int for value in values) and tuple(values) == tuple(
        range(int(values[0]), int(values[0]) + len(values))
    ):
        code_expr = (
            selector if int(values[0]) == 0
            else ast.BinOp(selector, ast.Sub(), ast.Constant(int(values[0])))
        )
    else:
        code_expr = ast.Constant(len(values) - 1)
        for index in range(len(values) - 2, -1, -1):
            code_expr = ast.IfExp(
                test=ast.Compare(
                    left=copy.deepcopy(selector),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(values[index])],
                ),
                body=ast.Constant(index),
                orelse=code_expr,
            )
    return FiniteSymbolicPointFieldPlan(
        selector_name=name,
        selector_values=values,
        field_names=tuple(fields),
        code_expr=ast.fix_missing_locations(code_expr),
        domain_source=source,
    )


@dataclass
class InputSpec:
    key: str
    dtype: str
    ndim: int
    scope: str  # point | global
    getter: Callable[[Sequence[Any]], Any]
    description: str
    domain_guard: Callable[[], str | None] | None = None
    domain_token: str | None = None
    expected_shape: tuple[int, ...] | None = None
    normalized_table: NormalizedTableInput | None = None
    # Optional shared-batch path.  Multiple inputs may reuse one exact indexer
    # calculation inside a single raw_inputs() call without introducing a
    # persistent cache or changing invalidation semantics.
    fast_getter: Callable[[Sequence[Any], dict[Any, Any]], Any] | None = None
    point_row_source: NormalizedPointRowSource | None = None
    point_field: Any | None = None
    normalized_structured_column: NormalizedStructuredColumn | None = None
    enum_labels: tuple[str, ...] | None = None
    category_validity_key: str | None = None
    normalized_lookup_1d: NormalizedLookup1D | None = None
    lookup_1d_role: str | None = None
    hierarchical_itemspace_binding: HierarchicalItemSpaceParameterBinding | None = None
    # Optional I4a-2 preparation contract.  This is deliberately attached to
    # the ordinary scalar global-input specification rather than introducing a
    # second configuration channel.  When present, both Python binding and the
    # generated native ``init_inputs`` path must reject a value that is not the
    # exact proven integer.
    prepared_exact_int_value: int | None = None
    prepared_exact_int_authority: str | None = None
    prepared_exact_int_proof_domain_size: int | None = None
    prepared_exact_int_source_name: str | None = None


@dataclass(frozen=True)
class PointRowBinding:
    """Preparation-time binding for a normalized point-row table source."""

    cell_name: str
    semantic: NormalizedPointRowSource
    provider: Callable[[], pd.DataFrame] = field(compare=False, repr=False)




@dataclass(frozen=True)
class ValidatedEnumCell:
    """Preparation-time description of a validated categorical selector Cell."""

    field: str
    allowed_values: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedPointScalarCell:
    """Guard-bearing numeric/bool selector safely specialized by exact value.

    The source value comes directly from one model-point field (optionally through
    bool/int/float).  Specialization is allowed only when substituting the proven
    value makes every guard predicate in the selector unconditionally false.
    Runtime input validation then requires the field to remain exactly that value.
    """

    field: str
    cast: str | None



@dataclass(frozen=True)
class _IntDomain:
    """Inclusive finite integer interval used only for compile-time source proofs."""

    lo: int
    hi: int

    def __post_init__(self):
        if int(self.lo) > int(self.hi):
            raise ValueError("empty integer domain")

    def union(self, other: "_IntDomain") -> "_IntDomain":
        return _IntDomain(min(int(self.lo), int(other.lo)), max(int(self.hi), int(other.hi)))

    def intersect(self, lo: int | None = None, hi: int | None = None) -> "_IntDomain | None":
        left = int(self.lo) if lo is None else max(int(self.lo), int(lo))
        right = int(self.hi) if hi is None else min(int(self.hi), int(hi))
        return None if left > right else _IntDomain(left, right)

    @property
    def singleton(self) -> int | None:
        return int(self.lo) if int(self.lo) == int(self.hi) else None


@dataclass
class ReductionSpec:
    init: ast.AST
    target: str
    loop_var: str
    range_args: tuple[ast.AST, ...]
    body_expr: ast.AST
    filters: tuple[ast.AST, ...] = ()


@dataclass
class CanonicalVariant:
    uid: str
    source_name: str
    source_fullname: str
    dtype: str
    role: str  # coordinate | scalar | reduction | vector
    function: ast.FunctionDef | None
    reduction: ReductionSpec | None = None
    time_param_source: str | None = None
    aux_values: tuple[Any, ...] = ()
    guards: tuple[GuardSpec, ...] = ()


@dataclass
class CanonicalModel:
    model: Any
    space: Any
    output_name: str
    output_uid: str
    output_invocation: OutputInvocation
    run_domain: RunDomain
    result_domain: ResultDomain
    sample_keys: tuple[Any, ...]
    run_keys: tuple[Any, ...]
    trace: TraceCapture
    variants: dict[str, CanonicalVariant]
    inputs: dict[str, InputSpec]
    executable: ExecutableGraph | None = None
    domain_graph_shadow: DomainGraph | None = None
    canonical_semantic_graph: CanonicalSemanticGraph | None = None
    canonical_schedule_analysis: CanonicalComponentScheduleAnalysis | None = None
    canonical_execution_schedule: CanonicalExecutionSchedule | None = None
    stage_execution_plan: StageExecutionPlan | None = None
    stage_storage_plan: StageStoragePlan | None = None
    normalized_operators: tuple[NormalizedOperator, ...] = ()
    diagnostics: list[str] = field(default_factory=list)
    static_facts: tuple[StaticScalarFact, ...] = ()
    validated_static_facts: tuple[ValidatedStaticFact, ...] = ()
    finite_domain_facts: tuple[FiniteDomainFact, ...] = ()
    fixed_coordinate_facts: tuple[FixedCoordinateFact, ...] = ()
    coordinate_recurrence_facts: tuple[CoordinateRecurrenceDomainFact, ...] = ()
    runtime_scalar_helper_facts: tuple[RuntimeScalarHelperFact, ...] = ()
    source_backed_scheduled_facts: tuple[SourceBackedScheduledFact, ...] = ()
    step_lookup_domain_facts: tuple[StepLookupDomainFact, ...] = ()

    def raw_inputs(self, keys: Sequence[Any] | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        keys = tuple(self.run_keys if keys is None else keys)
        overrides = overrides or {}
        unknown = set(overrides) - set(self.inputs)
        if unknown:
            raise FrontendError(f"unknown input override keys: {sorted(unknown)!r}")
        # Per-call only.  This lets related getters share exact lookup/index work
        # without creating a second cache-invalidation system.
        batch_cache: dict[Any, Any] = {}
        out: dict[str, Any] = {}
        for name, spec in self.inputs.items():
            if name in overrides:
                out[name] = overrides[name]
            elif spec.fast_getter is not None:
                out[name] = spec.fast_getter(keys, batch_cache)
            else:
                out[name] = spec.getter(keys)
        return out

    def coerce_inputs(self, values: dict[str, Any]) -> dict[str, Any]:
        unknown = set(values) - set(self.inputs)
        missing = set(self.inputs) - set(values)
        if unknown or missing:
            raise FrontendError(f"input value keys mismatch; missing={sorted(missing)!r} unknown={sorted(unknown)!r}")
        out = {}
        for name, spec in self.inputs.items():
            value = values[name]
            if spec.ndim == 0:
                if spec.dtype == "int64":
                    out[name] = int(value)
                elif spec.dtype == "bool":
                    out[name] = bool(value)
                else:
                    out[name] = float(value)
            else:
                dt = np.int64 if spec.dtype == "int64" else np.bool_ if spec.dtype == "bool" else np.float64
                arr = np.ascontiguousarray(np.asarray(value, dtype=dt))
                if spec.enum_labels is not None:
                    # -1 is the reserved finite-enum ``other/missing`` sentinel.
                    # Exact table selectors validate it before indexing; equality
                    # dispatch naturally falls through to the source ``else`` path.
                    if np.any(arr < -1) or np.any(arr >= len(spec.enum_labels)):
                        raise FrontendError(
                            f"enum input {name!r} contains a code outside [-1, {len(spec.enum_labels)})"
                        )
                out[name] = arr
        for fact in self.validated_static_facts:
            spec = self.inputs.get(fact.validation_input_key)
            if spec is None:
                raise FrontendError(
                    f"validated static fact {fact.source_name!r} has no validation input"
                )
            arr = np.asarray(out[fact.validation_input_key])
            if spec.enum_labels is not None:
                try:
                    expected = spec.enum_labels.index(fact.value)
                except ValueError as exc:
                    raise FrontendError(
                        f"validated static fact {fact.source_name!r} value is outside its enum domain"
                    ) from exc
                kind = "enum"
            else:
                expected = int(fact.value) if spec.dtype == "int64" else bool(fact.value) if spec.dtype == "bool" else float(fact.value)
                kind = "scalar"
            if np.any(arr != expected):
                raise FrontendError(
                    f"validated static {kind} {fact.source_name!r} changed from proven value {fact.value!r}"
                )
        return out

    def bind_inputs(self, keys: Sequence[Any] | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.coerce_inputs(self.raw_inputs(keys, overrides))

@dataclass(frozen=True)
class CanonicalProofSnapshot:
    """Immutable proof bundle at the canonical/pre-scheduler boundary.

    V02347 strengthens the snapshot so an observational canonical semantic graph
    can be built without first constructing ``ExecutableGraph``.  The object still
    grants no execution permission: source semantics, PureMap proofs and loop
    topology are evidence only until a later scheduler authorizes them.
    """

    output_uid: str
    run_key_count: int
    variants: dict[str, CanonicalVariant] = field(compare=False, repr=False)
    formula_hash: str = ""
    structural_hash: str = ""
    coordinate_domains: tuple[tuple[str, int, int], ...] = ()
    static_facts: tuple[StaticScalarFact, ...] = ()
    validated_static_facts: tuple[ValidatedStaticFact, ...] = ()
    finite_domain_facts: tuple[FiniteDomainFact, ...] = ()
    fixed_coordinate_facts: tuple[FixedCoordinateFact, ...] = ()
    coordinate_recurrence_facts: tuple[CoordinateRecurrenceDomainFact, ...] = ()
    runtime_scalar_helper_facts: tuple[RuntimeScalarHelperFact, ...] = ()
    source_backed_scheduled_facts: tuple[SourceBackedScheduledFact, ...] = ()
    step_lookup_domain_facts: tuple[StepLookupDomainFact, ...] = ()
    normalized_operators: tuple[NormalizedOperator, ...] = ()
    source_value_semantics: tuple[ValueSemanticsEvidence, ...] = ()
    pure_map_specs: tuple[ExecutablePureMapSpec, ...] = ()
    loop_topology_facts: tuple[CanonicalLoopTopologyFact, ...] = ()
    recovered_loop_topology_facts: tuple[RecoveredLoopTopologyFact, ...] = ()
    scheduler_evidence_status: str = "complete_without_recovered_evidence"
    executable_schedule_built: bool = False
    failure_reason: str | None = None

    def _copy(self, **changes: Any) -> "CanonicalProofSnapshot":
        data = {
            "output_uid": self.output_uid,
            "run_key_count": self.run_key_count,
            "variants": self.variants,
            "formula_hash": self.formula_hash,
            "structural_hash": self.structural_hash,
            "coordinate_domains": self.coordinate_domains,
            "static_facts": self.static_facts,
            "validated_static_facts": self.validated_static_facts,
            "finite_domain_facts": self.finite_domain_facts,
            "fixed_coordinate_facts": self.fixed_coordinate_facts,
            "coordinate_recurrence_facts": self.coordinate_recurrence_facts,
            "runtime_scalar_helper_facts": self.runtime_scalar_helper_facts,
            "source_backed_scheduled_facts": self.source_backed_scheduled_facts,
            "step_lookup_domain_facts": self.step_lookup_domain_facts,
            "normalized_operators": self.normalized_operators,
            "source_value_semantics": self.source_value_semantics,
            "pure_map_specs": self.pure_map_specs,
            "loop_topology_facts": self.loop_topology_facts,
            "recovered_loop_topology_facts": self.recovered_loop_topology_facts,
            "scheduler_evidence_status": self.scheduler_evidence_status,
            "executable_schedule_built": self.executable_schedule_built,
            "failure_reason": self.failure_reason,
        }
        data.update(changes)
        return CanonicalProofSnapshot(**data)

    def with_failure(self, reason: str) -> "CanonicalProofSnapshot":
        return self._copy(failure_reason=str(reason))

    def with_executable_schedule(self) -> "CanonicalProofSnapshot":
        return self._copy(executable_schedule_built=True)

    def with_recovered_loop_evidence(self, evidence: Any | None) -> "CanonicalProofSnapshot":
        facts = freeze_recovered_loop_topology(evidence)
        status = (
            "complete_with_recovered_evidence" if facts
            else "complete_without_recovered_evidence"
        )
        return self._copy(
            recovered_loop_topology_facts=facts,
            scheduler_evidence_status=status,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "modelx_graph.canonical_proof_snapshot.v3",
            "output_uid": self.output_uid,
            "run_key_count": int(self.run_key_count),
            "formula_hash": self.formula_hash,
            "structural_hash": self.structural_hash,
            "variant_count": len(self.variants),
            "variants": [
                {
                    "uid": uid,
                    "source_name": cv.source_name,
                    "source_fullname": cv.source_fullname,
                    "role": cv.role,
                    "dtype": cv.dtype,
                    "time_param_source": cv.time_param_source,
                    "aux_values": list(cv.aux_values),
                }
                for uid, cv in sorted(self.variants.items())
            ],
            "coordinate_domains": [
                {"uid": uid, "lo": lo, "hi": hi}
                for uid, lo, hi in self.coordinate_domains
            ],
            "static_facts": [x.manifest() for x in self.static_facts],
            "validated_static_facts": [x.manifest() for x in self.validated_static_facts],
            "finite_domain_facts": [x.manifest() for x in self.finite_domain_facts],
            "fixed_coordinate_facts": [x.manifest() for x in self.fixed_coordinate_facts],
            "coordinate_recurrence_facts": [x.manifest() for x in self.coordinate_recurrence_facts],
            "runtime_scalar_helper_facts": [x.manifest() for x in self.runtime_scalar_helper_facts],
            "source_backed_scheduled_facts": [x.manifest() for x in self.source_backed_scheduled_facts],
            "step_lookup_domain_facts": [x.manifest() for x in self.step_lookup_domain_facts],
            "normalized_operators": [x.manifest() for x in self.normalized_operators],
            "source_value_semantics": [x.manifest() for x in self.source_value_semantics],
            "pure_map_specs": [x.manifest() for x in self.pure_map_specs],
            "loop_topology_facts": [x.manifest() for x in self.loop_topology_facts],
            "recovered_loop_topology_facts": [x.manifest() for x in self.recovered_loop_topology_facts],
            "scheduler_evidence_status": self.scheduler_evidence_status,
            "executable_schedule_built": bool(self.executable_schedule_built),
            "failure_reason": self.failure_reason,
        }


class _Substitute(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, ast.AST]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(copy.deepcopy(self.mapping[node.id]), node)
        return node


class _ConservativeFolder(ast.NodeTransformer):
    """Fold only expressions made entirely from Python literals.

    The compiler never uses model-specific names or values to decide a branch.
    """
    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.copy_location(ast.Constant(not bool(node.operand.value)), node)
        if isinstance(node.operand, ast.Constant) and isinstance(node.op, (ast.UAdd, ast.USub)):
            try:
                value = +node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
                return ast.copy_location(ast.Constant(value), node)
            except Exception:
                pass
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        node = self.generic_visit(node)
        values = list(node.values)
        if isinstance(node.op, ast.And):
            for value in values:
                if isinstance(value, ast.Constant) and not bool(value.value):
                    return ast.copy_location(ast.Constant(False), node)
            values = [v for v in values if not (isinstance(v, ast.Constant) and bool(v.value))]
        elif isinstance(node.op, ast.Or):
            for value in values:
                if isinstance(value, ast.Constant) and bool(value.value):
                    return ast.copy_location(ast.Constant(True), node)
            values = [v for v in values if not (isinstance(v, ast.Constant) and not bool(v.value))]
        if not values:
            return ast.copy_location(ast.Constant(isinstance(node.op, ast.And)), node)
        if len(values) == 1:
            return values[0]
        node.values = values
        return node
    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        try:
            test = ast.literal_eval(node.test)
        except Exception:
            return node
        return node.body if bool(test) else node.orelse

    def visit_IfExp(self, node: ast.IfExp):
        node = self.generic_visit(node)
        try:
            test = ast.literal_eval(node.test)
        except Exception:
            return node
        return node.body if bool(test) else node.orelse

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        try:
            expr = ast.Expression(node)
            val = eval(compile(ast.fix_missing_locations(expr), "<const>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return node
        return ast.copy_location(ast.Constant(bool(val)), node)



class _CoordinateConstantFolder(ast.NodeTransformer):
    """Fold literal arithmetic used only for coordinate specialization.

    This deliberately does not run over general numerical formulas.  It is a small
    proof engine for index/coordinate expressions after already-proven static facts
    have been substituted.
    """

    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)
        if not (isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant)):
            return node
        try:
            value = eval(
                compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(node))), "<coordinate-const>", "eval"),
                {"__builtins__": {}}, {},
            )
        except Exception:
            return node
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, int, float, str)):
            return ast.copy_location(ast.Constant(value=value), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = self.generic_visit(node)
        if not isinstance(node.operand, ast.Constant):
            return node
        try:
            value = eval(
                compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(node))), "<coordinate-const>", "eval"),
                {"__builtins__": {}}, {},
            )
        except Exception:
            return node
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, int, float, str)):
            return ast.copy_location(ast.Constant(value=value), node)
        return node

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Name) and node.func.id in {"int", "float"}
            and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant)
        ):
            return node
        try:
            value = int(node.args[0].value) if node.func.id == "int" else float(node.args[0].value)
        except Exception:
            return node
        return ast.copy_location(ast.Constant(value=value), node)



def _strip_docstring(fn: ast.FunctionDef) -> ast.FunctionDef:
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str):
        fn.body = fn.body[1:]
    return fn


def _prune_source_unreachable_tails(fn: ast.FunctionDef) -> ast.FunctionDef:
    """Remove statements unreachable by Python source control flow.

    This is deliberately weaker than general control-flow simplification.  It
    understands only unconditional ``return``/``raise`` termination and the case
    where both branches of an ``if`` terminate.  In particular it never uses
    representative trace absence as evidence.  The pass runs *after* literal
    auxiliary/static substitution, so a specialized branch folded to a direct
    return also cuts the source tail that can no longer execute.
    """

    def block(stmts: list[ast.stmt]) -> tuple[list[ast.stmt], bool]:
        out: list[ast.stmt] = []
        terminated = False
        for original in stmts:
            if terminated:
                break
            st = copy.deepcopy(original)
            if isinstance(st, ast.If):
                st.body, body_term = block(list(st.body))
                st.orelse, else_term = block(list(st.orelse))
                out.append(st)
                # An if without an else always has a fall-through path.
                terminated = bool(st.orelse) and body_term and else_term
                continue
            out.append(st)
            if isinstance(st, (ast.Return, ast.Raise)):
                terminated = True
        return out, terminated

    out = copy.deepcopy(fn)
    out.body, _ = block(list(out.body))
    return ast.fix_missing_locations(out)


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception as exc:
        raise FrontendError(f"non-time specialization argument is not a literal: {ast.unparse(node)}") from exc


def _safe_numeric_axis(labels: Iterable[Any]) -> tuple[list[int], int, int]:
    vals = []
    for x in labels:
        if isinstance(x, str):
            try:
                x = int(x)
            except Exception as exc:
                raise FrontendError(f"non-numeric table label {x!r} is not yet supported natively") from exc
        if not isinstance(x, (int, np.integer)):
            raise FrontendError(f"non-integral table label {x!r} is not yet supported natively")
        vals.append(int(x))
    uniq = sorted(set(vals))
    if len(uniq) != len(vals):
        raise FrontendError("duplicate table labels are not supported")
    if len(uniq) <= 1:
        step = 1
    else:
        diffs = {b-a for a,b in zip(uniq,uniq[1:])}
        if len(diffs) != 1 or next(iter(diffs)) <= 0:
            raise FrontendError(f"irregular numeric table labels require a fallback lookup: {uniq[:8]}...")
        step = next(iter(diffs))
    return vals, uniq[0] if uniq else 0, step


def _axis_index_expr(expr: ast.AST, start: int, step: int) -> ast.AST:
    # ``:`` selects the entire labelled axis and therefore remains ``:`` after
    # conversion to the zero-based dense input ABI.  Treating a Slice as an
    # ordinary numeric expression used to turn ``[:]`` on a 1-based axis into
    # the nonsensical ``[:-1]``, silently dropping the final scenario.
    if isinstance(expr, ast.Slice):
        if expr.lower is None and expr.upper is None and expr.step is None:
            return copy.deepcopy(expr)
        # Bounded label slices have pandas inclusive/end-point semantics that
        # are not equivalent to blindly transforming Python slice bounds. Fail
        # closed until that semantic mapping is proved explicitly.
        raise FrontendError(
            "bounded slices on labelled numeric axes require explicit semantic lowering"
        )
    out = copy.deepcopy(expr)
    if start:
        out = ast.BinOp(out, ast.Sub(), ast.Constant(start))
    if step != 1:
        out = ast.BinOp(out, ast.FloorDiv(), ast.Constant(step))
    return out


class InputRegistry:
    def __init__(self, frontend: "GraphModelCompiler"):
        self.frontend = frontend
        self.specs: dict[str, InputSpec] = {}
        self.counter = 0
        self.memo: dict[tuple, str] = {}
        # Prepared ordered Mapping traversals own three related CSR inputs.
        # Keep their tuple separately rather than overloading the historical
        # one-token/one-input memo contract.
        self.mapping_ragged_memo: dict[tuple, tuple[str, str, str, str]] = {}
        self.point_lookup_memo: dict[tuple, tuple[str, str]] = {}

    def _key(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def frozen_point_scalar(
        self,
        token: tuple[Any, ...],
        dtype: str,
        getter: Callable[[Sequence[Any]], Any],
        description: str,
    ) -> str:
        """Register a full-RunDomain-proven numeric scalar as a point buffer."""
        if token in self.memo:
            return self.memo[token]
        if dtype not in {"float64", "int64", "bool"}:
            raise FrontendError(f"{description} has unsupported dtype {dtype!r}")
        sample = np.asarray(getter(self.frontend.run_keys))
        if sample.shape != (len(self.frontend.run_keys),):
            raise FrontendError(
                f"{description} produced shape {sample.shape!r}, expected "
                f"{(len(self.frontend.run_keys),)!r}"
            )
        key = self._key("fsp")
        self.memo[token] = key
        self.specs[key] = InputSpec(
            key=key,
            dtype=dtype,
            ndim=1,
            scope="point",
            getter=getter,
            description=description,
            domain_token=repr((token, len(self.frontend.run_keys))),
            expected_shape=(len(self.frontend.run_keys),),
        )
        return key

    def ranked_point_field(self, field: Any) -> str:
        """Bind one numeric column to the explicit rank-1 result axis.

        This is the array-output analogue of ``point_field``.  The owning
        modelx Space can be non-parameterized, so structural RunDomain keys
        remain ``((),)`` while the independent result axis uses the DataFrame
        index captured during modelx-cython tracing.
        """
        frame = self.frontend.point_frame
        keys = tuple(self.frontend.point_keys)
        if frame is None or not (
            type(frame).__name__ == "DataFrame"
            and type(frame).__module__.startswith("pandas.")
        ):
            raise FrontendError("rank-1 point field has no proven DataFrame axis")
        if field not in frame.columns:
            raise FrontendError(f"rank-1 point frame has no field {field!r}")
        values = frame[field]
        if values.dtype.kind in "iub":
            dtype = "int64"
        elif values.dtype.kind in "fc":
            dtype = "float64"
        else:
            raise FrontendError(
                f"rank-1 point field {field!r} has unsupported dtype {values.dtype}"
            )
        token = ("ranked_point_field", repr(field), keys)
        if token in self.memo:
            return self.memo[token]
        key = self._key("rp")
        self.memo[token] = key
        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()

        def getter(selected, source=frame, column=field):
            if not source.index.equals(expected_index) or not source.columns.equals(expected_columns):
                raise FrontendError("rank-1 point frame schema changed after compilation")
            requested = tuple(selected)
            if len(requested) != len(set(requested)):
                raise FrontendError("rank-1 result axis contains duplicate keys")
            try:
                out = source.loc[list(requested), column].to_numpy()
            except Exception as exc:
                raise FrontendError(
                    f"rank-1 point field {column!r} cannot bind the requested axis"
                ) from exc
            np_dtype = np.int64 if dtype == "int64" else np.float64
            return np.asarray(out, dtype=np_dtype)

        self.specs[key] = InputSpec(
            key=key,
            dtype=dtype,
            ndim=1,
            scope="point",
            getter=getter,
            description=f"rank-1 point field {field!r}",
            domain_token=repr(("rank1", tuple(frame.index.names), keys)),
            expected_shape=(len(keys),),
            point_field=str(field),
        )
        return key

    def ranked_structured_field(
        self,
        provider: StructuredFramePlan,
        field: Any,
        *,
        provider_id: str,
        category_labels: Sequence[Any] | None = None,
    ) -> str:
        """Bind one typed column from a logical structured provider.

        The provider owns column lineage and row-domain transforms (source,
        join, repeat/gather).  Binding evaluates those transforms directly from
        frozen source tables; it never executes the modelx provider Cell or
        constructs its pandas DataFrame result.
        """
        field = str(field)
        if provider.output_size != len(self.frontend.point_keys):
            raise FrontendError(
                f"structured provider {provider_id!r} has row domain "
                f"{provider.output_size}, expected ranked axis {len(self.frontend.point_keys)}"
            )
        try:
            type_evidence = provider.column_type(
                field, self.frontend.refs, labels=category_labels
            )
        except StructuredProviderError as exc:
            raise FrontendError(str(exc)) from exc
        dtype = type_evidence.physical_dtype
        labels_token = None if category_labels is None else tuple(category_labels)
        demands = provider.source_demands((field,))
        row_maps = provider.row_map_chain()
        semantic = NormalizedStructuredColumn(
            provider_id=str(provider_id),
            field=field,
            output_size=int(provider.output_size),
            logical_kind=type_evidence.logical_kind,
            physical_dtype=type_evidence.physical_dtype,
            nullable=bool(type_evidence.nullable),
            category_labels=tuple(repr(x) for x in type_evidence.category_labels),
            source_dtype=type_evidence.source_dtype,
            provider_manifest_json=json.dumps(
                provider.manifest(), sort_keys=True, separators=(",", ":")
            ),
            source_demands=tuple(
                NormalizedStructuredSourceDemand(
                    source_ref=row.source_ref,
                    columns=tuple(row.columns),
                    needs_index=bool(row.needs_index),
                )
                for row in demands
            ),
            row_maps=tuple(
                NormalizedStructuredRowMap(
                    kind=row.kind,
                    output_size=int(row.output_size),
                    parent_size=int(row.parent_size),
                    repeat=(int(row.repeat) if row.kind == "repeat_each" else None),
                    gather_size=(len(row.gather) if row.gather else None),
                    gather_sha256=(
                        row.manifest().get("gather_sha256") if row.gather else None
                    ),
                )
                for row in row_maps
            ),
        )
        token = (
            "ranked_structured_field", provider_id, field, labels_token,
            repr(semantic.manifest()),
        )
        if token in self.memo:
            return self.memo[token]
        key = self._key("rsp")
        self.memo[token] = key
        full_axis = tuple(self.frontend.point_keys)
        axis_pos = {value: i for i, value in enumerate(full_axis)}

        def bind_values(
            selected, *, cache=None, frame_plan=provider, column=field,
            labels=labels_token, expected_axis=full_axis, positions=axis_pos,
        ):
            try:
                _dtype, encoded, _nullable = frame_plan.encoded_column_values(
                    column, self.frontend.refs, labels=labels, cache=cache
                )
            except StructuredProviderError as exc:
                raise FrontendError(str(exc)) from exc
            requested = tuple(selected)
            if requested == expected_axis:
                out = encoded
            else:
                try:
                    idx = np.fromiter(
                        (positions[value] for value in requested),
                        dtype=np.int64, count=len(requested),
                    )
                except KeyError as exc:
                    raise FrontendError(
                        f"structured column {column!r} requested an unknown ranked key"
                    ) from exc
                out = encoded[idx]
            np_dtype = np.int64 if dtype == "int64" else np.float64
            return np.ascontiguousarray(out, dtype=np_dtype)

        def getter(selected):
            return bind_values(selected)

        def fast_getter(selected, cache):
            return bind_values(selected, cache=cache)

        self.specs[key] = InputSpec(
            key=key,
            dtype=dtype,
            ndim=1,
            scope="point",
            getter=getter,
            description=f"structured provider {provider_id}:{field}",
            domain_token=repr(("structured", semantic.manifest())),
            expected_shape=(provider.output_size,),
            fast_getter=fast_getter,
            point_field=field,
            normalized_structured_column=semantic,
            enum_labels=None if labels_token is None else tuple(str(x) for x in labels_token),
        )
        return key

    @staticmethod
    def _numeric_scalar(value: Any, description: str) -> tuple[str, Any]:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, np.bool_)):
            return "bool", bool(value)
        if isinstance(value, (int, np.integer)):
            return "int64", int(value)
        if isinstance(value, (float, np.floating)):
            value = float(value)
            if not np.isfinite(value):
                raise FrontendError(f"{description} returned a non-finite scalar")
            return "float64", value
        raise FrontendError(
            f"{description} returned unsupported non-numeric {type(value).__name__}"
        )

    @classmethod
    def _numeric_dtype(cls, values: Sequence[Any], description: str) -> str:
        if not values:
            raise FrontendError(f"{description} has an empty materialization domain")
        dtypes = [cls._numeric_scalar(value, description)[0] for value in values]
        if "float64" in dtypes:
            return "float64"
        if "int64" in dtypes:
            return "int64"
        return "bool"

    @classmethod
    def _uniform_primitive_dtype(
        cls, values: Sequence[Any], description: str
    ) -> str:
        """Prove one exact native primitive type over an authoritative domain.

        This is intentionally stricter than ``_numeric_dtype``.  It is used when
        the source container itself is object/mixed-typed, so numeric promotion
        would amount to inventing a type the source evidence did not establish.
        """
        if not values:
            raise FrontendError(f"{description} has an empty proof domain")
        observed = [cls._numeric_scalar(value, description)[0] for value in values]
        unique = tuple(dict.fromkeys(observed))
        if len(unique) != 1:
            raise FrontendError(
                f"{description} is heterogeneous over the complete proof domain: {unique!r}"
            )
        return unique[0]

    @classmethod
    def _coerce_uniform_primitive_values(
        cls, values: Sequence[Any], dtype: str, description: str
    ) -> np.ndarray:
        observed = cls._uniform_primitive_dtype(values, description)
        if observed != dtype:
            raise FrontendError(
                f"{description} changed primitive type from {dtype} to {observed}"
            )
        return cls._coerce_numeric_values(values, dtype, description)

    @classmethod
    def _coerce_numeric_values(
        cls, values: Sequence[Any], dtype: str, description: str
    ) -> np.ndarray:
        normalized: list[Any] = []
        for value in values:
            observed, numeric = cls._numeric_scalar(value, description)
            if dtype == "bool" and observed != "bool":
                raise FrontendError(f"{description} changed from bool to {observed}")
            if dtype == "int64" and observed not in {"bool", "int64"}:
                raise FrontendError(f"{description} changed from integer to {observed}")
            normalized.append(numeric)
        np_dtype = np.bool_ if dtype == "bool" else np.int64 if dtype == "int64" else np.float64
        return np.asarray(normalized, dtype=np_dtype)

    def regional_fallback_point(self, cell_name: str, dtype: str) -> str:
        token=("regional_fallback_point", cell_name)
        if token in self.memo:
            return self.memo[token]
        key=self._key("fb")
        self.memo[token]=key
        if dtype not in ("int64", "bool", "float64"):
            raise FrontendError(f"regional fallback Cell {cell_name} has unsupported observed dtype {dtype}")
        def getter(keys, space=self.frontend.space, name=cell_name):
            return np.asarray([getattr(self.frontend._space_instance(k), name)() for k in keys])
        self.specs[key]=InputSpec(
            key, dtype, 1, "point", getter,
            f"regional fallback point scalar {cell_name}; overrideable by a callback"
        )
        return key

    def reference_metadata_scalar(self, cell_name: str) -> str:
        """Bind one point-independent finite reference-metadata Cell as scalar input.

        Admission is performed by ``GraphModelCompiler``.  Binding re-evaluates the
        Cell over the exact run domain and requires one identical numeric value, so
        table/reference changes cannot silently reuse a stale compile-time literal.
        """
        token=("reference_metadata_scalar", cell_name)
        if token in self.memo:
            return self.memo[token]
        if not self.frontend.run_keys:
            raise FrontendError("reference metadata scalar requires a non-empty run domain")

        def evaluate(keys, name=cell_name):
            values=[]
            for key in keys:
                value=getattr(self.frontend._space_instance(key), name)()
                if isinstance(value, np.generic):
                    value=value.item()
                if not isinstance(value,(bool,int,float)):
                    raise FrontendError(
                        f"reference metadata Cell {name!r} returned non-numeric {type(value).__name__}"
                    )
                values.append(value)
            if not values:
                raise FrontendError(f"reference metadata Cell {name!r} has an empty run domain")
            first=values[0]
            if not all(type(value) is type(first) and value == first for value in values[1:]):
                raise FrontendError(
                    f"reference metadata Cell {name!r} is not point-independent over the frozen run domain"
                )
            return first

        sample=evaluate(self.frontend.run_keys)
        if isinstance(sample,bool): dtype="bool"
        elif isinstance(sample,int): dtype="int64"
        else: dtype="float64"
        key=self._key("meta")
        self.memo[token]=key
        self.specs[key]=InputSpec(
            key,dtype,0,"global",evaluate,
            f"finite point-independent reference metadata Cell {cell_name}",
            domain_token=f"reference_metadata:{cell_name}:{len(self.frontend.run_keys)}",
            expected_shape=(),
        )
        return key

    def scalar_ref(self, ref_name: str, ref_obj) -> str:
        token=("scalar_ref",ref_name)
        if token in self.memo: return self.memo[token]
        key = self._key("g")
        self.memo[token]=key
        dtype = "int64" if isinstance(ref_obj, (int, np.integer, bool, np.bool_)) else "float64"
        def getter(keys, space=self.frontend.space, name=ref_name):
            return space.refs[name]
        self.specs[key] = InputSpec(key, dtype, 0, "global", getter, f"Reference {ref_name}")
        return key

    def validated_exact_int_scalar_ref(
        self,
        ref_name: str,
        ref_obj: Any,
        *,
        expected_value: int,
        proof_domain_size: int,
        authority: str = "complete_run_domain_static_reference",
    ) -> str:
        """Strengthen an ordinary scalar ref with an exact-int input contract.

        The key and ABI remain the same as ``scalar_ref``.  The additional
        contract exists only when preparation has independently proven an exact
        integer value over the complete frozen artifact domain.  Rebinding the
        reference after preparation therefore fails during input binding instead
        of silently invalidating a fixed-count-control proof.
        """

        if isinstance(expected_value, bool) or type(expected_value) is not int:
            raise FrontendError("prepared exact-int scalar ref requires a Python int, not bool/float")
        if expected_value < -(1 << 63) or expected_value > (1 << 63) - 1:
            raise FrontendError("prepared exact-int scalar ref lies outside signed int64")
        key = self.scalar_ref(ref_name, ref_obj)
        spec = self.specs[key]
        if spec.dtype != "int64" or spec.ndim != 0 or spec.scope != "global":
            raise FrontendError(
                f"prepared exact-int scalar ref {ref_name!r} lacks scalar int64 global ABI"
            )
        if spec.prepared_exact_int_value is not None:
            if (
                spec.prepared_exact_int_value != expected_value
                or spec.prepared_exact_int_source_name != ref_name
            ):
                raise FrontendError(
                    f"prepared exact-int scalar ref {ref_name!r} received contradictory evidence"
                )
            return key

        original_getter = spec.getter

        def exact_getter(
            keys,
            *,
            getter=original_getter,
            expected=int(expected_value),
            name=ref_name,
        ):
            value = getter(keys)
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool) or type(value) is not int or int(value) != expected:
                raise FrontendError(
                    f"prepared exact-int reference {name!r} changed after compilation: "
                    f"expected {expected!r}, got {value!r}"
                )
            return int(value)

        spec.getter = exact_getter
        spec.prepared_exact_int_value = int(expected_value)
        spec.prepared_exact_int_authority = str(authority)
        spec.prepared_exact_int_proof_domain_size = int(proof_domain_size)
        spec.prepared_exact_int_source_name = ref_name
        spec.domain_token = repr((
            "prepared_exact_int",
            ref_name,
            int(expected_value),
            int(proof_domain_size),
            str(authority),
        ))
        spec.description = (
            spec.description
            + f"; exact prepared int {expected_value} ({authority}, domain={proof_domain_size})"
        )
        return key

    def point_field(
        self, source: PointRowBinding, field: Any, *,
        enum_labels: tuple[str, ...] | None = None,
        enum_required_keys: Sequence[Any] | None = None,
        enum_allow_missing: bool = False,
    ) -> str:
        semantic = source.semantic
        enum_labels = None if enum_labels is None else tuple(enum_labels)
        enum_required_keys = None if enum_required_keys is None else tuple(enum_required_keys)
        token=(
            "point_field", semantic.source_kind, semantic.source_ref,
            semantic.source_cell, semantic.key_position, repr(field), enum_labels,
            enum_required_keys, bool(enum_allow_missing),
        )
        if token in self.memo:
            return self.memo[token]
        frame = source.provider()
        if not isinstance(frame, pd.DataFrame):
            raise FrontendError("normalized point-row source no longer returns a DataFrame")
        if field not in frame.columns:
            raise FrontendError(
                f"normalized point-row source has no field {field!r}"
            )
        vals = frame[field]
        row_pos = int(semantic.key_position)
        nparams = len(self.frontend.space_params)
        run_keys = self.frontend.run_keys

        def rows_for(keys, pos=row_pos, nparams=nparams):
            if nparams > 1:
                return [k[pos] if isinstance(k, tuple) else k for k in keys]
            return keys

        object_primitive_dtype: str | None = None
        enum_codes: dict[str, int] | None = None
        if enum_labels is not None:
            if not enum_labels or len(enum_labels) > 16 or len(set(enum_labels)) != len(enum_labels):
                raise FrontendError("validated enum point field requires 1..16 unique labels")
            if not all(isinstance(label, str) for label in enum_labels):
                raise FrontendError("validated enum labels must be strings")
            enum_codes = {label: i for i, label in enumerate(enum_labels)}
            dtype = "int64"
        elif vals.dtype.kind in "iub":
            dtype = "int64"
        elif vals.dtype.kind in "fc":
            dtype = "float64"
        elif vals.dtype.kind == "O":
            try:
                complete_values = frame.loc[list(rows_for(run_keys)), field].tolist()
                dtype = self._uniform_primitive_dtype(
                    complete_values,
                    f"point field {field!r} object->primitive narrowing",
                )
            except Exception as exc:
                if isinstance(exc, FrontendError):
                    raise
                raise FrontendError(
                    f"point field {field!r} could not be proven primitive over the complete RunDomain"
                ) from exc
            object_primitive_dtype = dtype
        else:
            raise FrontendError(
                f"point field {field!r} has unsupported non-numeric dtype {vals.dtype}"
            )
        key = self._key("p")
        self.memo[token]=key
        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()

        run_pos = frame.index.get_indexer(rows_for(run_keys))
        if np.any(run_pos < 0):
            raise FrontendError(
                f"run_keys contain labels not present in normalized point-row source "
                f"{semantic.source_ref!r}"
            )
        run_slice = None
        if len(run_pos):
            first = int(run_pos[0])
            if np.array_equal(run_pos, np.arange(first, first + len(run_pos))):
                run_slice = slice(first, first + len(run_pos))

        def current_frame(provider=source.provider, ei=expected_index, ec=expected_columns):
            table = provider()
            if not isinstance(table, pd.DataFrame):
                raise FrontendError("normalized point-row provider changed type after compilation")
            if not table.index.equals(ei) or not table.columns.equals(ec):
                raise FrontendError("normalized point-row table domain changed after compilation")
            return table

        def encode_enum(values, *, codes=enum_codes, f=field):
            if codes is None:
                return values
            out = np.empty(len(values), dtype=np.int64)
            for j, value in enumerate(values):
                if value in codes:
                    out[j] = codes[value]
                    continue
                if enum_allow_missing:
                    try:
                        missing = bool(pd.isna(value))
                    except Exception:
                        missing = False
                    if missing:
                        out[j] = -1
                        continue
                    raise FrontendError(
                        f"point enum field {f!r} contains unknown non-missing label {value!r}"
                    )
                out[j] = codes.get(value, -1)
            return out

        def primitive(values, *, dtype=object_primitive_dtype, f=field):
            if dtype is None:
                return values
            return self._coerce_uniform_primitive_values(
                list(values), dtype, f"point field {f!r} object->primitive narrowing"
            )

        def getter(keys, f=field, rows_for=rows_for, current=current_frame, encode=encode_enum):
            table = current()
            rows = list(rows_for(keys))
            values = table.loc[rows, f].to_numpy()
            return encode(primitive(values))

        def fast_getter(
            keys, cache, *, f=field, rows_for=rows_for, current=current_frame,
            run_keys=run_keys, run_pos=run_pos, run_slice=run_slice,
            cache_id=(semantic.source_kind, semantic.source_ref, semantic.source_cell),
        ):
            table_key = ("point_row_table", cache_id)
            table = cache.get(table_key)
            if table is None:
                table = current()
                cache[table_key] = table
            selector_key = ("point_row_rows", cache_id, row_pos, tuple(keys))
            selector = cache.get(selector_key)
            if selector is None:
                if keys is run_keys or keys == run_keys:
                    selector = run_slice if run_slice is not None else run_pos
                else:
                    pos_arr = table.index.get_indexer(rows_for(keys))
                    if np.any(pos_arr < 0):
                        raise KeyError("ItemSpace key is missing from normalized point-row table index")
                    selector = pos_arr
                    if len(pos_arr):
                        first = int(pos_arr[0])
                        if np.array_equal(pos_arr, np.arange(first, first + len(pos_arr))):
                            selector = slice(first, first + len(pos_arr))
                cache[selector_key] = selector
            values = table[f].to_numpy(copy=False)[selector]
            return encode_enum(primitive(values))

        def domain_guard(
            current=current_frame, *, f=field, codes=enum_codes, rows_for=rows_for, run_keys=run_keys,
            required_keys=enum_required_keys, allow_missing=enum_allow_missing,
        ):
            try:
                table = current()
            except FrontendError as exc:
                return f"point_row_domain:{exc}"
            if codes is not None:
                try:
                    checked_keys = run_keys if required_keys is None else required_keys
                    rows = list(rows_for(checked_keys))
                    values = table.loc[rows, f].tolist()
                except Exception as exc:
                    return f"point_row_enum_domain:{exc}"
                def invalid(value):
                    if value in codes:
                        return False
                    if allow_missing:
                        try:
                            return not bool(pd.isna(value))
                        except Exception:
                            return True
                    return True
                unknown = tuple(sorted({repr(v) for v in values if invalid(v)}))
                if unknown:
                    return (
                        f"point_row_enum_domain:field {f!r} contains labels outside frozen "
                        f"domain {tuple(codes)!r}: {unknown!r}"
                    )
            return None

        domain_token = repr((tuple(frame.index.tolist()), tuple(frame.columns.tolist())))
        validity_key = None
        if enum_codes is not None and enum_allow_missing:
            validity_key = self._key("pv")

            def validity(values, *, codes=enum_codes):
                return np.asarray([value in codes for value in values], dtype=np.bool_)

            def validity_getter(keys, f=field, rows_for=rows_for, current=current_frame):
                table = current()
                rows = list(rows_for(keys))
                values = table.loc[rows, f].tolist()
                encode_enum(values)
                return validity(values)

            def validity_fast_getter(
                keys, cache, *, f=field, rows_for=rows_for, current=current_frame,
                run_keys=run_keys, run_pos=run_pos, run_slice=run_slice,
                cache_id=(semantic.source_kind, semantic.source_ref, semantic.source_cell),
            ):
                table_key = ("point_row_table", cache_id)
                table = cache.get(table_key)
                if table is None:
                    table = current(); cache[table_key] = table
                selector_key = ("point_row_rows", cache_id, row_pos, tuple(keys))
                selector = cache.get(selector_key)
                if selector is None:
                    if keys is run_keys or keys == run_keys:
                        selector = run_slice if run_slice is not None else run_pos
                    else:
                        pos_arr = table.index.get_indexer(rows_for(keys))
                        if np.any(pos_arr < 0):
                            raise KeyError("ItemSpace key is missing from normalized point-row table index")
                        selector = pos_arr
                    cache[selector_key] = selector
                values = table[f].to_numpy(copy=False)[selector]
                encode_enum(values)
                return validity(values)

            self.specs[validity_key] = InputSpec(
                validity_key, "bool", 1, "point", validity_getter,
                f"point row category validity {field!r} by ItemSpace key",
                domain_guard=domain_guard, domain_token=domain_token, expected_shape=None,
                fast_getter=validity_fast_getter, point_row_source=semantic,
                point_field=field,
            )
        self.specs[key] = InputSpec(
            key, dtype, 1, "point", getter,
            f"point row field {field!r} by ItemSpace key"
            + ("; complete-RunDomain object->primitive proof" if object_primitive_dtype else ""),
            domain_guard=domain_guard, domain_token=domain_token, expected_shape=None,
            fast_getter=fast_getter, point_row_source=semantic, point_field=field,
            enum_labels=enum_labels, category_validity_key=validity_key,
        )
        return key

    def finite_symbolic_point_field_dispatch(
        self,
        source: PointRowBinding,
        selector_values: Sequence[Any],
        field_names: Sequence[str],
        *,
        domain_source: str,
    ) -> tuple[str, str]:
        """Freeze a finite symbolic point-field family into a numeric table.

        Source field names and their symbolic construction are preparation-only.
        Native execution receives a dense ``point ordinal x selector code`` table
        plus the existing point ordinal input.  Every selected field is proved to
        have one identical primitive dtype over the complete RunDomain and the
        point-row schema/type proof is repeated whenever inputs are rebound.
        """
        selector_values = tuple(selector_values)
        field_names = tuple(str(name) for name in field_names)
        domain_source = str(domain_source)
        if not selector_values or len(selector_values) != len(field_names):
            raise FrontendError("finite symbolic point-field dispatch has an invalid case domain")
        if len(selector_values) > 32 or len(set(selector_values)) != len(selector_values):
            raise FrontendError("finite symbolic point-field dispatch requires 1..32 unique selector values")
        if len(self.frontend.run_keys) * len(field_names) > RUNTIME_SCALAR_HELPER_EXPANSION_LIMIT:
            raise FrontendError("finite symbolic point-field dispatch materialization exceeds safety limit")

        semantic = source.semantic
        token = (
            "finite_symbolic_point_field_dispatch",
            semantic.source_kind,
            semantic.source_ref,
            semantic.source_cell,
            semantic.key_position,
            tuple(repr(value) for value in selector_values),
            field_names,
            domain_source,
        )
        cached = self.memo.get(token)
        if cached is not None:
            return cached, self.point_ordinal()

        frame = source.provider()
        if not isinstance(frame, pd.DataFrame):
            raise FrontendError("normalized point-row source no longer returns a DataFrame")
        missing = tuple(name for name in field_names if name not in frame.columns)
        if missing:
            raise FrontendError(
                f"finite symbolic point-field dispatch fields are missing: {missing!r}"
            )

        row_pos = int(semantic.key_position)
        nparams = len(self.frontend.space_params)
        run_keys = self.frontend.run_keys

        def rows_for(keys, pos=row_pos, nparams=nparams):
            if nparams > 1:
                return [k[pos] if isinstance(k, tuple) else k for k in keys]
            return keys

        proof_rows = list(rows_for(run_keys))
        try:
            proof_values = frame.loc[proof_rows, list(field_names)].to_numpy().reshape(-1).tolist()
        except Exception as exc:
            raise FrontendError(
                "finite symbolic point-field dispatch could not read the complete RunDomain"
            ) from exc
        dtype = self._uniform_primitive_dtype(
            proof_values, "finite symbolic point-field dispatch"
        )
        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()

        def current_frame(provider=source.provider, ei=expected_index, ec=expected_columns):
            table = provider()
            if not isinstance(table, pd.DataFrame):
                raise FrontendError("normalized point-row provider changed type after compilation")
            if not table.index.equals(ei) or not table.columns.equals(ec):
                raise FrontendError("normalized point-row table domain changed after compilation")
            return table

        def materialize(keys, current=current_frame, fields=field_names, dtype=dtype):
            table = current()
            rows = list(rows_for(keys))
            try:
                raw = table.loc[rows, list(fields)].to_numpy().reshape(-1).tolist()
            except Exception as exc:
                raise FrontendError(
                    "finite symbolic point-field dispatch binding domain changed"
                ) from exc
            values = self._coerce_uniform_primitive_values(
                raw, dtype, "finite symbolic point-field dispatch"
            )
            return values.reshape(len(rows), len(fields))

        def domain_guard(current=current_frame, dtype=dtype):
            try:
                table = current()
                raw = table.loc[proof_rows, list(field_names)].to_numpy().reshape(-1).tolist()
                observed = self._uniform_primitive_dtype(
                    raw, "finite symbolic point-field dispatch"
                )
            except FrontendError as exc:
                return f"finite_symbolic_point_field:{exc}"
            except Exception as exc:
                return f"finite_symbolic_point_field:{type(exc).__name__}:{exc}"
            if observed != dtype:
                return f"finite_symbolic_point_field:type_changed:{dtype}->{observed}"
            return None

        key = self._key("fsp")
        self.memo[token] = key
        self.specs[key] = InputSpec(
            key=key,
            dtype=dtype,
            ndim=2,
            scope="global",
            getter=materialize,
            description=(
                "finite symbolic point-field dispatch; preparation-owned field names; "
                f"cases={tuple(zip(selector_values, field_names))!r}; source={domain_source}"
            ),
            domain_guard=domain_guard,
            domain_token=repr((
                "finite_symbolic_point_field_dispatch",
                tuple(frame.index.tolist()),
                tuple(frame.columns.tolist()),
                tuple(repr(value) for value in selector_values),
                field_names,
                dtype,
                domain_source,
            )),
            expected_shape=(len(run_keys), len(field_names)),
            point_row_source=semantic,
        )
        return key, self.point_ordinal()

    def external_dataframe_step_lookup(
        self, ref_name: str, cell_name: str, cell, *, row_key: Any, column: Any,
        below_first: str, default_value: float | None = None, whole_table: bool = False,
    ) -> tuple[str, str]:
        """Materialize a sorted numeric 1-D axis/value pair for step lookup.

        Source-table interpretation stays in preparation.  Runtime backends receive
        two ordinary global float64 arrays plus normalized lookup provenance.
        """
        if below_first not in {"default", "first"}:
            raise FrontendError(f"unsupported step-lookup below-first policy {below_first!r}")
        if below_first == "default" and default_value is None:
            raise FrontendError("step lookup default policy requires a default value")
        token = (
            "external_dataframe_step_lookup", ref_name, cell_name, bool(whole_table), repr(row_key), repr(column),
            below_first, None if default_value is None else float(default_value),
        )
        if token in self.memo:
            axis_key, value_key = self.memo[token]
            return axis_key, value_key
        frame = cell()
        if not isinstance(frame, pd.DataFrame):
            raise FrontendError(f"external table {ref_name}.{cell_name} no longer returns a DataFrame")
        if whole_table:
            if isinstance(frame.index, pd.MultiIndex):
                raise FrontendError("whole-table step lookup requires a one-dimensional index")
            sub = frame
        else:
            try:
                sub = frame.loc[row_key]
            except Exception as exc:
                raise FrontendError(
                    f"external step lookup slice {ref_name}.{cell_name}.loc[{row_key!r}] failed"
                ) from exc
            if isinstance(sub, pd.Series):
                sub = sub.to_frame().T
        if not isinstance(sub, pd.DataFrame):
            raise FrontendError("normalized step lookup requires a DataFrame row slice")
        if column not in sub.columns:
            raise FrontendError(f"external step lookup has no column {column!r}")
        try:
            axis = np.asarray([float(x) for x in sub.index], dtype=np.float64)
        except Exception as exc:
            raise FrontendError("normalized step lookup axis must be numeric") from exc
        if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
            raise FrontendError("normalized step lookup axis must be a non-empty finite 1-D numeric axis")
        if len(np.unique(axis)) != len(axis):
            raise FrontendError("normalized step lookup axis must be unique")
        vals_raw = np.asarray(sub[column])
        if vals_raw.dtype.kind not in "iufc":
            raise FrontendError(f"normalized step lookup column {column!r} must be numeric")
        values = np.asarray(vals_raw, dtype=np.float64)
        order = np.argsort(axis, kind="stable")
        axis = np.ascontiguousarray(axis[order], dtype=np.float64)
        values = np.ascontiguousarray(values[order], dtype=np.float64)
        if np.any(np.diff(axis) <= 0):
            raise FrontendError("normalized step lookup axis must be strictly increasing")

        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()
        semantic = NormalizedLookup1D(
            kind="step", source_ref=ref_name, source_cell=cell_name, source_shape=tuple(frame.shape),
            row_selector=row_key, selected_column=column, axis_labels=tuple(float(x) for x in axis),
            below_first=below_first,
            default_value=(None if default_value is None else float(default_value)),
        )
        axis_key = self._key("lkx")
        value_key = self._key("lky")
        self.memo[token] = (axis_key, value_key)

        def current(c=cell, ei=expected_index, ec=expected_columns, rk=row_key, col=column, whole=whole_table):
            table = c()
            if not isinstance(table, pd.DataFrame):
                raise FrontendError("external lookup provider changed type after compilation")
            if not table.index.equals(ei) or not table.columns.equals(ec):
                raise FrontendError("external lookup table domain changed after compilation")
            if whole:
                ss = table
            else:
                ss = table.loc[rk]
                if isinstance(ss, pd.Series):
                    ss = ss.to_frame().T
            if not isinstance(ss, pd.DataFrame) or col not in ss.columns:
                raise FrontendError("external lookup slice changed after compilation")
            return ss

        def axis_getter(keys, current=current):
            ss = current()
            aa = np.asarray([float(x) for x in ss.index], dtype=np.float64)
            oo = np.argsort(aa, kind="stable")
            return np.ascontiguousarray(aa[oo], dtype=np.float64)

        def values_getter(keys, current=current, col=column):
            ss = current()
            aa = np.asarray([float(x) for x in ss.index], dtype=np.float64)
            oo = np.argsort(aa, kind="stable")
            return np.ascontiguousarray(np.asarray(ss[col], dtype=np.float64)[oo], dtype=np.float64)

        def domain_guard(current=current, expected_axis=axis, rn=ref_name, cn=cell_name):
            try:
                ss = current()
                aa = np.asarray([float(x) for x in ss.index], dtype=np.float64)
                oo = np.argsort(aa, kind="stable")
                aa = np.ascontiguousarray(aa[oo], dtype=np.float64)
            except Exception as exc:
                return f"lookup1d_domain:{rn}.{cn}:{exc}"
            if aa.shape != expected_axis.shape or not np.array_equal(aa, expected_axis):
                return f"lookup1d_domain:{rn}.{cn}:axis_changed"
            return None

        domain_token = repr((tuple(frame.index.tolist()), tuple(frame.columns.tolist()), row_key, column))
        self.specs[axis_key] = InputSpec(
            axis_key, "float64", 1, "global", axis_getter,
            f"normalized step lookup axis {ref_name}.{cell_name}[{row_key!r}]",
            domain_guard=domain_guard, domain_token=domain_token, expected_shape=tuple(axis.shape),
            normalized_lookup_1d=semantic, lookup_1d_role="axis",
        )
        self.specs[value_key] = InputSpec(
            value_key, "float64", 1, "global", values_getter,
            f"normalized step lookup values {ref_name}.{cell_name}[{row_key!r}, {column!r}]",
            domain_guard=domain_guard, domain_token=domain_token, expected_shape=tuple(values.shape),
            normalized_lookup_1d=semantic, lookup_1d_role="values",
        )
        return axis_key, value_key

    def external_dataframe_interp_lookup(
        self, ref_name: str, cell_name: str, cell, *, row_key: Any, column: Any,
        interpolation: str, extrapolation: str = "endpoint_segment",
    ) -> tuple[str, str]:
        """Materialize a sorted numeric 1-D axis/value pair for interpolation.

        Preparation owns all pandas/modelx interpretation.  Runtime backends see
        only contiguous float64 arrays and a small mathematical intrinsic.  The
        supported extrapolation continues the nearest endpoint segment, matching
        the CI_UK_S pivot-table formula.
        """
        if interpolation not in {"linear", "log_linear"}:
            raise FrontendError(f"unsupported interpolation mode {interpolation!r}")
        if extrapolation != "endpoint_segment":
            raise FrontendError(f"unsupported interpolation extrapolation {extrapolation!r}")
        token = (
            "external_dataframe_interp_lookup", ref_name, cell_name, repr(row_key), repr(column),
            interpolation, extrapolation,
        )
        if token in self.memo:
            return self.memo[token]
        frame = cell()
        if not isinstance(frame, pd.DataFrame):
            raise FrontendError(f"external table {ref_name}.{cell_name} no longer returns a DataFrame")
        try:
            sub = frame.loc[row_key]
        except Exception as exc:
            raise FrontendError(
                f"external interpolation slice {ref_name}.{cell_name}.loc[{row_key!r}] failed"
            ) from exc
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        if not isinstance(sub, pd.DataFrame):
            raise FrontendError("normalized interpolation requires a DataFrame row slice")
        if column not in sub.columns:
            raise FrontendError(f"external interpolation has no column {column!r}")
        try:
            axis = np.asarray([float(x) for x in sub.index], dtype=np.float64)
        except Exception as exc:
            raise FrontendError("normalized interpolation axis must be numeric") from exc
        raw = np.asarray(sub[column])
        if raw.dtype.kind not in "iuf":
            raise FrontendError(f"normalized interpolation column {column!r} must be real numeric")
        values = np.asarray(raw, dtype=np.float64)
        if axis.ndim != 1 or axis.size < 2 or values.shape != axis.shape:
            raise FrontendError("normalized interpolation requires at least two 1-D axis/value points")
        if not (np.all(np.isfinite(axis)) and np.all(np.isfinite(values))):
            raise FrontendError("normalized interpolation axis and values must be finite")
        if len(np.unique(axis)) != len(axis):
            raise FrontendError("normalized interpolation axis must be unique")
        order = np.argsort(axis, kind="stable")
        axis = np.ascontiguousarray(axis[order], dtype=np.float64)
        values = np.ascontiguousarray(values[order], dtype=np.float64)
        if np.any(np.diff(axis) <= 0):
            raise FrontendError("normalized interpolation axis must be strictly increasing")
        if interpolation == "log_linear" and np.any(values <= 0.0):
            raise FrontendError("log-linear interpolation requires strictly positive values")

        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()
        semantic = NormalizedLookup1D(
            kind="interp", source_ref=ref_name, source_cell=cell_name, source_shape=tuple(frame.shape),
            row_selector=row_key, selected_column=column, axis_labels=tuple(float(x) for x in axis),
            interpolation=interpolation, extrapolation=extrapolation,
        )
        axis_key = self._key("lkx")
        value_key = self._key("lky")
        self.memo[token] = (axis_key, value_key)

        def current(c=cell, ei=expected_index, ec=expected_columns, rk=row_key, col=column):
            table = c()
            if not isinstance(table, pd.DataFrame):
                raise FrontendError("external interpolation provider changed type after compilation")
            if not table.index.equals(ei) or not table.columns.equals(ec):
                raise FrontendError("external interpolation table domain changed after compilation")
            ss = table.loc[rk]
            if isinstance(ss, pd.Series):
                ss = ss.to_frame().T
            if not isinstance(ss, pd.DataFrame) or col not in ss.columns:
                raise FrontendError("external interpolation slice changed after compilation")
            return ss

        def normalized_pair(current=current):
            ss = current()
            aa = np.asarray([float(x) for x in ss.index], dtype=np.float64)
            vv = np.asarray(ss[column], dtype=np.float64)
            oo = np.argsort(aa, kind="stable")
            return (
                np.ascontiguousarray(aa[oo], dtype=np.float64),
                np.ascontiguousarray(vv[oo], dtype=np.float64),
            )

        def axis_getter(keys, pair=normalized_pair):
            return pair()[0]

        def values_getter(keys, pair=normalized_pair):
            return pair()[1]

        def domain_guard(
            pair=normalized_pair, expected_axis=axis, mode=interpolation, rn=ref_name, cn=cell_name
        ):
            try:
                aa, vv = pair()
            except Exception as exc:
                return f"lookup1d_domain:{rn}.{cn}:{exc}"
            if aa.shape != expected_axis.shape or not np.array_equal(aa, expected_axis):
                return f"lookup1d_domain:{rn}.{cn}:axis_changed"
            if not np.all(np.isfinite(vv)):
                return f"lookup1d_domain:{rn}.{cn}:values_nonfinite"
            if mode == "log_linear" and np.any(vv <= 0.0):
                return f"lookup1d_domain:{rn}.{cn}:values_nonpositive"
            return None

        domain_token = repr((
            tuple(frame.index.tolist()), tuple(frame.columns.tolist()), row_key, column,
            interpolation, extrapolation,
        ))
        common = dict(
            domain_guard=domain_guard, domain_token=domain_token, expected_shape=tuple(axis.shape),
            normalized_lookup_1d=semantic,
        )
        self.specs[axis_key] = InputSpec(
            axis_key, "float64", 1, "global", axis_getter,
            f"normalized interpolation axis {ref_name}.{cell_name}[{row_key!r}]",
            lookup_1d_role="axis", **common,
        )
        self.specs[value_key] = InputSpec(
            value_key, "float64", 1, "global", values_getter,
            f"normalized interpolation values {ref_name}.{cell_name}[{row_key!r}, {column!r}]",
            lookup_1d_role="values", **common,
        )
        return axis_key, value_key

    def external_dataframe_interval_lookup(
        self, ref_name: str, cell_name: str, cell, *, lower_column: Any, upper_column: Any,
        value_column: Any, closure: str = "(lo,hi]", fallback: str = "last",
        default_value: float | None = None,
    ) -> tuple[str, str, str]:
        """Normalize a numeric interval table to lower/upper/value float64 arrays.

        Pandas iteration is a preparation concern, not a runtime representation.
        The normalized ABI therefore records the small mathematical policy needed
        by the source formula: lower-bound closure plus what to do when no interval
        matches.  ``last`` preserves the historical Term_US shock-lapse contract;
        ``default_or_last_above`` returns a numeric default for unmatched queries at
        or below the final upper bound and holds the final value above it.
        """
        if closure not in {"(lo,hi]", "[lo,hi]"}:
            raise FrontendError(f"unsupported interval closure {closure!r}")
        if fallback not in {"last", "default_or_last_above"}:
            raise FrontendError(f"unsupported interval fallback {fallback!r}")
        if fallback == "default_or_last_above":
            if not isinstance(default_value, (int, float, np.integer, np.floating)) or isinstance(default_value, (bool, np.bool_)):
                raise FrontendError("default_or_last_above interval fallback requires a numeric default")
            default_value = float(default_value)
            if not np.isfinite(default_value):
                raise FrontendError("interval fallback default must be finite")
        elif default_value is not None:
            default_value = float(default_value)
        token = (
            "external_dataframe_interval_lookup", ref_name, cell_name, repr(lower_column),
            repr(upper_column), repr(value_column), closure, fallback, default_value,
        )
        if token in self.memo:
            return self.memo[token]
        frame = cell()
        if not isinstance(frame, pd.DataFrame):
            raise FrontendError(f"external table {ref_name}.{cell_name} no longer returns a DataFrame")
        for col in (lower_column, upper_column, value_column):
            if col not in frame.columns:
                raise FrontendError(f"external interval lookup has no column {col!r}")
            raw = np.asarray(frame[col])
            if raw.dtype.kind not in "iufc":
                raise FrontendError(f"normalized interval lookup column {col!r} must be numeric")
        lower = np.ascontiguousarray(np.asarray(frame[lower_column], dtype=np.float64), dtype=np.float64)
        upper = np.ascontiguousarray(np.asarray(frame[upper_column], dtype=np.float64), dtype=np.float64)
        values = np.ascontiguousarray(np.asarray(frame[value_column], dtype=np.float64), dtype=np.float64)
        if lower.ndim != 1 or lower.size == 0 or upper.shape != lower.shape or values.shape != lower.shape:
            raise FrontendError("normalized interval lookup requires non-empty equal-length 1-D columns")
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(np.isfinite(values))):
            raise FrontendError("normalized interval lookup columns must be finite")
        if np.any(lower >= upper):
            raise FrontendError("normalized interval lookup requires lower < upper for every row")

        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()
        semantic = NormalizedLookup1D(
            kind="interval", source_ref=ref_name, source_cell=cell_name, source_shape=tuple(frame.shape),
            selected_column=value_column, lower_column=lower_column, upper_column=upper_column,
            lower_labels=tuple(float(x) for x in lower), upper_labels=tuple(float(x) for x in upper),
            closure=closure, fallback=fallback, default_value=default_value,
        )
        lower_key = self._key("lklo")
        upper_key = self._key("lkhi")
        value_key = self._key("lky")
        self.memo[token] = (lower_key, upper_key, value_key)

        def current(c=cell, ei=expected_index, ec=expected_columns):
            table = c()
            if not isinstance(table, pd.DataFrame):
                raise FrontendError("external interval lookup provider changed type after compilation")
            if not table.index.equals(ei) or not table.columns.equals(ec):
                raise FrontendError("external interval lookup table domain changed after compilation")
            return table

        def column_getter(keys, current=current, col=None):
            table = current()
            return np.ascontiguousarray(np.asarray(table[col], dtype=np.float64), dtype=np.float64)

        def domain_guard(current=current, elo=lower, ehi=upper, rn=ref_name, cn=cell_name):
            try:
                table = current()
                lo = np.ascontiguousarray(np.asarray(table[lower_column], dtype=np.float64), dtype=np.float64)
                hi = np.ascontiguousarray(np.asarray(table[upper_column], dtype=np.float64), dtype=np.float64)
            except Exception as exc:
                return f"lookup1d_domain:{rn}.{cn}:{exc}"
            if lo.shape != elo.shape or hi.shape != ehi.shape or not np.array_equal(lo, elo) or not np.array_equal(hi, ehi):
                return f"lookup1d_domain:{rn}.{cn}:interval_changed"
            return None

        domain_token = repr((tuple(frame.index.tolist()), tuple(frame.columns.tolist()), lower_column, upper_column, value_column))
        common = dict(
            domain_guard=domain_guard, domain_token=domain_token, expected_shape=tuple(lower.shape),
            normalized_lookup_1d=semantic,
        )
        self.specs[lower_key] = InputSpec(
            lower_key, "float64", 1, "global", lambda keys, g=column_getter, c=lower_column: g(keys, col=c),
            f"normalized interval lookup lower bounds {ref_name}.{cell_name}[{lower_column!r}]",
            lookup_1d_role="lower", **common,
        )
        self.specs[upper_key] = InputSpec(
            upper_key, "float64", 1, "global", lambda keys, g=column_getter, c=upper_column: g(keys, col=c),
            f"normalized interval lookup upper bounds {ref_name}.{cell_name}[{upper_column!r}]",
            lookup_1d_role="upper", **common,
        )
        self.specs[value_key] = InputSpec(
            value_key, "float64", 1, "global", lambda keys, g=column_getter, c=value_column: g(keys, col=c),
            f"normalized interval lookup values {ref_name}.{cell_name}[{value_column!r}]",
            lookup_1d_role="values", **common,
        )
        return lower_key, upper_key, value_key

    def point_coordinate(self, position: int) -> str:
        token=("point_coordinate", int(position))
        if token in self.memo:
            return self.memo[token]
        if position < 0 or position >= len(self.frontend.space_params):
            raise FrontendError(f"invalid ItemSpace coordinate position {position}")
        key=self._key("key")
        self.memo[token]=key
        sample=self.frontend.run_keys
        def extract(k):
            if len(self.frontend.space_params) == 1:
                return k
            if not isinstance(k, tuple) or len(k) != len(self.frontend.space_params):
                raise FrontendError(
                    f"multi-parameter ItemSpace key must be a tuple of length {len(self.frontend.space_params)}"
                )
            return k[position]
        vals=[extract(k) for k in sample]
        if not all(isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_)) for x in vals):
            raise FrontendError("direct use of a non-integral ItemSpace coordinate requires a Python fallback")
        ranked_axis = tuple(getattr(self.frontend, "point_keys", self.frontend.run_keys)) != tuple(self.frontend.run_keys)
        if ranked_axis and len(sample) != 1:
            raise FrontendError("ranked ItemSpace coordinates require one structural invocation")
        def getter(keys, pos=position, nparams=len(self.frontend.space_params)):
            # Portfolio row identifiers are not Space invocation arguments.
            # A ranked valuation shares the single bound Space coordinate
            # across every row of its independent result axis.
            if ranked_axis:
                vals = [extract(sample[0])] * len(keys)
            elif nparams == 1:
                vals=keys
            else:
                vals=[k[pos] for k in keys]
            return np.asarray(vals,dtype=np.int64)
        pname=self.frontend.space_params[position]
        self.specs[key]=InputSpec(key,"int64",1,"point",getter,f"ItemSpace coordinate {position} ({pname})")
        return key

    def point_keys(self) -> str:
        # Compatibility helper for the original one-parameter path.
        return self.point_coordinate(0)

    def global_series(self, ref_name: str, series: pd.Series) -> tuple[str, Callable[[ast.AST], ast.AST]]:
        if isinstance(series.index, pd.MultiIndex):
            return self.global_multiindex_series(ref_name, series)
        labels, start, step = _safe_numeric_axis(series.index)
        token=("series",ref_name,start,step,len(labels))
        if token in self.memo:
            key=self.memo[token]
            return key, lambda expr: _axis_index_expr(expr, start, step)
        order = np.asarray(sorted(range(len(labels)), key=lambda i: labels[i]), dtype=np.intp)
        identity_order = np.array_equal(order, np.arange(len(order), dtype=np.intp))
        key = self._key("a1")
        self.memo[token]=key
        def getter(keys, space=self.frontend.space, rn=ref_name, order=order, identity_order=identity_order):
            s = space.refs[rn]
            arr = s.to_numpy(dtype=np.float64, copy=False)
            return arr if identity_order else arr[order]
        expected_index_obj = series.index
        expected_index = tuple(series.index.tolist())
        def domain_guard(space=self.frontend.space, rn=ref_name, expected_obj=expected_index_obj):
            current = space.refs[rn]
            return None if (current.index is expected_obj or current.index.equals(expected_obj)) else f"table_domain:{rn}:index_changed"
        self.specs[key] = InputSpec(key, "float64", 1, "global", getter, f"Series Reference {ref_name}", domain_guard, repr(expected_index), (len(expected_index),))
        return key, lambda expr: _axis_index_expr(expr, start, step)

    def global_dataframe_exact_nd(
        self,
        ref_name: str,
        df: pd.DataFrame,
        *,
        fixed_axes: Mapping[int, Any],
        dynamic_axes: Sequence[NormalizedTableAxis],
    ) -> str:
        """Normalize an exact labelled DataFrame reference to numeric N-D ABI.

        Unlike ``global_dataframe`` this path does not require numeric column
        labels.  Each physical dimension is either fixed at compile time or is
        supplied through a proven finite categorical / regular numeric selector.
        The runtime sees only a dense numeric ndarray whose axis order is frozen by
        the normalized selector labels.
        """
        if not isinstance(df, pd.DataFrame):
            raise FrontendError(f"Reference {ref_name} is no longer a DataFrame")
        if df.ndim != 2:
            raise FrontendError("exact DataFrame reference normalization requires rank 2")
        axes = tuple(dynamic_axes)
        fixed = {int(k): v for k, v in fixed_axes.items()}
        if any(k not in (0, 1) for k in fixed):
            raise FrontendError("exact DataFrame reference fixed axis must be 0 or 1")
        by_level = {int(a.level): a for a in axes}
        if set(by_level).intersection(fixed):
            raise FrontendError("exact DataFrame reference axis cannot be both fixed and dynamic")
        if set(by_level).union(fixed) != {0, 1}:
            raise FrontendError("exact DataFrame reference must account for both row and column axes")

        source_domains = (tuple(df.index.tolist()), tuple(df.columns.tolist()))
        selected: list[tuple[Any, ...]] = []
        for level, domain in enumerate(source_domains):
            if level in fixed:
                label = fixed[level]
                if label not in domain:
                    raise FrontendError(
                        f"DataFrame Reference {ref_name} axis {level} has no label {label!r}"
                    )
                selected.append((label,))
            else:
                labels = tuple(by_level[level].labels)
                if not labels or any(label not in domain for label in labels):
                    raise FrontendError(
                        f"DataFrame Reference {ref_name} dynamic axis {level} is outside the frozen domain"
                    )
                selected.append(labels)

        token = (
            "dataframe_exact_nd", ref_name,
            tuple((k, repr(v)) for k, v in sorted(fixed.items())),
            tuple((a.level, a.selector_kind, tuple(map(repr, a.labels))) for a in axes),
        )
        if token in self.memo:
            return self.memo[token]

        try:
            sample = df.loc[list(selected[0]), list(selected[1])].to_numpy(copy=False)
            numeric = np.asarray(sample, dtype=np.float64)
        except Exception as exc:
            raise FrontendError(
                f"DataFrame Reference {ref_name} exact lookup values must be numeric"
            ) from exc
        if numeric.shape != (len(selected[0]), len(selected[1])):
            raise FrontendError(f"DataFrame Reference {ref_name} exact lookup shape proof failed")

        key = self._key("a2x")
        self.memo[token] = key
        expected_index_obj = df.index
        expected_columns_obj = df.columns

        def current(space=self.frontend.space, rn=ref_name, ei=expected_index_obj, ec=expected_columns_obj):
            frame = space.refs[rn]
            if not isinstance(frame, pd.DataFrame):
                raise FrontendError(f"DataFrame Reference {rn} changed type after compilation")
            if not (frame.index is ei or frame.index.equals(ei)):
                raise FrontendError(f"DataFrame Reference {rn} index changed after compilation")
            if not (frame.columns is ec or frame.columns.equals(ec)):
                raise FrontendError(f"DataFrame Reference {rn} columns changed after compilation")
            return frame

        def getter(keys, current=current, rows=selected[0], cols=selected[1]):
            frame = current()
            return np.ascontiguousarray(
                np.asarray(frame.loc[list(rows), list(cols)].to_numpy(copy=False), dtype=np.float64)
            )

        def domain_guard(current=current, rn=ref_name):
            try:
                current()
            except FrontendError as exc:
                return f"table_domain:{rn}:{exc}"
            return None

        semantic = NormalizedTableInput(
            source_ref=ref_name, source_cell="<reference>", source_shape=tuple(df.shape),
            index_names=(df.index.name, "__columns__"), selected_column=None,
            fixed_row_labels=tuple(sorted(fixed.items())), dynamic_level=None,
            dynamic_labels=(), axis_start=None, axis_step=None, result_kind="exact_nd",
            layout="dense_cartesian", dynamic_axes=axes, sparse_key_input=None,
        )
        self.specs[key] = InputSpec(
            key, "float64", 2, "global", getter,
            f"normalized exact DataFrame Reference {ref_name}",
            domain_guard=domain_guard, domain_token=repr(source_domains),
            expected_shape=(len(selected[0]), len(selected[1])), normalized_table=semantic,
        )
        return key

    def global_dataframe(self, ref_name: str, df: pd.DataFrame) -> tuple[str, Callable[[ast.AST], ast.AST], Callable[[ast.AST], ast.AST]]:
        rows, r0, rs = _safe_numeric_axis(df.index)
        cols, c0, cs = _safe_numeric_axis(df.columns)
        token=("dataframe",ref_name,r0,rs,c0,cs,len(rows),len(cols))
        if token in self.memo:
            key=self.memo[token]
            return key, lambda e: _axis_index_expr(e, r0, rs), lambda e: _axis_index_expr(e, c0, cs)
        rorder = np.asarray(sorted(range(len(rows)), key=lambda i: rows[i]), dtype=np.intp)
        corder = np.asarray(sorted(range(len(cols)), key=lambda i: cols[i]), dtype=np.intp)
        r_identity = np.array_equal(rorder, np.arange(len(rorder), dtype=np.intp))
        c_identity = np.array_equal(corder, np.arange(len(corder), dtype=np.intp))
        key = self._key("a2")
        self.memo[token]=key
        def getter(keys, space=self.frontend.space, rn=ref_name, ro=rorder, co=corder, ri=r_identity, ci=c_identity):
            frame = space.refs[rn]
            arr = frame.to_numpy(dtype=np.float64, copy=False)
            if not ri:
                arr = arr[ro, :]
            if not ci:
                arr = arr[:, co]
            return arr
        expected_index_obj = df.index; expected_columns_obj = df.columns
        expected_index = tuple(df.index.tolist()); expected_columns = tuple(df.columns.tolist())
        def domain_guard(space=self.frontend.space, rn=ref_name, er=expected_index_obj, ec=expected_columns_obj):
            current = space.refs[rn]
            if not (current.index is er or current.index.equals(er)):
                return f"table_domain:{rn}:index_changed"
            if not (current.columns is ec or current.columns.equals(ec)):
                return f"table_domain:{rn}:columns_changed"
            return None
        self.specs[key] = InputSpec(key, "float64", 2, "global", getter, f"DataFrame Reference {ref_name}", domain_guard, repr((expected_index, expected_columns)), (len(expected_index), len(expected_columns)))
        return key, lambda e: _axis_index_expr(e, r0, rs), lambda e: _axis_index_expr(e, c0, cs)

    def global_multiindex_series(self, ref_name: str, series: pd.Series) -> tuple[str, Callable[[ast.AST], ast.AST]]:
        if series.index.nlevels != 2:
            raise FrontendError("only two-level numeric MultiIndex Series are supported by the native table frontend")
        lev0 = sorted(set(series.index.get_level_values(0)))
        lev1 = sorted(set(series.index.get_level_values(1)))
        _, r0, rs = _safe_numeric_axis(lev0)
        _, c0, cs = _safe_numeric_axis(lev1)
        # Require a complete Cartesian grid so missing labels cannot silently turn into zeros.
        if len(series) != len(lev0) * len(lev1):
            raise FrontendError("sparse MultiIndex Series requires a lookup fallback; native dense conversion would change missing-key semantics")
        token=("multiindex",ref_name,r0,rs,c0,cs,len(lev0),len(lev1))
        if token in self.memo:
            key=self.memo[token]
            self.frontend.multiindex_axes[key]=(r0,rs,c0,cs)
            return key, (lambda expr: expr)
        key = self._key("a2")
        self.memo[token]=key
        desired = [(a, b) for a in lev0 for b in lev1]
        order = series.index.get_indexer(desired)
        if np.any(order < 0):
            raise FrontendError("MultiIndex Cartesian-domain proof failed")
        order = np.asarray(order, dtype=np.intp)
        identity_order = np.array_equal(order, np.arange(len(order), dtype=np.intp))
        def getter(keys, space=self.frontend.space, rn=ref_name, l0=lev0, l1=lev1, order=order, identity_order=identity_order):
            s = space.refs[rn]
            arr = s.to_numpy(dtype=np.float64, copy=False)
            if not identity_order:
                arr = arr[order]
            return arr.reshape((len(l0), len(l1)))
        expected_index_obj = series.index
        expected_index = tuple(series.index.tolist())
        def domain_guard(space=self.frontend.space, rn=ref_name, expected_obj=expected_index_obj):
            current = space.refs[rn]
            return None if (current.index is expected_obj or current.index.equals(expected_obj)) else f"table_domain:{rn}:multiindex_changed"
        self.specs[key] = InputSpec(key, "float64", 2, "global", getter, f"MultiIndex Series Reference {ref_name}", domain_guard, repr(expected_index), (len(lev0), len(lev1)))
        # Caller handles tuple indices; returned mapper is unused for 2-D.
        def mapper(expr):
            return expr
        self.frontend.multiindex_axes[key] = (r0, rs, c0, cs)
        return key, mapper

    def local_cell_multiindex_series(self, cell_name: str, cell) -> str:
        """Normalize a zero-argument local Cell returning a dense numeric 2-D Series.

        The pandas/modelx object is preparation-only.  Backends receive one dense
        numeric table input whose exact MultiIndex domain is validated whenever
        inputs are bound.
        """
        value = cell()
        if not isinstance(value, pd.Series) or not isinstance(value.index, pd.MultiIndex):
            raise FrontendError(
                f"local Cells value {cell_name!r} is not a MultiIndex Series"
            )
        if value.index.nlevels != 2:
            raise FrontendError(
                f"local Cells value {cell_name!r} requires exactly two index levels"
            )
        raw = value.to_numpy(copy=False)
        if raw.dtype.kind != "f":
            raise FrontendError(
                f"local Cells value {cell_name!r} requires float dtype for indexed-input normalization; got {raw.dtype}"
            )
        lev0 = sorted(set(value.index.get_level_values(0)))
        lev1 = sorted(set(value.index.get_level_values(1)))
        _, r0, rs = _safe_numeric_axis(lev0)
        _, c0, cs = _safe_numeric_axis(lev1)
        if len(value) != len(lev0) * len(lev1):
            raise FrontendError(
                f"local Cells value {cell_name!r} is sparse; dense indexed-input normalization would change missing-key semantics"
            )
        token=("local_cell_multiindex", cell_name, r0, rs, c0, cs, len(lev0), len(lev1))
        if token in self.memo:
            key=self.memo[token]
            self.frontend.multiindex_axes[key]=(r0,rs,c0,cs)
            return key
        desired=[(a,b) for a in lev0 for b in lev1]
        order=value.index.get_indexer(desired)
        if np.any(order < 0):
            raise FrontendError(
                f"local Cells value {cell_name!r} failed Cartesian-domain proof"
            )
        order=np.asarray(order,dtype=np.intp)
        identity=np.array_equal(order,np.arange(len(order),dtype=np.intp))
        expected_index=value.index.copy()
        expected_shape=(len(lev0),len(lev1))
        key=self._key("lc2")
        self.memo[token]=key

        def current(c=cell, ei=expected_index, cn=cell_name):
            series=c()
            if not isinstance(series,pd.Series) or not isinstance(series.index,pd.MultiIndex):
                raise FrontendError(f"local Cells indexed input {cn!r} changed type")
            if not series.index.equals(ei):
                raise FrontendError(f"local Cells indexed input {cn!r} changed domain")
            arr=series.to_numpy(copy=False)
            if arr.dtype.kind != "f":
                raise FrontendError(f"local Cells indexed input {cn!r} changed dtype")
            return arr

        def getter(keys,current=current,order=order,identity=identity,shape=expected_shape):
            arr=current()
            if not identity:
                arr=arr[order]
            return np.ascontiguousarray(arr,dtype=np.float64).reshape(shape)

        def domain_guard(current=current,cn=cell_name):
            try:
                current()
            except FrontendError as exc:
                return f"local_cell_indexed_input:{cn}:{exc}"
            return None

        self.specs[key]=InputSpec(
            key,"float64",2,"global",getter,
            f"dense MultiIndex Series from local Cells {cell_name}",
            domain_guard=domain_guard,
            domain_token=repr((tuple(expected_index.tolist()), expected_shape)),
            expected_shape=expected_shape,
        )
        self.frontend.multiindex_axes[key]=(r0,rs,c0,cs)
        return key

    def external_scalar(self, ref_name: str, cell_name: str, cell) -> str:
        """Marshal a zero-argument numeric Cells value from another Space.

        A structurally relative child Space may inherit one or more coordinates from
        the active RunDomain ItemSpace.  In that case the Cells value is a point
        input, not a global scalar.  The child/modelx object remains preparation-only;
        generated code receives only the numeric ABI value.
        """
        binding = self.frontend._hierarchical_itemspace_parameter_binding(ref_name, cell_name)
        if binding is not None:
            token=(
                "external_scalar_hierarchical", ref_name, cell_name,
                binding.owner_space_fullname, binding.child_space_fullname,
                binding.parameter_names, binding.parameter_positions,
                binding.dependency_cells,
            )
            if token in self.memo:
                return self.memo[token]
            description = (
                f"hierarchical ItemSpace-bound external Cells scalar {ref_name}.{cell_name} "
                f"via {binding.parameter_names!r}"
            )

            def evaluate(keys, b=binding, description=description):
                keys = tuple(keys)
                self.frontend._validate_hierarchical_itemspace_binding(b, keys)
                values=[]
                for run_key in keys:
                    concrete = self.frontend._context_cell_for_key(
                        b.ref_name, b.cell_name, run_key
                    )
                    if tuple(getattr(concrete, "parameters", ()) or ()):
                        raise FrontendError(
                            f"{description} changed to a parameterized Cells member"
                        )
                    try:
                        values.append(concrete())
                    except Exception as exc:
                        raise FrontendError(
                            f"{description} could not be evaluated in its owning ItemSpace: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                return values

            sample_values = evaluate(self.frontend.run_keys)
            dtype = self._numeric_dtype(sample_values, description)
            key=self._key("extp")
            self.memo[token]=key

            def getter(keys, evaluate=evaluate, dtype=dtype, description=description):
                return self._coerce_numeric_values(evaluate(keys), dtype, description)

            self.specs[key]=InputSpec(
                key, dtype, 1, "point", getter, description,
                domain_token=repr((
                    binding.owner_space_fullname, binding.child_space_fullname,
                    binding.parameter_names, binding.parameter_positions,
                )),
                hierarchical_itemspace_binding=binding,
            )
            return key

        token=("external_scalar", ref_name, cell_name)
        if token in self.memo:
            return self.memo[token]
        key=self._key("exts")
        self.memo[token]=key
        value=cell()
        if isinstance(value,(bool,np.bool_,int,np.integer)):
            dtype="int64"
        elif isinstance(value,(float,np.floating)):
            dtype="float64"
        else:
            raise FrontendError(f"external Cells scalar {ref_name}.{cell_name} has unsupported type {type(value).__name__}")
        def getter(keys, c=cell):
            return c()
        self.specs[key]=InputSpec(key,dtype,0,"global",getter,f"external Cells scalar {ref_name}.{cell_name}")
        return key

    def external_static_item(
        self, ref_name: str, cell_name: str, cell, item_key: Any
    ) -> str:
        """Extract one statically named primitive from an object/mixed container.

        Only the selected item crosses the preparation boundary.  The container is
        restricted to Mapping/Series semantics and its key/index domain is frozen so
        a later schema change cannot silently reinterpret the compiled expression.
        """
        token = ("external_static_item", ref_name, cell_name, repr(item_key))
        if token in self.memo:
            return self.memo[token]

        description = f"static item {ref_name}.{cell_name}[{item_key!r}]"

        def inspect_container(container):
            if isinstance(container, Mapping):
                domain = ("mapping", tuple(container.keys()))
            elif isinstance(container, pd.Series):
                domain = ("series", tuple(container.index.tolist()))
            else:
                raise FrontendError(
                    f"{description} requires a preparation-owned Mapping or Series; "
                    f"got {type(container).__name__}"
                )
            try:
                value = container[item_key]
            except Exception as exc:
                raise FrontendError(f"{description} source lookup failed: {type(exc).__name__}: {exc}") from exc
            dtype = self._uniform_primitive_dtype((value,), description)
            return domain, dtype, value

        initial = cell()
        expected_domain, dtype, _sample = inspect_container(initial)
        key = self._key("extitem")
        self.memo[token] = key

        def getter(keys, c=cell, expected=expected_domain, dtype=dtype):
            current = c()
            domain, current_dtype, value = inspect_container(current)
            if domain != expected:
                raise FrontendError(f"{description} container domain changed after compilation")
            if current_dtype != dtype:
                raise FrontendError(
                    f"{description} changed primitive type from {dtype} to {current_dtype}"
                )
            return self._coerce_uniform_primitive_values((value,), dtype, description)[0].item()

        def domain_guard(c=cell, expected=expected_domain):
            try:
                domain, _dtype, _value = inspect_container(c())
            except FrontendError as exc:
                return f"static_item:{ref_name}.{cell_name}:{exc}"
            if domain != expected:
                return f"static_item:{ref_name}.{cell_name}:domain_changed"
            return None

        self.specs[key] = InputSpec(
            key, dtype, 0, "global", getter, description,
            domain_guard=domain_guard,
            domain_token=repr((ref_name, cell_name, expected_domain, repr(item_key), dtype)),
            expected_shape=(),
        )
        return key

    def local_cell_mapping_coordinate_get(
        self,
        cell_name: str,
        key_expr: ast.AST,
        default_value: Any,
        coordinate_name: str,
        coordinate_lo: int,
        coordinate_hi: int,
        coordinate_owner_uid: str,
    ) -> tuple[str, str]:
        """Materialize ``cell().get(key_expr, default)`` over point x coordinate.

        The mapping and Python ``.get`` operation exist only during input binding.
        Generated valuation code sees an ordinary dense numeric table indexed by the
        existing point ordinal and the already-proven coordinate position.
        """
        coordinate_name = str(coordinate_name)
        coordinate_lo = int(coordinate_lo)
        coordinate_hi = int(coordinate_hi)
        coordinate_owner_uid = str(coordinate_owner_uid)
        key_expr = copy.deepcopy(key_expr)
        if coordinate_lo > coordinate_hi:
            raise FrontendError("finite mapping get has an empty coordinate domain")
        if not self.bound_numeric_expr_supported(key_expr, coordinate_name):
            raise FrontendError("finite mapping get key is outside the bounded native expression subset")
        if not self.bound_numeric_expr_uses_coordinate(key_expr, coordinate_name):
            raise FrontendError("finite mapping get currently requires coordinate-dependent key evidence")
        default_dtype, default_numeric = self._numeric_scalar(default_value, "finite mapping get default")
        width = coordinate_hi - coordinate_lo + 1
        if width * len(self.frontend.run_keys) > RUNTIME_SCALAR_HELPER_EXPANSION_LIMIT:
            raise FrontendError("finite mapping get materialization exceeds safety limit")

        expr_token = ast.dump(key_expr, include_attributes=False)
        token = (
            "local_cell_mapping_coordinate_get", cell_name, expr_token,
            repr(default_numeric), default_dtype, coordinate_owner_uid,
            coordinate_name, coordinate_lo, coordinate_hi,
        )
        if token in self.memo:
            return self.memo[token], self.point_ordinal()

        description = f"finite prepared mapping get from local Cells {cell_name}"

        def validate_coordinate_domain() -> None:
            mx_domain = getattr(
                self.frontend, "modelx_cython_parameter_domains", {}
            ).get((coordinate_owner_uid, coordinate_name))
            if mx_domain is not None:
                lo, hi, _source = mx_domain
                if lo is None or hi is None or int(lo) != coordinate_lo or int(hi) != coordinate_hi:
                    raise FrontendError(f"{description} modelx-cython coordinate domain changed")
                return
            tr = self.frontend.variant_trace_by_uid.get(coordinate_owner_uid)
            if tr is None or tr.key.time_pos is None:
                raise FrontendError(f"{description} coordinate owner is unavailable")
            pos = int(tr.key.time_pos)
            if pos < 0 or pos >= len(tr.schema.parameters):
                raise FrontendError(f"{description} coordinate owner has invalid parameter geometry")
            if tr.schema.parameters[pos] != coordinate_name:
                raise FrontendError(f"{description} coordinate parameter identity changed")
            current = (
                self.frontend.specialized_coordinate_domains.get(coordinate_owner_uid)
                or self.frontend.coordinate_domains.get(coordinate_owner_uid)
            )
            if current is None or int(current.lo) != coordinate_lo or int(current.hi) != coordinate_hi:
                raise FrontendError(f"{description} coordinate domain changed after compilation")

        expected_domains: dict[Any, tuple[Any, ...]] = {}

        def mapping_for(run_key):
            space = self.frontend._space_instance(run_key)
            try:
                member = getattr(space, cell_name)
                mapping = member()
            except Exception as exc:
                raise FrontendError(
                    f"{description} could not evaluate its source mapping: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(mapping, Mapping):
                raise FrontendError(
                    f"{description} requires Mapping result, got {type(mapping).__name__}"
                )
            domain: list[Any] = []
            for raw_key in mapping.keys():
                if isinstance(raw_key, np.generic):
                    raw_key = raw_key.item()
                if isinstance(raw_key, bool) or not isinstance(raw_key, (int, np.integer)):
                    raise FrontendError(
                        f"{description} mapping key {raw_key!r} is not a native integer key"
                    )
                domain.append(int(raw_key))
            if len(set(domain)) != len(domain):
                raise FrontendError(f"{description} mapping key normalization is not unique")
            return mapping, tuple(sorted(domain))

        def evaluate(keys, *, establish: bool):
            validate_coordinate_domain()
            rows: list[list[Any]] = []
            domains: list[tuple[Any, ...]] = []
            all_source_values: list[Any] = [default_numeric]
            for run_key in keys:
                mapping, domain = mapping_for(run_key)
                domains.append(domain)
                all_source_values.extend(mapping.values())
                row: list[Any] = []
                for coordinate in range(coordinate_lo, coordinate_hi + 1):
                    actual_key = self._bound_numeric_expr_value(
                        key_expr, run_key, (coordinate_name, coordinate)
                    )
                    if isinstance(actual_key, np.generic):
                        actual_key = actual_key.item()
                    if isinstance(actual_key, bool) or not isinstance(actual_key, (int, np.integer)):
                        raise FrontendError(
                            f"{description} runtime key {actual_key!r} is not a proven integer"
                        )
                    value = mapping.get(int(actual_key), default_numeric)
                    self._numeric_scalar(value, description)
                    row.append(value)
                rows.append(row)
            dtype = self._numeric_dtype(all_source_values, description)
            if establish:
                expected_domains.clear()
                expected_domains.update(zip(keys, domains))
            else:
                for run_key, domain in zip(keys, domains):
                    if run_key not in expected_domains or expected_domains[run_key] != domain:
                        raise FrontendError(f"{description} mapping domain changed after compilation")
            # Source default/value types may promote only through the ordinary
            # numeric ABI; strings/None/object values are rejected above.
            flat = [value for row in rows for value in row]
            return rows, dtype, flat

        sample_rows, dtype, _flat = evaluate(self.frontend.run_keys, establish=True)
        # Recheck the declared default against the chosen numeric ABI.  Bool/int
        # widening is allowed by the existing numeric ABI; nonnumeric values never
        # reach this point.
        if default_dtype == "float64" and dtype != "float64":
            raise FrontendError(f"{description} default/value dtype contract is inconsistent")
        table_key = self._key("mapget")
        self.memo[token] = table_key

        def getter(keys, dtype=dtype, width=width):
            rows, current_dtype, flat = evaluate(keys, establish=False)
            if current_dtype != dtype:
                raise FrontendError(
                    f"{description} value dtype changed from {dtype} to {current_dtype}"
                )
            arr = self._coerce_numeric_values(flat, dtype, description)
            return np.ascontiguousarray(arr.reshape((len(rows), width)))

        self.specs[table_key] = InputSpec(
            table_key, dtype, 2, "global", getter, description,
            domain_token=repr((
                cell_name, expr_token, coordinate_owner_uid, coordinate_name,
                coordinate_lo, coordinate_hi, tuple((repr(k), v) for k, v in expected_domains.items()), repr(default_numeric), dtype,
            )),
            expected_shape=(len(self.frontend.run_keys), width),
        )
        return table_key, self.point_ordinal()

    def local_cell_mapping_ragged(
        self, cell_name: str
    ) -> tuple[str, str, str, str]:
        """Prepare an ordered zero-argument local Mapping as native CSR arrays.

        Python Mapping iteration order is semantic, so preparation preserves the
        exact insertion order for every RunDomain point.  The generated runtime
        sees only ``offsets``, integer ``keys`` and primitive numeric ``values``;
        no Mapping/Python object survives into valuation.
        """
        token = ("local_cell_mapping_ragged", str(cell_name))
        if token in self.mapping_ragged_memo:
            return self.mapping_ragged_memo[token]
        description = f"ordered prepared mapping traversal from local Cells {cell_name}"
        expected_domains: dict[Any, tuple[int, ...]] = {}

        def mapping_for(run_key):
            space = self.frontend._space_instance(run_key)
            try:
                member = getattr(space, cell_name)
                mapping = member()
            except Exception as exc:
                raise FrontendError(
                    f"{description} could not evaluate its source mapping: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(mapping, Mapping):
                raise FrontendError(
                    f"{description} requires Mapping result, got {type(mapping).__name__}"
                )
            keys: list[int] = []
            values: list[Any] = []
            for raw_key, value in mapping.items():
                if isinstance(raw_key, np.generic):
                    raw_key = raw_key.item()
                if isinstance(raw_key, bool) or not isinstance(raw_key, (int, np.integer)):
                    raise FrontendError(
                        f"{description} mapping key {raw_key!r} is not a native integer key"
                    )
                keys.append(int(raw_key))
                self._numeric_scalar(value, description)
                values.append(value)
            if len(set(keys)) != len(keys):
                raise FrontendError(f"{description} mapping key normalization is not unique")
            return tuple(keys), tuple(values)

        def materialize(keys, *, establish: bool):
            selected = tuple(keys)
            offsets = [0]
            flat_keys: list[int] = []
            flat_values: list[Any] = []
            domains: list[tuple[int, ...]] = []
            for run_key in selected:
                domain, values = mapping_for(run_key)
                domains.append(domain)
                flat_keys.extend(domain)
                flat_values.extend(values)
                offsets.append(len(flat_keys))
            if establish:
                expected_domains.clear()
                expected_domains.update(zip(selected, domains))
            else:
                for run_key, domain in zip(selected, domains):
                    if run_key not in expected_domains or expected_domains[run_key] != domain:
                        raise FrontendError(
                            f"{description} mapping key/order domain changed after compilation"
                        )
            # An entirely empty mapping domain has no primitive value type to
            # prove. Fail closed instead of inventing a dtype from absence.
            if not flat_values:
                raise FrontendError(f"{description} has no values over the complete proof domain")
            dtype = self._numeric_dtype(flat_values, description)
            return (
                np.asarray(offsets, dtype=np.int64),
                np.asarray(flat_keys, dtype=np.int64),
                self._coerce_numeric_values(flat_values, dtype, description),
                dtype,
            )

        sample_offsets, sample_keys, sample_values, dtype = materialize(
            self.frontend.run_keys, establish=True
        )
        if len(sample_keys) > RUNTIME_SCALAR_HELPER_EXPANSION_LIMIT:
            raise FrontendError(f"{description} materialization exceeds safety limit")

        offsets_key = self._key("mapoff")
        keys_key = self._key("mapkey")
        values_key = self._key("mapval")
        ordinal_key = self.point_ordinal()
        self.mapping_ragged_memo[token] = (
            offsets_key, keys_key, values_key, ordinal_key
        )
        cache_token = ("prepared_mapping_ragged", str(cell_name))

        def packed(selected, batch_cache):
            cache_key = (cache_token, tuple(selected))
            if cache_key not in batch_cache:
                current = materialize(selected, establish=False)
                if current[3] != dtype:
                    raise FrontendError(
                        f"{description} value dtype changed from {dtype} to {current[3]}"
                    )
                batch_cache[cache_key] = current[:3]
            return batch_cache[cache_key]

        def make_fast_getter(index: int):
            def fast_getter(selected, batch_cache):
                return packed(selected, batch_cache)[index]
            return fast_getter

        domain_token = repr((
            cell_name,
            tuple((repr(k), v) for k, v in expected_domains.items()),
            dtype,
            "ordered_csr",
        ))
        total = int(sample_values.shape[0])
        self.specs[offsets_key] = InputSpec(
            offsets_key, "int64", 1, "global", lambda selected: materialize(selected, establish=False)[0],
            description + " offsets", domain_token=domain_token,
            expected_shape=(len(self.frontend.run_keys) + 1,),
            fast_getter=make_fast_getter(0),
        )
        self.specs[keys_key] = InputSpec(
            keys_key, "int64", 1, "global", lambda selected: materialize(selected, establish=False)[1],
            description + " keys", domain_token=domain_token,
            expected_shape=(total,), fast_getter=make_fast_getter(1),
        )
        self.specs[values_key] = InputSpec(
            values_key, dtype, 1, "global", lambda selected: materialize(selected, establish=False)[2],
            description + " values", domain_token=domain_token,
            expected_shape=(total,), fast_getter=make_fast_getter(2),
        )
        return offsets_key, keys_key, values_key, ordinal_key

    def external_array(self, ref_name: str, cell_name: str, cell, point_position: int | None) -> str:
        """Marshal an external numeric Cells array.

        ``point_position`` identifies the ItemSpace coordinate used to index the
        array.  It is deliberately positional/name-agnostic: a multi-parameter
        Space may use any coordinate as the independent array index.  ``None``
        means the complete array is a global kernel input.
        """
        point_indexed = point_position is not None
        token=("external",ref_name,cell_name,point_position)
        if token in self.memo: return self.memo[token]
        key = self._key("extp" if point_indexed else "ext")
        self.memo[token]=key
        value = np.asarray(cell())
        if value.dtype.kind not in "iufb" or value.ndim not in (1,2):
            raise FrontendError(f"external Cells array {ref_name}.{cell_name} has unsupported dtype/shape {value.dtype} {value.shape}")
        dtype = "int64" if value.dtype.kind in "iub" else "float64"
        fast_getter = None
        if point_indexed:
            run_keys = self.frontend.run_keys
            pos = int(point_position)
            nparams = len(self.frontend.space_params)
            def coordinate(keys, pos=pos, nparams=nparams):
                if nparams == 1:
                    return np.asarray(keys, dtype=np.int64)
                return np.asarray([k[pos] if isinstance(k, tuple) else k for k in keys], dtype=np.int64)
            run_idx = coordinate(run_keys)
            run_slice = None
            if len(run_idx):
                start = int(run_idx[0])
                if np.array_equal(run_idx, np.arange(start, start + len(run_idx), dtype=np.int64)):
                    run_slice = slice(start, start + len(run_idx))
            def getter(keys, c=cell, coordinate=coordinate):
                arr = np.asarray(c())
                idx = coordinate(keys)
                return arr[idx]
            def fast_getter(keys, cache, c=cell, run_keys=run_keys, run_idx=run_idx, run_slice=run_slice, coordinate=coordinate):
                arr = np.asarray(c())
                if keys is run_keys or keys == run_keys:
                    selector = run_slice if run_slice is not None else run_idx
                else:
                    idx = coordinate(keys)
                    selector = idx
                    if len(idx):
                        start = int(idx[0])
                        if np.array_equal(idx, np.arange(start, start + len(idx), dtype=np.int64)):
                            selector = slice(start, start + len(idx))
                return arr[selector]
            ndim = max(1, value.ndim - 0)
            scope = "point"
        else:
            def getter(keys, c=cell):
                return np.asarray(c())
            ndim = value.ndim
            scope = "global"
        self.specs[key] = InputSpec(
            key, dtype, ndim, scope, getter, f"external Cells array {ref_name}.{cell_name}",
            expected_shape=(tuple(value.shape) if scope == "global" else None),
            domain_token=(repr(tuple(value.shape)) if scope == "global" else None),
            fast_getter=fast_getter,
        )
        return key

    def _bound_numeric_expr_value(
        self,
        node: ast.AST,
        run_key: Any,
        coordinate_binding: tuple[str, int] | None = None,
    ) -> Any:
        """Evaluate one already-lowered ABI scalar expression for one RunDomain key.

        This is intentionally *not* a Python/source evaluator.  It accepts only
        primitive constants, arithmetic, and scalar values already owned by the
        existing ``global_input``/``point_input`` ABI.  V02387 optionally admits one
        explicitly supplied integer coordinate name/value pair.  Parameterized-Space
        materialization therefore composes existing input bindings instead of
        introducing a second model-expression interpreter or discovering coordinates
        from arbitrary source names.
        """
        if isinstance(node, ast.Constant):
            ok, value = self.frontend._context_primitive(node.value)
            if not ok:
                raise FrontendError("bound numeric expression contains a non-primitive constant")
            return value
        if isinstance(node, ast.Name):
            if (
                coordinate_binding is not None
                and isinstance(node.ctx, ast.Load)
                and node.id == coordinate_binding[0]
            ):
                return int(coordinate_binding[1])
            raise FrontendError(
                f"bound numeric expression contains unsupported runtime name {node.id!r}"
            )
        if isinstance(node, ast.UnaryOp):
            value = self._bound_numeric_expr_value(node.operand, run_key, coordinate_binding)
            try:
                if isinstance(node.op, ast.UAdd): return +value
                if isinstance(node.op, ast.USub): return -value
            except Exception as exc:
                raise FrontendError("bound numeric unary expression could not be evaluated") from exc
            raise FrontendError("bound numeric expression has an unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = self._bound_numeric_expr_value(node.left, run_key, coordinate_binding)
            right = self._bound_numeric_expr_value(node.right, run_key, coordinate_binding)
            try:
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
                if isinstance(node.op, ast.Div): return left / right
                if isinstance(node.op, ast.FloorDiv): return left // right
                if isinstance(node.op, ast.Mod): return left % right
                if isinstance(node.op, ast.Pow): return left ** right
            except Exception as exc:
                raise FrontendError("bound numeric binary expression could not be evaluated") from exc
            raise FrontendError("bound numeric expression has an unsupported binary operator")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in {"global_input", "point_input"} and len(node.args) == 1:
                key_node = node.args[0]
                if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                    raise FrontendError(f"{name} requires a literal registered input key")
                key = key_node.value
                spec = self.specs.get(key)
                if spec is None:
                    raise FrontendError(f"bound numeric expression references unknown input {key!r}")
                if name == "global_input":
                    if spec.scope != "global" or spec.ndim != 0:
                        raise FrontendError(f"input {key!r} is not a global scalar")
                    value = spec.getter((run_key,))
                else:
                    if spec.scope != "point" or spec.ndim != 1:
                        raise FrontendError(f"input {key!r} is not a point scalar")
                    # Point ordinal is relative to a bound batch, not an intrinsic
                    # semantic point value.  It cannot be replayed one key at a time.
                    if str(spec.domain_token or "").startswith("point_ordinal:"):
                        raise FrontendError("point ordinal is not a materializable semantic argument")
                    arr = np.asarray(spec.getter((run_key,)))
                    if arr.ndim != 1 or len(arr) != 1:
                        raise FrontendError(f"point input {key!r} did not bind one scalar")
                    value = arr[0]
                if isinstance(value, np.generic):
                    value = value.item()
                ok, value = self.frontend._context_primitive(value)
                if not ok:
                    raise FrontendError(f"input {key!r} did not bind a primitive scalar")
                return value
            if name in {"int", "float", "bool", "abs", "min", "max"}:
                values = [
                    self._bound_numeric_expr_value(arg, run_key, coordinate_binding)
                    for arg in node.args
                ]
                fn = {"int": int, "float": float, "bool": bool, "abs": abs, "min": min, "max": max}[name]
                try:
                    return fn(*values)
                except Exception as exc:
                    raise FrontendError(f"bound numeric intrinsic {name} failed") from exc
        raise FrontendError(
            f"unsupported bound numeric ABI expression: {ast.dump(node, include_attributes=False)}"
        )

    def bound_numeric_expr_supported(
        self, node: ast.AST, coordinate_name: str | None = None
    ) -> bool:
        """Structural check for the bounded ABI-expression subset above.

        ``coordinate_name`` is deliberately explicit.  A random source ``Name`` never
        becomes materializable merely because a caller happened to have an integer for
        it at trace time.
        """
        if isinstance(node, ast.Constant):
            return self.frontend._context_primitive(node.value)[0]
        if isinstance(node, ast.Name):
            return (
                coordinate_name is not None
                and isinstance(node.ctx, ast.Load)
                and node.id == coordinate_name
            )
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.UAdd, ast.USub)) and self.bound_numeric_expr_supported(
                node.operand, coordinate_name
            )
        if isinstance(node, ast.BinOp):
            return isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
            ) and self.bound_numeric_expr_supported(
                node.left, coordinate_name
            ) and self.bound_numeric_expr_supported(node.right, coordinate_name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id in {"global_input", "point_input"}:
                return (
                    len(node.args) == 1
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value in self.specs
                )
            if node.func.id in {"int", "float", "bool", "abs", "min", "max"}:
                return bool(node.args) and all(
                    self.bound_numeric_expr_supported(arg, coordinate_name) for arg in node.args
                )
        return False

    @staticmethod
    def bound_numeric_expr_uses_coordinate(node: ast.AST, coordinate_name: str) -> bool:
        return any(
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id == coordinate_name
            for child in ast.walk(node)
        )

    def context_static_attribute(self, ref_name: str, attr_name: str) -> str:
        """Bind one primitive model-ref attribute without baking its observed value.

        The attribute is re-resolved on every ``bind_inputs`` call and must remain a
        point-independent numeric primitive with a stable dtype.
        """
        token = ("context_static_attribute", ref_name, attr_name)
        if token in self.memo:
            return self.memo[token]
        node = ast.Attribute(ast.Name(ref_name, ast.Load()), attr_name, ast.Load())
        description = f"model-bound primitive attribute {ref_name}.{attr_name}"

        def evaluate(keys):
            values: list[Any] = []
            for run_key in keys:
                ok, value = self.frontend._context_static_value(node, run_key)
                if not ok:
                    raise FrontendError(f"{description} is no longer a primitive resolvable attribute")
                self._numeric_scalar(value, description)
                values.append(value)
            if not values:
                raise FrontendError(f"{description} has an empty bound RunDomain")
            first = values[0]
            if not all(type(value) is type(first) and value == first for value in values[1:]):
                raise FrontendError(f"{description} is not point-independent over the bound RunDomain")
            return values

        sample = evaluate(self.frontend.run_keys)
        dtype = self._numeric_dtype(sample, description)
        key = self._key("ctxattr")
        self.memo[token] = key

        def getter(keys, evaluate=evaluate, dtype=dtype, description=description):
            values = evaluate(keys)
            arr = self._coerce_numeric_values(values, dtype, description)
            if np.any(arr != arr[0]):
                raise FrontendError(f"{description} changed across the bound RunDomain")
            return arr[0].item() if hasattr(arr[0], "item") else arr[0]

        self.specs[key] = InputSpec(
            key, dtype, 0, "global", getter, description,
            expected_shape=(),
            domain_token=repr((ref_name, attr_name, "primitive_static_attribute")),
        )
        return key

    def parameterized_space_method_point(
        self,
        ref_name: str,
        selector_nodes: Sequence[ast.AST],
        member_name: str,
        argument_nodes: Sequence[ast.AST],
    ) -> str:
        """Materialize ``Space[selectors].Cells(args...)`` over the frozen RunDomain.

        The selected ItemSpace and Cells objects exist only during preparation.
        Generated code receives one numeric point input for the semantic call.
        """
        selector_nodes = tuple(copy.deepcopy(node) for node in selector_nodes)
        argument_nodes = tuple(copy.deepcopy(node) for node in argument_nodes)
        selector_token = tuple(ast.dump(node, include_attributes=False) for node in selector_nodes)
        argument_token = tuple(ast.dump(node, include_attributes=False) for node in argument_nodes)
        token = (
            "parameterized_space_method_point", ref_name, selector_token, member_name, argument_token,
        )
        if token in self.memo:
            return self.memo[token]
        if not selector_nodes:
            raise FrontendError("parameterized Space selection requires at least one selector")
        if not all(self.bound_numeric_expr_supported(node) for node in selector_nodes):
            raise FrontendError(f"parameterized Space {ref_name} has an unsupported selector expression")
        if not all(self.bound_numeric_expr_supported(node) for node in argument_nodes):
            raise FrontendError(
                f"parameterized Space method {ref_name}[...].{member_name} has a non-point-scalar argument"
            )
        description = f"model-bound parameterized Space numeric method {ref_name}[...].{member_name}"

        def evaluate(keys):
            values: list[Any] = []
            for run_key in keys:
                selectors = [self._bound_numeric_expr_value(node, run_key) for node in selector_nodes]
                item = self.frontend._context_selected_itemspace(ref_name, selectors, run_key)
                member = self.frontend._context_itemspace_cell(item, ref_name, member_name)
                args = [self._bound_numeric_expr_value(node, run_key) for node in argument_nodes]
                try:
                    value = member(*args)
                except Exception as exc:
                    raise FrontendError(
                        f"{description} could not be materialized: {type(exc).__name__}: {exc}"
                    ) from exc
                self._numeric_scalar(value, description)
                values.append(value)
            return values

        sample = evaluate(self.frontend.run_keys)
        dtype = self._numeric_dtype(sample, description)
        key = self._key("ctxmethod")
        self.memo[token] = key

        def getter(keys, evaluate=evaluate, dtype=dtype, description=description):
            return self._coerce_numeric_values(evaluate(keys), dtype, description)

        self.specs[key] = InputSpec(
            key, dtype, 1, "point", getter, description,
            domain_token=repr((ref_name, selector_token, member_name, argument_token)),
        )
        return key

    def parameterized_space_method_coordinate(
        self,
        ref_name: str,
        selector_nodes: Sequence[ast.AST],
        member_name: str,
        argument_nodes: Sequence[ast.AST],
        coordinate_name: str,
        coordinate_lo: int,
        coordinate_hi: int,
        coordinate_owner_uid: str,
        coordinate_step: int = 1,
    ) -> tuple[str, str]:
        """Materialize one selected-Cells call over point x exact coordinate.

        Rows are frozen RunDomain points.  Columns are the exact finite integer
        coordinate domain already proven for ``coordinate_owner_uid``.  The selected
        ItemSpace and Cells objects remain preparation-only; generated code uses the
        ordinary two-dimensional ``table_input`` ABI plus ``point_ordinal``.
        """
        selector_nodes = tuple(copy.deepcopy(node) for node in selector_nodes)
        argument_nodes = tuple(copy.deepcopy(node) for node in argument_nodes)
        coordinate_name = str(coordinate_name)
        coordinate_lo = int(coordinate_lo)
        coordinate_hi = int(coordinate_hi)
        coordinate_step = int(coordinate_step)
        coordinate_owner_uid = str(coordinate_owner_uid)

        if not selector_nodes:
            raise FrontendError("parameterized Space selection requires at least one selector")
        if coordinate_step != 1:
            raise FrontendError("parameterized Space coordinate materialization requires unit step")
        if coordinate_lo > coordinate_hi:
            raise FrontendError("parameterized Space coordinate materialization has an empty domain")
        if not all(self.bound_numeric_expr_supported(node) for node in selector_nodes):
            raise FrontendError(f"parameterized Space {ref_name} has an unsupported selector expression")
        if not argument_nodes or not all(
            self.bound_numeric_expr_supported(node, coordinate_name) for node in argument_nodes
        ):
            raise FrontendError(
                f"parameterized Space method {ref_name}[...].{member_name} has an unsupported coordinate argument"
            )
        if not any(
            self.bound_numeric_expr_uses_coordinate(node, coordinate_name)
            for node in argument_nodes
        ):
            raise FrontendError(
                f"parameterized Space method {ref_name}[...].{member_name} is not coordinate dependent"
            )

        width = ((coordinate_hi - coordinate_lo) // coordinate_step) + 1
        if width * len(self.frontend.run_keys) > RUNTIME_SCALAR_HELPER_EXPANSION_LIMIT:
            raise FrontendError(
                f"parameterized Space method {ref_name}[...].{member_name} materialization exceeds safety limit"
            )

        selector_token = tuple(ast.dump(node, include_attributes=False) for node in selector_nodes)
        argument_token = tuple(ast.dump(node, include_attributes=False) for node in argument_nodes)
        coordinate_domain_identity = (
            coordinate_owner_uid, coordinate_name, coordinate_lo, coordinate_hi, coordinate_step,
            "exact_rank1_coordinate_v1",
        )
        token = (
            "parameterized_space_method_coordinate",
            ref_name,
            selector_token,
            member_name,
            argument_token,
            coordinate_domain_identity,
        )
        if token in self.memo:
            return self.memo[token], self.point_ordinal()

        description = (
            f"model-bound parameterized Space numeric coordinate method "
            f"{ref_name}[...].{member_name}"
        )

        def validate_coordinate_domain() -> None:
            tr = self.frontend.variant_trace_by_uid.get(coordinate_owner_uid)
            if tr is None or tr.key.time_pos is None:
                raise FrontendError(f"{description} coordinate owner is no longer available")
            pos = int(tr.key.time_pos)
            if pos < 0 or pos >= len(tr.schema.parameters):
                raise FrontendError(f"{description} coordinate owner has invalid parameter geometry")
            if tr.schema.parameters[pos] != coordinate_name:
                raise FrontendError(f"{description} coordinate parameter identity changed")
            current = (
                self.frontend.specialized_coordinate_domains.get(coordinate_owner_uid)
                or self.frontend.coordinate_domains.get(coordinate_owner_uid)
            )
            if current is None:
                raise FrontendError(f"{description} coordinate domain is no longer proven")
            if int(current.lo) != coordinate_lo or int(current.hi) != coordinate_hi:
                raise FrontendError(f"{description} coordinate domain changed after compilation")

        def evaluate(keys):
            validate_coordinate_domain()
            rows: list[list[Any]] = []
            for run_key in keys:
                selectors = [
                    self._bound_numeric_expr_value(node, run_key) for node in selector_nodes
                ]
                item = self.frontend._context_selected_itemspace(ref_name, selectors, run_key)
                member = self.frontend._context_itemspace_cell(item, ref_name, member_name)
                row: list[Any] = []
                for coordinate in range(coordinate_lo, coordinate_hi + 1, coordinate_step):
                    binding = (coordinate_name, int(coordinate))
                    args = [
                        self._bound_numeric_expr_value(node, run_key, binding)
                        for node in argument_nodes
                    ]
                    try:
                        value = member(*args)
                    except Exception as exc:
                        raise FrontendError(
                            f"{description} could not be materialized at {coordinate_name}={coordinate}: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    self._numeric_scalar(value, description)
                    row.append(value)
                rows.append(row)
            return rows

        sample_rows = evaluate(self.frontend.run_keys)
        flat_sample = [value for row in sample_rows for value in row]
        dtype = self._numeric_dtype(flat_sample, description)
        table_key = self._key("ctxmethodcurve")
        self.memo[token] = table_key

        def getter(keys, evaluate=evaluate, dtype=dtype, description=description, width=width):
            rows = evaluate(keys)
            if any(len(row) != width for row in rows):
                raise FrontendError(f"{description} coordinate table width changed")
            flat = [value for row in rows for value in row]
            arr = self._coerce_numeric_values(flat, dtype, description)
            return np.ascontiguousarray(arr.reshape((len(rows), width)))

        self.specs[table_key] = InputSpec(
            table_key,
            dtype,
            2,
            "global",
            getter,
            description,
            domain_token=repr(
                (ref_name, selector_token, member_name, argument_token, coordinate_domain_identity)
            ),
            expected_shape=(len(self.frontend.run_keys), width),
        )
        return table_key, self.point_ordinal()

    def context_scalar_accessor(
        self, ref_name: str, cell_name: str, argument_nodes: Sequence[ast.AST]
    ) -> str:
        """Materialize one direct model-bound Cells call over point/global ABI scalars.

        V02390 deliberately reuses the bounded numeric ABI-expression evaluator used
        by parameterized-Space methods.  The context/Cells object semantics remain
        unchanged: ``_context_cell_for_key`` is still the sole concrete-member
        authority, while only numeric argument binding is generalized.
        """
        arg_nodes = tuple(copy.deepcopy(node) for node in argument_nodes)
        arg_token = tuple(ast.dump(node, include_attributes=False) for node in arg_nodes)
        token = ("context_scalar_accessor", ref_name, cell_name, arg_token)
        if token in self.memo:
            return self.memo[token]
        if not all(self.bound_numeric_expr_supported(node) for node in arg_nodes):
            raise FrontendError(
                f"model-bound context accessor {ref_name}.{cell_name} has an unsupported bounded argument"
            )

        description = f"model-bound numeric scalar accessor {ref_name}.{cell_name}"

        def evaluate(keys):
            values: list[Any] = []
            for run_key in keys:
                cell = self.frontend._context_cell_for_key(ref_name, cell_name, run_key)
                args = [self._bound_numeric_expr_value(node, run_key) for node in arg_nodes]
                try:
                    value = cell(*args)
                except Exception as exc:
                    raise FrontendError(
                        f"{description} could not be materialized: {type(exc).__name__}: {exc}"
                    ) from exc
                self._numeric_scalar(value, description)
                values.append(value)
            return values

        sample = evaluate(self.frontend.run_keys)
        dtype = self._numeric_dtype(sample, description)
        normalized_sample = self._coerce_numeric_values(sample, dtype, description)
        stable = bool(
            len(normalized_sample)
            and all(
                type(sample[i]) is type(sample[0]) and sample[i] == sample[0]
                for i in range(1, len(sample))
            )
        )
        key = self._key("ctxs" if stable else "ctxp")
        self.memo[token] = key
        if stable:
            def getter(keys, evaluate=evaluate, dtype=dtype, description=description):
                values = evaluate(keys)
                arr = self._coerce_numeric_values(values, dtype, description)
                if not len(arr):
                    raise FrontendError(f"{description} has an empty bound RunDomain")
                if np.any(arr != arr[0]):
                    raise FrontendError(
                        f"{description} is no longer point-independent over the bound RunDomain"
                    )
                return arr[0].item() if hasattr(arr[0], "item") else arr[0]
            self.specs[key] = InputSpec(
                key, dtype, 0, "global", getter, description,
                domain_token=repr((ref_name, cell_name, arg_token, "stable")),
                expected_shape=(),
            )
        else:
            def getter(keys, evaluate=evaluate, dtype=dtype, description=description):
                return self._coerce_numeric_values(evaluate(keys), dtype, description)
            self.specs[key] = InputSpec(
                key, dtype, 1, "point", getter, description,
                domain_token=repr((ref_name, cell_name, arg_token, "point")),
            )
        return key

    def context_point_accessor(
        self,
        ref_name: str,
        cell_name: str,
        argument_nodes: Sequence[ast.AST],
        point_position: int,
    ) -> str:
        """Materialize ``ref.member(static...)[point_coordinate]`` as point input."""
        arg_nodes = tuple(copy.deepcopy(node) for node in argument_nodes)
        arg_token = tuple(ast.dump(node, include_attributes=False) for node in arg_nodes)
        point_position = int(point_position)
        if point_position < 0 or point_position >= len(self.frontend.space_params):
            raise FrontendError("model-bound point accessor has an invalid point coordinate position")
        token = (
            "context_point_accessor", ref_name, cell_name, arg_token, point_position,
        )
        if token in self.memo:
            return self.memo[token]
        for node in arg_nodes:
            if not self.frontend._context_static_argument_is_fixed(node):
                raise FrontendError(
                    f"model-bound context accessor {ref_name}.{cell_name} has a dynamic argument"
                )
        description = f"model-bound numeric point accessor {ref_name}.{cell_name}"

        def selector(run_key):
            if len(self.frontend.space_params) == 1:
                return run_key
            if not isinstance(run_key, tuple) or len(run_key) != len(self.frontend.space_params):
                raise FrontendError(
                    f"{description} requires a tuple RunDomain key of length {len(self.frontend.space_params)}"
                )
            return run_key[point_position]

        def evaluate(keys):
            values: list[Any] = []
            for run_key in keys:
                cell = self.frontend._context_cell_for_key(ref_name, cell_name, run_key)
                args: list[Any] = []
                for node in arg_nodes:
                    ok, value = self.frontend._context_static_value(node, run_key)
                    if not ok:
                        raise FrontendError(
                            f"{description} static argument is no longer resolvable"
                        )
                    args.append(value)
                try:
                    container = cell(*args)
                    value = container[selector(run_key)]
                except Exception as exc:
                    raise FrontendError(
                        f"{description} could not be indexed/materialized: {type(exc).__name__}: {exc}"
                    ) from exc
                self._numeric_scalar(value, description)
                values.append(value)
            return values

        sample = evaluate(self.frontend.run_keys)
        dtype = self._numeric_dtype(sample, description)
        key = self._key("ctxp")
        self.memo[token] = key

        def getter(keys, evaluate=evaluate, dtype=dtype, description=description):
            return self._coerce_numeric_values(evaluate(keys), dtype, description)

        self.specs[key] = InputSpec(
            key, dtype, 1, "point", getter, description,
            domain_token=repr((ref_name, cell_name, arg_token, point_position)),
        )
        return key

    def point_ordinal(self) -> str:
        """Return the existing point-input ABI representation of row ordinal."""
        token = ("point_ordinal",)
        if token in self.memo:
            return self.memo[token]
        key = self._key("ctxord")
        self.memo[token] = key
        self.specs[key] = InputSpec(
            key, "int64", 1, "point",
            lambda keys: np.arange(len(keys), dtype=np.int64),
            "bound RunDomain point ordinal for model-bound coordinate input",
            domain_token=f"point_ordinal:{len(self.frontend.run_keys)}",
        )
        return key

    def context_coordinate_accessor(
        self,
        ref_name: str,
        cell_name: str,
        argument_nodes: Sequence[ast.AST],
        coordinate_name: str,
        coordinate_lo: int,
        coordinate_hi: int,
        coordinate_owner_uid: str,
        coordinate_step: int = 1,
    ) -> tuple[str, str]:
        """Materialize one direct context Cells call over point x exact coordinate.

        Every method argument is evaluated by the same bounded ABI-expression
        evaluator as V02387 selected-Space methods.  The exact coordinate domain
        remains owned by the canonical coordinate variant and is revalidated at
        bind time; generated code sees only the ordinary table/point ABI.
        """
        arg_nodes = tuple(copy.deepcopy(node) for node in argument_nodes)
        coordinate_name = str(coordinate_name)
        coordinate_lo = int(coordinate_lo)
        coordinate_hi = int(coordinate_hi)
        coordinate_owner_uid = str(coordinate_owner_uid)
        coordinate_step = int(coordinate_step)
        if coordinate_step != 1:
            raise FrontendError("model-bound coordinate accessor requires unit step")
        if coordinate_lo > coordinate_hi:
            raise FrontendError("model-bound coordinate accessor has an empty coordinate domain")
        if not arg_nodes or not all(
            self.bound_numeric_expr_supported(node, coordinate_name) for node in arg_nodes
        ):
            raise FrontendError(
                f"model-bound coordinate accessor {ref_name}.{cell_name} has an unsupported bounded argument"
            )
        if not any(
            self.bound_numeric_expr_uses_coordinate(node, coordinate_name) for node in arg_nodes
        ):
            raise FrontendError(
                f"model-bound coordinate accessor {ref_name}.{cell_name} is not coordinate dependent"
            )

        width = ((coordinate_hi - coordinate_lo) // coordinate_step) + 1
        if width * len(self.frontend.run_keys) > RUNTIME_SCALAR_HELPER_EXPANSION_LIMIT:
            raise FrontendError(
                f"model-bound coordinate accessor {ref_name}.{cell_name} materialization exceeds safety limit"
            )

        arg_token = tuple(ast.dump(node, include_attributes=False) for node in arg_nodes)
        coordinate_domain_identity = (
            coordinate_owner_uid, coordinate_name, coordinate_lo, coordinate_hi, coordinate_step,
            "exact_rank1_coordinate_v1",
        )
        token = (
            "context_coordinate_accessor", ref_name, cell_name, arg_token,
            coordinate_domain_identity,
        )
        if token in self.memo:
            return self.memo[token], self.point_ordinal()

        description = f"model-bound numeric coordinate accessor {ref_name}.{cell_name}"

        def validate_coordinate_domain() -> None:
            tr = self.frontend.variant_trace_by_uid.get(coordinate_owner_uid)
            if tr is None or tr.key.time_pos is None:
                raise FrontendError(f"{description} coordinate owner is no longer available")
            pos = int(tr.key.time_pos)
            if pos < 0 or pos >= len(tr.schema.parameters):
                raise FrontendError(f"{description} coordinate owner has invalid parameter geometry")
            if tr.schema.parameters[pos] != coordinate_name:
                raise FrontendError(f"{description} coordinate parameter identity changed")
            current = (
                self.frontend.specialized_coordinate_domains.get(coordinate_owner_uid)
                or self.frontend.coordinate_domains.get(coordinate_owner_uid)
            )
            if current is None:
                raise FrontendError(f"{description} coordinate domain is no longer proven")
            if int(current.lo) != coordinate_lo or int(current.hi) != coordinate_hi:
                raise FrontendError(f"{description} coordinate domain changed after compilation")

        def evaluate(keys):
            validate_coordinate_domain()
            rows: list[list[Any]] = []
            for run_key in keys:
                cell = self.frontend._context_cell_for_key(ref_name, cell_name, run_key)
                row: list[Any] = []
                for coordinate in range(coordinate_lo, coordinate_hi + 1, coordinate_step):
                    binding = (coordinate_name, int(coordinate))
                    args = [
                        self._bound_numeric_expr_value(node, run_key, binding)
                        for node in arg_nodes
                    ]
                    try:
                        value = cell(*args)
                    except Exception as exc:
                        raise FrontendError(
                            f"{description} could not be materialized at {coordinate_name}={coordinate}: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    self._numeric_scalar(value, description)
                    row.append(value)
                rows.append(row)
            return rows

        sample_rows = evaluate(self.frontend.run_keys)
        flat_sample = [value for row in sample_rows for value in row]
        dtype = self._numeric_dtype(flat_sample, description)
        table_key = self._key("ctxcurve")
        self.memo[token] = table_key

        def getter(keys, evaluate=evaluate, dtype=dtype, description=description, width=width):
            rows = evaluate(keys)
            if any(len(row) != width for row in rows):
                raise FrontendError(f"{description} coordinate table width changed")
            flat = [value for row in rows for value in row]
            arr = self._coerce_numeric_values(flat, dtype, description)
            return np.ascontiguousarray(arr.reshape((len(rows), width)))

        self.specs[table_key] = InputSpec(
            table_key, dtype, 2, "global", getter, description,
            expected_shape=(len(self.frontend.run_keys), width),
            domain_token=repr((ref_name, cell_name, arg_token, coordinate_domain_identity)),
        )
        return table_key, self.point_ordinal()

    def external_dataframe_scalar(
        self, ref_name: str, cell_name: str, cell, row_key: Any, column: Any
    ) -> str:
        """Normalize one labelled external-DataFrame lookup to a numeric scalar input."""
        token = ("external_dataframe_scalar", ref_name, cell_name, repr(row_key), repr(column))
        if token in self.memo:
            return self.memo[token]
        frame = cell()
        if not isinstance(frame, pd.DataFrame):
            raise FrontendError(
                f"external table {ref_name}.{cell_name} no longer returns a DataFrame"
            )
        if column not in frame.columns:
            raise FrontendError(
                f"external table {ref_name}.{cell_name} has no column {column!r}"
            )
        try:
            sample = frame.loc[row_key, column]
        except Exception as exc:
            raise FrontendError(
                f"external table lookup {ref_name}.{cell_name}.loc[{row_key!r}, {column!r}] failed"
            ) from exc
        if isinstance(sample, (bool, np.bool_, int, np.integer)):
            dtype = "int64"
        elif isinstance(sample, (float, np.floating)):
            dtype = "float64"
        else:
            raise FrontendError(
                f"external table lookup {ref_name}.{cell_name} column {column!r} is non-numeric"
            )
        key = self._key("extt0")
        self.memo[token] = key
        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()

        def getter(keys, c=cell, rk=row_key, col=column, ei=expected_index, ec=expected_columns):
            current = c()
            if not isinstance(current, pd.DataFrame):
                raise FrontendError("external table provider changed type after compilation")
            if not current.index.equals(ei) or not current.columns.equals(ec):
                raise FrontendError("external table domain changed after compilation")
            return current.loc[rk, col]

        def domain_guard(c=cell, ei=expected_index, ec=expected_columns, rn=ref_name, cn=cell_name):
            current = c()
            if not isinstance(current, pd.DataFrame):
                return f"table_domain:{rn}.{cn}:type_changed"
            if not current.index.equals(ei):
                return f"table_domain:{rn}.{cn}:index_changed"
            if not current.columns.equals(ec):
                return f"table_domain:{rn}.{cn}:columns_changed"
            return None

        nlevels = int(frame.index.nlevels) if isinstance(frame.index, pd.MultiIndex) else 1
        row_parts = row_key if isinstance(row_key, tuple) else (row_key,)
        fixed = tuple((i, row_parts[i]) for i in range(min(nlevels, len(row_parts))))
        semantic = NormalizedTableInput(
            source_ref=ref_name, source_cell=cell_name, source_shape=tuple(frame.shape),
            index_names=tuple(frame.index.names), selected_column=column,
            fixed_row_labels=fixed, dynamic_level=None, dynamic_labels=(),
            axis_start=None, axis_step=None, result_kind="scalar",
        )
        self.specs[key] = InputSpec(
            key, dtype, 0, "global", getter,
            f"normalized external DataFrame scalar {ref_name}.{cell_name}[{column!r}]",
            domain_guard=domain_guard,
            domain_token=repr((tuple(frame.index.tolist()), tuple(frame.columns.tolist()))),
            expected_shape=(), normalized_table=semantic,
        )
        return key

    def external_dataframe_point_lookup(
        self,
        ref_name: str,
        cell_name: str,
        cell,
        *,
        column: Any,
        selector_token: Any,
        selector_getter: Callable[[Any], Any],
    ) -> tuple[str, str]:
        """Prepare one exact DataFrame lookup per RunDomain point.

        This is the object/provider escape hatch for selectors that are numeric or
        finite but whose *global* categorical domain contains labels absent from
        the particular external table.  Preparation evaluates the exact selector
        for each point and materializes ``value`` plus ``valid`` buffers.  Runtime
        code remains numeric and fail-closed when the source ``.loc`` would miss.
        """
        token = (
            "external_dataframe_point_lookup", ref_name, cell_name,
            repr(column), repr(selector_token),
        )
        if token in self.point_lookup_memo:
            return self.point_lookup_memo[token]
        description = (
            f"point-prepared exact external DataFrame lookup "
            f"{ref_name}.{cell_name}[{column!r}]"
        )
        frame = cell()
        if not isinstance(frame, pd.DataFrame):
            raise FrontendError(
                f"external table {ref_name}.{cell_name} no longer returns a DataFrame"
            )
        if column not in frame.columns:
            raise FrontendError(
                f"external table {ref_name}.{cell_name} has no column {column!r}"
            )
        kind = np.asarray(frame[column]).dtype.kind
        if kind in "iub":
            dtype = "int64"
            default = 0
        elif kind in "fc":
            dtype = "float64"
            default = 0.0
        else:
            raise FrontendError(
                f"external table {ref_name}.{cell_name} column {column!r} is non-numeric"
            )
        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()

        def current_table():
            current = cell()
            if not isinstance(current, pd.DataFrame):
                raise FrontendError("external point table provider changed type after compilation")
            if not current.index.equals(expected_index) or not current.columns.equals(expected_columns):
                raise FrontendError("external point table domain changed after compilation")
            return current

        def materialize(selected):
            table = current_table()
            values: list[Any] = []
            valid: list[bool] = []
            for run_key in selected:
                try:
                    row_key = selector_getter(run_key)
                except Exception as exc:
                    raise FrontendError(
                        f"{description} selector evaluation failed for {run_key!r}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                try:
                    value = table.loc[row_key, column]
                except KeyError:
                    values.append(default)
                    valid.append(False)
                    continue
                # A unique exact row selector must produce one scalar.  Duplicate
                # labels/partial slicing are intentionally outside this bridge.
                if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
                    raise FrontendError(
                        f"{description} selector {row_key!r} did not produce a scalar"
                    )
                self._numeric_scalar(value, description)
                values.append(value)
                valid.append(True)
            return (
                self._coerce_numeric_values(values, dtype, description),
                np.asarray(valid, dtype=np.bool_),
            )

        sample_values, sample_valid = materialize(self.frontend.run_keys)
        value_key = self._key("ptlookup")
        valid_key = self._key("ptvalid")
        self.point_lookup_memo[token] = (value_key, valid_key)
        cache_token = ("external_dataframe_point_lookup", token)

        def packed(selected, batch_cache):
            cache_key = (cache_token, tuple(selected))
            if cache_key not in batch_cache:
                batch_cache[cache_key] = materialize(selected)
            return batch_cache[cache_key]

        def value_fast(selected, batch_cache):
            return packed(selected, batch_cache)[0]

        def valid_fast(selected, batch_cache):
            return packed(selected, batch_cache)[1]

        def domain_guard():
            try:
                current_table()
            except FrontendError as exc:
                return f"table_domain:{ref_name}.{cell_name}:{exc}"
            return None

        domain_token = repr((
            ref_name, cell_name, repr(column), selector_token,
            tuple(frame.index.tolist()), tuple(frame.columns.tolist()),
            "point_prepared_exact",
        ))
        shape = (len(self.frontend.run_keys),)
        self.specs[value_key] = InputSpec(
            value_key, dtype, 1, "point", lambda selected: materialize(selected)[0],
            description + " values", domain_guard=domain_guard,
            domain_token=domain_token, expected_shape=shape,
            fast_getter=value_fast,
        )
        self.specs[valid_key] = InputSpec(
            valid_key, "bool", 1, "point", lambda selected: materialize(selected)[1],
            description + " validity", domain_guard=domain_guard,
            domain_token=domain_token, expected_shape=shape,
            fast_getter=valid_fast,
        )
        return value_key, valid_key

    def external_dataframe_axis(
        self, ref_name: str, cell_name: str, cell, *, fixed_levels: dict[int, Any],
        dynamic_level: int, column: Any,
    ) -> tuple[str, int, int]:
        """Normalize a labelled table slice with one dynamic numeric row axis.

        Categorical/static row levels are selected during preparation.  The remaining
        numeric level becomes a dense 1-D numeric input plus an affine label->position
        mapping shared by Python and Cython through the canonical ``array_input`` op.
        """
        fixed_tuple = tuple(sorted((int(k), v) for k, v in fixed_levels.items()))
        token = (
            "external_dataframe_axis", ref_name, cell_name,
            tuple((k, repr(v)) for k, v in fixed_tuple), int(dynamic_level), repr(column),
        )
        if token in self.memo:
            key = self.memo[token]
            sem = self.specs[key].normalized_table
            assert sem is not None and sem.axis_start is not None and sem.axis_step is not None
            return key, int(sem.axis_start), int(sem.axis_step)
        frame = cell()
        if not isinstance(frame, pd.DataFrame):
            raise FrontendError(
                f"external table {ref_name}.{cell_name} no longer returns a DataFrame"
            )
        if column not in frame.columns:
            raise FrontendError(
                f"external table {ref_name}.{cell_name} has no column {column!r}"
            )
        nlevels = int(frame.index.nlevels) if isinstance(frame.index, pd.MultiIndex) else 1
        if dynamic_level < 0 or dynamic_level >= nlevels:
            raise FrontendError("external table dynamic level is outside the row index")
        if any(level < 0 or level >= nlevels or level == dynamic_level for level, _ in fixed_tuple):
            raise FrontendError("invalid fixed level in normalized external table lookup")
        positions: list[int] = []
        labels: list[Any] = []
        for pos, label in enumerate(frame.index):
            parts = label if isinstance(label, tuple) else (label,)
            if all(parts[level] == value for level, value in fixed_tuple):
                positions.append(pos)
                labels.append(parts[dynamic_level])
        if not positions:
            raise FrontendError(
                f"external table {ref_name}.{cell_name} has no rows for fixed labels {fixed_tuple!r}"
            )
        numeric_labels, start, step = _safe_numeric_axis(labels)
        if len(set(numeric_labels)) != len(numeric_labels):
            raise FrontendError("external table dynamic row axis is not unique after fixed-level selection")
        order = np.asarray(sorted(range(len(numeric_labels)), key=lambda i: numeric_labels[i]), dtype=np.intp)
        positions_arr = np.asarray(positions, dtype=np.intp)[order]
        sorted_labels = tuple(numeric_labels[i] for i in order)
        col_kind = np.asarray(frame[column]).dtype.kind
        if col_kind in "iub":
            dtype = "int64"
        elif col_kind in "fc":
            dtype = "float64"
        else:
            raise FrontendError(
                f"external table {ref_name}.{cell_name} column {column!r} is non-numeric"
            )
        key = self._key("extt1")
        self.memo[token] = key
        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()

        def getter(keys, c=cell, col=column, pos=positions_arr, ei=expected_index, ec=expected_columns):
            current = c()
            if not isinstance(current, pd.DataFrame):
                raise FrontendError("external table provider changed type after compilation")
            if not current.index.equals(ei) or not current.columns.equals(ec):
                raise FrontendError("external table domain changed after compilation")
            arr = current[col].to_numpy(copy=False)[pos]
            return np.ascontiguousarray(arr)

        def domain_guard(c=cell, ei=expected_index, ec=expected_columns, rn=ref_name, cn=cell_name):
            current = c()
            if not isinstance(current, pd.DataFrame):
                return f"table_domain:{rn}.{cn}:type_changed"
            if not current.index.equals(ei):
                return f"table_domain:{rn}.{cn}:index_changed"
            if not current.columns.equals(ec):
                return f"table_domain:{rn}.{cn}:columns_changed"
            return None

        semantic = NormalizedTableInput(
            source_ref=ref_name, source_cell=cell_name, source_shape=tuple(frame.shape),
            index_names=tuple(frame.index.names), selected_column=column,
            fixed_row_labels=fixed_tuple, dynamic_level=int(dynamic_level),
            dynamic_labels=sorted_labels, axis_start=int(start), axis_step=int(step),
            result_kind="axis1d",
        )
        self.specs[key] = InputSpec(
            key, dtype, 1, "global", getter,
            f"normalized external DataFrame axis {ref_name}.{cell_name}[{column!r}]",
            domain_guard=domain_guard,
            domain_token=repr((tuple(frame.index.tolist()), tuple(frame.columns.tolist()))),
            expected_shape=(len(sorted_labels),), normalized_table=semantic,
        )
        return key, int(start), int(step)

    def external_dataframe_exact_nd(
        self,
        ref_name: str,
        cell_name: str,
        cell,
        *,
        fixed_levels: dict[int, Any],
        dynamic_axes: Sequence[NormalizedTableAxis],
        column: Any,
        allow_missing_categorical_labels: bool = False,
    ) -> tuple[str, str | None, str]:
        """Normalize an exact N-D labelled lookup to numeric backend inputs.

        Static row levels are removed during preparation.  Runtime selector axes
        are already encoded as integer positions by the source normalizer.  A
        Cartesian table becomes one dense ndarray; a non-Cartesian finite table
        becomes an integer key matrix plus a value vector.  Missing sparse keys
        therefore retain source ``.loc`` semantics rather than being silently
        filled with a numeric sentinel.
        """
        if not dynamic_axes:
            raise FrontendError("exact N-D table normalization requires at least one dynamic axis")
        fixed_tuple = tuple(sorted((int(k), v) for k, v in fixed_levels.items()))
        axes = tuple(dynamic_axes)
        token = (
            "external_dataframe_exact_nd", ref_name, cell_name,
            tuple((k, repr(v)) for k, v in fixed_tuple), repr(column),
            tuple((
                a.level, a.selector_kind, tuple(map(repr, a.labels)),
                None if a.reachable_labels is None else tuple(map(repr, a.reachable_labels)),
            ) for a in axes),
            bool(allow_missing_categorical_labels),
        )
        if token in self.memo:
            value_key, sparse_key, layout = self.memo[token]
            return value_key, sparse_key, layout

        frame = cell()
        if not isinstance(frame, pd.DataFrame):
            raise FrontendError(f"external table {ref_name}.{cell_name} no longer returns a DataFrame")
        if column not in frame.columns:
            raise FrontendError(f"external table {ref_name}.{cell_name} has no column {column!r}")
        nlevels = int(frame.index.nlevels) if isinstance(frame.index, pd.MultiIndex) else 1
        dynamic_levels = tuple(int(a.level) for a in axes)
        if len(set(dynamic_levels)) != len(dynamic_levels):
            raise FrontendError("normalized exact table lookup repeats a dynamic index level")
        if any(level < 0 or level >= nlevels for level in dynamic_levels):
            raise FrontendError("normalized exact table lookup has a dynamic level outside the table index")
        if any(level < 0 or level >= nlevels or level in dynamic_levels for level, _ in fixed_tuple):
            raise FrontendError("normalized exact table lookup has an invalid fixed level")

        label_codes: list[dict[Any, int]] = []
        for axis in axes:
            if not axis.labels or len(set(axis.labels)) != len(axis.labels):
                raise FrontendError(
                    f"normalized exact table axis {axis.index_name!r} must have a non-empty unique label domain"
                )
            label_codes.append({label: i for i, label in enumerate(axis.labels)})

        positions: list[int] = []
        encoded_keys: list[tuple[int, ...]] = []
        for pos, label in enumerate(frame.index):
            parts = label if isinstance(label, tuple) else (label,)
            if not all(parts[level] == value for level, value in fixed_tuple):
                continue
            try:
                key_tuple = tuple(
                    label_codes[j][parts[axis.level]] for j, axis in enumerate(axes)
                )
            except KeyError:
                # The source table may contain labels outside the proven runtime
                # selector domain. They are irrelevant to this compiled lookup.
                continue
            positions.append(pos)
            encoded_keys.append(key_tuple)
        if not positions:
            raise FrontendError(
                f"external table {ref_name}.{cell_name} has no rows in the normalized selector domain"
            )
        # A finite categorical selector is a semantic label domain, not merely a
        # convenient integer coordinate.  If one of its labels is absent from
        # the frozen table axis under the selected fixed prefix, that value would
        # inevitably raise at source runtime. Reject it here rather than allowing
        # an invalid code (notably -1) to become accidental Python/C negative
        # indexing in a dense backend. Sparse cross-axis combinations remain
        # legal; only an axis label that is absent entirely is contradictory.
        for j, axis in enumerate(axes):
            if axis.selector_kind != "finite_categorical":
                continue
            observed = {key[j] for key in encoded_keys}
            proof_labels = axis.labels if axis.reachable_labels is None else axis.reachable_labels
            missing = tuple(
                label for label in proof_labels
                if label_codes[j].get(label) not in observed
            )
            if missing and not allow_missing_categorical_labels:
                raise FrontendError(
                    f"external table {ref_name}.{cell_name} categorical axis "
                    f"{axis.index_name!r} is missing selector labels {missing!r}"
                )
        if len(set(encoded_keys)) != len(encoded_keys):
            raise FrontendError("normalized exact table lookup does not have unique dynamic keys")

        shape = tuple(len(a.labels) for a in axes)
        expected_count = int(np.prod(shape, dtype=np.int64))
        full_keys = set(__import__("itertools").product(*(range(n) for n in shape)))
        dense = len(encoded_keys) == expected_count and set(encoded_keys) == full_keys
        layout = "dense_cartesian" if dense else "sparse_finite_keys"

        col_kind = np.asarray(frame[column]).dtype.kind
        if col_kind in "iub":
            dtype = "int64"
            np_dtype = np.int64
        elif col_kind in "fc":
            dtype = "float64"
            np_dtype = np.float64
        else:
            raise FrontendError(
                f"external table {ref_name}.{cell_name} column {column!r} is non-numeric"
            )

        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()
        positions_arr = np.asarray(positions, dtype=np.intp)

        def current(c=cell, ei=expected_index, ec=expected_columns):
            table = c()
            if not isinstance(table, pd.DataFrame):
                raise FrontendError("external exact table provider changed type after compilation")
            if not table.index.equals(ei) or not table.columns.equals(ec):
                raise FrontendError("external exact table domain changed after compilation")
            return table

        def domain_guard(current=current, rn=ref_name, cn=cell_name):
            try:
                current()
            except FrontendError as exc:
                return f"table_domain:{rn}.{cn}:{exc}"
            return None

        value_key = self._key("exttn")
        sparse_key: str | None = None
        semantic = NormalizedTableInput(
            source_ref=ref_name,
            source_cell=cell_name,
            source_shape=tuple(frame.shape),
            index_names=tuple(frame.index.names),
            selected_column=column,
            fixed_row_labels=fixed_tuple,
            dynamic_level=None,
            dynamic_labels=(),
            axis_start=None,
            axis_step=None,
            result_kind="exact_nd",
            layout=layout,
            dynamic_axes=axes,
        )

        if dense:
            position_by_key = {key: pos for key, pos in zip(encoded_keys, positions)}
            order = np.asarray(
                [position_by_key[key] for key in __import__("itertools").product(*(range(n) for n in shape))],
                dtype=np.intp,
            )

            def getter(keys, current=current, col=column, order=order, shape=shape, dt=np_dtype):
                arr = current()[col].to_numpy(copy=False)[order]
                return np.ascontiguousarray(arr, dtype=dt).reshape(shape)

            self.specs[value_key] = InputSpec(
                value_key, dtype, len(shape), "global", getter,
                f"normalized dense exact table {ref_name}.{cell_name}[{column!r}]",
                domain_guard=domain_guard,
                domain_token=repr((tuple(frame.index.tolist()), tuple(frame.columns.tolist()), layout, shape)),
                expected_shape=shape,
                normalized_table=semantic,
            )
        else:
            encoded_arr = np.ascontiguousarray(np.asarray(encoded_keys, dtype=np.int64))
            sparse_key = self._key("exttk")
            semantic = NormalizedTableInput(
                **{**semantic.__dict__, "sparse_key_input": sparse_key}
            )

            def key_getter(keys, arr=encoded_arr):
                return arr

            def value_getter(keys, current=current, col=column, pos=positions_arr, dt=np_dtype):
                arr = current()[col].to_numpy(copy=False)[pos]
                return np.ascontiguousarray(arr, dtype=dt)

            self.specs[sparse_key] = InputSpec(
                sparse_key, "int64", 2, "global", key_getter,
                f"normalized sparse exact table keys {ref_name}.{cell_name}[{column!r}]",
                domain_token=repr((layout, tuple(encoded_keys))),
                expected_shape=tuple(encoded_arr.shape),
            )
            self.specs[value_key] = InputSpec(
                value_key, dtype, 1, "global", value_getter,
                f"normalized sparse exact table values {ref_name}.{cell_name}[{column!r}]",
                domain_guard=domain_guard,
                domain_token=repr((tuple(frame.index.tolist()), tuple(frame.columns.tolist()), layout)),
                expected_shape=(len(positions),),
                normalized_table=semantic,
            )

        self.memo[token] = (value_key, sparse_key, layout)
        return value_key, sparse_key, layout

    def external_dataframe_row_axis_extreme(
        self,
        ref_name: str,
        cell_name: str,
        cell,
        *,
        fixed_levels: Mapping[int, Any],
        dynamic_axes: Sequence[NormalizedTableAxis],
        residual_level: int,
        extreme: str,
    ) -> tuple[str, str | None, str]:
        """Normalize ``row_view.index.min/max`` to numeric metadata ABI.

        The source row view must fix every MultiIndex level except one residual
        integral axis.  Runtime selectors address only the fixed-prefix domain;
        preparation derives the exact min/max residual label for every admitted
        prefix.  Pandas objects therefore remain preparation-only.
        """
        if extreme not in {"min", "max"}:
            raise FrontendError(f"unsupported row-view index extreme {extreme!r}")
        fixed_tuple = tuple(sorted((int(k), v) for k, v in fixed_levels.items()))
        axes = tuple(dynamic_axes)
        token = (
            "external_dataframe_row_axis_extreme", ref_name, cell_name,
            tuple((k, repr(v)) for k, v in fixed_tuple),
            tuple((a.level, a.selector_kind, tuple(map(repr, a.labels))) for a in axes),
            int(residual_level), extreme,
        )
        if token in self.memo:
            value_key, sparse_key, layout = self.memo[token]
            return value_key, sparse_key, layout

        frame = cell()
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.MultiIndex):
            raise FrontendError(
                f"external table {ref_name}.{cell_name} row-axis metadata requires a MultiIndex DataFrame"
            )
        nlevels = int(frame.index.nlevels)
        residual_level = int(residual_level)
        dynamic_levels = tuple(int(a.level) for a in axes)
        if residual_level < 0 or residual_level >= nlevels:
            raise FrontendError("row-view residual axis is outside the table index")
        if len(set(dynamic_levels)) != len(dynamic_levels):
            raise FrontendError("row-view metadata repeats a dynamic prefix level")
        accounted = set(dynamic_levels) | {k for k, _ in fixed_tuple} | {residual_level}
        if accounted != set(range(nlevels)):
            raise FrontendError(
                "row-view axis metadata requires every non-residual index level to be fixed or dynamic"
            )
        if residual_level in dynamic_levels or any(k == residual_level for k, _ in fixed_tuple):
            raise FrontendError("row-view residual axis cannot also be a prefix selector")

        label_codes: list[dict[Any, int]] = []
        for axis in axes:
            if not axis.labels or len(set(axis.labels)) != len(axis.labels):
                raise FrontendError("row-view metadata dynamic axis must have unique labels")
            label_codes.append({label: i for i, label in enumerate(axis.labels)})

        grouped: dict[tuple[int, ...], list[int]] = {}
        for label in frame.index:
            parts = label if isinstance(label, tuple) else (label,)
            if not all(parts[level] == value for level, value in fixed_tuple):
                continue
            try:
                key_tuple = tuple(
                    label_codes[j][parts[axis.level]] for j, axis in enumerate(axes)
                )
            except KeyError:
                continue
            raw = parts[residual_level]
            if not isinstance(raw, (int, np.integer)) or isinstance(raw, (bool, np.bool_)):
                raise FrontendError(
                    f"external table {ref_name}.{cell_name} residual index level "
                    f"{residual_level} must be integral for row-view min/max metadata"
                )
            grouped.setdefault(key_tuple, []).append(int(raw))
        if not grouped:
            raise FrontendError(
                f"external table {ref_name}.{cell_name} has no rows in the normalized row-view prefix domain"
            )

        encoded_keys = tuple(sorted(grouped))
        values_by_key = {
            key: (min(values) if extreme == "min" else max(values))
            for key, values in grouped.items()
        }
        shape = tuple(len(a.labels) for a in axes)
        full_keys = set(__import__("itertools").product(*(range(n) for n in shape)))
        dense = set(encoded_keys) == full_keys
        layout = "dense_cartesian" if dense else "sparse_finite_keys"

        expected_index = frame.index.copy()
        expected_columns = frame.columns.copy()

        def current(c=cell, ei=expected_index, ec=expected_columns):
            table = c()
            if not isinstance(table, pd.DataFrame):
                raise FrontendError("row-view metadata provider changed type after compilation")
            if not table.index.equals(ei) or not table.columns.equals(ec):
                raise FrontendError("row-view metadata table domain changed after compilation")
            return table

        def domain_guard(current=current, rn=ref_name, cn=cell_name):
            try:
                current()
            except FrontendError as exc:
                return f"table_domain:{rn}.{cn}:{exc}"
            return None

        value_key = self._key("extmeta")
        sparse_key: str | None = None
        semantic = NormalizedTableInput(
            source_ref=ref_name,
            source_cell=cell_name,
            source_shape=tuple(frame.shape),
            index_names=tuple(frame.index.names),
            selected_column=None,
            fixed_row_labels=fixed_tuple,
            dynamic_level=None,
            dynamic_labels=(),
            axis_start=None,
            axis_step=None,
            result_kind="axis_metadata_scalar",
            layout=layout,
            dynamic_axes=axes,
            sparse_key_input=None,
            metadata_kind=f"index_{extreme}",
            metadata_level=residual_level,
        )

        if dense:
            order = tuple(__import__("itertools").product(*(range(n) for n in shape)))
            dense_values = np.asarray([values_by_key[key] for key in order], dtype=np.int64).reshape(shape)

            def getter(keys, current=current, arr=dense_values):
                current()
                return np.ascontiguousarray(arr, dtype=np.int64)

            self.specs[value_key] = InputSpec(
                value_key, "int64", len(shape), "global", getter,
                f"normalized row-view index {extreme} metadata {ref_name}.{cell_name}",
                domain_guard=domain_guard,
                domain_token=repr((tuple(frame.index.tolist()), tuple(frame.columns.tolist()), extreme, residual_level, shape)),
                expected_shape=shape,
                normalized_table=semantic,
            )
        else:
            encoded_arr = np.ascontiguousarray(np.asarray(encoded_keys, dtype=np.int64))
            values_arr = np.ascontiguousarray(
                np.asarray([values_by_key[key] for key in encoded_keys], dtype=np.int64)
            )
            sparse_key = self._key("extmetak")
            semantic = NormalizedTableInput(
                **{**semantic.__dict__, "sparse_key_input": sparse_key}
            )

            def key_getter(keys, current=current, arr=encoded_arr):
                current()
                return arr

            def value_getter(keys, current=current, arr=values_arr):
                current()
                return arr

            self.specs[sparse_key] = InputSpec(
                sparse_key, "int64", 2, "global", key_getter,
                f"normalized sparse row-view metadata keys {ref_name}.{cell_name}",
                domain_guard=domain_guard,
                domain_token=repr((layout, encoded_keys)),
                expected_shape=tuple(encoded_arr.shape),
            )
            self.specs[value_key] = InputSpec(
                value_key, "int64", 1, "global", value_getter,
                f"normalized sparse row-view index {extreme} metadata {ref_name}.{cell_name}",
                domain_guard=domain_guard,
                domain_token=repr((tuple(frame.index.tolist()), tuple(frame.columns.tolist()), extreme, residual_level, layout)),
                expected_shape=(len(encoded_keys),),
                normalized_table=semantic,
            )

        self.memo[token] = (value_key, sparse_key, layout)
        return value_key, sparse_key, layout


class GraphModelCompiler:
    """Graph-guided, name-agnostic frontend for the native optimizer.

    The caller identifies the Space, output Cell and representative ItemSpace keys.
    All semantic properties (time parameter, variants, types, lags and phase
    boundaries) are inferred from the modelx calculation graph and formulas.
    """

    def __init__(
        self,
        model,
        *,
        space,
        output: str,
        sample_keys: Sequence[Any],
        run_keys: Sequence[Any] | None = None,
        output_invocation: OutputInvocation | None = None,
        run_domain: RunDomain | None = None,
        fallback_cells: Sequence[str] = (),
        trace_capture: TraceCapture | None = None,
        realized_graph_only: bool = False,
        recovered_loop_evidence: Any | None = None,
        recovered_loop_evidence_provider: Any | None = None,
        canonical_schedule_authority: bool = False,
        legacy_comparison: bool | None = None,
    ):
        self.model = model
        self.space = space
        self.output_invocation = (
            output_invocation
            if output_invocation is not None
            else OutputInvocation.fixed(str(output), (), ())
        )
        if self.output_invocation.output_name != str(output):
            raise FrontendError(
                "OutputInvocation output name does not match GraphModelCompiler output"
            )
        self.output = self.output_invocation.output_name
        self.sample_keys = tuple(sample_keys)
        self.fallback_cells = frozenset(fallback_cells)
        self.realized_graph_only = bool(realized_graph_only)
        self.recovered_loop_evidence = recovered_loop_evidence
        self.recovered_loop_evidence_provider = recovered_loop_evidence_provider
        self.canonical_schedule_authority_enabled = bool(canonical_schedule_authority)
        # Canonical-authority compilation is production-direct by default.  Mature
        # ExecutableGraph construction survives only for the pre-canonical
        # compatibility path or when comparison mode is explicitly requested.
        self.legacy_comparison_enabled = (
            not self.canonical_schedule_authority_enabled
            if legacy_comparison is None else bool(legacy_comparison)
        )
        unknown_fallbacks = self.fallback_cells - set(space.cells)
        if unknown_fallbacks:
            raise FrontendError(f"unknown fallback Cells: {sorted(unknown_fallbacks)!r}")
        for name in self.fallback_cells:
            if space.cells[name].parameters:
                raise FrontendError(
                    f"regional fallback Cell {name!r} must currently be zero-argument within the ItemSpace"
                )
        self.fallback_input_by_name: dict[str, str] = {}
        self.refs = dict(space.refs)
        self.space_params = tuple(space.parameters or ())
        self.space_param = self.space_params[0] if self.space_params else None
        if run_domain is not None:
            if tuple(run_domain.itemspace_parameter_names) != self.space_params:
                raise FrontendError(
                    "RunDomain ItemSpace parameters do not match the selected Space"
                )
            if str(getattr(space, "fullname", None) or getattr(space, "name", "<space>")) != run_domain.space_fullname:
                raise FrontendError(
                    "RunDomain structural Space does not match the selected Space"
                )
            if run_keys is not None and tuple(run_keys) != tuple(run_domain.run_keys):
                raise FrontendError("run_keys disagree with explicit RunDomain")
            self.run_domain = run_domain
            self.run_keys = tuple(run_domain.run_keys)
        else:
            resolved_keys = tuple(run_keys) if run_keys is not None else self._infer_run_keys()
            try:
                self.run_domain = RunDomain.finite(
                    space_fullname=str(getattr(space, "fullname", None) or getattr(space, "name", "<space>")),
                    itemspace_parameter_names=self.space_params,
                    run_keys=resolved_keys,
                )
            except (TypeError, ValueError) as exc:
                raise FrontendError(str(exc)) from exc
            self.run_keys = tuple(self.run_domain.run_keys)
        self.result_domain = ResultDomain.scalar_per_run_key(self.run_domain)
        self.multiindex_axes: dict[str, tuple[int,int,int,int]] = {}
        self.registry = InputRegistry(self)
        self.model_point_row = self._detect_point_row_cell()

        self.trace = (
            trace_capture
            if trace_capture is not None
            else capture_trace(
                model,
                lambda k: getattr(self._space_instance(k), self.output).node(
                    *self.output_invocation.argument_values
                ),
                self.sample_keys,
            )
        )
        self.trace_source = "realized_trace" if trace_capture is not None else "capture_trace"

        # Trace closures may contain Cells from external Spaces with the same
        # short name as a Cell in the selected Space.  Compiler identity is the
        # full Cell schema, never the name alone.  This map is also intentionally
        # source-sensitive so an inherited/overridden formula cannot be mistaken
        # for another Space's implementation.
        self.selected_schema_uid_by_name = {
            name: CellSchema(
                cell.fullname, name, tuple(cell.parameters), cell.formula.source
            ).uid
            for name, cell in space.cells.items()
            if cell.formula is not None
        }
        self.selected_trace_variants = {
            vk: tr for vk, tr in self.trace.variants.items()
            if self.selected_schema_uid_by_name.get(tr.schema.name) == tr.schema.uid
        }

        explicit = [
            tr.schema.fullname for tr in self.selected_trace_variants.values()
            if tr.has_input_values and tr.schema.name not in self.fallback_cells
        ]
        if explicit:
            raise FrontendError(
                "explicit Cell input values are present in the sampled closure; native seed/input lowering "
                "is not implemented, so fallback is required: " + ", ".join(sorted(set(explicit)))
            )
        self.base_funcs = {name: ast.parse(cell.formula.source).body[0] for name, cell in space.cells.items() if cell.formula is not None}
        # Source ValueSemantics is a positive admission gate for execution PureMaps.
        # Reconstruct one source module from the exact selected-Space formulas; free
        # model inputs remain immutable source evidence, while inter-Cell calls retain
        # their real names and recurrence structure.  This evidence never replaces
        # canonical specialization; it only prevents StateDerivedMaps from being
        # promoted by an accidentally acyclic specialized closure.
        _semantics_source = ast.unparse(
            ast.fix_missing_locations(
                ast.Module(
                    body=[copy.deepcopy(self.base_funcs[name]) for name in sorted(self.base_funcs)],
                    type_ignores=[],
                )
            )
        )
        self.source_value_semantics = {
            row.uid: row
            for row in classify_source_value_semantics(_semantics_source)
        }
        self.source_pure_map_names = frozenset(
            uid for uid, row in self.source_value_semantics.items()
            if row.semantic_class == "pure_map"
        )

        # A regional fallback Cell is a terminal dependency from the compiled
        # artifact's perspective.  Compute the source-level closure with those
        # identities cut before lowering any formulas.  This prevents unsupported
        # formulas upstream of a fallback boundary from being needlessly parsed as
        # native code (and is essential when several fallback islands exist in one
        # original modelx trace).
        self.compiled_source_names: set[str] = set()
        stack = [self.output]
        while stack:
            name = stack.pop()
            if name in self.compiled_source_names or name in self.fallback_cells:
                continue
            if name not in self.base_funcs:
                continue
            self.compiled_source_names.add(name)
            fn = self.base_funcs[name]
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    dep = node.func.id
                    if dep in self.base_funcs and dep not in self.fallback_cells:
                        stack.append(dep)
                elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                    dep = node.value.id
                    if dep in self.base_funcs and dep not in self.fallback_cells:
                        stack.append(dep)

        self.schema_by_name = {}
        for tr in self.selected_trace_variants.values():
            self.schema_by_name.setdefault(tr.schema.name, tr.schema)
        self.variant_trace_by_uid = {tr.uid: tr for tr in self.selected_trace_variants.values()}
        self.variants_by_cell_aux: dict[tuple[str, tuple[Any,...]], str] = {}
        for tr in self.selected_trace_variants.values():
            self.variants_by_cell_aux[(tr.schema.name, tr.key.aux_values)] = tr.uid
        self.validated_enum_cells = self._discover_validated_enum_cells()
        self.validated_point_scalar_cells = self._discover_validated_point_scalar_cells()
        self._static_variant_value_cache: dict[str, tuple[bool, Any]] = {}
        self._finite_variant_domain_cache: dict[str, tuple[bool, tuple[Any, ...]]] = {}
        self._static_reference_value_cache: dict[str, tuple[bool, Any]] = {}
        self._used_static_facts: dict[tuple[str, str], StaticScalarFact] = {}
        self._normalized_operators: dict[str, NormalizedOperator] = {}
        self._used_validated_static_facts: dict[str, ValidatedStaticFact] = {}
        self._used_finite_domain_facts: dict[tuple[str, str], FiniteDomainFact] = {}
        self._fixed_coordinate_uid_by_key: dict[tuple[str, int], str] = {}
        self._fixed_coordinate_requests: dict[str, tuple[str, int]] = {}
        self._used_fixed_coordinate_facts: dict[str, FixedCoordinateFact] = {}
        self._fixed_coordinate_limit = 128
        self._future_coordinate_source_cache: dict[str, bool] = {}
        self.coordinate_recurrence_facts: tuple[CoordinateRecurrenceDomainFact, ...] = ()
        self._runtime_scalar_helper_inline_stack: list[str] = []
        self._runtime_scalar_helper_expansion_limit = RUNTIME_SCALAR_HELPER_EXPANSION_LIMIT
        self._used_runtime_scalar_helper_facts: dict[str, RuntimeScalarHelperFact] = {}
        self._static_point_field_cache: dict[str, tuple[bool, Any]] = {}
        self._coordinate_constant_cache: dict[str, tuple[bool, Any]] = {}
        self.coordinate_domains: dict[str, _IntDomain] = {}
        # A later, demand-graph proof over already-specialized canonical source.
        # Unlike ``coordinate_domains`` above, this pass can consume literal
        # auxiliary specialization, source-semantic dead-tail pruning and
        # source-backed scheduled variants discovered during canonical demand.
        # It exists specifically to prove call-site preconditions such as an
        # irregular step lookup's first-anchor lower bound.  It must never be
        # populated from representative observed times.
        self.specialized_coordinate_domains: dict[str, _IntDomain] = {}
        self.primary_coordinate_domain: _IntDomain | None = None
        # Partial source-proven bounds for lookup/domain proofs when the physical
        # reduction range is not exact over the full RunDomain (for example a
        # policy-varying stop with a fixed positive start).  Scheduling continues
        # to use only ``primary_coordinate_domain`` exact intervals.
        self.primary_coordinate_bounds: tuple[int | None, int | None] = (None, None)
        self.primary_coordinate_affine_constraint: AffineRangeConstraint | None = None
        self.coordinate_domain_notes: list[str] = []
        self._source_recursive_reach_cache: dict[str, bool] = {}
        self._source_backed_helper_cache: dict[str, bool] = {}
        # Preparation-only proof caches.  Numeric/table normalization can ask the
        # same pure source helper question many times while visiting one model.
        # Keep immutable source metadata once and let each normalizer memoize the
        # resulting bound query.  This is proof reuse only: no runtime state or
        # execution scheduling authority is introduced here.
        self._pure_single_return_meta_cache: dict[
            str, tuple[tuple[str, ...], dict[str, ast.AST], ast.AST] | None
        ] = {}
        self._proof_diagnostics: dict[str, int] = {
            "numeric_bounds_requests": 0,
            "numeric_bounds_cache_hits": 0,
            "numeric_bounds_cache_stores": 0,
            "numeric_bounds_cycle_cuts": 0,
            "numeric_bounds_active_peak": 0,
            "numeric_bounds_alias_hops": 0,
            "numeric_bounds_alias_cycle_cuts": 0,
            "pure_single_return_meta_hits": 0,
            "pure_single_return_meta_misses": 0,
            "pure_single_return_expansions": 0,
        }
        self._source_backed_scheduled_uid_by_key: dict[tuple[str, tuple[Any, ...]], str] = {}
        self._used_source_backed_scheduled_facts: dict[str, SourceBackedScheduledFact] = {}
        self._used_step_lookup_domain_facts: dict[tuple[Any, ...], StepLookupDomainFact] = {}
        self._canonical_proof_snapshot: CanonicalProofSnapshot | None = None

        # The broad frontend remains source-complete.  NativeBatch has a narrower
        # contract: one compiled artifact represents one realized graph family.
        # Source branches absent from that graph are not pulled into the artifact;
        # a policy that takes another graph belongs to another backend/artifact.
        if not self.realized_graph_only:
            self._preflight_observed_dependencies()

        self.ir = build_graph_ir(self.trace)
        self.vector_elements = self._discover_vector_helpers()
        # Coordinate-domain discovery is a proof prepass.  It may consult the same
        # static-value machinery used by real specialization, but must not itself
        # perturb the generated ABI/input order.  Retain caches, restore registrations.
        saved_registry=(dict(self.registry.specs),dict(self.registry.memo),int(self.registry.counter))
        saved_static=dict(self._used_static_facts)
        saved_validated=dict(self._used_validated_static_facts)
        saved_finite=dict(self._used_finite_domain_facts)
        recurrence_facts: tuple[CoordinateRecurrenceDomainFact, ...] = ()
        try:
            self.coordinate_domains = self._derive_coordinate_domains()
            recurrence_facts = self._derive_coordinate_recurrence_facts(
                self.coordinate_domains
            )
        finally:
            self.registry.specs,self.registry.memo,self.registry.counter=saved_registry
            self._used_static_facts=saved_static
            self._used_validated_static_facts=saved_validated
            self._used_finite_domain_facts=saved_finite
        self.coordinate_recurrence_facts = recurrence_facts
        self.canonical = self._build_canonical()


    def _discover_validated_enum_cells(self) -> dict[str, ValidatedEnumCell]:
        """Recognize a tiny validated categorical-selector source pattern.

        The accepted form is intentionally structural and name-agnostic::

            v = model_point()["field"]
            if v not in ("A", "B", ...):
                raise ValueError(...)
            return v

        The selector may then be proven static over the explicit ``run_keys``
        domain without erasing the original validation contract: a numeric enum
        validation input is retained in the optimized program.
        """
        if self.model_point_row is None:
            return {}
        point_name = self.model_point_row.cell_name
        out: dict[str, ValidatedEnumCell] = {}
        for name, original in self.base_funcs.items():
            fn = _strip_docstring(copy.deepcopy(original))
            if fn.args.args or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
                continue
            body = list(fn.body)
            if len(body) != 3:
                continue
            assign, guard, ret = body
            if not (
                isinstance(assign, ast.Assign) and len(assign.targets) == 1
                and isinstance(assign.targets[0], ast.Name)
                and isinstance(assign.value, ast.Subscript)
                and isinstance(assign.value.value, ast.Call)
                and isinstance(assign.value.value.func, ast.Name)
                and assign.value.value.func.id == point_name
                and not assign.value.value.args and not assign.value.value.keywords
                and isinstance(assign.value.slice, ast.Constant)
                and isinstance(assign.value.slice.value, str)
            ):
                continue
            local = assign.targets[0].id
            if not (
                isinstance(guard, ast.If) and not guard.orelse
                and isinstance(guard.test, ast.Compare)
                and isinstance(guard.test.left, ast.Name) and guard.test.left.id == local
                and len(guard.test.ops) == 1 and isinstance(guard.test.ops[0], ast.NotIn)
                and len(guard.test.comparators) == 1
                and isinstance(ret, ast.Return)
                and isinstance(ret.value, ast.Name) and ret.value.id == local
                and any(isinstance(node, ast.Raise) for node in ast.walk(guard))
            ):
                continue
            domain_node = guard.test.comparators[0]
            if not isinstance(domain_node, (ast.Tuple, ast.List, ast.Set)):
                continue
            labels: list[str] = []
            for elt in domain_node.elts:
                if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                    break
                labels.append(elt.value)
            else:
                if 1 <= len(labels) <= 16 and len(set(labels)) == len(labels):
                    out[name] = ValidatedEnumCell(
                        field=str(assign.value.slice.value), allowed_values=tuple(labels)
                    )
        return out

    def _discover_validated_point_scalar_cells(self) -> dict[str, ValidatedPointScalarCell]:
        """Recognize exact-value-safe numeric/bool point selectors with guards.

        Accepted selectors have one point-field assignment, one or more guard Ifs
        containing a raise, and a final return of the assigned local.  The guard is
        not discarded here: `_proven_static_cell_value` later substitutes the proven
        scalar and requires each guard test to fold to literal False.
        """
        if self.model_point_row is None:
            return {}
        point_name = self.model_point_row.cell_name
        out: dict[str, ValidatedPointScalarCell] = {}

        def point_field_expr(expr: ast.AST):
            cast = None
            inner = expr
            if (
                isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                and inner.func.id in {"bool", "int", "float"}
                and len(inner.args) == 1 and not inner.keywords
            ):
                cast = inner.func.id
                inner = inner.args[0]
            if not (
                isinstance(inner, ast.Subscript)
                and isinstance(inner.value, ast.Call)
                and isinstance(inner.value.func, ast.Name)
                and inner.value.func.id == point_name
                and not inner.value.args and not inner.value.keywords
                and isinstance(inner.slice, ast.Constant)
                and isinstance(inner.slice.value, str)
            ):
                return None
            return str(inner.slice.value), cast

        for name, original in self.base_funcs.items():
            if name in self.validated_enum_cells:
                continue
            fn = _strip_docstring(copy.deepcopy(original))
            if fn.args.args or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
                continue
            if len(fn.body) < 3:
                continue
            assign, *middle, ret = fn.body
            if not (
                isinstance(assign, ast.Assign) and len(assign.targets) == 1
                and isinstance(assign.targets[0], ast.Name)
                and isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
                and ret.value.id == assign.targets[0].id
            ):
                continue
            parsed = point_field_expr(assign.value)
            if parsed is None:
                continue
            local = assign.targets[0].id
            guards = [
                st for st in middle
                if isinstance(st, ast.If) and any(isinstance(node, ast.Raise) for node in ast.walk(st))
            ]
            if not guards or len(guards) != len(middle):
                continue
            # Every guard must actually depend on the assigned selector.
            if not all(any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == local for n in ast.walk(g.test)) for g in guards):
                continue
            field, cast = parsed
            out[name] = ValidatedPointScalarCell(field=field, cast=cast)
        return out

    def _proven_static_variant_value(self, uid: str) -> tuple[bool, Any]:
        """Prove one non-coordinate Cell variant is constant over the build run domain.

        This is compile-time preparation only.  It is used narrowly by table-key
        normalization so categorical labels can disappear before Python/Cython
        emission.  If values differ across ``run_keys`` the proof fails closed.
        """
        cached = self._static_variant_value_cache.get(uid)
        if cached is not None:
            return cached
        tr = self.variant_trace_by_uid.get(uid)
        if tr is None or tr.key.time_pos is not None:
            out = (False, None)
            self._static_variant_value_cache[uid] = out
            return out
        allowed = (str, bool, int, float, np.bool_, np.integer, np.floating)
        values: list[Any] = []
        try:
            for key in self.run_keys:
                item = self._space_instance(key)
                cell = getattr(item, tr.schema.name)
                value = cell(*tr.key.aux_values) if tr.key.aux_values else cell()
                if not isinstance(value, allowed):
                    out = (False, None)
                    self._static_variant_value_cache[uid] = out
                    return out
                if isinstance(value, np.generic):
                    value = value.item()
                values.append(value)
        except Exception:
            out = (False, None)
            self._static_variant_value_cache[uid] = out
            return out
        if not values:
            out = (False, None)
        else:
            first = values[0]
            try:
                stable = all(type(v) is type(first) and v == first for v in values[1:])
            except Exception:
                stable = False
            out = (bool(stable), first if stable else None)
        self._static_variant_value_cache[uid] = out
        return out

    def _proven_static_cell_value(self, name: str) -> tuple[bool, Any]:
        base = self.base_funcs.get(name)
        guarded = base is not None and any(isinstance(node, ast.Raise) for node in ast.walk(base))
        enum_spec = self.validated_enum_cells.get(name) if guarded else None
        scalar_spec = self.validated_point_scalar_cells.get(name) if guarded else None
        # Ordinary guard-bearing Cells remain runtime semantic operations.  Two
        # structurally verified exceptions retain explicit point-input validation:
        # validated enums and exact numeric/bool point selectors whose guard becomes
        # unreachable after substituting the proven selector value.
        if guarded and enum_spec is None and scalar_spec is None:
            return False, None
        uid = self.variants_by_cell_aux.get((name, ()))
        if uid is None:
            return False, None
        ok, value = self._proven_static_variant_value(uid)
        if not ok:
            return False, None
        if enum_spec is not None:
            if value not in enum_spec.allowed_values or self.model_point_row is None:
                return False, None
            validation_key = self.registry.point_field(
                self.model_point_row, enum_spec.field, enum_labels=enum_spec.allowed_values
            )
            self._used_validated_static_facts[name] = ValidatedStaticFact(
                source_name=name, value=value, allowed_values=enum_spec.allowed_values,
                proof_domain_size=len(self.run_keys), validation_input_key=validation_key,
            )
            return True, value
        if scalar_spec is not None:
            if self.model_point_row is None or isinstance(value, str):
                return False, None
            # Exact-value specialization is safe only when the selector value alone
            # proves every source guard unreachable.  Other runtime values therefore
            # cannot resurrect a discarded raise while this exact selector holds.
            base_fn = _strip_docstring(copy.deepcopy(base))
            local = base_fn.body[0].targets[0].id
            guards = [st for st in base_fn.body[1:-1] if isinstance(st, ast.If)]
            for guard in guards:
                test = _Substitute({local: ast.Constant(value=value)}).visit(copy.deepcopy(guard.test))
                test = _ConservativeFolder().visit(ast.fix_missing_locations(test))
                if not (isinstance(test, ast.Constant) and not bool(test.value)):
                    return False, None
            validation_key = self.registry.point_field(self.model_point_row, scalar_spec.field)
            self._used_validated_static_facts[name] = ValidatedStaticFact(
                source_name=name, value=value, allowed_values=(),
                proof_domain_size=len(self.run_keys), validation_input_key=validation_key,
            )
            return True, value
        self._used_static_facts[("cell", name)] = StaticScalarFact(
            "cell", name, value, len(self.run_keys)
        )
        return True, value

    def _proven_static_reference_value(self, name: str) -> tuple[bool, Any]:
        cached = self._static_reference_value_cache.get(name)
        if cached is not None:
            ok, value = cached
        else:
            if name not in self.refs:
                return False, None
            allowed = (str, bool, int, float, type(None))
            values: list[Any] = []
            # A scalar-looking Space ref is not automatically static: ItemSpace
            # parameters also appear in ``refs`` (for example ``point_id``).  A
            # branch may be pruned only when the reference has the same primitive
            # value on the *complete* frozen RunDomain used by this artifact.
            # This is preparation-time proof, not representative-sample evidence.
            try:
                for key in self.run_keys:
                    item = self._space_instance(key)
                    if name not in item.refs:
                        values = []
                        break
                    value = item.refs[name]
                    if isinstance(value, np.generic):
                        value = value.item()
                    if not isinstance(value, allowed):
                        values = []
                        break
                    values.append(value)
            except Exception:
                values = []
            if not values:
                cached = (False, None)
            else:
                first = values[0]
                try:
                    stable = all(type(v) is type(first) and v == first for v in values[1:])
                except Exception:
                    stable = False
                cached = (bool(stable), first if stable else None)
            self._static_reference_value_cache[name] = cached
            ok, value = cached
        if ok:
            self._used_static_facts[("reference", name)] = StaticScalarFact(
                "reference", name, value, len(self.run_keys)
            )
        return ok, value

    def _proven_numeric_point_field_bounds(
        self, field: str
    ) -> tuple[float | None, float | None]:
        """Return sound finite numeric bounds for one point field over ``run_keys``.

        Unlike representative-trace extrema, these bounds are derived from the exact
        frozen RunDomain used to build the artifact.  Missing/non-numeric/non-finite
        values fail closed.  This is a domain proof only; it never specializes the
        runtime point input to one value.
        """
        if self.model_point_row is None:
            return None, None
        try:
            frame = self.model_point_row.provider()
            if not isinstance(frame, pd.DataFrame) or field not in frame.columns:
                return None, None
            semantic = self.model_point_row.semantic
            pos = int(semantic.key_position)
            if len(self.space_params) > 1:
                rows = [k[pos] if isinstance(k, tuple) else k for k in self.run_keys]
            else:
                rows = list(self.run_keys)
            if not rows:
                return None, None
            values = np.asarray(frame.loc[rows, field])
            if values.ndim != 1 or values.size != len(rows):
                return None, None
            if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
                return None, None
            numeric = values.astype(np.float64, copy=False)
            if not np.all(np.isfinite(numeric)):
                return None, None
            return float(np.min(numeric)), float(np.max(numeric))
        except Exception:
            return None, None

    def _proven_integer_point_field_bounds(
        self, field: str
    ) -> tuple[int | None, int | None]:
        """Return exact integer-valued point-field bounds over the frozen RunDomain.

        Affine correlation is sound only when the source value is genuinely integral,
        not merely when its extrema happen to be whole numbers.  This helper therefore
        validates every selected RunDomain row and fails closed for fractional,
        missing, boolean, object, or non-finite values.
        """
        if self.model_point_row is None:
            return None, None
        try:
            frame = self.model_point_row.provider()
            if not isinstance(frame, pd.DataFrame) or field not in frame.columns:
                return None, None
            semantic = self.model_point_row.semantic
            pos = int(semantic.key_position)
            if len(self.space_params) > 1:
                rows = [k[pos] if isinstance(k, tuple) else k for k in self.run_keys]
            else:
                rows = list(self.run_keys)
            if not rows:
                return None, None
            values = np.asarray(frame.loc[rows, field])
            if values.ndim != 1 or values.size != len(rows):
                return None, None
            if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
                return None, None
            numeric = values.astype(np.float64, copy=False)
            if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.trunc(numeric)):
                return None, None
            return int(np.min(numeric)), int(np.max(numeric))
        except Exception:
            return None, None

    def _proven_static_point_field_value(self, field: str) -> tuple[bool, Any]:
        """Prove one normalized model-point field exact over every build run key.

        Unlike representative-trace specialization, this proof reads the normalized
        point table over the complete artifact run-key domain and retains an explicit
        point input as a runtime equality guard.  It is therefore safe to use only for
        source topology/predicate specialization; arithmetic point inputs are not
        silently baked into generated formulas.
        """
        if self.model_point_row is None:
            return False, None
        cached = self._static_point_field_cache.get(field)
        if cached is None:
            try:
                frame = self.model_point_row.provider()
                if not isinstance(frame, pd.DataFrame) or field not in frame.columns:
                    cached = (False, None)
                else:
                    semantic = self.model_point_row.semantic
                    pos = int(semantic.key_position)
                    if len(self.space_params) > 1:
                        rows = [k[pos] if isinstance(k, tuple) else k for k in self.run_keys]
                    else:
                        rows = list(self.run_keys)
                    values = list(frame.loc[rows, field])
                    if not values:
                        cached = (False, None)
                    else:
                        normalized: list[Any] = []
                        for value in values:
                            if isinstance(value, np.generic):
                                value = value.item()
                            normalized.append(value)
                        first = normalized[0]
                        # Missing values are intentionally not specialized here:
                        # NaN equality is not a valid exact runtime guard.
                        missing = bool(pd.isna(first)) if not isinstance(first, (tuple, list, dict)) else True
                        stable = (
                            not missing
                            and isinstance(first, (str, bool, int, float))
                            and all(type(v) is type(first) and v == first for v in normalized[1:])
                        )
                        cached = (bool(stable), first if stable else None)
            except Exception:
                cached = (False, None)
            self._static_point_field_cache[field] = cached
        ok, value = cached
        if not ok:
            return False, None

        if isinstance(value, str):
            try:
                frame = self.model_point_row.provider()
                labels = tuple(dict.fromkeys(
                    str(v) for v in frame[field].tolist()
                    if isinstance(v, str)
                ))
            except Exception:
                return False, None
            if not labels or len(labels) > 16 or value not in labels:
                return False, None
            validation_key = self.registry.point_field(
                self.model_point_row, field, enum_labels=labels
            )
            allowed_values = labels
        else:
            validation_key = self.registry.point_field(self.model_point_row, field)
            allowed_values = ()
        source_name = f"model_point[{field!r}]"
        self._used_validated_static_facts[source_name] = ValidatedStaticFact(
            source_name=source_name,
            value=value,
            allowed_values=allowed_values,
            proof_domain_size=len(self.run_keys),
            validation_input_key=validation_key,
        )
        return True, value

    def _proven_static_expr(
        self, node: ast.AST, locals_: dict[str, Any] | None = None
    ) -> tuple[bool, Any]:
        """Evaluate a deliberately tiny expression subset from full-domain facts."""
        locals_ = locals_ or {}
        if isinstance(node, ast.Constant):
            value = node.value.item() if isinstance(node.value, np.generic) else node.value
            return (isinstance(value, (str, bool, int, float, type(None))), value)
        if isinstance(node, ast.Name):
            if node.id in locals_ and not isinstance(locals_[node.id], _IntDomain):
                return True, locals_[node.id]
            if node.id in self.refs:
                return self._proven_static_reference_value(node.id)
            return False, None
        if (
            isinstance(node, ast.Subscript)
            and self.model_point_row is not None
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == self.model_point_row.cell_name
            and not node.value.args and not node.value.keywords
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
        ):
            return self._proven_static_point_field_value(str(node.slice.value))
        if isinstance(node, ast.Subscript):
            ok, base = self._proven_static_expr(node.value, locals_)
            if not ok or not isinstance(base, (str, tuple)):
                return False, None
            if isinstance(node.slice, ast.Slice):
                parts = []
                for part in (node.slice.lower, node.slice.upper, node.slice.step):
                    if part is None:
                        parts.append(None)
                        continue
                    ok_part, value_part = self._proven_static_expr(part, locals_)
                    if not ok_part or not isinstance(value_part, int) or isinstance(value_part, bool):
                        return False, None
                    parts.append(int(value_part))
                key = slice(*parts)
            else:
                ok_key, key = self._proven_static_expr(node.slice, locals_)
                if not ok_key or not isinstance(key, (str, int)) or isinstance(key, bool):
                    return False, None
            try:
                value = base[key]
            except Exception:
                return False, None
            if isinstance(value, tuple):
                return True, value
            if isinstance(value, (str, bool, int, float, type(None))):
                return True, value
            return False, None
        if isinstance(node, ast.UnaryOp):
            ok, value = self._proven_static_expr(node.operand, locals_)
            if not ok:
                return False, None
            try:
                if isinstance(node.op, ast.Not): return True, not bool(value)
                if isinstance(node.op, ast.UAdd): return True, +value
                if isinstance(node.op, ast.USub): return True, -value
            except Exception:
                pass
            return False, None
        if isinstance(node, ast.BinOp):
            ok1, left = self._proven_static_expr(node.left, locals_)
            ok2, right = self._proven_static_expr(node.right, locals_)
            if not (ok1 and ok2):
                return False, None
            try:
                if isinstance(node.op, ast.Add): value = left + right
                elif isinstance(node.op, ast.Sub): value = left - right
                elif isinstance(node.op, ast.Mult): value = left * right
                elif isinstance(node.op, ast.Div): value = left / right
                elif isinstance(node.op, ast.FloorDiv): value = left // right
                elif isinstance(node.op, ast.Mod): value = left % right
                elif isinstance(node.op, ast.Pow): value = left ** right
                else: return False, None
                if isinstance(value, (str, bool, int, float)):
                    return True, value
            except Exception:
                pass
            return False, None
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators):
            ok, left = self._proven_static_expr(node.left, locals_)
            if not ok:
                return False, None
            for op, rhs_node in zip(node.ops, node.comparators):
                ok, right = self._proven_static_expr(rhs_node, locals_)
                if not ok:
                    return False, None
                try:
                    if isinstance(op, ast.Lt): verdict = left < right
                    elif isinstance(op, ast.LtE): verdict = left <= right
                    elif isinstance(op, ast.Gt): verdict = left > right
                    elif isinstance(op, ast.GtE): verdict = left >= right
                    elif isinstance(op, ast.Eq): verdict = left == right
                    elif isinstance(op, ast.NotEq): verdict = left != right
                    elif isinstance(op, ast.In): verdict = left in right
                    elif isinstance(op, ast.NotIn): verdict = left not in right
                    elif isinstance(op, (ast.Is, ast.IsNot)):
                        # Identity is only a stable compile-time relation for the
                        # singleton ``None`` object in this proof subset.  String,
                        # numeric and container identity can differ between the
                        # DataFrame/source objects used for proof and the runtime
                        # objects, so folding those would be unsound.
                        if left is not None and right is not None:
                            return False, None
                        verdict = (left is right)
                        if isinstance(op, ast.IsNot):
                            verdict = not verdict
                    else: return False, None
                except Exception:
                    return False, None
                if not verdict:
                    return True, False
                left = right
            return True, True
        if isinstance(node, ast.BoolOp):
            values: list[Any] = []
            for elt in node.values:
                ok, value = self._proven_static_expr(elt, locals_)
                if not ok:
                    return False, None
                values.append(value)
            if isinstance(node.op, ast.And):
                return True, all(bool(v) for v in values)
            if isinstance(node.op, ast.Or):
                return True, any(bool(v) for v in values)
            return False, None
        if isinstance(node, ast.IfExp):
            ok, test = self._proven_static_expr(node.test, locals_)
            if not ok:
                return False, None
            return self._proven_static_expr(node.body if bool(test) else node.orelse, locals_)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            values=[]
            for elt in node.elts:
                ok, value = self._proven_static_expr(elt, locals_)
                if not ok: return False, None
                values.append(value)
            return True, tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if not node.args and not node.keywords and name in self.base_funcs:
                    return self._proven_static_cell_value(name)
                if name in {"bool", "int", "float", "str", "abs", "min", "max"}:
                    vals=[]
                    for arg in node.args:
                        ok, value = self._proven_static_expr(arg, locals_)
                        if not ok: return False, None
                        vals.append(value)
                    if node.keywords: return False, None
                    try:
                        fn={"bool":bool,"int":int,"float":float,"str":str,"abs":abs,"min":min,"max":max}[name]
                        return True, fn(*vals)
                    except Exception:
                        return False, None
                if name == "__is_missing__" and len(node.args) == 1 and not node.keywords:
                    ok, value = self._proven_static_expr(node.args[0], locals_)
                    return (True, bool(pd.isna(value))) if ok else (False, None)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and len(node.args) == 1 and not node.keywords
            ):
                obj = self.refs.get(node.func.value.id)
                module_name = getattr(obj, "__name__", None)
                if (
                    (module_name == "pandas" and node.func.attr in {"isna", "isnull"})
                    or (module_name == "numpy" and node.func.attr == "isnan")
                ):
                    ok, value = self._proven_static_expr(node.args[0], locals_)
                    return (True, bool(pd.isna(value))) if ok else (False, None)
        return False, None

    def _proven_coordinate_cell_constant(self, name: str) -> tuple[bool, Any]:
        """Prove a coordinate Cell constant only when every source return says so.

        This is intentionally stronger than evaluating representative coordinates.
        It is currently restricted to one coordinate parameter and no auxiliary
        parameters.  Every syntactic Return expression must independently reduce to
        the same scalar under guarded full-run static facts.
        """
        cached = self._coordinate_constant_cache.get(name)
        if cached is not None:
            return cached
        base = self.base_funcs.get(name)
        rows = [tr for tr in self.variant_trace_by_uid.values() if tr.schema.name == name]
        positions = {tr.key.time_pos for tr in rows if tr.key.time_pos is not None}
        if base is None or len(positions) != 1 or len(base.args.args) != 1:
            out=(False,None); self._coordinate_constant_cache[name]=out; return out
        returns=[node for node in ast.walk(base) if isinstance(node, ast.Return)]
        values=[]
        for ret in returns:
            if ret.value is None:
                out=(False,None); self._coordinate_constant_cache[name]=out; return out
            ok,value=self._proven_static_expr(ret.value)
            if not ok or not isinstance(value,(bool,int,float)):
                out=(False,None); self._coordinate_constant_cache[name]=out; return out
            values.append(value)
        if not values:
            out=(False,None)
        else:
            first=values[0]
            out=(all(type(v) is type(first) and v == first for v in values[1:]), first)
        self._coordinate_constant_cache[name]=out
        if out[0]:
            self._used_static_facts[("coordinate_cell",name)] = StaticScalarFact(
                "coordinate_cell", name, out[1], len(self.run_keys)
            )
        return out

    def _proven_finite_cell_domain(self, name: str, *, max_items: int = 16) -> tuple[bool, tuple[Any, ...]]:
        uid = self.variants_by_cell_aux.get((name, ()))
        if uid is None:
            return False, ()
        cached = self._finite_variant_domain_cache.get(uid)
        if cached is None:
            tr = self.variant_trace_by_uid.get(uid)
            if tr is None or tr.key.time_pos is not None:
                cached = (False, ())
            else:
                domains: list[tuple[Any, ...]] = []
                allowed = (str, bool, int, float, np.bool_, np.integer, np.floating)
                try:
                    for key in self.run_keys:
                        value = getattr(self._space_instance(key), name)()
                        if not isinstance(value, (tuple, list)) or len(value) > max_items:
                            cached = (False, ())
                            break
                        items: list[Any] = []
                        for item in value:
                            if not isinstance(item, allowed):
                                cached = (False, ())
                                break
                            if isinstance(item, np.generic):
                                item = item.item()
                            items.append(item)
                        else:
                            domains.append(tuple(items))
                            continue
                        break
                    else:
                        if domains and all(domain == domains[0] for domain in domains[1:]):
                            cached = (True, domains[0])
                        else:
                            cached = (False, ())
                except Exception:
                    cached = (False, ())
            self._finite_variant_domain_cache[uid] = cached
        ok, values = cached
        if ok:
            self._used_finite_domain_facts[("cell", name)] = FiniteDomainFact(
                "cell", name, values, len(self.run_keys)
            )
        return ok, values

    @staticmethod
    def _function_bound_names(fn: ast.FunctionDef) -> set[str]:
        names = {arg.arg for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}
        if fn.args.vararg is not None:
            names.add(fn.args.vararg.arg)
        if fn.args.kwarg is not None:
            names.add(fn.args.kwarg.arg)
        for node in ast.walk(fn):
            targets: list[ast.AST] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                if isinstance(node, ast.Assign):
                    targets.extend(node.targets)
                else:
                    targets.append(node.target)
            elif isinstance(node, (ast.For, ast.comprehension)):
                targets.append(node.target)
            for target in targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name):
                        names.add(child.id)
        return names

    def _int_domain_expr(
        self, node: ast.AST, env: dict[str, Any] | None = None, *,
        _stack: tuple[str, ...] = (),
    ) -> _IntDomain | None:
        """Conservatively prove a finite integer interval from source semantics.

        This proof may use exact frozen-RunDomain point-field bounds and may follow
        pure single-return source helpers.  It never uses representative trace
        extrema as semantic bounds.
        """
        env = env or {}
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return _IntDomain(int(node.value), int(node.value))
        if isinstance(node, ast.Name):
            value = env.get(node.id)
            if isinstance(value, _IntDomain):
                return value
            if isinstance(value, int) and not isinstance(value, bool):
                return _IntDomain(int(value), int(value))
            ok, scalar = self._proven_static_expr(node, env)
            if ok and isinstance(scalar, int) and not isinstance(scalar, bool):
                return _IntDomain(int(scalar), int(scalar))
            return None
        ok, scalar = self._proven_static_expr(node, env)
        if ok and isinstance(scalar, int) and not isinstance(scalar, bool):
            return _IntDomain(int(scalar), int(scalar))
        if (
            isinstance(node, ast.Subscript)
            and self.model_point_row is not None
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == self.model_point_row.cell_name
            and not node.value.args and not node.value.keywords
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            lo, hi = self._proven_numeric_point_field_bounds(str(node.slice.value))
            if lo is not None and hi is not None and float(lo).is_integer() and float(hi).is_integer():
                return _IntDomain(int(lo), int(hi))
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            inner = self._int_domain_expr(node.operand, env, _stack=_stack)
            if inner is None:
                return None
            return inner if isinstance(node.op, ast.UAdd) else _IntDomain(-inner.hi, -inner.lo)
        if isinstance(node, ast.BinOp):
            left = self._int_domain_expr(node.left, env, _stack=_stack)
            right = self._int_domain_expr(node.right, env, _stack=_stack)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return _IntDomain(left.lo + right.lo, left.hi + right.hi)
            if isinstance(node.op, ast.Sub):
                return _IntDomain(left.lo - right.hi, left.hi - right.lo)
            if isinstance(node.op, ast.Mult):
                vals = (left.lo * right.lo, left.lo * right.hi, left.hi * right.lo, left.hi * right.hi)
                return _IntDomain(min(vals), max(vals))
            if isinstance(node.op, ast.FloorDiv) and right.singleton not in (None, 0):
                d = int(right.singleton)
                vals = (left.lo // d, left.hi // d)
                return _IntDomain(min(vals), max(vals))
            if isinstance(node.op, ast.Mod) and right.singleton is not None and int(right.singleton) > 0:
                d = int(right.singleton)
                if left.singleton is not None:
                    v = int(left.singleton) % d
                    return _IntDomain(v, v)
                return _IntDomain(0, d - 1)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "int" and len(node.args) == 1 and not node.keywords:
                return self._int_domain_expr(node.args[0], env, _stack=_stack)
            if name in {"min", "max"} and node.args and not node.keywords:
                domains = [self._int_domain_expr(arg, env, _stack=_stack) for arg in node.args]
                if all(d is not None for d in domains):
                    ds = [d for d in domains if d is not None]
                    if name == "min":
                        return _IntDomain(min(d.lo for d in ds), min(d.hi for d in ds))
                    return _IntDomain(max(d.lo for d in ds), max(d.hi for d in ds))
            if name in self.base_funcs and name not in _stack:
                bound = self._bind_source_call(name, node, env)
                if bound is not None:
                    assigned, _time_pos = bound
                    base = _strip_docstring(copy.deepcopy(self.base_funcs[name]))
                    if (
                        not base.args.posonlyargs and not base.args.kwonlyargs
                        and base.args.vararg is None and base.args.kwarg is None
                        and len(base.body) == 1 and isinstance(base.body[0], ast.Return)
                        and base.body[0].value is not None
                    ):
                        value = _Substitute(assigned).visit(copy.deepcopy(base.body[0].value))
                        value = _ConservativeFolder().visit(ast.fix_missing_locations(value))
                        if isinstance(value, ast.AST):
                            return self._int_domain_expr(value, env, _stack=_stack + (name,))
        return None

    _PRIMARY_AFFINE_COORD = "@primary_coordinate"
    _POINT_AFFINE_PREFIX = "@point:"

    def _expand_pure_single_return_call(
        self, node: ast.Call, *, _stack: tuple[str, ...] = ()
    ) -> tuple[ast.AST, str] | None:
        """Inline one pure single-return source/variant call for semantic proofs."""
        if not isinstance(node.func, ast.Name):
            return None
        name = node.func.id
        if name in _stack:
            return None
        source_name: str | None = None
        assigned: dict[str, ast.AST] | None = None
        stack_token = name
        if name in self.variant_trace_by_uid:
            tr = self.variant_trace_by_uid[name]
            source_name = tr.schema.name
            meta = self._pure_single_return_metadata(source_name)
            if meta is None:
                return None
            params, _defaults, _return_expr = meta
            assigned = {}
            if tr.key.time_pos is None:
                if node.args or node.keywords:
                    return None
                aux_iter = iter(tr.key.aux_values)
                try:
                    for param in params:
                        assigned[param] = ast.Constant(next(aux_iter))
                except StopIteration:
                    return None
            else:
                if len(node.args) != 1 or node.keywords:
                    return None
                aux_iter = iter(tr.key.aux_values)
                try:
                    for pos, param in enumerate(params):
                        assigned[param] = (
                            copy.deepcopy(node.args[0])
                            if pos == tr.key.time_pos
                            else ast.Constant(next(aux_iter))
                        )
                except StopIteration:
                    return None
        elif name in self.base_funcs:
            source_name = name
            meta = self._pure_single_return_metadata(source_name)
            if meta is None:
                return None
            params, defaults, _return_expr = meta
            assigned = {}
            if len(node.args) > len(params):
                return None
            for param, arg in zip(params, node.args):
                assigned[param] = copy.deepcopy(arg)
            for kw in node.keywords:
                if kw.arg is None or kw.arg not in params or kw.arg in assigned:
                    return None
                assigned[kw.arg] = copy.deepcopy(kw.value)
            for param in params:
                if param not in assigned:
                    if param not in defaults:
                        return None
                    assigned[param] = copy.deepcopy(defaults[param])
        else:
            return None
        if source_name is None or assigned is None:
            return None
        meta = self._pure_single_return_metadata(source_name)
        if meta is None:
            return None
        _params, _defaults, return_expr = meta
        self._proof_diagnostics["pure_single_return_expansions"] += 1
        value = _Substitute(assigned).visit(copy.deepcopy(return_expr))
        value = _ConservativeFolder().visit(ast.fix_missing_locations(value))
        return (value, stack_token) if isinstance(value, ast.AST) else None

    def _pure_single_return_metadata(
        self, source_name: str
    ) -> tuple[tuple[str, ...], dict[str, ast.AST], ast.AST] | None:
        """Return immutable metadata for a source helper eligible for proof inlining.

        The old proof path deep-copied and stripped the complete FunctionDef on
        every query, often twice per call.  Large actuarial formulas make that
        surprisingly expensive.  Cache only source metadata and still deepcopy
        the return expression before substitution, so AST mutation semantics stay
        exactly as conservative as before.
        """
        if source_name in self._pure_single_return_meta_cache:
            self._proof_diagnostics["pure_single_return_meta_hits"] += 1
            return self._pure_single_return_meta_cache[source_name]
        self._proof_diagnostics["pure_single_return_meta_misses"] += 1
        base = self.base_funcs.get(source_name)
        if not isinstance(base, ast.FunctionDef):
            self._pure_single_return_meta_cache[source_name] = None
            return None
        body = [
            st for st in base.body
            if not (
                isinstance(st, ast.Expr)
                and isinstance(st.value, ast.Constant)
                and isinstance(st.value.value, str)
            )
        ]
        if (
            base.args.posonlyargs or base.args.kwonlyargs
            or base.args.vararg is not None or base.args.kwarg is not None
            or len(body) != 1 or not isinstance(body[0], ast.Return)
            or body[0].value is None
        ):
            self._pure_single_return_meta_cache[source_name] = None
            return None
        params = tuple(a.arg for a in base.args.args)
        defaults: dict[str, ast.AST] = {}
        if base.args.defaults:
            for param, default in zip(params[-len(base.args.defaults):], base.args.defaults):
                defaults[param] = default
        out = (params, defaults, body[0].value)
        self._pure_single_return_meta_cache[source_name] = out
        return out

    def proof_diagnostics(self) -> dict[str, int]:
        """Preparation-only counters for semantic proof work."""
        return dict(self._proof_diagnostics)

    def _affine_int_expr(
        self, node: ast.AST, env: dict[str, Any] | None = None, *,
        _stack: tuple[str, ...] = (),
    ) -> AffineIntForm | None:
        """Normalize a deliberately small integer-affine source subset."""
        env = env or {}
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return AffineIntForm.scalar(int(node.value))
        if isinstance(node, ast.Name):
            value = env.get(node.id)
            if isinstance(value, AffineIntForm):
                return value
            if isinstance(value, int) and not isinstance(value, bool):
                return AffineIntForm.scalar(int(value))
            ok, scalar = self._proven_static_expr(node, env)
            if ok and isinstance(scalar, int) and not isinstance(scalar, bool):
                return AffineIntForm.scalar(int(scalar))
            return None
        ok, scalar = self._proven_static_expr(node, env)
        if ok and isinstance(scalar, int) and not isinstance(scalar, bool):
            return AffineIntForm.scalar(int(scalar))
        if (
            isinstance(node, ast.Subscript)
            and self.model_point_row is not None
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == self.model_point_row.cell_name
            and not node.value.args and not node.value.keywords
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            field = str(node.slice.value)
            lo, hi = self._proven_integer_point_field_bounds(field)
            if lo is None or hi is None:
                return None
            return AffineIntForm.symbol(self._POINT_AFFINE_PREFIX + field)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            inner = self._affine_int_expr(node.operand, env, _stack=_stack)
            if inner is None:
                return None
            return inner if isinstance(node.op, ast.UAdd) else inner.scale(-1)
        if isinstance(node, ast.BinOp):
            left = self._affine_int_expr(node.left, env, _stack=_stack)
            right = self._affine_int_expr(node.right, env, _stack=_stack)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left.add(right)
            if isinstance(node.op, ast.Sub):
                return left.sub(right)
            if isinstance(node.op, ast.Mult):
                if not left.terms:
                    return right.scale(left.constant)
                if not right.terms:
                    return left.scale(right.constant)
                return None
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "int" and len(node.args) == 1 and not node.keywords:
                # Preserve only already-integral affine inputs.  General truncation is
                # nonlinear and intentionally outside this proof subset.
                return self._affine_int_expr(node.args[0], env, _stack=_stack)
            expanded = self._expand_pure_single_return_call(node, _stack=_stack)
            if expanded is not None:
                value, token = expanded
                return self._affine_int_expr(value, env, _stack=_stack + (token,))
        return None

    def _affine_symbol_bounds_many(
        self, *forms: AffineIntForm | None
    ) -> dict[str, tuple[int, int]] | None:
        out: dict[str, tuple[int, int]] = {}
        for form in forms:
            if form is None:
                continue
            for symbol, _coeff in form.terms:
                if symbol == self._PRIMARY_AFFINE_COORD or symbol in out:
                    continue
                if not symbol.startswith(self._POINT_AFFINE_PREFIX):
                    return None
                field = symbol[len(self._POINT_AFFINE_PREFIX):]
                lo, hi = self._proven_integer_point_field_bounds(field)
                if lo is None or hi is None:
                    return None
                out[symbol] = (int(lo), int(hi))
        return out

    def _range_affine_constraint_from_args(
        self, args: tuple[ast.AST, ...], env: dict[str, Any] | None = None
    ) -> AffineRangeConstraint | None:
        """Freeze affine coordinate inequalities for semantic proofs only."""
        env = env or {}
        if len(args) == 1:
            start_node, stop_node, step_node = ast.Constant(0), args[0], ast.Constant(1)
        elif len(args) == 2:
            start_node, stop_node, step_node = args[0], args[1], ast.Constant(1)
        elif len(args) == 3:
            start_node, stop_node, step_node = args
        else:
            return None
        step = self._int_domain_expr(step_node, env)
        if step is None or step.singleton is None or int(step.singleton) == 0:
            return None
        start = self._affine_int_expr(start_node, env)
        stop = self._affine_int_expr(stop_node, env)
        if start is None or stop is None:
            return None
        step_value = int(step.singleton)
        if step_value > 0:
            lower = start
            upper = stop.sub(AffineIntForm.scalar(1))
        else:
            lower = stop.add(AffineIntForm.scalar(1))
            upper = start
        return AffineRangeConstraint(self._PRIMARY_AFFINE_COORD, lower=lower, upper=upper)

    def _bound_affine_under_primary_range(self, form: AffineIntForm) -> _IntDomain | None:
        constraint = self.primary_coordinate_affine_constraint
        if constraint is None:
            return None
        symbol_bounds = self._affine_symbol_bounds_many(form, constraint.lower, constraint.upper)
        if symbol_bounds is None:
            return None
        lo, hi = affine_bounds_under_range(form, constraint, symbol_bounds)
        if lo is None or hi is None or int(lo) > int(hi):
            return None
        return _IntDomain(int(lo), int(hi))

    def _correlated_int_domain_expr(
        self, node: ast.AST, *, coordinate_name: str = "t", _stack: tuple[str, ...] = ()
    ) -> _IntDomain | None:
        """Bound a small integer expression while preserving primary-range correlation."""
        env = {coordinate_name: AffineIntForm.symbol(self._PRIMARY_AFFINE_COORD)}
        affine = self._affine_int_expr(node, env, _stack=_stack)
        if affine is not None:
            bounded = self._bound_affine_under_primary_range(affine)
            if bounded is not None:
                return bounded
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            inner = self._correlated_int_domain_expr(node.operand, coordinate_name=coordinate_name, _stack=_stack)
            if inner is None:
                return None
            return inner if isinstance(node.op, ast.UAdd) else _IntDomain(-inner.hi, -inner.lo)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.FloorDiv):
                right = self._int_domain_expr(node.right)
                if right is None or right.singleton is None or int(right.singleton) <= 0:
                    return None
                numerator = self._affine_int_expr(node.left, env, _stack=_stack)
                left = (
                    self._bound_affine_under_primary_range(numerator)
                    if numerator is not None else
                    self._correlated_int_domain_expr(node.left, coordinate_name=coordinate_name, _stack=_stack)
                )
                if left is None:
                    return None
                d = int(right.singleton)
                return _IntDomain(left.lo // d, left.hi // d)
            left = self._correlated_int_domain_expr(node.left, coordinate_name=coordinate_name, _stack=_stack)
            right = self._correlated_int_domain_expr(node.right, coordinate_name=coordinate_name, _stack=_stack)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return _IntDomain(left.lo + right.lo, left.hi + right.hi)
            if isinstance(node.op, ast.Sub):
                return _IntDomain(left.lo - right.hi, left.hi - right.lo)
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "int" and len(node.args) == 1 and not node.keywords:
                return self._correlated_int_domain_expr(node.args[0], coordinate_name=coordinate_name, _stack=_stack)
            expanded = self._expand_pure_single_return_call(node, _stack=_stack)
            if expanded is not None:
                value, token = expanded
                return self._correlated_int_domain_expr(
                    value, coordinate_name=coordinate_name, _stack=_stack + (token,)
                )
        return None

    @staticmethod
    def _compare_domain_truth(left: _IntDomain, op: ast.cmpop, right: _IntDomain) -> bool | None:
        if isinstance(op, ast.Lt):
            if left.hi < right.lo: return True
            if left.lo >= right.hi: return False
        elif isinstance(op, ast.LtE):
            if left.hi <= right.lo: return True
            if left.lo > right.hi: return False
        elif isinstance(op, ast.Gt):
            if left.lo > right.hi: return True
            if left.hi <= right.lo: return False
        elif isinstance(op, ast.GtE):
            if left.lo >= right.hi: return True
            if left.hi < right.lo: return False
        elif isinstance(op, ast.Eq):
            if left.singleton is not None and right.singleton is not None:
                return left.singleton == right.singleton
            if left.hi < right.lo or right.hi < left.lo: return False
        elif isinstance(op, ast.NotEq):
            if left.hi < right.lo or right.hi < left.lo: return True
            if left.singleton is not None and right.singleton is not None:
                return left.singleton != right.singleton
        return None

    def _domain_truth(self, node: ast.AST, env: dict[str, Any] | None = None) -> bool | None:
        env=env or {}
        ok,value=self._proven_static_expr(node,env)
        if ok:
            return bool(value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            val=self._domain_truth(node.operand,env)
            return None if val is None else not val
        if isinstance(node, ast.BoolOp):
            vals=[self._domain_truth(v,env) for v in node.values]
            if isinstance(node.op,ast.And):
                if any(v is False for v in vals): return False
                if all(v is True for v in vals): return True
            elif isinstance(node.op,ast.Or):
                if any(v is True for v in vals): return True
                if all(v is False for v in vals): return False
            return None
        if isinstance(node, ast.Compare) and len(node.ops)==1 and len(node.comparators)==1:
            left=self._int_domain_expr(node.left,env); right=self._int_domain_expr(node.comparators[0],env)
            if left is not None and right is not None:
                return self._compare_domain_truth(left,node.ops[0],right)
        return None

    def _refine_domain_env(self, node: ast.AST, env: dict[str, Any], truth: bool) -> dict[str, Any] | None:
        """Refine one simple comparison; unknown predicates keep the input domain."""
        out=dict(env)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op,ast.Not):
            return self._refine_domain_env(node.operand,env,not truth)
        if isinstance(node,ast.BoolOp):
            if isinstance(node.op,ast.And) and truth:
                cur=out
                for value in node.values:
                    cur=self._refine_domain_env(value,cur,True)
                    if cur is None:return None
                return cur
            if isinstance(node.op,ast.Or) and not truth:
                cur=out
                for value in node.values:
                    cur=self._refine_domain_env(value,cur,False)
                    if cur is None:return None
                return cur
            return out
        if not (isinstance(node,ast.Compare) and len(node.ops)==1 and len(node.comparators)==1):
            return out
        left,right=node.left,node.comparators[0]; op=node.ops[0]
        if isinstance(right,ast.Name) and not isinstance(left,ast.Name):
            # Reverse c OP x to x reverse(OP) c.
            rev={ast.Lt:ast.Gt(),ast.LtE:ast.GtE(),ast.Gt:ast.Lt(),ast.GtE:ast.LtE(),ast.Eq:ast.Eq(),ast.NotEq:ast.NotEq()}
            op=rev.get(type(op),op); left,right=right,left
        if not (isinstance(left,ast.Name) and isinstance(out.get(left.id),_IntDomain)):
            return out
        rhs=self._int_domain_expr(right,out)
        if rhs is None or rhs.singleton is None:
            return out
        d=out[left.id]; c=int(rhs.singleton)
        if not truth:
            inv={ast.Lt:ast.GtE(),ast.LtE:ast.Gt(),ast.Gt:ast.LtE(),ast.GtE:ast.Lt(),ast.Eq:ast.NotEq(),ast.NotEq:ast.Eq()}
            op=inv.get(type(op),op)
        if isinstance(op,ast.Lt): nd=d.intersect(hi=c-1)
        elif isinstance(op,ast.LtE): nd=d.intersect(hi=c)
        elif isinstance(op,ast.Gt): nd=d.intersect(lo=c+1)
        elif isinstance(op,ast.GtE): nd=d.intersect(lo=c)
        elif isinstance(op,ast.Eq): nd=d.intersect(lo=c,hi=c)
        elif isinstance(op,ast.NotEq):
            if d.singleton==c:return None
            # An interval cannot represent an interior hole, but excluding one
            # *boundary* value is exact.  This matters for source shapes such as
            # ``if t == 0: return ...``: the fall-through path is then provably
            # ``t >= 1`` rather than the old coarse ``t >= 0``.
            if c == d.lo:
                nd=d.intersect(lo=c+1)
            elif c == d.hi:
                nd=d.intersect(hi=c-1)
            else:
                return out
        else:return out
        if nd is None:return None
        out[left.id]=nd; return out

    def _specialize_static_facts_and_domains(
        self, fn: ast.FunctionDef, *, coordinate_env: dict[str, Any] | None = None
    ) -> ast.FunctionDef:
        """Prune control flow and expand tiny domains using full-run-key proofs.

        Scalar facts are intentionally substituted only in branch predicates.  A
        stable observed arithmetic value is *not* enough reason to turn a normal
        point input into a compile-time constant.  This keeps the native ABI honest
        while still removing source branches that cannot execute in this artifact's
        proven graph family.
        """
        frontend = self
        bound = self._function_bound_names(fn)
        coordinate_env = dict(coordinate_env or {})

        # A very small local alias form is useful for selectors written as
        # ``family = mva_family(); if family == ...``.  Only top-level, single-
        # assignment aliases are eligible; conditionally reassigned numeric locals
        # remain runtime values (the dev13 table tests enforce that distinction).
        assign_count: dict[str, int] = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assign_count[target.id] = assign_count.get(target.id, 0) + 1
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                assign_count[node.target.id] = assign_count.get(node.target.id, 0) + 1
        local_static: dict[str, Any] = {}
        for st in fn.body:
            if not (
                isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and assign_count.get(st.targets[0].id) == 1
            ):
                continue
            target = st.targets[0].id
            rhs = st.value
            ok, value = self._proven_static_expr(rhs, local_static)
            if ok:
                local_static[target] = value

        class PredicateFacts(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call):
                # A coordinate Cell can be substituted in a predicate only when
                # every source return is independently the same full-domain scalar.
                source_name = None
                if isinstance(node.func, ast.Name):
                    if node.func.id in frontend.base_funcs:
                        source_name = node.func.id
                    else:
                        source_name = frontend.formula_uid_to_source_name.get(node.func.id)
                if source_name in frontend.base_funcs:
                    ok,value=frontend._proven_coordinate_cell_constant(source_name)
                    if ok:
                        return ast.copy_location(ast.Constant(value=value),node)
                if (
                    isinstance(node.func, ast.Name)
                    and not node.args and not node.keywords
                    and source_name in frontend.base_funcs
                ):
                    ok, value = frontend._proven_static_cell_value(source_name)
                    if ok:
                        return ast.copy_location(ast.Constant(value=value), node)
                return self.generic_visit(node)

            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Load) and node.id in local_static:
                    return ast.copy_location(ast.Constant(value=local_static[node.id]), node)
                if isinstance(node.ctx, ast.Load) and node.id not in bound and node.id in frontend.refs:
                    ok, value = frontend._proven_static_reference_value(node.id)
                    if ok:
                        return ast.copy_location(ast.Constant(value=value), node)
                return node

        facts = PredicateFacts()

        class Specialize(ast.NodeTransformer):
            def _test(self, node: ast.AST) -> ast.AST:
                out = facts.visit(copy.deepcopy(node))
                out = _ConservativeFolder().visit(ast.fix_missing_locations(out))
                verdict=frontend._domain_truth(out,{**coordinate_env,**local_static})
                if verdict is not None:
                    out=ast.copy_location(ast.Constant(bool(verdict)),out)
                return ast.fix_missing_locations(out)

            def visit_If(self, node: ast.If):
                node = copy.deepcopy(node)
                node.test = self._test(node.test)
                # generic_visit owns statement-list flattening when a nested static
                # branch itself disappears.  Do not manually append a returned list.
                node = self.generic_visit(node)
                if isinstance(node.test, ast.Constant):
                    return node.body if bool(node.test.value) else node.orelse
                return node

            def visit_IfExp(self, node: ast.IfExp):
                node = copy.deepcopy(node)
                node.test = self._test(node.test)
                node = self.generic_visit(node)
                if isinstance(node.test, ast.Constant):
                    return node.body if bool(node.test.value) else node.orelse
                return node

            def visit_GeneratorExp(self, node: ast.GeneratorExp):
                node = copy.deepcopy(node)
                for comp in node.generators:
                    iterator = comp.iter
                    if (
                        isinstance(iterator, ast.Call)
                        and isinstance(iterator.func, ast.Name)
                        and not iterator.args and not iterator.keywords
                        and iterator.func.id in frontend.base_funcs
                    ):
                        ok, values = frontend._proven_finite_cell_domain(iterator.func.id)
                        if ok:
                            comp.iter = ast.Tuple(
                                elts=[ast.Constant(value=value) for value in values],
                                ctx=ast.Load(),
                            )
                return self.generic_visit(node)

        out = Specialize().visit(copy.deepcopy(fn))
        assert isinstance(out, ast.FunctionDef)
        out = _ConservativeFolder().visit(ast.fix_missing_locations(out))
        assert isinstance(out, ast.FunctionDef)

        class StaticReferenceValues(ast.NodeTransformer):
            """Materialize proven non-numeric configuration refs outside predicates.

            Numeric/bool refs already have a typed ``global_input`` ABI in the
            ordinary expression lowerer. String/None configuration refs do not,
            and leaving them in an otherwise-native formula would surface later as
            an unresolved Python name (for example IUL ``seg_credit_base_conv``).
            Only complete-RunDomain-proven refs are replaced, and function/local
            bindings continue to shadow Space refs exactly as in source Python.
            """

            def visit_Name(self, node: ast.Name):  # noqa: N802
                if not (
                    isinstance(node.ctx, ast.Load)
                    and node.id not in bound
                    and node.id in frontend.refs
                ):
                    return node
                ok, value = frontend._proven_static_reference_value(node.id)
                if ok and (isinstance(value, str) or value is None):
                    return ast.copy_location(ast.Constant(value=value), node)
                return node

        out = StaticReferenceValues().visit(out)
        assert isinstance(out, ast.FunctionDef)
        out = _ConservativeFolder().visit(ast.fix_missing_locations(out))
        assert isinstance(out, ast.FunctionDef)
        out = unroll_finite_literal_generators(out)
        out = _ConservativeFolder().visit(ast.fix_missing_locations(out))
        assert isinstance(out, ast.FunctionDef)

        def strip_dead_tail(stmts: list[ast.stmt]) -> list[ast.stmt]:
            kept: list[ast.stmt] = []
            for st in stmts:
                if isinstance(st, ast.If):
                    st.body = strip_dead_tail(list(st.body))
                    st.orelse = strip_dead_tail(list(st.orelse))
                kept.append(st)
                if isinstance(st, (ast.Return, ast.Raise)):
                    break
            return kept

        out.body = strip_dead_tail(list(out.body))

        class StaticRoundPrecision(ast.NodeTransformer):
            """Specialize only the precision argument of ``round`` from guards.

            Normal arithmetic values remain runtime inputs: the dev29 contract is
            deliberately *not* changed into wholesale constant propagation.  The
            precision parameter is different because Cython needs it structurally
            static to preserve Python's decimal-rounding semantics without a dynamic
            fallback.  Full-run proof/guards are therefore consumed only at this
            call-site position.
            """

            def visit_Call(self, node: ast.Call):
                node = self.generic_visit(node)
                if not (
                    isinstance(node.func, ast.Name) and node.func.id == "round"
                    and len(node.args) == 2 and not node.keywords
                ):
                    return node
                ok, value = frontend._proven_static_expr(node.args[1], local_static)
                if not ok:
                    return node
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                    raise FrontendError(
                        "proven-static round ndigits must be an integer"
                    )
                return ast.copy_location(
                    ast.Call(
                        func=copy.deepcopy(node.func),
                        args=[node.args[0], ast.Constant(int(value))],
                        keywords=[],
                    ),
                    node,
                )

        out = StaticRoundPrecision().visit(out)
        assert isinstance(out, ast.FunctionDef)
        out = ast.fix_missing_locations(out)
        # Static branch folding can create literal locals that were not literal in
        # source (PA's kind = "A" if static_selector() else "B").  Propagate only
        # single-assignment literal locals, never arithmetic/runtime locals.
        counts: dict[str,int]={}
        literal_locals: dict[str,Any]={}
        for node in ast.walk(out):
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):
                counts[node.id]=counts.get(node.id,0)+1
        for st in out.body:
            if (
                isinstance(st,ast.Assign) and len(st.targets)==1 and isinstance(st.targets[0],ast.Name)
                and counts.get(st.targets[0].id)==1 and isinstance(st.value,ast.Constant)
            ):
                literal_locals[st.targets[0].id]=st.value.value
        if literal_locals:
            out=_Substitute({k:ast.Constant(v) for k,v in literal_locals.items()}).visit(out)
            assert isinstance(out,ast.FunctionDef)
            out=ast.fix_missing_locations(out)
        if local_static:
            loads = {
                node.id for node in ast.walk(out)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            out.body = [
                st for st in out.body
                if not (
                    isinstance(st, ast.Assign) and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name)
                    and st.targets[0].id in local_static
                    and st.targets[0].id not in loads
                )
            ]
        return ast.fix_missing_locations(out)

    def _range_partial_bounds_from_args(
        self, args: tuple[ast.AST, ...], env: dict[str, Any] | None = None
    ) -> tuple[int | None, int | None]:
        """Return sound one-sided bounds for values yielded by ``range``.

        Unlike exact physical range resolution, this retains safe bounds when a
        start/stop varies over the frozen RunDomain.  It is used only for semantic
        domain proofs, never to choose physical scheduling geometry.
        """
        env = env or {}
        if len(args) == 1:
            start_node, stop_node, step_node = ast.Constant(0), args[0], ast.Constant(1)
        elif len(args) == 2:
            start_node, stop_node, step_node = args[0], args[1], ast.Constant(1)
        elif len(args) == 3:
            start_node, stop_node, step_node = args
        else:
            return None, None
        start = self._int_domain_expr(start_node, env)
        stop = self._int_domain_expr(stop_node, env)
        step = self._int_domain_expr(step_node, env)
        if step is None or step.singleton is None or int(step.singleton) == 0:
            return None, None
        step_value = int(step.singleton)
        if step_value > 0:
            lo = None if start is None else int(start.lo)
            hi = None if stop is None else int(stop.hi) - 1
            return lo, hi
        lo = None if stop is None else int(stop.lo) + 1
        hi = None if start is None else int(start.hi)
        return lo, hi

    def _range_domain_from_args(
        self, args: tuple[ast.AST, ...], env: dict[str, Any] | None = None
    ) -> tuple[_IntDomain, int] | None:
        """Resolve a Python ``range`` to one exact finite integer interval."""
        env=env or {}
        vals=[]
        for arg in args:
            dom=self._int_domain_expr(arg,env)
            if dom is None or dom.singleton is None:
                return None
            vals.append(int(dom.singleton))
        if len(vals)==1:
            start,stop,step=0,vals[0],1
        elif len(vals)==2:
            start,stop,step=vals[0],vals[1],1
        elif len(vals)==3:
            start,stop,step=vals
        else:
            return None
        if step==0:
            return None
        seq=range(start,stop,step)
        if len(seq)==0:
            return None
        first=seq[0]; last=seq[-1]
        return _IntDomain(min(first,last),max(first,last)), int(step)

    def _coordinate_time_position(self, name: str) -> int | None:
        rows=[tr for tr in self.variant_trace_by_uid.values() if tr.schema.name==name and tr.key.time_pos is not None]
        positions={tr.key.time_pos for tr in rows}
        return next(iter(positions)) if len(positions)==1 else None

    def _bind_source_call(
        self, name: str, call: ast.Call, env: dict[str, Any]
    ) -> tuple[dict[str, ast.AST], int | None] | None:
        base=self.base_funcs.get(name)
        if base is None:
            return None
        params=[a.arg for a in base.args.args]
        defaults={}
        if base.args.defaults:
            for p,d in zip(params[-len(base.args.defaults):],base.args.defaults): defaults[p]=d
        assigned={p:a for p,a in zip(params,call.args)}
        for kw in call.keywords:
            if kw.arg is None or kw.arg in assigned:
                return None
            assigned[kw.arg]=kw.value
        for p in params:
            if p not in assigned:
                if p not in defaults:return None
                assigned[p]=defaults[p]
        return assigned,self._coordinate_time_position(name)

    @staticmethod
    def _merge_domain_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not states:
            return []
        if len(states)==1:
            return states
        keys=set.intersection(*(set(st) for st in states)) if states else set()
        merged: dict[str,Any]={}
        for key in keys:
            vals=[st[key] for st in states]
            if all(isinstance(v,_IntDomain) for v in vals):
                dom=vals[0]
                for value in vals[1:]:dom=dom.union(value)
                merged[key]=dom
            else:
                first=vals[0]
                try:same=all(type(v) is type(first) and v==first for v in vals[1:])
                except Exception:same=False
                if same:merged[key]=first
        return [merged]

    def _derive_coordinate_domains(self) -> dict[str, _IntDomain]:
        """Propagate finite source-proven coordinate intervals from output demand.

        Two root shapes are admitted:

        * a normalized reduction, whose exact ``range`` remains the physical root;
        * a fixed OutputInvocation, whose literal/static arguments seed demand into
          scheduled coordinate families called by the output formula.

        The latter is essential for finite reverse recurrences such as ``f(0)`` with
        source-guarded ``f(t + 1)``.  No observed trace extrema are consumed: trace
        metadata identifies coordinate/auxiliary roles only; bounds come from the
        declared OutputInvocation, source control flow, run-domain validated static
        facts and exact integer algebra.
        """
        output_matches=[
            uid for uid,tr in self.variant_trace_by_uid.items()
            if tr.schema.name==self.output
            and tr.key.time_pos is None
            and tuple(tr.key.aux_values)==tuple(self.output_invocation.argument_values)
            and tr.schema.name in self.base_funcs
        ]
        if len(output_matches)!=1:
            return {}
        output_uid=output_matches[0]
        output_fn=_strip_docstring(copy.deepcopy(self.base_funcs[self.output]))
        output_fn=self._specialize_static_facts_and_domains(output_fn)
        reduction=self._normalize_reduction(output_fn)
        root_domain: _IntDomain | None = None
        fixed_output_env: dict[str, Any] | None = None
        if reduction is not None:
            self.primary_coordinate_affine_constraint = self._range_affine_constraint_from_args(
                reduction.range_args
            )
            self.primary_coordinate_bounds = self._range_partial_bounds_from_args(reduction.range_args)
            root=self._range_domain_from_args(reduction.range_args)
            if root is None:
                return {}
            root_domain,_root_step=root
            # The root reduction range is an exact physical-loop domain even when
            # the wider inter-Cell fixed point later exceeds its proof budget.
            self.primary_coordinate_domain = root_domain
            self.primary_coordinate_bounds = (int(root_domain.lo), int(root_domain.hi))
        else:
            params=[a.arg for a in output_fn.args.args]
            values=tuple(self.output_invocation.argument_values)
            if len(params)!=len(values):
                return {}
            fixed_output_env={}
            for name,value in zip(params,values):
                if isinstance(value, np.generic):
                    value=value.item()
                if isinstance(value, bool):
                    fixed_output_env[name]=value
                elif isinstance(value, int):
                    fixed_output_env[name]=_IntDomain(int(value),int(value))
                elif isinstance(value, (float,str,type(None))):
                    fixed_output_env[name]=value
                else:
                    return {}

        domains: dict[str,_IntDomain]={}
        pending: list[str]=[]

        def add_domain(uid: str, dom: _IntDomain):
            old=domains.get(uid)
            new=dom if old is None else old.union(dom)
            if old is None or new!=old:
                domains[uid]=new
                if uid not in pending:pending.append(uid)

        def record_call(call: ast.Call, env: dict[str,Any]):
            if not (isinstance(call.func,ast.Name) and call.func.id in self.base_funcs):
                return
            name=call.func.id
            bound=self._bind_source_call(name,call,env)
            if bound is None:return
            assigned,tpos=bound
            if tpos is None:return
            params=[a.arg for a in self.base_funcs[name].args.args]
            aux=[]
            for j,p in enumerate(params):
                if j==tpos:continue
                ok,value=self._proven_static_expr(assigned[p],env)
                if not ok:return
                aux.append(value)
            uid=self.variants_by_cell_aux.get((name,tuple(aux)))
            if uid is None:return
            dom=self._int_domain_expr(assigned[params[tpos]],env)
            if dom is not None:add_domain(uid,dom)

        def scan_expr(node: ast.AST | None, env: dict[str,Any]):
            if node is None:return
            if isinstance(node,ast.IfExp):
                scan_expr(node.test,env)
                verdict=self._domain_truth(node.test,env)
                if verdict is True:
                    scan_expr(node.body,self._refine_domain_env(node.test,env,True) or env)
                elif verdict is False:
                    scan_expr(node.orelse,self._refine_domain_env(node.test,env,False) or env)
                else:
                    a=self._refine_domain_env(node.test,env,True)
                    b=self._refine_domain_env(node.test,env,False)
                    if a is not None:scan_expr(node.body,a)
                    if b is not None:scan_expr(node.orelse,b)
                return
            if isinstance(node,ast.Call):
                for arg in node.args:scan_expr(arg,env)
                for kw in node.keywords:scan_expr(kw.value,env)
                record_call(node,env)
                return
            for child in ast.iter_child_nodes(node):
                scan_expr(child,env)

        def scan_block(stmts: list[ast.stmt], initial: dict[str,Any]) -> list[dict[str,Any]]:
            states=[dict(initial)]
            for st in stmts:
                next_states=[]
                for env in states:
                    if isinstance(st,ast.Return):
                        scan_expr(st.value,env); continue
                    if isinstance(st,ast.Raise):
                        scan_expr(st.exc,env); continue
                    if isinstance(st,ast.Assign) and len(st.targets)==1 and isinstance(st.targets[0],ast.Name):
                        scan_expr(st.value,env)
                        nxt=dict(env); name=st.targets[0].id
                        ok,value=self._proven_static_expr(st.value,env)
                        if ok:nxt[name]=value
                        else:
                            dom=self._int_domain_expr(st.value,env)
                            if dom is not None:nxt[name]=dom
                            else:nxt.pop(name,None)
                        next_states.append(nxt); continue
                    if isinstance(st,ast.If):
                        scan_expr(st.test,env)
                        verdict=self._domain_truth(st.test,env)
                        branches=[]
                        if verdict is not False:
                            tenv=self._refine_domain_env(st.test,env,True)
                            if tenv is not None:branches.extend(scan_block(list(st.body),tenv))
                        if verdict is not True:
                            fenv=self._refine_domain_env(st.test,env,False)
                            if fenv is not None:
                                branches.extend(scan_block(list(st.orelse),fenv) if st.orelse else [fenv])
                        next_states.extend(branches); continue
                    if isinstance(st,ast.Expr):
                        scan_expr(st.value,env); next_states.append(env); continue
                    if isinstance(st,ast.AugAssign):
                        scan_expr(st.value,env); scan_expr(st.target,env)
                        nxt=dict(env)
                        if isinstance(st.target,ast.Name):nxt.pop(st.target.id,None)
                        next_states.append(nxt); continue
                    # Local imperative constructs are outside this proof foundation;
                    # inspect their expressions conservatively but do not infer bounds.
                    for child in ast.iter_child_nodes(st):scan_expr(child,env)
                    next_states.append(env)
                states=self._merge_domain_states(next_states)
                if not states:break
            return states

        if reduction is not None:
            assert root_domain is not None
            root_env={reduction.loop_var:root_domain}
            for filt in reduction.filters:scan_expr(filt,root_env)
            scan_expr(reduction.body_expr,root_env)
        else:
            assert fixed_output_env is not None
            scan_block(list(output_fn.body), fixed_output_env)

        iterations=0
        while pending and iterations<4096:
            iterations+=1
            uid=pending.pop(0); tr=self.variant_trace_by_uid[uid]; dom=domains[uid]
            base=copy.deepcopy(self.base_funcs.get(tr.schema.name))
            if base is None:continue
            params=[a.arg for a in base.args.args]; tpos=tr.key.time_pos
            if tpos is None:continue
            env: dict[str,Any]={params[tpos]:dom}
            aux_iter=iter(tr.key.aux_values)
            for j,p in enumerate(params):
                if j!=tpos:env[p]=next(aux_iter)
            scan_block(list(_strip_docstring(base).body),env)

        if pending:
            # A partial fixed point is an under-approximation and therefore must
            # never authorize branch deletion.  If a pathological source graph
            # does not converge within the bounded proof budget, discard the
            # entire coordinate-domain proof rather than using narrower domains.
            self.coordinate_domain_notes = [
                "source-proven coordinate-domain analysis abandoned: fixed-point budget exceeded"
            ]
            return {}

        self.coordinate_domain_notes=[
            f"source-proven coordinate domain {self.variant_trace_by_uid[uid].schema.fullname}: [{dom.lo}, {dom.hi}]"
            for uid,dom in sorted(domains.items())
        ]
        return domains

    def _derive_specialized_coordinate_domains(
        self,
        specialized: dict[str, tuple[Any, ast.FunctionDef, str | None, str | None]],
        output_uid: str,
    ) -> dict[str, _IntDomain]:
        """Prove coordinate intervals on the specialized demand graph.

        The older source-wide prepass runs before demand specialization and is
        intentionally conservative.  In particular it cannot always retain the
        path precision created later by literal auxiliary substitution,
        source-backed scheduled variants and dead-tail removal.  Lookup-domain
        safety needs the *callee call domain*, not merely the physical loop range.

        This pass therefore runs after all demanded source variants have been
        specialized but before table lowering.  Calls already name canonical UIDs,
        so no representative observed coordinate values are needed.  Source
        control flow and exact finite ``range`` semantics are the only reachability
        evidence.  If the finite-domain fixed point does not converge within the
        proof budget, the entire result is discarded rather than using an unsafe
        under-approximation.
        """
        row = specialized.get(output_uid)
        if row is None:
            return {}
        _tr, output_fn, _forced_role, _time_param_source = row
        reduction = self._normalize_reduction(copy.deepcopy(output_fn))
        if reduction is None:
            # Fixed OutputInvocation roots have no physical reduction range, but
            # the earlier source-demand fixed point can still prove exact finite
            # call domains for scheduled coordinate families.  Reuse only facts
            # for UIDs that survived demand specialization; this remains source /
            # invocation evidence and never imports representative trace extrema.
            return {
                uid: dom for uid, dom in self.coordinate_domains.items()
                if uid in specialized
            }
        root = self._range_domain_from_args(reduction.range_args)
        if root is None:
            # Source specialization has already canonicalized zero-argument Cells
            # in the range bounds (for example a zero-argument projection-length Cell) to UID calls.  The
            # earlier source-wide prepass resolved the original range before that
            # rewrite and retains the exact physical interval here.  Reusing that
            # exact root interval is safe; it is not representative trace data.
            if self.primary_coordinate_domain is None:
                return {}
            root_domain = self.primary_coordinate_domain
        else:
            root_domain, _root_step = root

        domains: dict[str, _IntDomain] = {}
        pending: list[str] = []

        def add_domain(uid: str, dom: _IntDomain) -> None:
            old = domains.get(uid)
            new = dom if old is None else old.union(dom)
            if old is None or new != old:
                domains[uid] = new
                if uid not in pending:
                    pending.append(uid)

        def record_call(call: ast.Call, env: dict[str, Any]) -> None:
            if not (
                isinstance(call.func, ast.Name)
                and call.func.id in specialized
                and call.func.id in self.variant_trace_by_uid
            ):
                return
            uid = call.func.id
            tr = self.variant_trace_by_uid[uid]
            if tr.key.time_pos is None:
                return
            # Canonical scheduled coordinate calls have one coordinate argument.
            # Fixed-coordinate synthetic scalars have none and need no call domain.
            if len(call.args) != 1 or call.keywords:
                return
            dom = self._int_domain_expr(call.args[0], env)
            if dom is not None:
                add_domain(uid, dom)

        def scan_expr(node: ast.AST | None, env: dict[str, Any]) -> None:
            if node is None:
                return
            if isinstance(node, ast.IfExp):
                scan_expr(node.test, env)
                verdict = self._domain_truth(node.test, env)
                if verdict is True:
                    scan_expr(node.body, self._refine_domain_env(node.test, env, True) or env)
                elif verdict is False:
                    scan_expr(node.orelse, self._refine_domain_env(node.test, env, False) or env)
                else:
                    tenv = self._refine_domain_env(node.test, env, True)
                    fenv = self._refine_domain_env(node.test, env, False)
                    if tenv is not None:
                        scan_expr(node.body, tenv)
                    if fenv is not None:
                        scan_expr(node.orelse, fenv)
                return
            if isinstance(node, ast.Call):
                for arg in node.args:
                    scan_expr(arg, env)
                for kw in node.keywords:
                    scan_expr(kw.value, env)
                record_call(node, env)
                return
            for child in ast.iter_child_nodes(node):
                scan_expr(child, env)

        def scan_block(stmts: list[ast.stmt], initial: dict[str, Any]) -> list[dict[str, Any]]:
            states = [dict(initial)]
            for st in stmts:
                next_states: list[dict[str, Any]] = []
                for env in states:
                    if isinstance(st, ast.Return):
                        scan_expr(st.value, env)
                        continue
                    if isinstance(st, ast.Raise):
                        scan_expr(st.exc, env)
                        continue
                    if (
                        isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], ast.Name)
                    ):
                        scan_expr(st.value, env)
                        nxt = dict(env)
                        name = st.targets[0].id
                        ok, value = self._proven_static_expr(st.value, env)
                        if ok:
                            nxt[name] = value
                        else:
                            dom = self._int_domain_expr(st.value, env)
                            if dom is not None:
                                nxt[name] = dom
                            else:
                                nxt.pop(name, None)
                        next_states.append(nxt)
                        continue
                    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                        scan_expr(st.value, env)
                        nxt = dict(env)
                        if st.value is None:
                            nxt.pop(st.target.id, None)
                        else:
                            dom = self._int_domain_expr(st.value, env)
                            if dom is not None:
                                nxt[st.target.id] = dom
                            else:
                                nxt.pop(st.target.id, None)
                        next_states.append(nxt)
                        continue
                    if isinstance(st, ast.If):
                        scan_expr(st.test, env)
                        verdict = self._domain_truth(st.test, env)
                        branches: list[dict[str, Any]] = []
                        if verdict is not False:
                            tenv = self._refine_domain_env(st.test, env, True)
                            if tenv is not None:
                                branches.extend(scan_block(list(st.body), tenv))
                        if verdict is not True:
                            fenv = self._refine_domain_env(st.test, env, False)
                            if fenv is not None:
                                branches.extend(
                                    scan_block(list(st.orelse), fenv) if st.orelse else [fenv]
                                )
                        next_states.extend(branches)
                        continue
                    if isinstance(st, ast.Expr):
                        scan_expr(st.value, env)
                        next_states.append(env)
                        continue
                    if isinstance(st, ast.AugAssign):
                        scan_expr(st.value, env)
                        scan_expr(st.target, env)
                        nxt = dict(env)
                        if isinstance(st.target, ast.Name):
                            nxt.pop(st.target.id, None)
                        next_states.append(nxt)
                        continue
                    # Imperative/local loop constructs are outside this interval
                    # proof.  Inspect calls conservatively but retain no new local
                    # bounds from them.
                    for child in ast.iter_child_nodes(st):
                        scan_expr(child, env)
                    next_states.append(env)
                states = self._merge_domain_states(next_states)
                if not states:
                    break
            return states

        root_env = {reduction.loop_var: root_domain}
        for filt in reduction.filters:
            scan_expr(filt, root_env)
        scan_expr(reduction.body_expr, root_env)

        iterations = 0
        while pending and iterations < 8192:
            iterations += 1
            uid = pending.pop(0)
            row = specialized.get(uid)
            if row is None:
                continue
            tr, fn, _forced_role, _time_param_source = row
            if tr.key.time_pos is None or not fn.args.args:
                continue
            scan_block(list(fn.body), {fn.args.args[0].arg: domains[uid]})

        if pending:
            return {}
        return domains

    def _derive_coordinate_recurrence_facts(
        self, domains: dict[str, _IntDomain]
    ) -> tuple[CoordinateRecurrenceDomainFact, ...]:
        """Freeze exact active/boundary domains for guarded unit recurrences.

        This pass runs only after the source-demand fixed point converges.  It
        rescans each demanded coordinate formula with its final domain and records
        the subdomain on which an exact self edge is reachable.  Consequently a
        fixed invocation such as ``f(0)`` with ``if t > n: return seed`` and
        ``f(t + 1)`` becomes demand ``[0,n+1]``, active ``[0,n]`` and boundary
        ``{n+1}`` without consulting observed trace coordinates.

        G2a intentionally freezes only exact unit self recurrences.  Wider/mixed
        offsets remain represented by ordinary canonical transition evidence and
        receive no boundary-domain fact from this foundation.
        """
        facts: list[CoordinateRecurrenceDomainFact] = []

        for uid, demand in sorted(domains.items()):
            tr = self.variant_trace_by_uid.get(uid)
            if tr is None or tr.key.time_pos is None:
                continue
            base = self.base_funcs.get(tr.schema.name)
            if base is None:
                continue
            params = [a.arg for a in base.args.args]
            tpos = tr.key.time_pos
            if tpos >= len(params):
                continue
            tparam = params[tpos]
            active_rows: list[tuple[int, _IntDomain]] = []

            def record_self_call(call: ast.Call, env: dict[str, Any]) -> None:
                if not (
                    isinstance(call.func, ast.Name)
                    and call.func.id == tr.schema.name
                ):
                    return
                bound = self._bind_source_call(tr.schema.name, call, env)
                if bound is None:
                    return
                assigned, call_tpos = bound
                if call_tpos != tpos:
                    return
                expr = assigned[params[tpos]]
                off = self._source_coordinate_relative_offset(expr, tparam)
                caller = env.get(tparam)
                if off is None or not isinstance(caller, _IntDomain):
                    return
                active_rows.append((int(off), caller))

            def scan_expr(node: ast.AST | None, env: dict[str, Any]) -> None:
                if node is None:
                    return
                if isinstance(node, ast.IfExp):
                    scan_expr(node.test, env)
                    verdict = self._domain_truth(node.test, env)
                    if verdict is True:
                        tenv = self._refine_domain_env(node.test, env, True) or env
                        scan_expr(node.body, tenv)
                    elif verdict is False:
                        fenv = self._refine_domain_env(node.test, env, False) or env
                        scan_expr(node.orelse, fenv)
                    else:
                        tenv = self._refine_domain_env(node.test, env, True)
                        fenv = self._refine_domain_env(node.test, env, False)
                        if tenv is not None:
                            scan_expr(node.body, tenv)
                        if fenv is not None:
                            scan_expr(node.orelse, fenv)
                    return
                if isinstance(node, ast.Call):
                    record_self_call(node, env)
                for child in ast.iter_child_nodes(node):
                    scan_expr(child, env)

            def scan_block(stmts: list[ast.stmt], initial: dict[str, Any]) -> list[dict[str, Any]]:
                states = [dict(initial)]
                for st in stmts:
                    next_states: list[dict[str, Any]] = []
                    for env in states:
                        if isinstance(st, ast.Return):
                            scan_expr(st.value, env)
                            continue
                        if isinstance(st, ast.Raise):
                            scan_expr(st.exc, env)
                            continue
                        if (
                            isinstance(st, ast.Assign)
                            and len(st.targets) == 1
                            and isinstance(st.targets[0], ast.Name)
                        ):
                            scan_expr(st.value, env)
                            nxt = dict(env)
                            name = st.targets[0].id
                            ok, value = self._proven_static_expr(st.value, env)
                            if ok:
                                nxt[name] = value
                            else:
                                local_domain = self._int_domain_expr(st.value, env)
                                if local_domain is not None:
                                    nxt[name] = local_domain
                                else:
                                    nxt.pop(name, None)
                            next_states.append(nxt)
                            continue
                        if isinstance(st, ast.If):
                            scan_expr(st.test, env)
                            verdict = self._domain_truth(st.test, env)
                            if verdict is not False:
                                tenv = self._refine_domain_env(st.test, env, True)
                                if tenv is not None:
                                    next_states.extend(scan_block(list(st.body), tenv))
                            if verdict is not True:
                                fenv = self._refine_domain_env(st.test, env, False)
                                if fenv is not None:
                                    next_states.extend(
                                        scan_block(list(st.orelse), fenv) if st.orelse else [fenv]
                                    )
                            continue
                        if isinstance(st, ast.Expr):
                            scan_expr(st.value, env)
                            next_states.append(env)
                            continue
                        for child in ast.iter_child_nodes(st):
                            scan_expr(child, env)
                        next_states.append(env)
                    states = self._merge_domain_states(next_states)
                    if not states:
                        break
                return states

            env: dict[str, Any] = {tparam: demand}
            aux_iter = iter(tr.key.aux_values)
            for index, name in enumerate(params):
                if index != tpos:
                    env[name] = next(aux_iter)
            scan_block(list(_strip_docstring(copy.deepcopy(base)).body), env)

            nonzero = [(off, dom) for off, dom in active_rows if off != 0]
            offsets = tuple(sorted({off for off, _ in nonzero}))
            if len(offsets) != 1 or abs(offsets[0]) != 1:
                continue
            offset = int(offsets[0])
            active: _IntDomain | None = None
            for row_off, row_domain in nonzero:
                if row_off != offset:
                    continue
                active = row_domain if active is None else active.union(row_domain)
            if active is None:
                continue

            boundary: tuple[int, ...] = ()
            scan_requirement = "descending" if offset > 0 else "ascending"
            if (
                offset == 1
                and int(demand.lo) == int(active.lo)
                and int(demand.hi) == int(active.hi) + 1
            ):
                boundary = (int(demand.hi),)
            elif (
                offset == -1
                and int(demand.hi) == int(active.hi)
                and int(demand.lo) == int(active.lo) - 1
            ):
                boundary = (int(demand.lo),)
            else:
                continue

            facts.append(CoordinateRecurrenceDomainFact(
                source_uid=str(uid),
                source_name=str(tr.schema.name),
                coordinate_parameter=str(tparam),
                demand_lo=int(demand.lo),
                demand_hi=int(demand.hi),
                active_lo=int(active.lo),
                active_hi=int(active.hi),
                recurrence_offsets=(offset,),
                boundary_coordinates=boundary,
                scan_requirement=scan_requirement,
                output_invocation_uid=str(self.output_invocation.uid),
            ))

        return tuple(facts)

    def _normalize_coordinate_expression(
        self, node: ast.AST, *, bound_names: set[str] | None = None
    ) -> ast.AST:
        """Substitute only already-proven static facts inside a coordinate expression."""
        frontend = self
        bound = set(bound_names or ())

        class CoordinateFacts(ast.NodeTransformer):
            def visit_Call(self, call: ast.Call):
                if (
                    isinstance(call.func, ast.Name) and not call.args and not call.keywords
                    and call.func.id in frontend.base_funcs and call.func.id not in bound
                ):
                    ok, value = frontend._proven_static_cell_value(call.func.id)
                    if ok:
                        return ast.copy_location(ast.Constant(value=value), call)
                return self.generic_visit(call)

            def visit_Name(self, name: ast.Name):
                if (
                    isinstance(name.ctx, ast.Load) and name.id not in bound
                    and name.id in frontend.refs
                ):
                    ok, value = frontend._proven_static_reference_value(name.id)
                    if ok:
                        return ast.copy_location(ast.Constant(value=value), name)
                return name

        out = CoordinateFacts().visit(copy.deepcopy(node))
        out = _CoordinateConstantFolder().visit(ast.fix_missing_locations(out))
        out = _ConservativeFolder().visit(ast.fix_missing_locations(out))
        if isinstance(out, list):
            raise FrontendError("coordinate constant folding unexpectedly produced statements")
        return ast.fix_missing_locations(out)

    def _fold_fixed_local_constants(self, fn: ast.FunctionDef) -> ast.FunctionDef:
        """Propagate literal locals after a coordinate parameter has been fixed.

        Only assignments that reduce to literal constants are propagated.  Dynamic
        locals and non-constant control flow remain untouched.
        """
        env: dict[str, ast.AST] = {}

        def assigned_names(node: ast.AST) -> set[str]:
            out: set[str] = set()
            for child in ast.walk(node):
                targets: list[ast.AST] = []
                if isinstance(child, ast.Assign):
                    targets.extend(child.targets)
                elif isinstance(child, ast.AnnAssign):
                    targets.append(child.target)
                elif isinstance(child, ast.AugAssign):
                    targets.append(child.target)
                for target in targets:
                    for part in ast.walk(target):
                        if isinstance(part, ast.Name):
                            out.add(part.id)
            return out

        def process(stmts: list[ast.stmt], current: dict[str, ast.AST]) -> list[ast.stmt]:
            out: list[ast.stmt] = []
            for original in stmts:
                st = _Substitute(current).visit(copy.deepcopy(original))
                st = _CoordinateConstantFolder().visit(ast.fix_missing_locations(st))
                st = _ConservativeFolder().visit(ast.fix_missing_locations(st))
                if isinstance(st, list):
                    out.extend(process(list(st), current))
                    if out and isinstance(out[-1], (ast.Return, ast.Raise)):
                        break
                    continue
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    target = st.targets[0].id
                    if isinstance(st.value, ast.Constant):
                        current[target] = copy.deepcopy(st.value)
                    else:
                        current.pop(target, None)
                elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    target = st.target.id
                    if isinstance(st.value, ast.Constant):
                        current[target] = copy.deepcopy(st.value)
                    else:
                        current.pop(target, None)
                elif isinstance(st, ast.If):
                    for target in assigned_names(st):
                        current.pop(target, None)
                out.append(st)
                if isinstance(st, (ast.Return, ast.Raise)):
                    break
            return out

        out = copy.deepcopy(fn)
        out.body = process(list(out.body), env)
        return ast.fix_missing_locations(out)

    def _source_has_future_coordinate_dependency(self, base_uid: str) -> bool:
        cached = self._future_coordinate_source_cache.get(base_uid)
        if cached is not None:
            return cached
        tr = self.variant_trace_by_uid.get(base_uid)
        if tr is None or tr.key.time_pos is None:
            self._future_coordinate_source_cache[base_uid] = False
            return False
        fn = self.base_funcs.get(tr.schema.name)
        if fn is None:
            self._future_coordinate_source_cache[base_uid] = False
            return False
        current_t = tr.schema.parameters[tr.key.time_pos]

        for call in (node for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            callee = call.func.id
            candidates = [
                row for row in self.variant_trace_by_uid.values()
                if row.schema.name == callee and row.key.time_pos is not None
            ]
            if not candidates:
                continue
            positions = {row.key.time_pos for row in candidates}
            if len(positions) != 1:
                continue
            pos = next(iter(positions))
            callee_params = list(candidates[0].schema.parameters)
            arg = call.args[pos] if pos < len(call.args) else None
            if arg is None and pos < len(callee_params):
                pname = callee_params[pos]
                arg = next((kw.value for kw in call.keywords if kw.arg == pname), None)
            if arg is not None:
                off = self._source_coordinate_relative_offset(arg, current_t)
                if off is not None and off > 0:
                    self._future_coordinate_source_cache[base_uid] = True
                    return True
        self._future_coordinate_source_cache[base_uid] = False
        return False

    @staticmethod
    def _source_coordinate_relative_offset(expr: ast.AST, variable: str) -> int | None:
        if isinstance(expr, ast.Name) and expr.id == variable:
            return 0
        if (
            isinstance(expr, ast.BinOp)
            and isinstance(expr.left, ast.Name)
            and expr.left.id == variable
            and isinstance(expr.right, ast.Constant)
            and isinstance(expr.right.value, int)
            and not isinstance(expr.right.value, bool)
        ):
            if isinstance(expr.op, ast.Add):
                return int(expr.right.value)
            if isinstance(expr.op, ast.Sub):
                return -int(expr.right.value)
        return None

    def _request_fixed_coordinate(self, base_uid: str, coordinate: int) -> str:
        tr = self.variant_trace_by_uid.get(base_uid)
        if tr is None or tr.key.time_pos is None:
            raise FrontendError(f"fixed-coordinate specialization requires a coordinate variant, got {base_uid}")
        if self._source_has_future_coordinate_dependency(base_uid):
            raise FrontendError(
                f"future dependency in coordinate family {tr.schema.name} requires fallback"
            )
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise FrontendError("fixed-coordinate specialization requires an integer coordinate")
        key = (base_uid, int(coordinate))
        existing = self._fixed_coordinate_uid_by_key.get(key)
        if existing is not None:
            return existing
        if len(self._fixed_coordinate_uid_by_key) >= self._fixed_coordinate_limit:
            raise FrontendError(
                f"fixed-coordinate specialization exceeds the generic cap of {self._fixed_coordinate_limit} variants"
            )
        digest = hashlib.sha1(f"{base_uid}|{int(coordinate)}".encode("utf-8")).hexdigest()[:12]
        synthetic_uid = f"vfix_{digest}"
        self._fixed_coordinate_uid_by_key[key] = synthetic_uid
        self._fixed_coordinate_requests[synthetic_uid] = key
        self._used_fixed_coordinate_facts[synthetic_uid] = FixedCoordinateFact(
            source_uid=base_uid, source_name=tr.schema.name,
            coordinate=int(coordinate), synthetic_uid=synthetic_uid,
        )
        return synthetic_uid

    def _preflight_observed_dependencies(self) -> None:
        observed = set(self.schema_by_name)
        for source_name in sorted(self.compiled_source_names):
            fn = self.base_funcs[source_name]
            for node in ast.walk(fn):
                dep = None
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    dep = node.func.id
                elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                    dep = node.value.id
                if (
                    dep is not None
                    and dep in self.base_funcs
                    and dep not in self.fallback_cells
                    and dep not in observed
                ):
                    raise FrontendError(
                        f"Cells {dep} was not observed in the representative trace; "
                        "fallback or broader representative sampling is required"
                    )

    @staticmethod
    def _is_modelx_cells_object(value: Any) -> bool:
        """Return whether ``value`` is a concrete modelx Cells object.

        Context-accessor normalization is intentionally limited to modelx Cells.
        In particular, an arbitrary Python attribute that merely happens to be
        callable must never become a preparation-time execution escape hatch.
        """
        cls = getattr(value, "__class__", None)
        return bool(
            cls is not None
            and getattr(cls, "__name__", "") == "Cells"
            and str(getattr(cls, "__module__", "")).startswith("modelx.")
        )

    @staticmethod
    def _is_modelx_space_object(value: Any) -> bool:
        cls = getattr(value, "__class__", None)
        return bool(
            cls is not None
            and getattr(cls, "__name__", "") in {"UserSpace", "ItemSpace"}
            and str(getattr(cls, "__module__", "")).startswith("modelx.")
        )

    @staticmethod
    def _is_modelx_itemspace_object(value: Any) -> bool:
        cls = getattr(value, "__class__", None)
        return bool(
            cls is not None
            and getattr(cls, "__name__", "") == "ItemSpace"
            and str(getattr(cls, "__module__", "")).startswith("modelx.")
        )

    def _context_parameterized_space_for_key(
        self, ref_name: str, key: Any | None = None
    ) -> Any:
        root = self._context_root_for_key(ref_name, key)
        if not self._is_modelx_space_object(root):
            raise FrontendError(f"model-bound context root {ref_name!r} is not a modelx Space")
        params = tuple(getattr(root, "parameters", ()) or ())
        if not params:
            raise FrontendError(f"model-bound context Space {ref_name!r} is not parameterized")
        return root

    def _context_selected_itemspace(
        self, ref_name: str, selectors: Sequence[Any], key: Any | None = None
    ) -> Any:
        root = self._context_parameterized_space_for_key(ref_name, key)
        params = tuple(getattr(root, "parameters", ()) or ())
        if len(selectors) != len(params):
            raise FrontendError(
                f"model-bound context Space {ref_name!r} selection arity changed; "
                f"expected {len(params)}, got {len(selectors)}"
            )
        selector = selectors[0] if len(selectors) == 1 else tuple(selectors)
        try:
            item = root[selector]
        except Exception as exc:
            raise FrontendError(
                f"model-bound context Space {ref_name!r} could not select an ItemSpace: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not self._is_modelx_itemspace_object(item):
            raise FrontendError(
                f"model-bound context Space {ref_name!r} selection did not return an ItemSpace"
            )
        return item

    def _context_itemspace_cell(
        self, item: Any, ref_name: str, member_name: str
    ) -> Any:
        candidate = None
        cells = getattr(item, "cells", None)
        if cells is not None and member_name in cells:
            candidate = cells[member_name]
        if candidate is None:
            try:
                candidate = getattr(item, member_name)
            except Exception:
                candidate = None
        if not self._is_modelx_cells_object(candidate):
            raise FrontendError(
                f"selected model-bound Space member {ref_name}[...].{member_name} "
                "is not a concrete Cells object"
            )
        return candidate

    @staticmethod
    def _formula_unbound_load_names(source: str) -> set[str]:
        """Return lexical load names not bound inside one formula source.

        This is a source-structure query only.  It never executes source and is used
        solely to identify owner-parameter dependencies in a child Space Cells
        closure.
        """
        try:
            tree = ast.parse(source)
        except (SyntaxError, TypeError) as exc:
            raise FrontendError("could not parse child Space formula for hierarchy binding") from exc
        bound: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
        return {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        } - bound

    def _hierarchical_itemspace_parameter_binding(
        self, ref_name: str, cell_name: str
    ) -> HierarchicalItemSpaceParameterBinding | None:
        """Prove that a relative child-Space Cells closure inherits RunDomain keys.

        Admission is intentionally narrow: the referenced root must be an
        unparameterized modelx child Space under the selected parameterized Space,
        the selected-Space reference must actually be relative, and the Cells
        dependency closure must contain at least one bare load of a selected-Space
        declared parameter that is not shadowed by a child ref/Cells/local binding.
        Parameterized intermediate Spaces are rejected because they would introduce
        a second ItemSpace context not represented by the current RunDomain.
        """
        if not self.space_params or ref_name not in self.refs:
            return None
        root = self.refs[ref_name]
        if not self._is_modelx_space_object(root):
            return None
        if tuple(getattr(root, "parameters", ()) or ()):
            return None
        cells = getattr(root, "cells", None)
        if cells is None or cell_name not in cells:
            return None

        # A same-looking absolute Space reference is not an inherited ItemSpace
        # binding.  Require modelx's own relative-reference metadata when present.
        impl = getattr(self.space, "_impl", None)
        impl_refs = getattr(impl, "refs", None)
        try:
            ref_impl = impl_refs[ref_name] if impl_refs is not None else None
        except Exception:
            ref_impl = None
        if ref_impl is None or not bool(getattr(ref_impl, "is_relative", False)):
            return None

        # The child must belong to exactly this selected Space context.  Any
        # parameterized intermediate ancestor would need an additional selector and
        # is outside this foundation.
        cur = root
        found_owner = False
        while True:
            parent = getattr(cur, "parent", None)
            if parent is self.space:
                found_owner = True
                break
            if parent is None or parent is getattr(self.space, "model", None):
                break
            if tuple(getattr(parent, "parameters", ()) or ()):
                return None
            cur = parent
        if not found_owner:
            return None

        owner_params = tuple(self.space_params)
        owner_set = set(owner_params)
        inherited: set[str] = set()
        seen: set[tuple[str, str]] = set()
        dependency_fullnames: set[str] = set()
        stack: list[tuple[Any, str]] = [(root, cell_name)]

        def descendant_without_competing_parameters(space_obj: Any) -> bool:
            cur_space = space_obj
            while cur_space is not None:
                parent = getattr(cur_space, "parent", None)
                if parent is self.space:
                    return True
                if parent is None or parent is getattr(self.space, "model", None):
                    return False
                if tuple(getattr(parent, "parameters", ()) or ()):
                    return False
                cur_space = parent
            return False

        while stack:
            current_space, name = stack.pop()
            current_cells = getattr(current_space, "cells", None)
            if current_cells is None or name not in current_cells:
                return None
            current_cell = current_cells[name]
            cell_fullname = str(getattr(current_cell, "fullname", None) or name)
            node_key = (str(getattr(current_space, "fullname", None) or id(current_space)), cell_fullname)
            if node_key in seen:
                continue
            seen.add(node_key)
            dependency_fullnames.add(cell_fullname)
            formula = getattr(current_cell, "formula", None)
            source = getattr(formula, "source", None)
            if not isinstance(source, str) or not source.strip():
                return None
            loads = self._formula_unbound_load_names(source)
            current_refs_obj = getattr(current_space, "refs", {})
            current_refs = set(current_refs_obj.keys())
            current_cell_names = set(current_cells.keys())
            # Explicit child refs/Cells shadow an owner parameter of the same
            # spelling.  Only genuinely unresolved bare loads may be inherited.
            unresolved = loads - current_refs - current_cell_names
            inherited.update(unresolved & owner_set)

            for dep in loads & current_cell_names:
                stack.append((current_space, dep))

            # Relative Cells refs can bridge sibling child Spaces while preserving
            # the same owning ItemSpace.  Follow only those structurally relative
            # references whose target Space is still an unparameterized descendant
            # of the selected RunDomain Space.
            impl = getattr(current_space, "_impl", None)
            impl_refs = getattr(impl, "refs", None)
            for dep in loads & current_refs:
                try:
                    ref_value = current_refs_obj[dep]
                    ref_impl = impl_refs[dep] if impl_refs is not None else None
                except Exception:
                    continue
                if not (
                    self._is_modelx_cells_object(ref_value)
                    and ref_impl is not None
                    and bool(getattr(ref_impl, "is_relative", False))
                ):
                    continue
                target_space = getattr(ref_value, "parent", None)
                target_name = getattr(ref_value, "name", None)
                if (
                    target_space is not None
                    and isinstance(target_name, str)
                    and descendant_without_competing_parameters(target_space)
                ):
                    stack.append((target_space, target_name))

        if not inherited:
            return None
        names = tuple(name for name in owner_params if name in inherited)
        positions = tuple(owner_params.index(name) for name in names)
        if len(set(positions)) != len(positions):
            return None

        binding = HierarchicalItemSpaceParameterBinding(
            ref_name=ref_name,
            cell_name=cell_name,
            owner_space_fullname=str(
                getattr(self.space, "fullname", None) or getattr(self.space, "name", "<space>")
            ),
            child_space_fullname=str(
                getattr(root, "fullname", None) or getattr(root, "name", "<space>")
            ),
            parameter_names=names,
            parameter_positions=positions,
            dependency_cells=tuple(sorted(dependency_fullnames)),
        )
        # The compile-time RunDomain must already demonstrate exact activation of
        # every inherited parameter through the relative child Space.
        self._validate_hierarchical_itemspace_binding(binding, self.run_keys)
        return binding

    def _run_key_parameter_value(self, key: Any, position: int) -> Any:
        if position < 0 or position >= len(self.space_params):
            raise FrontendError("hierarchical ItemSpace parameter position is invalid")
        if len(self.space_params) == 1:
            value = key
        else:
            if not isinstance(key, tuple) or len(key) != len(self.space_params):
                raise FrontendError(
                    f"multi-parameter ItemSpace key must be a tuple of length {len(self.space_params)}"
                )
            value = key[position]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (tuple, list, dict, set, np.ndarray)):
            raise FrontendError(
                "composite ItemSpace parameters are outside hierarchical scalar binding"
            )
        return value

    def _validate_hierarchical_itemspace_binding(
        self, binding: HierarchicalItemSpaceParameterBinding, keys: Sequence[Any]
    ) -> None:
        if binding.owner_space_fullname != str(
            getattr(self.space, "fullname", None) or getattr(self.space, "name", "<space>")
        ):
            raise FrontendError("hierarchical ItemSpace owner changed after compilation")
        if tuple(binding.parameter_names) != tuple(
            self.space_params[pos] for pos in binding.parameter_positions
        ):
            raise FrontendError("hierarchical ItemSpace parameter schema changed after compilation")
        for key in keys:
            owner = self._space_instance(key)
            dynamic_root = self._context_root_for_key(binding.ref_name, key)
            if dynamic_root is self.refs[binding.ref_name]:
                raise FrontendError(
                    f"relative child Space {binding.ref_name!r} did not activate an ItemSpace context"
                )
            # The dynamic child may be more than one unparameterized level below the
            # owner, but its parent chain must terminate at this exact ItemSpace.
            cur = dynamic_root
            found_owner = False
            while cur is not None:
                parent = getattr(cur, "parent", None)
                if parent is owner:
                    found_owner = True
                    break
                if parent is None:
                    break
                if tuple(getattr(parent, "parameters", ()) or ()):
                    raise FrontendError(
                        "hierarchical binding encountered a competing parameterized ItemSpace context"
                    )
                cur = parent
            if not found_owner:
                raise FrontendError(
                    f"relative child Space {binding.ref_name!r} is not owned by the active RunDomain ItemSpace"
                )
            refs = getattr(dynamic_root, "refs", None)
            if refs is None:
                raise FrontendError("hierarchical child Space has no resolved reference environment")
            for name, pos in zip(binding.parameter_names, binding.parameter_positions):
                if name not in refs:
                    raise FrontendError(
                        f"hierarchical child Space did not inherit owner parameter {name!r}"
                    )
                actual = refs[name]
                if isinstance(actual, np.generic):
                    actual = actual.item()
                expected = self._run_key_parameter_value(key, pos)
                try:
                    equal = type(actual) is type(expected) and actual == expected
                except Exception:
                    equal = False
                if not bool(equal):
                    # Numeric scalar wrappers are harmless, but rich/coercive
                    # equality is deliberately not accepted.
                    if not (
                        isinstance(actual, (bool, int, float, str))
                        and isinstance(expected, (bool, int, float, str))
                        and actual == expected
                    ):
                        raise FrontendError(
                            f"hierarchical owner parameter {name!r} does not match the active RunDomain key"
                        )

    def _context_root_for_key(self, ref_name: str, key: Any | None = None) -> Any:
        """Resolve one model-bound root through the active ItemSpace context.

        Access through ``_space_instance`` is important for modelx interface
        references whose bound parameters are activated by the enclosing ItemSpace
        (for example a scenario Space selected by one Projection coordinate).
        """
        if ref_name not in self.refs:
            raise FrontendError(f"unknown model-bound context root {ref_name!r}")
        if key is None:
            return self.refs[ref_name]
        item = self._space_instance(key)
        refs = getattr(item, "refs", None)
        if refs is not None and ref_name in refs:
            return refs[ref_name]
        try:
            return getattr(item, ref_name)
        except Exception:
            return self.refs[ref_name]

    def _context_cell_for_key(
        self, ref_name: str, member_name: str, key: Any | None = None
    ) -> Any:
        """Resolve a concrete Cells member, including a Cells-valued Space alias."""
        root = self._context_root_for_key(ref_name, key)
        candidate = None
        cells = getattr(root, "cells", None)
        if cells is not None and member_name in cells:
            candidate = cells[member_name]
        if candidate is None:
            refs = getattr(root, "refs", None)
            if refs is not None and member_name in refs:
                candidate = refs[member_name]
        if candidate is None:
            try:
                candidate = getattr(root, member_name)
            except Exception:
                candidate = None
        if not self._is_modelx_cells_object(candidate):
            raise FrontendError(
                f"model-bound context member {ref_name}.{member_name} is not a concrete Cells object"
            )
        return candidate

    @staticmethod
    def _context_primitive(value: Any) -> tuple[bool, Any]:
        if isinstance(value, np.generic):
            value = value.item()
        return (
            isinstance(value, (str, bool, int, float, type(None))),
            value,
        )

    def _context_static_value(self, node: ast.AST, key: Any) -> tuple[bool, Any]:
        """Resolve a tiny immutable/static argument subset for context Cells calls.

        Values are re-resolved for every bound run key rather than baked into the
        generated formula.  This keeps modelx rebinding on the preparation side of
        the existing numeric ABI and avoids introducing runtime context objects.
        """
        if isinstance(node, ast.Constant):
            return self._context_primitive(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.space_params or node.id not in self.refs:
                return False, None
            item = self._space_instance(key)
            refs = getattr(item, "refs", None)
            value = refs[node.id] if refs is not None and node.id in refs else self.refs[node.id]
            return self._context_primitive(value)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base_name = node.value.id
            if base_name not in self.refs:
                return False, None
            try:
                base = self._context_root_for_key(base_name, key)
                refs = getattr(base, "refs", None)
                if refs is not None and node.attr in refs:
                    value = refs[node.attr]
                else:
                    value = getattr(base, node.attr)
            except Exception:
                return False, None
            return self._context_primitive(value)
        if isinstance(node, ast.UnaryOp):
            ok, value = self._context_static_value(node.operand, key)
            if not ok:
                return False, None
            try:
                if isinstance(node.op, ast.UAdd):
                    return self._context_primitive(+value)
                if isinstance(node.op, ast.USub):
                    return self._context_primitive(-value)
                if isinstance(node.op, ast.Not):
                    return True, not bool(value)
            except Exception:
                return False, None
        if isinstance(node, ast.BinOp):
            ok_l, left = self._context_static_value(node.left, key)
            ok_r, right = self._context_static_value(node.right, key)
            if not (ok_l and ok_r):
                return False, None
            try:
                if isinstance(node.op, ast.Add): value = left + right
                elif isinstance(node.op, ast.Sub): value = left - right
                elif isinstance(node.op, ast.Mult): value = left * right
                elif isinstance(node.op, ast.Div): value = left / right
                elif isinstance(node.op, ast.FloorDiv): value = left // right
                elif isinstance(node.op, ast.Mod): value = left % right
                else: return False, None
            except Exception:
                return False, None
            return self._context_primitive(value)
        return False, None

    def _context_static_argument_is_fixed(self, node: ast.AST) -> bool:
        values: list[Any] = []
        for key in self.run_keys:
            ok, value = self._context_static_value(node, key)
            if not ok:
                return False
            values.append(value)
        if not values:
            return False
        first = values[0]
        try:
            return all(type(value) is type(first) and value == first for value in values[1:])
        except Exception:
            return False

    def _space_instance(self, key: Any):
        """Return the structural instance for one RunDomain key.

        Parameterized Spaces use normal ItemSpace indexing.  A non-parameterized
        Space is itself the sole structural instance and is represented in
        RunDomain by the singleton key ``()``.
        """
        if not self.space_params:
            if key != ():
                raise FrontendError(
                    f"non-parameterized Space expects RunDomain key (), got {key!r}"
                )
            return self.space
        return self.space[key]

    def modelx_output_value(self, key: Any) -> Any:
        item = self._space_instance(key)
        cell = getattr(item, self.output)
        return cell(*self.output_invocation.argument_values)

    def _infer_run_keys(self):
        if not self.space.parameters:
            return ((),)
        if self.space.parameters:
            row = self._detect_point_row_cell()
            if row is not None:
                frame = row.provider()
                if isinstance(frame, pd.DataFrame):
                    return tuple(frame.index)
        raise FrontendError("run_keys are required when the compiler cannot infer ItemSpace keys from a DataFrame row lookup")

    def _detect_point_row_cell(self) -> PointRowBinding | None:
        """Normalize supported model-point row access to one semantic source.

        Supported source spellings are deliberately narrow:

        ``direct_table.loc[point_id]``
        ``external_space.zero_arg_table_cell().loc[point_id]``

        Both become the same point-input ABI later.  Only preparation retains the
        provider callback; emitters receive numeric arrays plus provenance metadata.
        """
        if not self.space_params:
            return None
        positions = {name: i for i, name in enumerate(self.space_params)}

        def semantic_for(
            *, source_kind: str, source_ref: str, source_cell: str | None,
            frame: pd.DataFrame, key_name: str, key_pos: int,
        ) -> NormalizedPointRowSource:
            return NormalizedPointRowSource(
                source_kind=source_kind, source_ref=source_ref, source_cell=source_cell,
                source_shape=tuple(frame.shape), index_names=tuple(frame.index.names),
                key_parameter=key_name, key_position=int(key_pos),
            )

        for name, cell in self.space.cells.items():
            if cell.formula is None or cell.parameters:
                continue
            fn = self._parse_formula_source(name, cell.formula.source)
            body = [
                x for x in fn.body
                if not (
                    isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant)
                    and isinstance(x.value.value, str)
                )
            ]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            r = body[0].value
            if not (
                isinstance(r, ast.Subscript)
                and isinstance(r.value, ast.Attribute)
                and r.value.attr == "loc"
                and isinstance(r.slice, ast.Name)
                and r.slice.id in positions
            ):
                continue
            key_name = r.slice.id
            key_pos = positions[key_name]
            table_expr = r.value.value

            # Existing direct DataFrame Reference form.
            if isinstance(table_expr, ast.Name):
                refname = table_expr.id
                frame = self.refs.get(refname)
                if isinstance(frame, pd.DataFrame):
                    semantic = semantic_for(
                        source_kind="direct_reference", source_ref=refname,
                        source_cell=None, frame=frame, key_name=key_name, key_pos=key_pos,
                    )
                    return PointRowBinding(
                        cell_name=name, semantic=semantic,
                        provider=lambda rn=refname, frontend=self: frontend.space.refs[rn],
                    )

            # Local zero-argument Cell returning a DataFrame.  This is the same
            # semantic point-row source as a direct/external table; the extra Cell
            # layer is preparation-only and must not survive the numeric ABI.
            if (
                isinstance(table_expr, ast.Call)
                and isinstance(table_expr.func, ast.Name)
                and not table_expr.args and not table_expr.keywords
                and table_expr.func.id in self.space.cells
            ):
                source_cell_name = table_expr.func.id
                source_cell = self.space.cells[source_cell_name]
                if not tuple(source_cell.parameters or ()):
                    try:
                        frame = source_cell()
                    except Exception:
                        frame = None
                    if isinstance(frame, pd.DataFrame):
                        semantic = semantic_for(
                            source_kind="local_cell", source_ref=self.space.name,
                            source_cell=source_cell_name, frame=frame,
                            key_name=key_name, key_pos=key_pos,
                        )
                        return PointRowBinding(
                            cell_name=name, semantic=semantic, provider=source_cell,
                        )

            # Qualified zero-argument external Cell returning a DataFrame.
            if (
                isinstance(table_expr, ast.Call)
                and isinstance(table_expr.func, ast.Attribute)
                and isinstance(table_expr.func.value, ast.Name)
                and not table_expr.args and not table_expr.keywords
            ):
                refname = table_expr.func.value.id
                cell_name = table_expr.func.attr
                obj = self.refs.get(refname)
                if obj is None or not hasattr(obj, "cells") or cell_name not in obj.cells:
                    continue
                source_cell = obj.cells[cell_name]
                if tuple(source_cell.parameters or ()):
                    continue
                try:
                    frame = source_cell()
                except Exception:
                    continue
                if not isinstance(frame, pd.DataFrame):
                    continue
                semantic = semantic_for(
                    source_kind="external_cell", source_ref=refname,
                    source_cell=cell_name, frame=frame, key_name=key_name, key_pos=key_pos,
                )
                return PointRowBinding(
                    cell_name=name, semantic=semantic, provider=source_cell,
                )
        return None

    def _is_numpy_array_call(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "array" or not isinstance(node.func.value, ast.Name):
            return False
        obj = self.refs.get(node.func.value.id)
        return obj is np

    def _unwrap_generator(self, node: ast.AST) -> ast.GeneratorExp | None:
        if isinstance(node, ast.GeneratorExp):
            return node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("list", "tuple") and len(node.args)==1 and isinstance(node.args[0], ast.GeneratorExp):
            return node.args[0]
        return None

    def _discover_vector_helpers(self):
        result = {}
        for name, fn in self.base_funcs.items():
            body=[x for x in fn.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
            if len(body)!=1 or not isinstance(body[0],ast.Return):
                continue
            ret=body[0].value
            if not (isinstance(ret,ast.Call) and self._is_numpy_array_call(ret) and len(ret.args)==1):
                continue
            gen=self._unwrap_generator(ret.args[0])
            if gen is None or len(gen.generators)!=1 or gen.generators[0].ifs:
                continue
            comp=gen.generators[0]
            if not isinstance(comp.target,ast.Name) or not (isinstance(comp.iter,ast.Call) and isinstance(comp.iter.func,ast.Name) and comp.iter.func.id=="range"):
                continue
            result[name]=(comp.target.id, copy.deepcopy(comp.iter), copy.deepcopy(gen.elt))
        return result

    def _inline_vector_subscripts(self, node: ast.AST, depth=0) -> ast.AST:
        if depth > 10:
            raise FrontendError("nested vector helper expansion exceeded safety depth")
        outer=self
        class Rewriter(ast.NodeTransformer):
            def visit_Subscript(self, n):
                n=self.generic_visit(n)
                if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and not n.value.args and n.value.func.id in outer.vector_elements and not isinstance(n.slice,ast.Slice):
                    vname=n.value.func.id
                    var, rng, elt=outer.vector_elements[vname]
                    repl=_Substitute({var:copy.deepcopy(n.slice)}).visit(copy.deepcopy(elt))
                    return ast.copy_location(outer._inline_vector_subscripts(ast.fix_missing_locations(repl), depth+1),n)
                return n
        return ast.fix_missing_locations(Rewriter().visit(copy.deepcopy(node)))

    def _stable_realized_dep_names(self, uid: str) -> set[str] | None:
        """Return one direct same-Space dependency signature if every call agrees.

        The check is intentionally concrete rather than variant-aggregate: a time
        family may legitimately take different branches at different coordinates.
        Such a family is not branch-specialized.
        """
        tr = self.variant_trace_by_uid[uid]
        signatures: set[frozenset[str]] = set()
        for runtime_node, key in self.trace.node_variant.items():
            if key != tr.key:
                continue
            deps: set[str] = set()
            for pred in self.trace.graph.predecessors(runtime_node):
                pkey = self.trace.node_variant.get(pred)
                dep = self.trace.variants.get(pkey) if pkey is not None else None
                if dep is not None and dep.schema.name in self.base_funcs:
                    deps.add(dep.schema.name)
            signatures.add(frozenset(deps))
            if len(signatures) > 1:
                return None
        if len(signatures) != 1:
            return None
        return set(next(iter(signatures)))

    def _same_space_calls(self, node: ast.AST) -> set[str]:
        return {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in self.base_funcs
        }

    def _enumerate_straight_paths(
        self, stmts: Sequence[ast.stmt]
    ) -> list[tuple[list[ast.stmt], set[str]]] | None:
        """Enumerate if/return execution paths for conservative graph pruning."""

        max_paths = 32

        def walk(rest: list[ast.stmt], prefix: list[ast.stmt]):
            if not rest:
                return [(list(prefix), set())]
            st, tail = rest[0], rest[1:]
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                return walk(tail, prefix)
            if isinstance(st, (ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)):
                return None
            if isinstance(st, (ast.Return, ast.Raise)):
                return [(list(prefix) + [copy.deepcopy(st)], self._same_space_calls(st))]
            if isinstance(st, ast.If):
                cond = self._same_space_calls(st.test)
                out: list[tuple[list[ast.stmt], set[str]]] = []
                branches = [list(st.body), list(st.orelse)] if st.orelse else [list(st.body), []]
                for branch in branches:
                    got = walk(branch + tail, list(prefix))
                    if got is None:
                        return None
                    out.extend((seq, cond | calls) for seq, calls in got)
                    if len(out) > max_paths:
                        return None
                return out
            got = walk(tail, list(prefix) + [copy.deepcopy(st)])
            if got is None:
                return None
            own = self._same_space_calls(st)
            return [(seq, own | calls) for seq, calls in got]

        return walk(list(stmts), [])

    def _prune_to_realized_path(self, uid: str, fn: ast.FunctionDef) -> ast.FunctionDef:
        """Prune only a uniquely proven fixed-graph control-flow path.

        No traced scalar/table result is substituted.  The only specialization is
        graph topology, and only when every concrete invocation of the variant has
        the same direct dependency signature and exactly one source path matches it.
        """
        if not self.realized_graph_only:
            return fn
        observed = self._stable_realized_dep_names(uid)
        if observed is None:
            return fn
        source_calls = self._same_space_calls(fn)
        if not (source_calls - observed):
            return fn
        paths = self._enumerate_straight_paths(fn.body)
        if not paths or len(paths) <= 1:
            return fn
        matches = [seq for seq, calls in paths if calls == observed]
        if len(matches) != 1:
            return fn
        out = copy.deepcopy(fn)
        out.body = matches[0]
        return ast.fix_missing_locations(out)

    def _runtime_scalar_helper_candidates(self, name: str, *, allow_coordinate: bool = False):
        """Return observed variants eligible for upstream scalar-helper inlining.

        The original subset admits only non-coordinate helpers with numeric/bool
        runtime arguments.  ``allow_coordinate`` adds one deliberately narrow case:
        a scalar Cell whose representative trace classified one parameter as a
        coordinate, but whose body can still be fully inlined before scheduling.
        Numeric/bool non-coordinate arguments may remain runtime scalar expressions.
        Categorical non-coordinate arguments must become literals (or full-run-domain
        static proofs) at the call boundary; they do not create a new runtime
        categorical dimension.
        """
        base = self.base_funcs.get(name)
        if base is None:
            return ()
        params = [a.arg for a in base.args.args]
        if not params:
            return ()
        rows = [tr for tr in self.variant_trace_by_uid.values() if tr.schema.name == name]
        if not rows:
            return ()
        positions = {tr.key.time_pos for tr in rows}
        coordinate_classified = positions != {None}
        if coordinate_classified:
            if not allow_coordinate or None in positions or len(positions) != 1:
                return ()
        numeric = (bool, int, float, np.bool_, np.integer, np.floating)
        scalar = (str, bool, int, float, np.bool_, np.integer, np.floating)
        for tr in rows:
            if tr.dtype not in ("bool", "int64", "float64"):
                return ()
            expected_aux = len(params) - (1 if coordinate_classified else 0)
            if len(tr.key.aux_values) != expected_aux:
                return ()
            allowed = scalar if coordinate_classified else numeric
            if any(not isinstance(value, allowed) for value in tr.key.aux_values):
                return ()
        return tuple(rows)

    def _proven_static_helper_argument(self, expr: ast.AST) -> tuple[bool, Any]:
        """Resolve a helper selector to a literal only with a full-domain proof.

        Literal arguments are accepted directly.  A Cell call is accepted only if
        it resolves to a non-coordinate observed variant whose value is invariant
        over every ``run_key`` in the compiled artifact.  This is intentionally not
        a generic constant folder for arbitrary expressions.
        """
        try:
            value = ast.literal_eval(expr)
        except Exception:
            value = None
        else:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (str, bool, int, float)):
                return True, value

        if (
            isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
            and expr.func.id in self.base_funcs
        ):
            try:
                uid, coordinate_expr = self._variant_for_call(expr.func.id, expr, {})
            except FrontendError:
                return False, None
            if coordinate_expr is not None:
                return False, None
            ok, value = self._proven_static_variant_value(uid)
            if ok:
                if isinstance(value, np.generic):
                    value = value.item()
                self._used_static_facts[("cell", ast.unparse(expr))] = StaticScalarFact(
                    "cell", ast.unparse(expr), value, len(self.run_keys)
                )
                return True, value

        if (
            isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
            and expr.func.id in {"str", "bool", "int", "float"}
            and len(expr.args) == 1 and not expr.keywords
        ):
            ok, inner = self._proven_static_helper_argument(expr.args[0])
            if ok:
                try:
                    return True, {"str": str, "bool": bool, "int": int, "float": float}[expr.func.id](inner)
                except Exception:
                    return False, None
        return False, None

    def _sparse_observed_coordinate_helper_candidates(
        self, name: str, params: list[str], assigned: dict[str, ast.AST]
    ) -> tuple[tuple[Any, ...], int | None]:
        """Recover one primary-time helper parameter from a sparse fixed trace.

        Some pure wrappers are executed only when an event occurs, so modelx may
        observe ``helper(36, 'OPTION')`` as a fixed non-coordinate variant even
        though the source caller is the monthly family ``caller(t, bucket)`` and
        passes its primary ``t`` straight through.  Reclassifying the trace would
        be unsafe; instead this call-site-only proof permits upstream helper
        inlining when exactly one parameter is literally the canonical primary
        coordinate ``t``.  The observed values for that parameter must be integer
        coordinates.  All other categorical arguments remain subject to the same
        full-domain static proof as ordinary coordinate-classified helpers.
        """
        rows = tuple(tr for tr in self.variant_trace_by_uid.values() if tr.schema.name == name)
        if not rows or any(tr.key.time_pos is not None for tr in rows):
            return (), None
        if any(tr.dtype not in ("bool", "int64", "float64") for tr in rows):
            return (), None
        if any(len(tr.key.aux_values) != len(params) for tr in rows):
            return (), None
        candidates: list[int] = []
        for pos, param in enumerate(params):
            expr = assigned.get(param)
            if not (isinstance(expr, ast.Name) and expr.id == "t"):
                continue
            observed = [tr.key.aux_values[pos] for tr in rows]
            if observed and all(
                isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                for value in observed
            ):
                candidates.append(pos)
        if len(candidates) != 1:
            return (), None
        scalar = (str, bool, int, float, np.bool_, np.integer, np.floating)
        coordinate_pos = candidates[0]
        for tr in rows:
            if any(
                pos != coordinate_pos and not isinstance(value, scalar)
                for pos, value in enumerate(tr.key.aux_values)
            ):
                return (), None
        return rows, coordinate_pos

    def _runtime_scalar_helper_expr_from_block(
        self, stmts: list[ast.stmt], env: dict[str, ast.AST]
    ) -> ast.AST:
        """Convert a pure straight/conditional helper body to one expression.

        This is hygienic upstream inlining, not a second runtime calling convention.
        Assignments become lexical substitutions; conditionals become ``IfExp``.
        Loops, mutation, raises, augmented assignment and other statement effects are
        intentionally outside the first helper subset.
        """
        if not stmts:
            raise FrontendError("runtime scalar helper has a path without a return value")
        original = copy.deepcopy(stmts[0])
        rest = list(stmts[1:])
        st = _Substitute(env).visit(original)
        st = _ConservativeFolder().visit(ast.fix_missing_locations(st))
        if isinstance(st, list):
            return self._runtime_scalar_helper_expr_from_block(list(st) + rest, dict(env))
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            return self._runtime_scalar_helper_expr_from_block(rest, dict(env))
        if isinstance(st, ast.Pass):
            return self._runtime_scalar_helper_expr_from_block(rest, dict(env))
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            next_env = dict(env)
            next_env[st.targets[0].id] = copy.deepcopy(st.value)
            return self._runtime_scalar_helper_expr_from_block(rest, next_env)
        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            next_env = dict(env)
            next_env[st.target.id] = copy.deepcopy(st.value)
            return self._runtime_scalar_helper_expr_from_block(rest, next_env)
        if isinstance(st, ast.Return):
            if st.value is None:
                raise FrontendError("runtime scalar helper has a bare return")
            return ast.fix_missing_locations(copy.deepcopy(st.value))
        if isinstance(st, ast.If):
            test = copy.deepcopy(st.test)
            if isinstance(test, ast.Constant):
                branch = list(st.body if bool(test.value) else st.orelse)
                return self._runtime_scalar_helper_expr_from_block(branch + rest, dict(env))
            body = self._runtime_scalar_helper_expr_from_block(list(st.body) + rest, dict(env))
            other = self._runtime_scalar_helper_expr_from_block(list(st.orelse) + rest, dict(env))
            return ast.fix_missing_locations(ast.IfExp(test=test, body=body, orelse=other))
        raise FrontendError(
            f"runtime scalar helper statement {type(st).__name__} is outside the pure inlining subset"
        )

    def _prepare_runtime_scalar_helper_function(
        self, name: str, call: ast.Call
    ) -> tuple[ast.FunctionDef, tuple[str, ...], int, bool, bool, bool] | None:
        """Bind and normalize one pure scalar helper without erasing its locals.

        dev19-dev24 represented a helper call as one recursively substituted
        expression.  That is compact for tiny helpers, but it duplicates reused
        locals exponentially in deeper pure helper graphs (RILA's Black-Scholes
        chain is the first real-product example).  Keep this preparation step
        separate from the representation chosen by the caller: the legacy
        expression inliner can still use it, while dev25 can lower the same proven
        helper body into hygienic local statements.

        The returned FunctionDef has all call arguments bound and has already
        passed the same static-fact, table and local-syntax normalization used by
        the expression path.  It is *not* a runtime helper ABI: callers must erase
        it by inlining before executable graph scheduling.
        """
        base = copy.deepcopy(self.base_funcs[name])
        params = [a.arg for a in base.args.args]
        defaults: dict[str, ast.AST] = {}
        if base.args.defaults:
            for param, default in zip(params[-len(base.args.defaults):], base.args.defaults):
                defaults[param] = copy.deepcopy(default)
        assigned: dict[str, ast.AST] = {}
        for param, arg in zip(params, call.args):
            assigned[param] = copy.deepcopy(arg)
        for kw in call.keywords:
            if kw.arg is None or kw.arg in assigned:
                return None
            assigned[kw.arg] = copy.deepcopy(kw.value)
        for param in params:
            if param not in assigned:
                if param not in defaults:
                    return None
                assigned[param] = copy.deepcopy(defaults[param])
        if len(assigned) != len(params):
            return None

        rows = self._runtime_scalar_helper_candidates(name)
        coordinate_classified = False
        coordinate_recovered_from_callsite = False
        source_backed_unobserved = False
        coordinate_pos_override: int | None = None
        if not rows:
            rows = self._runtime_scalar_helper_candidates(name, allow_coordinate=True)
            coordinate_classified = bool(rows)
        if not rows:
            rows, coordinate_pos_override = self._sparse_observed_coordinate_helper_candidates(
                name, params, assigned
            )
            if rows:
                coordinate_classified = True
                coordinate_recovered_from_callsite = True
        if not rows:
            if not self._source_backed_unobserved_helper_ok(name):
                return None
            # No trace schema is invented.  This helper is erased by hygienic
            # source inlining before scheduling, so its call arguments remain the
            # caller's ordinary scalar/coordinate expressions and observed Cells
            # in its body remain graph boundaries.
            source_backed_unobserved = True

        if coordinate_classified:
            if coordinate_pos_override is None:
                positions = {tr.key.time_pos for tr in rows}
                if None in positions or len(positions) != 1:
                    return None
                coordinate_pos = next(iter(positions))
            else:
                coordinate_pos = coordinate_pos_override
            numeric = (bool, int, float, np.bool_, np.integer, np.floating)
            aux_by_pos: dict[int, list[Any]] = {pos: [] for pos in range(len(params)) if pos != coordinate_pos}
            for tr in rows:
                if coordinate_recovered_from_callsite:
                    for pos in aux_by_pos:
                        aux_by_pos[pos].append(tr.key.aux_values[pos])
                else:
                    aux_iter = iter(tr.key.aux_values)
                    for pos in range(len(params)):
                        if pos == coordinate_pos:
                            continue
                        aux_by_pos[pos].append(next(aux_iter))
            for pos, param in enumerate(params):
                if pos == coordinate_pos:
                    continue
                observed = aux_by_pos[pos]
                if observed and all(isinstance(value, numeric) for value in observed):
                    # Numeric/bool helper inputs are ordinary runtime scalars.  If a
                    # downstream operation needs them to be static, that operation's
                    # own proof/normalizer must establish it or fail closed.
                    continue
                ok, value = self._proven_static_helper_argument(assigned[param])
                if not ok:
                    return None
                assigned[param] = ast.Constant(value=value)

        # Bind helper arguments first.  In particular, a coordinate-classified pure
        # helper such as CI_UK_S.pivot_interp has categorical table selectors that
        # become static only at the caller.  Normalizing pandas before substitution
        # would therefore miss the preparation-time slice and reject a supported
        # mathematical lookup for purely syntactic reasons.
        base = _Substitute(assigned).visit(base)
        base = _ConservativeFolder().visit(ast.fix_missing_locations(base))
        if not isinstance(base, ast.FunctionDef):
            return None
        base = _prune_source_unreachable_tails(base)
        base = self._specialize_static_facts_and_domains(base)
        base = _ConservativeFolder().visit(ast.fix_missing_locations(base))
        if not isinstance(base, ast.FunctionDef):
            return None
        base = _prune_source_unreachable_tails(base)
        base = self._normalize_external_tables(base)
        base = self._normalize_simple_local_syntax(base)
        return (
            base, tuple(params), len(rows), coordinate_classified,
            coordinate_recovered_from_callsite, source_backed_unobserved,
        )

    def _record_runtime_scalar_helper_fact(
        self, name: str, params: tuple[str, ...], observed_variant_count: int,
        coordinate_classified: bool, coordinate_recovered_from_callsite: bool = False,
        source_backed_unobserved: bool = False,
    ) -> None:
        self._used_runtime_scalar_helper_facts[name] = RuntimeScalarHelperFact(
            source_name=name, parameters=params,
            observed_variant_count=observed_variant_count,
            coordinate_classified=coordinate_classified,
            coordinate_recovered_from_callsite=coordinate_recovered_from_callsite,
            source_backed_unobserved=source_backed_unobserved,
        )

    def _inline_runtime_scalar_helper_call(self, name: str, call: ast.Call) -> ast.AST | None:
        prepared = self._prepare_runtime_scalar_helper_function(name, call)
        if prepared is None:
            return None
        (
            base, params, observed_variant_count, coordinate_classified,
            coordinate_recovered, source_backed_unobserved,
        ) = prepared
        expr = self._runtime_scalar_helper_expr_from_block(list(base.body), {})
        self._record_runtime_scalar_helper_fact(
            name, params, observed_variant_count, coordinate_classified, coordinate_recovered,
            source_backed_unobserved,
        )
        return ast.copy_location(expr, call)

    def _source_backed_unobserved_helper_ok(self, name: str) -> bool:
        """Prove a completely unobserved Cell can disappear by source inlining.

        Observed Cells are terminal graph boundaries for this proof, including
        recursive state.  Only the closure made exclusively of *unobserved* helper
        Cells is recursively inspected, so a helper such as ``excess_factor(t)``
        may legitimately read observed recursive account state without flattening
        that state into the helper.  Any cycle among unobserved helpers, object
        boundary, or statement side effect fails closed.
        """
        if name in self._source_backed_helper_cache:
            return self._source_backed_helper_cache[name]
        if name in self.schema_by_name or name not in self.base_funcs:
            self._source_backed_helper_cache[name] = False
            return False

        numeric_dtypes = {"bool", "int64", "float64"}
        active: set[str] = set()
        local_cache: dict[str, bool] = {}

        def walk(current: str) -> bool:
            if current in local_cache:
                return local_cache[current]
            if current in active:
                return False
            if current in self.schema_by_name:
                if (
                    self.model_point_row is not None
                    and current == self.model_point_row.cell_name
                ):
                    # The model-point row is an object only at preparation time.
                    # Its field reads are normalized later into guarded numeric
                    # point inputs, so it is a safe source-helper boundary without
                    # creating an object runtime ABI.
                    return True
                rows = [tr for tr in self.variant_trace_by_uid.values() if tr.schema.name == current]
                return bool(rows) and all(tr.dtype in numeric_dtypes for tr in rows)
            fn = self.base_funcs.get(current)
            if fn is None:
                return False
            active.add(current)
            returns = [node for node in ast.walk(fn) if isinstance(node, ast.Return)]
            if not returns or any(node.value is None for node in returns):
                active.remove(current)
                local_cache[current] = False
                return False
            prohibited = (
                ast.Global, ast.Nonlocal, ast.AugAssign, ast.Delete,
                ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith,
                ast.Match, ast.Yield, ast.YieldFrom, ast.Await,
            )
            if any(isinstance(node, prohibited) for node in ast.walk(fn)):
                active.remove(current)
                local_cache[current] = False
                return False
            for ret in returns:
                value = ret.value
                if isinstance(value, ast.Constant) and (
                    value.value is None or isinstance(value.value, (str, bytes, complex))
                ):
                    active.remove(current)
                    local_cache[current] = False
                    return False
                if isinstance(value, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
                    active.remove(current)
                    local_cache[current] = False
                    return False
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                dep = node.func.id
                if dep not in self.base_funcs:
                    continue
                if not walk(dep):
                    active.remove(current)
                    local_cache[current] = False
                    return False
            active.remove(current)
            local_cache[current] = True
            return True

        result = walk(name)
        self._source_backed_helper_cache.update(local_cache)
        self._source_backed_helper_cache[name] = result
        return result

    def _source_reaches_recursive_cycle(self, name: str) -> bool:
        """Conservatively identify source closures that contain recursive state.

        This is used only to decide whether a non-affine coordinate call should be
        preserved as a scheduled Cell instead of treated as an acyclic helper.  It
        does not classify ordinary parameterized helper calls and it does not make
        any scheduling decision by itself.
        """
        if name in self._source_recursive_reach_cache:
            return self._source_recursive_reach_cache[name]

        seen: set[str] = set()
        active: set[str] = set()

        def walk(current: str) -> bool:
            if current in active:
                return True
            if current in seen:
                return False
            seen.add(current)
            active.add(current)
            fn = self.base_funcs.get(current)
            if fn is not None:
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in self.base_funcs
                        and walk(node.func.id)
                    ):
                        active.remove(current)
                        return True
            active.remove(current)
            return False

        result = walk(name)
        self._source_recursive_reach_cache[name] = result
        return result

    def _inline_runtime_scalar_helpers_as_statements(
        self, fn: ast.FunctionDef
    ) -> tuple[ast.FunctionDef, dict[str, ast.AST]]:
        """Erase parameterized pure helpers while preserving their local temporaries.

        The old helper path substitutes every local into the returned expression.
        That is intentionally retained as a small/fallback representation, but it
        is a poor fit for formulas that reuse intermediates.  This pass performs a
        tiny A-normal/statement-level lowering instead: a helper call becomes a set
        of hygienically named assignments plus one result temporary.  Existing
        scheduled Cells remain calls and are canonicalized by ``Calls`` later.

        Only eager expression positions are hoisted.  Python short-circuit forms
        (BoolOp, chained Compare, IfExp, comprehensions/lambdas) are left intact so
        we never change which helper calls execute.  If such a position needs
        helper expansion, the legacy expression path remains the conservative
        backstop and its node budget still applies.
        """
        frontend = self
        existing = self._function_bound_names(fn)
        coordinate_aliases: dict[str, ast.AST] = {}
        counter = 0
        statement_expansion_nodes = 0

        def fresh(prefix: str) -> str:
            nonlocal counter
            while True:
                name = f"__mx_h_{prefix}_{counter}"
                counter += 1
                if name not in existing:
                    existing.add(name)
                    return name

        def count_added(nodes: list[ast.stmt], helper_name: str) -> None:
            nonlocal statement_expansion_nodes
            statement_expansion_nodes += sum(1 for st in nodes for _ in ast.walk(st))
            if statement_expansion_nodes > frontend._runtime_scalar_helper_expansion_limit:
                raise FrontendError(
                    "runtime scalar helper expansion budget exceeded "
                    f"({statement_expansion_nodes} > {frontend._runtime_scalar_helper_expansion_limit}) "
                    f"during statement lowering of {helper_name}"
                )

        def substitute_local(expr: ast.AST, env: dict[str, ast.AST]) -> ast.AST:
            out = _Substitute(env).visit(copy.deepcopy(expr))
            out = _ConservativeFolder().visit(ast.fix_missing_locations(out))
            if isinstance(out, list):
                raise FrontendError("runtime scalar helper expression folding produced statements")
            return ast.fix_missing_locations(out)

        def helper_call_kind(call: ast.Call) -> str | None:
            """Return source Cell name iff this call needs pure-helper inlining.

            Besides parameterized helpers with runtime numeric auxiliaries, dev25
            also uses the same fail-closed inliner to erase a *non-affine coordinate
            composition*.  The executable scheduler supports ``t + constant``.  If
            a pure Cell is called at some richer coordinate (RILA's
            ``index_level(term_start_month(t))``), inlining can move that coordinate
            expression into a table/scalar formula instead of pretending it is a
            new scheduled state dimension.  Recursive state fails closed because
            forcing its body would encounter the same helper on the inline stack.
            """
            if not (isinstance(call.func, ast.Name) and call.func.id in frontend.base_funcs):
                return None
            try:
                _uid, coordinate_expr = frontend._variant_for_call(call.func.id, call, {})
            except FrontendError as exc:
                if str(exc).startswith("non-time specialization argument is not a literal:"):
                    return call.func.id
                if (
                    str(exc).startswith(f"Cells {call.func.id} was not observed")
                    and frontend._source_backed_unobserved_helper_ok(call.func.id)
                ):
                    return call.func.id
                return None
            if coordinate_expr is None:
                return None

            # Recover SSA-like helper aliases only for the scheduling decision.
            # Arithmetic formula uses still read the local temporary itself.
            coordinate_expr = copy.deepcopy(coordinate_expr)
            for _ in range(len(coordinate_aliases) + 1):
                before = ast.dump(coordinate_expr, include_attributes=False)
                coordinate_expr = _Substitute(coordinate_aliases).visit(coordinate_expr)
                coordinate_expr = ast.fix_missing_locations(coordinate_expr)
                if ast.dump(coordinate_expr, include_attributes=False) == before:
                    break
            coordinate_expr = frontend._normalize_coordinate_expression(
                coordinate_expr, bound_names=existing
            )

            if (
                isinstance(coordinate_expr, ast.Constant)
                and isinstance(coordinate_expr.value, int)
                and not isinstance(coordinate_expr.value, bool)
            ):
                return None
            if isinstance(coordinate_expr, ast.Name) and coordinate_expr.id == "t":
                return None
            if (
                isinstance(coordinate_expr, ast.BinOp)
                and isinstance(coordinate_expr.left, ast.Name)
                and coordinate_expr.left.id == "t"
                and isinstance(coordinate_expr.right, ast.Constant)
                and isinstance(coordinate_expr.right.value, int)
                and not isinstance(coordinate_expr.right.value, bool)
                and isinstance(coordinate_expr.op, (ast.Add, ast.Sub))
            ):
                return None
            # A non-affine call into a Cell whose source closure reaches a cycle is
            # scheduled state. Preserve the outer semantic Cell rather than
            # partially flattening its implementation until a nested recursive
            # call eventually trips the inline stack. Pure coordinate helpers
            # continue through statement inlining.
            if frontend._source_reaches_recursive_cycle(call.func.id):
                # The dev26 scheduled-history foundation is explicitly tied to the
                # primary projection coordinate. A recursive Cell queried only by an
                # independent/local coordinate (IUL segment month ``m`` is the real
                # product example) is a genuine additional state dimension, not a
                # snapshot of the current monthly state. Keep the dev25 fail-closed
                # helper-recursion behavior for that class instead of routing it into
                # history scheduling / expensive loop recovery.
                mentions_primary_t = any(
                    isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "t"
                    for node in ast.walk(coordinate_expr)
                )
                if mentions_primary_t:
                    return None
            return call.func.id

        def lift_expr(expr: ast.AST, helper_stack: tuple[str, ...]) -> tuple[list[ast.stmt], ast.AST]:
            """Hoist helper calls from eager scalar expression positions."""
            expr = copy.deepcopy(expr)

            # Preserve Python's conditional/short-circuit evaluation exactly.  The
            # existing expression inliner can still handle a small helper under one
            # of these nodes, or fail closed via its expansion budget.
            if isinstance(expr, (ast.BoolOp, ast.IfExp, ast.Lambda, ast.GeneratorExp,
                                 ast.ListComp, ast.SetComp, ast.DictComp)):
                return [], expr
            if isinstance(expr, ast.Compare) and len(expr.ops) > 1:
                return [], expr

            if isinstance(expr, ast.Call):
                pre: list[ast.stmt] = []
                new_args: list[ast.AST] = []
                for arg in expr.args:
                    p, value = lift_expr(arg, helper_stack)
                    pre.extend(p)
                    new_args.append(value)
                new_keywords: list[ast.keyword] = []
                for kw in expr.keywords:
                    p, value = lift_expr(kw.value, helper_stack)
                    pre.extend(p)
                    new_keywords.append(ast.keyword(arg=kw.arg, value=value))
                rebuilt = ast.copy_location(
                    ast.Call(func=copy.deepcopy(expr.func), args=new_args, keywords=new_keywords), expr
                )
                helper_name = helper_call_kind(rebuilt)
                if helper_name is None:
                    return pre, rebuilt
                if helper_name in helper_stack:
                    chain = " -> ".join((*helper_stack, helper_name))
                    raise FrontendError(f"runtime scalar helper recursion detected during helper/coordinate inlining: {chain}")
                prepared = frontend._prepare_runtime_scalar_helper_function(helper_name, rebuilt)
                if prepared is None:
                    # Leave it for the legacy fail-closed call rewriter.  In
                    # particular, do not reinterpret a categorical runtime axis as
                    # an ordinary numeric helper just to make the call disappear.
                    return pre, rebuilt
                (
                    helper_fn, params, observed_variant_count, coordinate_classified,
                    coordinate_recovered, source_backed_unobserved,
                ) = prepared
                result_name = fresh("result")
                try:
                    helper_stmts = lower_helper_block(
                        list(helper_fn.body), {}, result_name, (*helper_stack, helper_name)
                    )
                except FrontendError as exc:
                    # A coordinate-classified Cell that becomes recursive when we
                    # inspect its body is scheduled state, not an acyclic helper.
                    # Preserve the call so the executable scheduler can prove or
                    # reject the derived-coordinate history read.
                    if coordinate_classified and str(exc).startswith(
                        "runtime scalar helper recursion detected during helper/coordinate inlining:"
                    ):
                        return pre, rebuilt
                    raise
                count_added(helper_stmts, helper_name)
                frontend._record_runtime_scalar_helper_fact(
                    helper_name, params, observed_variant_count, coordinate_classified,
                    coordinate_recovered, source_backed_unobserved,
                )
                pre.extend(helper_stmts)
                return pre, ast.copy_location(ast.Name(result_name, ast.Load()), expr)

            # Eager expression nodes whose child evaluation order is simple and
            # relevant to the scalar subset.  Build them explicitly instead of a
            # generic ast walk so inserted statements keep Python evaluation order.
            if isinstance(expr, ast.BinOp):
                p1, left = lift_expr(expr.left, helper_stack)
                p2, right = lift_expr(expr.right, helper_stack)
                return p1 + p2, ast.copy_location(ast.BinOp(left, copy.deepcopy(expr.op), right), expr)
            if isinstance(expr, ast.UnaryOp):
                pre, operand = lift_expr(expr.operand, helper_stack)
                return pre, ast.copy_location(ast.UnaryOp(copy.deepcopy(expr.op), operand), expr)
            if isinstance(expr, ast.Compare):
                # One comparison has no chained-comparison short circuit.
                p1, left = lift_expr(expr.left, helper_stack)
                p2, right = lift_expr(expr.comparators[0], helper_stack)
                return p1 + p2, ast.copy_location(
                    ast.Compare(left, [copy.deepcopy(expr.ops[0])], [right]), expr
                )
            if isinstance(expr, ast.Attribute):
                pre, value = lift_expr(expr.value, helper_stack)
                return pre, ast.copy_location(ast.Attribute(value, expr.attr, copy.deepcopy(expr.ctx)), expr)
            if isinstance(expr, ast.Subscript):
                p1, value = lift_expr(expr.value, helper_stack)
                p2, sl = lift_expr(expr.slice, helper_stack)
                return p1 + p2, ast.copy_location(ast.Subscript(value, sl, copy.deepcopy(expr.ctx)), expr)
            if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
                pre: list[ast.stmt] = []
                elts: list[ast.AST] = []
                for elt in expr.elts:
                    p, value = lift_expr(elt, helper_stack)
                    pre.extend(p)
                    elts.append(value)
                if isinstance(expr, ast.Tuple):
                    rebuilt = ast.Tuple(elts, copy.deepcopy(expr.ctx))
                elif isinstance(expr, ast.List):
                    rebuilt = ast.List(elts, copy.deepcopy(expr.ctx))
                else:
                    rebuilt = ast.Set(elts)
                return pre, ast.copy_location(rebuilt, expr)
            if isinstance(expr, ast.Dict):
                pre: list[ast.stmt] = []
                keys: list[ast.AST | None] = []
                vals: list[ast.AST] = []
                for key, val in zip(expr.keys, expr.values):
                    if key is not None:
                        pk, key2 = lift_expr(key, helper_stack)
                        pre.extend(pk)
                    else:
                        key2 = None
                    pv, val2 = lift_expr(val, helper_stack)
                    pre.extend(pv)
                    keys.append(key2)
                    vals.append(val2)
                return pre, ast.copy_location(ast.Dict(keys, vals), expr)

            # Constants, Names and source forms without helper-bearing eager
            # children are safe to leave for the ordinary Cells/intrinsic rewriter.
            return [], expr

        def lower_helper_block(
            stmts: list[ast.stmt], env: dict[str, ast.AST], result_name: str,
            helper_stack: tuple[str, ...],
        ) -> list[ast.stmt]:
            if not stmts:
                raise FrontendError("runtime scalar helper has a path without a return value")
            st = copy.deepcopy(stmts[0])
            rest = list(stmts[1:])
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                return lower_helper_block(rest, dict(env), result_name, helper_stack)
            if isinstance(st, ast.Pass):
                return lower_helper_block(rest, dict(env), result_name, helper_stack)
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                rhs = substitute_local(st.value, env)
                pre, rhs = lift_expr(rhs, helper_stack)
                local_name = fresh(f"local_{st.targets[0].id}")
                assign = ast.copy_location(
                    ast.Assign([ast.Name(local_name, ast.Store())], rhs), st
                )
                # Helper locals are SSA-like (every lowering path gets a fresh
                # name).  Keep their defining expressions separately so the graph
                # call rewriter can recover a coordinate expression such as ``t``
                # from ``m = t`` without substituting arithmetic locals everywhere.
                coordinate_aliases[local_name] = copy.deepcopy(rhs)
                next_env = dict(env)
                next_env[st.targets[0].id] = ast.Name(local_name, ast.Load())
                return pre + [assign] + lower_helper_block(rest, next_env, result_name, helper_stack)
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                rhs = substitute_local(st.value, env)
                pre, rhs = lift_expr(rhs, helper_stack)
                local_name = fresh(f"local_{st.target.id}")
                assign = ast.copy_location(
                    ast.Assign([ast.Name(local_name, ast.Store())], rhs), st
                )
                coordinate_aliases[local_name] = copy.deepcopy(rhs)
                next_env = dict(env)
                next_env[st.target.id] = ast.Name(local_name, ast.Load())
                return pre + [assign] + lower_helper_block(rest, next_env, result_name, helper_stack)
            if isinstance(st, ast.Return):
                if st.value is None:
                    raise FrontendError("runtime scalar helper has a bare return")
                value = substitute_local(st.value, env)
                pre, value = lift_expr(value, helper_stack)
                out = ast.copy_location(
                    ast.Assign([ast.Name(result_name, ast.Store())], value), st
                )
                return pre + [out]
            if isinstance(st, ast.Raise):
                # Preserve terminal helper guards until the established canonical
                # guard-normalization pass.  Local substitutions are applied here
                # so a helper such as ``msg = \"bad\"; raise ValueError(msg)`` can
                # still become the supported literal guard form downstream.
                #
                # Deliberately do not append ``rest``: raise is terminal control
                # flow.  Exception type/message validation and guard-code ownership
                # remain exclusively in normalize_formula_guards().
                exc = None if st.exc is None else substitute_local(st.exc, env)
                cause = None if st.cause is None else substitute_local(st.cause, env)
                return [
                    ast.copy_location(
                        ast.Raise(exc=copy.deepcopy(exc), cause=copy.deepcopy(cause)),
                        st,
                    )
                ]
            if isinstance(st, ast.If):
                test = substitute_local(st.test, env)
                try:
                    constant_test = ast.literal_eval(test)
                except Exception:
                    constant_test = None
                    is_constant = False
                else:
                    is_constant = True
                if is_constant:
                    branch = list(st.body if bool(constant_test) else st.orelse)
                    return lower_helper_block(branch + rest, dict(env), result_name, helper_stack)
                pre, test = lift_expr(test, helper_stack)
                body = lower_helper_block(list(st.body) + rest, dict(env), result_name, helper_stack)
                other = lower_helper_block(list(st.orelse) + rest, dict(env), result_name, helper_stack)
                cond = ast.copy_location(ast.If(test=test, body=body, orelse=other), st)
                return pre + [cond]
            raise FrontendError(
                f"runtime scalar helper statement {type(st).__name__} is outside the pure statement inlining subset"
            )

        def transform_block(stmts: list[ast.stmt]) -> list[ast.stmt]:
            out: list[ast.stmt] = []
            for st in stmts:
                st = copy.deepcopy(st)
                if isinstance(st, ast.Assign):
                    pre, value = lift_expr(st.value, ())
                    st.value = value
                    out.extend(pre)
                    out.append(st)
                elif isinstance(st, ast.AnnAssign) and st.value is not None:
                    pre, value = lift_expr(st.value, ())
                    st.value = value
                    out.extend(pre)
                    out.append(st)
                elif isinstance(st, ast.Return) and st.value is not None:
                    pre, value = lift_expr(st.value, ())
                    st.value = value
                    out.extend(pre)
                    out.append(st)
                elif isinstance(st, ast.If):
                    pre, test = lift_expr(st.test, ())
                    st.test = test
                    st.body = transform_block(list(st.body))
                    st.orelse = transform_block(list(st.orelse))
                    out.extend(pre)
                    out.append(st)
                else:
                    out.append(st)
            return out

        out = copy.deepcopy(fn)
        out.body = transform_block(list(out.body))
        return ast.fix_missing_locations(out), coordinate_aliases

    def _request_source_backed_scheduled_variant(
        self,
        name: str,
        *,
        params: list[str],
        assigned: dict[str, ast.AST],
        observed_time_pos: int | None = None,
    ) -> tuple[str, ast.AST | None] | None:
        """Create one demanded coordinate variant from source, never from values.

        This is deliberately *not* helper inlining.  The returned variant remains a
        normal scheduled graph operation and may participate in lagged recursion.
        The first subset accepts exactly one primary-coordinate argument and only
        literal/full-domain-static auxiliary arguments.  Source lowering, graph
        cycle checks, history/ring proofs and backend safety remain authoritative
        downstream.
        """
        if (
            name not in self.base_funcs
            or name not in self.compiled_source_names
            or name in self.fallback_cells
            or name in self.vector_elements
            or (self.model_point_row is not None and name == self.model_point_row.cell_name)
        ):
            return None
        fn = self.base_funcs[name]
        if (
            fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg is not None
            or fn.args.kwarg is not None or len(fn.args.args) != len(params)
        ):
            return None
        # Side-effectful Python control flow is not admitted merely because source
        # exists.  Ordinary scalar If/Return/Assign and guard raises continue to be
        # validated later by the normal formula safety pass.
        forbidden = (
            ast.Global, ast.Nonlocal, ast.Delete, ast.Yield, ast.YieldFrom, ast.Await,
            ast.With, ast.AsyncWith, ast.Try, ast.AsyncFor, ast.AsyncFunctionDef,
            ast.ClassDef, ast.Import, ast.ImportFrom,
        )
        if any(isinstance(node, forbidden) for node in ast.walk(fn)):
            return None

        if observed_time_pos is not None:
            time_pos = int(observed_time_pos)
            if not (0 <= time_pos < len(params)):
                return None
        else:
            dynamic_positions: list[int] = []
            for pos, param in enumerate(params):
                ok, _value = self._proven_static_helper_argument(assigned[param])
                if not ok:
                    dynamic_positions.append(pos)
            if len(dynamic_positions) != 1:
                return None
            time_pos = dynamic_positions[0]

        time_param = params[time_pos]
        time_expr = copy.deepcopy(assigned[time_param])
        # First proof subset: the demanded coordinate must be expressed directly
        # from the caller's primary ``t``.  More elaborate coordinate-helper
        # algebra belongs to the existing coordinate-relation proof layer rather
        # than being guessed here.
        if not any(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "t"
            for node in ast.walk(time_expr)
        ):
            return None

        aux_values: list[Any] = []
        for pos, param in enumerate(params):
            if pos == time_pos:
                continue
            ok, value = self._proven_static_helper_argument(assigned[param])
            if not ok:
                return None
            if isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (str, bool, int, float)):
                return None
            aux_values.append(value)
        aux = tuple(aux_values)
        token = (name, aux)
        existing = self._source_backed_scheduled_uid_by_key.get(token)
        if existing is not None:
            return existing, time_expr
        observed_uid = self.variants_by_cell_aux.get(token)
        if observed_uid is not None:
            return observed_uid, time_expr

        try:
            cell = self.space.cells[name]
            source = cell.formula.source
            fullname = getattr(cell, "fullname", None) or name
        except Exception:
            return None
        schema = CellSchema(
            fullname=str(fullname), name=name, parameters=tuple(params), source=str(source)
        )
        key = VariantKey(schema.uid, time_pos, aux)
        # The first scheduled-source subset is numeric state.  Dtype is intentionally
        # conservative: bool/int source-backed state should get an explicit source
        # dtype proof in a later extension rather than relying on value sampling.
        tr = VariantTrace(key=key, schema=schema, dtype="float64")
        uid = tr.uid
        if uid in self.variant_trace_by_uid:
            return uid, time_expr
        self.schema_by_name.setdefault(name, schema)
        self.variant_trace_by_uid[uid] = tr
        self.variants_by_cell_aux[token] = uid
        self._source_backed_scheduled_uid_by_key[token] = uid
        if self.primary_coordinate_domain is not None:
            self.coordinate_domains[uid] = self.primary_coordinate_domain
        fact = SourceBackedScheduledFact(
            source_name=name,
            synthetic_uid=uid,
            parameters=tuple(params),
            time_param=time_param,
            aux_values=aux,
            dtype="float64",
            observed=False,
        )
        self._used_source_backed_scheduled_facts[uid] = fact
        return uid, time_expr

    def _variant_for_call(self, name: str, call: ast.Call, caller_mapping: dict[str, ast.AST]) -> tuple[str, ast.AST | None]:
        if name not in self.base_funcs:
            raise FrontendError(f"unknown Cells call {name}")
        base=self.base_funcs[name]
        params=[a.arg for a in base.args.args]
        defaults={}
        if base.args.defaults:
            for p,d in zip(params[-len(base.args.defaults):],base.args.defaults): defaults[p]=d
        assigned: dict[str,ast.AST]={}
        for p,a in zip(params,call.args): assigned[p]=a
        for kw in call.keywords:
            if kw.arg is None:
                raise FrontendError("**kwargs Cells calls are not natively specialized")
            if kw.arg in assigned:
                raise FrontendError(f"duplicate argument {kw.arg!r} in call to {name}")
            assigned[kw.arg]=kw.value
        for p in params:
            if p not in assigned:
                if p in defaults: assigned[p]=defaults[p]
                else: raise FrontendError(f"missing argument {p!r} in call to {name}")
        assigned={p:_Substitute(caller_mapping).visit(copy.deepcopy(v)) for p,v in assigned.items()}

        schema=self.schema_by_name.get(name)
        if schema is None:
            # Acyclic source-backed helpers retain the upstream inlining path.  A
            # recursive/uninlineable demanded Cell may instead become a scheduled
            # source-backed coordinate variant if it satisfies the narrow proof.
            if self._source_backed_unobserved_helper_ok(name):
                raise FrontendError(f"Cells {name} was not observed in the representative trace")
            scheduled = self._request_source_backed_scheduled_variant(
                name, params=params, assigned=assigned
            )
            if scheduled is not None:
                return scheduled
            raise FrontendError(f"Cells {name} was not observed in the representative trace")

        # Schema time position comes from trace, never from parameter spelling.
        time_pos=None
        aux=()
        candidates=[tr for tr in self.variant_trace_by_uid.values() if tr.schema.name==name and tr.key.time_pos is not None]
        if candidates:
            positions={tr.key.time_pos for tr in candidates}
            if len(positions)!=1: raise FrontendError(f"ambiguous time parameter for {name}")
            time_pos=next(iter(positions))
            auxvals=[]
            for j,p in enumerate(params):
                if j==time_pos: continue
                auxvals.append(_literal(assigned[p]))
            aux=tuple(auxvals)
            time_expr=assigned[params[time_pos]]
        else:
            # Non-time parameters are specialized from representative literal calls.
            aux=tuple(_literal(assigned[p]) for p in params)
            time_expr=None
        uid=self.variants_by_cell_aux.get((name,aux))
        if uid is None:
            if not self._source_backed_unobserved_helper_ok(name):
                scheduled = self._request_source_backed_scheduled_variant(
                    name, params=params, assigned=assigned, observed_time_pos=time_pos
                )
                if scheduled is not None:
                    return scheduled
            raise FrontendError(f"call variant {name}{aux!r} was not observed in sample trace; fallback required")
        return uid,time_expr

    def _reference_metadata_scalar_ok(self, name: str) -> bool:
        """Admit one finite point-independent reference-metadata scalar.

        The computation is executed during input preparation, never in the native
        recursive runtime.  Accepted source forms are deliberately bounded to
        immutable DataFrame/Series index metadata: explicit ``for`` loops and
        finite list/set/generator comprehensions over ``table.index``.  ``table``
        may be a direct Reference or a zero-argument table provider reached through
        a model Reference/Space (for example ``data.surr_charge_table()``).

        ``name`` may be a canonical modelx-cython UID.  Source proof always maps
        back to the original Cell formula so canonical renaming does not create a
        second semantic language.
        """
        source_name = getattr(self, "formula_uid_to_source_name", {}).get(name, name)
        base = None
        if source_name in self.space.cells:
            cell = self.space.cells[source_name]
            if cell.formula is not None:
                try:
                    base = self._parse_formula_source(source_name, cell.formula.source)
                except Exception:
                    return False
        if base is None:
            base = self.base_funcs.get(source_name) or self.base_funcs.get(name)
        if base is None:
            return False
        if base.args.posonlyargs or base.args.args or base.args.kwonlyargs or base.args.vararg or base.args.kwarg:
            return False

        def provider_table(expr: ast.AST) -> bool:
            if isinstance(expr, ast.Name):
                return isinstance(self.refs.get(expr.id), (pd.DataFrame, pd.Series))
            if (
                isinstance(expr, ast.Call) and not expr.args and not expr.keywords
                and isinstance(expr.func, ast.Attribute)
                and isinstance(expr.func.value, ast.Name)
                and expr.func.value.id in self.refs
            ):
                try:
                    root = self._context_root_for_key(expr.func.value.id, self.run_keys[0])
                    value = getattr(root, expr.func.attr)()
                except Exception:
                    return False
                return isinstance(value, (pd.DataFrame, pd.Series))
            return False

        table_aliases: set[str] = set()
        for st in base.body:
            if (
                isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and provider_table(st.value)
            ):
                table_aliases.add(st.targets[0].id)

        def table_expr(expr: ast.AST) -> bool:
            return (isinstance(expr, ast.Name) and expr.id in table_aliases) or provider_table(expr)

        def index_iter(expr: ast.AST) -> bool:
            return isinstance(expr, ast.Attribute) and expr.attr == "index" and table_expr(expr.value)

        def table_loc_row(expr: ast.AST) -> bool:
            return (
                isinstance(expr, ast.Subscript)
                and isinstance(expr.value, ast.Attribute)
                and expr.value.attr == "loc"
                and table_expr(expr.value.value)
            )

        row_aliases: set[str] = set()
        for node in ast.walk(base):
            if (
                isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and table_loc_row(node.value)
            ):
                row_aliases.add(node.targets[0].id)

        loops = [node for node in ast.walk(base) if isinstance(node, ast.For)]
        comps = [
            gen
            for node in ast.walk(base)
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp))
            for gen in node.generators
        ]
        if not loops and not comps:
            return False
        for loop in loops:
            if not (isinstance(loop.target, ast.Name) and index_iter(loop.iter)):
                return False
        for gen in comps:
            if gen.is_async or not (isinstance(gen.target, ast.Name) and index_iter(gen.iter)):
                return False

        forbidden = (
            ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
            ast.Delete, ast.AugAssign, ast.NamedExpr, ast.Yield, ast.YieldFrom,
            ast.Await, ast.Global, ast.Nonlocal,
        )
        if any(isinstance(node, forbidden) for node in ast.walk(base)):
            return False

        builtin_calls = {
            "abs", "all", "any", "bool", "float", "int", "len", "list",
            "max", "min", "range", "round", "set", "sum", "tuple",
        }
        source_cell_names = {
            cell_name
            for cell_name, cell in self.space.cells.items()
            if cell.formula is not None
        }
        for node in ast.walk(base):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(not isinstance(target, ast.Name) for target in targets):
                    return False
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if (
                        node.func.id in self.base_funcs
                        or node.func.id in getattr(self, "formula_uid_to_source_name", {})
                        or node.func.id in source_cell_names
                    ):
                        return False
                    if node.func.id not in builtin_calls:
                        return False
                elif isinstance(node.func, ast.Attribute):
                    # Zero-arg external table provider or bounded row reduction.
                    if provider_table(node):
                        continue
                    if node.func.attr not in {"all", "any"}:
                        return False
                    if not any(
                        isinstance(part, ast.Name) and part.id in row_aliases
                        for part in ast.walk(node.func.value)
                    ):
                        return False
                else:
                    return False
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in self.space_params:
                    return False
                if self.model_point_row is not None and node.id == self.model_point_row.cell_name:
                    return False
        return True

    def _materialize_static_value_operators(
        self, uid: str, fn: ast.FunctionDef
    ) -> ast.FunctionDef:
        """Materialize safe immutable static index/slice/cast expressions.

        This pass is intentionally narrower than generic constant folding.  It
        targets expressions whose non-numeric/indexing shape would otherwise leak
        into numeric backends, and admits them only when ``_proven_static_expr``
        can evaluate the exact source expression from complete-run-domain facts.
        """
        frontend = self
        def current_static_facts() -> dict[str, StaticScalarFact]:
            return {
                fact.source_name: fact
                for fact in frontend._used_static_facts.values()
            }

        def dependencies(node: ast.AST) -> tuple[tuple[str, Any], ...]:
            facts = current_static_facts()
            names = sorted({
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                and child.func.id in facts
            })
            return tuple((name, facts[name].value) for name in names)

        def dtype_of(value: Any) -> str:
            if isinstance(value, bool):
                return "bool"
            if isinstance(value, int):
                return "int64"
            if isinstance(value, float):
                return "float64"
            if isinstance(value, str):
                return "string"
            return type(value).__name__

        class Materialize(ast.NodeTransformer):
            def _replace(self, node: ast.AST):
                ok, value = frontend._proven_static_expr(node)
                if not ok or not isinstance(value, (str, bool, int, float)):
                    return None
                deps = dependencies(node)
                if not deps:
                    return None
                facts = current_static_facts()
                frontend._normalized_operators[uid] = NormalizedOperator(
                    uid=uid,
                    kind="static_value",
                    result_dtype=dtype_of(value),
                    result_shape="scalar",
                    static_inputs=deps,
                    provenance=tuple(
                        f"full_run_static:{name}:{facts[name].proof_domain_size}"
                        for name, _ in deps
                    ),
                    state_semantic="state_free",
                    python_supported=True,
                    cython_supported=True,
                )
                return ast.copy_location(ast.Constant(value=value), node)

            def visit_Call(self, node: ast.Call):
                # Casts around a proven static index/slice are materialized as one
                # typed operator, avoiding string/object syntax in numeric IR.
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in {"bool", "int", "float", "str"}
                    and any(isinstance(child, ast.Subscript) for child in ast.walk(node))
                ):
                    replacement = self._replace(node)
                    if replacement is not None:
                        return replacement
                return self.generic_visit(node)

            def visit_Subscript(self, node: ast.Subscript):
                replacement = self._replace(node)
                if replacement is not None:
                    return replacement
                return self.generic_visit(node)

        out = Materialize().visit(copy.deepcopy(fn))
        if not isinstance(out, ast.FunctionDef):
            raise FrontendError(f"static operator materialization removed formula {fn.name}")
        return ast.fix_missing_locations(out)

    def _specialize_function(
        self, uid: str, *, fixed_time: int | None = None, synthetic_uid: str | None = None
    ) -> ast.FunctionDef | None:
        tr=self.variant_trace_by_uid[uid]
        name=tr.schema.name
        if name not in self.base_funcs:
            return None
        fn=copy.deepcopy(self.base_funcs[name])
        # Finite reference-only metadata loops are preparation inputs, not runtime
        # control flow.  Lower them before statement helper processing so an
        # otherwise unsupported ``For`` never enters canonical execution.
        if tr.key.time_pos is None and not tr.key.aux_values and self._reference_metadata_scalar_ok(name):
            key=self.registry.reference_metadata_scalar(name)
            spec=self.registry.specs[key]
            self._normalized_operators[uid]=NormalizedOperator(
                uid=uid,kind="reference_metadata_scalar",
                result_dtype=spec.dtype,result_shape="scalar",
                static_inputs=(("input_key",key),("source_cell",name)),
                provenance=(
                    "finite_reference_index_loop","point_independent_run_domain",
                    "numeric_global_input",
                ),
                state_semantic="state_free",python_supported=True,cython_supported=True,
            )
            return ast.fix_missing_locations(ast.FunctionDef(
                name=synthetic_uid or uid,
                args=ast.arguments(posonlyargs=[],args=[],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),
                body=[ast.Return(ast.Call(ast.Name("global_input",ast.Load()),[ast.Constant(key)],[]))],
                decorator_list=[],returns=None,type_comment=None,
            ))
        source_params=[a.arg for a in fn.args.args]
        coord_env={}
        if tr.key.time_pos is not None:
            source_param=source_params[tr.key.time_pos]
            if fixed_time is not None:
                coord_env[source_param]=_IntDomain(int(fixed_time),int(fixed_time))
            elif uid in self.coordinate_domains:
                coord_env[source_param]=self.coordinate_domains[uid]
        fn=self._specialize_static_facts_and_domains(fn,coordinate_env=coord_env)
        fn=self._materialize_static_value_operators(uid, fn)
        fn=self._prune_to_realized_path(uid, fn)
        params=[a.arg for a in fn.args.args]
        tpos=tr.key.time_pos
        mapping={}
        aux_iter=iter(tr.key.aux_values)
        for j,p in enumerate(params):
            if tpos is not None and j==tpos:
                mapping[p] = ast.Constant(int(fixed_time)) if fixed_time is not None else ast.Name("t",ast.Load())
            else:
                mapping[p]=ast.Constant(next(aux_iter))
        fn=_Substitute(mapping).visit(fn)
        if fixed_time is not None:
            fn=_CoordinateConstantFolder().visit(fn)
        fn=_ConservativeFolder().visit(fn)
        if isinstance(fn,list):
            raise FrontendError(f"constant folding unexpectedly replaced function {name}")
        fn=ast.fix_missing_locations(fn)
        fn=_prune_source_unreachable_tails(fn)
        if fixed_time is not None:
            fn=self._fold_fixed_local_constants(fn)
            fn=self._specialize_static_facts_and_domains(fn)
            fn=self._fold_fixed_local_constants(fn)
            fn=_prune_source_unreachable_tails(fn)
        fn.name=synthetic_uid or uid
        fn.args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg("t")] if (tpos is not None and fixed_time is None) else [],
            vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]
        )
        fn=_strip_docstring(fn)
        # Preserve pure-helper locals before the legacy expression inliner gets a
        # chance to substitute them repeatedly.  Ordinary scheduled Cells remain
        # source-name calls here and are canonicalized by Calls below.
        fn, helper_coordinate_aliases = self._inline_runtime_scalar_helpers_as_statements(fn)
        bound_names=self._function_bound_names(fn)

        frontend=self
        class Calls(ast.NodeTransformer):
            def __init__(self):
                super().__init__()
                self.helper_expansion_nodes = 0

            def visit_Call(self,n):
                # Do not generic_visit first: Cells argument expressions must be rewritten under caller mapping exactly once.
                if isinstance(n.func,ast.Name) and n.func.id in frontend.fallback_input_by_name:
                    if n.args or n.keywords:
                        raise FrontendError(f"regional fallback Cell {n.func.id!r} must be called without arguments")
                    key=frontend.fallback_input_by_name[n.func.id]
                    return ast.copy_location(
                        ast.Call(ast.Name("point_input",ast.Load()),[ast.Constant(key)],[]), n
                    )
                if isinstance(n.func,ast.Name) and (n.func.id in frontend.vector_elements or (frontend.model_point_row and n.func.id == frontend.model_point_row.cell_name)):
                    return self.generic_visit(n)
                if isinstance(n.func,ast.Name) and n.func.id in frontend.base_funcs:
                    if n.func.id in frontend._runtime_scalar_helper_inline_stack:
                        chain = " -> ".join(frontend._runtime_scalar_helper_inline_stack + [n.func.id])
                        raise FrontendError(f"runtime scalar helper recursion detected: {chain}")
                    try:
                        cuid,texpr=frontend._variant_for_call(n.func.id,n,{})
                    except FrontendError as exc:
                        source_backed = (
                            str(exc).startswith(f"Cells {n.func.id} was not observed")
                            and frontend._source_backed_unobserved_helper_ok(n.func.id)
                        )
                        if str(exc).startswith("non-time specialization argument is not a literal:") or source_backed:
                            frontend._runtime_scalar_helper_inline_stack.append(n.func.id)
                            try:
                                inlined = frontend._inline_runtime_scalar_helper_call(n.func.id, n)
                                if inlined is not None:
                                    self.helper_expansion_nodes += sum(1 for _ in ast.walk(inlined))
                                    if self.helper_expansion_nodes > frontend._runtime_scalar_helper_expansion_limit:
                                        raise FrontendError(
                                            "runtime scalar helper expansion budget exceeded "
                                            f"({self.helper_expansion_nodes} > {frontend._runtime_scalar_helper_expansion_limit}) "
                                            f"while inlining {n.func.id}"
                                        )
                                    return self.visit(inlined)
                            finally:
                                popped = frontend._runtime_scalar_helper_inline_stack.pop()
                                assert popped == n.func.id
                        raise
                    if texpr is None:
                        return ast.copy_location(ast.Call(ast.Name(cuid,ast.Load()),[],[]),n)
                    # Statement-level helper lowering preserves arithmetic locals,
                    # but graph scheduling still needs the defining expression of a
                    # local used as a Cell coordinate.  Fresh helper locals are
                    # single-assignment, so substituting only here is hygienic and
                    # does not reintroduce expression duplication in the formula.
                    for _ in range(len(helper_coordinate_aliases) + 1):
                        before = ast.dump(texpr, include_attributes=False)
                        texpr = _Substitute(helper_coordinate_aliases).visit(copy.deepcopy(texpr))
                        texpr = ast.fix_missing_locations(texpr)
                        if ast.dump(texpr, include_attributes=False) == before:
                            break
                    texpr = frontend._normalize_coordinate_expression(
                        texpr, bound_names=bound_names
                    )
                    if (
                        isinstance(texpr, ast.Constant)
                        and isinstance(texpr.value, int)
                        and not isinstance(texpr.value, bool)
                    ):
                        # A constant call into a future-recursive coordinate family
                        # must remain parameterized.  Expanding it into a synthetic
                        # fixed scalar would recursively request t, t+1, t+2, ...
                        # and therefore hides the recurrence from the canonical
                        # scheduler.  Ordinary acyclic/fixed-coordinate families
                        # keep the historical specialization path.
                        if frontend._source_has_future_coordinate_dependency(cuid):
                            return ast.copy_location(
                                ast.Call(
                                    ast.Name(cuid, ast.Load()),
                                    [ast.Constant(int(texpr.value))],
                                    [],
                                ),
                                n,
                            )
                        fixed_uid = frontend._request_fixed_coordinate(cuid, int(texpr.value))
                        return ast.copy_location(ast.Call(ast.Name(fixed_uid,ast.Load()),[],[]),n)
                    return ast.copy_location(
                        ast.Call(ast.Name(cuid,ast.Load()),[self.visit(texpr)],[]), n
                    )
                return self.generic_visit(n)
            def visit_Subscript(self,n):
                # modelx Cells subscript call syntax: f[t] / f[t, aux]
                if isinstance(n.value,ast.Name) and n.value.id in frontend.base_funcs:
                    sl=n.slice
                    args=list(sl.elts) if isinstance(sl,ast.Tuple) else [sl]
                    call=ast.Call(ast.Name(n.value.id,ast.Load()),args,[])
                    return self.visit_Call(call)
                return self.generic_visit(n)
        fn=Calls().visit(fn)
        fn=ast.fix_missing_locations(fn)
        fn=self._inline_fixed_time_calls(fn)
        fn=self._inline_vector_subscripts(fn)
        return ast.fix_missing_locations(fn)

    def _inline_fixed_time_calls(self, fn: ast.FunctionDef) -> ast.FunctionDef:
        frontend=self
        class Inliner(ast.NodeTransformer):
            depth=0
            def visit_Call(self,n):
                n=self.generic_visit(n)
                if self.depth>10 or not (isinstance(n.func,ast.Name) and n.func.id in frontend.variant_trace_by_uid and len(n.args)==1):
                    return n
                arg=n.args[0]
                if not isinstance(arg,ast.Constant) or not isinstance(arg.value,int):
                    return n
                callee=frontend._specialize_function(n.func.id)
                if callee is None or not callee.args.args:
                    return n
                callee=_Substitute({"t":ast.Constant(int(arg.value))}).visit(callee)
                callee=_ConservativeFolder().visit(callee)
                if isinstance(callee,list): return n
                callee=_strip_docstring(callee)
                body=callee.body
                if len(body)==1 and isinstance(body[0],ast.Return):
                    self.depth+=1
                    val=self.visit(copy.deepcopy(body[0].value))
                    self.depth-=1
                    return ast.copy_location(val,n)
                return n
        return ast.fix_missing_locations(Inliner().visit(fn))

    def _normalize_scalar_intrinsics(self, fn: ast.FunctionDef) -> ast.FunctionDef:
        """Lower a tiny backend-neutral scalar-intrinsic subset.

        pandas/numpy missing-value spellings are source conveniences.  Only a
        numeric scalar predicate survives as ``__is_missing__(x)``; emitters map
        that semantic operation to their native floating-point implementation.
        """
        frontend = self
        float_is_builtin = frontend.refs.get('float', float) is float and not any(
            (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == 'float')
            or (isinstance(n, ast.arg) and n.arg == 'float')
            or (isinstance(n, ast.alias) and (n.asname or n.name.split('.')[0]) == 'float')
            or (isinstance(n, ast.ExceptHandler) and n.name == 'float')
            or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == 'float')
            for n in ast.walk(fn)
        )

        class Normalize(ast.NodeTransformer):
            def visit_Attribute(self, node: ast.Attribute):
                node = self.generic_visit(node)
                if not isinstance(node.value, ast.Name) or node.attr != "inf":
                    return node
                obj = frontend.refs.get(node.value.id)
                module_name = getattr(obj, "__name__", None)
                if module_name in ("math", "numpy"):
                    # Keep infinity as a backend-neutral numeric AST constant.
                    # Emitters choose their native spelling (math.inf / INFINITY).
                    return ast.copy_location(ast.Constant(value=float("inf")), node)
                return node

            def visit_Call(self, node: ast.Call):
                node = self.generic_visit(node)
                if (float_is_builtin and isinstance(node.func, ast.Name)
                    and node.func.id == 'float' and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value.strip().lower() in
                        {'inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity'}):
                    return ast.copy_location(ast.Constant(value=float(node.args[0].value)), node)
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                ):
                    receiver = node.func.value.id
                    module_name = getattr(frontend.refs.get(receiver), "__name__", None)
                    if (
                        module_name == "numpy"
                        and node.func.attr in {"maximum", "minimum"}
                        and len(node.args) == 2 and not node.keywords
                    ):
                        name = "max" if node.func.attr == "maximum" else "min"
                        return ast.copy_location(
                            ast.Call(ast.Name(name, ast.Load()), node.args, []), node
                        )
                    if (
                        module_name == "numpy"
                        and node.func.attr in {"around", "round"}
                        and 1 <= len(node.args) <= 2 and not node.keywords
                    ):
                        return ast.copy_location(
                            ast.Call(ast.Name("round", ast.Load()), node.args, []), node
                        )
                if not (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and len(node.args) == 1 and not node.keywords
                ):
                    return node
                receiver = node.func.value.id
                obj = frontend.refs.get(receiver)
                module_name = getattr(obj, "__name__", None)
                if module_name == "pandas" and node.func.attr in {"isna", "isnull"}:
                    return ast.copy_location(
                        ast.Call(ast.Name("__is_missing__", ast.Load()), [node.args[0]], []), node
                    )
                if module_name == "numpy" and node.func.attr == "isnan":
                    return ast.copy_location(
                        ast.Call(ast.Name("__is_missing__", ast.Load()), [node.args[0]], []), node
                    )
                if module_name in ("math", "numpy") and node.func.attr == "isinf":
                    return ast.copy_location(
                        ast.Call(ast.Name("__is_inf__", ast.Load()), [node.args[0]], []), node
                    )
                if module_name == "math" and node.func.attr == "erf":
                    return ast.copy_location(
                        ast.Call(ast.Name("__erf__", ast.Load()), [node.args[0]], []), node
                    )
                return node

        out = Normalize().visit(copy.deepcopy(fn))
        assert isinstance(out, ast.FunctionDef)
        return ast.fix_missing_locations(out)

    def _normalize_simple_local_syntax(self, fn: ast.FunctionDef) -> ast.FunctionDef:
        """Lower small Python-local conveniences to the verified scalar subset.

        This pass changes syntax, not model semantics:
        * fixed-arity tuple/list unpacking uses temporaries so all RHS expressions
          are evaluated left-to-right before any target is assigned;
        * numeric local AugAssign becomes an ordinary assignment;
        * a local list used only as a min/max accumulator via ``append`` is
          scalarized to repeated min/max updates.
        """
        existing = self._function_bound_names(fn)
        counter = 0

        def fresh(prefix: str) -> str:
            nonlocal counter
            while True:
                name = f"__mx_{prefix}_{counter}"
                counter += 1
                if name not in existing:
                    existing.add(name)
                    return name

        class Basic(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign):
                node = self.generic_visit(node)
                if not (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], (ast.Tuple, ast.List))
                    and isinstance(node.value, (ast.Tuple, ast.List))
                    and len(node.targets[0].elts) == len(node.value.elts)
                    and len(node.targets[0].elts) >= 1
                    and all(isinstance(x, ast.Name) for x in node.targets[0].elts)
                    and all(not isinstance(x, ast.Starred) for x in node.targets[0].elts)
                ):
                    return node
                temps = [fresh("unpack") for _ in node.value.elts]
                out: list[ast.stmt] = []
                for tmp, value in zip(temps, node.value.elts):
                    out.append(ast.copy_location(
                        ast.Assign([ast.Name(tmp, ast.Store())], value), node
                    ))
                for target, tmp in zip(node.targets[0].elts, temps):
                    out.append(ast.copy_location(
                        ast.Assign([ast.Name(target.id, ast.Store())], ast.Name(tmp, ast.Load())), node
                    ))
                return out

            def visit_AugAssign(self, node: ast.AugAssign):
                node = self.generic_visit(node)
                if not isinstance(node.target, ast.Name):
                    return node
                return ast.copy_location(
                    ast.Assign(
                        [ast.Name(node.target.id, ast.Store())],
                        ast.BinOp(ast.Name(node.target.id, ast.Load()), node.op, node.value),
                    ),
                    node,
                )

        out = Basic().visit(copy.deepcopy(fn))
        assert isinstance(out, ast.FunctionDef)
        out = ast.fix_missing_locations(out)

        # Identify list locals whose *only* reads are append operations and one
        # consistent min/max aggregate.  Anything else remains untouched/fail-closed.
        candidates: dict[str, str] = {}
        for st in out.body:
            if not (
                isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and isinstance(st.value, ast.List) and len(st.value.elts) == 1
            ):
                continue
            name = st.targets[0].id
            total_loads = sum(
                1 for n in ast.walk(out)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == name
            )
            append_loads = 0
            aggregates: list[str] = []
            for n in ast.walk(out):
                if (
                    isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == name
                    and n.func.attr == "append" and len(n.args) == 1 and not n.keywords
                ):
                    append_loads += 1
                if (
                    isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in {"min", "max"} and len(n.args) == 1
                    and isinstance(n.args[0], ast.Name) and n.args[0].id == name
                ):
                    aggregates.append(n.func.id)
            if aggregates and len(set(aggregates)) == 1 and total_loads == append_loads + len(aggregates):
                candidates[name] = aggregates[0]

        if not candidates:
            return out

        class Accumulator(ast.NodeTransformer):
            def visit_Assign(self, node: ast.Assign):
                node = self.generic_visit(node)
                if (
                    len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in candidates
                    and isinstance(node.value, ast.List) and len(node.value.elts) == 1
                ):
                    node.value = node.value.elts[0]
                return node

            def visit_Expr(self, node: ast.Expr):
                node = self.generic_visit(node)
                call = node.value
                if (
                    isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in candidates and call.func.attr == "append"
                    and len(call.args) == 1 and not call.keywords
                ):
                    name = call.func.value.id
                    return ast.copy_location(
                        ast.Assign(
                            [ast.Name(name, ast.Store())],
                            ast.Call(
                                ast.Name(candidates[name], ast.Load()),
                                [ast.Name(name, ast.Load()), call.args[0]], [],
                            ),
                        ),
                        node,
                    )
                return node

            def visit_Call(self, node: ast.Call):
                node = self.generic_visit(node)
                if (
                    isinstance(node.func, ast.Name) and node.func.id in {"min", "max"}
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in candidates and candidates[node.args[0].id] == node.func.id
                ):
                    return ast.copy_location(ast.Name(node.args[0].id, ast.Load()), node)
                return node

        out = Accumulator().visit(out)
        assert isinstance(out, ast.FunctionDef)
        return ast.fix_missing_locations(out)

    def _normalize_local_mapping_iteration(
        self, fn: ast.FunctionDef, *, operator_uid: str | None = None
    ) -> ast.FunctionDef:
        """Lower preparation-owned Mapping traversal to ordered native CSR loops.

        Supported source surfaces are deliberately narrow and semantic:
        ``sum(cell().values())``, ``for value in cell().values()``,
        ``for key in cell().keys()`` and ``for key, value in cell().items()``
        for a zero-argument local Cells mapping.  Mapping insertion order is
        preserved exactly by ``InputRegistry.local_cell_mapping_ragged``.
        """
        frontend = self
        registry = self.registry
        out = copy.deepcopy(fn)
        used_names = {
            node.id for node in ast.walk(out) if isinstance(node, ast.Name)
        } | {arg.arg for arg in out.args.args}
        counter = [0]

        def fresh(prefix: str) -> str:
            while True:
                counter[0] += 1
                name = f"__mx_{prefix}_{counter[0]}"
                if name not in used_names:
                    used_names.add(name)
                    return name

        def local_mapping_call(node: ast.AST, method: str):
            if not (
                isinstance(node, ast.Call) and not node.args and not node.keywords
                and isinstance(node.func, ast.Attribute) and node.func.attr == method
                and isinstance(node.func.value, ast.Call)
                and not node.func.value.args and not node.func.value.keywords
                and isinstance(node.func.value.func, ast.Name)
            ):
                return None
            called = node.func.value.func.id
            tr = frontend.variant_trace_by_uid.get(called)
            source_name = None
            if tr is not None and not tr.schema.parameters:
                source_name = tr.schema.name
            if source_name is None:
                source_name = getattr(frontend, "formula_uid_to_source_name", {}).get(called)
            if source_name is None and called in frontend.base_funcs:
                source_name = called
            if source_name is None:
                return None
            cell = frontend.space.cells.get(source_name)
            if cell is None or tuple(cell.parameters or ()):
                return None
            try:
                keys = registry.local_cell_mapping_ragged(source_name)
            except FrontendError:
                return None
            return source_name, keys

        def point_row(ordinal_key: str) -> ast.Call:
            return ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(ordinal_key)], [])

        def arr(key: str, index: ast.AST) -> ast.Call:
            return ast.Call(ast.Name("array_input", ast.Load()), [ast.Constant(key), index], [])

        def prepared_range(keys: tuple[str, str, str, str], location: ast.AST):
            offsets_key, _keys_key, _values_key, ordinal_key = keys
            row = point_row(ordinal_key)
            start = arr(offsets_key, copy.deepcopy(row))
            stop = arr(
                offsets_key,
                ast.BinOp(copy.deepcopy(row), ast.Add(), ast.Constant(1)),
            )
            call = ast.Call(ast.Name("range", ast.Load()), [start, stop], [])
            return ast.copy_location(call, location)

        class Lower(ast.NodeTransformer):
            def _mark_loop(self, loop: ast.For, keys: tuple[str, str, str, str]) -> ast.For:
                # Private AST metadata is compiler-generated and survives deepcopy;
                # admission/codegen still validate the range shape before trusting it.
                loop._mx_prepared_mapping_range = True
                loop._mx_prepared_mapping_keys = tuple(keys)
                return loop

            def visit_Assign(self, node: ast.Assign):
                # Rewrite exact ``target = sum(mapping_cell().values())``.  This
                # needs statement expansion, so it intentionally lives here rather
                # than as a generic expression rewrite.
                if (
                    len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "sum" and len(node.value.args) == 1
                    and not node.value.keywords
                ):
                    matched = local_mapping_call(node.value.args[0], "values")
                    if matched is not None:
                        source_name, keys = matched
                        _off, _key, values_key, _ord = keys
                        dtype = registry.specs[values_key].dtype
                        target = node.targets[0].id
                        j = fresh("map_j")
                        init = ast.copy_location(
                            ast.Assign(
                                [ast.Name(target, ast.Store())],
                                ast.Constant(0.0 if dtype == "float64" else 0),
                            ), node,
                        )
                        update = ast.Assign(
                            [ast.Name(target, ast.Store())],
                            ast.BinOp(
                                ast.Name(target, ast.Load()), ast.Add(),
                                arr(values_key, ast.Name(j, ast.Load())),
                            ),
                        )
                        loop = ast.For(
                            target=ast.Name(j, ast.Store()),
                            iter=prepared_range(keys, node),
                            body=[update], orelse=[], type_comment=None,
                        )
                        self._mark_loop(loop, keys)
                        if operator_uid is not None:
                            frontend._normalized_operators[operator_uid] = NormalizedOperator(
                                uid=operator_uid,
                                kind="prepared_mapping_iteration",
                                result_dtype=dtype,
                                result_shape="scalar",
                                static_inputs=(("source_cell", source_name), ("layout", "ordered_csr")),
                                dynamic_inputs=(),
                                provenance=("finite_prepared_mapping", "ordered_mapping_values", "native_csr"),
                                state_semantic="state_free",
                                python_supported=True,
                                cython_supported=True,
                            )
                        return [init, ast.copy_location(loop, node)]
                return self.generic_visit(node)

            def visit_For(self, node: ast.For):
                matched = None
                mode = None
                for candidate in ("items", "values", "keys"):
                    probe = local_mapping_call(node.iter, candidate)
                    if probe is not None:
                        matched = probe
                        mode = candidate
                        break
                if matched is None:
                    return self.generic_visit(node)
                source_name, keys = matched
                _off, keys_key, values_key, _ord = keys
                j = fresh("map_j")
                prefix: list[ast.stmt] = []
                if mode == "items":
                    if not (
                        isinstance(node.target, (ast.Tuple, ast.List))
                        and len(node.target.elts) == 2
                        and all(isinstance(x, ast.Name) for x in node.target.elts)
                    ):
                        return self.generic_visit(node)
                    key_target = node.target.elts[0].id
                    value_target = node.target.elts[1].id
                    prefix.extend([
                        ast.Assign([ast.Name(key_target, ast.Store())], arr(keys_key, ast.Name(j, ast.Load()))),
                        ast.Assign([ast.Name(value_target, ast.Store())], arr(values_key, ast.Name(j, ast.Load()))),
                    ])
                elif mode == "values":
                    if not isinstance(node.target, ast.Name):
                        return self.generic_visit(node)
                    prefix.append(ast.Assign(
                        [ast.Name(node.target.id, ast.Store())],
                        arr(values_key, ast.Name(j, ast.Load())),
                    ))
                else:
                    if not isinstance(node.target, ast.Name):
                        return self.generic_visit(node)
                    prefix.append(ast.Assign(
                        [ast.Name(node.target.id, ast.Store())],
                        arr(keys_key, ast.Name(j, ast.Load())),
                    ))
                body: list[ast.stmt] = []
                for child in node.body:
                    visited = self.visit(child)
                    if visited is None:
                        continue
                    if isinstance(visited, list):
                        body.extend(visited)
                    else:
                        body.append(visited)
                loop = ast.For(
                    target=ast.Name(j, ast.Store()),
                    iter=prepared_range(keys, node),
                    body=prefix + body,
                    orelse=[], type_comment=None,
                )
                self._mark_loop(loop, keys)
                if operator_uid is not None:
                    frontend._normalized_operators[operator_uid] = NormalizedOperator(
                        uid=operator_uid,
                        kind="prepared_mapping_iteration",
                        result_dtype=registry.specs[values_key].dtype,
                        result_shape="scalar",
                        static_inputs=(("source_cell", source_name), ("layout", "ordered_csr")),
                        dynamic_inputs=(),
                        provenance=("finite_prepared_mapping", f"ordered_mapping_{mode}", "native_csr"),
                        state_semantic="state_free",
                        python_supported=True,
                        cython_supported=True,
                    )
                return ast.copy_location(loop, node)

        out = Lower().visit(out)
        assert isinstance(out, ast.FunctionDef)
        return ast.fix_missing_locations(out)

    def _normalize_external_tables(
        self, fn: ast.FunctionDef, *, operator_uid: str | None = None,
        semantic_ops: Sequence[Any] = (),
    ) -> ast.FunctionDef:
        """Lower supported external DataFrame semantics before either backend sees them.

        Exact numeric ``.loc`` lookups may combine compile-time-fixed, finite
        categorical and regular numeric runtime selectors.  Multi-dimensional
        runtime keys are normalized to dense or sparse numeric inputs; pandas
        objects and label semantics never survive into the optimized program.
        """
        frontend = self
        registry = self.registry

        @dataclass(frozen=True)
        class TableSource:
            ref_name: str
            cell_name: str
            cell: Any
            sample: pd.DataFrame

        @dataclass(frozen=True)
        class RowSource:
            table: TableSource
            row_expr: ast.AST

        @dataclass(frozen=True)
        class SeriesSource:
            """Preparation-only numeric 1-D provider view.

            ``row_expr`` fixes every MultiIndex prefix level either statically or
            through a proven finite categorical selector, leaving exactly one
            residual numeric index axis.  The pandas Series itself never survives
            normalization; consumers are rewritten to existing exact-table/value
            and row-axis metadata inputs.
            """

            table: TableSource
            row_expr: ast.AST = field(compare=False, repr=False)
            fixed_levels: tuple[tuple[int, Any], ...]
            dynamic_level: int
            column: Any

            @property
            def has_dynamic_prefix(self) -> bool:
                return len(self.fixed_levels) != int(self.dynamic_level)

        @dataclass(frozen=True)
        class FiniteEnumValue:
            """Numeric code expression plus the exact finite source label domain.

            This is deliberately an upstream/frontend proof object.  The canonical
            executable graph sees only the integer ``code_expr``; string labels are
            retained solely while normalizing source control flow/table dispatch.
            """

            labels: tuple[str, ...]
            code_expr: ast.AST = field(compare=False, repr=False)
            source_name: str | None = None

        @dataclass(frozen=True)
        class FiniteNumericExactValue:
            """Prepared finite numeric selector for exact table-prefix dispatch."""

            labels: tuple[Any, ...]
            code_expr: ast.AST = field(compare=False, repr=False)
            reachable_labels: tuple[Any, ...] | None = None
            source_name: str | None = None

        @dataclass(frozen=True)
        class FiniteDispatch:
            fixed_levels: tuple[tuple[int, Any], ...]
            enum_level: int
            enum_value: FiniteEnumValue = field(compare=False, repr=False)
            numeric_level: int
            numeric_expr: ast.AST = field(compare=False, repr=False)

        class Normalize(ast.NodeTransformer):
            def __init__(self):
                self.table_aliases: dict[str, TableSource] = {}
                self.table_column_aliases: dict[str, tuple[TableSource, ast.AST]] = {}
                self.row_aliases: dict[str, RowSource] = {}
                self.series_aliases: dict[str, SeriesSource] = {}
                # Direct DataFrame-reference column views, e.g. ``s = table[col]``.
                # These are preparation-only symbolic Series objects.  Supported
                # consumers (index min/max and exact row lookup) are rewritten
                # before the ordinary expression lowerer sees the incomplete
                # one-index DataFrame operation.
                self.direct_dataframe_series_aliases: dict[str, tuple[str, ast.AST]] = {}
                self.finite_enum_aliases: dict[str, FiniteEnumValue] = {}
                self.value_aliases: dict[str, ast.AST] = {}
                self.external_cache: dict[tuple[str, str], TableSource | None] = {}
                self.series_helper_cache: dict[str, SeriesSource | None] = {}
                self.finite_enum_cache: dict[str, FiniteEnumValue | None] = {}
                self.store_counts: dict[str, int] = {}
                self.membership_guarded_rows: list[tuple[str, str]] = []
                self.active_membership_guarded_rows: list[tuple[str, str]] = []
                self.numeric_bounds_cache: dict[
                    tuple[str, tuple[str, ...]], tuple[float | None, float | None]
                ] = {}
                self.numeric_bounds_active: set[tuple[str, tuple[str, ...]]] = set()
                for child in ast.walk(fn):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        self.store_counts[child.id] = self.store_counts.get(child.id, 0) + 1
                # Preserve a tiny but important source contract: a direct
                # ``if key not in table.index: return default`` makes a following
                # lookup total over enum labels absent from the table.  We record
                # only the local aliases here; table/row identity is checked after
                # aliases have been proven by this same normalizer.
                for index, st in enumerate(fn.body):
                    if not (
                        isinstance(st, ast.If) and not st.orelse and len(st.body) == 1
                        and isinstance(st.body[0], ast.Return) and st.body[0].value is not None
                        and isinstance(st.body[0].value, ast.Constant)
                        and isinstance(st.body[0].value.value, (int, float, np.integer, np.floating))
                        and not isinstance(st.body[0].value.value, (bool, np.bool_))
                        and isinstance(st.test, ast.Compare) and len(st.test.ops) == 1
                        and isinstance(st.test.ops[0], ast.NotIn)
                        and len(st.test.comparators) == 1
                        and isinstance(st.test.left, ast.Name)
                        and isinstance(st.test.comparators[0], ast.Attribute)
                        and st.test.comparators[0].attr == "index"
                        and isinstance(st.test.comparators[0].value, ast.Name)
                    ):
                        continue
                    key_name = st.test.left.id
                    # A membership-return check is a lookup guard only when it
                    # precedes every source ``.loc[key]`` access.  Merely finding a
                    # matching ``if`` somewhere in the function would be a rather
                    # inventive interpretation of Python control flow.
                    unsafe_prior_lookup = any(
                        isinstance(child, ast.Subscript)
                        and isinstance(child.value, ast.Attribute)
                        and child.value.attr == "loc"
                        and any(
                            isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
                            and name.id == key_name
                            for name in ast.walk(child.slice)
                        )
                        for prior in fn.body[:index]
                        for child in ast.walk(prior)
                    )
                    if unsafe_prior_lookup:
                        continue
                    self.membership_guarded_rows.append(
                        (key_name, st.test.comparators[0].value.id)
                    )

            @staticmethod
            def _same_expr(left: ast.AST, right: ast.AST) -> bool:
                return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)

            def _numeric_bounds(
                self, node: ast.AST, *, _stack: tuple[str, ...] = ()
            ) -> tuple[float | None, float | None]:
                """Return sound numeric lower/upper bounds for a tiny pure subset.

                The analysis exists to prove lookup-domain preconditions, not to fold
                arbitrary formulas.  Unknown pieces remain unbounded.  Calls into source
                Cells are followed only when their body is a single pure ``return``
                expression; recursion and statementful helpers fail closed.  This lets
                formulas such as ``policy_year(t) = max(0, t // 12) + 1`` prove a
                lower bound of one without learning anything about the concrete value of
                ``t`` or relying on representative-trace coverage.
                """
                frontend._proof_diagnostics["numeric_bounds_requests"] += 1
                node = self._analysis_expand_alias(node)
                key = (ast.dump(node, include_attributes=False), _stack)
                cached = self.numeric_bounds_cache.get(key)
                if cached is not None:
                    frontend._proof_diagnostics["numeric_bounds_cache_hits"] += 1
                    return cached
                if key in self.numeric_bounds_active:
                    # Structural recurrence in evidence analysis is never a proof
                    # of finite bounds.  Cut it immediately and fail closed.
                    frontend._proof_diagnostics["numeric_bounds_cycle_cuts"] += 1
                    return None, None
                self.numeric_bounds_active.add(key)
                frontend._proof_diagnostics["numeric_bounds_active_peak"] = max(
                    frontend._proof_diagnostics["numeric_bounds_active_peak"],
                    len(self.numeric_bounds_active),
                )
                try:
                    result = self._numeric_bounds_uncached(node, _stack=_stack)
                finally:
                    self.numeric_bounds_active.remove(key)
                self.numeric_bounds_cache[key] = result
                frontend._proof_diagnostics["numeric_bounds_cache_stores"] += 1
                return result

            def _numeric_bounds_uncached(
                self, node: ast.AST, *, _stack: tuple[str, ...]
            ) -> tuple[float | None, float | None]:
                ok_static, static_value = frontend._proven_static_expr(node)
                if (
                    ok_static
                    and isinstance(static_value, (int, float, np.integer, np.floating))
                    and not isinstance(static_value, (bool, np.bool_))
                    and np.isfinite(float(static_value))
                ):
                    value = float(static_value)
                    return value, value
                if (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"global_input", "array_input", "table_input"}
                    and node.args and isinstance(node.args[0], ast.Constant)
                ):
                    spec = registry.specs.get(node.args[0].value)
                    sem = None if spec is None else spec.normalized_table
                    if sem is not None and sem.result_kind == "axis_metadata_scalar":
                        try:
                            arr = np.asarray(spec.getter(frontend.run_keys), dtype=float)
                            if arr.size and np.isfinite(arr).all():
                                return float(np.min(arr)), float(np.max(arr))
                        except Exception:
                            pass
                if isinstance(node, ast.Name) and operator_uid is not None:
                    guarded_lo = getattr(
                        frontend, "source_guarded_parameter_lower_bounds", {}
                    ).get((operator_uid, node.id))
                    if guarded_lo is not None:
                        # Preserve any independently proved upper bound. The source-call
                        # evidence only strengthens the lower edge; it does not invent a
                        # finite maximum.
                        upper = None
                        mx_domain = getattr(
                            frontend, "modelx_cython_parameter_domains", {}
                        ).get((operator_uid, node.id))
                        if mx_domain is not None and mx_domain[1] is not None:
                            upper = float(mx_domain[1])
                        if node.id == "t":
                            dom = (
                                frontend.specialized_coordinate_domains.get(fn.name)
                                or frontend.coordinate_domains.get(fn.name)
                                or frontend.primary_coordinate_domain
                            )
                            if dom is not None:
                                upper = float(dom.hi) if upper is None else min(upper, float(dom.hi))
                            elif upper is None:
                                _lo, hi = frontend.primary_coordinate_bounds
                                upper = None if hi is None else float(hi)
                        return float(guarded_lo), upper
                if isinstance(node, ast.Name) and node.id == "t":
                    dom = (
                        frontend.specialized_coordinate_domains.get(fn.name)
                        or frontend.coordinate_domains.get(fn.name)
                        or frontend.primary_coordinate_domain
                    )
                    if dom is not None:
                        return float(dom.lo), float(dom.hi)
                    lo, hi = frontend.primary_coordinate_bounds
                    return (None if lo is None else float(lo), None if hi is None else float(hi))
                if isinstance(node, ast.Name) and operator_uid is not None:
                    # The fast modelx-cython frontend may own a complete finite
                    # parameter interval even when this formula is not a canonical
                    # time-coordinate variant.  Provider normalization may consume
                    # that authority exactly as it consumes source-proven ``t``
                    # domains.  Representative trace extrema never enter this map;
                    # the producer admits trace min/max only when the trace covers
                    # the complete frozen RunDomain.
                    mx_domain = getattr(
                        frontend, "modelx_cython_parameter_domains", {}
                    ).get((operator_uid, node.id))
                    if mx_domain is not None:
                        lo, hi, _authority = mx_domain
                        if lo is not None and hi is not None:
                            return float(lo), float(hi)
                if (
                    isinstance(node, ast.Subscript)
                    and frontend.model_point_row is not None
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == frontend.model_point_row.cell_name
                    and not node.value.args and not node.value.keywords
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                ):
                    return frontend._proven_numeric_point_field_bounds(str(node.slice.value))
                if isinstance(node, ast.Constant):
                    value = node.value
                    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
                        value, (bool, np.bool_)
                    ):
                        value = float(value)
                        if np.isfinite(value):
                            return value, value
                    return None, None
                if isinstance(node, ast.UnaryOp):
                    lo, hi = self._numeric_bounds(node.operand, _stack=_stack)
                    if isinstance(node.op, ast.UAdd):
                        return lo, hi
                    if isinstance(node.op, ast.USub):
                        return (None if hi is None else -hi, None if lo is None else -lo)
                    return None, None
                if isinstance(node, ast.BinOp):
                    llo, lhi = self._numeric_bounds(node.left, _stack=_stack)
                    rlo, rhi = self._numeric_bounds(node.right, _stack=_stack)
                    if isinstance(node.op, ast.Add):
                        return (
                            None if llo is None or rlo is None else llo + rlo,
                            None if lhi is None or rhi is None else lhi + rhi,
                        )
                    if isinstance(node.op, ast.Sub):
                        return (
                            None if llo is None or rhi is None else llo - rhi,
                            None if lhi is None or rlo is None else lhi - rlo,
                        )
                    if isinstance(node.op, ast.Mult) and None not in (llo, lhi, rlo, rhi):
                        vals = (llo * rlo, llo * rhi, lhi * rlo, lhi * rhi)
                        return min(vals), max(vals)
                    if isinstance(node.op, ast.FloorDiv) and rlo == rhi and rlo is not None and rlo > 0:
                        return (
                            None if llo is None else float(np.floor(llo / rlo)),
                            None if lhi is None else float(np.floor(lhi / rlo)),
                        )
                    return None, None
                if isinstance(node, ast.IfExp):
                    blo, bhi = self._numeric_bounds(node.body, _stack=_stack)
                    olo, ohi = self._numeric_bounds(node.orelse, _stack=_stack)
                    return (
                        None if blo is None or olo is None else min(blo, olo),
                        None if bhi is None or ohi is None else max(bhi, ohi),
                    )
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    name = node.func.id
                    if name in {"int", "float"} and len(node.args) == 1 and not node.keywords:
                        return self._numeric_bounds(node.args[0], _stack=_stack)
                    if name in {"max", "min"} and node.args and not node.keywords:
                        bounds = [self._numeric_bounds(arg, _stack=_stack) for arg in node.args]
                        los = [lo for lo, _ in bounds if lo is not None]
                        his = [hi for _, hi in bounds if hi is not None]
                        if name == "max":
                            # max(x, c) is >= c even when x is otherwise unbounded.
                            lo = max(los) if los else None
                            hi = max(his) if len(his) == len(bounds) else None
                        else:
                            lo = min(los) if len(los) == len(bounds) else None
                            # min(x, c) is <= c even when x is otherwise unbounded.
                            hi = min(his) if his else None
                        return lo, hi
                    if name == "abs" and len(node.args) == 1 and not node.keywords:
                        lo, hi = self._numeric_bounds(node.args[0], _stack=_stack)
                        upper = None
                        if lo is not None and hi is not None:
                            upper = max(abs(lo), abs(hi))
                        return 0.0, upper
                    expanded = frontend._expand_pure_single_return_call(node, _stack=_stack)
                    if expanded is not None:
                        value, stack_token = expanded
                        return self._numeric_bounds(value, _stack=_stack + (stack_token,))
                return None, None

            def _rewrite_interp_lookup_function(self, fn: ast.FunctionDef) -> ast.FunctionDef:
                """Recognize a bounded pivot interpolation idiom as InterpLookup1D.

                This is intentionally a mathematical recognizer, not general pandas
                lowering.  It accepts a static DataFrame prefix slice, a sorted
                numeric pivot axis, nearest-end-segment extrapolation, and either
                linear or multiplicative log-linear interpolation between adjacent
                pivots.  Any structural deviation fails closed.
                """
                work = _strip_docstring(copy.deepcopy(fn))
                body = list(work.body)
                # Allow one bounded query alias (``x = age(t)``) and one bounded
                # interpolation-result temporary before a final scalar wrapper.
                # This is still a mathematical recognizer, not general local
                # dataflow lowering.
                if len(body) < 6 or len(body) > 8:
                    return fn

                # sub = table.loc[static_prefix]
                sub_st = body[0]
                if not (
                    isinstance(sub_st, ast.Assign) and len(sub_st.targets) == 1
                    and isinstance(sub_st.targets[0], ast.Name)
                ):
                    return fn
                sub_name = sub_st.targets[0].id
                row = self._row_proxy(sub_st.value)
                if row is None:
                    return fn

                # The interpolation buffer itself remains statically materialized,
                # but a source-proven finite categorical prefix may select among a
                # bounded family of those buffers at runtime. This keeps pandas and
                # semantic labels in preparation while the generated program sees
                # only integer category codes.
                prefix_expr = self._expand_alias(copy.deepcopy(row.row_expr))
                prefix_nodes = list(prefix_expr.elts) if isinstance(prefix_expr, ast.Tuple) else [prefix_expr]
                row_dimensions: list[
                    tuple[Any | None, FiniteEnumValue | FiniteNumericExactValue | None]
                ] = []
                for level, part in enumerate(prefix_nodes):
                    ok, value = self._static_value(part)
                    if ok:
                        row_dimensions.append((value, None))
                        continue
                    enum_value = self._finite_enum(part)
                    if enum_value is not None:
                        row_dimensions.append((None, enum_value))
                        continue
                    # Numeric policy selectors (for example occupation class or
                    # deferred period) may also define a finite table prefix. Reuse
                    # the same complete-RunDomain proof as exact irregular lookup
                    # rather than requiring them to masquerade as categories.
                    index = row.table.sample.index
                    if isinstance(index, pd.MultiIndex):
                        labels = tuple(sorted(set(index.get_level_values(level))))
                    elif level == 0:
                        labels = tuple(sorted(set(index)))
                    else:
                        return fn
                    proof = self._complete_run_finite_numeric_selector(part, labels)
                    if proof is None:
                        return fn
                    source_labels, runtime_expr, reachable_labels = proof
                    source_labels, code_expr = self._finite_numeric_exact_code(
                        runtime_expr, source_labels
                    )
                    source_name = None
                    expanded_part = self._expand_alias(copy.deepcopy(part))
                    if isinstance(expanded_part, ast.Call) and isinstance(expanded_part.func, ast.Name):
                        source_name = getattr(frontend, "formula_uid_to_source_name", {}).get(
                            expanded_part.func.id, expanded_part.func.id
                        )
                    row_dimensions.append((None, FiniteNumericExactValue(
                        labels=source_labels,
                        code_expr=code_expr,
                        reachable_labels=reachable_labels,
                        source_name=source_name,
                    )))

                # ages = sorted(int(a) for a in sub.index)
                ages_st = body[1]
                if not (
                    isinstance(ages_st, ast.Assign) and len(ages_st.targets) == 1
                    and isinstance(ages_st.targets[0], ast.Name)
                    and isinstance(ages_st.value, ast.Call)
                    and isinstance(ages_st.value.func, ast.Name) and ages_st.value.func.id == "sorted"
                    and len(ages_st.value.args) == 1 and not ages_st.value.keywords
                    and isinstance(ages_st.value.args[0], ast.GeneratorExp)
                ):
                    return fn
                ages_name = ages_st.targets[0].id
                gen = ages_st.value.args[0]
                if len(gen.generators) != 1:
                    return fn
                comp = gen.generators[0]
                if not (
                    isinstance(comp.target, ast.Name) and not comp.ifs and not comp.is_async
                    and isinstance(comp.iter, ast.Attribute) and comp.iter.attr == "index"
                    and isinstance(comp.iter.value, ast.Name) and comp.iter.value.id == sub_name
                    and isinstance(gen.elt, ast.Call) and isinstance(gen.elt.func, ast.Name)
                    and gen.elt.func.id in {"int", "float"} and len(gen.elt.args) == 1
                    and not gen.elt.keywords and isinstance(gen.elt.args[0], ast.Name)
                    and gen.elt.args[0].id == comp.target.id
                ):
                    return fn

                cursor = 2
                query_alias: ast.Assign | None = None
                if (
                    cursor + 1 < len(body)
                    and isinstance(body[cursor], ast.Assign)
                    and len(body[cursor].targets) == 1
                    and isinstance(body[cursor].targets[0], ast.Name)
                    and isinstance(body[cursor + 1], ast.If)
                ):
                    query_alias = body[cursor]
                    cursor += 1

                if cursor >= len(body):
                    return fn
                bracket = body[cursor]
                cursor += 1
                if not (
                    isinstance(bracket, ast.If) and len(bracket.body) == 1
                    and len(bracket.orelse) == 1 and isinstance(bracket.orelse[0], ast.If)
                ):
                    return fn
                outer = bracket
                inner = bracket.orelse[0]

                def axis_sub(expr: ast.AST, index: int) -> bool:
                    if not (
                        isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name)
                        and expr.value.id == ages_name
                    ):
                        return False
                    try:
                        return int(ast.literal_eval(expr.slice)) == index
                    except Exception:
                        return False

                if not (
                    isinstance(outer.test, ast.Compare) and len(outer.test.ops) == 1
                    and isinstance(outer.test.ops[0], ast.LtE) and len(outer.test.comparators) == 1
                    and axis_sub(outer.test.comparators[0], 0)
                ):
                    return fn
                query = copy.deepcopy(outer.test.left)
                if query_alias is not None:
                    alias_name = query_alias.targets[0].id
                    if not isinstance(query, ast.Name) or query.id != alias_name:
                        return fn
                if not (
                    isinstance(inner.test, ast.Compare) and len(inner.test.ops) == 1
                    and isinstance(inner.test.ops[0], ast.GtE) and len(inner.test.comparators) == 1
                    and self._same_expr(inner.test.left, query) and axis_sub(inner.test.comparators[0], -1)
                    and len(inner.body) == 1 and len(inner.orelse) == 2
                ):
                    return fn

                def pair_assign(st: ast.stmt, left_idx: int, right_idx: int):
                    if not (
                        isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], (ast.Tuple, ast.List))
                        and len(st.targets[0].elts) == 2
                        and all(isinstance(x, ast.Name) for x in st.targets[0].elts)
                        and isinstance(st.value, (ast.Tuple, ast.List)) and len(st.value.elts) == 2
                        and axis_sub(st.value.elts[0], left_idx) and axis_sub(st.value.elts[1], right_idx)
                    ):
                        return None
                    return st.targets[0].elts[0].id, st.targets[0].elts[1].id

                first_pair = pair_assign(outer.body[0], 0, 1)
                last_pair = pair_assign(inner.body[0], -2, -1)
                if first_pair is None or last_pair != first_pair:
                    return fn
                lo_name, hi_name = first_pair

                def interior_pick(st: ast.stmt, func_name: str, op_type, target_name: str) -> bool:
                    if not (
                        isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], ast.Name) and st.targets[0].id == target_name
                        and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name)
                        and st.value.func.id == func_name and len(st.value.args) == 1 and not st.value.keywords
                        and isinstance(st.value.args[0], ast.GeneratorExp)
                    ):
                        return False
                    g = st.value.args[0]
                    if len(g.generators) != 1:
                        return False
                    c = g.generators[0]
                    if not (
                        isinstance(c.target, ast.Name) and isinstance(c.iter, ast.Name)
                        and c.iter.id == ages_name and len(c.ifs) == 1 and not c.is_async
                        and isinstance(g.elt, ast.Name) and g.elt.id == c.target.id
                    ):
                        return False
                    cond = c.ifs[0]
                    return (
                        isinstance(cond, ast.Compare) and len(cond.ops) == 1
                        and isinstance(cond.ops[0], op_type) and len(cond.comparators) == 1
                        and isinstance(cond.left, ast.Name) and cond.left.id == c.target.id
                        and self._same_expr(cond.comparators[0], query)
                    )

                if not interior_pick(inner.orelse[0], "max", ast.LtE, lo_name):
                    return fn
                if not interior_pick(inner.orelse[1], "min", ast.Gt, hi_name):
                    return fn

                def value_assign(st: ast.stmt, index_name: str):
                    if not (
                        isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], ast.Name)
                    ):
                        return None
                    expr = st.value
                    if (
                        isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
                        and expr.func.id == "float" and len(expr.args) == 1 and not expr.keywords
                    ):
                        expr = expr.args[0]
                    if not (
                        isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Attribute)
                        and expr.value.attr == "loc" and isinstance(expr.value.value, ast.Name)
                        and expr.value.value.id == sub_name and isinstance(expr.slice, ast.Tuple)
                        and len(expr.slice.elts) == 2 and isinstance(expr.slice.elts[0], ast.Name)
                        and expr.slice.elts[0].id == index_name
                    ):
                        return None
                    return st.targets[0].id, self._expand_alias(copy.deepcopy(expr.slice.elts[1]))

                if cursor + 1 >= len(body):
                    return fn
                lo_value = value_assign(body[cursor], lo_name)
                hi_value = value_assign(body[cursor + 1], hi_name)
                cursor += 2
                if lo_value is None or hi_value is None or not self._same_expr(lo_value[1], hi_value[1]):
                    return fn
                r_lo_name, column_expr = lo_value
                r_hi_name, _ = hi_value
                ok_column, static_column = self._static_value(column_expr)
                column_enum = None if ok_column else self._finite_enum(column_expr)
                if ok_column:
                    if static_column not in row.table.sample.columns:
                        return fn
                    column_choices = ((static_column, None),)
                elif column_enum is not None:
                    if any(label not in row.table.sample.columns for label in column_enum.labels):
                        return fn
                    column_choices = tuple((label, code) for code, label in enumerate(column_enum.labels))
                else:
                    return fn
                tail = body[cursor:]
                interpolation_temp: str | None = None
                if (
                    len(tail) == 1
                    and isinstance(tail[0], ast.Return)
                    and tail[0].value is not None
                ):
                    ret = tail[0]
                    interpolation_expr = ret.value
                elif (
                    len(tail) == 2
                    and isinstance(tail[0], ast.Assign)
                    and len(tail[0].targets) == 1
                    and isinstance(tail[0].targets[0], ast.Name)
                    and isinstance(tail[1], ast.Return)
                    and tail[1].value is not None
                ):
                    interpolation_temp = tail[0].targets[0].id
                    interpolation_expr = tail[0].value
                    ret = tail[1]
                    if not any(
                        isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id == interpolation_temp
                        for node in ast.walk(ret.value)
                    ):
                        return fn
                else:
                    return fn

                frac = ast.BinOp(
                    left=ast.BinOp(copy.deepcopy(query), ast.Sub(), ast.Name(lo_name, ast.Load())),
                    op=ast.Div(),
                    right=ast.BinOp(ast.Name(hi_name, ast.Load()), ast.Sub(), ast.Name(lo_name, ast.Load())),
                )

                log_expr = ast.BinOp(
                    left=ast.Name(r_lo_name, ast.Load()), op=ast.Mult(),
                    right=ast.BinOp(
                        left=ast.BinOp(ast.Name(r_hi_name, ast.Load()), ast.Div(), ast.Name(r_lo_name, ast.Load())),
                        op=ast.Pow(), right=frac,
                    ),
                )
                linear_expr = ast.BinOp(
                    left=ast.Name(r_lo_name, ast.Load()), op=ast.Add(),
                    right=ast.BinOp(
                        left=ast.BinOp(ast.Name(r_hi_name, ast.Load()), ast.Sub(), ast.Name(r_lo_name, ast.Load())),
                        op=ast.Mult(), right=frac,
                    ),
                )
                # Python's left-associative ``*``/``/`` parsing also commonly
                # spells the same linear interpolation as
                # ``delta * (x - lo) / (hi - lo)``.  Preserve that exact source
                # form as a second bounded recognizer shape rather than asking
                # users to parenthesize algebra for the compiler's convenience.
                linear_expr_left_assoc = ast.BinOp(
                    left=ast.Name(r_lo_name, ast.Load()), op=ast.Add(),
                    right=ast.BinOp(
                        left=ast.BinOp(
                            left=ast.BinOp(
                                ast.Name(r_hi_name, ast.Load()), ast.Sub(), ast.Name(r_lo_name, ast.Load())
                            ),
                            op=ast.Mult(),
                            right=ast.BinOp(copy.deepcopy(query), ast.Sub(), ast.Name(lo_name, ast.Load())),
                        ),
                        op=ast.Div(),
                        right=ast.BinOp(ast.Name(hi_name, ast.Load()), ast.Sub(), ast.Name(lo_name, ast.Load())),
                    ),
                )
                if self._same_expr(interpolation_expr, log_expr):
                    mode = "log_linear"
                    mode_code = 1
                elif (
                    self._same_expr(interpolation_expr, linear_expr)
                    or self._same_expr(interpolation_expr, linear_expr_left_assoc)
                ):
                    mode = "linear"
                    mode_code = 0
                else:
                    return fn

                # Cartesian-expand only the finite semantic selectors. Each leaf
                # reuses the existing InterpLookup1D materializer/intrinsic; runtime
                # dispatch is integer-only and bounded by the proven category domains.
                import itertools
                dimension_choices: list[tuple[tuple[Any, int | None], ...]] = []
                dynamic_selectors: list[
                    tuple[int, FiniteEnumValue | FiniteNumericExactValue]
                ] = []
                for dim, (static_value, enum_value) in enumerate(row_dimensions):
                    if enum_value is None:
                        dimension_choices.append(((static_value, None),))
                    else:
                        dimension_choices.append(tuple((label, code) for code, label in enumerate(enum_value.labels)))
                        dynamic_selectors.append((dim, enum_value))

                leaves: list[tuple[tuple[int, ...], ast.AST]] = []
                for row_choice in itertools.product(*dimension_choices):
                    row_values = tuple(value for value, _code in row_choice)
                    row_key = row_values[0] if len(row_values) == 1 else row_values
                    row_codes = tuple(
                        int(code) for _value, code in row_choice if code is not None
                    )
                    for column, column_code in column_choices:
                        axis_key, value_key = registry.external_dataframe_interp_lookup(
                            row.table.ref_name, row.table.cell_name, row.table.cell,
                            row_key=row_key, column=column, interpolation=mode,
                            extrapolation="endpoint_segment",
                        )
                        call = ast.Call(
                            func=ast.Name("__interp_lookup_1d__", ast.Load()),
                            args=[ast.Constant(axis_key), ast.Constant(value_key), copy.deepcopy(query), ast.Constant(mode_code)],
                            keywords=[],
                        )
                        codes = row_codes + (() if column_code is None else (int(column_code),))
                        leaves.append((codes, call))

                selectors = [copy.deepcopy(enum.code_expr) for _dim, enum in dynamic_selectors]
                if column_enum is not None:
                    selectors.append(copy.deepcopy(column_enum.code_expr))
                if not selectors:
                    result = leaves[0][1]
                else:
                    # Build a total dispatch over the finite product. The final leaf
                    # is the exhaustive fallthrough because every selector is already
                    # range-proven by the category ABI.
                    result = leaves[-1][1]
                    for codes, branch in reversed(leaves[:-1]):
                        tests = [
                            ast.Compare(copy.deepcopy(selector), [ast.Eq()], [ast.Constant(code)])
                            for selector, code in zip(selectors, codes)
                        ]
                        test = tests[0] if len(tests) == 1 else ast.BoolOp(ast.And(), tests)
                        result = ast.IfExp(test=test, body=branch, orelse=result)

                if interpolation_temp is not None:
                    class ReplaceInterpolationTemp(ast.NodeTransformer):
                        def visit_Name(self, node: ast.Name):
                            if isinstance(node.ctx, ast.Load) and node.id == interpolation_temp:
                                return ast.copy_location(copy.deepcopy(result), node)
                            return node

                    final_expr = ReplaceInterpolationTemp().visit(copy.deepcopy(ret.value))
                else:
                    final_expr = result

                out = copy.deepcopy(fn)
                out_body: list[ast.stmt] = []
                if query_alias is not None:
                    out_body.append(copy.deepcopy(query_alias))
                out_body.append(ast.copy_location(ast.Return(final_expr), ret))
                out.body = out_body
                return ast.fix_missing_locations(out)

            def _finite_step_integer_parameter(
                self, table: TableSource, level: int, expr: ast.AST,
            ) -> FiniteNumericExactValue | None:
                """Return a finite exact numeric selector for one step-lookup prefix.

                This is shared physical-planning evidence.  Source recognizers should
                only establish *what* the lookup means; this helper establishes whether
                a dynamic integer prefix can be represented by bounded native dispatch.
                """
                expanded = self._expand_alias(copy.deepcopy(expr))
                if not (operator_uid is not None and isinstance(expanded, ast.Name)):
                    return None
                key = (operator_uid, expanded.id)
                domain = tuple(getattr(
                    frontend, "finite_symbolic_parameter_domains", {}
                ).get(key, ()))
                if not domain or len(domain) > 32 or any(type(value) is not int for value in domain):
                    return None

                level_labels = tuple(sorted(set(
                    table.sample.index.get_level_values(int(level))
                )))
                try:
                    source_labels, numeric_labels = self._finite_numeric_exact_labels(level_labels)
                except FrontendError:
                    return None
                source_by_numeric = {
                    int(numeric): source
                    for source, numeric in zip(source_labels, numeric_labels)
                }
                if any(int(value) not in source_by_numeric for value in domain):
                    return None
                selected = tuple(source_by_numeric[int(value)] for value in sorted(domain))
                try:
                    selected, code_expr = self._finite_numeric_exact_code(expanded, selected)
                except FrontendError:
                    return None
                return FiniteNumericExactValue(
                    labels=selected,
                    code_expr=code_expr,
                    reachable_labels=selected,
                    source_name=getattr(
                        frontend, "finite_symbolic_parameter_domain_sources", {}
                    ).get(key, "selected_closure_finite_integer_parameter_domain"),
                )

            def _plan_step_lookup_prefix(
                self, table: TableSource, prefix_expr: ast.AST,
            ) -> tuple[Any, tuple[tuple[tuple[int, Any], ...], tuple[tuple[int, Any], ...]] | None] | None:
                """Plan a MultiIndex prefix for the shared StepLookup1D physicalizer.

                Returns ``(row_key, dispatch)``.  ``dispatch`` is ``None`` for a
                completely static prefix; otherwise it contains fixed levels and
                finite dynamic selectors.  Both legacy source syntax and semantic
                :class:`AsofLookupIR` use this exact planner.
                """
                if not isinstance(table.sample.index, pd.MultiIndex):
                    return None
                nlevels = int(table.sample.index.nlevels)
                prefix = self._expand_alias(copy.deepcopy(prefix_expr))
                parts = list(prefix.elts) if isinstance(prefix, ast.Tuple) else [prefix]
                if nlevels < 2 or len(parts) != nlevels - 1:
                    return None

                fixed: dict[int, Any] = {}
                dynamic: list[tuple[int, FiniteEnumValue | FiniteNumericExactValue]] = []
                for level, expr in enumerate(parts):
                    ok, value = self._static_value(expr)
                    if ok:
                        fixed[int(level)] = value
                        continue
                    selector: FiniteEnumValue | FiniteNumericExactValue | None = self._finite_enum(expr)
                    if selector is None:
                        selector = self._finite_step_integer_parameter(table, int(level), expr)
                    if selector is None:
                        return None
                    dynamic.append((int(level), selector))

                if not dynamic:
                    values = [fixed[level] for level in range(nlevels - 1)]
                    row_key = values[0] if len(values) == 1 else tuple(values)
                    return row_key, None

                combinations = math.prod(len(selector.labels) for _level, selector in dynamic)
                if combinations <= 0 or combinations > 64:
                    return None
                return None, (tuple(sorted(fixed.items())), tuple(dynamic))

            def _step_lookup_row_choices(
                self, table: TableSource, row_key: Any, whole_table: bool,
                dispatch,
            ) -> tuple[
                tuple[tuple[tuple[int, ...], Any, bool, pd.DataFrame | pd.Series], ...],
                tuple[ast.AST, ...],
            ]:
                """Physicalize finite row-prefix dispatch without creating lookup leaves."""
                if dispatch is None:
                    sub = table.sample if whole_table else table.sample.loc[row_key]
                    if not isinstance(sub, (pd.DataFrame, pd.Series)):
                        raise FrontendError("step lookup did not leave a residual row axis")
                    return (((), row_key, whole_table, sub),), ()

                import itertools
                fixed_items, dynamic = dispatch
                nlevels = int(table.sample.index.nlevels)
                choices = [
                    tuple((code, label) for code, label in enumerate(selector.labels))
                    for _level, selector in dynamic
                ]
                rows: list[tuple[tuple[int, ...], Any, bool, pd.DataFrame | pd.Series]] = []
                for branch in itertools.product(*choices):
                    selected = dict(fixed_items)
                    codes: list[int] = []
                    labels: list[Any] = []
                    for (level, _selector), (code, label) in zip(dynamic, branch):
                        selected[int(level)] = label
                        codes.append(int(code))
                        labels.append(label)
                    values: list[Any] = []
                    for level in range(nlevels - 1):
                        if level not in selected:  # pragma: no cover - planner proves this
                            raise FrontendError("finite step-dispatch prefix has an unproved row level")
                        values.append(selected[level])
                    branch_row_key = values[0] if len(values) == 1 else tuple(values)
                    try:
                        sub = table.sample.loc[branch_row_key]
                    except Exception as exc:
                        raise FrontendError(
                            f"finite step selector combination {tuple(labels)!r} has no row slice in "
                            f"{table.ref_name}.{table.cell_name}"
                        ) from exc
                    if not isinstance(sub, (pd.DataFrame, pd.Series)):
                        raise FrontendError(
                            "finite step selector combination did not leave a residual row axis"
                        )
                    rows.append((tuple(codes), branch_row_key, False, sub))
                if not rows:
                    raise FrontendError("finite step selector has an empty domain")
                selectors = tuple(
                    self.visit(copy.deepcopy(selector.code_expr))
                    for _level, selector in dynamic
                )
                return tuple(rows), selectors

            def _plan_step_lookup_columns(
                self, table: TableSource, column_expr: ast.AST,
            ) -> tuple[tuple[tuple[Any, int | None], ...], FiniteEnumValue | None] | None:
                """Resolve a static or finite dynamic value-column selector once."""
                ok_column, static_column = self._static_value(column_expr)
                if ok_column:
                    if static_column not in table.sample.columns:
                        return None
                    return ((static_column, None),), None
                column_enum = self._finite_enum(column_expr)
                if column_enum is None or any(
                    label not in table.sample.columns for label in column_enum.labels
                ):
                    return None
                return (
                    tuple((label, int(code)) for code, label in enumerate(column_enum.labels)),
                    column_enum,
                )

            def _step_lookup_leaf_call(
                self, table: TableSource, *, row_key: Any, whole_table: bool,
                column: Any, query: ast.AST, below_first: str,
                default_value: float | None,
            ) -> ast.Call:
                """Create one prepared StepLookup1D leaf using the existing ABI."""
                axis_key, value_key = registry.external_dataframe_step_lookup(
                    table.ref_name, table.cell_name, table.cell,
                    row_key=row_key, column=column, below_first=below_first,
                    default_value=default_value, whole_table=whole_table,
                )
                return ast.Call(
                    func=ast.Name("__step_lookup_1d__", ast.Load()),
                    args=[
                        ast.Constant(axis_key), ast.Constant(value_key), copy.deepcopy(query),
                        ast.Constant(0.0 if default_value is None else float(default_value)),
                        ast.Constant(1 if below_first == "first" else 0),
                    ], keywords=[],
                )

            def _emit_step_lookup_dispatch(
                self, table: TableSource, *, row_choices, row_selectors,
                column_spec, query: ast.AST, below_first: str,
                default_value: float | None,
            ) -> tuple[ast.AST, int]:
                """Emit one total finite dispatch over existing StepLookup1D leaves."""
                column_choices, column_enum = column_spec
                selectors = list(row_selectors)
                if column_enum is not None:
                    selectors.append(self.visit(copy.deepcopy(column_enum.code_expr)))

                leaves: list[tuple[tuple[int, ...], ast.AST]] = []
                for row_codes, branch_row_key, branch_whole_table, _sub in row_choices:
                    for column, column_code in column_choices:
                        call = self._step_lookup_leaf_call(
                            table, row_key=branch_row_key, whole_table=branch_whole_table,
                            column=column, query=query, below_first=below_first,
                            default_value=default_value,
                        )
                        codes = row_codes + (() if column_code is None else (int(column_code),))
                        leaves.append((codes, call))
                if not leaves:
                    raise FrontendError("step lookup physical plan has no leaves")
                if not selectors:
                    return leaves[0][1], len(leaves)

                # Every selector is independently range-proven by the finite-category
                # ABI.  The final leaf is therefore an exhaustive fallthrough.
                result = leaves[-1][1]
                for codes, branch in reversed(leaves[:-1]):
                    tests = [
                        ast.Compare(copy.deepcopy(selector), [ast.Eq()], [ast.Constant(code)])
                        for selector, code in zip(selectors, codes)
                    ]
                    test = tests[0] if len(tests) == 1 else ast.BoolOp(ast.And(), tests)
                    result = ast.IfExp(test=test, body=branch, orelse=result)
                return result, len(leaves)

            def _rewrite_step_lookup_result(
                self, fn: ast.FunctionDef, *, prefix: Sequence[ast.stmt], origin: ast.AST,
                table: TableSource, row_key: Any, whole_table: bool, dispatch,
                column_spec, query: ast.AST, below_first: str,
                default_value: float | None,
            ) -> ast.FunctionDef:
                """Shared function rewrite used by all StepLookup1D source frontends."""
                row_choices, row_selectors = self._step_lookup_row_choices(
                    table, row_key, whole_table, dispatch
                )
                result, _leaf_count = self._emit_step_lookup_dispatch(
                    table, row_choices=row_choices, row_selectors=row_selectors,
                    column_spec=column_spec, query=query, below_first=below_first,
                    default_value=default_value,
                )
                out = copy.deepcopy(fn)
                out.body = copy.deepcopy(list(prefix)) + [
                    ast.copy_location(ast.Return(result), origin)
                ]
                return ast.fix_missing_locations(out)

            def _rewrite_semantic_asof_lookup_function(self, fn: ast.FunctionDef) -> ast.FunctionDef:
                """Physicalize one first-class :class:`AsofLookupIR`.

                Semantic recognition remains source-only.  All row-prefix planning,
                value-column planning, prepared StepLookup1D leaves and finite
                dispatch are delegated to the shared physical planner used by the
                legacy step-lookup syntax.  ``missing='raise'`` is represented by a
                native guard only when the ordered query cannot be proved in-domain.
                """
                ops = tuple(op for op in semantic_ops if isinstance(op, AsofLookupIR))
                if len(ops) != 1:
                    return fn
                op = ops[0]
                if (
                    op.direction != "backward"
                    or not op.allow_exact
                    or op.missing != "raise"
                    or (op.source_ast_sha256 and op.source_ast_sha256 != ast_fingerprint(fn))
                ):
                    return fn

                table = self._table(copy.deepcopy(op.table_expr))
                if table is None or not isinstance(table.sample.index, pd.MultiIndex):
                    return fn
                prefix_plan = self._plan_step_lookup_prefix(table, op.equality_prefix)
                if prefix_plan is None:
                    return fn
                row_key, dispatch = prefix_plan
                try:
                    row_choices, row_selectors = self._step_lookup_row_choices(
                        table, row_key, False, dispatch
                    )
                except FrontendError:
                    return fn

                column_spec = self._plan_step_lookup_columns(table, op.value_selector)
                if column_spec is None:
                    return fn

                axis_mins: list[float] = []
                for _codes, _row_key, _whole_table, sub in row_choices:
                    try:
                        values = [float(value) for value in sub.index]
                    except Exception:
                        return fn
                    if not values or not all(np.isfinite(value) for value in values):
                        return fn
                    axis_mins.append(float(min(values)))
                if not axis_mins:
                    return fn

                lower_bound, upper_bound = self._numeric_bounds(
                    copy.deepcopy(op.ordered_query)
                )
                required_min = max(axis_mins)
                guard_required = lower_bound is None or float(lower_bound) < required_min

                result, leaf_count = self._emit_step_lookup_dispatch(
                    table,
                    row_choices=row_choices,
                    row_selectors=row_selectors,
                    column_spec=column_spec,
                    query=op.ordered_query,
                    # ``missing='raise'`` is enforced by the guard below.  Once it
                    # passes, the existing hold-first leaf is semantically unreachable
                    # below its first anchor and can stay entirely native/nogil.
                    below_first="first",
                    default_value=None,
                )

                guard_stmt: ast.stmt | None = None
                if guard_required:
                    min_leaves = [
                        (codes, ast.Constant(float(axis_min)))
                        for (codes, _row_key, _whole_table, _sub), axis_min
                        in zip(row_choices, axis_mins)
                    ]
                    min_expr: ast.AST = min_leaves[-1][1]
                    for codes, branch_min in reversed(min_leaves[:-1]):
                        tests = [
                            ast.Compare(
                                copy.deepcopy(selector), [ast.Eq()], [ast.Constant(code)]
                            )
                            for selector, code in zip(row_selectors, codes)
                        ]
                        test = tests[0] if len(tests) == 1 else ast.BoolOp(ast.And(), tests)
                        min_expr = ast.IfExp(test=test, body=branch_min, orelse=min_expr)
                    guard_stmt = ast.If(
                        test=ast.Compare(
                            copy.deepcopy(op.ordered_query), [ast.Lt()], [min_expr]
                        ),
                        body=[ast.Raise(
                            exc=ast.Call(
                                ast.Name("ValueError", ast.Load()),
                                [ast.Constant("max() arg is an empty sequence")], [],
                            ),
                            cause=None,
                        )],
                        orelse=[],
                    )

                if operator_uid is not None:
                    frontend._normalized_operators[operator_uid] = NormalizedOperator(
                        uid=operator_uid,
                        kind="asof_lookup",
                        result_dtype="float64",
                        result_shape="scalar",
                        static_inputs=(
                            ("source_ref", table.ref_name),
                            ("source_cell", table.cell_name),
                            ("direction", op.direction),
                            ("allow_exact", bool(op.allow_exact)),
                            (
                                "missing",
                                "raise_runtime_guard"
                                if guard_required else "raise_proven_unreachable",
                            ),
                            ("branch_count", int(leaf_count)),
                            ("branch_axis_mins", tuple(float(value) for value in axis_mins)),
                            (
                                "proven_query_lower_bound",
                                None if lower_bound is None else float(lower_bound),
                            ),
                            (
                                "proven_query_upper_bound",
                                None if upper_bound is None else float(upper_bound),
                            ),
                        ),
                        dynamic_inputs=(
                            ast.unparse(op.equality_prefix),
                            ast.unparse(op.ordered_query),
                            ast.unparse(op.value_selector),
                        ),
                        provenance=(
                            "semantic_asof_lookup_ir",
                            "shared_step_lookup_physicalizer",
                            "prepared_step_lookup",
                            "finite_prefix_dispatch",
                            "finite_column_dispatch",
                            *(
                                ("native_missing_domain_guard",)
                                if guard_required else ("ordered_domain_proof",)
                            ),
                        ),
                        state_semantic="state_free",
                        python_supported=True,
                        cython_supported=True,
                    )

                out = copy.deepcopy(fn)
                prefix_body: list[ast.stmt] = []
                if (
                    out.body
                    and isinstance(out.body[0], ast.Expr)
                    and isinstance(out.body[0].value, ast.Constant)
                    and isinstance(out.body[0].value.value, str)
                ):
                    prefix_body.append(copy.deepcopy(out.body[0]))
                if guard_stmt is not None:
                    prefix_body.append(guard_stmt)
                prefix_body.append(ast.Return(result))
                out.body = prefix_body
                return ast.fix_missing_locations(out)

            def _rewrite_step_lookup_function(self, fn: ast.FunctionDef) -> ast.FunctionDef:
                """Recognize explicit last-anchor-<=x idioms as StepLookup1D.

                This method now performs source-shape recognition only.  Prefix and
                column planning plus physical StepLookup1D emission are shared with
                :class:`AsofLookupIR` through the helpers above.

                Supported tails are deliberately mathematical rather than pandas-general:

                * explicit numeric default below the first anchor;
                * hold-first below the first anchor via ``max(keys) if keys else min(index)``;
                * no fallback only when the query syntax itself proves a lower bound at
                  or above the first anchor (for example ``max(t, 0)`` with axis min 0).
                """
                body = list(fn.body)

                def step_parts(st: ast.stmt):
                    if not (
                        isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], ast.Name)
                    ):
                        return None
                    row = self._row_proxy(st.value)
                    if row is not None:
                        prefix_plan = self._plan_step_lookup_prefix(row.table, row.row_expr)
                        if prefix_plan is None:
                            return None
                        row_key, dispatch = prefix_plan
                        return st.targets[0].id, row.table, row_key, False, dispatch
                    table = self._table(st.value)
                    if table is not None and not isinstance(table.sample.index, pd.MultiIndex):
                        return st.targets[0].id, table, None, True, None
                    return None

                def key_list(st: ast.stmt, sub_name: str):
                    if not (
                        isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], ast.Name)
                        and isinstance(st.value, ast.ListComp)
                        and len(st.value.generators) == 1
                    ):
                        return None
                    comp = st.value.generators[0]
                    if not (
                        isinstance(comp.target, ast.Name)
                        and isinstance(comp.iter, ast.Attribute) and comp.iter.attr == "index"
                        and isinstance(comp.iter.value, ast.Name) and comp.iter.value.id == sub_name
                        and len(comp.ifs) == 1 and not comp.is_async
                        and isinstance(st.value.elt, ast.Name) and st.value.elt.id == comp.target.id
                    ):
                        return None
                    cond = comp.ifs[0]
                    if not (
                        isinstance(cond, ast.Compare) and len(cond.ops) == 1
                        and isinstance(cond.ops[0], ast.LtE) and len(cond.comparators) == 1
                        and isinstance(cond.left, ast.Name) and cond.left.id == comp.target.id
                    ):
                        return None
                    return st.targets[0].id, copy.deepcopy(cond.comparators[0])

                def lookup_return(
                    st: ast.stmt, sub_name: str, table: TableSource, index_expr_pred,
                ):
                    if not isinstance(st, ast.Return) or st.value is None:
                        return None
                    expr = st.value
                    if (
                        isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
                        and expr.func.id == "float" and len(expr.args) == 1 and not expr.keywords
                    ):
                        expr = expr.args[0]
                    if not (
                        isinstance(expr, ast.Subscript)
                        and isinstance(expr.value, ast.Attribute) and expr.value.attr == "loc"
                        and isinstance(expr.value.value, ast.Name) and expr.value.value.id == sub_name
                    ):
                        return None
                    slice_node = expr.slice
                    if not (isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2):
                        return None
                    row_expr, col_expr = slice_node.elts
                    if not index_expr_pred(row_expr):
                        return None
                    return self._plan_step_lookup_columns(table, col_expr)

                def emit(
                    prefix, origin, table, row_key, whole_table, column_spec, query,
                    below_first, default_value, dispatch=None,
                ):
                    return self._rewrite_step_lookup_result(
                        fn,
                        prefix=prefix,
                        origin=origin,
                        table=table,
                        row_key=row_key,
                        whole_table=whole_table,
                        dispatch=dispatch,
                        column_spec=column_spec,
                        query=query,
                        below_first=below_first,
                        default_value=default_value,
                    )

                for i, st in enumerate(body):
                    parts = step_parts(st)
                    if parts is None or i + 2 >= len(body):
                        continue
                    sub_name, table, row_key, whole_table, dispatch = parts
                    kl = key_list(body[i + 1], sub_name)
                    if kl is None:
                        continue
                    keys_name, query = kl

                    # Explicit constant default then max(keys).
                    if i + 3 < len(body):
                        default_st = body[i + 2]
                        if (
                            isinstance(default_st, ast.If)
                            and isinstance(default_st.test, ast.UnaryOp)
                            and isinstance(default_st.test.op, ast.Not)
                            and isinstance(default_st.test.operand, ast.Name)
                            and default_st.test.operand.id == keys_name
                            and len(default_st.body) == 1
                            and isinstance(default_st.body[0], ast.Return)
                            and not default_st.orelse
                        ):
                            ok_default, default_value = self._static_value(default_st.body[0].value)
                            if (
                                ok_default and not isinstance(default_value, bool)
                                and isinstance(default_value, (int, float, np.integer, np.floating))
                            ):
                                def is_max_keys(x):
                                    return (
                                        isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                                        and x.func.id == "max" and len(x.args) == 1 and not x.keywords
                                        and isinstance(x.args[0], ast.Name) and x.args[0].id == keys_name
                                    )
                                column_spec = lookup_return(
                                    body[i + 3], sub_name, table, is_max_keys
                                )
                                if column_spec is not None and i + 4 == len(body):
                                    return emit(
                                        body[:i], body[i + 3], table, row_key, whole_table,
                                        column_spec, query, "default", float(default_value),
                                        dispatch=dispatch,
                                    )

                    # key = max(keys) if keys else min(sub.index); return sub.loc[key, col]
                    key_st = body[i + 2]
                    if (
                        isinstance(key_st, ast.Assign) and len(key_st.targets) == 1
                        and isinstance(key_st.targets[0], ast.Name)
                        and isinstance(key_st.value, ast.IfExp)
                        and isinstance(key_st.value.test, ast.Name)
                        and key_st.value.test.id == keys_name
                    ):
                        selected_name = key_st.targets[0].id
                        yes, no = key_st.value.body, key_st.value.orelse
                        yes_ok = (
                            isinstance(yes, ast.Call) and isinstance(yes.func, ast.Name)
                            and yes.func.id == "max" and len(yes.args) == 1 and not yes.keywords
                            and isinstance(yes.args[0], ast.Name) and yes.args[0].id == keys_name
                        )
                        no_ok = (
                            isinstance(no, ast.Call) and isinstance(no.func, ast.Name)
                            and no.func.id == "min" and len(no.args) == 1 and not no.keywords
                            and isinstance(no.args[0], ast.Attribute) and no.args[0].attr == "index"
                            and isinstance(no.args[0].value, ast.Name) and no.args[0].value.id == sub_name
                        )
                        if yes_ok and no_ok and i + 3 < len(body):
                            column_spec = lookup_return(
                                body[i + 3], sub_name, table,
                                lambda x: isinstance(x, ast.Name) and x.id == selected_name,
                            )
                            if column_spec is not None and i + 4 == len(body):
                                return emit(
                                    body[:i], body[i + 3], table, row_key, whole_table,
                                    column_spec, query, "first", None, dispatch=dispatch,
                                )

                    # Bare max(keys) is safe only if the query itself proves a lower bound.
                    def is_max_keys(x):
                        return (
                            isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                            and x.func.id == "max" and len(x.args) == 1 and not x.keywords
                            and isinstance(x.args[0], ast.Name) and x.args[0].id == keys_name
                        )
                    column_spec = lookup_return(body[i + 2], sub_name, table, is_max_keys)
                    if column_spec is not None and i + 3 == len(body):
                        try:
                            row_choices, _row_selectors = self._step_lookup_row_choices(
                                table, row_key, whole_table, dispatch
                            )
                            axis_mins = [float(min(sub.index)) for *_prefix, sub in row_choices]
                            axis_min = max(axis_mins)
                        except Exception:
                            continue
                        lower_bound, upper_bound = self._numeric_bounds(query)
                        if lower_bound is not None and lower_bound >= axis_min:
                            tr = frontend.variant_trace_by_uid.get(fn.name)
                            source_name = tr.schema.name if tr is not None else fn.name
                            column_choices, _column_enum = column_spec
                            for (
                                _codes, branch_row_key, _branch_whole_table, sub
                            ) in row_choices:
                                branch_axis_min = float(min(sub.index))
                                for column, _column_code in column_choices:
                                    fact_key = (
                                        fn.name,
                                        ast.dump(query, include_attributes=False),
                                        branch_axis_min,
                                        table.ref_name,
                                        table.cell_name,
                                        repr(branch_row_key),
                                        repr(column),
                                    )
                                    frontend._used_step_lookup_domain_facts[fact_key] = StepLookupDomainFact(
                                        source_uid=fn.name,
                                        source_name=source_name,
                                        query_expr=ast.unparse(query),
                                        axis_min=branch_axis_min,
                                        proven_lower_bound=float(lower_bound),
                                        proven_upper_bound=(
                                            None if upper_bound is None else float(upper_bound)
                                        ),
                                    )
                            return emit(
                                body[:i], body[i + 2], table, row_key, whole_table,
                                column_spec, query, "first", None, dispatch=dispatch,
                            )
                return fn

            def _rewrite_interval_lookup_function(self, fn: ast.FunctionDef) -> ast.FunctionDef:
                """Normalize one finite numeric ``iterrows`` interval-table idiom.

                This is provider normalization, not general Python-loop lowering.
                The source DataFrame is frozen during preparation into typed lower,
                upper and value arrays; Cython sees only a scalar interval intrinsic.

                Supported source semantics are deliberately explicit:

                * ``lower < q <= upper`` with final-row fallback;
                * ``lower <= q <= upper`` with a numeric default for unmatched
                  queries at/below the final upper bound and final-row hold above it;
                * a value column selected by one proven finite categorical selector.

                The latter covers finite-category-selected value columns generically.  The
                category mapping and table columns are preparation-time metadata;
                no pandas object or Python string crosses the native runtime ABI.
                """
                body = list(fn.body)

                def unwrap_cast(expr: ast.AST) -> tuple[ast.AST, str | None]:
                    if (
                        isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
                        and expr.func.id in {"float", "int"}
                        and len(expr.args) == 1 and not expr.keywords
                    ):
                        return expr.args[0], expr.func.id
                    return expr, None

                def row_col(expr: ast.AST, row_name: str):
                    expr, cast = unwrap_cast(expr)
                    if not (
                        isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name)
                        and expr.value.id == row_name
                    ):
                        return None
                    ok, value = self._static_value(expr.slice)
                    return (value, cast) if ok else None

                def row_value_slice(expr: ast.AST, row_name: str):
                    expr, _cast = unwrap_cast(expr)
                    if not (
                        isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name)
                        and expr.value.id == row_name
                    ):
                        return None
                    return copy.deepcopy(expr.slice)

                @staticmethod
                def is_minus_one(node: ast.AST) -> bool:
                    return (
                        isinstance(node, ast.Constant) and node.value == -1
                    ) or (
                        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                        and isinstance(node.operand, ast.Constant) and node.operand.value == 1
                    )

                def resolve_table(expr: ast.AST, stop: int):
                    direct = self._table(expr)
                    if direct is not None:
                        return direct, None
                    if not isinstance(expr, ast.Name):
                        return None, None
                    alias = expr.id
                    matches: list[tuple[int, TableSource]] = []
                    for j, prior in enumerate(body[:stop]):
                        if not (
                            isinstance(prior, ast.Assign) and len(prior.targets) == 1
                            and isinstance(prior.targets[0], ast.Name)
                            and prior.targets[0].id == alias
                        ):
                            continue
                        table = self._external_table(prior.value)
                        if table is not None:
                            matches.append((j, table))
                    if len(matches) != 1:
                        return None, None
                    return matches[0][1], matches[0][0]

                def numeric_column_ok(table: TableSource, column: Any, cast: str | None) -> bool:
                    if column not in table.sample.columns:
                        return False
                    raw = np.asarray(table.sample[column])
                    if raw.dtype.kind not in "iufc":
                        return False
                    if cast == "int":
                        try:
                            vals = np.asarray(raw, dtype=np.float64)
                        except Exception:
                            return False
                        if not np.all(np.isfinite(vals)) or not np.all(vals == np.trunc(vals)):
                            return False
                    return True

                def parse_category_condition(test: ast.AST):
                    if not (
                        isinstance(test, ast.Compare) and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
                    ):
                        return None
                    pairs = ((test.left, test.comparators[0]), (test.comparators[0], test.left))
                    for selector, label_node in pairs:
                        if not isinstance(label_node, ast.Constant):
                            continue
                        enum = self._finite_enum(selector)
                        if enum is None:
                            continue
                        value = label_node.value
                        if isinstance(value, str):
                            if value not in enum.labels:
                                continue
                            label = value
                        elif isinstance(value, int) and not isinstance(value, bool):
                            # Category comparisons may already have passed through the
                            # ordinary category-ABI rewrite before provider lowering.
                            # Integer code k is exactly labels[k] by CategoryEncoding.
                            if value < 0 or value >= len(enum.labels):
                                continue
                            label = enum.labels[value]
                        else:
                            continue
                        return enum, label, copy.deepcopy(selector)
                    return None

                def terminal_guard(stmts: list[ast.stmt]) -> bool:
                    if len(stmts) != 1:
                        return False
                    st = stmts[0]
                    if isinstance(st, ast.Raise):
                        return True
                    return (
                        isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                        and isinstance(st.value.func, ast.Name)
                        and st.value.func.id == "__guard_fail__"
                    )

                def finite_column_dispatch(column_name: str, stop: int, table: TableSource):
                    """Resolve ``if enum == label: column = '...'`` to numeric dispatch."""
                    candidates = [
                        (j, st) for j, st in enumerate(body[:stop]) if isinstance(st, ast.If)
                    ]
                    for stmt_index, root_if in candidates:
                        mapping: dict[str, Any] = {}
                        enum_value = None
                        selector_dump = None
                        current: ast.If | None = root_if
                        terminal: list[ast.stmt] = []
                        valid = True
                        while current is not None:
                            parsed = parse_category_condition(current.test)
                            if parsed is None or len(current.body) != 1:
                                valid = False
                                break
                            enum, label, selector = parsed
                            assign = current.body[0]
                            if not (
                                isinstance(assign, ast.Assign) and len(assign.targets) == 1
                                and isinstance(assign.targets[0], ast.Name)
                                and assign.targets[0].id == column_name
                                and isinstance(assign.value, ast.Constant)
                                and isinstance(assign.value.value, str)
                            ):
                                valid = False
                                break
                            token = ast.dump(selector, include_attributes=False)
                            if enum_value is None:
                                enum_value = enum
                                selector_dump = token
                            elif tuple(enum.labels) != tuple(enum_value.labels) or token != selector_dump:
                                valid = False
                                break
                            if label in mapping:
                                valid = False
                                break
                            mapping[label] = assign.value.value
                            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                                current = current.orelse[0]
                            else:
                                terminal = list(current.orelse)
                                current = None
                        if not valid or enum_value is None or not terminal_guard(terminal):
                            continue
                        if set(mapping) != set(enum_value.labels):
                            continue
                        choices = tuple(
                            (mapping[label], code) for code, label in enumerate(enum_value.labels)
                        )
                        if any(
                            column not in table.sample.columns
                            or np.asarray(table.sample[column]).dtype.kind not in "iufc"
                            for column, _code in choices
                        ):
                            continue
                        return enum_value, choices, stmt_index
                    return None

                def value_column_spec(slice_node: ast.AST, stop: int, table: TableSource):
                    ok, static_column = self._static_value(slice_node)
                    if ok:
                        if static_column not in table.sample.columns:
                            return None
                        if np.asarray(table.sample[static_column]).dtype.kind not in "iufc":
                            return None
                        return ((static_column, None),), None, None
                    enum = self._finite_enum(slice_node)
                    if enum is not None and all(label in table.sample.columns for label in enum.labels):
                        if all(np.asarray(table.sample[label]).dtype.kind in "iufc" for label in enum.labels):
                            return tuple((label, code) for code, label in enumerate(enum.labels)), enum, None
                    if isinstance(slice_node, ast.Name):
                        dispatch = finite_column_dispatch(slice_node.id, stop, table)
                        if dispatch is not None:
                            enum, choices, stmt_index = dispatch
                            return choices, enum, stmt_index
                    return None

                def same_slice(left: ast.AST, right: ast.AST) -> bool:
                    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)

                def direct_last_value(expr: ast.AST, stop: int):
                    expr, _cast = unwrap_cast(expr)
                    if not isinstance(expr, ast.Subscript):
                        return None
                    value_slice = copy.deepcopy(expr.slice)
                    iloc = expr.value
                    if not (
                        isinstance(iloc, ast.Subscript) and isinstance(iloc.value, ast.Attribute)
                        and iloc.value.attr == "iloc" and is_minus_one(iloc.slice)
                    ):
                        return None
                    table, alias_index = resolve_table(iloc.value.value, stop)
                    if table is None:
                        return None
                    return table, value_slice, alias_index

                def last_alias_assignment(st: ast.stmt, stop: int):
                    if not (
                        isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], ast.Name)
                        and isinstance(st.value, ast.Subscript)
                        and isinstance(st.value.value, ast.Attribute)
                        and st.value.value.attr == "iloc"
                        and is_minus_one(st.value.slice)
                    ):
                        return None
                    table, alias_index = resolve_table(st.value.value.value, stop)
                    if table is None:
                        return None
                    return st.targets[0].id, table, alias_index

                def last_alias_col(expr: ast.AST, alias: str):
                    expr, cast = unwrap_cast(expr)
                    if not (
                        isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name)
                        and expr.value.id == alias
                    ):
                        return None
                    return copy.deepcopy(expr.slice), cast

                for i, st in enumerate(body):
                    if not isinstance(st, ast.For):
                        continue
                    if not (
                        isinstance(st.target, (ast.Tuple, ast.List)) and len(st.target.elts) == 2
                        and isinstance(st.target.elts[1], ast.Name)
                        and isinstance(st.iter, ast.Call) and not st.iter.args and not st.iter.keywords
                        and isinstance(st.iter.func, ast.Attribute) and st.iter.func.attr == "iterrows"
                    ):
                        continue
                    row_name = st.target.elts[1].id
                    table, table_alias_index = resolve_table(st.iter.func.value, i)
                    if table is None or len(st.body) != 1 or st.orelse:
                        continue
                    branch = st.body[0]
                    if not (
                        isinstance(branch, ast.If) and len(branch.body) == 1
                        and isinstance(branch.body[0], ast.Return) and branch.body[0].value is not None
                        and not branch.orelse
                    ):
                        continue
                    cond = branch.test
                    if not (
                        isinstance(cond, ast.Compare) and len(cond.ops) == 2
                        and len(cond.comparators) == 2
                        and isinstance(cond.ops[1], ast.LtE)
                    ):
                        continue
                    if isinstance(cond.ops[0], ast.Lt):
                        closure = "(lo,hi]"
                        closure_code = 0
                    elif isinstance(cond.ops[0], ast.LtE):
                        closure = "[lo,hi]"
                        closure_code = 1
                    else:
                        continue
                    lower_info = row_col(cond.left, row_name)
                    upper_info = row_col(cond.comparators[1], row_name)
                    value_slice = row_value_slice(branch.body[0].value, row_name)
                    if lower_info is None or upper_info is None or value_slice is None:
                        continue
                    lower_col, lower_cast = lower_info
                    upper_col, upper_cast = upper_info
                    if not (
                        numeric_column_ok(table, lower_col, lower_cast)
                        and numeric_column_ok(table, upper_col, upper_cast)
                    ):
                        continue
                    query = copy.deepcopy(cond.comparators[0])
                    column_spec = value_column_spec(value_slice, i, table)
                    if column_spec is None:
                        continue
                    column_choices, column_enum, column_dispatch_index = column_spec

                    fallback = None
                    default_value = 0.0
                    extra_alias_index = None

                    # Historical interval idiom: immediate final-row fallback.
                    if i + 1 < len(body) and isinstance(body[i + 1], ast.Return) and body[i + 1].value is not None:
                        final = direct_last_value(body[i + 1].value, i + 1)
                        if final is not None:
                            fallback_table, fallback_slice, extra_alias_index = final
                            if (
                                fallback_table.ref_name == table.ref_name
                                and fallback_table.cell_name == table.cell_name
                                and same_slice(fallback_slice, value_slice)
                                and i + 2 == len(body)
                            ):
                                fallback = "last"

                    # Generic contract: bind final row, return it only
                    # above the last upper bound, otherwise return a numeric default.
                    if fallback is None and i + 2 < len(body):
                        last = last_alias_assignment(body[i + 1], i + 1)
                        ret = body[i + 2]
                        if last is not None and isinstance(ret, ast.Return) and isinstance(ret.value, ast.IfExp) and i + 3 == len(body):
                            alias, fallback_table, extra_alias_index = last
                            test = ret.value.test
                            body_col = last_alias_col(ret.value.body, alias)
                            orelse = ret.value.orelse
                            default_ok = isinstance(orelse, ast.Constant) and isinstance(
                                orelse.value, (int, float, np.integer, np.floating)
                            ) and not isinstance(orelse.value, (bool, np.bool_))
                            if (
                                isinstance(test, ast.Compare) and len(test.ops) == 1
                                and isinstance(test.ops[0], ast.Gt) and len(test.comparators) == 1
                                and self._same_expr(test.left, query)
                                and body_col is not None and same_slice(body_col[0], value_slice)
                                and default_ok
                                and fallback_table.ref_name == table.ref_name
                                and fallback_table.cell_name == table.cell_name
                            ):
                                upper_fallback = last_alias_col(test.comparators[0], alias)
                                if (
                                    upper_fallback is not None
                                    and upper_fallback[0] is not None
                                    and self._static_value(upper_fallback[0]) == (True, upper_col)
                                    and numeric_column_ok(table, upper_col, upper_fallback[1])
                                ):
                                    fallback = "default_or_last_above"
                                    default_value = float(orelse.value)
                    if fallback is None:
                        continue

                    fallback_code = 0 if fallback == "last" else 1
                    leaves: list[tuple[int | None, ast.AST]] = []
                    for value_col, code in column_choices:
                        lo_key, hi_key, value_key = registry.external_dataframe_interval_lookup(
                            table.ref_name, table.cell_name, table.cell,
                            lower_column=lower_col, upper_column=upper_col,
                            value_column=value_col, closure=closure, fallback=fallback,
                            default_value=(None if fallback == "last" else default_value),
                        )
                        call = ast.Call(
                            func=ast.Name("__interval_lookup_1d__", ast.Load()),
                            args=[
                                ast.Constant(lo_key), ast.Constant(hi_key), ast.Constant(value_key),
                                copy.deepcopy(query), ast.Constant(closure_code),
                                ast.Constant(fallback_code), ast.Constant(float(default_value)),
                            ],
                            keywords=[],
                        )
                        leaves.append((code, call))

                    if column_enum is None:
                        result = leaves[0][1]
                    else:
                        selector = copy.deepcopy(column_enum.code_expr)
                        result = leaves[-1][1]
                        for (code, branch_call) in reversed(leaves[:-1]):
                            assert code is not None
                            result = ast.IfExp(
                                test=ast.Compare(
                                    copy.deepcopy(selector), [ast.Eq()], [ast.Constant(int(code))]
                                ),
                                body=branch_call,
                                orelse=result,
                            )

                    removed_prefix = {
                        idx for idx in (table_alias_index, column_dispatch_index, extra_alias_index)
                        if idx is not None and idx < i
                    }
                    prefix = [
                        copy.deepcopy(prior) for j, prior in enumerate(body[:i])
                        if j not in removed_prefix
                    ]
                    out = copy.deepcopy(fn)
                    out.body = prefix + [ast.copy_location(ast.Return(result), st)]
                    return ast.fix_missing_locations(out)
                return fn


            def _external_table(self, node: ast.AST) -> TableSource | None:
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and not node.args and not node.keywords
                ):
                    return None
                rn = node.func.value.id
                cn = node.func.attr
                cache_key = (rn, cn)
                if cache_key in self.external_cache:
                    return self.external_cache[cache_key]
                obj = frontend.refs.get(rn)
                if obj is None or not hasattr(obj, "cells") or cn not in obj.cells:
                    self.external_cache[cache_key] = None
                    return None
                cell = obj.cells[cn]
                try:
                    value = cell()
                except Exception:
                    self.external_cache[cache_key] = None
                    return None
                if not isinstance(value, pd.DataFrame):
                    self.external_cache[cache_key] = None
                    return None
                src = TableSource(rn, cn, cell, value)
                self.external_cache[cache_key] = src
                return src

            def _table(self, node: ast.AST) -> TableSource | None:
                if isinstance(node, ast.Name) and node.id in self.table_aliases:
                    return self.table_aliases[node.id]
                return self._external_table(node)

            def _table_column(
                self, node: ast.AST
            ) -> tuple[TableSource, ast.AST] | None:
                """Preparation-only exact DataFrame column view."""
                if not isinstance(node, ast.Subscript):
                    return None
                table = self._table(node.value)
                if table is None:
                    return None
                ok, column = self._static_value(node.slice)
                if not ok or column not in table.sample.columns:
                    return None
                return table, copy.deepcopy(node.slice)

            def _series_from_expr(self, node: ast.AST) -> SeriesSource | None:
                """Recognize one residual numeric-axis DataFrame view exactly.

                Accepted prefix selectors are either compile-time static or proven
                finite categorical values.  This is a descriptor proof only: the
                pandas Series never becomes a runtime value.
                """
                if not (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Subscript)
                    and isinstance(node.value.value, ast.Attribute)
                    and node.value.value.attr == "loc"
                ):
                    return None
                table = self._table(node.value.value.value)
                if table is None or not isinstance(table.sample.index, pd.MultiIndex):
                    return None
                nlevels = int(table.sample.index.nlevels)
                prefix_node = self._expand_alias(copy.deepcopy(node.value.slice))
                if isinstance(prefix_node, ast.Tuple):
                    prefix_parts = list(prefix_node.elts)
                else:
                    prefix_parts = [prefix_node]
                if len(prefix_parts) != nlevels - 1 or nlevels < 2:
                    return None
                fixed: list[tuple[int, Any]] = []
                for level, expr in enumerate(prefix_parts):
                    ok, value = self._static_value(expr)
                    if ok:
                        fixed.append((level, value))
                        continue
                    if self._finite_enum(expr) is None:
                        return None
                ok, column = self._static_value(node.slice)
                if not ok or column not in table.sample.columns:
                    return None
                if np.asarray(table.sample[column]).dtype.kind not in "iubfc":
                    return None
                residual_level = nlevels - 1
                # Every row reachable through the admitted prefix domain must expose
                # a finite numeric residual label.  Exact per-prefix completeness is
                # checked later by the reused row-metadata/table materializers.
                for raw in table.sample.index.get_level_values(residual_level):
                    if (
                        not isinstance(raw, (int, float, np.integer, np.floating))
                        or isinstance(raw, (bool, np.bool_))
                        or not np.isfinite(float(raw))
                    ):
                        return None
                source = SeriesSource(
                    table=table,
                    row_expr=copy.deepcopy(prefix_node),
                    fixed_levels=tuple(fixed),
                    dynamic_level=residual_level,
                    column=column,
                )
                if not source.has_dynamic_prefix:
                    # Retain the mature compact 1-D path for fully static prefixes.
                    registry.external_dataframe_axis(
                        table.ref_name, table.cell_name, table.cell,
                        fixed_levels=dict(source.fixed_levels),
                        dynamic_level=source.dynamic_level,
                        column=source.column,
                    )
                return source

            def _series(self, node: ast.AST) -> SeriesSource | None:
                if isinstance(node, ast.Name) and node.id in self.series_aliases:
                    return self.series_aliases[node.id]
                direct = self._series_from_expr(node)
                if direct is not None:
                    return direct
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and not node.args and not node.keywords
                ):
                    return None
                call_name = node.func.id
                if call_name in self.series_helper_cache:
                    return self.series_helper_cache[call_name]
                source_name: str | None = None
                if call_name in frontend.variant_trace_by_uid:
                    tr = frontend.variant_trace_by_uid[call_name]
                    if tr.key.time_pos is None and not tr.key.aux_values and tr.dtype == "object":
                        source_name = tr.schema.name
                elif call_name in frontend.base_funcs:
                    source_name = call_name
                base = frontend.base_funcs.get(source_name) if source_name is not None else None
                if base is None or base.args.args or base.args.posonlyargs or base.args.kwonlyargs:
                    self.series_helper_cache[call_name] = None
                    return None
                body = _strip_docstring(copy.deepcopy(base)).body
                if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
                    self.series_helper_cache[call_name] = None
                    return None
                out = self._series_from_expr(body[0].value)
                self.series_helper_cache[call_name] = out
                return out

            def _point_field_finite_enum(
                self, source_name: str, node: ast.Call
            ) -> FiniteEnumValue | None:
                """Return the numeric enum ABI for a proven point-field selector.

                Guard-bearing selectors reuse ``ValidatedEnumCell`` evidence
                discovered earlier by the frontend; this table-normalization pass
                does not rediscover or reinterpret their validation control flow.
                Unguarded direct ``return model_point()["field"]`` selectors are
                admitted from the exact frozen RunDomain as before.
                """
                if frontend.model_point_row is None:
                    return None
                point_row_names = {frontend.model_point_row.cell_name}
                point_source_name = getattr(frontend, "formula_uid_to_source_name", {}).get(
                    frontend.model_point_row.cell_name
                )
                if point_source_name is not None:
                    point_row_names.add(point_source_name)

                field: str | None = None
                labels: tuple[str, ...] | None = None
                validated = frontend.validated_enum_cells.get(source_name)
                if validated is not None:
                    if node.args or node.keywords:
                        return None
                    field = validated.field
                    labels = tuple(validated.allowed_values)
                else:
                    original = frontend.base_funcs.get(source_name)
                    if original is None and isinstance(node.func, ast.Name):
                        original = frontend.base_funcs.get(node.func.id)
                    if original is None:
                        return None
                    direct = _strip_docstring(copy.deepcopy(original))
                    positional = [*direct.args.posonlyargs, *direct.args.args]
                    if len(positional) != len(node.args) or node.keywords:
                        return None

                    # Parameterized finite point-field dispatch, e.g.
                    # ``if life == 1: return point()["primary_sex"]``.
                    # Keep the selector numeric and bind each statically named
                    # field through the existing point-category ABI. This is a
                    # finite dispatch proof, not a general dynamic row lookup.
                    dispatch_rows: list[tuple[ast.AST, str]] = []
                    dispatch_ok = bool(positional)
                    for st in direct.body:
                        if isinstance(st, ast.Raise):
                            continue
                        if (
                            isinstance(st, ast.Expr)
                            and isinstance(st.value, ast.Call)
                            and isinstance(st.value.func, ast.Name)
                            and st.value.func.id == "__guard_fail__"
                        ):
                            continue
                        if not (
                            isinstance(st, ast.If) and not st.orelse
                            and len(st.body) == 1 and isinstance(st.body[0], ast.Return)
                            and isinstance(st.body[0].value, ast.Subscript)
                            and isinstance(st.body[0].value.value, ast.Call)
                            and isinstance(st.body[0].value.value.func, ast.Name)
                            and st.body[0].value.value.func.id in point_row_names
                            and not st.body[0].value.value.args
                            and not st.body[0].value.value.keywords
                            and isinstance(st.body[0].value.slice, ast.Constant)
                            and isinstance(st.body[0].value.slice.value, str)
                        ):
                            dispatch_ok = False
                            break
                        dispatch_rows.append((copy.deepcopy(st.test), str(st.body[0].value.slice.value)))
                    if dispatch_ok and dispatch_rows:
                        frame = frontend.model_point_row.provider()
                        fields = tuple(dict.fromkeys(field_name for _test, field_name in dispatch_rows))
                        if isinstance(frame, pd.DataFrame) and all(name in frame.columns for name in fields):
                            all_values: list[Any] = []
                            field_values: dict[str, list[Any]] = {}
                            valid = True
                            for field_name in fields:
                                values = frame[field_name].tolist()
                                field_values[field_name] = values
                                if not values or not all(
                                    isinstance(value, str) or bool(pd.isna(value)) for value in values
                                ):
                                    valid = False
                                    break
                                all_values.extend(value for value in values if isinstance(value, str))
                            union_labels = tuple(dict.fromkeys(all_values))
                            if valid and union_labels and len(union_labels) <= 16:
                                argument_map = {
                                    param.arg: copy.deepcopy(arg)
                                    for param, arg in zip(positional, node.args)
                                }
                                result: ast.AST = ast.Constant(-1)
                                for test, field_name in reversed(dispatch_rows):
                                    values = field_values[field_name]
                                    has_other = any(not isinstance(value, str) for value in values)
                                    key = registry.point_field(
                                        frontend.model_point_row, field_name,
                                        enum_labels=union_labels, enum_allow_missing=has_other,
                                    )
                                    field_code: ast.AST = ast.Call(
                                        ast.Name("point_input", ast.Load()),
                                        [ast.Constant(key)], [],
                                    )
                                    substituted = _Substitute(argument_map).visit(copy.deepcopy(test))
                                    substituted = ast.fix_missing_locations(substituted)
                                    result = ast.IfExp(
                                        test=self.visit(substituted),
                                        body=field_code,
                                        orelse=result,
                                    )
                                return FiniteEnumValue(
                                    labels=union_labels,
                                    code_expr=ast.fix_missing_locations(result),
                                    source_name=source_name,
                                )

                    mapping: dict[str, ast.AST] = {}
                    for param, arg in zip(positional, node.args):
                        ok, value = self._static_value(arg)
                        if not ok:
                            return None
                        mapping[param.arg] = ast.Constant(value)
                    if mapping:
                        direct = _Substitute(mapping).visit(direct)
                        direct = _ConservativeFolder().visit(direct)
                        if not isinstance(direct, ast.FunctionDef):
                            return None
                        direct = _prune_source_unreachable_tails(direct)
                    if not (
                        not direct.args.kwonlyargs and direct.args.vararg is None
                        and direct.args.kwarg is None and len(direct.body) == 1
                        and isinstance(direct.body[0], ast.Return)
                        and isinstance(direct.body[0].value, ast.Subscript)
                        and isinstance(direct.body[0].value.value, ast.Call)
                        and isinstance(direct.body[0].value.value.func, ast.Name)
                        and direct.body[0].value.value.func.id in point_row_names
                        and not direct.body[0].value.value.args
                        and not direct.body[0].value.value.keywords
                        and isinstance(direct.body[0].value.slice, ast.Constant)
                        and isinstance(direct.body[0].value.slice.value, str)
                    ):
                        return None
                    field = str(direct.body[0].value.slice.value)

                try:
                    frame = frontend.model_point_row.provider()
                    values = frame[field].tolist()
                except Exception:
                    return None
                if not values:
                    return None

                if labels is None:
                    if not all(
                        isinstance(value, str) or bool(pd.isna(value)) for value in values
                    ):
                        return None
                    labels = tuple(dict.fromkeys(
                        value for value in values if isinstance(value, str)
                    ))
                    if not labels or len(labels) > 16:
                        return None
                else:
                    # A guard-bearing selector promises this exact source domain;
                    # every value in the frozen RunDomain must satisfy it now.
                    if not all(
                        isinstance(value, str) and value in labels for value in values
                    ):
                        return None

                key = registry.point_field(
                    frontend.model_point_row, field, enum_labels=labels
                )
                point_code = ast.Call(
                    ast.Name("point_input", ast.Load()), [ast.Constant(key)], []
                )
                has_other = any(not isinstance(value, str) for value in values)
                code_expr = (
                    ast.Call(
                        ast.Name("__exact_axis_code__", ast.Load()),
                        [point_code, ast.Constant(0), ast.Constant(1), ast.Constant(len(labels))], [],
                    )
                    if has_other else point_code
                )
                return FiniteEnumValue(
                    labels=labels, code_expr=code_expr, source_name=source_name,
                )

            def _encode_nested_enum_comparisons(self, expr: ast.AST) -> ast.AST:
                """Replace proven categorical string comparisons by integer-code tests.

                This is used only while proving another finite categorical helper.
                Unsupported object/string dependencies remain in the tree and are
                rejected by the caller, so composition is fail-closed.
                """
                outer = self

                class Encode(ast.NodeTransformer):
                    def visit_Compare(self, node: ast.Compare):
                        if (
                            len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
                            and len(node.comparators) == 1
                        ):
                            pairs = (
                                (node.left, node.comparators[0], False),
                                (node.comparators[0], node.left, True),
                            )
                            for enum_node, label_node, reversed_order in pairs:
                                enum_value = outer._finite_enum(enum_node)
                                if enum_value is None or not (
                                    isinstance(label_node, ast.Constant)
                                    and isinstance(label_node.value, str)
                                ):
                                    continue
                                code = outer._enum_label_code(enum_value, label_node.value)
                                if code is None:
                                    value = isinstance(node.ops[0], ast.NotEq)
                                    return ast.copy_location(ast.Constant(value), node)
                                left = copy.deepcopy(enum_value.code_expr)
                                right = ast.Constant(code)
                                if reversed_order:
                                    left, right = right, left
                                return ast.copy_location(
                                    ast.Compare(left, [copy.deepcopy(node.ops[0])], [right]), node
                                )
                        return self.generic_visit(node)

                return ast.fix_missing_locations(Encode().visit(copy.deepcopy(expr)))

            @staticmethod
            def _finite_string_value_labels(expr: ast.AST) -> tuple[str, ...] | None:
                """Return the finite string value domain of a pure conditional expression."""
                if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                    return (str(expr.value),)
                if isinstance(expr, ast.IfExp):
                    left = Normalize._finite_string_value_labels(expr.body)
                    right = Normalize._finite_string_value_labels(expr.orelse)
                    if left is None or right is None:
                        return None
                    return tuple(dict.fromkeys((*left, *right)))
                return None

            @staticmethod
            def _encode_finite_string_values(
                expr: ast.AST, codes: Mapping[str, int]
            ) -> ast.AST | None:
                """Encode only the value leaves of a proven finite string expression."""
                if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                    if expr.value not in codes:
                        return None
                    return ast.copy_location(ast.Constant(int(codes[expr.value])), expr)
                if isinstance(expr, ast.IfExp):
                    body = Normalize._encode_finite_string_values(expr.body, codes)
                    orelse = Normalize._encode_finite_string_values(expr.orelse, codes)
                    if body is None or orelse is None:
                        return None
                    return ast.copy_location(
                        ast.IfExp(test=copy.deepcopy(expr.test), body=body, orelse=orelse), expr
                    )
                return None

            def _reachable_category_labels(
                self, node: ast.AST, labels: Sequence[Any]
            ) -> tuple[Any, ...] | None:
                """Return a consumer-scoped complete-RunDomain reachable subset.

                The category ABI still owns the global code domain. This proof only
                permits a sparse table to omit labels that source control makes
                unreachable from the selected output; it never renumbers labels.
                """
                expr = self._expand_alias(copy.deepcopy(node))
                if not (
                    operator_uid is not None
                    and isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
                    and not expr.args and not expr.keywords
                ):
                    return None
                mapping = getattr(frontend, "static_category_reachable_result_domains", {})
                reachable = mapping.get((operator_uid, expr.func.id))
                if reachable is None:
                    return None
                reachable = tuple(x for x in labels if x in set(reachable))
                return reachable or None

            def _finite_enum(self, node: ast.AST) -> FiniteEnumValue | None:
                """Prove one object Cell call is a pure finite string selector.

                The source Cell is specialized through the ordinary frontend first,
                so full-domain static branches have already disappeared and nested
                numeric helpers are already canonical UID calls.  We then require
                every reachable return to be a literal string and encode those
                labels as small integer codes.  No object/string value survives into
                canonical execution.
                """
                if isinstance(node, ast.Name):
                    parameter_categories = getattr(frontend, "static_category_parameter_domains", {})
                    labels = tuple(parameter_categories.get((operator_uid, node.id), ()))
                    if labels:
                        if len(labels) > 16 or any(not isinstance(label, str) for label in labels):
                            return None
                        return FiniteEnumValue(
                            labels=labels, code_expr=copy.deepcopy(node),
                            source_name=f"{operator_uid}:{node.id}" if operator_uid is not None else node.id,
                        )
                    if node.id in self.finite_enum_aliases:
                        return self.finite_enum_aliases[node.id]
                expanded_node = self._expand_alias(copy.deepcopy(node))
                if isinstance(expanded_node, ast.IfExp):
                    # A local/source expression such as
                    # ``"a" if numeric_predicate else "b"`` is already a
                    # complete finite value-domain proof.  Encode only the
                    # string leaves here; the ordinary visitor remains
                    # responsible for lowering the numeric predicate itself.
                    encoded_condition = self._encode_nested_enum_comparisons(expanded_node)
                    labels = self._finite_string_value_labels(encoded_condition)
                    if labels and len(labels) <= 16:
                        codes = {label: i for i, label in enumerate(labels)}
                        encoded = self._encode_finite_string_values(encoded_condition, codes)
                        if encoded is not None:
                            encoded = ast.fix_missing_locations(encoded)
                            if not any(
                                isinstance(child, ast.Call)
                                and isinstance(child.func, ast.Name)
                                and child.func.id in frontend.variant_trace_by_uid
                                and frontend.variant_trace_by_uid[child.func.id].dtype == "object"
                                for child in ast.walk(encoded)
                            ):
                                return FiniteEnumValue(
                                    labels=labels,
                                    code_expr=encoded,
                                    source_name="<finite_expression>",
                                )
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and not node.keywords
                ):
                    return None
                token = ast.dump(node, include_attributes=False)
                if token in self.finite_enum_cache:
                    return self.finite_enum_cache[token]

                call_name = node.func.id
                static_categories = getattr(frontend, "static_category_result_domains", {})
                if call_name in static_categories:
                    labels = tuple(static_categories[call_name])
                    if not labels or len(labels) > 16 or any(not isinstance(label, str) for label in labels):
                        self.finite_enum_cache[token] = None
                        return None
                    out = FiniteEnumValue(
                        labels=labels, code_expr=copy.deepcopy(node), source_name=call_name
                    )
                    self.finite_enum_cache[token] = out
                    return out
                tr = frontend.variant_trace_by_uid.get(call_name)
                source_name = tr.schema.name if tr is not None else getattr(
                    frontend, "formula_uid_to_source_name", {}
                ).get(call_name)
                if source_name is None and call_name in frontend.base_funcs:
                    source_name = call_name
                if source_name is None:
                    self.finite_enum_cache[token] = None
                    return None
                if tr is not None and (
                    tr.dtype != "object" or frontend._source_reaches_recursive_cycle(source_name)
                ):
                    self.finite_enum_cache[token] = None
                    return None

                point_enum = self._point_field_finite_enum(source_name, node)
                if point_enum is not None:
                    self.finite_enum_cache[token] = point_enum
                    return point_enum

                if tr is None:
                    self.finite_enum_cache[token] = None
                    return None
                prepared = frontend._specialize_function(node.func.id)
                if prepared is None or prepared.args.vararg is not None or prepared.args.kwarg is not None:
                    self.finite_enum_cache[token] = None
                    return None
                params = [arg.arg for arg in prepared.args.args]
                if len(params) != len(node.args):
                    self.finite_enum_cache[token] = None
                    return None
                try:
                    expr = frontend._runtime_scalar_helper_expr_from_block(list(prepared.body), {})
                except FrontendError:
                    self.finite_enum_cache[token] = None
                    return None
                if params:
                    expr = _Substitute({
                        param: copy.deepcopy(arg) for param, arg in zip(params, node.args)
                    }).visit(expr)
                    expr = ast.fix_missing_locations(expr)
                expr = self._encode_nested_enum_comparisons(expr)
                labels = self._finite_string_value_labels(expr)
                if not labels or len(labels) > 16:
                    self.finite_enum_cache[token] = None
                    return None
                codes = {label: i for i, label in enumerate(labels)}
                encoded = self._encode_finite_string_values(expr, codes)
                if encoded is None:
                    self.finite_enum_cache[token] = None
                    return None
                encoded = ast.fix_missing_locations(encoded)
                if any(
                    isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id in frontend.variant_trace_by_uid
                    and frontend.variant_trace_by_uid[child.func.id].dtype == "object"
                    for child in ast.walk(encoded)
                ):
                    # A condition that still depends on an object-valued Cell has
                    # not actually been converted into a numeric finite selector.
                    self.finite_enum_cache[token] = None
                    return None
                out = FiniteEnumValue(
                    labels=labels, code_expr=encoded, source_name=tr.schema.name
                )
                self.finite_enum_cache[token] = out
                return out

            @staticmethod
            def _enum_label_code(value: FiniteEnumValue, label: str) -> int | None:
                try:
                    return value.labels.index(label)
                except ValueError:
                    return None

            def _finite_dispatch(
                self, table: TableSource, row: ast.AST
            ) -> FiniteDispatch | None:
                if not isinstance(table.sample.index, pd.MultiIndex):
                    return None
                parts = self._row_parts(row, table)
                fixed: dict[int, Any] = {}
                enum_rows: list[tuple[int, FiniteEnumValue]] = []
                numeric_rows: list[tuple[int, ast.AST]] = []
                for level, expr in enumerate(parts):
                    ok, value = self._static_value(expr)
                    if ok:
                        fixed[level] = value
                        continue
                    enum_value = self._finite_enum(expr)
                    if enum_value is not None:
                        enum_rows.append((level, enum_value))
                    else:
                        numeric_rows.append((level, self._expand_alias(expr)))
                if len(enum_rows) != 1 or len(numeric_rows) != 1:
                    return None
                enum_level, enum_value = enum_rows[0]
                numeric_level, numeric_expr = numeric_rows[0]
                return FiniteDispatch(
                    fixed_levels=tuple(sorted(fixed.items())),
                    enum_level=int(enum_level), enum_value=enum_value,
                    numeric_level=int(numeric_level), numeric_expr=copy.deepcopy(numeric_expr),
                )

            def _dispatch_numeric_labels(
                self, table: TableSource, dispatch: FiniteDispatch, label: str
            ) -> tuple[Any, ...]:
                fixed = dict(dispatch.fixed_levels)
                fixed[dispatch.enum_level] = label
                labels: list[Any] = []
                for row in table.sample.index:
                    parts = row if isinstance(row, tuple) else (row,)
                    if all(parts[level] == value for level, value in fixed.items()):
                        labels.append(parts[dispatch.numeric_level])
                return tuple(labels)

            @staticmethod
            def _finite_numeric_exact_labels(
                labels: Sequence[Any], *, cap: int = 128
            ) -> tuple[tuple[Any, ...], tuple[int, ...]]:
                """Normalize a bounded finite integer label set without assuming spacing."""
                if not labels or len(labels) > int(cap):
                    raise FrontendError(
                        "finite irregular exact axis is empty or exceeds the bounded native dispatch cap"
                    )
                pairs: list[tuple[int, Any]] = []
                for label in labels:
                    raw = label
                    if isinstance(raw, str):
                        try:
                            raw = int(raw)
                        except Exception as exc:
                            raise FrontendError(
                                f"non-integral finite exact table label {label!r}"
                            ) from exc
                    if not isinstance(raw, (int, np.integer)) or isinstance(raw, (bool, np.bool_)):
                        raise FrontendError(
                            f"non-integral finite exact table label {label!r}"
                        )
                    pairs.append((int(raw), label))
                pairs.sort(key=lambda item: item[0])
                numeric = tuple(value for value, _source in pairs)
                if len(set(numeric)) != len(numeric):
                    raise FrontendError("finite exact table labels are not numerically unique")
                return tuple(source for _value, source in pairs), numeric

            def _finite_numeric_exact_code(
                self, expr: ast.AST, labels: Sequence[Any]
            ) -> tuple[tuple[Any, ...], ast.AST]:
                source_labels, numeric = self._finite_numeric_exact_labels(labels)
                x = self.visit(copy.deepcopy(expr))
                result: ast.AST = ast.Constant(-1)
                for code, value in reversed(tuple(enumerate(numeric))):
                    result = ast.IfExp(
                        test=ast.Compare(
                            copy.deepcopy(x), [ast.Eq()], [ast.Constant(int(value))]
                        ),
                        body=ast.Constant(int(code)),
                        orelse=result,
                    )
                return source_labels, ast.fix_missing_locations(result)

            @staticmethod
            def _exact_integral_selector_value(value: Any) -> int:
                """Normalize one runtime numeric selector without rounding.

                Integral floats are accepted because Python/pandas numeric label
                equality treats ``4`` and ``4.0`` as the same exact key.  Booleans,
                non-finite values and fractional floats fail closed.
                """
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (bool, np.bool_)):
                    raise FrontendError("finite numeric exact selector cannot be boolean")
                if isinstance(value, (int, np.integer)):
                    return int(value)
                if isinstance(value, (float, np.floating)):
                    value = float(value)
                    if not np.isfinite(value) or not value.is_integer():
                        raise FrontendError(
                            f"finite numeric exact selector {value!r} is not an exact integer label"
                        )
                    return int(value)
                raise FrontendError(
                    f"finite numeric exact selector returned non-numeric {type(value).__name__}"
                )

            def _complete_run_finite_numeric_selector(
                self, expr: ast.AST, labels: Sequence[Any]
            ) -> tuple[tuple[Any, ...], ast.AST, tuple[Any, ...]] | None:
                """Prove and bind one policy-only irregular numeric selector.

                The accepted dynamic form is deliberately narrow: a zero-argument
                Cell call in the selected Space.  It is evaluated on every key in
                the frozen RunDomain, and every observed value must belong exactly
                to the table label set.  The admitted selector is then rebound as a
                prepared int64 point input whose getter repeats the membership
                validation, so a later out-of-domain value fails before native
                execution instead of becoming ``-1``/negative indexing.

                Representative trace samples are never sufficient for this proof.
                """
                source_labels, numeric_labels = self._finite_numeric_exact_labels(labels)
                expanded = self._expand_alias(copy.deepcopy(expr))
                if not (
                    isinstance(expanded, ast.Call)
                    and isinstance(expanded.func, ast.Name)
                    and not expanded.args and not expanded.keywords
                ):
                    return None
                source_name = getattr(frontend, "formula_uid_to_source_name", {}).get(
                    expanded.func.id, expanded.func.id
                )
                if source_name == expanded.func.id:
                    traced = frontend.variant_trace_by_uid.get(expanded.func.id)
                    if traced is not None:
                        source_name = traced.schema.name
                # The historical frontend catalogs source functions by source
                # name, while FastFormulaTranslationService intentionally
                # catalogs modelx-cython canonical UIDs and carries a UID->source
                # bridge beside them.  Accept either catalog identity here; live
                # complete-RunDomain evaluation still uses the original Cell name.
                base = frontend.base_funcs.get(expanded.func.id)
                if base is None:
                    base = frontend.base_funcs.get(source_name)
                if base is None:
                    return None
                if (
                    base.args.posonlyargs or base.args.args or base.args.kwonlyargs
                    or base.args.vararg is not None or base.args.kwarg is not None
                ):
                    return None

                allowed = set(int(x) for x in numeric_labels)

                def evaluate(keys, name=source_name, allowed_values=allowed):
                    values: list[int] = []
                    for key in keys:
                        value = getattr(frontend._space_instance(key), name)()
                        numeric = self._exact_integral_selector_value(value)
                        if numeric not in allowed_values:
                            raise FrontendError(
                                f"finite numeric exact selector {name!r} produced {numeric!r} "
                                f"outside frozen table labels {tuple(sorted(allowed_values))!r}"
                            )
                        values.append(numeric)
                    return np.asarray(values, dtype=np.int64)

                try:
                    complete = evaluate(frontend.run_keys)
                except Exception:
                    return None
                if complete.shape != (len(frontend.run_keys),):
                    return None

                token = (
                    "finite_numeric_exact_selector", source_name,
                    tuple(int(x) for x in numeric_labels), len(frontend.run_keys),
                )
                input_key = registry.frozen_point_scalar(
                    token,
                    "int64",
                    evaluate,
                    f"complete-RunDomain finite numeric exact selector {source_name}",
                )
                runtime_expr = ast.Call(
                    ast.Name("point_input", ast.Load()), [ast.Constant(input_key)], []
                )
                source_by_numeric = {
                    int(numeric): source
                    for source, numeric in zip(source_labels, numeric_labels)
                }
                reachable_numeric = tuple(
                    value for value in numeric_labels if int(value) in set(map(int, complete.tolist()))
                )
                reachable_source = tuple(source_by_numeric[int(value)] for value in reachable_numeric)
                frontend._used_finite_domain_facts[("numeric_selector", source_name)] = FiniteDomainFact(
                    "numeric_selector", source_name, tuple(int(x) for x in reachable_numeric),
                    len(frontend.run_keys),
                )
                return source_labels, runtime_expr, reachable_source

            def _numeric_membership_test(self, expr: ast.AST, labels: Sequence[Any]) -> ast.AST:
                try:
                    numeric, start, step = _safe_numeric_axis(labels)
                except FrontendError:
                    _source_labels, numeric = self._finite_numeric_exact_labels(labels)
                    x = self.visit(copy.deepcopy(expr))
                    tests = [
                        ast.Compare(
                            copy.deepcopy(x), [ast.Eq()], [ast.Constant(int(value))]
                        )
                        for value in numeric
                    ]
                    if len(tests) == 1:
                        return tests[0]
                    return ast.BoolOp(ast.Or(), tests)
                lo, hi = min(numeric), max(numeric)
                x = self.visit(copy.deepcopy(expr))
                tests: list[ast.AST] = [
                    ast.Compare(copy.deepcopy(x), [ast.GtE()], [ast.Constant(lo)]),
                    ast.Compare(copy.deepcopy(x), [ast.LtE()], [ast.Constant(hi)]),
                ]
                if step != 1:
                    tests.append(ast.Compare(
                        ast.BinOp(
                            ast.BinOp(copy.deepcopy(x), ast.Sub(), ast.Constant(start)),
                            ast.Mod(), ast.Constant(step),
                        ),
                        [ast.Eq()], [ast.Constant(0)],
                    ))
                return ast.BoolOp(ast.And(), tests)

            def _row_is_membership_guarded(
                self, table: TableSource, row: ast.AST
            ) -> bool:
                try:
                    actual = self._row_parts(row, table)
                except FrontendError:
                    return False
                actual_sig = tuple(ast.dump(part, include_attributes=False) for part in actual)
                for key_name, table_name in (
                    *self.membership_guarded_rows,
                    *self.active_membership_guarded_rows,
                ):
                    candidate = self.table_aliases.get(table_name)
                    if candidate is None or (
                        candidate.ref_name != table.ref_name or candidate.cell_name != table.cell_name
                    ):
                        continue
                    if key_name not in self.value_aliases:
                        continue
                    try:
                        expected = self._row_parts(ast.Name(key_name, ast.Load()), table)
                    except FrontendError:
                        continue
                    expected_sig = tuple(ast.dump(part, include_attributes=False) for part in expected)
                    if expected_sig == actual_sig:
                        return True
                return False

            @staticmethod
            def _inline_membership_guard(
                test: ast.AST,
            ) -> tuple[str, str, bool] | None:
                """Recognize one local ``key in table.index`` branch contract.

                The returned boolean tells whether the true branch is guarded.
                Guard scope is applied only while visiting that branch.
                """
                if not (
                    isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], (ast.In, ast.NotIn))
                    and len(test.comparators) == 1
                    and isinstance(test.left, ast.Name)
                    and isinstance(test.comparators[0], ast.Attribute)
                    and test.comparators[0].attr == "index"
                    and isinstance(test.comparators[0].value, ast.Name)
                ):
                    return None
                return (
                    test.left.id,
                    test.comparators[0].value.id,
                    isinstance(test.ops[0], ast.In),
                )

            def _series_axis(self, source: SeriesSource) -> tuple[str, int, int, tuple[float, ...]]:
                key, start, step = registry.external_dataframe_axis(
                    source.table.ref_name, source.table.cell_name, source.table.cell,
                    fixed_levels=dict(source.fixed_levels),
                    dynamic_level=source.dynamic_level,
                    column=source.column,
                )
                sem = registry.specs[key].normalized_table
                assert sem is not None
                labels = tuple(float(x) for x in sem.dynamic_labels)
                return key, start, step, labels

            def _expand_alias(self, node: ast.AST, depth: int = 0) -> ast.AST:
                if depth > 8:
                    return node
                if isinstance(node, ast.Name) and node.id in self.value_aliases:
                    return self._expand_alias(copy.deepcopy(self.value_aliases[node.id]), depth + 1)
                return node

            def _analysis_expand_alias(self, node: ast.AST) -> ast.AST:
                """Resolve proof-only scalar aliases without cloning their ASTs.

                Numeric-bound analysis never mutates its input tree.  Returning the
                stored alias expression directly avoids repeatedly deep-copying a
                large helper expression at every recursive proof node.  Alias
                cycles/depth overflow are conservative: leave the last Name
                unresolved so the bound proof returns unknown.
                """
                current, hops, cycle = _resolve_proof_alias(
                    node, self.value_aliases, max_hops=8
                )
                frontend._proof_diagnostics["numeric_bounds_alias_hops"] += hops
                if cycle:
                    frontend._proof_diagnostics["numeric_bounds_alias_cycle_cuts"] += 1
                return current

            def _static_value(self, node: ast.AST) -> tuple[bool, Any]:
                node = self._expand_alias(node)
                if isinstance(node, ast.Constant):
                    return True, node.value
                if isinstance(node, ast.Tuple):
                    values = []
                    for item in node.elts:
                        ok, value = self._static_value(item)
                        if not ok:
                            return False, None
                        values.append(value)
                    return True, tuple(values)
                if isinstance(node, ast.UnaryOp):
                    ok, value = self._static_value(node.operand)
                    if not ok:
                        return False, None
                    try:
                        if isinstance(node.op, ast.USub): return True, -value
                        if isinstance(node.op, ast.UAdd): return True, +value
                        if isinstance(node.op, ast.Not): return True, not value
                    except Exception:
                        return False, None
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if not node.args and not node.keywords:
                        if node.func.id in frontend.variant_trace_by_uid:
                            return frontend._proven_static_variant_value(node.func.id)
                        if node.func.id in frontend.base_funcs:
                            ok, value = frontend._proven_static_cell_value(node.func.id)
                            if ok:
                                return True, value
                        source_name = getattr(
                            frontend, "formula_uid_to_source_name", {}
                        ).get(node.func.id)
                        if source_name is not None:
                            ok, value = frontend._proven_static_cell_value(source_name)
                            if ok:
                                return True, value
                    if node.func.id in {"int", "float", "str", "bool"} and len(node.args) == 1 and not node.keywords:
                        ok, value = self._static_value(node.args[0])
                        if not ok:
                            return False, None
                        try:
                            return True, {"int": int, "float": float, "str": str, "bool": bool}[node.func.id](value)
                        except Exception:
                            return False, None
                return False, None

            def _row_parts(self, row: ast.AST, table: TableSource) -> list[ast.AST]:
                row = self._expand_alias(row)
                nlevels = int(table.sample.index.nlevels) if isinstance(table.sample.index, pd.MultiIndex) else 1
                if nlevels == 1:
                    return [row]
                if isinstance(row, ast.Tuple) and len(row.elts) == nlevels:
                    return [self._expand_alias(x) for x in row.elts]
                raise FrontendError(
                    f"external table {table.ref_name}.{table.cell_name} requires a {nlevels}-part row key"
                )

            def _column_value(self, node: ast.AST, table: TableSource) -> Any:
                ok, value = self._static_value(node)
                if not ok:
                    raise FrontendError(
                        f"external table {table.ref_name}.{table.cell_name} requires a compile-time column label"
                    )
                if value not in table.sample.columns:
                    raise FrontendError(
                        f"external table {table.ref_name}.{table.cell_name} has no column {value!r}"
                    )
                return value

            def _record_exact_table_operator(
                self,
                *,
                table: TableSource,
                column: Any,
                value_key: str,
                sparse_key: str | None,
                layout: str,
                axes: Sequence[NormalizedTableAxis],
                dynamic_exprs: Sequence[ast.AST],
                fixed_levels: Mapping[int, Any],
                provenance_tag: str = "external_dataframe_exact_lookup",
            ) -> None:
                if operator_uid is None:
                    return
                binding = (
                    ("source_ref", table.ref_name),
                    ("source_cell", table.cell_name),
                    ("column", repr(column)),
                    ("layout", layout),
                    ("value_input_key", value_key),
                    ("sparse_key_input", sparse_key),
                    ("fixed_levels", tuple(sorted((int(k), repr(v)) for k, v in fixed_levels.items()))),
                    ("axes", tuple(
                        (
                            axis.level, axis.index_name, axis.selector_kind,
                            tuple(axis.labels), axis.axis_start, axis.axis_step,
                            axis.axis_input_key,
                            axis.reachable_labels,
                        )
                        for axis in axes
                    )),
                )
                prior = frontend._normalized_operators.get(operator_uid)
                prior_bindings: tuple[tuple[str, Any], ...] = ()
                prior_dynamic: tuple[str, ...] = ()
                prior_provenance: tuple[str, ...] = ()
                if prior is not None and prior.kind == "table_lookup":
                    prior_bindings = prior.static_inputs
                    prior_dynamic = prior.dynamic_inputs
                    prior_provenance = prior.provenance
                lookup_no = sum(1 for name, _ in prior_bindings if name.startswith("lookup_"))
                frontend._normalized_operators[operator_uid] = NormalizedOperator(
                    uid=operator_uid,
                    kind="table_lookup",
                    result_dtype="float64",
                    result_shape="scalar",
                    static_inputs=prior_bindings + ((f"lookup_{lookup_no}", binding),),
                    dynamic_inputs=prior_dynamic + tuple(ast.unparse(x) for x in dynamic_exprs),
                    provenance=tuple(dict.fromkeys((*prior_provenance,
                        provenance_tag,
                        "numeric_selector_encoding",
                        layout,
                        "runtime_domain_guard",
                    ))),
                    state_semantic="state_free",
                    python_supported=True,
                    cython_supported=True,
                )

            def _point_selector_value(self, node: ast.AST, run_key: Any) -> Any:
                """Evaluate a tiny exact-selector expression against one modelx point.

                This is preparation-only.  Numeric valuation expressions never use
                this evaluator; it exists solely to materialize external-table keys
                that are already represented by ordinary modelx Cells.
                """
                node = self._expand_alias(node)
                if isinstance(node, ast.Constant):
                    return node.value
                if isinstance(node, ast.Tuple):
                    return tuple(self._point_selector_value(x, run_key) for x in node.elts)
                if isinstance(node, ast.List):
                    return [self._point_selector_value(x, run_key) for x in node.elts]
                if isinstance(node, ast.Name):
                    if node.id in frontend.space_params:
                        pos = frontend.space_params.index(node.id)
                        if len(frontend.space_params) == 1:
                            return run_key[0] if isinstance(run_key, tuple) else run_key
                        return run_key[pos] if isinstance(run_key, tuple) else run_key
                    ok, value = frontend._context_static_value(node, run_key)
                    if ok:
                        return value
                    raise FrontendError(
                        f"point table selector name {node.id!r} is not preparation-resolvable"
                    )
                if isinstance(node, ast.UnaryOp):
                    value = self._point_selector_value(node.operand, run_key)
                    if isinstance(node.op, ast.UAdd): return +value
                    if isinstance(node.op, ast.USub): return -value
                    if isinstance(node.op, ast.Not): return not bool(value)
                if isinstance(node, ast.BinOp):
                    left = self._point_selector_value(node.left, run_key)
                    right = self._point_selector_value(node.right, run_key)
                    if isinstance(node.op, ast.Add): return left + right
                    if isinstance(node.op, ast.Sub): return left - right
                    if isinstance(node.op, ast.Mult): return left * right
                    if isinstance(node.op, ast.Div): return left / right
                    if isinstance(node.op, ast.FloorDiv): return left // right
                    if isinstance(node.op, ast.Mod): return left % right
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
                    args = [self._point_selector_value(arg, run_key) for arg in node.args]
                    if node.func.id in {"int", "float", "bool", "str"} and len(args) == 1:
                        return {"int": int, "float": float, "bool": bool, "str": str}[node.func.id](args[0])
                    source_name = getattr(frontend, "formula_uid_to_source_name", {}).get(
                        node.func.id, node.func.id
                    )
                    if source_name == node.func.id:
                        traced = frontend.variant_trace_by_uid.get(node.func.id)
                        if traced is not None:
                            source_name = traced.schema.name
                    if source_name in frontend.base_funcs or source_name in frontend.space.cells:
                        member = getattr(frontend._space_instance(run_key), source_name)
                        value = member(*args)
                        return value.item() if isinstance(value, np.generic) else value
                raise FrontendError(
                    "point external-table selector is outside the preparation evaluator: "
                    + ast.unparse(node)
                )

            def _point_prepared_exact_lookup(
                self,
                table: TableSource,
                parts: Sequence[ast.AST],
                column: Any,
                origin: ast.AST,
            ) -> ast.AST:
                selector_parts = tuple(
                    self._expand_alias(copy.deepcopy(part)) for part in parts
                )
                selector_token = tuple(
                    ast.dump(part, include_attributes=False) for part in selector_parts
                )

                def selector_getter(run_key, selector_parts=selector_parts):
                    values = tuple(
                        self._point_selector_value(part, run_key)
                        for part in selector_parts
                    )
                    return values[0] if len(values) == 1 else values

                value_key, valid_key = registry.external_dataframe_point_lookup(
                    table.ref_name,
                    table.cell_name,
                    table.cell,
                    column=column,
                    selector_token=selector_token,
                    selector_getter=selector_getter,
                )
                if operator_uid is not None:
                    prior = frontend._normalized_operators.get(operator_uid)
                    prior_bindings = prior.static_inputs if prior is not None else ()
                    prior_dynamic = prior.dynamic_inputs if prior is not None else ()
                    prior_provenance = prior.provenance if prior is not None else ()
                    lookup_no = sum(
                        1 for name, _ in prior_bindings if name.startswith("point_lookup_")
                    )
                    frontend._normalized_operators[operator_uid] = NormalizedOperator(
                        uid=operator_uid,
                        kind="table_lookup",
                        result_dtype=registry.specs[value_key].dtype,
                        result_shape="scalar",
                        static_inputs=prior_bindings + ((
                            f"point_lookup_{lookup_no}",
                            (
                                ("source_ref", table.ref_name),
                                ("source_cell", table.cell_name),
                                ("column", repr(column)),
                                ("value_input_key", value_key),
                                ("valid_input_key", valid_key),
                                ("layout", "point_prepared_exact"),
                            ),
                        ),),
                        dynamic_inputs=prior_dynamic + tuple(
                            ast.unparse(part) for part in selector_parts
                        ),
                        provenance=tuple(dict.fromkeys((
                            *prior_provenance,
                            "point_prepared_exact_table_lookup",
                            "runtime_validity_guard",
                            "runtime_domain_guard",
                        ))),
                        state_semantic="state_free",
                        python_supported=True,
                        cython_supported=True,
                    )
                return ast.copy_location(
                    ast.Call(
                        ast.Name("__prepared_point_lookup__", ast.Load()),
                        [ast.Constant(value_key), ast.Constant(valid_key)],
                        [],
                    ),
                    origin,
                )

            def _exact_nd_lookup(
                self,
                table: TableSource,
                parts: Sequence[ast.AST],
                fixed: Mapping[int, Any],
                dynamic: Sequence[tuple[int, ast.AST]],
                column: Any,
                origin: ast.AST,
                membership_guarded: bool = False,
            ) -> ast.AST:
                axes: list[NormalizedTableAxis] = []
                selector_codes: list[ast.AST] = []
                selector_sources: list[ast.AST] = []
                fixed_dict = dict(fixed)
                nlevels = int(table.sample.index.nlevels) if isinstance(table.sample.index, pd.MultiIndex) else 1
                index_names = tuple(table.sample.index.names) if isinstance(table.sample.index, pd.MultiIndex) else (table.sample.index.name,)

                for level, raw_expr in dynamic:
                    expr = self._expand_alias(copy.deepcopy(raw_expr))
                    selector_sources.append(copy.deepcopy(expr))
                    enum_value = self._finite_enum(expr)
                    if enum_value is not None:
                        axes.append(NormalizedTableAxis(
                            level=int(level), index_name=index_names[level],
                            selector_kind="finite_categorical",
                            labels=tuple(enum_value.labels),
                            reachable_labels=self._reachable_category_labels(expr, enum_value.labels),
                        ))
                        selector_codes.append(ast.Call(
                            ast.Name("__exact_axis_code__", ast.Load()),
                            [
                                self.visit(copy.deepcopy(enum_value.code_expr)),
                                ast.Constant(0), ast.Constant(1),
                                ast.Constant(len(enum_value.labels)),
                            ], [],
                        ))
                        continue

                    labels: list[Any] = []
                    for row_label in table.sample.index:
                        row_parts = row_label if isinstance(row_label, tuple) else (row_label,)
                        if all(row_parts[k] == value for k, value in fixed_dict.items()):
                            labels.append(row_parts[level])
                    unique_labels = tuple(sorted(set(labels)))
                    try:
                        numeric_labels, start, step = _safe_numeric_axis(unique_labels)
                    except FrontendError as exc:
                        complete_proof = None
                        if not membership_guarded:
                            complete_proof = self._complete_run_finite_numeric_selector(
                                expr, unique_labels
                            )
                            if complete_proof is None:
                                raise FrontendError(
                                    f"external table {table.ref_name}.{table.cell_name} dynamic level "
                                    f"{level} selector {ast.unparse(expr)!r} is neither a proven finite "
                                    f"categorical selector nor a regular numeric exact axis: {exc}"
                                ) from exc
                        if complete_proof is None:
                            source_labels, code_expr = self._finite_numeric_exact_code(
                                expr, unique_labels
                            )
                            reachable_labels = None
                        else:
                            source_labels, runtime_expr, reachable_labels = complete_proof
                            source_labels, code_expr = self._finite_numeric_exact_code(
                                runtime_expr, source_labels
                            )
                        axes.append(NormalizedTableAxis(
                            level=int(level), index_name=index_names[level],
                            selector_kind="finite_numeric_exact",
                            labels=source_labels,
                            reachable_labels=reachable_labels,
                        ))
                        selector_codes.append(code_expr)
                        continue
                    # Preserve the source labels used to slice/materialize the
                    # pandas table while deriving numeric runtime codes from their
                    # integral interpretation.  This matters for legitimate axes
                    # such as columns ``"0".."5"`` whose runtime selector is an
                    # integer but whose source labels are strings.
                    numeric_to_source = sorted(
                        zip(numeric_labels, unique_labels), key=lambda item: item[0]
                    )
                    source_labels = tuple(label for _numeric, label in numeric_to_source)
                    visited = self.visit(copy.deepcopy(expr))
                    selector_codes.append(ast.Call(
                        ast.Name("__exact_axis_code__", ast.Load()),
                        [
                            visited,
                            ast.Constant(int(start)), ast.Constant(int(step)),
                            ast.Constant(len(source_labels)),
                        ], [],
                    ))
                    axes.append(NormalizedTableAxis(
                        level=int(level), index_name=index_names[level],
                        selector_kind="numeric_exact", labels=source_labels,
                        axis_start=int(start), axis_step=int(step),
                    ))

                try:
                    value_key, sparse_key, layout = registry.external_dataframe_exact_nd(
                        table.ref_name, table.cell_name, table.cell,
                        fixed_levels=fixed_dict, dynamic_axes=tuple(axes), column=column,
                        allow_missing_categorical_labels=membership_guarded,
                    )
                except FrontendError as exc:
                    # A table may legitimately cover only the branch-specific
                    # subset of a broader finite categorical domain.  Rather than
                    # rejecting the whole Cell or manufacturing absent table rows,
                    # bind the exact selector per RunDomain point and carry an
                    # explicit validity bit into the generated branch.
                    if "is missing selector labels" not in str(exc):
                        raise
                    return self._point_prepared_exact_lookup(
                        table, parts, column, origin
                    )
                self._record_exact_table_operator(
                    table=table, column=column, value_key=value_key,
                    sparse_key=sparse_key, layout=layout, axes=axes,
                    dynamic_exprs=selector_sources,
                    fixed_levels=fixed_dict,
                )
                if layout == "dense_cartesian":
                    return ast.copy_location(
                        ast.Call(
                            ast.Name("table_input", ast.Load()),
                            [ast.Constant(value_key), *selector_codes], [],
                        ),
                        origin,
                    )
                assert sparse_key is not None
                return ast.copy_location(
                    ast.Call(
                        ast.Name("__sparse_table_lookup__", ast.Load()),
                        [ast.Constant(sparse_key), ast.Constant(value_key), *selector_codes], [],
                    ),
                    origin,
                )

            def _lookup(self, table: TableSource, row: ast.AST, column_node: ast.AST, origin: ast.AST) -> ast.AST:
                ok_column, _static_column = self._static_value(column_node)
                if not ok_column:
                    column_enum = self._finite_enum(column_node)
                    if column_enum is not None:
                        labels = tuple(column_enum.labels)
                        if not labels or any(label not in table.sample.columns for label in labels):
                            raise FrontendError(
                                f"external table {table.ref_name}.{table.cell_name} finite column selector "
                                "exceeds the frozen column domain"
                            )
                        kinds = tuple(np.asarray(table.sample[label]).dtype.kind for label in labels)
                        if any(kind not in "iubfc" for kind in kinds):
                            raise FrontendError(
                                f"external table {table.ref_name}.{table.cell_name} finite column selector "
                                "includes a non-numeric column"
                            )

                        selector = self.visit(copy.deepcopy(column_enum.code_expr))
                        branches = [
                            self._lookup(
                                table, copy.deepcopy(row), ast.Constant(label), origin
                            )
                            for label in labels
                        ]
                        result = branches[-1]
                        for code, branch in reversed(tuple(enumerate(branches[:-1]))):
                            result = ast.IfExp(
                                test=ast.Compare(
                                    copy.deepcopy(selector), [ast.Eq()], [ast.Constant(code)]
                                ),
                                body=branch,
                                orelse=result,
                            )
                        if operator_uid is not None:
                            prior = frontend._normalized_operators.get(operator_uid)
                            if prior is not None and prior.kind == "table_lookup":
                                frontend._normalized_operators[operator_uid] = NormalizedOperator(
                                    uid=prior.uid,
                                    kind=prior.kind,
                                    result_dtype=prior.result_dtype,
                                    result_shape=prior.result_shape,
                                    static_inputs=prior.static_inputs + ((
                                        "finite_column_labels", tuple(map(repr, labels))
                                    ),),
                                    dynamic_inputs=prior.dynamic_inputs + (
                                        ast.unparse(self._expand_alias(copy.deepcopy(column_node))),
                                    ),
                                    provenance=tuple(dict.fromkeys((
                                        *prior.provenance,
                                        "finite_categorical_column_dispatch",
                                    ))),
                                    state_semantic=prior.state_semantic,
                                    python_supported=prior.python_supported,
                                    cython_supported=prior.cython_supported,
                                )
                        return ast.copy_location(ast.fix_missing_locations(result), origin)
                column = self._column_value(column_node, table)
                parts = self._row_parts(row, table)
                fixed: dict[int, Any] = {}
                dynamic: list[tuple[int, ast.AST]] = []
                for level, expr in enumerate(parts):
                    ok, value = self._static_value(expr)
                    if ok:
                        fixed[level] = value
                    else:
                        dynamic.append((level, self._expand_alias(expr)))
                if not dynamic:
                    row_key: Any
                    values = tuple(fixed[i] for i in range(len(parts)))
                    row_key = values[0] if len(values) == 1 else values
                    key = registry.external_dataframe_scalar(
                        table.ref_name, table.cell_name, table.cell, row_key, column
                    )
                    return ast.copy_location(
                        ast.Call(ast.Name("global_input", ast.Load()), [ast.Constant(key)], []), origin
                    )
                # A single categorical axis belongs to the same exact finite-table
                # semantics as N-D categorical lookup.  Route it through the mature
                # exact-ND materializer so semantic category labels, not raw integer
                # codes or pandas row positions, determine the table mapping.
                if len(dynamic) == 1 and self._finite_enum(dynamic[0][1]) is not None:
                    return self._exact_nd_lookup(
                        table, parts, fixed, dynamic, column, origin,
                        membership_guarded=self._row_is_membership_guarded(table, row),
                    )
                if len(dynamic) == 1:
                    level, _expr = dynamic[0]
                    labels: list[Any] = []
                    for row_label in table.sample.index:
                        row_parts = row_label if isinstance(row_label, tuple) else (row_label,)
                        if all(row_parts[k] == value for k, value in fixed.items()):
                            labels.append(row_parts[level])
                    try:
                        _safe_numeric_axis(tuple(sorted(set(labels))))
                    except FrontendError:
                        # Reuse exact-ND finite-axis semantics for an irregular
                        # one-dimensional axis too.  The exact-ND normalizer owns
                        # membership-guard and complete-RunDomain selector proof;
                        # regular numeric axes retain the lean array path below.
                        return self._exact_nd_lookup(
                            table, parts, fixed, dynamic, column, origin,
                            membership_guarded=self._row_is_membership_guarded(table, row),
                        )
                if len(dynamic) >= 2:
                    return self._exact_nd_lookup(
                        table, parts, fixed, dynamic, column, origin,
                        membership_guarded=self._row_is_membership_guarded(table, row),
                    )
                if len(dynamic) != 1:
                    dispatch = self._finite_dispatch(table, row)
                    if dispatch is None:
                        raise FrontendError(
                            f"external table {table.ref_name}.{table.cell_name} lookup has {len(dynamic)} dynamic row levels; "
                            "current optimized table subset supports at most one numeric level unless one other level "
                            "is a structurally proven finite categorical selector"
                        )
                    guarded = self._row_is_membership_guarded(table, row)
                    enum_expr = self.visit(copy.deepcopy(dispatch.enum_value.code_expr))
                    numeric_bounds = self._numeric_bounds(dispatch.numeric_expr)
                    result: ast.AST | None = None
                    col_kind = np.asarray(table.sample[column]).dtype.kind
                    fallback: ast.AST = ast.Constant(0 if col_kind in "iub" else 0.0)
                    result = fallback
                    for code, label in reversed(tuple(enumerate(dispatch.enum_value.labels))):
                        labels = self._dispatch_numeric_labels(table, dispatch, label)
                        if not labels:
                            if not guarded:
                                raise FrontendError(
                                    f"finite categorical selector label {label!r} has no rows in "
                                    f"{table.ref_name}.{table.cell_name} and the source lookup is not protected by "
                                    "a direct membership-return guard"
                                )
                            continue
                        numeric, axis_start, axis_step = _safe_numeric_axis(labels)
                        lo, hi = min(numeric), max(numeric)
                        qlo, qhi = numeric_bounds
                        if not guarded and (
                            qlo is None or qhi is None or qlo < lo or qhi > hi
                            or (axis_step != 1 and not (
                                qlo == qhi and (qlo - axis_start) % axis_step == 0
                            ))
                        ):
                            raise FrontendError(
                                "finite categorical table lookup requires either source-proven in-domain numeric "
                                f"bounds or a direct membership-return guard; label={label!r}, "
                                f"proven={qlo!r},{qhi!r}, axis={lo!r},{hi!r}"
                            )
                        fixed_levels = dict(dispatch.fixed_levels)
                        fixed_levels[dispatch.enum_level] = label
                        key, start, step = registry.external_dataframe_axis(
                            table.ref_name, table.cell_name, table.cell,
                            fixed_levels=fixed_levels,
                            dynamic_level=dispatch.numeric_level, column=column,
                        )
                        # ``external_dataframe_axis`` is the authoritative source of
                        # executable axis semantics.  The local validation above uses
                        # the same labels only to decide whether an unguarded lookup is
                        # total over the proven query domain.
                        assert int(start) == int(axis_start) and int(step) == int(axis_step)
                        mapped = _axis_index_expr(
                            self.visit(copy.deepcopy(dispatch.numeric_expr)), start, step
                        )
                        value = ast.Call(
                            ast.Name("array_input", ast.Load()), [ast.Constant(key), mapped], []
                        )
                        test = ast.Compare(
                            copy.deepcopy(enum_expr), [ast.Eq()], [ast.Constant(code)]
                        )
                        result = ast.IfExp(test=test, body=value, orelse=result)
                    return ast.copy_location(ast.fix_missing_locations(result), origin)
                level, expr = dynamic[0]
                key, start, step = registry.external_dataframe_axis(
                    table.ref_name, table.cell_name, table.cell,
                    fixed_levels=fixed, dynamic_level=level, column=column,
                )
                mapped = _axis_index_expr(self.visit(copy.deepcopy(expr)), start, step)
                return ast.copy_location(
                    ast.Call(ast.Name("array_input", ast.Load()), [ast.Constant(key), mapped], []), origin
                )

            def _loc_parts(self, node: ast.AST) -> tuple[TableSource, ast.AST] | None:
                if not (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "loc"
                ):
                    return None
                table = self._table(node.value.value)
                if table is None:
                    return None
                return table, node.slice

            def _at_parts(
                self, node: ast.AST
            ) -> tuple[TableSource, ast.AST, ast.AST] | None:
                """Recognize exact scalar pandas ``.at`` table/Series reads.

                ``DataFrame.at[row, column]`` and ``DataFrame[column].at[row]``
                carry the same exact-label semantics already owned by ``_lookup``.
                Reuse that mature preparation-time table normalization rather than
                allowing pandas' scalar accessor object to leak into the recursive
                runtime.  Deliberately do not generalize to ``.iat`` (positional
                semantics) or assignment through ``.at``.
                """
                if not (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "at"
                ):
                    return None

                owner = node.value.value

                if isinstance(owner, ast.Name) and owner.id in self.table_column_aliases:
                    table, column = self.table_column_aliases[owner.id]
                    return table, node.slice, column

                # DataFrame.at[row, column]
                table = self._table(owner)
                if table is not None:
                    if not (
                        isinstance(node.slice, ast.Tuple)
                        and len(node.slice.elts) == 2
                    ):
                        return None
                    row, column = node.slice.elts
                    return table, row, column

                # DataFrame[column].at[row] -> exact Series scalar lookup.
                if isinstance(owner, ast.Subscript):
                    table = self._table(owner.value)
                    if table is not None:
                        return table, node.slice, owner.slice
                return None

            def _is_scalar_loc(self, table: TableSource, slice_node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
                if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) != 2:
                    return None
                first, second = slice_node.elts
                if isinstance(first, ast.Tuple):
                    return first, second
                expanded = self._expand_alias(first)
                if isinstance(expanded, ast.Tuple):
                    ok, col = self._static_value(second)
                    if ok and col in table.sample.columns:
                        return expanded, second
                nlevels = int(table.sample.index.nlevels) if isinstance(table.sample.index, pd.MultiIndex) else 1
                if nlevels == 1:
                    return first, second
                ok, col = self._static_value(second)
                if ok and col in table.sample.columns:
                    return first, second
                return None

            def _row_proxy(self, node: ast.AST) -> RowSource | None:
                loc = self._loc_parts(node)
                if loc is None:
                    return None
                table, slice_node = loc
                if self._is_scalar_loc(table, slice_node) is not None:
                    return None
                return RowSource(table, self._expand_alias(slice_node))

            def _row_view_axis_extreme_lookup(
                self, row: RowSource, *, extreme: str, origin: ast.AST
            ) -> ast.AST | None:
                table = row.table
                if not isinstance(table.sample.index, pd.MultiIndex):
                    return None
                nlevels = int(table.sample.index.nlevels)
                prefix = self._expand_alias(copy.deepcopy(row.row_expr))
                prefix_parts = list(prefix.elts) if isinstance(prefix, ast.Tuple) else [prefix]
                # Pandas partial MultiIndex row selection fixes leading levels.
                # This normalized subset requires exactly one residual level.
                if len(prefix_parts) != nlevels - 1:
                    return None
                residual_level = len(prefix_parts)
                index_names = tuple(table.sample.index.names)

                fixed: dict[int, Any] = {}
                dynamic: list[tuple[int, ast.AST]] = []
                for level, expr in enumerate(prefix_parts):
                    ok, value = self._static_value(expr)
                    if ok:
                        fixed[level] = value
                    else:
                        dynamic.append((level, self._expand_alias(copy.deepcopy(expr))))

                axes: list[NormalizedTableAxis] = []
                selector_codes: list[ast.AST] = []
                selector_sources: list[ast.AST] = []
                for level, raw_expr in dynamic:
                    expr = self._expand_alias(copy.deepcopy(raw_expr))
                    selector_sources.append(copy.deepcopy(expr))
                    enum_value = self._finite_enum(expr)
                    if enum_value is not None:
                        axes.append(NormalizedTableAxis(
                            level=int(level), index_name=index_names[level],
                            selector_kind="finite_categorical",
                            labels=tuple(enum_value.labels),
                        ))
                        selector_codes.append(ast.Call(
                            ast.Name("__exact_axis_code__", ast.Load()),
                            [
                                self.visit(copy.deepcopy(enum_value.code_expr)),
                                ast.Constant(0), ast.Constant(1),
                                ast.Constant(len(enum_value.labels)),
                            ], [],
                        ))
                        continue

                    labels: list[Any] = []
                    for row_label in table.sample.index:
                        parts = row_label if isinstance(row_label, tuple) else (row_label,)
                        if all(parts[k] == value for k, value in fixed.items()):
                            labels.append(parts[level])
                    unique_labels = tuple(sorted(set(labels)))
                    try:
                        numeric_labels, start, step = _safe_numeric_axis(unique_labels)
                    except FrontendError as exc:
                        raise FrontendError(
                            f"external table {table.ref_name}.{table.cell_name} row-view prefix level "
                            f"{level} selector {ast.unparse(expr)!r} is neither a proven finite "
                            f"categorical selector nor a regular numeric exact axis: {exc}"
                        ) from exc
                    numeric_to_source = sorted(
                        zip(numeric_labels, unique_labels), key=lambda item: item[0]
                    )
                    source_labels = tuple(label for _numeric, label in numeric_to_source)
                    visited = self.visit(copy.deepcopy(expr))
                    selector_codes.append(ast.Call(
                        ast.Name("__exact_axis_code__", ast.Load()),
                        [
                            visited,
                            ast.Constant(int(start)), ast.Constant(int(step)),
                            ast.Constant(len(source_labels)),
                        ], [],
                    ))
                    axes.append(NormalizedTableAxis(
                        level=int(level), index_name=index_names[level],
                        selector_kind="numeric_exact", labels=source_labels,
                        axis_start=int(start), axis_step=int(step),
                    ))

                value_key, sparse_key, layout = registry.external_dataframe_row_axis_extreme(
                    table.ref_name, table.cell_name, table.cell,
                    fixed_levels=fixed, dynamic_axes=tuple(axes),
                    residual_level=residual_level, extreme=extreme,
                )
                self._record_exact_table_operator(
                    table=table,
                    column=(f"__index_{extreme}__", residual_level),
                    value_key=value_key,
                    sparse_key=sparse_key,
                    layout=layout,
                    axes=axes,
                    dynamic_exprs=selector_sources,
                    fixed_levels=fixed,
                    provenance_tag="external_row_view_axis_metadata",
                )
                if not axes:
                    # All prefix levels are static, so the prepared metadata input
                    # is a scalar singleton domain.
                    return ast.copy_location(
                        ast.Call(ast.Name("global_input", ast.Load()), [ast.Constant(value_key)], []),
                        origin,
                    )
                if layout == "dense_cartesian":
                    return ast.copy_location(
                        ast.Call(
                            ast.Name("table_input", ast.Load()),
                            [ast.Constant(value_key), *selector_codes], [],
                        ),
                        origin,
                    )
                assert sparse_key is not None
                return ast.copy_location(
                    ast.Call(
                        ast.Name("__sparse_table_lookup__", ast.Load()),
                        [ast.Constant(sparse_key), ast.Constant(value_key), *selector_codes], [],
                    ),
                    origin,
                )

            def _series_row_source(self, source: SeriesSource) -> RowSource:
                return RowSource(source.table, copy.deepcopy(source.row_expr))

            def _series_dynamic_lookup(
                self, source: SeriesSource, query: ast.AST, origin: ast.AST
            ) -> ast.AST:
                prefix = self._expand_alias(copy.deepcopy(source.row_expr))
                parts = list(prefix.elts) if isinstance(prefix, ast.Tuple) else [prefix]
                full_row = ast.Tuple(
                    elts=[*parts, self._expand_alias(copy.deepcopy(query))], ctx=ast.Load()
                )

                # Preserve exact Series indexing semantics by proving the numeric
                # query is within every admitted residual axis.  ``index.max()``
                # metadata participates in the tiny numeric bound prover above, so
                # source clamps such as min(age(t), scale.index.max()) remain compact.
                residual_by_prefix: dict[tuple[Any, ...], list[float]] = {}
                frame = source.table.sample
                for label in frame.index:
                    row = label if isinstance(label, tuple) else (label,)
                    key = tuple(row[level] for level in range(source.dynamic_level))
                    residual_by_prefix.setdefault(key, []).append(float(row[source.dynamic_level]))
                if not residual_by_prefix:
                    raise FrontendError("provider view has an empty residual numeric axis")
                required_lo = max(min(vals) for vals in residual_by_prefix.values())
                allowed_hi = min(max(vals) for vals in residual_by_prefix.values())
                qlo, qhi = self._numeric_bounds(self._expand_alias(copy.deepcopy(query)))
                if qlo is None or qhi is None or qlo < required_lo or qhi > allowed_hi:
                    raise FrontendError(
                        "provider-view scalar lookup requires a source-proven in-domain "
                        f"numeric query {ast.unparse(query)!r}; proven bounds={qlo!r},{qhi!r}, "
                        f"common residual axis={required_lo!r},{allowed_hi!r}"
                    )
                return self._lookup(
                    source.table, full_row, ast.Constant(source.column), origin
                )

            def _table_index_extreme(self, node: ast.AST) -> ast.AST | None:
                if not (
                    isinstance(node, ast.Call) and not node.args and not node.keywords
                    and isinstance(node.func, ast.Attribute) and node.func.attr in {"min", "max"}
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "index"
                ):
                    return None
                base = node.func.value.value
                if isinstance(base, ast.Name) and base.id in self.series_aliases:
                    source = self.series_aliases[base.id]
                    if source.has_dynamic_prefix:
                        return self._row_view_axis_extreme_lookup(
                            self._series_row_source(source),
                            extreme=node.func.attr, origin=node,
                        )
                    _, _, _, labels = self._series_axis(source)
                    if not labels:
                        return None
                    extreme = min(labels) if node.func.attr == "min" else max(labels)
                    if float(extreme).is_integer():
                        extreme = int(extreme)
                    return ast.copy_location(ast.Constant(extreme), node)
                if isinstance(base, ast.Name) and base.id in self.row_aliases:
                    row = self.row_aliases[base.id]
                    ok, row_key = self._static_value(row.row_expr)
                    if not ok:
                        return self._row_view_axis_extreme_lookup(
                            row, extreme=node.func.attr, origin=node
                        )
                    try:
                        sub = row.table.sample.loc[row_key]
                    except Exception as exc:
                        raise FrontendError(
                            f"external table row slice {row.table.ref_name}.{row.table.cell_name}[{row_key!r}] failed"
                        ) from exc
                    if isinstance(sub, pd.Series):
                        # A Series row has column labels rather than a numeric row axis.
                        return None
                    if not isinstance(sub, pd.DataFrame) or isinstance(sub.index, pd.MultiIndex):
                        return None
                    labels, _, _ = _safe_numeric_axis(sub.index)
                    extreme = min(labels) if node.func.attr == "min" else max(labels)
                    return ast.copy_location(ast.Constant(extreme), node)
                table = self._table(base)
                if table is None:
                    return None
                if isinstance(table.sample.index, pd.MultiIndex):
                    raise FrontendError(
                        f"external table {table.ref_name}.{table.cell_name}.index.{node.func.attr}() on MultiIndex is outside the normalized subset"
                    )
                labels, _, _ = _safe_numeric_axis(table.sample.index)
                extreme = min(labels) if node.func.attr == "min" else max(labels)
                return ast.copy_location(ast.Constant(extreme), node)

            def _multiindex_level_extreme(self, node: ast.AST) -> ast.AST | None:
                """Resolve a pure min/max projection of one MultiIndex level.

                This is preparation-time metadata normalization, not a general
                generator-expression evaluator.  The accepted source shape is
                deliberately exact::

                    max(level_name for level_a, level_name in table.index)

                (and the analogous ``min`` form).  The generator must have one
                unfiltered synchronous iterator over the immutable prepared table
                index, and the yielded expression must be exactly one tuple-bound
                index component.  Anything more dynamic remains ordinary Python
                source and therefore fails closed later in table normalization.
                """
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {"min", "max"}
                    and len(node.args) == 1
                    and not node.keywords
                    and isinstance(node.args[0], ast.GeneratorExp)
                ):
                    return None
                gen = node.args[0]
                if len(gen.generators) != 1:
                    return None
                comp = gen.generators[0]
                if comp.ifs or comp.is_async:
                    return None
                if not (
                    isinstance(comp.iter, ast.Attribute)
                    and comp.iter.attr == "index"
                ):
                    return None
                table = self._table(comp.iter.value)
                if table is None or not isinstance(table.sample.index, pd.MultiIndex):
                    return None
                if not isinstance(comp.target, (ast.Tuple, ast.List)):
                    return None
                targets = list(comp.target.elts)
                nlevels = int(table.sample.index.nlevels)
                if len(targets) != nlevels or not all(isinstance(x, ast.Name) for x in targets):
                    return None
                names = [x.id for x in targets]
                if len(set(names)) != len(names):
                    return None
                if not isinstance(gen.elt, ast.Name) or gen.elt.id not in names:
                    return None
                level = names.index(gen.elt.id)
                values = list(table.sample.index.get_level_values(level))
                if not values:
                    raise FrontendError(
                        f"external table {table.ref_name}.{table.cell_name} has an empty index level"
                    )
                normalized: list[int | float] = []
                for value in values:
                    if (
                        not isinstance(value, (int, float, np.integer, np.floating))
                        or isinstance(value, (bool, np.bool_))
                        or not np.isfinite(float(value))
                    ):
                        raise FrontendError(
                            f"external table {table.ref_name}.{table.cell_name} index level {level} "
                            "is not finite numeric and cannot be aggregated during preparation"
                        )
                    normalized.append(value.item() if isinstance(value, np.generic) else value)
                extreme = min(normalized) if node.func.id == "min" else max(normalized)
                return ast.copy_location(ast.Constant(extreme), node)

            def _membership(self, left: ast.AST, comparator: ast.AST, origin: ast.AST) -> ast.AST | None:
                if not (isinstance(comparator, ast.Attribute) and comparator.attr == "index"):
                    return None
                table = self._table(comparator.value)
                if table is None:
                    return None
                if isinstance(table.sample.index, pd.MultiIndex):
                    parts = self._row_parts(left, table)
                    fixed = {}
                    dynamic = []
                    for level, expr in enumerate(parts):
                        ok, value = self._static_value(expr)
                        if ok: fixed[level] = value
                        else: dynamic.append((level, self._expand_alias(expr)))
                    if len(dynamic) != 1:
                        dispatch = self._finite_dispatch(table, left)
                        if dispatch is None:
                            raise FrontendError(
                                "normalized table membership requires exactly one dynamic numeric row level, "
                                "or one such level plus one structurally proven finite categorical selector"
                            )
                        enum_expr = self.visit(copy.deepcopy(dispatch.enum_value.code_expr))
                        branches: list[ast.AST] = []
                        for code, label in enumerate(dispatch.enum_value.labels):
                            labels = self._dispatch_numeric_labels(table, dispatch, label)
                            if not labels:
                                continue
                            numeric_test = self._numeric_membership_test(
                                dispatch.numeric_expr, labels
                            )
                            branches.append(ast.BoolOp(ast.And(), [
                                ast.Compare(
                                    copy.deepcopy(enum_expr), [ast.Eq()], [ast.Constant(code)]
                                ),
                                numeric_test,
                            ]))
                        if not branches:
                            return ast.copy_location(ast.Constant(False), origin)
                        replacement = branches[0] if len(branches) == 1 else ast.BoolOp(ast.Or(), branches)
                        return ast.copy_location(ast.fix_missing_locations(replacement), origin)
                    dyn_level, dyn_expr = dynamic[0]
                    labels = []
                    for label in table.sample.index:
                        vals = label if isinstance(label, tuple) else (label,)
                        if all(vals[k] == v for k, v in fixed.items()):
                            labels.append(vals[dyn_level])
                else:
                    dyn_expr = self._expand_alias(left)
                    labels = list(table.sample.index)
                return ast.copy_location(
                    ast.fix_missing_locations(
                        self._numeric_membership_test(dyn_expr, labels)
                    ),
                    origin,
                )

            def _direct_dataframe_column(self, node: ast.AST):
                if not (
                    isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id in frontend.refs
                ):
                    return None
                frame = frontend.refs[node.value.id]
                if not isinstance(frame, pd.DataFrame) or isinstance(node.slice, ast.Tuple):
                    return None
                return node.value.id, copy.deepcopy(node.slice)

            def _direct_series_index_extreme(self, node: ast.Call) -> ast.AST | None:
                if not (
                    isinstance(node.func, ast.Name) and node.func.id in {"min", "max"}
                    and len(node.args) == 1 and not node.keywords
                    and isinstance(node.args[0], ast.Attribute)
                    and node.args[0].attr == "index"
                    and isinstance(node.args[0].value, ast.Name)
                    and node.args[0].value.id in self.direct_dataframe_series_aliases
                ):
                    return None
                alias = node.args[0].value.id
                ref_name, _column = self.direct_dataframe_series_aliases[alias]
                frame = frontend.refs[ref_name]
                labels, _, _ = _safe_numeric_axis(frame.index)
                if not labels:
                    raise FrontendError(f"DataFrame Reference {ref_name} has an empty index")
                value = min(labels) if node.func.id == "min" else max(labels)
                return ast.copy_location(ast.Constant(int(value)), node)

            def visit_Assign(self, node: ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    direct_series = self._direct_dataframe_column(node.value)
                    if direct_series is not None:
                        self.direct_dataframe_series_aliases[name] = direct_series
                        return None
                    table = self._table(node.value)
                    if table is not None:
                        self.table_aliases[name] = table
                        return None
                    table_column = self._table_column(node.value)
                    if table_column is not None:
                        self.table_column_aliases[name] = table_column
                        return None
                    series = self._series(node.value)
                    if series is not None:
                        self.series_aliases[name] = series
                        return None
                    row = self._row_proxy(node.value)
                    if row is not None:
                        self.row_aliases[name] = row
                        return None
                    enum_value = self._finite_enum(node.value)
                    if enum_value is not None:
                        # Keep the local in emitted numeric code, but replace the
                        # object/string Cell call with its proven integer code.
                        code_expr = self.visit(copy.deepcopy(enum_value.code_expr))
                        self.finite_enum_aliases[name] = FiniteEnumValue(
                            labels=enum_value.labels,
                            code_expr=ast.Name(name, ast.Load()),
                            source_name=enum_value.source_name,
                        )
                        return ast.copy_location(
                            ast.Assign(targets=copy.deepcopy(node.targets), value=code_expr), node
                        )
                out = self.generic_visit(node)
                if (
                    isinstance(out, ast.Assign) and len(out.targets) == 1
                    and isinstance(out.targets[0], ast.Name)
                    and self.store_counts.get(out.targets[0].id) == 1
                    and _proof_alias_rhs_is_snapshot_safe(
                        out.targets[0].id, out.value
                    )
                ):
                    # Single-store locals are safe proof aliases.  They are never
                    # substituted into emitted code by this pass; they exist only so
                    # range checks can see through patterns such as
                    # ``y = min(policy_year(t), int(scale.index.max()))``.
                    self.value_aliases[out.targets[0].id] = copy.deepcopy(out.value)
                return out

            def visit_Call(self, node: ast.Call):
                replacement = self._direct_series_index_extreme(node)
                if replacement is not None:
                    return replacement
                replacement = self._multiindex_level_extreme(node)
                if replacement is not None:
                    return replacement
                replacement = self._table_index_extreme(node)
                if replacement is not None:
                    return replacement
                return self.generic_visit(node)

            def visit_Compare(self, node: ast.Compare):
                if (
                    len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
                    and len(node.comparators) == 1
                ):
                    pairs = (
                        (node.left, node.comparators[0], False),
                        (node.comparators[0], node.left, True),
                    )
                    for enum_node, label_node, reversed_order in pairs:
                        enum_value = self._finite_enum(enum_node)
                        if enum_value is None or not (
                            isinstance(label_node, ast.Constant) and isinstance(label_node.value, str)
                        ):
                            continue
                        code = self._enum_label_code(enum_value, label_node.value)
                        if code is None:
                            value = isinstance(node.ops[0], ast.NotEq)
                            return ast.copy_location(ast.Constant(value), node)
                        left = self.visit(copy.deepcopy(enum_value.code_expr))
                        right = ast.Constant(code)
                        if reversed_order:
                            left, right = right, left
                        return ast.copy_location(
                            ast.Compare(left, [copy.deepcopy(node.ops[0])], [right]), node
                        )
                if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) and len(node.comparators) == 1:
                    replacement = self._membership(node.left, node.comparators[0], node)
                    if replacement is not None:
                        if isinstance(node.ops[0], ast.NotIn):
                            replacement = ast.UnaryOp(ast.Not(), replacement)
                        return ast.copy_location(replacement, node)
                return self.generic_visit(node)

            def visit_IfExp(self, node: ast.IfExp):
                guard = self._inline_membership_guard(node.test)
                if guard is None:
                    return self.generic_visit(node)
                key_name, table_name, guarded_on_true = guard
                test = self.visit(copy.deepcopy(node.test))

                def visit_guarded(expr: ast.AST) -> ast.AST:
                    self.active_membership_guarded_rows.append((key_name, table_name))
                    try:
                        return self.visit(copy.deepcopy(expr))
                    finally:
                        self.active_membership_guarded_rows.pop()

                if guarded_on_true:
                    body = visit_guarded(node.body)
                    orelse = self.visit(copy.deepcopy(node.orelse))
                else:
                    body = self.visit(copy.deepcopy(node.body))
                    orelse = visit_guarded(node.orelse)
                return ast.copy_location(
                    ast.fix_missing_locations(
                        ast.IfExp(test=test, body=body, orelse=orelse)
                    ),
                    node,
                )

            def visit_Subscript(self, node: ast.Subscript):
                at = self._at_parts(node)
                if at is not None:
                    table, row, column = at
                    result = self._lookup(table, row, column, node)
                    if operator_uid is not None:
                        prior = frontend._normalized_operators.get(operator_uid)
                        if prior is not None and prior.kind == "table_lookup":
                            frontend._normalized_operators[operator_uid] = NormalizedOperator(
                                uid=prior.uid,
                                kind=prior.kind,
                                result_dtype=prior.result_dtype,
                                result_shape=prior.result_shape,
                                static_inputs=prior.static_inputs,
                                dynamic_inputs=prior.dynamic_inputs,
                                provenance=tuple(dict.fromkeys((
                                    *prior.provenance,
                                    "pandas_at_exact_scalar_read",
                                ))),
                                state_semantic=prior.state_semantic,
                                python_supported=prior.python_supported,
                                cython_supported=prior.cython_supported,
                            )
                    return result
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id in self.direct_dataframe_series_aliases
                ):
                    ref_name, column = self.direct_dataframe_series_aliases[node.value.id]
                    replacement = ast.Subscript(
                        value=ast.Subscript(
                            value=ast.Name(ref_name, ast.Load()),
                            slice=copy.deepcopy(column), ctx=ast.Load(),
                        ),
                        slice=self.visit(copy.deepcopy(node.slice)), ctx=ast.Load(),
                    )
                    return ast.copy_location(ast.fix_missing_locations(replacement), node)
                if isinstance(node.value, ast.Name) and node.value.id in self.series_aliases:
                    source = self.series_aliases[node.value.id]
                    query = self._expand_alias(copy.deepcopy(node.slice))
                    if source.has_dynamic_prefix:
                        return self._series_dynamic_lookup(source, query, node)
                    key, start, step, labels = self._series_axis(source)
                    lo, hi = self._numeric_bounds(query)
                    axis_lo, axis_hi = min(labels), max(labels)
                    if lo is None or hi is None or lo < axis_lo or hi > axis_hi:
                        correlated = frontend._correlated_int_domain_expr(
                            query, coordinate_name="t"
                        )
                        if correlated is not None:
                            corr_lo, corr_hi = float(correlated.lo), float(correlated.hi)
                            if corr_lo >= axis_lo and corr_hi <= axis_hi:
                                lo, hi = corr_lo, corr_hi
                    if lo is None or hi is None or lo < axis_lo or hi > axis_hi:
                        raise FrontendError(
                            "static-prefix Series lookup requires a source-proven in-domain "
                            f"numeric query {ast.unparse(query)!r}; "
                            f"proven bounds={lo!r},{hi!r}, axis={axis_lo!r},{axis_hi!r}"
                        )
                    mapped = _axis_index_expr(self.visit(copy.deepcopy(query)), start, step)
                    return ast.copy_location(
                        ast.Call(ast.Name("array_input", ast.Load()), [ast.Constant(key), mapped], []),
                        node,
                    )
                # Scalar lookup through a previously normalized row slice, e.g.
                # ``grid = table.loc[static_key]`` then ``grid.loc[x, column]``.
                if (
                    isinstance(node.value, ast.Attribute) and node.value.attr == "loc"
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id in self.row_aliases
                    and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2
                ):
                    row = self.row_aliases[node.value.value.id]
                    nlevels = int(row.table.sample.index.nlevels) if isinstance(row.table.sample.index, pd.MultiIndex) else 1
                    prefix = self._expand_alias(copy.deepcopy(row.row_expr))
                    prefix_parts = list(prefix.elts) if isinstance(prefix, ast.Tuple) else [prefix]
                    # A local ``tbl = frame.loc[static_prefix]`` is merely a
                    # preparation-time view when that prefix fixes all but one row
                    # level.  Reconstitute the full row key and reuse the ordinary
                    # normalized lookup path instead of giving the local DataFrame
                    # view runtime semantics.
                    if len(prefix_parts) == nlevels - 1:
                        full_row = ast.Tuple(
                            elts=[*prefix_parts, copy.deepcopy(node.slice.elts[0])],
                            ctx=ast.Load(),
                        )
                        return self._lookup(row.table, full_row, node.slice.elts[1], node)
                loc = self._loc_parts(node)
                if loc is not None:
                    table, slice_node = loc
                    scalar = self._is_scalar_loc(table, slice_node)
                    if scalar is not None:
                        return self._lookup(table, scalar[0], scalar[1], node)
                if isinstance(node.value, ast.Name) and node.value.id in self.row_aliases:
                    row = self.row_aliases[node.value.id]
                    return self._lookup(row.table, row.row_expr, node.slice, node)
                return self.generic_visit(node)

        normalizer = Normalize()
        prepared_fn = normalizer._rewrite_semantic_asof_lookup_function(copy.deepcopy(fn))
        prepared_fn = normalizer._rewrite_interp_lookup_function(prepared_fn)
        prepared_fn = normalizer._rewrite_step_lookup_function(prepared_fn)
        prepared_fn = normalizer._rewrite_interval_lookup_function(prepared_fn)
        out = normalizer.visit(prepared_fn)
        if out is None or not isinstance(out, ast.FunctionDef):
            raise FrontendError(f"external table normalization removed formula {fn.name}")
        out = ast.fix_missing_locations(out)

        removed = (
            set(normalizer.table_aliases) | set(normalizer.table_column_aliases)
            | set(normalizer.row_aliases)
            | set(normalizer.series_aliases)
        )
        residual = {
            node.id for node in ast.walk(out)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in removed
        }
        if residual:
            raise FrontendError(
                f"{fn.name}: external DataFrame operation is outside the optimized table subset: "
                + ", ".join(sorted(residual))
            )

        loads = {
            node.id for node in ast.walk(out)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        cleaned: list[ast.stmt] = []
        for st in out.body:
            if (
                isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name)
                and st.targets[0].id in normalizer.value_aliases
                and st.targets[0].id not in loads
            ):
                continue
            cleaned.append(st)
        out.body = cleaned

        # A helper may have been table-normalized before it was inlined into this
        # canonical formula.  In that case the final pass sees only ``table_input``
        # or ``__sparse_table_lookup__`` and would otherwise lose the per-formula
        # NormalizedOperator permission record.  Recover that record strictly from
        # the already-frozen numeric InputSpec semantics; no pandas operation is
        # reinterpreted here.
        if operator_uid is not None:
            prior = self._normalized_operators.get(operator_uid)
            existing_value_keys: set[str] = set()
            if prior is not None and prior.kind == "table_lookup":
                for name, payload in prior.static_inputs:
                    if not name.startswith("lookup_"):
                        continue
                    try:
                        existing_value_keys.add(str(dict(payload)["value_input_key"]))
                    except Exception:
                        pass
            for call in (
                node for node in ast.walk(out)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            ):
                value_key = None
                selector_args: list[ast.AST] = []
                if (
                    call.func.id == "table_input" and len(call.args) >= 1
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                ):
                    value_key = str(call.args[0].value)
                    selector_args = list(call.args[1:])
                elif (
                    call.func.id == "__sparse_table_lookup__" and len(call.args) >= 2
                    and isinstance(call.args[1], ast.Constant)
                    and isinstance(call.args[1].value, str)
                ):
                    value_key = str(call.args[1].value)
                    selector_args = list(call.args[2:])
                if value_key is None or value_key in existing_value_keys:
                    continue
                spec = registry.specs.get(value_key)
                sem = None if spec is None else spec.normalized_table
                if sem is None or sem.result_kind != "exact_nd":
                    continue
                binding = (
                    ("source_ref", sem.source_ref),
                    ("source_cell", sem.source_cell),
                    ("column", repr(sem.selected_column)),
                    ("layout", sem.layout),
                    ("value_input_key", value_key),
                    ("sparse_key_input", sem.sparse_key_input),
                    ("fixed_levels", tuple((int(k), repr(v)) for k, v in sem.fixed_row_labels)),
                    ("axes", tuple(
                        (
                            axis.level, axis.index_name, axis.selector_kind,
                            tuple(axis.labels), axis.axis_start, axis.axis_step,
                            axis.axis_input_key,
                        )
                        for axis in sem.dynamic_axes
                    )),
                )
                prior = self._normalized_operators.get(operator_uid)
                prior_bindings = prior.static_inputs if prior is not None and prior.kind == "table_lookup" else ()
                prior_dynamic = prior.dynamic_inputs if prior is not None and prior.kind == "table_lookup" else ()
                prior_provenance = prior.provenance if prior is not None and prior.kind == "table_lookup" else ()
                lookup_no = sum(1 for name, _ in prior_bindings if name.startswith("lookup_"))
                self._normalized_operators[operator_uid] = NormalizedOperator(
                    uid=operator_uid,
                    kind="table_lookup",
                    result_dtype=spec.dtype,
                    result_shape="scalar",
                    static_inputs=prior_bindings + ((f"lookup_{lookup_no}", binding),),
                    dynamic_inputs=prior_dynamic + tuple(ast.unparse(x) for x in selector_args),
                    provenance=tuple(dict.fromkeys((
                        *prior_provenance,
                        "normalized_table_input_recovered",
                        "numeric_selector_encoding",
                        sem.layout,
                        "runtime_domain_guard",
                    ))),
                    state_semantic="state_free",
                    python_supported=True,
                    cython_supported=True,
                )
                existing_value_keys.add(value_key)
        return ast.fix_missing_locations(out)

    def _normalize_reduction(self, fn: ast.FunctionDef) -> ReductionSpec | None:
        body=[x for x in fn.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
        # Explicit accumulator loop, semantics preserved including initial value and range args.
        if len(body)==3 and isinstance(body[0],ast.Assign) and len(body[0].targets)==1 and isinstance(body[0].targets[0],ast.Name) and isinstance(body[1],ast.For) and isinstance(body[2],ast.Return):
            acc=body[0].targets[0].id; loop=body[1]; ret=body[2]
            if isinstance(ret.value,ast.Name) and ret.value.id==acc and isinstance(loop.target,ast.Name) and isinstance(loop.iter,ast.Call) and isinstance(loop.iter.func,ast.Name) and loop.iter.func.id=="range" and len(loop.body)==1 and isinstance(loop.body[0],ast.AugAssign) and isinstance(loop.body[0].target,ast.Name) and loop.body[0].target.id==acc and isinstance(loop.body[0].op,ast.Add):
                return ReductionSpec(copy.deepcopy(body[0].value),acc,loop.target.id,tuple(copy.deepcopy(loop.iter.args)),copy.deepcopy(loop.body[0].value),())

        # Pure local scalar aliases before a terminal generator reduction may be
        # substituted into the reduction domain.  This reuses the mature
        # ReductionSpec rather than creating a separate generator runtime.  Keep
        # the admitted subset intentionally small: single-name assignments whose
        # RHS contains no Cell/provider call except native scalar intrinsics.
        local_subs: dict[str, ast.AST] = {}
        if len(body) > 1:
            for st in body[:-1]:
                if not (
                    isinstance(st, ast.Assign) and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name)
                ):
                    return None
                rhs = _Substitute(local_subs).visit(copy.deepcopy(st.value))
                unsafe = False
                for node in ast.walk(rhs):
                    if isinstance(node, ast.Call):
                        if not (
                            isinstance(node.func, ast.Name)
                            and node.func.id in {"min", "max", "abs", "int", "float", "bool", "round"}
                        ):
                            unsafe = True
                            break
                if unsafe:
                    return None
                local_subs[st.targets[0].id] = ast.fix_missing_locations(rhs)
        if not body or not isinstance(body[-1],ast.Return): return None
        ret=_Substitute(local_subs).visit(copy.deepcopy(body[-1].value))
        if not (isinstance(ret,ast.Call) and isinstance(ret.func,ast.Name) and ret.func.id=="sum" and len(ret.args)==1 and not ret.keywords): return None
        arg=self._inline_vector_subscripts(ret.args[0])
        # sum(EXPR for x in range(...) if ...)
        gen=self._unwrap_generator(arg)
        if gen is not None:
            if len(gen.generators)!=1: return None
            c=gen.generators[0]
            if not (isinstance(c.target,ast.Name) and isinstance(c.iter,ast.Call) and isinstance(c.iter.func,ast.Name) and c.iter.func.id=="range"):
                return None
            return ReductionSpec(ast.Constant(0.0),"_acc",c.target.id,tuple(copy.deepcopy(c.iter.args)),copy.deepcopy(gen.elt),tuple(copy.deepcopy(c.ifs)))

        # Proven aligned vector product: sum(list(EXPR for x in range(...)) * VECTOR[:stop])
        if isinstance(arg,ast.BinOp) and isinstance(arg.op,ast.Mult):
            for gside,vside in ((arg.left,arg.right),(arg.right,arg.left)):
                gen=self._unwrap_generator(gside)
                if gen is None or len(gen.generators)!=1: continue
                c=gen.generators[0]
                if not (isinstance(c.target,ast.Name) and isinstance(c.iter,ast.Call) and isinstance(c.iter.func,ast.Name) and c.iter.func.id=="range"):
                    continue
                if not (isinstance(vside,ast.Subscript) and isinstance(vside.slice,ast.Slice)):
                    continue
                sl=vside.slice
                # Only transform when the slice is exactly the same 0:stop:1 domain as range(stop).
                if len(c.iter.args)!=1 or sl.lower is not None or sl.step is not None or sl.upper is None:
                    continue
                if ast.dump(c.iter.args[0],include_attributes=False)!=ast.dump(sl.upper,include_attributes=False):
                    continue
                vector_value=vside.value
                if not (isinstance(vector_value,ast.Call) and isinstance(vector_value.func,ast.Name) and not vector_value.args and vector_value.func.id in self.vector_elements):
                    continue
                var,_,elt=self.vector_elements[vector_value.func.id]
                elem=_Substitute({var:ast.Name(c.target.id,ast.Load())}).visit(copy.deepcopy(elt))
                elem=self._inline_vector_subscripts(elem)
                bodyexpr=ast.BinOp(copy.deepcopy(gen.elt),ast.Mult(),elem)
                return ReductionSpec(ast.Constant(0.0),"_acc",c.target.id,tuple(copy.deepcopy(c.iter.args)),ast.fix_missing_locations(bodyexpr),tuple(copy.deepcopy(c.ifs)))
        return None

    def _expr_lowerer(self, operator_uid: str | None = None):
        frontend=self
        registry=self.registry
        point_row=self.model_point_row
        point_key_input=[None]
        class Lower(ast.NodeTransformer):
            def __init__(self):
                super().__init__()
                self._single_assign_names: set[str] = set()
                self._function_parameters: tuple[str, ...] = ()
                self._space_alias_roots: dict[str, str] = {}
                self._space_bindings: dict[str, tuple[str, tuple[ast.AST, ...]]] = {}
                self._numeric_bindings: dict[str, ast.AST] = {}
                self._space_assignment_names: set[str] = set()

            @staticmethod
            def _subscript_elements(node: ast.Subscript) -> tuple[ast.AST, ...]:
                return tuple(node.slice.elts) if isinstance(node.slice, ast.Tuple) else (node.slice,)

            def prepare_function(self, fn: ast.FunctionDef) -> None:
                """Pre-prove nonescaping local binders for parameterized modelx Spaces."""
                self._function_parameters = tuple(arg.arg for arg in fn.args.args)
                assign_count: dict[str, int] = {}
                assignments: dict[str, ast.Assign] = {}
                parents: dict[int, ast.AST] = {}

                def target_names(target: ast.AST) -> tuple[str, ...]:
                    if isinstance(target, ast.Name):
                        return (target.id,)
                    if isinstance(target, (ast.Tuple, ast.List)):
                        out: list[str] = []
                        for elt in target.elts:
                            out.extend(target_names(elt))
                        return tuple(out)
                    return ()

                for parent in ast.walk(fn):
                    for child in ast.iter_child_nodes(parent):
                        parents[id(child)] = parent
                    if isinstance(parent, ast.Assign):
                        for target in parent.targets:
                            for name in target_names(target):
                                assign_count[name] = assign_count.get(name, 0) + 1
                        if len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
                            assignments[parent.targets[0].id] = parent
                    elif isinstance(parent, ast.AnnAssign):
                        for name in target_names(parent.target):
                            assign_count[name] = assign_count.get(name, 0) + 1
                self._single_assign_names = {name for name, count in assign_count.items() if count == 1}

                for name in sorted(self._single_assign_names):
                    st = assignments.get(name)
                    if st is None or not (
                        isinstance(st.value, ast.Subscript)
                        and isinstance(st.value.value, ast.Name)
                    ):
                        continue
                    root_name = st.value.value.id
                    if root_name not in frontend.refs:
                        continue
                    try:
                        probe_key = frontend.run_keys[0] if frontend.run_keys else None
                        frontend._context_parameterized_space_for_key(root_name, probe_key)
                    except FrontendError:
                        continue

                    loads = [
                        node for node in ast.walk(fn)
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name
                    ]
                    if not loads:
                        continue
                    admitted = True
                    for load in loads:
                        attr = parents.get(id(load))
                        call = parents.get(id(attr)) if isinstance(attr, ast.Attribute) else None
                        if not (
                            isinstance(attr, ast.Attribute)
                            and attr.value is load
                            and isinstance(call, ast.Call)
                            and call.func is attr
                        ):
                            admitted = False
                            break
                    if admitted:
                        self._space_alias_roots[name] = root_name

            def _expand_numeric_expr(self, node: ast.AST, seen: frozenset[str] = frozenset()) -> ast.AST:
                outer = self
                class Expand(ast.NodeTransformer):
                    def visit_Name(self, name: ast.Name):
                        if (
                            isinstance(name.ctx, ast.Load)
                            and name.id in outer._numeric_bindings
                            and name.id not in seen
                        ):
                            return outer._expand_numeric_expr(
                                copy.deepcopy(outer._numeric_bindings[name.id]), seen | {name.id}
                            )
                        return copy.deepcopy(name)
                return ast.fix_missing_locations(Expand().visit(copy.deepcopy(node)))

            def cleanup_function(self, fn: ast.FunctionDef) -> ast.FunctionDef:
                """Erase compile-time-only Space binders only after all alias loads vanish."""
                loads = {
                    node.id for node in ast.walk(fn)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                }
                removable = self._space_assignment_names - loads
                if not removable:
                    return fn

                def clean(stmts: list[ast.stmt]) -> list[ast.stmt]:
                    out: list[ast.stmt] = []
                    for st in stmts:
                        if (
                            isinstance(st, ast.Assign) and len(st.targets) == 1
                            and isinstance(st.targets[0], ast.Name)
                            and st.targets[0].id in removable
                        ):
                            continue
                        if isinstance(st, ast.If):
                            st = copy.deepcopy(st)
                            st.body = clean(list(st.body))
                            st.orelse = clean(list(st.orelse))
                        out.append(st)
                    return out

                fn = copy.deepcopy(fn)
                fn.body = clean(list(fn.body))
                return ast.fix_missing_locations(fn)

            def visit_Name(self,n):
                if isinstance(n.ctx,ast.Load):
                    if n.id in frontend.space_params:
                        pos=frontend.space_params.index(n.id)
                        key=registry.point_coordinate(pos)
                        return ast.copy_location(
                            ast.Call(ast.Name("point_input",ast.Load()),[ast.Constant(key)],[]),n
                        )
                    if n.id in frontend.refs:
                        v=frontend.refs[n.id]
                        if isinstance(v,(int,float,np.integer,np.floating,bool,np.bool_)):
                            key=registry.scalar_ref(n.id,v)
                            return ast.copy_location(ast.Call(ast.Name("global_input",ast.Load()),[ast.Constant(key)],[]),n)
                return n

            def visit_Attribute(self, n):
                if (
                    isinstance(n.ctx, ast.Load)
                    and isinstance(n.value, ast.Name)
                    and n.value.id in frontend.refs
                ):
                    try:
                        key = registry.context_static_attribute(n.value.id, n.attr)
                    except FrontendError:
                        pass
                    else:
                        return ast.copy_location(
                            ast.Call(ast.Name("global_input", ast.Load()), [ast.Constant(key)], []), n
                        )
                return self.generic_visit(n)

            def visit_Assign(self, n):
                n = self.generic_visit(n)
                if len(n.targets) != 1:
                    return n
                target_node = n.targets[0]
                if isinstance(target_node, (ast.Tuple, ast.List)) and isinstance(n.value, (ast.Tuple, ast.List)):
                    def bind_pattern(target: ast.AST, value: ast.AST) -> None:
                        if isinstance(target, ast.Name):
                            if target.id in self._single_assign_names:
                                self._numeric_bindings[target.id] = copy.deepcopy(value)
                            return
                        if (
                            isinstance(target, (ast.Tuple, ast.List))
                            and isinstance(value, (ast.Tuple, ast.List))
                            and len(target.elts) == len(value.elts)
                        ):
                            for left, right in zip(target.elts, value.elts):
                                bind_pattern(left, right)
                    bind_pattern(target_node, n.value)
                    return n
                if not isinstance(target_node, ast.Name):
                    return n
                target = target_node.id
                if target not in self._single_assign_names:
                    return n
                if (
                    target in self._space_alias_roots
                    and isinstance(n.value, ast.Subscript)
                    and isinstance(n.value.value, ast.Name)
                    and n.value.value.id == self._space_alias_roots[target]
                ):
                    selectors = tuple(
                        self._expand_numeric_expr(part)
                        for part in self._subscript_elements(n.value)
                    )
                    self._space_bindings[target] = (n.value.value.id, selectors)
                    self._space_assignment_names.add(target)
                    return n
                self._numeric_bindings[target] = copy.deepcopy(n.value)
                return n

            def _external(self,call):
                if not (isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and not call.args and not call.keywords): return None
                rn=call.func.value.id
                obj=frontend.refs.get(rn)
                if obj is None or not hasattr(obj,"cells") or call.func.attr not in obj.cells: return None
                return rn,call.func.attr,obj.cells[call.func.attr]

            def _context_accessor(self, call):
                """Return a concrete modelx Cells-backed attribute call if provable.

                This deliberately accepts only direct ``root.member(...)`` calls where
                ``root`` is a frozen model reference.  Local rich-object methods remain
                a separate operator problem unless another typed lowering proves them.
                """
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and not call.keywords
                ):
                    return None
                rn = call.func.value.id
                if rn not in frontend.refs:
                    return None
                try:
                    frontend._context_cell_for_key(rn, call.func.attr, None)
                except FrontendError:
                    return None
                return rn, call.func.attr

            def _coordinate_name_and_domain(self):
                if operator_uid is None:
                    return None
                # The fast modelx-cython catalogue may have authoritative parameter
                # type/range evidence even when the older GraphModelCompiler trace
                # has no VariantTrace for this helper.  Reuse that evidence rather
                # than guessing from source names or representative extrema.
                mx_domains = getattr(frontend, "modelx_cython_parameter_domains", {})
                candidates = [
                    (name, bounds)
                    for (uid, name), bounds in mx_domains.items()
                    if uid == operator_uid
                    and name in self._function_parameters
                    and bounds[0] is not None and bounds[1] is not None
                ]
                if len(candidates) == 1:
                    name, (lo, hi, _source) = candidates[0]
                    return name, _IntDomain(int(lo), int(hi))
                tr = frontend.variant_trace_by_uid.get(operator_uid)
                if tr is None or tr.key.time_pos is None:
                    return None
                pos = int(tr.key.time_pos)
                if pos < 0 or pos >= len(tr.schema.parameters):
                    return None
                domain = (
                    frontend.specialized_coordinate_domains.get(operator_uid)
                    or frontend.coordinate_domains.get(operator_uid)
                )
                if domain is None:
                    return None
                return tr.schema.parameters[pos], domain

            def _selected_space_method_call(
                self, call: ast.Call
            ) -> tuple[str, tuple[ast.AST, ...], str, tuple[ast.AST, ...]] | None:
                """Describe one proven selected-Space Cells call without selecting it at runtime.

                V02386/V02387 originally admitted only the local-alias surface::

                    obj = root[selectors]
                    obj.member(args)

                V02389 admits the semantically identical direct receiver surface::

                    root[selectors].member(args)

                Both shapes return the same root/selectors/member/arguments descriptor
                and therefore share InputRegistry proof, memoization, materialization,
                rebinding and domain guards.  This is intentionally not a generic
                attribute/subscript interpreter.
                """
                if not (
                    isinstance(call.func, ast.Attribute)
                    and not call.keywords
                ):
                    return None

                def lowered(node: ast.AST) -> ast.AST:
                    # Direct receivers are seen before ``generic_visit`` would lower
                    # their selector/argument children (for example an owning Space
                    # parameter such as ``ScenID``).  Lower just those bounded numeric
                    # expressions, then expand single-assignment numeric aliases.
                    visited = self.visit(copy.deepcopy(node))
                    if visited is None or isinstance(visited, list):
                        return copy.deepcopy(node)
                    return self._expand_numeric_expr(visited)

                receiver = call.func.value
                if isinstance(receiver, ast.Name) and receiver.id in self._space_bindings:
                    root_name, selectors = self._space_bindings[receiver.id]
                    return (
                        root_name,
                        tuple(lowered(selector) for selector in selectors),
                        call.func.attr,
                        tuple(lowered(arg) for arg in call.args),
                    )

                if (
                    isinstance(receiver, ast.Subscript)
                    and isinstance(receiver.value, ast.Name)
                    and receiver.value.id in frontend.refs
                ):
                    root_name = receiver.value.id
                    # Structural precheck only.  Full selector arity, ItemSpace and
                    # concrete Cells-member proof remains owned by InputRegistry.
                    try:
                        probe_key = frontend.run_keys[0] if frontend.run_keys else None
                        frontend._context_parameterized_space_for_key(root_name, probe_key)
                    except FrontendError:
                        return None
                    selectors = tuple(
                        lowered(part) for part in self._subscript_elements(receiver)
                    )
                    return (
                        root_name,
                        selectors,
                        call.func.attr,
                        tuple(lowered(arg) for arg in call.args),
                    )
                return None

            def _bounded_context_argument(
                self, node: ast.AST, coordinate_name: str | None = None
            ) -> ast.AST:
                """Lower one context-accessor argument into the bounded numeric ABI subset.

                Ordinary nested input/context reads are lowered first.  If that is
                still not representable, V02390 may hygienically substitute exactly
                one existing pure single-return source/variant helper and retry.
                The substituted result must itself reduce completely to the bounded
                ABI-expression subset; arbitrary helper execution never becomes part
                of preparation or generated runtime.
                """
                def lowered(expr: ast.AST) -> ast.AST:
                    visited = self.visit(copy.deepcopy(expr))
                    if visited is None or isinstance(visited, list):
                        return copy.deepcopy(expr)
                    return self._expand_numeric_expr(visited)

                direct = lowered(node)
                if registry.bound_numeric_expr_supported(direct, coordinate_name):
                    return direct

                if isinstance(direct, ast.Call) and isinstance(direct.func, ast.Name):
                    expanded = frontend._expand_pure_single_return_call(direct)
                    if expanded is not None:
                        expr, _helper_token = expanded
                        candidate = lowered(expr)
                        if registry.bound_numeric_expr_supported(candidate, coordinate_name):
                            return candidate
                return direct

            def visit_Call(self,n):
                # Finite preparation-owned mapping lookup.  The source mapping is
                # a zero-argument local Cells value; only the numeric result table
                # crosses into the generated backend.  This deliberately recognizes
                # ``.get`` only with an explicit default and a source-proven bounded
                # coordinate owner, rather than implementing a runtime mapping VM.
                if (
                    isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get"
                    and isinstance(n.func.value, ast.Call)
                    and isinstance(n.func.value.func, ast.Name)
                    and not n.func.value.args and not n.func.value.keywords
                    and len(n.args) == 2 and not n.keywords
                ):
                    called = n.func.value.func.id
                    tr = frontend.variant_trace_by_uid.get(called)
                    source_name = None
                    if tr is not None and not tr.schema.parameters:
                        source_name = tr.schema.name
                    if source_name is None:
                        source_name = getattr(
                            frontend, "formula_uid_to_source_name", {}
                        ).get(called)
                    if source_name is None and called in frontend.base_funcs:
                        source_name = called
                    cell = None if source_name is None else frontend.space.cells.get(source_name)
                    coord = self._coordinate_name_and_domain()
                    ok_default, default_value = frontend._proven_static_expr(n.args[1])
                    if (
                        cell is not None
                        and not tuple(cell.parameters or ())
                        and coord is not None and operator_uid is not None
                        and ok_default
                    ):
                        coordinate_name, domain = coord
                        key_expr = self._bounded_context_argument(
                            n.args[0], coordinate_name
                        )
                        try:
                            table_key, ordinal_key = registry.local_cell_mapping_coordinate_get(
                                source_name,
                                key_expr,
                                default_value,
                                coordinate_name,
                                int(domain.lo),
                                int(domain.hi),
                                operator_uid,
                            )
                        except FrontendError:
                            pass
                        else:
                            column = _axis_index_expr(
                                ast.Name(coordinate_name, ast.Load()), int(domain.lo), 1
                            )
                            if operator_uid is not None:
                                frontend._normalized_operators[operator_uid] = NormalizedOperator(
                                    uid=operator_uid,
                                    kind="prepared_mapping_get",
                                    result_dtype=registry.specs[table_key].dtype,
                                    result_shape="scalar",
                                    static_inputs=(("input_key", table_key), ("source_cell", source_name)),
                                    dynamic_inputs=(ast.unparse(n.args[0]),),
                                    provenance=(
                                        "finite_prepared_mapping",
                                        "explicit_default_semantics",
                                        "complete_coordinate_domain",
                                    ),
                                    state_semantic="state_free",
                                    python_supported=True,
                                    cython_supported=True,
                                )
                            return ast.copy_location(
                                ast.Call(
                                    ast.Name("table_input", ast.Load()),
                                    [
                                        ast.Constant(table_key),
                                        ast.Call(
                                            ast.Name("point_input", ast.Load()),
                                            [ast.Constant(ordinal_key)], [],
                                        ),
                                        column,
                                    ],
                                    [],
                                ),
                                n,
                            )

                selected = self._selected_space_method_call(n)
                if selected is not None:
                    root_name, selectors, member_name, args = selected
                    try:
                        key = registry.parameterized_space_method_point(
                            root_name, selectors, member_name, args
                        )
                    except FrontendError:
                        pass
                    else:
                        return ast.copy_location(
                            ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(key)], []), n
                        )

                    # V02387: the same selected-ItemSpace/Cells proof may be
                    # materialized over exactly one source-proven coordinate axis.
                    # The point-only path above remains first so point scalars do not
                    # silently become two-dimensional tables.
                    coord = self._coordinate_name_and_domain()
                    if coord is not None and operator_uid is not None:
                        coordinate_name, domain = coord
                        if (
                            all(
                                registry.bound_numeric_expr_supported(arg, coordinate_name)
                                for arg in args
                            )
                            and any(
                                registry.bound_numeric_expr_uses_coordinate(arg, coordinate_name)
                                for arg in args
                            )
                        ):
                            try:
                                table_key, ordinal_key = (
                                    registry.parameterized_space_method_coordinate(
                                        root_name,
                                        selectors,
                                        member_name,
                                        args,
                                        coordinate_name,
                                        int(domain.lo),
                                        int(domain.hi),
                                        operator_uid,
                                        1,
                                    )
                                )
                            except FrontendError:
                                pass
                            else:
                                column = _axis_index_expr(
                                    ast.Name(coordinate_name, ast.Load()), int(domain.lo), 1
                                )
                                return ast.copy_location(
                                    ast.Call(
                                        ast.Name("table_input", ast.Load()),
                                        [
                                            ast.Constant(table_key),
                                            ast.Call(
                                                ast.Name("point_input", ast.Load()),
                                                [ast.Constant(ordinal_key)],
                                                [],
                                            ),
                                            column,
                                        ],
                                        [],
                                    ),
                                    n,
                                )

                ext=self._external(n)
                if ext:
                    rn,cn,cell=ext
                    key=registry.external_scalar(rn,cn,cell)
                    intrinsic = "point_input" if registry.specs[key].scope == "point" else "global_input"
                    return ast.copy_location(
                        ast.Call(ast.Name(intrinsic,ast.Load()),[ast.Constant(key)],[]),n
                    )

                ctx = self._context_accessor(n)
                if ctx:
                    rn, cn = ctx
                    coord = self._coordinate_name_and_domain()
                    if coord is not None and operator_uid is not None:
                        coordinate_name, domain = coord
                        coordinate_args = tuple(
                            self._bounded_context_argument(arg, coordinate_name) for arg in n.args
                        )
                        if (
                            coordinate_args
                            and all(
                                registry.bound_numeric_expr_supported(arg, coordinate_name)
                                for arg in coordinate_args
                            )
                            and any(
                                registry.bound_numeric_expr_uses_coordinate(arg, coordinate_name)
                                for arg in coordinate_args
                            )
                        ):
                            try:
                                table_key, ordinal_key = registry.context_coordinate_accessor(
                                    rn, cn, coordinate_args, coordinate_name,
                                    int(domain.lo), int(domain.hi), operator_uid, 1,
                                )
                            except FrontendError:
                                pass
                            else:
                                column = _axis_index_expr(
                                    ast.Name(coordinate_name, ast.Load()), int(domain.lo), 1
                                )
                                return ast.copy_location(
                                    ast.Call(
                                        ast.Name("table_input", ast.Load()),
                                        [
                                            ast.Constant(table_key),
                                            ast.Call(
                                                ast.Name("point_input", ast.Load()),
                                                [ast.Constant(ordinal_key)], [],
                                            ),
                                            column,
                                        ],
                                        [],
                                    ),
                                    n,
                                )

                    point_args = tuple(
                        self._bounded_context_argument(arg, None) for arg in n.args
                    )
                    if all(registry.bound_numeric_expr_supported(arg) for arg in point_args):
                        try:
                            key = registry.context_scalar_accessor(rn, cn, point_args)
                        except FrontendError:
                            pass
                        else:
                            intrinsic = (
                                "global_input"
                                if registry.specs[key].scope == "global"
                                else "point_input"
                            )
                            return ast.copy_location(
                                ast.Call(ast.Name(intrinsic, ast.Load()), [ast.Constant(key)], []),
                                n,
                            )
                return self.generic_visit(n)

            def _point_enum_selector(self, node: ast.AST):
                """Return (labels, numeric-code-expr) for a direct point-row string Cell.

                This is deliberately the same structural proof used by normalized
                external-table dispatch: the source helper must consist solely of
                ``return model_point()["field"]`` and the exact frozen point-row
                domain must contain a small finite set of strings.
                """
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and not node.args and not node.keywords and frontend.model_point_row is not None
                ):
                    return None
                call_name = node.func.id
                tr = frontend.variant_trace_by_uid.get(call_name)
                source_name = tr.schema.name if tr is not None else (
                    call_name if call_name in frontend.base_funcs else None
                )
                if source_name is None:
                    return None
                original = frontend.base_funcs.get(source_name)
                if original is None:
                    return None
                direct = _strip_docstring(copy.deepcopy(original))
                if not (
                    isinstance(direct, ast.FunctionDef)
                    and not direct.args.posonlyargs and not direct.args.args
                    and not direct.args.kwonlyargs and direct.args.vararg is None
                    and direct.args.kwarg is None and len(direct.body) == 1
                    and isinstance(direct.body[0], ast.Return)
                    and isinstance(direct.body[0].value, ast.Subscript)
                    and isinstance(direct.body[0].value.value, ast.Call)
                    and isinstance(direct.body[0].value.value.func, ast.Name)
                    and direct.body[0].value.value.func.id == frontend.model_point_row.cell_name
                    and not direct.body[0].value.value.args
                    and not direct.body[0].value.value.keywords
                    and isinstance(direct.body[0].value.slice, ast.Constant)
                    and isinstance(direct.body[0].value.slice.value, str)
                ):
                    return None
                field = str(direct.body[0].value.slice.value)
                frame = frontend.model_point_row.provider()
                if not isinstance(frame, pd.DataFrame) or field not in frame.columns:
                    return None
                values = frame[field].tolist()
                if not values or not all(
                    isinstance(value, str) or bool(pd.isna(value)) for value in values
                ):
                    return None
                labels = tuple(dict.fromkeys(value for value in values if isinstance(value, str)))
                if not labels or len(labels) > 16:
                    return None
                has_other = any(not isinstance(value, str) for value in values)
                key = registry.point_field(
                    frontend.model_point_row, field, enum_labels=labels,
                    enum_allow_missing=has_other,
                )
                try:
                    point_code = point_category_exact_axis_call(
                        registry, key, size=len(labels), require_validity=has_other,
                    )
                except CategoryValidityError as exc:
                    raise FrontendError(str(exc)) from exc
                return labels, point_code

            def _exact_dataframe_axis(self, expr: ast.AST, labels: tuple[Any, ...], level: int):
                ok, value = frontend._proven_static_expr(expr)
                if ok:
                    if value not in labels:
                        raise FrontendError(
                            f"DataFrame exact lookup axis {level} has no label {value!r}"
                        )
                    return None, ast.Constant(0), {level: value}

                enum = self._point_enum_selector(expr)
                if enum is not None:
                    enum_labels, code = enum
                    if any(label not in labels for label in enum_labels):
                        raise FrontendError(
                            f"DataFrame exact lookup categorical axis {level} exceeds frozen label domain"
                        )
                    axis = NormalizedTableAxis(
                        level=level, index_name=None if level == 0 else "__columns__",
                        selector_kind="finite_categorical", labels=tuple(enum_labels),
                    )
                    return axis, code, {}

                try:
                    numeric, start, step = _safe_numeric_axis(labels)
                except FrontendError as exc:
                    raise FrontendError(
                        f"DataFrame exact lookup dynamic axis {level} is neither a finite "
                        f"categorical point selector nor a regular numeric axis: {exc}"
                    ) from exc
                numeric_to_source = sorted(zip(numeric, labels), key=lambda item: item[0])
                source_labels = tuple(label for _numeric, label in numeric_to_source)
                visited = self.visit(copy.deepcopy(expr))
                axis = NormalizedTableAxis(
                    level=level, index_name=None if level == 0 else "__columns__",
                    selector_kind="numeric_exact", labels=source_labels,
                    axis_start=int(start), axis_step=int(step),
                )
                code = ast.Call(
                    ast.Name("__exact_axis_code__", ast.Load()),
                    [visited, ast.Constant(int(start)), ast.Constant(int(step)), ast.Constant(len(source_labels))], []
                )
                return axis, code, {}

            def visit_Subscript(self,n):
                # A zero-argument local Cell returning a dense numeric two-level
                # Series is an explicit state-free indexed input.  The object never
                # crosses the numeric backend boundary.
                local_object_cell_name = None
                if (
                    isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name)
                    and not n.value.args and not n.value.keywords
                ):
                    called = n.value.func.id
                    traced = frontend.variant_trace_by_uid.get(called)
                    if traced is not None and not traced.schema.parameters:
                        local_object_cell_name = traced.schema.name
                    else:
                        # The fast modelx-cython catalog UID-binds same-Space calls
                        # before this mature provider lowerer runs.  Object-returning
                        # helper Cells are intentionally absent from scalar trace
                        # variants, so recover only their lexical source identity
                        # through the explicit catalog bridge.  The registry below
                        # remains the authority for type/rank/domain admission.
                        source_name = getattr(
                            frontend, "formula_uid_to_source_name", {}
                        ).get(called)
                        if source_name is None and called in frontend.base_funcs:
                            source_name = called
                        cell = (
                            None if source_name is None
                            else frontend.space.cells.get(source_name)
                        )
                        if cell is not None and not tuple(cell.parameters or ()):
                            local_object_cell_name = source_name
                if (
                    local_object_cell_name is not None
                    and isinstance(n.slice, ast.Tuple) and len(n.slice.elts) == 2
                ):
                    cell_name = local_object_cell_name
                    cell = frontend.space.cells.get(cell_name)
                    if cell is not None:
                        try:
                            key = registry.local_cell_multiindex_series(cell_name, cell)
                        except FrontendError:
                            pass
                        else:
                            r0, rs, c0, cs = frontend.multiindex_axes[key]
                            row_expr = _axis_index_expr(self.visit(copy.deepcopy(n.slice.elts[0])), r0, rs)
                            col_expr = _axis_index_expr(self.visit(copy.deepcopy(n.slice.elts[1])), c0, cs)
                            if operator_uid is not None:
                                frontend._normalized_operators[operator_uid] = NormalizedOperator(
                                    uid=operator_uid,
                                    kind="indexed_input",
                                    result_dtype="float64",
                                    result_shape="scalar",
                                    static_inputs=(("input_key", key), ("source_cell", cell_name)),
                                    dynamic_inputs=(ast.unparse(n.slice.elts[0]), ast.unparse(n.slice.elts[1])),
                                    provenance=(
                                        "local_zero_arg_object_cell",
                                        "dense_numeric_two_level_multiindex",
                                        "runtime_domain_guard",
                                    ),
                                    state_semantic="state_free",
                                    python_supported=True,
                                    cython_supported=True,
                                )
                            return ast.copy_location(
                                ast.Call(ast.Name("table_input",ast.Load()),[ast.Constant(key),row_expr,col_expr],[]),n
                            )

                # DataFrame point-row Cells followed by a column selection.  The
                # call may already have been rewritten to its canonical variant UID;
                # match by trace schema as well as source Cell spelling.
                point_row_call = False
                if point_row and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and not n.value.args and not n.value.keywords:
                    called = n.value.func.id
                    if called == point_row.cell_name:
                        point_row_call = True
                    else:
                        trace = frontend.variant_trace_by_uid.get(called)
                        point_row_call = bool(trace is not None and trace.schema.name == point_row.cell_name)
                if point_row_call:
                    ok, field = frontend._proven_static_expr(n.slice)
                    if ok and isinstance(field, str):
                        key=registry.point_field(point_row,field)
                        return ast.copy_location(ast.Call(ast.Name("point_input",ast.Load()),[ast.Constant(key)],[]),n)
                    symbolic = _finite_symbolic_point_field_plan(frontend, operator_uid, n.slice)
                    if symbolic is not None:
                        try:
                            table_key, ordinal_key = registry.finite_symbolic_point_field_dispatch(
                                point_row,
                                symbolic.selector_values,
                                symbolic.field_names,
                                domain_source=symbolic.domain_source,
                            )
                        except FrontendError:
                            pass
                        else:
                            code_expr = self.visit(copy.deepcopy(symbolic.code_expr))
                            if operator_uid is not None:
                                frontend._normalized_operators[operator_uid] = NormalizedOperator(
                                    uid=operator_uid,
                                    kind="finite_symbolic_point_field_dispatch",
                                    result_dtype=registry.specs[table_key].dtype,
                                    result_shape="scalar",
                                    static_inputs=(
                                        ("input_key", table_key),
                                        ("selector_values", symbolic.selector_values),
                                        ("field_names", symbolic.field_names),
                                        ("domain_source", symbolic.domain_source),
                                    ),
                                    dynamic_inputs=(symbolic.selector_name,),
                                    provenance=(
                                        "finite_symbolic_source_expression",
                                        "complete_RunDomain_field_type_proof",
                                        "preparation_owned_field_names",
                                        "native_integer_selection",
                                    ),
                                    state_semantic="state_free",
                                    python_supported=True,
                                    cython_supported=True,
                                )
                            return ast.copy_location(
                                ast.Call(
                                    ast.Name("table_input", ast.Load()),
                                    [
                                        ast.Constant(table_key),
                                        ast.Call(
                                            ast.Name("point_input", ast.Load()),
                                            [ast.Constant(ordinal_key)], [],
                                        ),
                                        code_expr,
                                    ],
                                    [],
                                ),
                                n,
                            )
                    # Finite conditional selection between two statically named
                    # point fields remains a scalar provider operation. Bind each
                    # field independently and keep only the numeric condition at
                    # runtime instead of materializing the point-row object.
                    if isinstance(n.slice, ast.IfExp):
                        ok_body, body_field = frontend._proven_static_expr(n.slice.body)
                        ok_else, else_field = frontend._proven_static_expr(n.slice.orelse)
                        if (
                            ok_body and ok_else
                            and isinstance(body_field, str)
                            and isinstance(else_field, str)
                        ):
                            body_key = registry.point_field(point_row, body_field)
                            else_key = registry.point_field(point_row, else_field)
                            return ast.copy_location(
                                ast.IfExp(
                                    test=self.visit(copy.deepcopy(n.slice.test)),
                                    body=ast.Call(
                                        ast.Name("point_input", ast.Load()),
                                        [ast.Constant(body_key)], [],
                                    ),
                                    orelse=ast.Call(
                                        ast.Name("point_input", ast.Load()),
                                        [ast.Constant(else_key)], [],
                                    ),
                                ),
                                n,
                            )

                if isinstance(n.value,ast.Call):
                    ext=self._external(n.value)
                    if ext:
                        rn,cn,cell=ext
                        ok_item, item_key = frontend._proven_static_expr(n.slice)
                        if ok_item:
                            try:
                                key = registry.external_static_item(rn, cn, cell, item_key)
                            except FrontendError:
                                pass
                            else:
                                if operator_uid is not None:
                                    frontend._normalized_operators[operator_uid] = NormalizedOperator(
                                        uid=operator_uid,
                                        kind="prepared_static_item",
                                        result_dtype=registry.specs[key].dtype,
                                        result_shape="scalar",
                                        static_inputs=(("input_key", key), ("item_key", item_key)),
                                        dynamic_inputs=(),
                                        provenance=(
                                            "mixed_container_static_item",
                                            "selected_primitive_only",
                                            "container_domain_guard",
                                        ),
                                        state_semantic="state_free",
                                        python_supported=True,
                                        cython_supported=True,
                                    )
                                return ast.copy_location(
                                    ast.Call(
                                        ast.Name("global_input", ast.Load()),
                                        [ast.Constant(key)], [],
                                    ),
                                    n,
                                )
                        if isinstance(n.slice,ast.Name) and n.slice.id in frontend.space_params:
                            pos = frontend.space_params.index(n.slice.id)
                            key=registry.external_array(rn,cn,cell,pos)
                            return ast.copy_location(ast.Call(ast.Name("point_input",ast.Load()),[ast.Constant(key)],[]),n)
                        key=registry.external_array(rn,cn,cell,None)
                        if isinstance(n.slice,ast.Tuple) and len(n.slice.elts)==2:
                            return ast.copy_location(ast.Call(ast.Name("table_input",ast.Load()),[ast.Constant(key),self.visit(n.slice.elts[0]),self.visit(n.slice.elts[1])],[]),n)
                        return ast.copy_location(ast.Call(ast.Name("array_input",ast.Load()),[ast.Constant(key),self.visit(n.slice)],[]),n)

                    # General model-bound Cells accessors that escaped the historical
                    # zero-argument ``obj.cells`` path.  The bounded first shape is a
                    # numeric vector selected by one exact ItemSpace coordinate; this
                    # covers Cells aliases and finite/static call arguments without
                    # introducing a runtime context proxy.
                    ctx = self._context_accessor(n.value)
                    if (
                        ctx is not None
                        and isinstance(n.slice, ast.Name)
                        and n.slice.id in frontend.space_params
                    ):
                        rn, cn = ctx
                        pos = frontend.space_params.index(n.slice.id)
                        try:
                            key = registry.context_point_accessor(
                                rn, cn, n.value.args, pos
                            )
                        except FrontendError:
                            pass
                        else:
                            return ast.copy_location(
                                ast.Call(
                                    ast.Name("point_input", ast.Load()),
                                    [ast.Constant(key)], [],
                                ),
                                n,
                            )

                # Exact DataFrame column-then-row: ref[col][row].  Handle the
                # outer shape before recursively visiting the inner subscript; the
                # latter is intentionally not a complete scalar lookup by itself.
                if isinstance(n.value,ast.Subscript) and isinstance(n.value.value,ast.Name):
                    rn=n.value.value.id; v=frontend.refs.get(rn)
                    if isinstance(v,pd.DataFrame):
                        col=n.value.slice
                        if isinstance(col,ast.Call) and isinstance(col.func,ast.Name) and col.func.id=="str" and len(col.args)==1 and not col.keywords:
                            col=col.args[0]
                        # Preserve the established affine lowering when both source
                        # axes are numeric (including numeric-string labels).  The
                        # richer exact N-D path is reserved for genuinely labelled
                        # categorical axes, avoiding needless source-shape churn in
                        # existing native controls.
                        try:
                            key,rm,cm=registry.global_dataframe(rn,v)
                        except FrontendError:
                            row_labels=tuple(v.index.tolist()); col_labels=tuple(v.columns.tolist())
                            row_axis,row_code,row_fixed=self._exact_dataframe_axis(n.slice,row_labels,0)
                            col_axis,col_code,col_fixed=self._exact_dataframe_axis(col,col_labels,1)
                            axes=tuple(x for x in (row_axis,col_axis) if x is not None)
                            fixed={**row_fixed,**col_fixed}
                            key=registry.global_dataframe_exact_nd(
                                rn,v,fixed_axes=fixed,dynamic_axes=axes,
                            )
                            return ast.copy_location(
                                ast.Call(ast.Name("table_input",ast.Load()),[ast.Constant(key),row_code,col_code],[]),n
                            )
                        else:
                            return ast.copy_location(
                                ast.Call(
                                    ast.Name("table_input",ast.Load()),
                                    [ast.Constant(key),rm(self.visit(n.slice)),cm(self.visit(col))],[],
                                ),n
                            )

                # Direct Series/DataFrame reference indexing.
                if isinstance(n.value,ast.Name) and n.value.id in frontend.refs:
                    rn=n.value.id; v=frontend.refs[rn]
                    if isinstance(v,pd.Series):
                        if isinstance(v.index,pd.MultiIndex):
                            if not (isinstance(n.slice,ast.Tuple) and len(n.slice.elts)==2):
                                raise FrontendError(f"MultiIndex Reference {rn} requires a two-element key")
                            key,_=registry.global_multiindex_series(rn,v)
                            r0,rs,c0,cs=frontend.multiindex_axes[key]
                            r=_axis_index_expr(self.visit(n.slice.elts[0]),r0,rs); c=_axis_index_expr(self.visit(n.slice.elts[1]),c0,cs)
                            return ast.copy_location(ast.Call(ast.Name("table_input",ast.Load()),[ast.Constant(key),r,c],[]),n)
                        key,mapidx=registry.global_series(rn,v)
                        return ast.copy_location(ast.Call(ast.Name("array_input",ast.Load()),[ast.Constant(key),mapidx(self.visit(n.slice))],[]),n)
                    if isinstance(v,pd.DataFrame):
                        if not (isinstance(n.slice,ast.Tuple) and len(n.slice.elts)==2):
                            raise FrontendError(f"DataFrame Reference {rn} direct indexing requires two indices")
                        key,rm,cm=registry.global_dataframe(rn,v)
                        return ast.copy_location(ast.Call(ast.Name("table_input",ast.Load()),[ast.Constant(key),rm(self.visit(n.slice.elts[0])),cm(self.visit(n.slice.elts[1]))],[]),n)

                return self.generic_visit(n)
        return Lower()

    def _build_canonical(self) -> CanonicalModel:
        canonical: dict[str,CanonicalVariant]={}
        diagnostics=list(self.trace.diagnostics)
        diagnostics.extend(d for d in self.ir.diagnostics if d not in diagnostics)
        diagnostics.extend(d for d in self.coordinate_domain_notes if d not in diagnostics)
        # Explicit regional fallback boundaries are identities supplied by the
        # caller, not semantic names.  They are marshaled as one scalar per model
        # point and may later be overridden by a callback/orchestrator.
        for name in sorted(self.fallback_cells):
            traces=[tr for tr in self.selected_trace_variants.values() if tr.schema.name == name]
            if len(traces) != 1:
                raise FrontendError(
                    f"regional fallback Cell {name!r} must resolve to one observed zero-argument variant"
                )
            self.fallback_input_by_name[name] = self.registry.regional_fallback_point(name, traces[0].dtype)

        # v0.6 executable roles are derived from trace-coordinate evidence and the
        # statically normalized formula itself.  Legacy planner kind/stage values
        # remain only as compatibility/audit metadata.
        coordinate_uids = {
            tr.uid for tr in self.variant_trace_by_uid.values()
            if tr.key.time_pos is not None
        }
        next_guard_code = 1

        def lower_into_canonical(
            uid: str, tr, fn: ast.FunctionDef, *, forced_role: str | None = None,
            time_param_source: str | None = None,
        ) -> None:
            nonlocal next_guard_code
            lower = self._expr_lowerer(uid)
            fn = self._normalize_external_tables(fn, operator_uid=uid)
            lower.prepare_function(fn)
            guards: tuple[GuardSpec, ...] = ()

            if forced_role is not None:
                role = forced_role
            elif tr.schema.name in self.vector_elements and tr.dtype == "object":
                role = "vector"
            elif tr.key.time_pos is not None:
                role = "coordinate"
            else:
                role = "scalar"

            reduction = None
            if role == "scalar":
                reduction = self._normalize_reduction(fn)
                if reduction is not None:
                    role = "reduction"

            if tr.dtype == "object" and role not in ("vector",):
                if forced_role is not None:
                    raise FrontendError(
                        f"fixed/native scalar specialization of object Cell {tr.schema.name!r} is unsupported"
                    )
                # Ordinary object helper Cells are consumed by input/table lowering
                # and do not enter the numeric kernel, matching the dev15 path.
                return

            if role == "reduction":
                reduction = ReductionSpec(
                    lower.visit(copy.deepcopy(reduction.init)), reduction.target, reduction.loop_var,
                    tuple(lower.visit(copy.deepcopy(a)) for a in reduction.range_args),
                    lower.visit(copy.deepcopy(reduction.body_expr)),
                    tuple(lower.visit(copy.deepcopy(f)) for f in reduction.filters),
                )
                ast.fix_missing_locations(reduction.body_expr)
            elif role != "vector":
                fn.body = [lower.visit(copy.deepcopy(x)) for x in fn.body]
                fn = ast.fix_missing_locations(fn)
                fn = lower.cleanup_function(fn)
                fn = self._normalize_scalar_intrinsics(fn)
                fn = self._normalize_simple_local_syntax(fn)
                try:
                    fn, guards, next_guard_code = normalize_formula_guards(
                        fn, uid=uid, next_code=next_guard_code
                    )
                    validate_formula_control_flow(fn)
                except (SemanticSafetyError, ProgramSemanticError) as exc:
                    raise FrontendError(str(exc)) from exc

            canonical[uid] = CanonicalVariant(
                uid=uid, source_name=tr.schema.name, source_fullname=tr.schema.fullname,
                dtype=tr.dtype if tr.dtype in ("int64", "bool") else "float64",
                role=role, function=fn, reduction=reduction,
                time_param_source=time_param_source,
                # Output arguments are invocation geometry, not helper-variant
                # axes.  _specialize_function already bound them into the root
                # formula, so the canonical root is a zero-argument scalar/reduction.
                aux_values=(() if uid == output_uid else tr.key.aux_values),
                guards=guards,
            )

        # Source specialization is demand-driven from the scalar output.  Earlier
        # releases specialized every observed trace variant and pruned reachability
        # afterwards.  Once parameterized pure helpers are erased by inlining, that
        # eagerly constructs large populations of helper variants that no generated
        # program can call (RILA observes many Black-Scholes argument combinations).
        # Specializing only dependencies discovered from already-specialized source
        # ASTs is semantically equivalent to the old pre-reachability filter, while
        # preserving the important phase-order rule: static topology proofs run before
        # an unsupported dead helper body is ever handed to table/backend lowering.
        output_matches = [
            uid for uid, tr in self.variant_trace_by_uid.items()
            if tr.schema.name == self.output
            and tr.key.time_pos is None
            and tuple(tr.key.aux_values)==tuple(self.output_invocation.argument_values)
            and tr.schema.name in self.base_funcs
            and tr.schema.name not in self.fallback_cells
            and tr.schema.name in self.compiled_source_names
        ]
        if len(output_matches) != 1:
            raise FrontendError(
                f"could not uniquely identify scalar output Cell {self.output!r} before lowering"
            )
        output_uid = output_matches[0]

        specialized: dict[str, tuple[Any, ast.FunctionDef, str | None, str | None]] = {}
        pending: list[str] = [output_uid]
        queued: set[str] = {output_uid}

        def enqueue_dependencies(fn: ast.FunctionDef) -> None:
            # _specialize_function has already canonicalized ordinary Cells calls to
            # observed variant UIDs and has registered any synthetic fixed-coordinate
            # calls.  Pure scalar helpers disappeared into local statements, so their
            # standalone observed variants are deliberately absent from this queue.
            for child in ast.walk(fn):
                if not (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)):
                    continue
                dep = child.func.id
                if dep in self.variant_trace_by_uid and dep not in queued:
                    tr = self.variant_trace_by_uid[dep]
                    if (
                        tr.schema.name in self.base_funcs
                        and tr.schema.name not in self.fallback_cells
                        and tr.schema.name in self.compiled_source_names
                    ):
                        queued.add(dep)
                        pending.append(dep)
                elif dep in self._fixed_coordinate_requests and dep not in queued:
                    queued.add(dep)
                    pending.append(dep)

        while pending:
            uid = pending.pop()
            if uid in specialized:
                continue
            if uid in self._fixed_coordinate_requests:
                base_uid, coordinate = self._fixed_coordinate_requests[uid]
                tr = self.variant_trace_by_uid[base_uid]
                fn = self._specialize_function(
                    base_uid, fixed_time=int(coordinate), synthetic_uid=uid
                )
                if fn is None:
                    raise FrontendError(
                        f"could not specialize fixed-coordinate variant {tr.schema.name}({coordinate})"
                    )
                specialized[uid] = (tr, fn, "scalar", None)
            else:
                tr = self.variant_trace_by_uid[uid]
                fn = self._specialize_function(uid)
                if fn is None:
                    continue
                specialized[uid] = (
                    tr, fn, None,
                    tr.schema.parameters[tr.key.time_pos] if tr.key.time_pos is not None else None,
                )
            enqueue_dependencies(fn)

            # A reachable specialization can discover fixed-coordinate calls while
            # processing nested expressions.  Enqueue any such calls even if a later
            # local rewrite happened to remove their syntactic call site; retaining an
            # extra proven scalar specialization is conservative and bounded.
            for synthetic_uid in self._fixed_coordinate_requests:
                if synthetic_uid not in queued:
                    queued.add(synthetic_uid)
                    pending.append(synthetic_uid)

        # Table/domain lowering happens only after the complete demanded source
        # graph has been specialized.  Derive one second, more precise coordinate
        # fixed point here so lookup preconditions see caller-refined domains rather
        # than only the physical root range.  This proof is source/control-flow
        # based and never consumes representative observed coordinate values.
        self.specialized_coordinate_domains = self._derive_specialized_coordinate_domains(
            specialized, output_uid
        )
        for uid, dom in sorted(self.specialized_coordinate_domains.items()):
            tr = self.variant_trace_by_uid.get(uid)
            if tr is None:
                continue
            note = (
                f"source-proven specialized call domain {tr.schema.fullname}"
                f"{tr.key.aux_values!r}: [{dom.lo}, {dom.hi}]"
            )
            if note not in diagnostics:
                diagnostics.append(note)

        # Keep canonical/input registration order stable with the historical eager
        # frontend.  Demand-driven specialization changes discovery order (DFS from
        # the output), but generated ABI names such as p_1/a1_5 should not churn just
        # because unreachable helpers stopped being compiled.  Normal observed
        # variants therefore lower in trace order; synthetic fixed-coordinate
        # variants follow their request order, matching the old two-phase loop.
        lower_order = [uid for uid in self.variant_trace_by_uid if uid in specialized]
        lower_order.extend(
            uid for uid in self._fixed_coordinate_requests
            if uid in specialized and uid not in self.variant_trace_by_uid
        )
        for uid in lower_order:
            tr, fn, forced_role, time_param_source = specialized[uid]
            lower_into_canonical(
                uid, tr, fn, forced_role=forced_role, time_param_source=time_param_source
            )

        if (
            output_uid not in canonical
            or canonical[output_uid].role not in ("scalar", "reduction")
            or canonical[output_uid].aux_values
        ):
            raise FrontendError(f"output Cell {self.output!r} did not lower as a scalar/reduction")

        # Lowering itself can erase additional dependencies (for example normalized
        # table/helper syntax), so retain the existing post-lowering reachability pass
        # as a second conservative cleanup.
        known=set(canonical)
        deps: dict[str,set[str]] = {uid:set() for uid in canonical}
        for uid,cv in canonical.items():
            nodes=[]
            if cv.function is not None:
                nodes.extend(cv.function.body)
            if cv.reduction is not None:
                nodes.extend([cv.reduction.init, cv.reduction.body_expr, *cv.reduction.filters, *cv.reduction.range_args])
            for node in nodes:
                for child in ast.walk(node):
                    if isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and child.func.id in known:
                        deps[uid].add(child.func.id)
        reachable=set()
        stack=[output_uid]
        while stack:
            uid=stack.pop()
            if uid in reachable:
                continue
            reachable.add(uid)
            stack.extend(deps.get(uid,()))
        canonical={uid:cv for uid,cv in canonical.items() if uid in reachable}

        # Static formula lowering has inspected every included numeric formula.
        # Strengthen observed loop evidence before freezing the canonical proof
        # boundary so the observational semantic bridge sees the same proof scope
        # as the legacy executable scheduler.
        if any(v.role == "coordinate" for v in canonical.values()):
            self.ir.authorize_loop_generalization(
                "exact coordinate-domain AST + fail-closed formula lowering validated by frontend"
            )

        coordinate_proof_domains = {
            uid: (int(dom.lo), int(dom.hi))
            for uid, dom in self.specialized_coordinate_domains.items()
            if uid in canonical
        }

        def proof_static_scalar_resolver(uid: str) -> int | None:
            cv = canonical.get(uid)
            if cv is None or cv.role != "scalar" or cv.aux_values:
                return None
            ok, value = self._proven_static_expr(
                ast.Call(func=ast.Name(id=cv.source_name, ctx=ast.Load()), args=[], keywords=[])
            )
            if ok and isinstance(value, int) and not isinstance(value, bool):
                return int(value)
            return None

        # Freeze the exact V02346 two-gate PureMap proof at the pre-scheduler
        # boundary.  The legacy scheduler reuses the same proof implementation, so
        # the new observational graph cannot silently broaden PureMap admission.
        snapshot_pure_maps = prove_executable_pure_maps(
            canonical,
            source_pure_map_names=self.source_pure_map_names,
            static_scalar_resolver=proof_static_scalar_resolver,
        )

        # Preserve all evidence needed by the pre-scheduler semantic graph.  Source
        # semantics remain provenance only; exact canonical UIDs remain identities.
        # Recovered loop evidence supplied lazily by NativeBatch is materialized by
        # capture_canonical_proof_snapshot() for audit tooling, not eagerly on the
        # production compile path.
        scheduler_evidence_status = (
            "complete_with_recovered_evidence" if self.recovered_loop_evidence is not None
            else "recovered_evidence_deferred" if self.recovered_loop_evidence_provider is not None
            else "complete_without_recovered_evidence"
        )
        self._canonical_proof_snapshot = CanonicalProofSnapshot(
            output_uid=output_uid,
            run_key_count=len(self.run_keys),
            variants=dict(canonical),
            formula_hash=str(self.ir.formula_hash),
            structural_hash=str(self.ir.structural_hash),
            coordinate_domains=tuple(sorted(
                (uid, int(dom.lo), int(dom.hi))
                for uid, dom in self.specialized_coordinate_domains.items()
                if uid in canonical
            )),
            static_facts=tuple(sorted(
                self._used_static_facts.values(), key=lambda x: (x.source_kind, x.source_name)
            )),
            validated_static_facts=tuple(sorted(
                self._used_validated_static_facts.values(), key=lambda x: x.source_name
            )),
            finite_domain_facts=tuple(sorted(
                self._used_finite_domain_facts.values(), key=lambda x: (x.source_kind, x.source_name)
            )),
            fixed_coordinate_facts=tuple(sorted(
                (fact for fact in self._used_fixed_coordinate_facts.values()
                 if fact.synthetic_uid in canonical),
                key=lambda x: x.synthetic_uid,
            )),
            coordinate_recurrence_facts=tuple(
                fact for fact in self.coordinate_recurrence_facts
                if fact.source_uid in canonical
            ),
            runtime_scalar_helper_facts=tuple(sorted(
                self._used_runtime_scalar_helper_facts.values(), key=lambda x: x.source_name
            )),
            source_backed_scheduled_facts=tuple(sorted(
                self._used_source_backed_scheduled_facts.values(),
                key=lambda x: (x.source_name, x.synthetic_uid),
            )),
            step_lookup_domain_facts=tuple(sorted(
                self._used_step_lookup_domain_facts.values(),
                key=lambda x: (x.source_name, x.source_uid, x.query_expr, x.axis_min),
            )),
            normalized_operators=tuple(sorted(self._normalized_operators.values(), key=lambda x: x.uid)),
            source_value_semantics=tuple(
                self.source_value_semantics[name] for name in sorted(self.source_value_semantics)
            ),
            pure_map_specs=tuple(snapshot_pure_maps[uid] for uid in sorted(snapshot_pure_maps)),
            loop_topology_facts=freeze_graph_ir_loop_topology(self.ir),
            recovered_loop_topology_facts=freeze_recovered_loop_topology(self.recovered_loop_evidence),
            scheduler_evidence_status=scheduler_evidence_status,
        )

        canonical_semantic_graph = None
        canonical_schedule_analysis = None
        canonical_execution_schedule = None
        stage_execution_plan = None
        stage_storage_plan = None
        if self.canonical_schedule_authority_enabled:
            # V02349 authority seam: construct schedule meaning before the legacy
            # executable builder.  Ineligibility is a typed contract result; an
            # analyzer exception is not silently converted into legacy authority.
            canonical_semantic_graph = build_canonical_semantic_graph(self._canonical_proof_snapshot)
            canonical_schedule_analysis = analyze_canonical_component_schedule(canonical_semantic_graph)
            canonical_execution_schedule = build_canonical_execution_schedule(
                canonical_schedule_analysis
            )
            # V02350A observational execution-plan seam.  This total canonical
            # stage/lifetime plan never changes production lowering in this
            # iteration; it records unsupported shapes as capability blockers.
            stage_execution_plan = build_stage_execution_plan(
                canonical_semantic_graph, canonical_schedule_analysis
            )
            stage_storage_plan = build_stage_storage_plan(
                canonical_semantic_graph, stage_execution_plan,
                analysis=canonical_schedule_analysis, variants=canonical,
            )
            diagnostics.append(
                "canonical stage execution plan built: "
                f"{stage_execution_plan.uid}; backend_eligible={stage_execution_plan.backend_eligible}"
            )
            diagnostics.append(
                "canonical stage storage plan built: "
                f"{stage_storage_plan.uid}; geometry_proven={stage_storage_plan.geometry_proven}"
            )
            if canonical_execution_schedule.eligible:
                diagnostics.append(
                    "canonical schedule authority eligible: "
                    f"{canonical_execution_schedule.uid}"
                )
            else:
                diagnostics.append(
                    "canonical schedule authority ineligible: "
                    + "; ".join(canonical_execution_schedule.capability_blockers)
                )

        executable = None
        domain_graph_shadow = None
        if self.legacy_comparison_enabled:
            try:
                executable = build_executable_graph(
                    self.ir, canonical, output_uid,
                    coordinate_domains=coordinate_proof_domains,
                    proof_static_scalar_resolver=proof_static_scalar_resolver,
                    source_pure_map_names=self.source_pure_map_names,
                    recovered_loop_evidence=self.recovered_loop_evidence,
                    recovered_loop_evidence_provider=self.recovered_loop_evidence_provider,
                    canonical_execution_schedule=canonical_execution_schedule,
                )
            except TemplateError as exc:
                raise FrontendError(f"legacy comparison executable lowering failed: {exc}") from exc
            # Comparison-only shadow retained for mature oracle tooling.  Canonical
            # Stage scheduling/storage never consumes this object.
            domain_graph_shadow = build_shadow_domain_graph(executable, canonical)
            if self._canonical_proof_snapshot is not None:
                self._canonical_proof_snapshot = (
                    self._canonical_proof_snapshot.with_executable_schedule()
                )
            diagnostics.append("legacy comparison executable built")
        else:
            diagnostics.append("legacy comparison executable skipped; canonical Stage pipeline owns production execution")

        return CanonicalModel(
            model=self.model, space=self.space, output_name=self.output, output_uid=output_uid,
            output_invocation=self.output_invocation,
            run_domain=self.run_domain,
            result_domain=self.result_domain,
            sample_keys=self.sample_keys, run_keys=self.run_keys, trace=self.trace,
            variants=canonical, inputs=self.registry.specs, executable=executable,
            domain_graph_shadow=domain_graph_shadow,
            canonical_semantic_graph=canonical_semantic_graph,
            canonical_schedule_analysis=canonical_schedule_analysis,
            canonical_execution_schedule=canonical_execution_schedule,
            stage_execution_plan=stage_execution_plan,
            stage_storage_plan=stage_storage_plan,
            normalized_operators=tuple(sorted(self._normalized_operators.values(), key=lambda x: x.uid)),
            diagnostics=diagnostics,
            static_facts=tuple(sorted(self._used_static_facts.values(), key=lambda x: (x.source_kind, x.source_name))),
            validated_static_facts=tuple(sorted(self._used_validated_static_facts.values(), key=lambda x: x.source_name)),
            finite_domain_facts=tuple(sorted(self._used_finite_domain_facts.values(), key=lambda x: (x.source_kind, x.source_name))),
            fixed_coordinate_facts=tuple(
                sorted(
                    (fact for fact in self._used_fixed_coordinate_facts.values() if fact.synthetic_uid in canonical),
                    key=lambda x: x.synthetic_uid,
                )
            ),
            coordinate_recurrence_facts=tuple(
                fact for fact in self.coordinate_recurrence_facts
                if fact.source_uid in canonical
            ),
            runtime_scalar_helper_facts=tuple(
                sorted(self._used_runtime_scalar_helper_facts.values(), key=lambda x: x.source_name)
            ),
            source_backed_scheduled_facts=tuple(
                sorted(
                    self._used_source_backed_scheduled_facts.values(),
                    key=lambda x: (x.source_name, x.synthetic_uid),
                )
            ),
            step_lookup_domain_facts=tuple(
                sorted(
                    self._used_step_lookup_domain_facts.values(),
                    key=lambda x: (x.source_name, x.source_uid, x.query_expr, x.axis_min),
                )
            ),
        )

_FAST_TRANSLATION_METHOD_NAMES = (
    '_affine_int_expr', '_affine_symbol_bounds_many', '_bind_source_call',
    '_bound_affine_under_primary_range', '_compare_domain_truth',
    '_context_cell_for_key', '_context_itemspace_cell',
    '_context_parameterized_space_for_key', '_context_primitive',
    '_context_root_for_key', '_context_selected_itemspace',
    '_context_static_argument_is_fixed', '_context_static_value',
    '_coordinate_time_position', '_correlated_int_domain_expr',
    '_detect_point_row_cell', '_discover_validated_enum_cells',
    '_discover_validated_point_scalar_cells', '_discover_vector_helpers',
    '_derive_coordinate_domains', '_derive_coordinate_recurrence_facts',
    '_derive_specialized_coordinate_domains', '_domain_truth',
    '_enumerate_straight_paths',
    '_expand_pure_single_return_call', '_expr_lowerer',
    '_fold_fixed_local_constants', '_formula_unbound_load_names',
    '_function_bound_names', '_hierarchical_itemspace_parameter_binding',
    '_inline_fixed_time_calls', '_inline_runtime_scalar_helper_call',
    '_inline_runtime_scalar_helpers_as_statements', '_inline_vector_subscripts',
    '_int_domain_expr', '_is_modelx_cells_object', '_is_modelx_itemspace_object',
    '_is_modelx_space_object', '_is_numpy_array_call',
    '_materialize_static_value_operators', '_normalize_coordinate_expression',
    '_normalize_external_tables', '_normalize_local_mapping_iteration', '_normalize_reduction',
    '_normalize_scalar_intrinsics', '_normalize_simple_local_syntax',
    '_prepare_runtime_scalar_helper_function', '_range_affine_constraint_from_args',
    '_range_domain_from_args', '_range_partial_bounds_from_args',
    '_proven_coordinate_cell_constant', '_proven_finite_cell_domain',
    '_proven_integer_point_field_bounds', '_proven_numeric_point_field_bounds',
    '_proven_static_cell_value', '_proven_static_expr',
    '_proven_static_helper_argument', '_proven_static_point_field_value',
    '_proven_static_reference_value', '_proven_static_variant_value',
    '_prune_to_realized_path', '_record_runtime_scalar_helper_fact',
    '_refine_domain_env',
    '_reference_metadata_scalar_ok', '_request_fixed_coordinate',
    '_request_source_backed_scheduled_variant', '_run_key_parameter_value',
    '_runtime_scalar_helper_candidates', '_runtime_scalar_helper_expr_from_block',
    '_same_space_calls', '_source_backed_unobserved_helper_ok',
    '_pure_single_return_metadata', 'proof_diagnostics',
    '_source_coordinate_relative_offset', '_source_has_future_coordinate_dependency',
    '_source_reaches_recursive_cycle', '_space_instance', '_merge_domain_states',
    '_sparse_observed_coordinate_helper_candidates', '_specialize_function',
    '_specialize_static_facts_and_domains', '_stable_realized_dep_names',
    '_unwrap_generator', '_validate_hierarchical_itemspace_binding',
    '_variant_for_call',
)


class FormulaTranslationContext:
    """Preparation-only formula/provider translator with no execution authority.

    The mature translation operations are shared with ``GraphModelCompiler`` while
    they are decommissioned from that historical host.  This context deliberately
    creates no GraphIR, canonical graph, schedule, Stage plan or storage plan.
    """

    _PRIMARY_AFFINE_COORD = GraphModelCompiler._PRIMARY_AFFINE_COORD
    _POINT_AFFINE_PREFIX = GraphModelCompiler._POINT_AFFINE_PREFIX

    def __init__(
        self,
        model,
        *,
        space,
        output: str,
        sample_keys: Sequence[Any],
        run_keys: Sequence[Any],
        point_keys: Sequence[Any] | None = None,
        point_frame: Any | None = None,
    ) -> None:
        self.model = model
        self.space = space
        self.output_invocation = OutputInvocation.fixed(str(output), (), ())
        self.output = str(output)
        self.sample_keys = tuple(sample_keys)
        self.run_keys = tuple(run_keys)
        self.point_keys = self.run_keys if point_keys is None else tuple(point_keys)
        self.point_frame = point_frame
        self.fallback_cells = frozenset()
        self.realized_graph_only = False
        self.fallback_input_by_name: dict[str, str] = {}
        self.refs = dict(space.refs)
        self.space_params = tuple(space.parameters or ())
        self.space_param = self.space_params[0] if self.space_params else None
        try:
            self.run_domain = RunDomain.finite(
                space_fullname=str(
                    getattr(space, "fullname", None) or getattr(space, "name", "<space>")
                ),
                itemspace_parameter_names=self.space_params,
                run_keys=self.run_keys,
            )
        except (TypeError, ValueError) as exc:
            raise FrontendError(str(exc)) from exc
        self.result_domain = ResultDomain.scalar_per_run_key(self.run_domain)
        self.multiindex_axes: dict[str, tuple[int, int, int, int]] = {}
        self.static_category_result_domains: dict[str, tuple[str, ...]] = {}
        self.static_category_parameter_domains: dict[tuple[str, str], tuple[str, ...]] = {}
        self.static_category_nullable_parameters: set[tuple[str, str]] = set()
        self.static_category_reachable_result_domains: dict[tuple[str, str], tuple[str, ...]] = {}
        # Complete finite integer parameter domains proved from selected source
        # call semantics.  These are preparation-only evidence for finite symbolic
        # dispatch; they do not alter modelx-cython parameter ABI or scheduling.
        self.finite_symbolic_parameter_domains: dict[tuple[str, str], tuple[int, ...]] = {}
        self.finite_symbolic_parameter_domain_sources: dict[tuple[str, str], str] = {}
        # Demand-driven source-call lower bounds used only to prove provider/query
        # preconditions. These come from complete selected-output call paths plus
        # dominating early-return guards, never representative trace extrema.
        self.source_guarded_parameter_lower_bounds: dict[tuple[str, str], float] = {}
        self.source_guarded_parameter_lower_bound_sources: dict[tuple[str, str], str] = {}
        # Fast modelx-cython formulas are catalogued under canonical ``mc_*``
        # UIDs. Translation-time semantic proofs still belong to the original
        # modelx Cell names. The graph-free fast service configures this bridge
        # so provider/static-value normalizers can reuse complete-RunDomain Cell
        # proofs after call renaming.
        self.formula_uid_to_source_name: dict[str, str] = {}
        self.registry = InputRegistry(self)
        self.model_point_row = self._detect_point_row_cell()
        _emit_translation_context_diagnostic("point_row_detected")

        self.trace = capture_trace(
            model,
            lambda key: getattr(self._space_instance(key), self.output).node(),
            self.sample_keys,
        )
        _emit_translation_context_diagnostic(
            "trace_captured",
            trace_variants=len(self.trace.variants),
            trace_nodes=len(self.trace.node_variant),
        )
        self.trace_source = "realized_trace"
        self.selected_schema_uid_by_name = {
            name: CellSchema(cell.fullname, name, tuple(cell.parameters), cell.formula.source).uid
            for name, cell in space.cells.items()
            if cell.formula is not None
        }
        self.selected_trace_variants = {
            key: row
            for key, row in self.trace.variants.items()
            if self.selected_schema_uid_by_name.get(row.schema.name) == row.schema.uid
        }
        explicit = [
            row.schema.fullname
            for row in self.selected_trace_variants.values()
            if row.has_input_values
        ]
        if explicit:
            raise FrontendError(
                "explicit Cell input values are present in the sampled closure; "
                "fast formula translation requires prepared provider inputs: "
                + ", ".join(sorted(set(explicit)))
            )

        self.base_funcs = {
            name: self._parse_formula_source(name, cell.formula.source)
            for name, cell in space.cells.items()
            if cell.formula is not None
        }
        _emit_translation_context_diagnostic("formula_catalogued", formula_count=len(self.base_funcs))
        semantics_source = ast.unparse(
            ast.fix_missing_locations(
                ast.Module(
                    body=[copy.deepcopy(self.base_funcs[name]) for name in sorted(self.base_funcs)],
                    type_ignores=[],
                )
            )
        )
        self.source_value_semantics = {
            row.uid: row for row in classify_source_value_semantics(semantics_source)
        }
        self.source_pure_map_names = frozenset(
            uid
            for uid, row in self.source_value_semantics.items()
            if row.semantic_class == "pure_map"
        )
        _emit_translation_context_diagnostic(
            "source_semantics_classified",
            semantic_rows=len(self.source_value_semantics),
        )
        self.compiled_source_names: set[str] = set()
        stack = [self.output]
        while stack:
            name = stack.pop()
            if name in self.compiled_source_names or name not in self.base_funcs:
                continue
            self.compiled_source_names.add(name)
            for node in ast.walk(self.base_funcs[name]):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in self.base_funcs:
                        stack.append(node.func.id)
                elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                    if node.value.id in self.base_funcs:
                        stack.append(node.value.id)
        _emit_translation_context_diagnostic(
            "source_closure_selected",
            compiled_formula_count=len(self.compiled_source_names),
        )
        self.schema_by_name = {}
        for row in self.selected_trace_variants.values():
            self.schema_by_name.setdefault(row.schema.name, row.schema)
        self.variant_trace_by_uid = {
            row.uid: row for row in self.selected_trace_variants.values()
        }
        self.variants_by_cell_aux = {
            (row.schema.name, row.key.aux_values): row.uid
            for row in self.selected_trace_variants.values()
        }
        self.validated_enum_cells = self._discover_validated_enum_cells()
        self.validated_point_scalar_cells = self._discover_validated_point_scalar_cells()
        _emit_translation_context_diagnostic(
            "validated_cells_discovered",
            enum_count=len(self.validated_enum_cells),
            point_scalar_count=len(self.validated_point_scalar_cells),
        )

        self._static_variant_value_cache = {}
        self._finite_variant_domain_cache = {}
        self._static_reference_value_cache = {}
        self._used_static_facts = {}
        self._normalized_operators = {}
        self._used_validated_static_facts = {}
        self._used_finite_domain_facts = {}
        self._fixed_coordinate_uid_by_key = {}
        self._fixed_coordinate_requests = {}
        self._used_fixed_coordinate_facts = {}
        self._fixed_coordinate_limit = 128
        self._future_coordinate_source_cache = {}
        self.coordinate_recurrence_facts = ()
        self._runtime_scalar_helper_inline_stack = []
        self._runtime_scalar_helper_expansion_limit = RUNTIME_SCALAR_HELPER_EXPANSION_LIMIT
        self._used_runtime_scalar_helper_facts = {}
        self._static_point_field_cache = {}
        self._coordinate_constant_cache = {}
        self.coordinate_domains = {}
        self.specialized_coordinate_domains = {}
        self.primary_coordinate_domain = None
        self.primary_coordinate_bounds = (None, None)
        self.primary_coordinate_affine_constraint = None
        self.coordinate_domain_notes = []
        self._source_recursive_reach_cache = {}
        self._source_backed_helper_cache = {}
        self._pure_single_return_meta_cache = {}
        self._proof_diagnostics = {
            "numeric_bounds_requests": 0,
            "numeric_bounds_cache_hits": 0,
            "numeric_bounds_cache_stores": 0,
            "numeric_bounds_cycle_cuts": 0,
            "numeric_bounds_active_peak": 0,
            "numeric_bounds_alias_hops": 0,
            "numeric_bounds_alias_cycle_cuts": 0,
            "pure_single_return_meta_hits": 0,
            "pure_single_return_meta_misses": 0,
            "pure_single_return_expansions": 0,
        }
        self._source_backed_scheduled_uid_by_key = {}
        self._used_source_backed_scheduled_facts = {}
        self._used_step_lookup_domain_facts = {}
        self._canonical_proof_snapshot = None
        self.vector_elements = self._discover_vector_helpers()
        saved_registry = (
            dict(self.registry.specs),
            dict(self.registry.memo),
            dict(self.registry.mapping_ragged_memo),
            dict(self.registry.point_lookup_memo),
            int(self.registry.counter),
        )
        saved_static = dict(self._used_static_facts)
        saved_validated = dict(self._used_validated_static_facts)
        saved_finite = dict(self._used_finite_domain_facts)
        recurrence_facts: tuple[CoordinateRecurrenceDomainFact, ...] = ()
        try:
            self.coordinate_domains = self._derive_coordinate_domains()
            _emit_translation_context_diagnostic(
                "coordinate_domains_derived",
                coordinate_domain_count=len(self.coordinate_domains),
            )
            recurrence_facts = self._derive_coordinate_recurrence_facts(
                self.coordinate_domains
            )
        finally:
            (
                self.registry.specs, self.registry.memo,
                self.registry.mapping_ragged_memo, self.registry.point_lookup_memo,
                self.registry.counter
            ) = saved_registry
            self._used_static_facts = saved_static
            self._used_validated_static_facts = saved_validated
            self._used_finite_domain_facts = saved_finite
        self.coordinate_recurrence_facts = recurrence_facts
        _emit_translation_context_diagnostic(
            "coordinate_recurrence_derived",
            recurrence_fact_count=len(self.coordinate_recurrence_facts),
        )

    @staticmethod
    def _parse_formula_source(name: str, source: str) -> ast.FunctionDef:
        """Normalize modelx's two formula source forms to one FunctionDef.

        ``Cells.formula.source`` retains a lambda when a Cell was originally
        declared with one.  The exported modelx-cython source already expands
        that lambda to a function, so the preparation-only provider translator
        must do the equivalent normalization before walking ``fn.body``.
        """
        node = ast.parse(source).body[0]
        if isinstance(node, ast.FunctionDef):
            return node
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Lambda)
        ):
            result = ast.FunctionDef(
                name=str(name),
                args=copy.deepcopy(node.value.args),
                body=[ast.Return(value=copy.deepcopy(node.value.body))],
                decorator_list=[],
            )
            return ast.fix_missing_locations(result)
        raise FrontendError(
            f"formula {name!r} is neither a function definition nor a lambda"
        )

    def authority_manifest(self) -> dict[str, bool | str]:
        return {
            "schema": "modelx_graph.formula_translation_context.v1",
            "graph_authority": False,
            "schedule_authority": False,
            "stage_authority": False,
            "storage_authority": False,
        }


for _translation_method_name in _FAST_TRANSLATION_METHOD_NAMES:
    setattr(
        FormulaTranslationContext,
        _translation_method_name,
        GraphModelCompiler.__dict__[_translation_method_name],
    )

# Complete the migration rather than leaving two translator copies.  The public
# compiler composes the preparation context with its graph/execution core; the core
# no longer owns any of the fast translator methods.
_GraphCompilerExecutionCore = GraphModelCompiler
for _translation_method_name in _FAST_TRANSLATION_METHOD_NAMES:
    delattr(_GraphCompilerExecutionCore, _translation_method_name)


class GraphModelCompiler(FormulaTranslationContext, _GraphCompilerExecutionCore):
    """Full compiler assembled from translation and execution-authority concerns."""

    __init__ = _GraphCompilerExecutionCore.__init__


del _translation_method_name


def capture_canonical_proof_snapshot(*args, **kwargs) -> CanonicalProofSnapshot:
    """Run the normal frontend and retain its pre-scheduler proof snapshot.

    Audit capture additionally materializes a deferred recovered-loop provider when
    one was supplied.  This keeps the *production* compile path lazy while making
    the returned evidence object self-contained enough for the observational
    canonical scheduler.  Provider failure is recorded as incomplete evidence; it
    never turns an otherwise valid canonical graph into an exception.
    """

    compiler = GraphModelCompiler.__new__(GraphModelCompiler)
    failure: FrontendError | None = None
    try:
        GraphModelCompiler.__init__(compiler, *args, **kwargs)
    except FrontendError as exc:
        failure = exc
    snapshot = getattr(compiler, "_canonical_proof_snapshot", None)
    if snapshot is None:
        if failure is not None:
            raise failure
        raise FrontendError("frontend completed without a canonical proof snapshot")

    provider = getattr(compiler, "recovered_loop_evidence_provider", None)
    existing = getattr(compiler, "recovered_loop_evidence", None)
    if existing is not None:
        snapshot = snapshot.with_recovered_loop_evidence(existing)
    elif provider is not None:
        try:
            snapshot = snapshot.with_recovered_loop_evidence(provider())
        except Exception as exc:
            snapshot = snapshot._copy(
                scheduler_evidence_status=f"recovered_evidence_failed:{type(exc).__name__}"
            )

    if failure is not None:
        snapshot = snapshot.with_failure(str(failure))
    return snapshot
