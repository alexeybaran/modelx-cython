from __future__ import annotations

"""Pre-scheduler canonical semantic graph.

The graph in this module is deliberately *observational*.  It is built from the
immutable canonical frontend proof snapshot before the legacy executable scheduler
is allowed to grant storage/codegen permission.  Graph construction is total for a
valid snapshot: unsupported relations/operators are preserved as explicit UNKNOWN
or unsupported evidence instead of raising a scheduling error.

Exact specialized canonical UIDs are the only graph identities.  Source-level
semantics are attached as provenance and may constrain a canonical execution
classification, but source names never become scheduling identities.
"""

import ast
import hashlib
from dataclasses import dataclass, replace
from typing import Any, Iterable


_PURE_EXTERNAL_CALLS = frozenset({
    "abs", "min", "max", "round", "int", "float", "bool",
    "point_input", "global_input", "array_input", "table_input",
    "__is_missing__", "__is_inf__", "__erf__",
    "__step_lookup_1d__", "__interval_lookup_1d__", "__interp_lookup_1d__",
    "__exact_axis_code__", "__sparse_table_lookup__", "__prepared_point_lookup__",
})
_PURE_MODULE_CALLS = frozenset({"exp", "log", "sqrt", "sin", "cos", "floor", "ceil"})


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


def _expr(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, include_attributes=False)


def _affine_offset(node: ast.AST, variable: str) -> int | None:
    if isinstance(node, ast.Name) and node.id == variable:
        return 0
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.left, ast.Name)
        and node.left.id == variable
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, int)
        and not isinstance(node.right.value, bool)
    ):
        if isinstance(node.op, ast.Sub):
            return -int(node.right.value)
        if isinstance(node.op, ast.Add):
            return int(node.right.value)
    return None


def _constant_int_expr(node: ast.AST) -> int | None:
    """Evaluate a tiny side-effect-free integer expression exactly.

    Canonical fixed OutputInvocations are source-specialized before this graph is
    built, so coordinate arguments such as ``0`` or ``0 + 1`` are ordinary
    constant arithmetic.  Keeping this evaluator deliberately tiny prevents an
    axis-less output from acquiring execution geometry through arbitrary Python
    evaluation.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        return None
    if isinstance(node, ast.UnaryOp):
        value = _constant_int_expr(node.operand)
        if value is None:
            return None
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        return None
    if isinstance(node, ast.BinOp):
        left = _constant_int_expr(node.left)
        right = _constant_int_expr(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right != 0:
            return left // right
        if isinstance(node.op, ast.Mod) and right != 0:
            return left % right
    return None


def _source_semantics_tuple(row: Any | None) -> tuple[str, str, str, str, tuple[str, ...], tuple[str, ...]]:
    if row is None:
        return ("unknown", "unknown", "unknown", "missing_source_semantics", (), ())
    return (
        str(getattr(row, "semantic_class", "unknown")),
        str(getattr(row, "state_dependency", "unknown")),
        str(getattr(row, "persistence", "unknown")),
        str(getattr(row, "proof_kind", "unknown")),
        tuple(str(x) for x in getattr(row, "operator_traits", ())),
        tuple(str(x) for x in getattr(row, "blockers", ())),
    )


@dataclass(frozen=True)
class CanonicalLoopTopologyFact:
    uid: str
    signature: str
    coordinate_positions: tuple[tuple[str, int], ...]
    observed_spans: tuple[tuple[str, int, int], ...]
    repetitions: int
    coordinate_stride: int
    boundary_coordinates: tuple[tuple[str, int], ...]
    generalization: str
    proof_notes: tuple[str, ...] = ()
    provenance: str = "graph_ir_loop_template"

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "signature": self.signature,
            "coordinate_positions": dict(self.coordinate_positions),
            "observed_spans": [list(x) for x in self.observed_spans],
            "repetitions": int(self.repetitions),
            "coordinate_stride": int(self.coordinate_stride),
            "boundary_coordinates": [list(x) for x in self.boundary_coordinates],
            "generalization": self.generalization,
            "proof_notes": list(self.proof_notes),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class RecoveredLoopTopologyFact:
    uid: str
    repetitions: int
    phase_count: int
    grammar_signature: str
    formula_family_signature: str
    formula_fullnames: tuple[str, ...]
    affine_coordinate_steps: tuple[tuple[str, int], ...]
    coordinate_domains: tuple[tuple[str, int, int, int, int, int | None, str], ...]
    exact_expansion_validated: bool
    provenance: str = "recovered_loop_evidence"

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "repetitions": int(self.repetitions),
            "phase_count": int(self.phase_count),
            "grammar_signature": self.grammar_signature,
            "formula_family_signature": self.formula_family_signature,
            "formula_fullnames": list(self.formula_fullnames),
            "affine_coordinate_steps": [list(x) for x in self.affine_coordinate_steps],
            "coordinate_domains": [
                {
                    "fullname": x[0], "occurrence_count": int(x[1]),
                    "minimum": int(x[2]), "maximum": int(x[3]),
                    "first": int(x[4]), "step": x[5], "binding_kind": x[6],
                }
                for x in self.coordinate_domains
            ],
            "exact_expansion_validated": bool(self.exact_expansion_validated),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CanonicalSemanticNode:
    uid: str
    source_name: str
    source_fullname: str
    canonical_role: str
    dtype: str
    parameters: tuple[str, ...]
    domain: tuple[int, int] | None
    canonical_ast: str
    canonical_source: str
    state_semantic: str  # state_free | auxiliary_recurrence | persistent_state | state_derived_map | unknown
    execution_semantic: str  # pure_map | auxiliary_recurrence | scheduled_state | derived_task | value | unknown
    source_semantic_class: str
    source_state_dependency: str
    source_persistence: str
    source_proof_kind: str
    operator_traits: tuple[str, ...] = ()
    semantic_blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_name": self.source_name,
            "source_fullname": self.source_fullname,
            "canonical_role": self.canonical_role,
            "dtype": self.dtype,
            "parameters": list(self.parameters),
            "domain": None if self.domain is None else {"lo": int(self.domain[0]), "hi": int(self.domain[1])},
            "canonical_ast": self.canonical_ast,
            "canonical_source": self.canonical_source,
            "state_semantic": self.state_semantic,
            "execution_semantic": self.execution_semantic,
            "source_semantic_class": self.source_semantic_class,
            "source_state_dependency": self.source_state_dependency,
            "source_persistence": self.source_persistence,
            "source_proof_kind": self.source_proof_kind,
            "operator_traits": list(self.operator_traits),
            "semantic_blockers": list(self.semantic_blockers),
        }


@dataclass(frozen=True)
class CanonicalSemanticAccess:
    uid: str
    source_uid: str
    target_uid: str
    context: str
    argument_exprs: tuple[str, ...]
    argument_asts: tuple[str, ...]
    keyword_exprs: tuple[tuple[str, str], ...]
    call_ast: str
    relation: str  # scalar | state_free | same | strict_before | strict_after | fixed | unknown
    min_offset: int | None
    max_offset: int | None
    scheduling_relevant: bool
    proof_kind: str
    fixed_coordinate: int | None = None
    blockers: tuple[str, ...] = ()
    lineno: int = -1
    col_offset: int = -1

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_uid": self.source_uid,
            "target_uid": self.target_uid,
            "context": self.context,
            "argument_exprs": list(self.argument_exprs),
            "argument_asts": list(self.argument_asts),
            "keyword_exprs": [list(x) for x in self.keyword_exprs],
            "call_ast": self.call_ast,
            "relation": self.relation,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "scheduling_relevant": bool(self.scheduling_relevant),
            "proof_kind": self.proof_kind,
            "fixed_coordinate": self.fixed_coordinate,
            "blockers": list(self.blockers),
            "lineno": int(self.lineno),
            "col_offset": int(self.col_offset),
        }


@dataclass(frozen=True)
class CanonicalOperatorUse:
    uid: str
    target_uid: str
    operator_kind: str  # call | control
    expression: str
    supported_by_pure_subset: bool
    lineno: int = -1
    col_offset: int = -1
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "target_uid": self.target_uid,
            "operator_kind": self.operator_kind,
            "expression": self.expression,
            "supported_by_pure_subset": bool(self.supported_by_pure_subset),
            "lineno": int(self.lineno),
            "col_offset": int(self.col_offset),
            "note": self.note,
        }


@dataclass(frozen=True)
class StateFreeCoordinateExecutionProof:
    """Graph-native proof that a lowered coordinate helper is executable on demand.

    This proof complements ``ExecutablePureMapSpec`` rather than replacing it.
    The older spec proves a source-level PureMap before canonical scheduling.  This
    record proves an equivalent execution ABI *after* frontend lowering, using the
    frozen canonical program, state semantics, dependencies, operators and domain.
    A failed proof is retained with typed blockers so ``state_free`` never becomes
    synonymous with executable by accident.
    """

    proof_uid: str
    value_uid: str
    coordinate_parameter: str
    coordinate_domain: tuple[int, int] | None
    coordinate_domain_proof_kind: str
    coordinate_domain_evidence_uids: tuple[str, ...]
    direct_dependency_uids: tuple[str, ...]
    scalar_dependencies: tuple[str, ...]
    pure_map_dependencies: tuple[str, ...]
    access_uids: tuple[str, ...]
    operator_uids: tuple[str, ...]
    approved: bool
    proof_kind: str = "canonical_lowered_coordinate_pure_v1"
    provenance: tuple[str, ...] = (
        "source_value_semantics_v1",
        "canonical_exact_state_free_refinement_v1",
        "canonical_lowered_numeric_execution_subset_v1",
    )
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "proof_uid": self.proof_uid,
            "value_uid": self.value_uid,
            "coordinate_parameter": self.coordinate_parameter,
            "coordinate_domain": (
                None
                if self.coordinate_domain is None
                else {"lo": int(self.coordinate_domain[0]), "hi": int(self.coordinate_domain[1])}
            ),
            "coordinate_domain_proof_kind": self.coordinate_domain_proof_kind,
            "coordinate_domain_evidence_uids": list(self.coordinate_domain_evidence_uids),
            "direct_dependency_uids": list(self.direct_dependency_uids),
            "scalar_dependencies": list(self.scalar_dependencies),
            "pure_map_dependencies": list(self.pure_map_dependencies),
            "access_uids": list(self.access_uids),
            "operator_uids": list(self.operator_uids),
            "approved": bool(self.approved),
            "proof_kind": self.proof_kind,
            "provenance": list(self.provenance),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class FormulaSchema:
    """Trace-independent structural formula identity used by semantic passes.

    This is deliberately smaller than ``CanonicalSemanticNode``: source text is
    excluded so downstream consumers cannot accidentally make syntax authoritative.
    """

    uid: str
    canonical_role: str
    dtype: str
    parameters: tuple[str, ...]
    state_semantic: str
    execution_semantic: str
    source_proof_kind: str
    operator_traits: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "canonical_role": self.canonical_role,
            "dtype": self.dtype,
            "parameters": list(self.parameters),
            "state_semantic": self.state_semantic,
            "execution_semantic": self.execution_semantic,
            "source_proof_kind": self.source_proof_kind,
            "operator_traits": list(self.operator_traits),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class DomainFact:
    """Proven execution-coordinate domain attached to one canonical value."""

    uid: str
    value_uid: str
    axis: str
    kind: str
    lo: int | None
    hi: int | None
    proof_kind: str
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "value_uid": self.value_uid,
            "axis": self.axis,
            "kind": self.kind,
            "lo": self.lo,
            "hi": self.hi,
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class CoordinateRelation:
    """Scheduling-safe coordinate relation detached from source syntax."""

    uid: str
    access_uid: str
    source_uid: str
    target_uid: str
    relation: str
    min_offset: int | None
    max_offset: int | None
    scheduling_relevant: bool
    proof_kind: str
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "access_uid": self.access_uid,
            "source_uid": self.source_uid,
            "target_uid": self.target_uid,
            "relation": self.relation,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "scheduling_relevant": bool(self.scheduling_relevant),
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class SemanticOperator:
    """Backend-neutral semantic operator observation, detached from Python objects."""

    uid: str
    target_uid: str
    operator_kind: str
    expression: str
    supported: bool
    proof_kind: str
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "target_uid": self.target_uid,
            "operator_kind": self.operator_kind,
            "expression": self.expression,
            "supported": bool(self.supported),
            "proof_kind": self.proof_kind,
            "note": self.note,
        }


@dataclass(frozen=True)
class CanonicalFootprintPath:
    """One exact persistent-leaf dependency frozen before scheduling."""

    uid: str
    root_uid: str
    persistent_source_uid: str
    min_offset: int | None
    max_offset: int | None
    availability: str
    access_uids: tuple[str, ...]
    proof_kinds: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "root_uid": self.root_uid,
            "persistent_source_uid": self.persistent_source_uid,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "availability": self.availability,
            "access_uids": list(self.access_uids),
            "proof_kinds": list(self.proof_kinds),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class CanonicalStateFootprintEvidence:
    uid: str
    node_uid: str
    state_semantic: str
    persistent_source_uids: tuple[str, ...]
    paths: tuple[CanonicalFootprintPath, ...]
    availability: str
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "node_uid": self.node_uid,
            "state_semantic": self.state_semantic,
            "persistent_source_uids": list(self.persistent_source_uids),
            "paths": [x.manifest() for x in self.paths],
            "availability": self.availability,
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class CanonicalTransitionEvidence:
    """Persistent leaf -> persistent transition dependency frozen pre-scheduler."""

    uid: str
    source_uid: str
    target_uid: str
    min_offset: int | None
    max_offset: int | None
    availability: str
    access_uids: tuple[str, ...]
    proof_kinds: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_uid": self.source_uid,
            "target_uid": self.target_uid,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "availability": self.availability,
            "access_uids": list(self.access_uids),
            "proof_kinds": list(self.proof_kinds),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class CanonicalRecurrenceDomainFact:
    """Syntax-free finite domain/boundary proof for one recurrent value."""

    uid: str
    value_uid: str
    coordinate_parameter: str
    demand_domain: tuple[int, int]
    active_domain: tuple[int, int]
    recurrence_offsets: tuple[int, ...]
    boundary_coordinates: tuple[int, ...]
    scan_requirement: str
    output_invocation_uid: str
    proof_kind: str
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "value_uid": self.value_uid,
            "coordinate_parameter": self.coordinate_parameter,
            "demand_domain": list(self.demand_domain),
            "active_domain": list(self.active_domain),
            "recurrence_offsets": list(self.recurrence_offsets),
            "boundary_coordinates": list(self.boundary_coordinates),
            "scan_requirement": self.scan_requirement,
            "output_invocation_uid": self.output_invocation_uid,
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class CanonicalSemanticGraph:
    schema: str
    output_uid: str
    run_key_count: int
    formula_hash: str
    structural_hash: str
    nodes: tuple[CanonicalSemanticNode, ...]
    accesses: tuple[CanonicalSemanticAccess, ...]
    operators: tuple[CanonicalOperatorUse, ...]
    loop_topology_facts: tuple[CanonicalLoopTopologyFact, ...]
    recovered_loop_topology_facts: tuple[RecoveredLoopTopologyFact, ...]
    pure_map_specs: tuple[Any, ...]
    scheduler_evidence_status: str
    auxiliary_recurrence_uids: tuple[str, ...] = ()
    state_free_execution_proofs: tuple[StateFreeCoordinateExecutionProof, ...] = ()
    recurrence_domain_facts: tuple[CanonicalRecurrenceDomainFact, ...] = ()
    formula_schemas: tuple[FormulaSchema, ...] = ()
    domain_facts: tuple[DomainFact, ...] = ()
    coordinate_relations: tuple[CoordinateRelation, ...] = ()
    semantic_operators: tuple[SemanticOperator, ...] = ()
    state_footprints: tuple[CanonicalStateFootprintEvidence, ...] = ()
    transition_evidence: tuple[CanonicalTransitionEvidence, ...] = ()
    semantic_analysis_schema: str = ""
    validation_notes: tuple[str, ...] = ()

    def node(self, uid: str) -> CanonicalSemanticNode:
        return next(x for x in self.nodes if x.uid == uid)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "output_uid": self.output_uid,
            "run_key_count": int(self.run_key_count),
            "formula_hash": self.formula_hash,
            "structural_hash": self.structural_hash,
            "nodes": [x.manifest() for x in self.nodes],
            "accesses": [x.manifest() for x in self.accesses],
            "operators": [x.manifest() for x in self.operators],
            "loop_topology_facts": [x.manifest() for x in self.loop_topology_facts],
            "recovered_loop_topology_facts": [x.manifest() for x in self.recovered_loop_topology_facts],
            "pure_map_specs": [x.manifest() for x in self.pure_map_specs],
            "scheduler_evidence_status": self.scheduler_evidence_status,
            "auxiliary_recurrence_uids": list(self.auxiliary_recurrence_uids),
            "state_free_execution_proofs": [x.manifest() for x in self.state_free_execution_proofs],
            "recurrence_domain_facts": [x.manifest() for x in self.recurrence_domain_facts],
            "formula_schemas": [x.manifest() for x in self.formula_schemas],
            "domain_facts": [x.manifest() for x in self.domain_facts],
            "coordinate_relations": [x.manifest() for x in self.coordinate_relations],
            "semantic_operators": [x.manifest() for x in self.semantic_operators],
            "state_footprints": [x.manifest() for x in self.state_footprints],
            "transition_evidence": [x.manifest() for x in self.transition_evidence],
            "semantic_analysis_schema": self.semantic_analysis_schema,
            "metrics": {
                "node_count": len(self.nodes),
                "access_count": len(self.accesses),
                "operator_count": len(self.operators),
                "pure_map_count": sum(x.execution_semantic == "pure_map" for x in self.nodes),
                "canonical_state_free_execution_proof_count": sum(
                    x.approved for x in self.state_free_execution_proofs
                ),
                "canonical_state_free_execution_rejection_count": sum(
                    not x.approved for x in self.state_free_execution_proofs
                ),
                "canonical_recurrence_domain_fact_count": len(self.recurrence_domain_facts),
                "auxiliary_recurrence_count": len(self.auxiliary_recurrence_uids),
                "persistent_state_count": sum(x.state_semantic == "persistent_state" for x in self.nodes),
                "state_derived_map_count": sum(x.state_semantic == "state_derived_map" for x in self.nodes),
                "state_free_count": sum(x.state_semantic == "state_free" for x in self.nodes),
                "auxiliary_recurrence_state_count": sum(x.state_semantic == "auxiliary_recurrence" for x in self.nodes),
                "unknown_state_semantic_count": sum(x.state_semantic == "unknown" for x in self.nodes),
                "unknown_execution_semantic_count": sum(x.execution_semantic == "unknown" for x in self.nodes),
                "unknown_relation_count": sum(x.relation == "unknown" for x in self.accesses),
                "scheduling_relevant_access_count": sum(x.scheduling_relevant for x in self.accesses),
                "unsupported_operator_count": sum(not x.supported_by_pure_subset for x in self.operators),
                "formula_schema_count": len(self.formula_schemas),
                "domain_fact_count": len(self.domain_facts),
                "coordinate_relation_count": len(self.coordinate_relations),
                "semantic_operator_count": len(self.semantic_operators),
                "state_footprint_count": len(self.state_footprints),
                "transition_evidence_count": len(self.transition_evidence),
                "unknown_footprint_count": sum(x.availability == "UNKNOWN" for x in self.state_footprints),
                "unknown_transition_count": sum(x.availability == "UNKNOWN" for x in self.transition_evidence),
            },
            "validation_notes": list(self.validation_notes),
        }


def freeze_graph_ir_loop_topology(ir: Any) -> tuple[CanonicalLoopTopologyFact, ...]:
    return tuple(
        CanonicalLoopTopologyFact(
            uid=str(row.uid),
            signature=str(row.signature),
            coordinate_positions=tuple(sorted((str(k), int(v)) for k, v in row.coordinate_positions.items())),
            observed_spans=tuple((str(a), int(b), int(c)) for a, b, c in row.observed_spans),
            repetitions=int(row.repetitions),
            coordinate_stride=int(row.coordinate_stride),
            boundary_coordinates=tuple((str(a), int(b)) for a, b in row.boundary_coordinates),
            generalization=str(row.generalization),
            proof_notes=tuple(str(x) for x in row.proof_notes),
        )
        for row in sorted(getattr(ir, "loop_templates", ()), key=lambda x: str(x.uid))
    )


def freeze_recovered_loop_topology(evidence: Any | None) -> tuple[RecoveredLoopTopologyFact, ...]:
    if evidence is None:
        return ()
    exact = bool(getattr(evidence, "exact_expansion_validated", False))
    rows: list[RecoveredLoopTopologyFact] = []
    for witness in sorted(getattr(evidence, "witnesses", ()), key=lambda x: str(x.uid)):
        domains = []
        for d in sorted(getattr(witness, "coordinate_domains", ()), key=lambda x: (str(x.fullname), int(x.minimum), int(x.maximum))):
            domains.append((
                str(d.fullname), int(d.occurrence_count), int(d.minimum), int(d.maximum),
                int(d.first), None if d.step is None else int(d.step), str(d.binding_kind),
            ))
        rows.append(RecoveredLoopTopologyFact(
            uid=str(witness.uid),
            repetitions=int(witness.repetitions),
            phase_count=int(witness.phase_count),
            grammar_signature=str(witness.grammar_signature),
            formula_family_signature=str(witness.formula_family_signature),
            formula_fullnames=tuple(sorted(str(x) for x in witness.formula_fullnames)),
            affine_coordinate_steps=tuple(sorted((str(a), int(b)) for a, b in witness.affine_coordinate_steps)),
            coordinate_domains=tuple(domains),
            exact_expansion_validated=exact,
        ))
    return tuple(rows)


def _external_call_supported(call: ast.Call) -> bool:
    if call.keywords:
        return False
    if isinstance(call.func, ast.Name):
        return call.func.id in _PURE_EXTERNAL_CALLS
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return (
            call.func.value.id in {"math", "np", "numpy"}
            and call.func.attr in _PURE_MODULE_CALLS
            and len(call.args) == 1
        )
    return False


def _variant_nodes(cv: Any) -> list[ast.AST]:
    if getattr(cv, "role", "scalar") == "reduction" and getattr(cv, "reduction", None) is not None:
        r = cv.reduction
        return [r.init, r.body_expr, *r.filters, *r.range_args]
    fn = getattr(cv, "function", None)
    return [] if fn is None else list(fn.body)



def _tarjan_components(nodes: set[str], edges: set[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    adjacency = {uid: set() for uid in nodes}
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    out: list[tuple[str, ...]] = []

    def visit(uid: str) -> None:
        nonlocal index
        indices[uid] = low[uid] = index
        index += 1
        stack.append(uid)
        on_stack.add(uid)
        for child in sorted(adjacency.get(uid, ())):
            if child not in indices:
                visit(child)
                low[uid] = min(low[uid], low[child])
            elif child in on_stack:
                low[uid] = min(low[uid], indices[child])
        if low[uid] == indices[uid]:
            members: list[str] = []
            while True:
                child = stack.pop()
                on_stack.remove(child)
                members.append(child)
                if child == uid:
                    break
            out.append(tuple(sorted(members)))

    for uid in sorted(nodes):
        if uid not in indices:
            visit(uid)
    return tuple(sorted(out))


def _block_always_returns(stmts: list[ast.stmt], member_uids: set[str]) -> bool:
    """Conservative base-branch proof for a causal recurrence."""
    for stmt in stmts:
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in member_uids
            for child in ast.walk(stmt)
        ):
            return False
        if isinstance(stmt, (ast.Return, ast.Raise)):
            return True
        if isinstance(stmt, ast.If):
            if stmt.orelse and _block_always_returns(stmt.body, member_uids) and _block_always_returns(stmt.orelse, member_uids):
                return True
    return False


def _initial_halfline_base_proven(
    fn: ast.FunctionDef, member_uids: set[str], direction: str
) -> bool:
    if len(fn.args.args) != 1:
        return False
    variable = fn.args.args[0].arg
    # Require an initial dominating half-line guard.  This is deliberately narrow:
    # it proves termination for arbitrary requested coordinates without guessing a
    # model-specific lower/upper bound.
    for stmt in fn.body:
        if not isinstance(stmt, ast.If):
            # Harmless local assignments before the base guard are allowed only if
            # they do not touch the recurrence.
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if any(
                    isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id in member_uids
                    for child in ast.walk(stmt)
                ):
                    return False
                continue
            return False
        test = stmt.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
            return False
        left, right, op = test.left, test.comparators[0], test.ops[0]
        if not (isinstance(left, ast.Name) and left.id == variable and isinstance(right, ast.Constant)
                and isinstance(right.value, int) and not isinstance(right.value, bool)):
            return False
        guard_direction = (
            'ascending' if isinstance(op, (ast.Lt, ast.LtE)) else
            'descending' if isinstance(op, (ast.Gt, ast.GtE)) else None
        )
        return guard_direction == direction and _block_always_returns(stmt.body, member_uids)
    return False


def _refine_exact_state_semantics(
    nodes: list[CanonicalSemanticNode],
    accesses: list[CanonicalSemanticAccess],
    operators: list[CanonicalOperatorUse],
    pure_uids: set[str],
) -> tuple[list[CanonicalSemanticNode], list[CanonicalSemanticAccess], tuple[str, ...]]:
    """Strengthen conservative source semantics from exact specialized IR.

    Two proofs are intentionally separated:
    * a state-free causal recurrence may recurse on its own private coordinate but
      cannot read policy carried state;
    * ordinary exact values become state-free only after every canonical dependency
      is already proven state-free and every external operator is in the verified
      pure subset.

    This removes source-level false positives without promoting anything to policy
    persistent state.
    """
    by_uid = {x.uid: x for x in nodes}
    accesses_by_target: dict[str, list[CanonicalSemanticAccess]] = {uid: [] for uid in by_uid}
    for access in accesses:
        accesses_by_target.setdefault(access.target_uid, []).append(access)
    operators_by_target: dict[str, list[CanonicalOperatorUse]] = {uid: [] for uid in by_uid}
    for op in operators:
        operators_by_target.setdefault(op.target_uid, []).append(op)

    state_free = {uid for uid, node in by_uid.items() if node.state_semantic == 'state_free'}
    policy_state_independent = set(state_free)
    persistent = {uid for uid, node in by_uid.items() if node.state_semantic == 'persistent_state'}

    coordinate_nodes = {
        uid for uid, node in by_uid.items()
        if uid not in persistent and node.canonical_role == 'coordinate' and len(node.parameters) == 1
    }
    recursion_edges = {
        (access.target_uid, access.source_uid)
        for access in accesses
        if access.target_uid in coordinate_nodes and access.source_uid in coordinate_nodes
    }
    recurrence_uids: set[str] = set()
    changed = True
    while changed:
        changed = False
        for members_tuple in _tarjan_components(coordinate_nodes - recurrence_uids, recursion_edges):
            members = set(members_tuple)
            internal = [
                access for access in accesses
                if access.target_uid in members and access.source_uid in members
            ]
            cyclic = len(members) > 1 or any(x.target_uid == x.source_uid for x in internal)
            if not cyclic:
                continue
            requirements: set[str] = set()
            valid = True
            for access in internal:
                if access.min_offset is None or access.max_offset is None:
                    valid = False; break
                if access.max_offset < 0:
                    requirements.add('ascending')
                elif access.min_offset > 0:
                    requirements.add('descending')
                else:
                    valid = False; break
            if not valid or len(requirements) != 1:
                continue
            direction = next(iter(requirements))
            external = [
                access for access in accesses
                if access.target_uid in members and access.source_uid not in members
            ]
            if any(access.source_uid not in policy_state_independent for access in external):
                continue
            if any(
                not op.supported_by_pure_subset
                for uid in members for op in operators_by_target.get(uid, ())
            ):
                continue
            member_functions: dict[str, ast.FunctionDef] = {}
            for uid in members:
                try:
                    parsed = ast.parse(by_uid[uid].canonical_source)
                    fn = next(x for x in parsed.body if isinstance(x, ast.FunctionDef))
                except Exception:
                    valid = False; break
                member_functions[uid] = fn
            if not valid or any(
                not _initial_halfline_base_proven(member_functions[uid], members, direction)
                for uid in members
            ):
                continue
            recurrence_uids.update(members)
            policy_state_independent.update(members)
            changed = True

    # Exact state-free propagation.  This may strengthen source-level state-derived
    # or unknown classifications when specialization removed every stateful branch.
    changed = True
    while changed:
        changed = False
        for uid, node in sorted(by_uid.items()):
            if uid in policy_state_independent:
                continue
            if any(not op.supported_by_pure_subset for op in operators_by_target.get(uid, ())):
                continue
            deps = accesses_by_target.get(uid, ())
            if all(access.source_uid in policy_state_independent for access in deps):
                state_free.add(uid)
                policy_state_independent.add(uid)
                changed = True

    refined_nodes: list[CanonicalSemanticNode] = []
    for node in nodes:
        if node.uid in recurrence_uids:
            refined_nodes.append(replace(
                node, state_semantic='auxiliary_recurrence', execution_semantic='auxiliary_recurrence',
                semantic_blockers=tuple(
                    b for b in node.semantic_blockers
                    if b != 'source_backed_recursion' and not b.startswith('dependency_stateful:')
                ),
            ))
        elif node.uid in state_free:
            operators_supported = all(
                op.supported_by_pure_subset for op in operators_by_target.get(node.uid, ())
            )
            execution = (
                'pure_map' if node.uid in pure_uids else
                'value' if operators_supported else
                node.execution_semantic
            )
            refined_nodes.append(replace(
                node, state_semantic='state_free', execution_semantic=execution,
                semantic_blockers=tuple(
                    b for b in node.semantic_blockers
                    if b != 'source_backed_recursion' and not b.startswith('dependency_stateful:')
                ),
            ))
        else:
            refined_nodes.append(node)

    refined_by_uid = {x.uid: x for x in refined_nodes}
    refined_accesses: list[CanonicalSemanticAccess] = []
    for access in accesses:
        source = refined_by_uid[access.source_uid]
        if source.state_semantic in {'state_free', 'auxiliary_recurrence'}:
            refined_accesses.append(replace(
                access,
                relation='state_free' if source.canonical_role == 'coordinate' else 'scalar',
                # State-free edges are non-causal, but an already-proven affine
                # call offset remains useful execution-domain evidence.  Do not
                # erase it merely because the scheduler need not order the edge.
                min_offset=access.min_offset, max_offset=access.max_offset,
                scheduling_relevant=False,
                proof_kind=(
                    'canonical_state_free_causal_recurrence_v1'
                    if source.uid in recurrence_uids
                    else 'canonical_exact_state_free_refinement_v1'
                ),
                blockers=(),
            ))
        else:
            refined_accesses.append(access)
    return refined_nodes, refined_accesses, tuple(sorted(recurrence_uids))


_LOWERED_COORDINATE_REJECT_NODES = (
    ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match,
    ast.Delete, ast.AugAssign, ast.GeneratorExp, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr,
    # Supported terminal source raises must already have been converted to the
    # canonical __guard_fail__ ABI before this proof is attempted.
    ast.Raise,
)


def _lowered_coordinate_shape_blockers(fn: ast.FunctionDef | None) -> tuple[str, ...]:
    """Check only canonical lowered-program shape, never actuarial/source meaning."""
    if fn is None:
        return ("canonical_function_missing",)
    blockers: set[str] = set()
    if len(fn.args.args) != 1 or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
        blockers.add("coordinate_signature_not_unary")
    for child in ast.walk(fn):
        if isinstance(child, _LOWERED_COORDINATE_REJECT_NODES):
            blockers.add(f"canonical_control_unsupported:{type(child).__name__}")
        if isinstance(child, ast.Call) and child.keywords:
            blockers.add("canonical_keyword_call_unsupported")
        # Local scalar temporaries are allowed; object/container mutation is not.
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if any(isinstance(x, (ast.Attribute, ast.Subscript)) and isinstance(x.ctx, ast.Store)
                       for x in ast.walk(target)):
                    blockers.add("canonical_mutation_unsupported")
        elif isinstance(child, ast.AnnAssign):
            target = child.target
            if isinstance(target, (ast.Attribute, ast.Subscript)):
                blockers.add("canonical_mutation_unsupported")
    return tuple(sorted(blockers))


def _lowered_guard_operator_supported(op: CanonicalOperatorUse) -> bool:
    """Admit only the canonical guard ABI, not arbitrary unsupported calls."""
    if op.operator_kind != "call":
        return False
    try:
        expr = ast.parse(op.expression, mode="eval").body
    except Exception:
        return False
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "__guard_fail__"
        and len(expr.args) == 1
        and not expr.keywords
    )


def _prove_lowered_state_free_coordinates(
    nodes: list[CanonicalSemanticNode],
    accesses: list[CanonicalSemanticAccess],
    operators: list[CanonicalOperatorUse],
    pure_specs: tuple[Any, ...],
    variants: dict[str, Any],
) -> tuple[
    list[CanonicalSemanticNode],
    list[CanonicalSemanticAccess],
    tuple[StateFreeCoordinateExecutionProof, ...],
]:
    """Grant on-demand execution only from frozen canonical semantic evidence.

    ``state_free`` remains merely a scheduling fact.  Approval additionally
    requires source semantic provenance, a frozen unary coordinate domain, a
    lowered side-effect-free/control-safe canonical body, supported normalized
    operators, and an executable state-free dependency closure.
    """
    by_uid = {x.uid: x for x in nodes}
    accesses_by_target: dict[str, list[CanonicalSemanticAccess]] = {uid: [] for uid in by_uid}
    for access in accesses:
        accesses_by_target.setdefault(access.target_uid, []).append(access)
    operators_by_target: dict[str, list[CanonicalOperatorUse]] = {uid: [] for uid in by_uid}
    for op in operators:
        operators_by_target.setdefault(op.target_uid, []).append(op)
    old_specs = {str(x.uid): x for x in pure_specs}
    uses_by_source: dict[str, list[CanonicalSemanticAccess]] = {uid: [] for uid in by_uid}
    for access in accesses:
        uses_by_source.setdefault(access.source_uid, []).append(access)

    candidates = {
        uid for uid, node in by_uid.items()
        if node.canonical_role == "coordinate"
        and node.state_semantic == "state_free"
        and node.execution_semantic == "unknown"
    }
    approved: set[str] = set()
    proofs: dict[str, StateFreeCoordinateExecutionProof] = {}
    active: set[str] = set()
    scalar_closure: dict[str, set[str]] = {}
    pure_closure: dict[str, set[str]] = {}

    def callable_domain(
        uid: str,
    ) -> tuple[tuple[int, int] | None, str, tuple[str, ...], tuple[str, ...]]:
        node = by_uid[uid]
        if node.domain is not None:
            return node.domain, "canonical_coordinate_domain_v1", (), ()
        bounds: list[tuple[int, int]] = []
        blockers: set[str] = set()
        evidence: set[str] = set()
        symbolic_forwarding = False
        uses = uses_by_source.get(uid, ())
        if not uses:
            return None, "canonical_coordinate_domain_unproved_v1", (), ("coordinate_call_domain_unproved:no_callsites",)
        for access in uses:
            caller = by_uid[access.target_uid]
            if access.min_offset is None or access.max_offset is None:
                blockers.add(f"coordinate_call_domain_unproved:{access.uid}")
                continue
            if caller.domain is not None:
                bounds.append((
                    int(caller.domain[0]) + int(access.min_offset),
                    int(caller.domain[1]) + int(access.max_offset),
                ))
                evidence.add(access.uid)
                continue
            if (
                caller.canonical_role == "coordinate"
                and caller.state_semantic in {"state_free", "auxiliary_recurrence"}
                and caller.execution_semantic in {"value", "pure_map", "auxiliary_recurrence"}
            ):
                symbolic_forwarding = True
                evidence.add(access.uid)
                continue
            blockers.add(f"coordinate_call_domain_unproved:{access.uid}")
        if blockers or not bounds:
            if blockers:
                return None, "canonical_coordinate_domain_unproved_v1", tuple(sorted(evidence)), tuple(sorted(blockers))
            if symbolic_forwarding:
                return (
                    None,
                    "canonical_forwarded_coordinate_contract_v1",
                    tuple(sorted(evidence)),
                    (),
                )
            return None, "canonical_coordinate_domain_unproved_v1", tuple(sorted(evidence)), ("coordinate_call_domain_unproved",)
        domain = (min(lo for lo, _ in bounds), max(hi for _, hi in bounds))
        return (
            domain,
            (
                "canonical_callsite_domain_contract_v1"
                if symbolic_forwarding
                else "canonical_callsite_domain_envelope_v1"
            ),
            tuple(sorted(evidence)),
            (),
        )

    def attempt(uid: str) -> bool:
        if uid in proofs:
            return proofs[uid].approved
        node = by_uid[uid]
        blockers: set[str] = set()
        direct_deps = tuple(sorted({x.source_uid for x in accesses_by_target.get(uid, ())}))
        access_uids = tuple(sorted(x.uid for x in accesses_by_target.get(uid, ())))
        operator_uids = tuple(sorted(x.uid for x in operators_by_target.get(uid, ())))

        if uid in active:
            blockers.add("canonical_coordinate_dependency_cycle")
        if node.source_semantic_class != "pure_map":
            blockers.add(f"source_semantic_not_pure_map:{node.source_semantic_class}")
        if node.source_state_dependency != "static_only":
            blockers.add(f"source_state_dependency_not_static_only:{node.source_state_dependency}")
        if node.semantic_blockers:
            blockers.update(f"source_semantic_blocker:{x}" for x in node.semantic_blockers)
        if len(node.parameters) != 1:
            blockers.add("coordinate_signature_not_unary")
        proven_domain, domain_proof_kind, domain_evidence_uids, domain_blockers = callable_domain(uid)
        blockers.update(domain_blockers)
        fn = getattr(variants.get(uid), "function", None)
        blockers.update(_lowered_coordinate_shape_blockers(fn))
        for op in operators_by_target.get(uid, ()):
            if not op.supported_by_pure_subset and not _lowered_guard_operator_supported(op):
                blockers.add(f"canonical_operator_unsupported:{op.uid}")
        for access in accesses_by_target.get(uid, ()):
            if access.scheduling_relevant:
                blockers.add(f"scheduling_relevant_dependency:{access.uid}")
            if access.blockers:
                blockers.update(f"dependency_access_blocker:{access.uid}:{x}" for x in access.blockers)

        direct_scalars: set[str] = set()
        direct_pure: set[str] = set()
        if not blockers:
            active.add(uid)
            try:
                for dep_uid in direct_deps:
                    dep = by_uid[dep_uid]
                    if dep.state_semantic != "state_free":
                        blockers.add(f"dependency_not_state_free:{dep_uid}")
                        continue
                    if dep.canonical_role == "scalar" and dep.execution_semantic in {"value", "pure_map"}:
                        direct_scalars.add(dep_uid)
                        old = old_specs.get(dep_uid)
                        if old is not None:
                            direct_scalars.update(str(x) for x in old.scalar_dependencies)
                            direct_pure.update(str(x) for x in old.pure_map_dependencies)
                        continue
                    if dep.canonical_role == "coordinate":
                        if dep.execution_semantic == "pure_map":
                            direct_pure.add(dep_uid)
                            old = old_specs.get(dep_uid)
                            if old is not None:
                                direct_scalars.update(str(x) for x in old.scalar_dependencies)
                                direct_pure.update(str(x) for x in old.pure_map_dependencies)
                            continue
                        if dep_uid in candidates and attempt(dep_uid):
                            direct_pure.add(dep_uid)
                            direct_scalars.update(scalar_closure.get(dep_uid, ()))
                            direct_pure.update(pure_closure.get(dep_uid, ()))
                            continue
                        blockers.add(f"dependency_execution_unapproved:{dep_uid}")
                        continue
                    blockers.add(f"dependency_role_unsupported:{dep_uid}:{dep.canonical_role}")
            finally:
                active.discard(uid)

        ok = not blockers
        if ok:
            scalar_closure[uid] = set(direct_scalars)
            pure_closure[uid] = set(direct_pure)
            approved.add(uid)
        proof = StateFreeCoordinateExecutionProof(
            proof_uid=_stable_id(
                "canonical_state_free_execution_proof",
                (
                    uid, node.parameters, proven_domain, domain_proof_kind,
                    domain_evidence_uids, tuple(sorted(blockers)), direct_deps, operator_uids,
                ),
            ),
            value_uid=uid,
            coordinate_parameter=(node.parameters[0] if len(node.parameters) == 1 else ""),
            coordinate_domain=proven_domain,
            coordinate_domain_proof_kind=domain_proof_kind,
            coordinate_domain_evidence_uids=domain_evidence_uids,
            direct_dependency_uids=direct_deps,
            scalar_dependencies=tuple(sorted(direct_scalars)),
            pure_map_dependencies=tuple(sorted(direct_pure)),
            access_uids=access_uids,
            operator_uids=operator_uids,
            approved=ok,
            blockers=tuple(sorted(blockers)),
        )
        proofs[uid] = proof
        return ok

    for uid in sorted(candidates):
        attempt(uid)

    if not approved:
        return nodes, accesses, tuple(proofs[uid] for uid in sorted(proofs))

    promoted_nodes = [
        replace(node, execution_semantic="pure_map") if node.uid in approved else node
        for node in nodes
    ]
    promoted_accesses = [
        replace(
            access,
            proof_kind="canonical_lowered_coordinate_pure_v1",
            blockers=(),
            relation="state_free" if by_uid[access.source_uid].canonical_role == "coordinate" else access.relation,
            scheduling_relevant=False,
        ) if access.source_uid in approved else access
        for access in accesses
    ]
    return (
        promoted_nodes,
        promoted_accesses,
        tuple(proofs[uid] for uid in sorted(proofs)),
    )

def build_canonical_semantic_graph(snapshot: Any) -> CanonicalSemanticGraph:
    """Build a lossless exact-UID semantic graph from a canonical proof snapshot.

    This function intentionally has no scheduling exceptions.  A relation that
    cannot be proved is emitted as ``relation='unknown'`` with blockers; an
    unsupported operator is emitted as an operator-use row.  The caller may later
    decide that such evidence blocks a schedule, but graph construction itself does
    not discard the already-lowered canonical program.
    """
    variants = dict(getattr(snapshot, "variants", {}))
    known = set(variants)
    domains = {str(uid): (int(lo), int(hi)) for uid, lo, hi in getattr(snapshot, "coordinate_domains", ())}
    source_rows = {str(x.uid): x for x in getattr(snapshot, "source_value_semantics", ())}
    pure_specs = tuple(sorted(getattr(snapshot, "pure_map_specs", ()), key=lambda x: str(x.uid)))
    pure_uids = {str(x.uid) for x in pure_specs}
    recurrence_domain_facts = tuple(
        CanonicalRecurrenceDomainFact(
            uid=_stable_id(
                "canonical_recurrence_domain",
                (
                    str(x.source_uid), int(x.demand_lo), int(x.demand_hi),
                    int(x.active_lo), int(x.active_hi), tuple(x.recurrence_offsets),
                    tuple(x.boundary_coordinates), str(x.scan_requirement),
                    str(x.output_invocation_uid), str(x.proof_kind), tuple(x.blockers),
                ),
            ),
            value_uid=str(x.source_uid),
            coordinate_parameter=str(x.coordinate_parameter),
            demand_domain=(int(x.demand_lo), int(x.demand_hi)),
            active_domain=(int(x.active_lo), int(x.active_hi)),
            recurrence_offsets=tuple(int(v) for v in x.recurrence_offsets),
            boundary_coordinates=tuple(int(v) for v in x.boundary_coordinates),
            scan_requirement=str(x.scan_requirement),
            output_invocation_uid=str(x.output_invocation_uid),
            proof_kind=str(x.proof_kind),
            blockers=tuple(str(v) for v in x.blockers),
        )
        for x in sorted(
            getattr(snapshot, "coordinate_recurrence_facts", ()),
            key=lambda row: (str(row.source_uid), tuple(row.recurrence_offsets)),
        )
        if str(x.source_uid) in variants
    )

    nodes: list[CanonicalSemanticNode] = []
    execution_by_uid: dict[str, str] = {}
    state_by_uid: dict[str, str] = {}
    for uid, cv in sorted(variants.items()):
        source = source_rows.get(str(getattr(cv, "source_name", "")))
        sem, dep, persistence, proof, traits, blockers = _source_semantics_tuple(source)
        role = str(getattr(cv, "role", "scalar"))
        if sem == "pure_map" or dep == "static_only":
            state_semantic = "state_free"
        elif (sem == "persistent_state" or persistence == "persistent") and role in {"coordinate", "vector"}:
            state_semantic = "persistent_state"
        elif sem == "state_derived_map" or dep == "reads_state":
            state_semantic = "state_derived_map"
        else:
            state_semantic = "unknown"

        if uid in pure_uids:
            execution = "pure_map"
        elif state_semantic == "persistent_state":
            execution = "scheduled_state"
        elif state_semantic == "state_derived_map":
            execution = "derived_task"
        elif role == "coordinate" and state_semantic == "state_free":
            # State independence is sufficient to remove a causal edge but not to
            # authorize an arbitrary-coordinate helper ABI.  WP currently lands
            # here: scheduling state-free, exact execution still unapproved.
            execution = "unknown"
        elif role in {"scalar", "reduction", "vector"}:
            execution = "value"
        else:
            execution = "unknown"
        fn = getattr(cv, "function", None)
        params = tuple(a.arg for a in fn.args.args) if fn is not None else ()
        if fn is not None:
            canonical_ast = ast.dump(fn, include_attributes=False)
            canonical_source = ast.unparse(fn)
        elif getattr(cv, "reduction", None) is not None:
            r = cv.reduction
            canonical_ast = "ReductionSpec(" + ";".join([
                ast.dump(r.init, include_attributes=False),
                str(r.target), str(r.loop_var),
                *(ast.dump(x, include_attributes=False) for x in r.range_args),
                ast.dump(r.body_expr, include_attributes=False),
                *(ast.dump(x, include_attributes=False) for x in r.filters),
            ]) + ")"
            canonical_source = canonical_ast
        else:
            canonical_ast = ""
            canonical_source = ""
        execution_by_uid[uid] = execution
        state_by_uid[uid] = state_semantic
        nodes.append(CanonicalSemanticNode(
            uid=uid,
            source_name=str(getattr(cv, "source_name", "")),
            source_fullname=str(getattr(cv, "source_fullname", uid)),
            canonical_role=role,
            dtype=str(getattr(cv, "dtype", "unknown")),
            parameters=params,
            domain=domains.get(uid),
            canonical_ast=canonical_ast,
            canonical_source=canonical_source,
            state_semantic=state_semantic,
            execution_semantic=execution,
            source_semantic_class=sem,
            source_state_dependency=dep,
            source_persistence=persistence,
            source_proof_kind=proof,
            operator_traits=traits,
            semantic_blockers=blockers,
        ))

    accesses: list[CanonicalSemanticAccess] = []
    operators: list[CanonicalOperatorUse] = []
    reject_controls = (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match,
                       ast.Delete, ast.AugAssign, ast.GeneratorExp, ast.ListComp, ast.SetComp,
                       ast.DictComp, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)

    for target_uid, cv in sorted(variants.items()):
        role = str(getattr(cv, "role", "scalar"))
        fn = getattr(cv, "function", None)
        reduction = getattr(cv, "reduction", None)
        if role == "coordinate" and fn is not None and len(fn.args.args) == 1:
            target_axis = fn.args.args[0].arg
        elif role == "reduction" and reduction is not None:
            target_axis = str(reduction.loop_var)
        else:
            target_axis = None
        context = "reduction" if role == "reduction" else role

        for root in _variant_nodes(cv):
            for node in ast.walk(root):
                if isinstance(node, reject_controls):
                    text = _expr(node) or type(node).__name__
                    operators.append(CanonicalOperatorUse(
                        uid=_stable_id("canonical_operator", (target_uid, type(node).__name__, text, getattr(node, "lineno", -1), getattr(node, "col_offset", -1))),
                        target_uid=target_uid,
                        operator_kind="control",
                        expression=text,
                        supported_by_pure_subset=False,
                        lineno=int(getattr(node, "lineno", -1)),
                        col_offset=int(getattr(node, "col_offset", -1)),
                        note="preserved unsupported/complex control construct; no execution permission",
                    ))
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in known:
                    source_uid = node.func.id
                    arg_exprs = tuple(_expr(x) or "<unparse-failed>" for x in node.args)
                    arg_asts = tuple(ast.dump(x, include_attributes=False) for x in node.args)
                    keyword_exprs = tuple(
                        (("**" if kw.arg is None else str(kw.arg)), _expr(kw.value) or "<unparse-failed>")
                        for kw in node.keywords
                    )
                    call_ast = ast.dump(node, include_attributes=False)
                    source_role = str(getattr(variants[source_uid], "role", "scalar"))
                    source_exec = execution_by_uid.get(source_uid, "unknown")
                    source_state = state_by_uid.get(source_uid, "unknown")
                    relation = "unknown"
                    min_offset = max_offset = None
                    fixed_coordinate = None
                    proof_kind = "canonical_unproved_relation_v1"
                    blockers: list[str] = []
                    scheduling_relevant = source_state != "state_free"
                    if source_exec == "pure_map":
                        relation = "state_free"
                        proof_kind = "executable_pure_map_v1"
                        scheduling_relevant = False
                    elif source_state == "state_free":
                        relation = "state_free" if source_role == "coordinate" else "scalar"
                        proof_kind = "source_state_free_semantics_v1"
                        scheduling_relevant = False
                        if source_role == "coordinate" and len(node.args) == 1 and target_axis is not None:
                            off = _affine_offset(node.args[0], target_axis)
                            if off is not None:
                                min_offset = max_offset = int(off)
                                proof_kind = "canonical_state_free_affine_call_v1"
                        if source_role == "coordinate" and source_exec != "pure_map":
                            blockers.append("state_free_but_execution_operator_unapproved")
                    elif source_role == "scalar":
                        relation = "unknown"
                        proof_kind = "state_dependent_scalar_relation_unproved_v1"
                        blockers.append("state_dependent_scalar_requires_derived_footprint")
                    elif source_role == "coordinate":
                        if len(node.args) != 1:
                            blockers.append("coordinate_arity_not_one")
                        elif target_axis is None:
                            fixed = _constant_int_expr(node.args[0])
                            if target_uid == str(snapshot.output_uid) and fixed is not None:
                                relation = "fixed"
                                fixed_coordinate = int(fixed)
                                proof_kind = "canonical_fixed_output_coordinate_v1"
                            else:
                                blockers.append("target_has_no_iteration_axis")
                        else:
                            off = _affine_offset(node.args[0], target_axis)
                            if off is None:
                                blockers.append("non_affine_or_unproved_coordinate_relation")
                            else:
                                min_offset = max_offset = int(off)
                                relation = "same" if off == 0 else "strict_before" if off < 0 else "strict_after"
                                proof_kind = "canonical_affine_offset_v1"
                    else:
                        relation = "scalar"
                        proof_kind = "canonical_noncoordinate_dependency_v1"
                        scheduling_relevant = False
                    accesses.append(CanonicalSemanticAccess(
                        uid=_stable_id("canonical_access", (
                            source_uid, target_uid, context, arg_asts, keyword_exprs, relation,
                            getattr(node, "lineno", -1), getattr(node, "col_offset", -1),
                        )),
                        source_uid=source_uid,
                        target_uid=target_uid,
                        context=context,
                        argument_exprs=arg_exprs,
                        argument_asts=arg_asts,
                        keyword_exprs=keyword_exprs,
                        call_ast=call_ast,
                        relation=relation,
                        min_offset=min_offset,
                        max_offset=max_offset,
                        scheduling_relevant=scheduling_relevant,
                        proof_kind=proof_kind,
                        fixed_coordinate=fixed_coordinate,
                        blockers=tuple(blockers),
                        lineno=int(getattr(node, "lineno", -1)),
                        col_offset=int(getattr(node, "col_offset", -1)),
                    ))
                else:
                    text = _expr(node) or ast.dump(node, include_attributes=False)
                    operators.append(CanonicalOperatorUse(
                        uid=_stable_id("canonical_operator", (target_uid, text, getattr(node, "lineno", -1), getattr(node, "col_offset", -1))),
                        target_uid=target_uid,
                        operator_kind="call",
                        expression=text,
                        supported_by_pure_subset=_external_call_supported(node),
                        lineno=int(getattr(node, "lineno", -1)),
                        col_offset=int(getattr(node, "col_offset", -1)),
                        note="external/operator call preserved independently of scheduling",
                    ))

    # ast.walk over nested controls can surface the same control node from more than
    # one enclosing root only for malformed/reused ASTs.  Deduplicate anyway so the
    # graph manifest is a deterministic evidence object.
    access_map = {x.uid: x for x in accesses}
    operator_map = {x.uid: x for x in operators}
    nodes, refined_accesses, auxiliary_recurrence_uids = _refine_exact_state_semantics(
        nodes, list(access_map.values()), list(operator_map.values()), pure_uids
    )
    nodes, refined_accesses, state_free_execution_proofs = _prove_lowered_state_free_coordinates(
        nodes,
        list(refined_accesses),
        list(operator_map.values()),
        pure_specs,
        variants,
    )
    access_map = {x.uid: x for x in refined_accesses}
    graph = CanonicalSemanticGraph(
        schema="modelx_graph.canonical_semantic_graph.v5",
        output_uid=str(snapshot.output_uid),
        run_key_count=int(snapshot.run_key_count),
        formula_hash=str(getattr(snapshot, "formula_hash", "")),
        structural_hash=str(getattr(snapshot, "structural_hash", "")),
        nodes=tuple(nodes),
        accesses=tuple(sorted(access_map.values(), key=lambda x: x.uid)),
        operators=tuple(sorted(operator_map.values(), key=lambda x: x.uid)),
        loop_topology_facts=tuple(getattr(snapshot, "loop_topology_facts", ())),
        recovered_loop_topology_facts=tuple(getattr(snapshot, "recovered_loop_topology_facts", ())),
        pure_map_specs=pure_specs,
        scheduler_evidence_status=str(getattr(snapshot, "scheduler_evidence_status", "unknown")),
        auxiliary_recurrence_uids=auxiliary_recurrence_uids,
        state_free_execution_proofs=state_free_execution_proofs,
        recurrence_domain_facts=recurrence_domain_facts,
        validation_notes=(
            "canonical semantic graph is the production pre-scheduler authority for Stage-supported compilation",
            "unknown relations/operators are retained as evidence rather than rejected during graph construction",
            "exact causal helper recurrences may be proven state-free without becoming policy persistent state",
            "lowered state-free coordinate execution requires a distinct canonical proof; state_free alone is not permission",
            "fixed scalar OutputInvocation coordinate calls retain absolute fixed-coordinate evidence without inventing a caller axis",
            "scheduling-relevant source/AST semantics are frozen before CanonicalComponentSchedule analysis",
        ),
    )
    formula_schemas = tuple(
        FormulaSchema(
            uid=node.uid, canonical_role=node.canonical_role, dtype=node.dtype,
            parameters=node.parameters, state_semantic=node.state_semantic,
            execution_semantic=node.execution_semantic,
            source_proof_kind=node.source_proof_kind,
            operator_traits=node.operator_traits, blockers=node.semantic_blockers,
        )
        for node in sorted(graph.nodes, key=lambda x: x.uid)
    )
    domain_facts = tuple(
        DomainFact(
            uid=_stable_id("canonical_domain_fact", (node.uid, node.parameters, node.domain, node.canonical_role)),
            value_uid=node.uid,
            axis=(
                node.parameters[0] if len(node.parameters) == 1
                else "scalar" if node.canonical_role == "scalar"
                else "canonical"
            ),
            kind=(
                "integer_interval" if node.domain is not None
                else "scalar" if node.canonical_role == "scalar"
                else "unknown"
            ),
            lo=(None if node.domain is None else int(node.domain[0])),
            hi=(None if node.domain is None else int(node.domain[1])),
            proof_kind=(
                "canonical_coordinate_domain_v1" if node.domain is not None
                else "canonical_scalar_domain_v1" if node.canonical_role == "scalar"
                else "canonical_domain_unproved_v1"
            ),
            blockers=(
                () if node.domain is not None or node.canonical_role == "scalar"
                else ("canonical_domain_unproved",)
            ),
        )
        for node in sorted(graph.nodes, key=lambda x: x.uid)
    )
    coordinate_relations = tuple(
        CoordinateRelation(
            uid=_stable_id("canonical_coordinate_relation", (access.uid, access.relation, access.min_offset, access.max_offset)),
            access_uid=access.uid, source_uid=access.source_uid, target_uid=access.target_uid,
            relation=access.relation, min_offset=access.min_offset, max_offset=access.max_offset,
            scheduling_relevant=access.scheduling_relevant, proof_kind=access.proof_kind,
            blockers=access.blockers,
        )
        for access in sorted(graph.accesses, key=lambda x: x.uid)
    )
    semantic_operators = tuple(
        SemanticOperator(
            uid=_stable_id("canonical_semantic_operator", (op.uid, op.operator_kind, op.expression)),
            target_uid=op.target_uid, operator_kind=op.operator_kind, expression=op.expression,
            supported=op.supported_by_pure_subset,
            proof_kind="canonical_operator_classification_v1", note=op.note,
        )
        for op in sorted(graph.operators, key=lambda x: x.uid)
    )
    graph = replace(
        graph,
        formula_schemas=formula_schemas,
        domain_facts=domain_facts,
        coordinate_relations=coordinate_relations,
        semantic_operators=semantic_operators,
    )

    # Local import avoids a module cycle while keeping ownership explicit: AST/source
    # analysis ends here, before the graph scheduler sees the program.
    from .canonical_semantic_analysis import build_canonical_semantic_evidence

    footprints, transitions = build_canonical_semantic_evidence(
        graph,
        functions={
            uid: cv.function
            for uid, cv in variants.items()
            if getattr(cv, "function", None) is not None
        },
    )
    return replace(
        graph,
        state_footprints=footprints,
        transition_evidence=transitions,
        semantic_analysis_schema="modelx_graph.canonical_semantic_evidence.v1",
    )
