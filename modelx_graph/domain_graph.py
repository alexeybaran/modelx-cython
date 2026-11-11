from __future__ import annotations

"""Shadow domain-relational graph IR.

This module is intentionally evidence-only in the first foundation.  It does not
participate in storage planning or code generation.  Its job is to express the
current executable graph, including today's derived-clock/history exceptions, in
one smaller vocabulary of iteration domains, kernels, access relations and
initial conditions.  It also contains source-only candidate scanners for graph
shapes that are not executable yet.

The shadow IR must never be treated as permission to execute an unsupported
relation.  Existing :mod:`template_ir` remains authoritative until an explicit
migration step proves equivalence and switches consumers over.
"""

import ast
import hashlib
import itertools
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .local_control_semantics import recognize_first_hit_range_stmt

from .coordinate_relation import (
    AffineAliasRead,
    AffineOffset,
    DerivedClockRead,
    DerivedHistoryRead,
)


class DomainGraphError(RuntimeError):
    """Fail-closed shadow-IR construction error."""


def _stable_id(prefix: str, parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:14]}"


def _expr(node: ast.AST | None) -> str | None:
    return None if node is None else ast.unparse(node)


@dataclass(frozen=True)
class DomainAxis:
    name: str
    lower_expr: str | None = None
    upper_expr: str | None = None
    step: int | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lower_expr": self.lower_expr,
            "upper_expr": self.upper_expr,
            "step": None if self.step is None else int(self.step),
        }


@dataclass(frozen=True)
class IterationDomain:
    uid: str
    kind: str  # scalar | ordered | event | indexed
    axes: tuple[DomainAxis, ...]
    order: str = "none"  # none | ascending | descending | mapped
    parent_uid: str | None = None
    mapping_expr: str | None = None
    proof_kind: str | None = None
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "kind": self.kind,
            "axes": [x.manifest() for x in self.axes],
            "order": self.order,
            "parent_uid": self.parent_uid,
            "mapping_expr": self.mapping_expr,
            "proof_kind": self.proof_kind,
            "note": self.note,
        }


@dataclass(frozen=True)
class Kernel:
    uid: str
    domain_uid: str
    member_uids: tuple[str, ...]
    kind: str  # primary | event | scalar | reduction
    persistent_uids: tuple[str, ...] = ()
    legacy_roles: tuple[tuple[str, str], ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "domain_uid": self.domain_uid,
            "member_uids": list(self.member_uids),
            "kind": self.kind,
            "persistent_uids": list(self.persistent_uids),
            "legacy_roles": [list(x) for x in self.legacy_roles],
            "note": self.note,
        }


@dataclass(frozen=True)
class AccessRelation:
    uid: str
    source_uid: str
    target_uid: str
    source_domain_uid: str
    target_domain_uid: str
    selector: str  # value | point | window | reduction
    index_expr: str | None
    order: str  # same | strict_before | before_or_same | strict_after | mixed | unknown
    min_lag: int | None = None
    max_lag: int | None = None
    min_offset: int | None = None
    max_offset: int | None = None
    proof_kind: str | None = None
    legacy_kind: str | None = None
    scheduling_relevant: bool = True
    scheduling_reason: str = "unclassified"
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_uid": self.source_uid,
            "target_uid": self.target_uid,
            "source_domain_uid": self.source_domain_uid,
            "target_domain_uid": self.target_domain_uid,
            "selector": self.selector,
            "index_expr": self.index_expr,
            "order": self.order,
            "min_lag": self.min_lag,
            "max_lag": self.max_lag,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "proof_kind": self.proof_kind,
            "legacy_kind": self.legacy_kind,
            "scheduling_relevant": self.scheduling_relevant,
            "scheduling_reason": self.scheduling_reason,
            "note": self.note,
        }


@dataclass(frozen=True)
class InitialCondition:
    uid: str
    state_uid: str
    domain_uid: str
    coordinate_expr: str
    kind: str
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "state_uid": self.state_uid,
            "domain_uid": self.domain_uid,
            "coordinate_expr": self.coordinate_expr,
            "kind": self.kind,
            "note": self.note,
        }


@dataclass(frozen=True)
class PurityEvidence:
    uid: str
    classification: str  # pure_map | stateful | unknown
    proof_kind: str
    dependencies: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    immutable_sources: tuple[str, ...] = ()
    note: str = ""

    @property
    def is_pure(self) -> bool:
        return self.classification == "pure_map"

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "classification": self.classification,
            "proof_kind": self.proof_kind,
            "dependencies": list(self.dependencies),
            "blockers": list(self.blockers),
            "immutable_sources": list(self.immutable_sources),
            "note": self.note,
        }


@dataclass(frozen=True)
class ValueSemanticsEvidence:
    """Proof-only source value semantics below the coarse purity split.

    ``state_dependency`` and ``persistence`` are intentionally orthogonal.  A
    coordinate function may read persistent state without itself being a carried
    state value.  Such a function is a ``state_derived_map`` rather than another
    member of the recurrence state vector.

    This evidence never grants execution permission.  In particular, operator
    traits such as a local loop, mutation, window reduction or dynamic coordinate
    remain explicit even when persistence can still be classified.
    """

    uid: str
    semantic_class: str  # pure_map | state_derived_map | persistent_state | unknown
    state_dependency: str  # static_only | reads_state | unknown
    persistence: str  # ephemeral | persistent | unknown
    proof_kind: str
    dependencies: tuple[str, ...] = ()
    recurrence_access_uids: tuple[str, ...] = ()
    operator_traits: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    note: str = ""

    @property
    def is_persistent(self) -> bool:
        return self.persistence == "persistent"

    @property
    def is_state_derived(self) -> bool:
        return self.semantic_class == "state_derived_map"

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "semantic_class": self.semantic_class,
            "state_dependency": self.state_dependency,
            "persistence": self.persistence,
            "proof_kind": self.proof_kind,
            "dependencies": list(self.dependencies),
            "recurrence_access_uids": list(self.recurrence_access_uids),
            "operator_traits": list(self.operator_traits),
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class DomainGraph:
    schema: str
    domains: tuple[IterationDomain, ...]
    kernels: tuple[Kernel, ...]
    accesses: tuple[AccessRelation, ...]
    initial_conditions: tuple[InitialCondition, ...]
    uid_domain: tuple[tuple[str, str], ...]
    legacy_summary: tuple[tuple[str, int], ...]
    purity_evidence: tuple[PurityEvidence, ...] = ()
    validation_notes: tuple[str, ...] = ()

    def domain_for_uid(self, uid: str) -> str | None:
        return dict(self.uid_domain).get(uid)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "domains": [x.manifest() for x in self.domains],
            "kernels": [x.manifest() for x in self.kernels],
            "accesses": [x.manifest() for x in self.accesses],
            "initial_conditions": [x.manifest() for x in self.initial_conditions],
            "uid_domain": dict(self.uid_domain),
            "legacy_summary": dict(self.legacy_summary),
            "purity_evidence": [x.manifest() for x in self.purity_evidence],
            "validation_notes": list(self.validation_notes),
        }


@dataclass(frozen=True)
class SourceGraphCandidate:
    kind: str
    function: str
    payload: tuple[tuple[str, Any], ...]

    def manifest(self) -> dict[str, Any]:
        return {"kind": self.kind, "function": self.function, **dict(self.payload)}


@dataclass(frozen=True)
class SourceAccessEvidence:
    uid: str
    source_uid: str
    target_uid: str
    selector: str  # point | window
    index_expr: str | None
    order: str  # same | strict_before | strict_after | fixed | window | unknown
    source_classification: str
    scheduling_relevant: bool
    scheduling_reason: str
    min_offset: int | None = None
    max_offset: int | None = None
    proof_kind: str | None = None
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_uid": self.source_uid,
            "target_uid": self.target_uid,
            "selector": self.selector,
            "index_expr": self.index_expr,
            "order": self.order,
            "source_classification": self.source_classification,
            "scheduling_relevant": self.scheduling_relevant,
            "scheduling_reason": self.scheduling_reason,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "proof_kind": self.proof_kind,
            "note": self.note,
        }


@dataclass(frozen=True)
class SourcePersistentComponent:
    uid: str
    persistent_uids: tuple[str, ...]
    contracted_edges: tuple[tuple[str, str], ...]
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "persistent_uids": list(self.persistent_uids),
            "contracted_edges": [list(x) for x in self.contracted_edges],
            "note": self.note,
        }


@dataclass(frozen=True)
class SourceStateStructure:
    schema: str
    axis: str
    semantics: tuple[ValueSemanticsEvidence, ...]
    accesses: tuple[SourceAccessEvidence, ...]
    raw_state_sccs: tuple[tuple[str, ...], ...]
    persistent_components: tuple[SourcePersistentComponent, ...]
    validation_notes: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        raw_nontrivial = [list(x) for x in self.raw_state_sccs if len(x) > 1]
        return {
            "schema": self.schema,
            "axis": self.axis,
            "semantics": [x.manifest() for x in self.semantics],
            "accesses": [x.manifest() for x in self.accesses],
            "raw_state_sccs": [list(x) for x in self.raw_state_sccs],
            "persistent_components": [x.manifest() for x in self.persistent_components],
            "metrics": {
                "raw_nontrivial_scc_count": len(raw_nontrivial),
                "raw_largest_scc_size": max((len(x) for x in self.raw_state_sccs), default=0),
                "persistent_value_count": sum(
                    1 for x in self.semantics if x.persistence == "persistent"
                ),
                "state_derived_value_count": sum(
                    1 for x in self.semantics if x.semantic_class == "state_derived_map"
                ),
                "persistent_component_count": len(self.persistent_components),
                "persistent_largest_component_size": max(
                    (len(x.persistent_uids) for x in self.persistent_components),
                    default=0,
                ),
            },
            "validation_notes": list(self.validation_notes),
        }


@dataclass(frozen=True)
class SourceCallSiteAvailability:
    """Proof-only availability requirement attached to one concrete source call site.

    ``root_target_uid`` is the function whose concrete call initiated the proof.
    ``target_uid`` / ``source_uid`` identify the immediate edge represented by this
    row.  ``call_path`` records any state-derived expansion used to reach the edge,
    so the same helper can carry different evidence at different callers/subdomains.
    """

    uid: str
    callsite_uid: str
    root_callsite_uid: str
    root_target_uid: str
    target_uid: str
    source_uid: str
    call_path: tuple[str, ...]
    selector: str  # point | window
    index_expr: str | None
    lineno: int
    col_offset: int
    guard_exprs: tuple[str, ...]
    target_subdomain: str
    source_semantic_class: str
    target_semantic_class: str
    transition_relevant: bool
    transition_sink_uids: tuple[str, ...]
    availability: str  # STATIC | CURRENT | PREFIX | SUFFIX | COMPLETE | UNKNOWN
    relation_min_offset: int | None = None
    relation_max_offset: int | None = None
    min_offset: int | None = None
    max_offset: int | None = None
    proof_kind: str | None = None
    persistent_source_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "callsite_uid": self.callsite_uid,
            "root_callsite_uid": self.root_callsite_uid,
            "root_target_uid": self.root_target_uid,
            "target_uid": self.target_uid,
            "source_uid": self.source_uid,
            "call_path": list(self.call_path),
            "selector": self.selector,
            "index_expr": self.index_expr,
            "lineno": int(self.lineno),
            "col_offset": int(self.col_offset),
            "guard_exprs": list(self.guard_exprs),
            "target_subdomain": self.target_subdomain,
            "source_semantic_class": self.source_semantic_class,
            "target_semantic_class": self.target_semantic_class,
            "transition_relevant": bool(self.transition_relevant),
            "transition_sink_uids": list(self.transition_sink_uids),
            "availability": self.availability,
            "relation_min_offset": self.relation_min_offset,
            "relation_max_offset": self.relation_max_offset,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "proof_kind": self.proof_kind,
            "persistent_source_uids": list(self.persistent_source_uids),
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class SourceCallSiteSummary:
    callsite_uid: str
    root_target_uid: str
    source_uid: str
    call_path: tuple[str, ...]
    guard_exprs: tuple[str, ...]
    transition_relevant: bool
    transition_sink_uids: tuple[str, ...]
    availability: str
    min_offset: int | None = None
    max_offset: int | None = None
    persistent_source_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "callsite_uid": self.callsite_uid,
            "root_target_uid": self.root_target_uid,
            "source_uid": self.source_uid,
            "call_path": list(self.call_path),
            "guard_exprs": list(self.guard_exprs),
            "transition_relevant": bool(self.transition_relevant),
            "transition_sink_uids": list(self.transition_sink_uids),
            "availability": self.availability,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "persistent_source_uids": list(self.persistent_source_uids),
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class SourceAvailabilityPlan:
    schema: str
    axis: str
    evidence: tuple[SourceCallSiteAvailability, ...]
    summaries: tuple[SourceCallSiteSummary, ...]
    transition_cone_uids: tuple[str, ...]
    validation_notes: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "axis": self.axis,
            "evidence": [x.manifest() for x in self.evidence],
            "summaries": [x.manifest() for x in self.summaries],
            "transition_cone_uids": list(self.transition_cone_uids),
            "metrics": {
                "evidence_count": len(self.evidence),
                "summary_count": len(self.summaries),
                "static_count": sum(x.availability == "STATIC" for x in self.summaries),
                "current_count": sum(x.availability == "CURRENT" for x in self.summaries),
                "prefix_count": sum(x.availability == "PREFIX" for x in self.summaries),
                "suffix_count": sum(x.availability == "SUFFIX" for x in self.summaries),
                "complete_count": sum(x.availability == "COMPLETE" for x in self.summaries),
                "unknown_count": sum(x.availability == "UNKNOWN" for x in self.summaries),
            },
            "validation_notes": list(self.validation_notes),
        }


@dataclass(frozen=True)
class DomainAccessAvailabilityEvidence:
    access_uid: str
    source_uid: str
    target_uid: str
    source_semantic_class: str
    target_semantic_class: str
    transition_relevant: bool
    availability: str
    min_offset: int | None = None
    max_offset: int | None = None
    proof_kind: str | None = None
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "access_uid": self.access_uid,
            "source_uid": self.source_uid,
            "target_uid": self.target_uid,
            "source_semantic_class": self.source_semantic_class,
            "target_semantic_class": self.target_semantic_class,
            "transition_relevant": bool(self.transition_relevant),
            "availability": self.availability,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class ComponentScheduleComponent:
    """One carried-state component in the proof-only shadow schedule.

    A component contains only values already proved persistent by the upstream
    semantics layer.  StateDerivedMaps are intentionally absent: they may connect
    components or become post-state work, but merely reading state never promotes
    them back into the carried-state vector.
    """

    uid: str
    persistent_uids: tuple[str, ...]
    intrinsic_scan: str  # any | ascending | descending | mixed | unknown
    recurrence_access_uids: tuple[str, ...] = ()
    domain_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "persistent_uids": list(self.persistent_uids),
            "intrinsic_scan": self.intrinsic_scan,
            "recurrence_access_uids": list(self.recurrence_access_uids),
            "domain_uids": list(self.domain_uids),
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class CrossComponentOrderProof:
    """Frozen monotone co-scan proof for one persistent component dependency.

    This record is deliberately independent of availability labels.  It asks a
    narrower scheduling question: is there one monotone coordinate order that is
    simultaneously compatible with the source component, target component, and
    every exact cross-component transition?  An empty proven intersection means
    the components require a full-stage history fence; unknown geometry remains a
    blocker rather than becoming an implicit fence.
    """

    uid: str
    source_component_uid: str
    target_component_uid: str
    source_scan_options: tuple[str, ...]
    target_scan_options: tuple[str, ...]
    dependency_scan_options: tuple[str, ...]
    co_scan_options: tuple[str, ...]
    required_offset_min: int | None
    required_offset_max: int | None
    requires_stage_fence: bool
    fence_reasons: tuple[str, ...] = ()
    evidence_uids: tuple[str, ...] = ()
    forcing_evidence_uids: tuple[str, ...] = ()
    proof_kind: str = "canonical_cross_component_monotone_order_v1"
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_component_uid": self.source_component_uid,
            "target_component_uid": self.target_component_uid,
            "source_scan_options": list(self.source_scan_options),
            "target_scan_options": list(self.target_scan_options),
            "dependency_scan_options": list(self.dependency_scan_options),
            "co_scan_options": list(self.co_scan_options),
            "required_offset_min": self.required_offset_min,
            "required_offset_max": self.required_offset_max,
            "requires_stage_fence": bool(self.requires_stage_fence),
            "fence_reasons": list(self.fence_reasons),
            "evidence_uids": list(self.evidence_uids),
            "forcing_evidence_uids": list(self.forcing_evidence_uids),
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageScanOrderProof:
    """Frozen global monotone-order proof for one persistent schedule stage.

    Pairwise cross-component proofs are necessary but not sufficient: several
    individually co-scannable dependencies can still leave a whole candidate
    stage with no common monotone direction.  This record freezes the final
    stage-wide intersection and any dependency edges that had to become stage
    partition fences to keep every physical stage single-direction.
    """

    uid: str
    stage_index: int
    component_uids: tuple[str, ...]
    dependency_order_proof_uids: tuple[str, ...]
    initial_scan_options: tuple[str, ...]
    final_scan_options: tuple[str, ...]
    selected_scan_direction: str | None
    partition_order_proof_uids: tuple[str, ...] = ()
    proof_kind: str = "canonical_stage_global_monotone_order_v1"
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "stage_index": int(self.stage_index),
            "component_uids": list(self.component_uids),
            "dependency_order_proof_uids": list(self.dependency_order_proof_uids),
            "initial_scan_options": list(self.initial_scan_options),
            "final_scan_options": list(self.final_scan_options),
            "selected_scan_direction": self.selected_scan_direction,
            "partition_order_proof_uids": list(self.partition_order_proof_uids),
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class ComponentScheduleConstraint:
    """One component dependency or materialization barrier proof."""

    uid: str
    source_component_uid: str
    target_component_uid: str
    kind: str  # persistent_flow | availability | scan_compatibility
    availability: str
    barrier_required: bool
    order_proof_uid: str | None = None
    source_uids: tuple[str, ...] = ()
    target_uids: tuple[str, ...] = ()
    evidence_uids: tuple[str, ...] = ()
    proof_kind: str | None = None
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_component_uid": self.source_component_uid,
            "target_component_uid": self.target_component_uid,
            "kind": self.kind,
            "availability": self.availability,
            "barrier_required": bool(self.barrier_required),
            "order_proof_uid": self.order_proof_uid,
            "source_uids": list(self.source_uids),
            "target_uids": list(self.target_uids),
            "evidence_uids": list(self.evidence_uids),
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class DerivedScheduleTask:
    """State-derived work whose placement is downstream of persistent state."""

    uid: str
    target_uid: str
    availability: str
    producer_component_uids: tuple[str, ...]
    stage_index: int | None
    persistent_source_uids: tuple[str, ...] = ()
    evidence_uids: tuple[str, ...] = ()
    materialized_source_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "target_uid": self.target_uid,
            "availability": self.availability,
            "producer_component_uids": list(self.producer_component_uids),
            "stage_index": self.stage_index,
            "persistent_source_uids": list(self.persistent_source_uids),
            "evidence_uids": list(self.evidence_uids),
            "materialized_source_uids": list(self.materialized_source_uids),
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class ComponentScheduleStage:
    index: int
    component_uids: tuple[str, ...]
    scan_directions: tuple[str, ...]
    derived_task_uids: tuple[str, ...] = ()
    materialized_source_uids: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "component_uids": list(self.component_uids),
            "scan_directions": list(self.scan_directions),
            "derived_task_uids": list(self.derived_task_uids),
            "materialized_source_uids": list(self.materialized_source_uids),
            "note": self.note,
        }


@dataclass(frozen=True)
class ComponentSchedulePlan:
    schema: str
    axis: str
    components: tuple[ComponentScheduleComponent, ...]
    constraints: tuple[ComponentScheduleConstraint, ...]
    derived_tasks: tuple[DerivedScheduleTask, ...]
    stages: tuple[ComponentScheduleStage, ...]
    persistent_schedule_proven: bool
    full_schedule_proven: bool
    cross_component_order_proofs: tuple[CrossComponentOrderProof, ...] = ()
    stage_scan_order_proofs: tuple[StageScanOrderProof, ...] = ()
    blockers: tuple[str, ...] = ()
    deferred_unknowns: tuple[str, ...] = ()
    validation_notes: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "axis": self.axis,
            "components": [x.manifest() for x in self.components],
            "constraints": [x.manifest() for x in self.constraints],
            "cross_component_order_proofs": [
                x.manifest() for x in self.cross_component_order_proofs
            ],
            "stage_scan_order_proofs": [
                x.manifest() for x in self.stage_scan_order_proofs
            ],
            "derived_tasks": [x.manifest() for x in self.derived_tasks],
            "stages": [x.manifest() for x in self.stages],
            "persistent_schedule_proven": bool(self.persistent_schedule_proven),
            "full_schedule_proven": bool(self.full_schedule_proven),
            "blockers": list(self.blockers),
            "deferred_unknowns": list(self.deferred_unknowns),
            "metrics": {
                "component_count": len(self.components),
                "cross_component_order_proof_count": len(self.cross_component_order_proofs),
                "stage_scan_order_proof_count": len(self.stage_scan_order_proofs),
                "barrier_count": sum(x.barrier_required for x in self.constraints),
                "stage_count": len(self.stages),
                "derived_task_count": len(self.derived_tasks),
            },
            "validation_notes": list(self.validation_notes),
        }



@dataclass(frozen=True)
class CanonicalOverlayFact:
    """One typed fact imported from or derived from the canonical frontend snapshot.

    The overlay is intentionally scoped to the exact compiled run-key domain.  A
    fact may strengthen raw-source UNKNOWN evidence only inside that scope; it is
    not a product-wide theorem and is never execution permission by itself.
    """

    uid: str
    fact_kind: str
    source_uid: str | None = None
    target_uid: str | None = None
    canonical_uids: tuple[str, ...] = ()
    coordinate: int | None = None
    availability: str | None = None
    relation_min_offset: int | None = None
    relation_max_offset: int | None = None
    min_offset: int | None = None
    max_offset: int | None = None
    persistent_source_uids: tuple[str, ...] = ()
    proof_scope: str = "canonical_run_key_domain"
    proof_domain_size: int = 0
    provenance: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "fact_kind": self.fact_kind,
            "source_uid": self.source_uid,
            "target_uid": self.target_uid,
            "canonical_uids": list(self.canonical_uids),
            "coordinate": self.coordinate,
            "availability": self.availability,
            "relation_min_offset": self.relation_min_offset,
            "relation_max_offset": self.relation_max_offset,
            "min_offset": self.min_offset,
            "max_offset": self.max_offset,
            "persistent_source_uids": list(self.persistent_source_uids),
            "proof_scope": self.proof_scope,
            "proof_domain_size": int(self.proof_domain_size),
            "provenance": list(self.provenance),
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class CanonicalOverlayApplication:
    callsite_uid: str
    target_uid: str
    source_uid: str
    fact_uid: str
    before_availability: str
    after_availability: str
    before_min_offset: int | None = None
    before_max_offset: int | None = None
    after_min_offset: int | None = None
    after_max_offset: int | None = None
    persistent_source_uids: tuple[str, ...] = ()
    changed: bool = False
    reason: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "callsite_uid": self.callsite_uid,
            "target_uid": self.target_uid,
            "source_uid": self.source_uid,
            "fact_uid": self.fact_uid,
            "before_availability": self.before_availability,
            "after_availability": self.after_availability,
            "before_min_offset": self.before_min_offset,
            "before_max_offset": self.before_max_offset,
            "after_min_offset": self.after_min_offset,
            "after_max_offset": self.after_max_offset,
            "persistent_source_uids": list(self.persistent_source_uids),
            "changed": bool(self.changed),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CanonicalOverlaySchedulePlan:
    schema: str
    axis: str
    proof_scope: str
    proof_domain_size: int
    raw_availability: SourceAvailabilityPlan
    overlaid_availability: SourceAvailabilityPlan
    facts: tuple[CanonicalOverlayFact, ...]
    applications: tuple[CanonicalOverlayApplication, ...]
    component_schedule: ComponentSchedulePlan
    validation_notes: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        changed = [x for x in self.applications if x.changed]
        return {
            "schema": self.schema,
            "axis": self.axis,
            "proof_scope": self.proof_scope,
            "proof_domain_size": int(self.proof_domain_size),
            "raw_availability": self.raw_availability.manifest(),
            "overlaid_availability": self.overlaid_availability.manifest(),
            "facts": [x.manifest() for x in self.facts],
            "applications": [x.manifest() for x in self.applications],
            "component_schedule": self.component_schedule.manifest(),
            "metrics": {
                "fact_count": len(self.facts),
                "application_count": len(self.applications),
                "changed_application_count": len(changed),
                "raw_unknown_count": sum(
                    x.availability == "UNKNOWN" for x in self.raw_availability.evidence
                ),
                "overlaid_unknown_count": sum(
                    x.availability == "UNKNOWN" for x in self.overlaid_availability.evidence
                ),
            },
            "validation_notes": list(self.validation_notes),
        }


@dataclass(frozen=True)
class SourcePhaseComponent:
    uid: str
    member_uids: tuple[str, ...]
    internal_access_uids: tuple[str, ...]
    required_scan: str  # any | ascending | descending | mixed | unknown
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "member_uids": list(self.member_uids),
            "internal_access_uids": list(self.internal_access_uids),
            "required_scan": self.required_scan,
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class SourcePhaseConstraint:
    access_uid: str
    source_component_uid: str
    target_component_uid: str
    barrier_required: bool
    compatible_same_pass_scans: tuple[str, ...]
    reason: str

    def manifest(self) -> dict[str, Any]:
        return {
            "access_uid": self.access_uid,
            "source_component_uid": self.source_component_uid,
            "target_component_uid": self.target_component_uid,
            "barrier_required": self.barrier_required,
            "compatible_same_pass_scans": list(self.compatible_same_pass_scans),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SourcePhase:
    index: int
    component_uids: tuple[str, ...]
    scan_directions: tuple[str, ...]
    cross_phase_source_uids: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "component_uids": list(self.component_uids),
            "scan_directions": list(self.scan_directions),
            "cross_phase_source_uids": list(self.cross_phase_source_uids),
            "note": self.note,
        }


@dataclass(frozen=True)
class SourcePhasePlan:
    schema: str
    axis: str
    accesses: tuple[SourceAccessEvidence, ...]
    components: tuple[SourcePhaseComponent, ...]
    constraints: tuple[SourcePhaseConstraint, ...]
    phases: tuple[SourcePhase, ...]
    scan_decomposable: bool
    blockers: tuple[str, ...]
    validation_notes: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "axis": self.axis,
            "accesses": [x.manifest() for x in self.accesses],
            "components": [x.manifest() for x in self.components],
            "constraints": [x.manifest() for x in self.constraints],
            "phases": [x.manifest() for x in self.phases],
            "scan_decomposable": self.scan_decomposable,
            "blockers": list(self.blockers),
            "validation_notes": list(self.validation_notes),
        }

def _range_bounds(args: tuple[ast.AST, ...]) -> tuple[str, str, int | None]:
    if len(args) == 1:
        return "0", ast.unparse(args[0]), 1
    if len(args) == 2:
        return ast.unparse(args[0]), ast.unparse(args[1]), 1
    if len(args) == 3:
        step = args[2].value if isinstance(args[2], ast.Constant) and isinstance(args[2].value, int) else None
        return ast.unparse(args[0]), ast.unparse(args[1]), step
    return "?", "?", None


def _order_from_offset(offset: int) -> str:
    if offset == 0:
        return "same"
    if offset < 0:
        return "strict_before"
    return "strict_after"


def _domain_key_from_clock(proof: Any, phase: int) -> tuple[int, int, int, int]:
    return (int(phase), int(proof.period), int(proof.shift), int(proof.bias))


def build_shadow_domain_graph(executable: Any, variants: dict[str, Any]) -> DomainGraph:
    """Translate the authoritative executable graph into a lossless shadow IR.

    Construction is intentionally downstream of all existing scheduling proofs.
    Nothing in this function can make an unsupported model executable.
    """

    domains: list[IterationDomain] = []
    kernels: list[Kernel] = []
    accesses: list[AccessRelation] = []
    initials: list[InitialCondition] = []
    uid_domain: dict[str, str] = {}
    validation: list[str] = []

    scalar_domain = IterationDomain(
        uid="domain:scalar", kind="scalar", axes=(), order="none",
        note="zero-dimensional preparation/reduction domain",
    )
    domains.append(scalar_domain)

    primary_domains: dict[int, str] = {}
    for block in executable.loops:
        lo, hi, step = _range_bounds(tuple(block.range_args))
        domain_uid = f"domain:primary:{int(block.index)}"
        primary_domains[int(block.index)] = domain_uid
        order = "ascending" if int(block.scan_direction) > 0 else "descending"
        domains.append(
            IterationDomain(
                uid=domain_uid,
                kind="ordered",
                axes=(DomainAxis(block.coordinate_symbol, lo, hi, step),),
                order=order,
                note=f"shadow of executable loop {block.uid}",
            )
        )
        members = tuple(block.families)
        for uid in members:
            uid_domain[uid] = domain_uid
        kernels.append(
            Kernel(
                uid=f"kernel:primary:{int(block.index)}",
                domain_uid=domain_uid,
                member_uids=members,
                kind="primary",
                persistent_uids=tuple(sorted(uid for uid in members if block.max_lag_by_family.get(uid, 0) > 0)),
                legacy_roles=tuple((uid, executable.roles.get(uid, "")) for uid in members),
                note="physical executable-loop combinational kernel",
            )
        )
        for uid in block.reductions:
            uid_domain[uid] = scalar_domain.uid
            kernels.append(
                Kernel(
                    uid=f"kernel:reduction:{uid}", domain_uid=scalar_domain.uid,
                    member_uids=(uid,), kind="reduction",
                    legacy_roles=((uid, executable.roles.get(uid, "reduction")),),
                )
            )
        for uid in block.post_scalars:
            uid_domain[uid] = scalar_domain.uid

    for uid in executable.pre_scalars:
        uid_domain[uid] = scalar_domain.uid
    for uid, role in executable.roles.items():
        if role in {"scalar", "reduction"}:
            uid_domain.setdefault(uid, scalar_domain.uid)

    clock_domains: dict[tuple[int, int, int, int], str] = {}

    def event_domain(phase: int, proof: Any, mapping_expr: ast.AST) -> str:
        key = _domain_key_from_clock(proof, phase)
        if key in clock_domains:
            return clock_domains[key]
        uid = "domain:event:%d:%d:%d:%d" % key
        parent = primary_domains.get(int(phase))
        domains.append(
            IterationDomain(
                uid=uid,
                kind="event",
                axes=(DomainAxis("event", None, None, 1),),
                order="mapped",
                parent_uid=parent,
                mapping_expr=ast.unparse(mapping_expr),
                proof_kind=getattr(proof, "proof_kind", None),
                note=getattr(proof, "note", ""),
            )
        )
        clock_domains[key] = uid
        return uid

    for uid, state in sorted(executable.derived_clock_states.items()):
        d = event_domain(state.phase, state.proof, state.mapping_expr)
        uid_domain[uid] = d
        kernels.append(
            Kernel(
                uid=f"kernel:event-state:{uid}", domain_uid=d,
                member_uids=(uid,), kind="event", persistent_uids=(uid,),
                legacy_roles=((uid, "derived_state"),),
                note="shadow of loop-carried derived-clock state",
            )
        )
        accesses.append(
            AccessRelation(
                uid=_stable_id("access", (uid, uid, d, str(state.self_offset), "derived_state_self")),
                source_uid=uid, target_uid=uid,
                source_domain_uid=d, target_domain_uid=d,
                selector="point", index_expr=f"event{int(state.self_offset):+d}",
                order=_order_from_offset(int(state.self_offset)),
                min_lag=(-int(state.self_offset) if int(state.self_offset) < 0 else None),
                max_lag=(-int(state.self_offset) if int(state.self_offset) < 0 else None),
                min_offset=int(state.self_offset), max_offset=int(state.self_offset),
                proof_kind="legacy_derived_state_self",
                legacy_kind="DerivedClockState",
            )
        )

    transition_overlays: dict[tuple[str, str, str], Any] = {}
    for root, region in sorted(executable.derived_clock_regions.items()):
        d = event_domain(region.phase, region.clock_proof, region.mapping_expr)
        for uid in region.member_uids:
            uid_domain[uid] = d
        kernels.append(
            Kernel(
                uid=f"kernel:event-region:{root}", domain_uid=d,
                member_uids=tuple(region.member_uids), kind="event",
                persistent_uids=tuple(region.persistent_uids),
                legacy_roles=tuple((uid, "derived_region") for uid in region.member_uids),
                note="shadow of executable derived-clock region",
            )
        )
        for row in region.transition_reads:
            key = (
                row.source_uid,
                row.target_uid,
                ast.dump(row.expr, include_attributes=False),
            )
            transition_overlays[key] = row

    def domain_of(uid: str) -> str:
        if uid in uid_domain:
            return uid_domain[uid]
        role = executable.roles.get(uid)
        if role == "coordinate":
            phase = int(executable.coordinate_phase.get(uid, 0))
            d = primary_domains.get(phase)
            if d is not None:
                uid_domain[uid] = d
                return d
        uid_domain[uid] = scalar_domain.uid
        return scalar_domain.uid

    for idx, use in enumerate(executable.uses):
        source_domain = domain_of(use.source)
        target_domain = domain_of(use.target)
        rel = use.relation
        selector = "value"
        index_expr: str | None = None
        order = "unknown"
        min_lag = max_lag = min_offset = max_offset = None
        proof_kind = None
        legacy_kind = None
        note = ""
        if isinstance(rel, AffineOffset):
            selector = "point"
            min_offset = max_offset = int(rel.offset)
            order = _order_from_offset(int(rel.offset))
            index_expr = f"t{int(rel.offset):+d}"
            if rel.offset < 0:
                min_lag = max_lag = -int(rel.offset)
            legacy_kind = "AffineOffset"
        elif isinstance(rel, AffineAliasRead):
            selector = "point"
            min_offset = max_offset = int(rel.offset)
            order = _order_from_offset(int(rel.offset))
            index_expr = rel.normalized_expr
            if rel.offset < 0:
                min_lag = max_lag = -int(rel.offset)
            proof_kind = "affine_alias"
            legacy_kind = "AffineAliasRead"
        elif isinstance(rel, DerivedHistoryRead):
            selector = "point"
            index_expr = ast.unparse(rel.expr)
            transition = transition_overlays.pop(
                (use.source, use.target, ast.dump(rel.expr, include_attributes=False)),
                None,
            )
            if transition is not None:
                proof_kind = transition.proof.proof_kind
                order = "strict_before"
                min_lag = max_lag = -int(transition.proof.primary_offset)
                min_offset = max_offset = int(transition.proof.primary_offset)
                legacy_kind = "TransitionLagProof"
                note = transition.proof.note
            else:
                legacy_kind = "DerivedHistoryRead"
                if rel.proof is not None:
                    proof_kind = rel.proof.proof_kind
                    order = rel.proof.relation
                    min_lag = rel.proof.min_lag
                    max_lag = rel.proof.max_lag
                    if rel.proof.min_lag is not None and rel.proof.max_lag is not None:
                        min_offset = -int(rel.proof.max_lag)
                        max_offset = -int(rel.proof.min_lag)
                    note = rel.proof.note
        elif isinstance(rel, DerivedClockRead):
            selector = "point"
            index_expr = ast.unparse(rel.expr)
            order = "same"
            proof_kind = rel.proof.proof_kind
            legacy_kind = "DerivedClockRead"
            note = rel.proof.note
        elif use.context == "reduction":
            selector = "reduction"
        accesses.append(
            AccessRelation(
                uid=_stable_id("access", (str(idx), use.source, use.target, use.context)),
                source_uid=use.source, target_uid=use.target,
                source_domain_uid=source_domain, target_domain_uid=target_domain,
                selector=selector, index_expr=index_expr, order=order,
                min_lag=min_lag, max_lag=max_lag,
                min_offset=min_offset, max_offset=max_offset,
                proof_kind=proof_kind, legacy_kind=legacy_kind, note=note,
            )
        )

    for uid, seed in sorted(executable.boundary_seeds.items()):
        d = domain_of(uid)
        initials.append(
            InitialCondition(
                uid=f"init:boundary:{uid}", state_uid=uid, domain_uid=d,
                coordinate_expr=str(int(seed.coordinate)), kind="boundary_seed",
                note=seed.note,
            )
        )

    for uid, state in sorted(executable.derived_clock_states.items()):
        d = domain_of(uid)
        initials.append(
            InitialCondition(
                uid=f"init:event-state:{uid}", state_uid=uid, domain_uid=d,
                coordinate_expr=str(int(state.proof.initial_bucket)),
                kind="event_state_initial_bucket",
                note=state.proof.note,
            )
        )
    for root, region in sorted(executable.derived_clock_regions.items()):
        d = domain_of(root)
        for uid in region.persistent_uids:
            initials.append(
                InitialCondition(
                    uid=f"init:event-region:{uid}", state_uid=uid, domain_uid=d,
                    coordinate_expr=str(int(region.clock_proof.initial_bucket)),
                    kind="event_region_initial_bucket",
                    note="; ".join(region.initial_safety_notes),
                )
            )

    legacy_summary = {
        "template_uses": len(executable.uses),
        "derived_clock_states": len(executable.derived_clock_states),
        "derived_clock_regions": len(executable.derived_clock_regions),
        "transition_reads": sum(len(r.transition_reads) for r in executable.derived_clock_regions.values()),
        "boundary_seeds": len(executable.boundary_seeds),
    }

    # Validate the shadow is a complete semantic adapter for the current special
    # scheduling constructs.  This is evidence only; it does not compare physical
    # storage because storage migration is explicitly out of scope for this step.
    if transition_overlays:
        raise DomainGraphError(
            "derived transition proof did not upgrade its underlying access: "
            f"{sorted((a, b) for a, b, _sig in transition_overlays)!r}"
        )
    legacy_access_count = len(executable.uses) + legacy_summary["derived_clock_states"]
    if len(accesses) != legacy_access_count:
        raise DomainGraphError(
            f"shadow access coverage mismatch: {len(accesses)} vs {legacy_access_count}"
        )
    for uid in executable.derived_clock_states:
        if uid_domain.get(uid, "").startswith("domain:event:") is False:
            raise DomainGraphError(f"derived state {uid} did not map to an event domain")
    for region in executable.derived_clock_regions.values():
        ds = {uid_domain.get(uid) for uid in region.member_uids}
        if len(ds) != 1 or None in ds:
            raise DomainGraphError("derived region members did not map to one event domain")
    if len(initials) < legacy_summary["boundary_seeds"]:
        raise DomainGraphError("boundary seed did not map to an initial condition")
    validation.append(
        "shadow adapter covers every legacy TemplateUse plus each derived-state self edge; transition proofs upgrade their underlying access"
    )
    validation.append(
        "derived clock states/regions map to generic event domains; boundary seeds map to initial conditions"
    )

    base = DomainGraph(
        schema="modelx_graph.domain_graph.shadow.v1",
        domains=tuple(domains), kernels=tuple(kernels), accesses=tuple(accesses),
        initial_conditions=tuple(initials),
        uid_domain=tuple(sorted(uid_domain.items())),
        legacy_summary=tuple(sorted(legacy_summary.items())),
        validation_notes=tuple(validation),
    )
    return with_domain_graph_purity(base, variants)


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
        if isinstance(node.op, ast.Add):
            return int(node.right.value)
        if isinstance(node.op, ast.Sub):
            return -int(node.right.value)
    return None


_PURE_BUILTINS = {
    "abs", "all", "any", "bool", "enumerate", "float", "int", "len",
    "max", "min", "range", "round", "str", "sum", "tuple", "zip",
    "ValueError", "TypeError",
}
_PURE_MODULE_ROOTS = {"math", "np", "numpy", "pd", "pandas", "data"}
_DYNAMIC_RUNTIME_ROOTS = {
    "space", "model", "cells", "refs", "runtime", "_space", "_model",
    "_cells", "_refs", "_impl", "modelx",
}
_PURE_VALUE_METHODS = {"split", "strip", "lower", "upper", "startswith", "endswith", "get"}
_MUTATING_METHODS = {
    "append", "extend", "insert", "pop", "remove", "clear", "update",
    "setdefault", "sort", "reverse", "add", "discard", "__setitem__",
}


def _root_name(node: ast.AST) -> str | None:
    cur = node
    while isinstance(cur, (ast.Attribute, ast.Subscript)):
        cur = cur.value
    if isinstance(cur, ast.Call):
        return _root_name(cur.func)
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def _assigned_names(fn: ast.FunctionDef) -> set[str]:
    out = {a.arg for a in fn.args.args}
    for node in ast.walk(fn):
        targets: list[ast.AST] = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            else:
                targets.append(node.target)
        elif isinstance(node, (ast.For, ast.comprehension)):
            targets.append(node.target)
        for target in targets:
            for child in ast.walk(target):
                if isinstance(child, ast.Name):
                    out.add(child.id)
    return out


def classify_source_purity(source: str) -> tuple[PurityEvidence, ...]:
    """Prove source functions pure-map/stateful using transitive source closure.

    This is deliberately permission-free evidence.  A function is a pure map only
    when every local dependency is itself proven pure and the body contains no
    persistent recursion, mutation, stateful statement loop, dynamic runtime-object
    access, or unresolved callable.  Ordinary globals and ``data``/numpy/pandas/math
    reads are treated as immutable input/table sources; they never create ordering.
    """
    module = ast.parse(source)
    functions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    known = set(functions)
    deps: dict[str, set[str]] = {name: set() for name in known}
    immutable: dict[str, set[str]] = {name: set() for name in known}
    blockers: dict[str, set[str]] = {name: set() for name in known}

    module_assigned = {
        target.id
        for node in module.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }

    for name, fn in functions.items():
        local_names = _assigned_names(fn)
        for node in ast.walk(fn):
            if isinstance(node, (ast.For, ast.While)):
                blockers[name].add("stateful_local_loop")
            if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal)):
                blockers[name].add("stateful_control_or_scope")
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, (ast.Attribute, ast.Subscript)) for t in targets):
                    blockers[name].add("mutation")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                    if callee in known:
                        deps[name].add(callee)
                        if callee == name:
                            blockers[name].add("source_backed_recursion")
                    elif callee in _PURE_BUILTINS:
                        immutable[name].add(f"builtin:{callee}")
                    elif callee not in local_names:
                        blockers[name].add(f"unproven_callable:{callee}")
                elif isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    root = _root_name(node.func.value)
                    if attr in _MUTATING_METHODS:
                        blockers[name].add("mutation")
                    elif attr in _PURE_VALUE_METHODS and root is not None:
                        immutable[name].add(f"pure_value_method:{attr}")
                    elif root in _DYNAMIC_RUNTIME_ROOTS:
                        blockers[name].add(f"dynamic_runtime_object:{root}")
                    elif root in _PURE_MODULE_ROOTS:
                        immutable[name].add(f"immutable_table_or_module:{root}")
                    elif root in known:
                        deps[name].add(root)
                    elif root is not None and root not in local_names:
                        blockers[name].add(f"unproven_attribute_call:{root}.{attr}")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                ident = node.id
                if ident in known or ident in local_names or ident in _PURE_BUILTINS:
                    continue
                if ident in _DYNAMIC_RUNTIME_ROOTS:
                    blockers[name].add(f"dynamic_runtime_object:{ident}")
                elif ident in _PURE_MODULE_ROOTS or ident in module_assigned:
                    immutable[name].add(f"immutable_global:{ident}")
                else:
                    # Free scalar names in lifelib formula modules are model inputs or
                    # full-run scalar references.  They are immutable for one run.
                    immutable[name].add(f"immutable_scalar:{ident}")

    # Propagate SCC/cycle statefulness first.  Any source cycle is unproven recursion.
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            if name in stack:
                cycle = stack[stack.index(name):]
                for member in cycle:
                    blockers[member].add("source_backed_recursion")
            return
        visiting.add(name); stack.append(name)
        for dep in deps[name]:
            visit(dep)
        stack.pop(); visiting.remove(name); visited.add(name)

    for name in sorted(known):
        visit(name)

    classification: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for name in sorted(known):
            direct_stateful = any(
                b in {"source_backed_recursion", "mutation", "stateful_local_loop", "stateful_control_or_scope"}
                or b.startswith("dynamic_runtime_object:")
                for b in blockers[name]
            )
            dep_states = [classification.get(d) for d in deps[name]]
            if direct_stateful or any(x == "stateful" for x in dep_states):
                new = "stateful"
            elif blockers[name] or any(x == "unknown" for x in dep_states):
                new = "unknown"
            elif all(x == "pure_map" for x in dep_states):
                new = "pure_map"
            else:
                continue
            if classification.get(name) != new:
                classification[name] = new; changed = True

    # Remaining unresolved dependency cycles fail closed as stateful recursion.
    for name in sorted(known):
        if name not in classification:
            blockers[name].add("source_backed_recursion")
            classification[name] = "stateful"

    # Add transitive blocker provenance for non-pure callers.
    for name in sorted(known):
        for dep in sorted(deps[name]):
            state = classification[dep]
            if state != "pure_map":
                blockers[name].add(f"dependency_{state}:{dep}")

    return tuple(
        PurityEvidence(
            uid=name, classification=classification[name],
            proof_kind="source_transitive_purity_v1",
            dependencies=tuple(sorted(deps[name])),
            blockers=tuple(sorted(blockers[name])),
            immutable_sources=tuple(sorted(immutable[name])),
            note=(
                "complete source closure is state-free" if classification[name] == "pure_map"
                else "fails closed because source closure contains or reaches stateful/unproven behavior"
            ),
        )
        for name in sorted(known)
    )


def _source_axis_functions(module: ast.Module, axis: str) -> dict[str, ast.FunctionDef]:
    out: dict[str, ast.FunctionDef] = {}
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        params = [a.arg for a in node.args.args]
        if axis in params:
            out[node.name] = node
    return out


def _call_arg_for_param(call: ast.Call, fn: ast.FunctionDef, param: str) -> ast.AST | None:
    params = [a.arg for a in fn.args.args]
    if param not in params:
        return None
    pos = params.index(param)
    if pos < len(call.args):
        return call.args[pos]
    for kw in call.keywords:
        if kw.arg == param:
            return kw.value
    return None


def _tarjan_source_components(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, list[str]] = {str(n): [] for n in nodes}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
        adjacency.setdefault(target, [])
    for source in adjacency:
        adjacency[source] = sorted(set(adjacency[source]))

    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for child in adjacency.get(node, ()):
            if child not in indices:
                strongconnect(child)
                lowlink[node] = min(lowlink[node], lowlink[child])
            elif child in on_stack:
                lowlink[node] = min(lowlink[node], indices[child])
        if lowlink[node] != indices[node]:
            return
        members: list[str] = []
        while True:
            child = stack.pop()
            on_stack.remove(child)
            members.append(child)
            if child == node:
                break
        components.append(tuple(sorted(members)))

    for node in sorted(adjacency):
        if node not in indices:
            strongconnect(node)
    return tuple(sorted(components, key=lambda c: c))


def _scan_set(required_scan: str) -> set[str]:
    if required_scan == "ascending":
        return {"ascending"}
    if required_scan == "descending":
        return {"descending"}
    if required_scan == "any":
        return {"ascending", "descending"}
    return set()


def _access_scan_set(order: str) -> set[str]:
    if order == "strict_before":
        return {"ascending"}
    if order == "strict_after":
        return {"descending"}
    if order in {"same", "fixed"}:
        return {"ascending", "descending"}
    return set()


def _collect_source_axis_accesses(
    module: ast.Module,
    axis_functions: dict[str, ast.FunctionDef],
    source_classification: dict[str, str],
    *,
    axis: str,
) -> tuple[SourceAccessEvidence, ...]:
    """Collect deterministic source-axis relations without granting semantics.

    The helper is shared by the phase and persistence/state-derived analyzers so
    both reason over the exact same source call-site geometry.  Purity/state
    classification is supplied by the caller and affects only the evidence flag
    that says whether an access is relevant to the older phase analyzer.
    """
    accesses: list[SourceAccessEvidence] = []

    for target_uid, fn in sorted(axis_functions.items()):
        window_calls: set[int] = set()
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"any", "all", "sum"}
                and len(node.args) == 1
                and isinstance(node.args[0], ast.GeneratorExp)
            ):
                continue
            gen = node.args[0]
            if len(gen.generators) != 1:
                continue
            comp = gen.generators[0]
            if (
                comp.is_async
                or comp.ifs
                or not isinstance(comp.target, ast.Name)
                or not isinstance(comp.iter, ast.Call)
                or not isinstance(comp.iter.func, ast.Name)
                or comp.iter.func.id != "range"
            ):
                continue
            body = gen.elt
            if not (
                isinstance(body, ast.Call)
                and isinstance(body.func, ast.Name)
                and body.func.id in axis_functions
            ):
                continue
            source_fn = axis_functions[body.func.id]
            source_arg = _call_arg_for_param(body, source_fn, axis)
            if not (
                isinstance(source_arg, ast.Name)
                and source_arg.id == comp.target.id
            ):
                continue
            window_calls.add(id(body))
            source_uid = body.func.id
            state = source_classification.get(source_uid, "unknown")
            relevant = state != "pure_map"
            index_expr = f"{comp.target.id} in {ast.unparse(comp.iter)}"
            accesses.append(
                SourceAccessEvidence(
                    uid=_stable_id(
                        "source_access",
                        (source_uid, target_uid, "window", index_expr),
                    ),
                    source_uid=source_uid,
                    target_uid=target_uid,
                    selector="window",
                    index_expr=index_expr,
                    order="window",
                    source_classification=state,
                    scheduling_relevant=relevant,
                    scheduling_reason=(
                        "stateful_or_unproven_source" if relevant else "pure_map_call"
                    ),
                    proof_kind="source_finite_window_candidate_v1",
                    note=(
                        "finite source window is scheduling evidence only; relative "
                        "causality is not assumed"
                    ),
                )
            )

        for call in ast.walk(fn):
            if id(call) in window_calls:
                continue
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in axis_functions
            ):
                continue
            source_uid = call.func.id
            source_fn = axis_functions[source_uid]
            coord = _call_arg_for_param(call, source_fn, axis)
            if coord is None:
                continue
            off = _affine_offset(coord, axis)
            selector = "point"
            min_offset = max_offset = None
            if off is not None:
                min_offset = max_offset = int(off)
                order = _order_from_offset(int(off))
                proof_kind = "source_affine_axis_v1"
            elif (
                isinstance(coord, ast.Constant)
                and isinstance(coord.value, int)
                and not isinstance(coord.value, bool)
            ):
                order = "fixed"
                proof_kind = "source_fixed_coordinate_v1"
            else:
                order = "unknown"
                proof_kind = "source_dynamic_coordinate_v1"
            state = source_classification.get(source_uid, "unknown")
            relevant = state != "pure_map"
            expr = ast.unparse(coord)
            accesses.append(
                SourceAccessEvidence(
                    uid=_stable_id(
                        "source_access",
                        (source_uid, target_uid, selector, expr, order),
                    ),
                    source_uid=source_uid,
                    target_uid=target_uid,
                    selector=selector,
                    index_expr=expr,
                    order=order,
                    source_classification=state,
                    scheduling_relevant=relevant,
                    scheduling_reason=(
                        "stateful_or_unproven_source" if relevant else "pure_map_call"
                    ),
                    min_offset=min_offset,
                    max_offset=max_offset,
                    proof_kind=proof_kind,
                    note="source-only axis relation; execution authority unchanged",
                )
            )

    access_map: dict[
        tuple[str, str, str, str | None, str, int | None, int | None, bool],
        SourceAccessEvidence,
    ] = {}
    for row in accesses:
        key = (
            row.source_uid,
            row.target_uid,
            row.selector,
            row.index_expr,
            row.order,
            row.min_offset,
            row.max_offset,
            row.scheduling_relevant,
        )
        access_map[key] = row
    return tuple(sorted(access_map.values(), key=lambda x: x.uid))


def classify_source_value_semantics(
    source: str,
    *,
    axis: str = "t",
) -> tuple[ValueSemanticsEvidence, ...]:
    """Classify source values as pure, state-derived, persistent, or unknown.

    Persistence is intentionally narrower than the older ``stateful`` purity class.
    A value is proven persistent only when it is the *source* of a shifted point
    access that lies on a feedback path in the known point-access graph.  A caller
    that merely reaches such a value is state-dependent but remains ephemeral.

    Window reductions and local statement loops are operator traits, not automatic
    persistence.  This lets the evidence distinguish a reduction over state history
    from the state history itself while remaining fail-closed for execution.
    """
    module = ast.parse(source)
    functions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    axis_functions = _source_axis_functions(module, axis)
    purity_rows = {x.uid: x for x in classify_source_purity(source)}
    accesses = _collect_source_axis_accesses(
        module,
        axis_functions,
        {uid: row.classification for uid, row in purity_rows.items()},
        axis=axis,
    )

    # The persistence proof uses only coordinate relations whose geometry is known.
    # Window/dynamic accesses are read patterns; they do not by themselves prove that
    # their source is a loop-carried value.
    point_accesses = [
        x for x in accesses
        if x.selector == "point"
        and x.order in {"same", "strict_before", "strict_after", "fixed"}
    ]
    point_components = _tarjan_source_components(
        axis_functions,
        ((x.source_uid, x.target_uid) for x in point_accesses),
    )
    component_by_uid = {
        member: index
        for index, members in enumerate(point_components)
        for member in members
    }
    internal_shifted = [
        x for x in point_accesses
        if x.order in {"strict_before", "strict_after"}
        and component_by_uid.get(x.source_uid) == component_by_uid.get(x.target_uid)
    ]
    persistent_uids = {x.source_uid for x in internal_shifted}
    recurrence_by_uid: dict[str, list[str]] = {uid: [] for uid in functions}
    for access in internal_shifted:
        recurrence_by_uid.setdefault(access.source_uid, []).append(access.uid)

    # Same-coordinate source recursion with no proven shifted carried value remains
    # unresolved.  A component that contains at least one shifted relation can still
    # contain ephemeral combinational helpers; those helpers are not promoted merely
    # because they share the feedback SCC.
    point_edges = {(x.source_uid, x.target_uid) for x in point_accesses}
    unresolved_persistence: set[str] = set()
    for members in point_components:
        cyclic = len(members) > 1 or any((uid, uid) in point_edges for uid in members)
        if not cyclic:
            continue
        has_shift = any(
            x.source_uid in members
            and x.target_uid in members
            and x.order in {"strict_before", "strict_after"}
            for x in point_accesses
        )
        if not has_shift:
            unresolved_persistence.update(members)

    deps = {uid: set(row.dependencies) for uid, row in purity_rows.items()}
    reads_state = set(persistent_uids)
    changed = True
    while changed:
        changed = False
        for uid in sorted(functions):
            if uid in reads_state:
                continue
            if any(dep in reads_state for dep in deps.get(uid, ())):
                reads_state.add(uid)
                changed = True

    traits: dict[str, set[str]] = {uid: set() for uid in functions}
    for uid, row in purity_rows.items():
        for blocker in row.blockers:
            if blocker == "stateful_local_loop":
                traits[uid].add("local_statement_loop")
            elif blocker == "mutation":
                traits[uid].add("mutation")
            elif blocker == "stateful_control_or_scope":
                traits[uid].add("stateful_control_or_scope")
            elif blocker.startswith("dynamic_runtime_object:"):
                traits[uid].add("dynamic_runtime_access")
            elif blocker.startswith("unproven_callable:") or blocker.startswith(
                "unproven_attribute_call:"
            ):
                traits[uid].add("unproven_callable")
    for access in accesses:
        if access.selector == "window":
            traits.setdefault(access.target_uid, set()).add("window_reduction")
        elif access.order == "unknown":
            traits.setdefault(access.target_uid, set()).add("dynamic_coordinate_access")

    rows: list[ValueSemanticsEvidence] = []
    for uid in sorted(functions):
        purity = purity_rows[uid]
        blockers = set(purity.blockers)
        if uid in persistent_uids:
            persistence = "persistent"
        elif uid in unresolved_persistence:
            persistence = "unknown"
            blockers.add("unresolved_same_coordinate_recursion")
        else:
            persistence = "ephemeral"

        if purity.classification == "pure_map":
            state_dependency = "static_only"
        elif uid in reads_state:
            state_dependency = "reads_state"
        else:
            state_dependency = "unknown"

        if purity.classification == "pure_map" and persistence == "ephemeral":
            semantic_class = "pure_map"
            note = "complete source closure is state-free and no carried state is required"
        elif persistence == "persistent":
            semantic_class = "persistent_state"
            note = (
                "value is the source of a proven shifted point access on a feedback path"
            )
        elif state_dependency == "reads_state" and persistence == "ephemeral":
            semantic_class = "state_derived_map"
            note = (
                "value reads persistent state but has no proven cross-coordinate carried-state role"
            )
        else:
            semantic_class = "unknown"
            note = "source evidence is insufficient to prove pure, derived, or persistent semantics"

        rows.append(
            ValueSemanticsEvidence(
                uid=uid,
                semantic_class=semantic_class,
                state_dependency=state_dependency,
                persistence=persistence,
                proof_kind="source_value_semantics_v1",
                dependencies=tuple(sorted(deps.get(uid, ()))),
                recurrence_access_uids=tuple(sorted(recurrence_by_uid.get(uid, ()))),
                operator_traits=tuple(sorted(traits.get(uid, ()))),
                blockers=tuple(sorted(blockers)),
                note=note,
            )
        )
    return tuple(rows)


def analyze_source_state_structure(
    source: str,
    *,
    axis: str = "t",
) -> SourceStateStructure:
    """Compare coarse stateful SCCs with the contracted persistent-state graph.

    The raw SCC side deliberately mirrors the V02341 source phase analyzer: every
    non-pure axis function and every scheduling-relevant state access participates.
    The persistent side then contracts ephemeral state-derived maps and retains only
    values proven to be carried across coordinates.  The result is diagnostic only;
    it does not infer scan direction or authorize a backend schedule.
    """
    module = ast.parse(source)
    axis_functions = _source_axis_functions(module, axis)
    purity = {x.uid: x for x in classify_source_purity(source)}
    semantics = classify_source_value_semantics(source, axis=axis)
    semantics_by_uid = {x.uid: x for x in semantics}
    accesses = _collect_source_axis_accesses(
        module,
        axis_functions,
        {uid: row.classification for uid, row in purity.items()},
        axis=axis,
    )

    raw_nodes = {
        uid for uid in axis_functions if purity[uid].classification != "pure_map"
    }
    raw_accesses = [
        x for x in accesses
        if x.scheduling_relevant
        and x.source_uid in raw_nodes
        and x.target_uid in raw_nodes
    ]
    raw_sccs = _tarjan_source_components(
        raw_nodes,
        ((x.source_uid, x.target_uid) for x in raw_accesses),
    )

    persistent = {
        uid for uid, row in semantics_by_uid.items()
        if uid in axis_functions and row.persistence == "persistent"
    }
    adjacency: dict[str, set[str]] = {uid: set() for uid in raw_nodes}
    for access in raw_accesses:
        adjacency.setdefault(access.source_uid, set()).add(access.target_uid)

    # Contract paths whose interior contains only ephemeral/non-persistent values.
    # Reaching another persistent value creates one edge in the reduced state graph.
    contracted_edges: set[tuple[str, str]] = set()
    for source_uid in sorted(persistent):
        pending = list(sorted(adjacency.get(source_uid, ())))
        visited: set[str] = set()
        while pending:
            target_uid = pending.pop(0)
            if target_uid in visited:
                continue
            visited.add(target_uid)
            if target_uid in persistent:
                contracted_edges.add((source_uid, target_uid))
                continue
            pending.extend(
                child for child in sorted(adjacency.get(target_uid, ()))
                if child not in visited
            )

    persistent_sccs = _tarjan_source_components(persistent, contracted_edges)
    components: list[SourcePersistentComponent] = []
    for members in persistent_sccs:
        internal_edges = tuple(sorted(
            edge for edge in contracted_edges
            if edge[0] in members and edge[1] in members
        ))
        components.append(
            SourcePersistentComponent(
                uid=_stable_id("persistent_scc", members),
                persistent_uids=members,
                contracted_edges=internal_edges,
                note=(
                    "ephemeral state-derived paths are contracted; component is proof-only"
                ),
            )
        )

    raw_largest = max((len(x) for x in raw_sccs), default=0)
    persistent_largest = max((len(x) for x in persistent_sccs), default=0)
    return SourceStateStructure(
        schema="modelx_graph.domain_graph.source_state_structure.v1",
        axis=axis,
        semantics=semantics,
        accesses=accesses,
        raw_state_sccs=raw_sccs,
        persistent_components=tuple(components),
        validation_notes=(
            "proof-only: executable scheduling/storage/codegen remain unchanged",
            (
                f"largest coarse stateful SCC={raw_largest}; "
                f"largest contracted persistent component={persistent_largest}"
            ),
            "window/dynamic/operator traits remain explicit and grant no execution permission",
        ),
    )


@dataclass(frozen=True)
class _DirectSourceCallSite:
    uid: str
    target_uid: str
    source_uid: str
    selector: str
    coord_expr: ast.AST
    guards: tuple[ast.AST, ...]
    lineno: int
    col_offset: int
    range_call: ast.Call | None = None
    generator_uid: str | None = None
    generator_filters: tuple[ast.AST, ...] = ()


@dataclass(frozen=True)
class _ProofAffine:
    coeffs: tuple[tuple[str, int], ...]
    const: int

    def as_dict(self) -> dict[str, int]:
        return dict(self.coeffs)


def _proof_affine(coeffs: dict[str, int] | None = None, const: int = 0) -> _ProofAffine:
    return _ProofAffine(
        tuple(sorted((str(k), int(v)) for k, v in (coeffs or {}).items() if int(v))),
        int(const),
    )


def _proof_affine_add(a: _ProofAffine, b: _ProofAffine, scale: int = 1) -> _ProofAffine:
    coeffs = a.as_dict()
    for key, value in b.coeffs:
        coeffs[key] = coeffs.get(key, 0) + int(scale) * int(value)
    return _proof_affine(coeffs, a.const + int(scale) * b.const)


def _proof_affine_scale(a: _ProofAffine, scale: int) -> _ProofAffine:
    return _proof_affine(
        {key: int(value) * int(scale) for key, value in a.coeffs},
        a.const * int(scale),
    )


def _proof_affine_div(a: _ProofAffine, divisor: int) -> _ProofAffine | None:
    divisor = int(divisor)
    if divisor <= 0 or any(int(value) % divisor for _key, value in a.coeffs):
        return None
    return _proof_affine(
        {key: int(value) // divisor for key, value in a.coeffs},
        a.const // divisor,
    )


def _proof_affine_mod(a: _ProofAffine, divisor: int) -> _ProofAffine | None:
    divisor = int(divisor)
    if divisor <= 0 or any(int(value) % divisor for _key, value in a.coeffs):
        return None
    return _proof_affine({}, a.const % divisor)


def _copy_expr(node: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(ast.parse(ast.unparse(node), mode="eval").body)


def _negate_expr(node: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(ast.UnaryOp(op=ast.Not(), operand=_copy_expr(node)))


def _block_always_terminates(statements: list[ast.stmt]) -> bool:
    for statement in statements:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return True
        if isinstance(statement, ast.If) and statement.orelse:
            if _block_always_terminates(statement.body) and _block_always_terminates(statement.orelse):
                return True
    return False


def _collect_direct_source_callsites(
    module: ast.Module,
    axis_functions: dict[str, ast.FunctionDef],
    *,
    axis: str,
) -> tuple[_DirectSourceCallSite, ...]:
    """Collect concrete source call sites with conservative path guards.

    This is separate from ``_collect_source_axis_accesses`` on purpose.  V02341's
    pair-level evidence remains byte-stable, while this collector preserves AST
    location, short-circuit guards and simple post-return subdomains.
    """
    rows: list[_DirectSourceCallSite] = []

    def record_point(
        call: ast.Call, target_uid: str, guards: tuple[ast.AST, ...]
    ) -> None:
        if not isinstance(call.func, ast.Name) or call.func.id not in axis_functions:
            return
        source_uid = call.func.id
        coord = _call_arg_for_param(call, axis_functions[source_uid], axis)
        if coord is None:
            return
        expr = ast.unparse(coord)
        rows.append(
            _DirectSourceCallSite(
                uid=_stable_id(
                    "callsite",
                    (
                        target_uid, source_uid, "point", str(getattr(call, "lineno", 0)),
                        str(getattr(call, "col_offset", 0)), expr,
                    ),
                ),
                target_uid=target_uid,
                source_uid=source_uid,
                selector="point",
                coord_expr=_copy_expr(coord),
                guards=tuple(_copy_expr(x) for x in guards),
                lineno=int(getattr(call, "lineno", 0)),
                col_offset=int(getattr(call, "col_offset", 0)),
            )
        )

    def record_window(
        body_call: ast.Call,
        target_uid: str,
        guards: tuple[ast.AST, ...],
        comp: ast.comprehension,
    ) -> bool:
        if not (
            isinstance(body_call.func, ast.Name)
            and body_call.func.id in axis_functions
            and isinstance(comp.target, ast.Name)
            and isinstance(comp.iter, ast.Call)
            and isinstance(comp.iter.func, ast.Name)
            and comp.iter.func.id == "range"
            and not comp.is_async
        ):
            return False
        source_uid = body_call.func.id
        coord = _call_arg_for_param(body_call, axis_functions[source_uid], axis)
        if coord is None:
            return False
        expr = ast.unparse(coord)
        rows.append(
            _DirectSourceCallSite(
                uid=_stable_id(
                    "callsite",
                    (
                        target_uid, source_uid, "window", str(getattr(body_call, "lineno", 0)),
                        str(getattr(body_call, "col_offset", 0)), expr, ast.unparse(comp.iter),
                    ),
                ),
                target_uid=target_uid,
                source_uid=source_uid,
                selector="window",
                coord_expr=_copy_expr(coord),
                guards=tuple(_copy_expr(x) for x in guards),
                lineno=int(getattr(body_call, "lineno", 0)),
                col_offset=int(getattr(body_call, "col_offset", 0)),
                range_call=ast.fix_missing_locations(_copy_expr(comp.iter)),
                generator_uid=comp.target.id,
                generator_filters=tuple(_copy_expr(x) for x in comp.ifs),
            )
        )
        return True

    def walk_expr(node: ast.AST | None, target_uid: str, guards: tuple[ast.AST, ...]) -> None:
        if node is None:
            return
        if isinstance(node, ast.BoolOp):
            prior: list[ast.AST] = []
            for value in node.values:
                if isinstance(node.op, ast.And):
                    extra = tuple(_copy_expr(x) for x in prior)
                else:
                    extra = tuple(_negate_expr(x) for x in prior)
                walk_expr(value, target_uid, guards + extra)
                prior.append(value)
            return
        if isinstance(node, ast.IfExp):
            walk_expr(node.test, target_uid, guards)
            walk_expr(node.body, target_uid, guards + (_copy_expr(node.test),))
            walk_expr(node.orelse, target_uid, guards + (_negate_expr(node.test),))
            return
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"any", "all", "sum", "min", "max"}
                and len(node.args) == 1
                and isinstance(node.args[0], ast.GeneratorExp)
                and len(node.args[0].generators) == 1
                and isinstance(node.args[0].elt, ast.Call)
            ):
                gen = node.args[0]
                comp = gen.generators[0]
                if record_window(gen.elt, target_uid, guards, comp):
                    walk_expr(comp.iter, target_uid, guards)
                    for filt in comp.ifs:
                        walk_expr(filt, target_uid, guards)
                    return
            record_point(node, target_uid, guards)
            for arg in node.args:
                walk_expr(arg, target_uid, guards)
            for keyword in node.keywords:
                walk_expr(keyword.value, target_uid, guards)
            return
        if isinstance(node, ast.GeneratorExp):
            walk_expr(node.elt, target_uid, guards)
            for comp in node.generators:
                walk_expr(comp.iter, target_uid, guards)
                for filt in comp.ifs:
                    walk_expr(filt, target_uid, guards)
            return
        for child in ast.iter_child_nodes(node):
            walk_expr(child, target_uid, guards)

    def walk_statements(
        statements: list[ast.stmt], target_uid: str, guards: tuple[ast.AST, ...]
    ) -> None:
        active = tuple(_copy_expr(x) for x in guards)
        for statement in statements:
            if isinstance(statement, ast.If):
                walk_expr(statement.test, target_uid, active)
                walk_statements(statement.body, target_uid, active + (_copy_expr(statement.test),))
                if statement.orelse:
                    walk_statements(statement.orelse, target_uid, active + (_negate_expr(statement.test),))
                body_ends = _block_always_terminates(statement.body)
                else_ends = bool(statement.orelse) and _block_always_terminates(statement.orelse)
                if body_ends and else_ends:
                    return
                if body_ends and not statement.orelse:
                    active = active + (_negate_expr(statement.test),)
                elif body_ends and statement.orelse and not else_ends:
                    active = active + (_negate_expr(statement.test),)
                elif else_ends and not body_ends:
                    active = active + (_copy_expr(statement.test),)
                continue
            if isinstance(statement, (ast.Return, ast.Raise)):
                if isinstance(statement, ast.Return):
                    walk_expr(statement.value, target_uid, active)
                elif isinstance(statement, ast.Raise):
                    walk_expr(statement.exc, target_uid, active)
                return
            if isinstance(statement, ast.Assign):
                walk_expr(statement.value, target_uid, active)
                continue
            if isinstance(statement, ast.AnnAssign):
                walk_expr(statement.value, target_uid, active)
                continue
            if isinstance(statement, ast.AugAssign):
                walk_expr(statement.value, target_uid, active)
                continue
            if isinstance(statement, ast.Expr):
                walk_expr(statement.value, target_uid, active)
                continue
            if isinstance(statement, (ast.For, ast.While)):
                if isinstance(statement, ast.For):
                    walk_expr(statement.iter, target_uid, active)
                else:
                    walk_expr(statement.test, target_uid, active)
                walk_statements(statement.body, target_uid, active)
                walk_statements(statement.orelse, target_uid, active)
                continue
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.expr):
                    walk_expr(child, target_uid, active)

    for target_uid, fn in sorted(axis_functions.items()):
        walk_statements(fn.body, target_uid, ())
    return tuple(sorted(rows, key=lambda x: (x.target_uid, x.lineno, x.col_offset, x.uid)))


def _single_return_expr(fn: ast.FunctionDef) -> ast.AST | None:
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
        return body[0].value
    return None


def _expand_source_proof_expr(
    node: ast.AST,
    target_fn: ast.FunctionDef,
    functions: dict[str, ast.FunctionDef],
    pure_uids: set[str],
    *,
    stack: tuple[str, ...] = (),
) -> ast.AST:
    """Inline one-shot locals and tiny pure helpers for proof only."""
    store_counts: dict[str, int] = {}
    for child in ast.walk(target_fn):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            store_counts[child.id] = store_counts.get(child.id, 0) + 1
    locals_map: dict[str, ast.AST] = {}
    for statement in target_fn.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and store_counts.get(statement.targets[0].id) == 1
        ):
            locals_map[statement.targets[0].id] = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
            and store_counts.get(statement.target.id) == 1
        ):
            locals_map[statement.target.id] = statement.value

    class Expand(ast.NodeTransformer):
        def __init__(self, current_stack: tuple[str, ...]):
            self.current_stack = current_stack
            self.local_stack: set[str] = set()

        def visit_Name(self, name: ast.Name):
            if (
                isinstance(name.ctx, ast.Load)
                and name.id in locals_map
                and name.id not in self.local_stack
            ):
                self.local_stack.add(name.id)
                value = self.visit(_copy_expr(locals_map[name.id]))
                self.local_stack.remove(name.id)
                return value
            return name

        def visit_Call(self, call: ast.Call):
            call = self.generic_visit(call)
            if not isinstance(call.func, ast.Name):
                return call
            uid = call.func.id
            if uid not in pure_uids or uid in self.current_stack:
                return call
            helper = functions.get(uid)
            if helper is None or not helper.args.args or call.keywords:
                return call
            returned = _single_return_expr(helper)
            params = [arg.arg for arg in helper.args.args]
            if returned is None or len(call.args) != len(params):
                return call
            replacements = {name: _copy_expr(value) for name, value in zip(params, call.args)}

            class Substitute(ast.NodeTransformer):
                def visit_Name(self, name: ast.Name):
                    if isinstance(name.ctx, ast.Load) and name.id in replacements:
                        return _copy_expr(replacements[name.id])
                    return name

            body = ast.fix_missing_locations(Substitute().visit(_copy_expr(returned)))
            return Expand(self.current_stack + (uid,)).visit(body)

    return ast.fix_missing_locations(Expand(stack).visit(_copy_expr(node)))


def _proof_moduli(node: ast.AST) -> tuple[int, ...]:
    values: set[int] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.BinOp)
            and isinstance(child.op, (ast.FloorDiv, ast.Mod))
            and isinstance(child.right, ast.Constant)
            and isinstance(child.right.value, int)
            and not isinstance(child.right.value, bool)
            and int(child.right.value) > 0
        ):
            values.add(int(child.right.value))
    return tuple(sorted(values))


def _proof_static_calls(node: ast.AST) -> set[str]:
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and not child.args
        and not child.keywords
    }


def _eval_proof_affine(node: ast.AST, env: dict[str, _ProofAffine]) -> _ProofAffine | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return _proof_affine(const=int(node.value))
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.args
        and not node.keywords
    ):
        return env.get(node.func.id)
    if isinstance(node, ast.UnaryOp):
        value = _eval_proof_affine(node.operand, env)
        if value is None:
            return None
        if isinstance(node.op, ast.USub):
            return _proof_affine_scale(value, -1)
        if isinstance(node.op, ast.UAdd):
            return value
        return None
    if isinstance(node, ast.BinOp):
        left = _eval_proof_affine(node.left, env)
        right = _eval_proof_affine(node.right, env)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return _proof_affine_add(left, right)
        if isinstance(node.op, ast.Sub):
            return _proof_affine_add(left, right, -1)
        if isinstance(node.op, ast.Mult):
            if not left.coeffs:
                return _proof_affine_scale(right, left.const)
            if not right.coeffs:
                return _proof_affine_scale(left, right.const)
            return None
        if isinstance(node.op, ast.FloorDiv) and not right.coeffs:
            return _proof_affine_div(left, right.const)
        if isinstance(node.op, ast.Mod) and not right.coeffs:
            return _proof_affine_mod(left, right.const)
        return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"min", "max"}
        and node.args
        and not node.keywords
    ):
        values = [_eval_proof_affine(arg, env) for arg in node.args]
        if any(value is None for value in values):
            return None
        typed = [value for value in values if value is not None]
        coeffs = typed[0].coeffs
        if any(value.coeffs != coeffs for value in typed):
            return None
        constants = [value.const for value in typed]
        selected = min(constants) if node.func.id == "min" else max(constants)
        return _ProofAffine(coeffs, selected)
    return None


def _compare_proof_affine(
    left: _ProofAffine | None, right: _ProofAffine | None, operator: ast.cmpop
) -> bool | None:
    if left is None or right is None:
        return None
    diff = _proof_affine_add(left, right, -1)
    if diff.coeffs:
        return None
    value = diff.const
    if isinstance(operator, ast.Eq):
        return value == 0
    if isinstance(operator, ast.NotEq):
        return value != 0
    if isinstance(operator, ast.Lt):
        return value < 0
    if isinstance(operator, ast.LtE):
        return value <= 0
    if isinstance(operator, ast.Gt):
        return value > 0
    if isinstance(operator, ast.GtE):
        return value >= 0
    return None


def _eval_proof_bool(node: ast.AST, env: dict[str, _ProofAffine]) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _eval_proof_bool(node.operand, env)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_eval_proof_bool(value, env) for value in node.values]
        if isinstance(node.op, ast.And):
            if any(value is False for value in values):
                return False
            if all(value is True for value in values):
                return True
            return None
        if any(value is True for value in values):
            return True
        if all(value is False for value in values):
            return False
        return None
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        return _compare_proof_affine(
            _eval_proof_affine(node.left, env),
            _eval_proof_affine(node.comparators[0], env),
            node.ops[0],
        )
    return None


def _exact_periodic_relative_bounds(
    node: ast.AST,
    guards: tuple[ast.AST, ...],
    *,
    axis: str,
) -> tuple[int, int] | None:
    """Prove exact relative bounds for a tiny periodic integer algebra.

    Every floor/mod divisor contributes to a finite residue modulus.  Replacing the
    coordinate and run-static zero-argument integer facts by quotient+residue forms
    is exact for this algebra.  A result is accepted only when ``expr - axis`` loses
    every quotient coefficient for every non-excluded residue class.
    """
    modulus = 1
    for expr in (node,) + guards:
        for value in _proof_moduli(expr):
            modulus = math.lcm(modulus, int(value))
    if modulus > 120:
        return None
    static_uids: set[str] = set()
    for expr in (node,) + guards:
        static_uids.update(_proof_static_calls(expr))
    combinations = modulus ** len(static_uids)
    if combinations > 4096:
        return None
    rows: list[int] = []
    static_names = tuple(sorted(static_uids))
    for static_residues in itertools.product(range(modulus), repeat=len(static_names)):
        for axis_residue in range(modulus):
            env: dict[str, _ProofAffine] = {
                axis: _proof_affine({f"q:{axis}": modulus}, axis_residue)
            }
            for uid, residue in zip(static_names, static_residues):
                env[uid] = _proof_affine({f"q:{uid}": modulus}, int(residue))
            excluded = False
            for guard in guards:
                value = _eval_proof_bool(guard, env)
                if value is False:
                    excluded = True
                    break
            if excluded:
                continue
            value = _eval_proof_affine(node, env)
            if value is None:
                continue
            diff = _proof_affine_add(value, env[axis], -1)
            if diff.coeffs:
                return None
            rows.append(int(diff.const))
    if not rows:
        return None
    return min(rows), max(rows)


def _relative_lower_bound(
    node: ast.AST, guards: tuple[ast.AST, ...], *, axis: str
) -> int | None:
    exact = _exact_periodic_relative_bounds(node, guards, axis=axis)
    if exact is not None:
        return exact[0]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
        if node.func.id == "max":
            values = [_relative_lower_bound(arg, guards, axis=axis) for arg in node.args]
            known = [value for value in values if value is not None]
            return max(known) if known else None
        if node.func.id == "min":
            values = [_relative_lower_bound(arg, guards, axis=axis) for arg in node.args]
            if all(value is not None for value in values):
                return min(int(value) for value in values if value is not None)
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, int)
        and not isinstance(node.right.value, bool)
    ):
        child = _relative_lower_bound(node.left, guards, axis=axis)
        if child is None:
            return None
        if isinstance(node.op, ast.Add):
            return child + int(node.right.value)
        if isinstance(node.op, ast.Sub):
            return child - int(node.right.value)
    return None


def _relative_upper_bound(
    node: ast.AST, guards: tuple[ast.AST, ...], *, axis: str
) -> int | None:
    exact = _exact_periodic_relative_bounds(node, guards, axis=axis)
    if exact is not None:
        return exact[1]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
        if node.func.id == "min":
            values = [_relative_upper_bound(arg, guards, axis=axis) for arg in node.args]
            known = [value for value in values if value is not None]
            return min(known) if known else None
        if node.func.id == "max":
            values = [_relative_upper_bound(arg, guards, axis=axis) for arg in node.args]
            if all(value is not None for value in values):
                return max(int(value) for value in values if value is not None)
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, int)
        and not isinstance(node.right.value, bool)
    ):
        child = _relative_upper_bound(node.left, guards, axis=axis)
        if child is None:
            return None
        if isinstance(node.op, ast.Add):
            return child + int(node.right.value)
        if isinstance(node.op, ast.Sub):
            return child - int(node.right.value)
    return None


def _required_guard_conjuncts(node: ast.AST, *, truth: bool = True) -> tuple[ast.AST, ...]:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _required_guard_conjuncts(node.operand, truth=not truth)
    if isinstance(node, ast.BoolOp):
        if (isinstance(node.op, ast.And) and truth) or (isinstance(node.op, ast.Or) and not truth):
            values: list[ast.AST] = []
            for child in node.values:
                values.extend(_required_guard_conjuncts(child, truth=truth))
            return tuple(values)
    return (_copy_expr(node) if truth else _negate_expr(node),)


def _normalized_guards(guards: tuple[ast.AST, ...]) -> tuple[ast.AST, ...]:
    rows: list[ast.AST] = []
    seen: set[str] = set()
    for guard in guards:
        for row in _required_guard_conjuncts(guard):
            # Collapse double negation after De Morgan expansion.
            while (
                isinstance(row, ast.UnaryOp)
                and isinstance(row.op, ast.Not)
                and isinstance(row.operand, ast.UnaryOp)
                and isinstance(row.operand.op, ast.Not)
            ):
                row = row.operand.operand
            text = ast.unparse(row)
            if text not in seen:
                rows.append(_copy_expr(row))
                seen.add(text)
    return tuple(rows)


def _guards_axis_lower_bound(guards: tuple[ast.AST, ...], *, axis: str) -> int | None:
    lower: int | None = None
    for guard in _normalized_guards(guards):
        negated = isinstance(guard, ast.UnaryOp) and isinstance(guard.op, ast.Not)
        candidate = guard.operand if negated else guard
        if not (
            isinstance(candidate, ast.Compare)
            and len(candidate.ops) == 1
            and len(candidate.comparators) == 1
            and isinstance(candidate.left, ast.Name)
            and candidate.left.id == axis
            and isinstance(candidate.comparators[0], ast.Constant)
            and isinstance(candidate.comparators[0].value, int)
            and not isinstance(candidate.comparators[0].value, bool)
        ):
            continue
        value = int(candidate.comparators[0].value)
        op = candidate.ops[0]
        bound: int | None = None
        if not negated:
            if isinstance(op, ast.GtE):
                bound = value
            elif isinstance(op, ast.Gt):
                bound = value + 1
        else:
            if isinstance(op, ast.Lt):
                bound = value
            elif isinstance(op, ast.LtE):
                bound = value + 1
        if bound is not None:
            lower = bound if lower is None else max(lower, bound)
    return lower


def _literal_range_values(call: ast.Call, *, cap: int = 64) -> tuple[int, ...] | None:
    if not (isinstance(call.func, ast.Name) and call.func.id == "range"):
        return None
    args: list[int] = []
    for node in call.args:
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            return None
        args.append(int(node.value))
    try:
        values = tuple(range(*args))
    except (TypeError, ValueError):
        return None
    return values if len(values) <= cap else None


def _substitute_name(node: ast.AST, name: str, replacement: ast.AST) -> ast.AST:
    class Substitute(ast.NodeTransformer):
        def visit_Name(self, child: ast.Name):
            if isinstance(child.ctx, ast.Load) and child.id == name:
                return _copy_expr(replacement)
            return child
    return ast.fix_missing_locations(Substitute().visit(_copy_expr(node)))


def _point_relative_bounds(
    site: _DirectSourceCallSite,
    target_fn: ast.FunctionDef,
    functions: dict[str, ast.FunctionDef],
    pure_uids: set[str],
    guards: tuple[ast.AST, ...],
    *,
    axis: str,
) -> tuple[int | None, int | None, str, tuple[str, ...]]:
    expr = _expand_source_proof_expr(site.coord_expr, target_fn, functions, pure_uids)
    expanded_guards = tuple(
        _expand_source_proof_expr(guard, target_fn, functions, pure_uids)
        for guard in guards
    )
    exact = _exact_periodic_relative_bounds(expr, expanded_guards, axis=axis)
    if exact is not None:
        return int(exact[0]), int(exact[1]), "source_periodic_relative_v1", ()

    # A zero floor preserves bounded strict-pastness when the unfloored coordinate
    # is already strictly before the caller and the caller is known positive.
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "max"
        and len(expr.args) == 2
    ):
        nonzero: ast.AST | None = None
        if isinstance(expr.args[0], ast.Constant) and expr.args[0].value == 0:
            nonzero = expr.args[1]
        elif isinstance(expr.args[1], ast.Constant) and expr.args[1].value == 0:
            nonzero = expr.args[0]
        if nonzero is not None:
            raw = _exact_periodic_relative_bounds(nonzero, expanded_guards, axis=axis)
            caller_lower = _guards_axis_lower_bound(expanded_guards, axis=axis)
            if raw is not None and raw[1] < 0 and caller_lower is not None and caller_lower >= 1:
                return int(raw[0]), -1, "source_zero_floor_strict_past_v1", ()

    return None, None, "source_dynamic_relative_unknown_v1", (
        f"unproved_point_coordinate:{ast.unparse(expr)}",
    )


def _window_relative_bounds(
    site: _DirectSourceCallSite,
    target_fn: ast.FunctionDef,
    functions: dict[str, ast.FunctionDef],
    pure_uids: set[str],
    guards: tuple[ast.AST, ...],
    *,
    axis: str,
) -> tuple[int | None, int | None, str, tuple[str, ...]]:
    if site.range_call is None or site.generator_uid is None:
        return None, None, "source_window_unknown_v1", ("missing_range_evidence",)
    expanded_guards = tuple(
        _expand_source_proof_expr(guard, target_fn, functions, pure_uids)
        for guard in guards
    )
    literal_values = _literal_range_values(site.range_call)
    if literal_values is not None:
        offsets: list[int] = []
        for value in literal_values:
            expr = _substitute_name(site.coord_expr, site.generator_uid, ast.Constant(value=value))
            expr = _expand_source_proof_expr(expr, target_fn, functions, pure_uids)
            exact = _exact_periodic_relative_bounds(expr, expanded_guards, axis=axis)
            if exact is None:
                return None, None, "source_literal_window_unknown_v1", (
                    f"unproved_literal_window_coordinate:{ast.unparse(expr)}",
                )
            offsets.extend((int(exact[0]), int(exact[1])))
        if offsets:
            return min(offsets), max(offsets), "source_literal_window_offsets_v1", ()
        return None, None, "source_literal_window_unknown_v1", ("empty_literal_window",)

    call = site.range_call
    if not (isinstance(call.func, ast.Name) and call.func.id == "range"):
        return None, None, "source_window_unknown_v1", ("unsupported_window_iterator",)
    args = list(call.args)
    if len(args) == 1:
        start = ast.Constant(value=0)
        stop = args[0]
        step = 1
    elif len(args) in {2, 3}:
        start, stop = args[0], args[1]
        if len(args) == 3:
            if not (
                isinstance(args[2], ast.Constant)
                and isinstance(args[2].value, int)
                and not isinstance(args[2].value, bool)
            ):
                return None, None, "source_window_unknown_v1", ("dynamic_range_step",)
            step = int(args[2].value)
        else:
            step = 1
    else:
        return None, None, "source_window_unknown_v1", ("unsupported_range_arity",)
    if step != 1:
        return None, None, "source_window_unknown_v1", ("nonunit_dynamic_range_step",)
    expanded_start = _expand_source_proof_expr(start, target_fn, functions, pure_uids)
    expanded_stop = _expand_source_proof_expr(stop, target_fn, functions, pure_uids)
    stop_inclusive = ast.fix_missing_locations(
        ast.BinOp(left=expanded_stop, op=ast.Sub(), right=ast.Constant(value=1))
    )
    lo = _relative_lower_bound(expanded_start, expanded_guards, axis=axis)
    hi = _relative_upper_bound(stop_inclusive, expanded_guards, axis=axis)
    if lo is None or hi is None:
        return None, None, "source_dynamic_window_unknown_v1", (
            f"unproved_window_bounds:{ast.unparse(expanded_start)}..{ast.unparse(stop_inclusive)}",
        )
    if isinstance(site.coord_expr, ast.Name) and site.coord_expr.id == site.generator_uid:
        return int(lo), int(hi), "source_dynamic_window_bounds_v1", ()
    return None, None, "source_dynamic_window_unknown_v1", (
        "dynamic_window_body_is_not_iterator_coordinate",
    )


def _availability_for_state_bounds(
    min_offset: int | None,
    max_offset: int | None,
    *,
    transition_relevant: bool,
) -> str:
    if min_offset is None or max_offset is None:
        return "UNKNOWN"
    if max_offset < 0:
        return "PREFIX"
    if min_offset == 0 and max_offset == 0:
        return "CURRENT"
    if max_offset > 0:
        return "SUFFIX" if transition_relevant else "COMPLETE"
    if min_offset < 0 and max_offset == 0:
        return "PREFIX"
    return "UNKNOWN"


def _aggregate_availability(values: Iterable[str]) -> str:
    rows = tuple(values)
    if not rows:
        return "STATIC"
    if "UNKNOWN" in rows:
        return "UNKNOWN"
    if "SUFFIX" in rows:
        return "SUFFIX"
    if "COMPLETE" in rows:
        return "COMPLETE"
    if "PREFIX" in rows:
        return "PREFIX"
    if "CURRENT" in rows:
        return "CURRENT"
    return "STATIC"


def _source_transition_cone(
    structure: SourceStateStructure,
) -> set[str]:
    """Return values whose same-coordinate result feeds a persistent transition."""
    persistent = {row.uid for row in structure.semantics if row.persistence == "persistent"}
    predecessors: dict[str, set[str]] = {}
    for access in structure.accesses:
        if access.selector != "point" or access.order not in {"same", "fixed"}:
            continue
        predecessors.setdefault(access.target_uid, set()).add(access.source_uid)
    cone = set(persistent)
    pending = list(sorted(persistent))
    while pending:
        target = pending.pop(0)
        for source in sorted(predecessors.get(target, ())):
            if source in cone:
                continue
            cone.add(source)
            pending.append(source)
    return cone


def _source_transition_sinks(
    structure: SourceStateStructure,
) -> dict[str, tuple[str, ...]]:
    """Map each same-coordinate predecessor to persistent states it can feed."""
    persistent = sorted(
        row.uid for row in structure.semantics if row.persistence == "persistent"
    )
    predecessors: dict[str, set[str]] = {}
    for access in structure.accesses:
        if access.selector != "point" or access.order not in {"same", "fixed"}:
            continue
        predecessors.setdefault(access.target_uid, set()).add(access.source_uid)
    sinks: dict[str, set[str]] = {uid: {uid} for uid in persistent}
    for sink in persistent:
        pending = [sink]
        seen = {sink}
        while pending:
            target = pending.pop(0)
            for source in sorted(predecessors.get(target, ())):
                sinks.setdefault(source, set()).add(sink)
                if source not in seen:
                    seen.add(source)
                    pending.append(source)
    return {uid: tuple(sorted(values)) for uid, values in sinks.items()}


def analyze_source_callsite_availability(
    source: str,
    *,
    axis: str = "t",
    max_expansion_depth: int = 12,
    max_expanded_calls: int = 128,
    root_targets: Iterable[str] | None = None,
    root_sources: Iterable[str] | None = None,
    include_trivial_same: bool = False,
    caller_domains: Mapping[str, tuple[int, int]] | None = None,
) -> SourceAvailabilityPlan:
    """Prove call-site/subdomain state availability without changing execution.

    Each concrete direct call site gets one evidence row.  For a StateDerivedMap
    callee, the row summarizes its transitive persistent-state footprint under that
    *caller's* guard/subdomain.  This avoids turning the call graph into an exploded
    path graph while still distinguishing the same helper at different callers.
    """
    module = ast.parse(source)
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    axis_functions = _source_axis_functions(module, axis)
    purity = {row.uid: row for row in classify_source_purity(source)}
    pure_uids = {uid for uid, row in purity.items() if row.classification == "pure_map"}
    structure = analyze_source_state_structure(source, axis=axis)
    semantics = {row.uid: row for row in structure.semantics}
    transition_cone = _source_transition_cone(structure)
    transition_sinks = _source_transition_sinks(structure)
    direct = _collect_direct_source_callsites(module, axis_functions, axis=axis)
    proven_caller_domains = {
        str(uid): (int(bounds[0]), int(bounds[1]))
        for uid, bounds in (caller_domains or {}).items()
    }

    def caller_domain_guards(uid: str) -> tuple[ast.AST, ...]:
        bounds = proven_caller_domains.get(uid)
        if bounds is None:
            return ()
        lo, hi = bounds
        return (
            ast.Compare(
                left=ast.Name(id=axis, ctx=ast.Load()),
                ops=[ast.GtE()],
                comparators=[ast.Constant(value=lo)],
            ),
            ast.Compare(
                left=ast.Name(id=axis, ctx=ast.Load()),
                ops=[ast.LtE()],
                comparators=[ast.Constant(value=hi)],
            ),
        )

    by_target: dict[str, list[_DirectSourceCallSite]] = {}
    for site in direct:
        by_target.setdefault(site.target_uid, []).append(site)

    footprint_memo: dict[
        tuple[str, int, int, tuple[str, ...], bool, bool],
        tuple[str, int | None, int | None, tuple[str, ...], tuple[str, ...]],
    ] = {}
    footprint_in_progress: set[tuple[str, int, int, tuple[str, ...], bool, bool]] = set()

    def persistent_one_step_eligible(uid: str) -> bool:
        fn = axis_functions.get(uid)
        if fn is None:
            return False
        positional = list(fn.args.posonlyargs) + list(fn.args.args)
        return (
            len(positional) == 1
            and positional[0].arg == axis
            and fn.args.vararg is None
            and fn.args.kwarg is None
            and not fn.args.kwonlyargs
        )

    def evaluate_site(
        site: _DirectSourceCallSite,
        *,
        context_min: int,
        context_max: int,
        inherited_guards: tuple[ast.AST, ...],
        transition_relevant: bool,
        stack: tuple[str, ...],
        depth: int,
        budget: list[int],
        allow_persistent_unfold: bool,
    ) -> tuple[str, int | None, int | None, tuple[str, ...], tuple[str, ...], str]:
        if budget[0] <= 0:
            return (
                "UNKNOWN", None, None, ("state_derived_expansion_budget_exceeded",),
                (), "source_expansion_budget_v1",
            )
        budget[0] -= 1
        target_fn = axis_functions[site.target_uid]
        local_guards = _normalized_guards(site.guards)
        domain_guards = caller_domain_guards(site.target_uid)
        use_inherited = context_min == 0 and context_max == 0
        proof_guards = _normalized_guards(
            (inherited_guards if use_inherited else ()) + local_guards + domain_guards
        )
        if site.selector == "point":
            rel_min, rel_max, coordinate_proof, proof_blockers = _point_relative_bounds(
                site, target_fn, functions, pure_uids, proof_guards, axis=axis
            )
        else:
            rel_min, rel_max, coordinate_proof, proof_blockers = _window_relative_bounds(
                site, target_fn, functions, pure_uids, proof_guards, axis=axis
            )
        if rel_min is None or rel_max is None:
            abs_min = abs_max = None
        else:
            abs_min = int(context_min) + int(rel_min)
            abs_max = int(context_max) + int(rel_max)

        source_row = semantics.get(site.source_uid)
        source_class = source_row.semantic_class if source_row is not None else "unknown"
        blockers = list(proof_blockers)
        leaves: list[str] = []
        proof_kind = coordinate_proof

        if source_class == "pure_map":
            return "STATIC", None, None, tuple(sorted(set(blockers))), (), proof_kind
        if source_class == "persistent_state":
            direct_availability = _availability_for_state_bounds(
                abs_min, abs_max, transition_relevant=transition_relevant
            )
            if (
                abs_min is not None and abs_max is not None
                and int(abs_min) == 1 and int(abs_max) == 1
                and allow_persistent_unfold
                and persistent_one_step_eligible(site.source_uid)
                and site.source_uid not in stack
                and depth < max_expansion_depth
            ):
                probe_budget = [int(budget[0])]
                unfolded = function_footprint(
                    site.source_uid,
                    context_min=int(abs_min if abs_min is not None else abs_max),
                    context_max=int(abs_max),
                    inherited_guards=proof_guards,
                    transition_relevant=transition_relevant,
                    stack=stack + (site.source_uid,),
                    depth=depth + 1,
                    budget=probe_budget,
                    allow_persistent_unfold=False,
                )
                if (
                    unfolded[0] != "UNKNOWN"
                    and unfolded[2] is not None
                    and int(unfolded[2]) <= 0
                    and not any(
                        item.startswith("future_state_required")
                        for item in unfolded[3]
                    )
                ):
                    budget[0] = probe_budget[0]
                    blockers.extend(unfolded[3])
                    return (
                        unfolded[0], unfolded[1], unfolded[2],
                        tuple(sorted(set(blockers))), unfolded[4],
                        f"{proof_kind}+persistent_one_step_unfold_v1",
                    )
            if direct_availability == "SUFFIX":
                blockers.append("future_state_required_inside_active_persistent_transition")
            return (
                direct_availability, abs_min, abs_max, tuple(sorted(set(blockers))),
                (site.source_uid,), proof_kind,
            )
        if source_class != "state_derived_map":
            blockers.append(f"unproved_source_semantics:{source_class}")
            return "UNKNOWN", None, None, tuple(sorted(set(blockers))), (), proof_kind
        if abs_min is None or abs_max is None:
            blockers.append("state_derived_call_coordinate_unproved")
            return "UNKNOWN", None, None, tuple(sorted(set(blockers))), (), proof_kind
        if depth >= max_expansion_depth:
            blockers.append("state_derived_expansion_depth_exceeded")
            return "UNKNOWN", None, None, tuple(sorted(set(blockers))), (), proof_kind
        if site.source_uid in stack:
            blockers.append("state_derived_expansion_cycle")
            return "UNKNOWN", None, None, tuple(sorted(set(blockers))), (), proof_kind

        nested = function_footprint(
            site.source_uid,
            context_min=abs_min,
            context_max=abs_max,
            inherited_guards=proof_guards,
            transition_relevant=transition_relevant,
            stack=stack + (site.source_uid,),
            depth=depth + 1,
            budget=budget,
            allow_persistent_unfold=allow_persistent_unfold,
        )
        blockers.extend(nested[3])
        leaves.extend(nested[4])
        proof_kind = f"{coordinate_proof}+state_derived_footprint_v1"
        return (
            nested[0], nested[1], nested[2], tuple(sorted(set(blockers))),
            tuple(sorted(set(leaves))), proof_kind,
        )

    def function_footprint(
        uid: str,
        *,
        context_min: int,
        context_max: int,
        inherited_guards: tuple[ast.AST, ...],
        transition_relevant: bool,
        stack: tuple[str, ...],
        depth: int,
        budget: list[int],
        allow_persistent_unfold: bool,
    ) -> tuple[str, int | None, int | None, tuple[str, ...], tuple[str, ...]]:
        guard_key = tuple(ast.unparse(x) for x in _normalized_guards(inherited_guards))
        key = (
            uid, int(context_min), int(context_max), guard_key,
            bool(transition_relevant), bool(allow_persistent_unfold),
        )
        if key in footprint_memo:
            return footprint_memo[key]
        if key in footprint_in_progress:
            return (
                "UNKNOWN", None, None,
                ("state_derived_footprint_cycle",), (),
            )
        footprint_in_progress.add(key)
        rows: list[tuple[str, int | None, int | None, tuple[str, ...], tuple[str, ...], str]] = []
        for child in by_target.get(uid, ()):
            rows.append(
                evaluate_site(
                    child,
                    context_min=context_min,
                    context_max=context_max,
                    inherited_guards=inherited_guards,
                    transition_relevant=transition_relevant,
                    stack=stack,
                    depth=depth,
                    budget=budget,
                    allow_persistent_unfold=allow_persistent_unfold,
                )
            )
        if not rows:
            result = (
                "UNKNOWN", None, None,
                ("state_derived_map_has_no_axis_access_evidence",), (),
            )
            footprint_memo[key] = result
            footprint_in_progress.discard(key)
            return result
        availability = _aggregate_availability(row[0] for row in rows)
        state_rows = [row for row in rows if row[0] != "STATIC"]
        if state_rows and all(row[1] is not None and row[2] is not None for row in state_rows):
            min_offset = min(int(row[1]) for row in state_rows if row[1] is not None)
            max_offset = max(int(row[2]) for row in state_rows if row[2] is not None)
        else:
            min_offset = max_offset = None
        blockers = tuple(sorted({item for row in rows for item in row[3]}))
        leaves = tuple(sorted({item for row in rows for item in row[4]}))
        result = (availability, min_offset, max_offset, blockers, leaves)
        footprint_memo[key] = result
        footprint_in_progress.discard(key)
        return result

    evidence: list[SourceCallSiteAvailability] = []
    summaries: list[SourceCallSiteSummary] = []
    selected_targets = None if root_targets is None else {str(x) for x in root_targets}
    selected_sources = None if root_sources is None else {str(x) for x in root_sources}
    coordinate_sensitive_traits = {
        "window_reduction", "dynamic_coordinate_access",
        "local_statement_loop", "mutation",
    }
    for site in direct:
        if selected_targets is not None and site.target_uid not in selected_targets:
            continue
        if selected_sources is not None and site.source_uid not in selected_sources:
            continue
        source_semantics_for_selection = semantics.get(site.source_uid)
        if selected_targets is None and selected_sources is None and not include_trivial_same:
            direct_offset = (
                _affine_offset(site.coord_expr, axis) if site.selector == "point" else None
            )
            source_class_for_selection = (
                source_semantics_for_selection.semantic_class
                if source_semantics_for_selection is not None else "unknown"
            )
            source_traits_for_selection = (
                set(source_semantics_for_selection.operator_traits)
                if source_semantics_for_selection is not None else set()
            )
            interesting = (
                site.selector == "window"
                or direct_offset is None
                or direct_offset != 0
                or source_class_for_selection in {"persistent_state", "unknown"}
                or bool(source_traits_for_selection & coordinate_sensitive_traits)
            )
            if not interesting:
                continue
        root_target_uid = site.target_uid
        transition_sink_uids = transition_sinks.get(root_target_uid, ())
        transition_relevant = bool(transition_sink_uids)
        root_guards = _normalized_guards(site.guards + caller_domain_guards(site.target_uid))
        result = evaluate_site(
            site,
            context_min=0,
            context_max=0,
            inherited_guards=(),
            transition_relevant=transition_relevant,
            stack=(root_target_uid,),
            depth=0,
            budget=[int(max_expanded_calls)],
            allow_persistent_unfold=True,
        )
        source_row = semantics.get(site.source_uid)
        target_row = semantics.get(site.target_uid)
        guard_text = tuple(ast.unparse(x) for x in root_guards)
        target_subdomain = " and ".join(guard_text) if guard_text else "full_axis_domain"
        target_fn = axis_functions[site.target_uid]
        if site.selector == "point":
            relation_min, relation_max, _relation_proof, _relation_blockers = _point_relative_bounds(
                site, target_fn, functions, pure_uids, root_guards, axis=axis
            )
        else:
            relation_min, relation_max, _relation_proof, _relation_blockers = _window_relative_bounds(
                site, target_fn, functions, pure_uids, root_guards, axis=axis
            )
        evidence_uid = _stable_id(
            "availability",
            (
                site.uid, result[0], str(result[1]), str(result[2]), target_subdomain,
                str(transition_relevant),
            ),
        )
        evidence.append(
            SourceCallSiteAvailability(
                uid=evidence_uid,
                callsite_uid=site.uid,
                root_callsite_uid=site.uid,
                root_target_uid=root_target_uid,
                target_uid=site.target_uid,
                source_uid=site.source_uid,
                call_path=(site.target_uid, site.source_uid),
                selector=site.selector,
                index_expr=ast.unparse(site.coord_expr),
                lineno=site.lineno,
                col_offset=site.col_offset,
                guard_exprs=guard_text,
                target_subdomain=target_subdomain,
                source_semantic_class=(
                    source_row.semantic_class if source_row is not None else "unknown"
                ),
                target_semantic_class=(
                    target_row.semantic_class if target_row is not None else "unknown"
                ),
                transition_relevant=transition_relevant,
                transition_sink_uids=transition_sink_uids,
                availability=result[0],
                relation_min_offset=relation_min,
                relation_max_offset=relation_max,
                min_offset=result[1],
                max_offset=result[2],
                proof_kind=result[5],
                persistent_source_uids=result[4],
                blockers=result[3],
                note=(
                    "direct call-site evidence; StateDerivedMap rows summarize transitive "
                    "persistent reads under this caller subdomain"
                ),
            )
        )
        summaries.append(
            SourceCallSiteSummary(
                callsite_uid=site.uid,
                root_target_uid=root_target_uid,
                source_uid=site.source_uid,
                call_path=(site.target_uid, site.source_uid),
                guard_exprs=guard_text,
                transition_relevant=transition_relevant,
                transition_sink_uids=transition_sink_uids,
                availability=result[0],
                min_offset=result[1],
                max_offset=result[2],
                persistent_source_uids=result[4],
                blockers=result[3],
                note="aggregate persistent-state footprint for this concrete direct call",
            )
        )

    return SourceAvailabilityPlan(
        schema="modelx_graph.domain_graph.source_callsite_availability.v1",
        axis=axis,
        evidence=tuple(sorted(evidence, key=lambda x: (x.root_target_uid, x.lineno, x.col_offset, x.uid))),
        summaries=tuple(sorted(summaries, key=lambda x: (x.root_target_uid, x.callsite_uid))),
        transition_cone_uids=tuple(sorted(transition_cone)),
        validation_notes=(
            "proof-only: authoritative ExecutableGraph scheduling/storage/codegen are unchanged",
            "evidence defaults to availability-sensitive call sites; trivial same-coordinate pure/derived calls are omitted unless requested",
            "concrete AST call sites retain short-circuit and post-return caller subdomains",
            "StateDerivedMap calls summarize persistent-state footprint under the caller subdomain",
            "optional caller_domains carry stronger source/canonical domain proofs into call-site geometry without product-specific compiler rules",
            "future state in an active persistent-transition cone is SUFFIX/fail-closed; post-processing future reads require COMPLETE history",
        ),
    )


def _contracted_persistent_edges_from_structure(
    structure: SourceStateStructure,
) -> tuple[tuple[str, str], ...]:
    """Recover the complete persistent condensation edges without changing V02342.

    V02342 intentionally stored only internal edges on each persistent component so
    its evidence manifest would stay compact.  The component scheduler also needs
    cross-component dataflow.  Recompute that condensation from the already-frozen
    source accesses instead of adding fields to ``SourceStateStructure`` and thereby
    perturbing the earlier evidence artifact.
    """
    semantics = {row.uid: row for row in structure.semantics}
    persistent = {
        uid for uid, row in semantics.items() if row.persistence == "persistent"
    }
    raw_nodes = {
        uid for uid, row in semantics.items() if row.semantic_class != "pure_map"
    }
    adjacency: dict[str, set[str]] = {uid: set() for uid in raw_nodes}
    for access in structure.accesses:
        if not access.scheduling_relevant:
            continue
        if access.source_uid not in raw_nodes or access.target_uid not in raw_nodes:
            continue
        adjacency.setdefault(access.source_uid, set()).add(access.target_uid)

    edges: set[tuple[str, str]] = set()
    for source_uid in sorted(persistent):
        pending = list(sorted(adjacency.get(source_uid, ())))
        visited: set[str] = set()
        while pending:
            target_uid = pending.pop(0)
            if target_uid in visited:
                continue
            visited.add(target_uid)
            if target_uid in persistent:
                edges.add((source_uid, target_uid))
                continue
            pending.extend(
                child for child in sorted(adjacency.get(target_uid, ()))
                if child not in visited
            )
    return tuple(sorted(edges))


def _scan_requirement_from_access(access: SourceAccessEvidence) -> str:
    if access.min_offset is not None and access.max_offset is not None:
        lo = int(access.min_offset)
        hi = int(access.max_offset)
        if hi < 0:
            return "ascending"
        if lo > 0:
            return "descending"
        if lo == 0 and hi == 0:
            return "any"
        return "mixed"
    if access.order in {"strict_before", "before_or_same"}:
        return "ascending"
    if access.order == "strict_after":
        return "descending"
    if access.order == "same":
        return "any"
    return "unknown"


def _graph_path_exists(
    adjacency: Mapping[str, set[str]], source: str, target: str
) -> bool:
    if source == target:
        return True
    pending = [source]
    seen: set[str] = set()
    while pending:
        uid = pending.pop(0)
        if uid in seen:
            continue
        seen.add(uid)
        for child in sorted(adjacency.get(uid, ())):
            if child == target:
                return True
            if child not in seen:
                pending.append(child)
    return False


def _topological_component_order(
    nodes: Iterable[str], edges: Iterable[tuple[str, str]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    nodes = tuple(sorted(set(nodes)))
    outgoing: dict[str, set[str]] = {uid: set() for uid in nodes}
    indegree = {uid: 0 for uid in nodes}
    for source, target in sorted(set(edges)):
        if source == target or source not in indegree or target not in indegree:
            continue
        if target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1
    ready = sorted(uid for uid in nodes if indegree[uid] == 0)
    order: list[str] = []
    while ready:
        uid = ready.pop(0)
        order.append(uid)
        for child in sorted(outgoing[uid]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) == len(nodes):
        return tuple(order), ()
    cyclic = tuple(sorted(uid for uid in nodes if indegree[uid] > 0))
    return tuple(order), cyclic


def _assign_component_stages(
    nodes: Iterable[str],
    dependency_edges: Iterable[tuple[str, str]],
    barrier_edges: Iterable[tuple[str, str]],
) -> tuple[dict[str, int], tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Assign minimal stages after collapsing co-schedulable dependency SCCs.

    Ordinary CURRENT/PREFIX component dependencies constrain *within-stage* dataflow
    but do not by themselves require a materialization phase.  Such dependencies may
    legitimately be cyclic for mapped/event state, as MYGA already proves.  A true
    completion barrier is stricter: its producer history must exist before the
    consumer stage starts.

    Therefore the correct graph operation is weighted SCC condensation:

    * collapse SCCs of the complete component dependency graph;
    * reject any barrier whose endpoints lie in one SCC;
    * assign weight 0 to ordinary cross-SCC dependencies and weight 1 to barriers;
    * take the longest weighted path through the condensation DAG.

    This is generic graph scheduling evidence.  It does not encode product/domain
    names and remains proof-only.
    """
    node_tuple = tuple(sorted(set(nodes)))
    dep_edges = {
        (source, target) for source, target in dependency_edges
        if source in node_tuple and target in node_tuple and source != target
    }
    barrier_set = {
        (source, target) for source, target in barrier_edges
        if source in node_tuple and target in node_tuple and source != target
    }
    components = _tarjan_source_components(node_tuple, dep_edges)
    group_by_node: dict[str, str] = {}
    members_by_group: dict[str, tuple[str, ...]] = {}
    for members in components:
        group_uid = _stable_id("schedule_component_scc", members)
        members_by_group[group_uid] = members
        for uid in members:
            group_by_node[uid] = group_uid

    blockers: list[str] = []
    for source, target in sorted(barrier_set):
        if group_by_node.get(source) == group_by_node.get(target):
            blockers.append(
                f"component_barrier_inside_dependency_scc:{source}->{target}"
            )

    weighted_edges: dict[tuple[str, str], int] = {}
    for source, target in sorted(dep_edges):
        source_group = group_by_node[source]
        target_group = group_by_node[target]
        if source_group == target_group:
            continue
        weight = 1 if (source, target) in barrier_set else 0
        edge = (source_group, target_group)
        weighted_edges[edge] = max(weighted_edges.get(edge, 0), weight)

    groups = tuple(sorted(members_by_group))
    topo, cyclic = _topological_component_order(groups, weighted_edges)
    if cyclic:
        # Tarjan condensation must be acyclic.  Keep this fail-closed invariant in
        # case future graph changes violate the assumptions above.
        blockers.append("component_condensation_cycle:" + ",".join(cyclic))

    stage_by_group = {uid: 0 for uid in groups}
    if not cyclic:
        incoming: dict[str, list[tuple[str, int]]] = {uid: [] for uid in groups}
        for (source_group, target_group), weight in weighted_edges.items():
            incoming.setdefault(target_group, []).append((source_group, weight))
        for uid in topo:
            for source_group, weight in incoming.get(uid, ()):
                stage_by_group[uid] = max(
                    stage_by_group[uid], stage_by_group[source_group] + int(weight)
                )

    stage_by_node = {
        uid: stage_by_group.get(group_by_node[uid], 0) for uid in node_tuple
    }
    return stage_by_node, components, tuple(sorted(set(blockers)))


def _transition_frontier_availability_rows(
    structure: SourceStateStructure,
    availability: SourceAvailabilityPlan,
) -> tuple[SourceCallSiteAvailability, ...]:
    """Keep the caller-nearest availability proof for each state-source/sink pair.

    V02343 deliberately records every availability-sensitive direct call as a
    possible proof root.  An internal StateDerivedMap therefore also has a
    full-domain row even when all of its actual persistent-transition callers add
    stronger guards.  The component scheduler must reason at those concrete caller
    sites, not accidentally reintroduce the helper's unconstrained internal row.

    For each persistent leaf -> transition sink pair, retain rows whose target is
    closest to the sink along same-coordinate transition dataflow.  Ties are kept,
    preserving distinct guarded callers.  Unknown rows with no resolved persistent
    leaf use their source identity as a conservative key and are filtered by the
    same caller-nearest rule.
    """
    adjacency: dict[str, set[str]] = {}
    for access in structure.accesses:
        if access.selector != "point" or access.order not in {"same", "fixed"}:
            continue
        adjacency.setdefault(access.source_uid, set()).add(access.target_uid)

    distance_cache: dict[tuple[str, str], int | None] = {}

    def distance(source_uid: str, sink_uid: str) -> int | None:
        key = (source_uid, sink_uid)
        if key in distance_cache:
            return distance_cache[key]
        if source_uid == sink_uid:
            distance_cache[key] = 0
            return 0
        pending = [(source_uid, 0)]
        seen: set[str] = set()
        while pending:
            uid, depth = pending.pop(0)
            if uid in seen:
                continue
            seen.add(uid)
            for child in sorted(adjacency.get(uid, ())):
                if child == sink_uid:
                    distance_cache[key] = depth + 1
                    return depth + 1
                if child not in seen:
                    pending.append((child, depth + 1))
        distance_cache[key] = None
        return None

    keyed_rows: list[tuple[SourceCallSiteAvailability, tuple[tuple[str, str, int], ...]]] = []
    minimum: dict[tuple[str, str], int] = {}
    for row in availability.evidence:
        if not row.transition_relevant or not row.transition_sink_uids:
            continue
        sources = row.persistent_source_uids or (f"unresolved:{row.source_uid}",)
        keys: list[tuple[str, str, int]] = []
        for source_uid in sources:
            for sink_uid in row.transition_sink_uids:
                dist = distance(row.target_uid, sink_uid)
                if dist is None:
                    continue
                keys.append((source_uid, sink_uid, dist))
                pair = (source_uid, sink_uid)
                minimum[pair] = min(minimum.get(pair, dist), dist)
        keyed_rows.append((row, tuple(keys)))

    keep: list[SourceCallSiteAvailability] = []
    for row, keys in keyed_rows:
        if not keys:
            # No same-coordinate route to the claimed sink could be reconstructed.
            # Keep the row so the scheduler fails closed rather than discarding it.
            keep.append(row)
            continue
        if any(minimum[(source_uid, sink_uid)] == dist for source_uid, sink_uid, dist in keys):
            keep.append(row)
    return tuple(sorted(keep, key=lambda x: x.uid))


def analyze_source_component_schedule(
    source: str,
    *,
    axis: str = "t",
    caller_domains: Mapping[str, tuple[int, int]] | None = None,
    max_expanded_calls: int = 32,
    max_expansion_depth: int = 6,
    availability_plan: SourceAvailabilityPlan | None = None,
) -> ComponentSchedulePlan:
    """Build a proof-only schedule over persistent components and availability.

    The schedule is deliberately downstream of V02342 persistence and V02343
    call-site availability.  It may insert a materialization barrier between two
    *different* persistent components, but it never legalizes a future read inside
    one persistent component.  Unknown transition-relevant evidence remains a hard
    blocker.  No backend consumes this plan in the current foundation.
    """
    structure = analyze_source_state_structure(source, axis=axis)
    availability = availability_plan or analyze_source_callsite_availability(
        source,
        axis=axis,
        caller_domains=caller_domains,
        max_expanded_calls=max_expanded_calls,
        max_expansion_depth=max_expansion_depth,
    )
    semantics = {row.uid: row for row in structure.semantics}
    access_by_uid = {row.uid: row for row in structure.accesses}

    component_by_value: dict[str, str] = {}
    components: list[ComponentScheduleComponent] = []
    component_scan: dict[str, str] = {}
    component_blockers: dict[str, list[str]] = {}
    for component in structure.persistent_components:
        recurrence_uids: set[str] = set()
        scan_requirements: set[str] = set()
        blockers: list[str] = []
        for uid in component.persistent_uids:
            component_by_value[uid] = component.uid
            row = semantics.get(uid)
            if row is None:
                blockers.append(f"missing_persistent_semantics:{uid}")
                continue
            for access_uid in row.recurrence_access_uids:
                recurrence_uids.add(access_uid)
                access = access_by_uid.get(access_uid)
                if access is None:
                    blockers.append(f"missing_recurrence_access:{access_uid}")
                    continue
                requirement = _scan_requirement_from_access(access)
                if requirement == "any":
                    continue
                if requirement in {"ascending", "descending"}:
                    scan_requirements.add(requirement)
                else:
                    blockers.append(
                        f"persistent_recurrence_scan_{requirement}:{access.source_uid}->{access.target_uid}"
                    )
        if len(scan_requirements) > 1:
            scan = "mixed"
            blockers.append("persistent_component_mixed_scan_direction")
        elif scan_requirements:
            scan = next(iter(scan_requirements))
        elif recurrence_uids:
            scan = "unknown"
            blockers.append("persistent_component_scan_unknown")
        else:
            scan = "any"
        component_scan[component.uid] = scan
        component_blockers[component.uid] = blockers
        components.append(
            ComponentScheduleComponent(
                uid=component.uid,
                persistent_uids=component.persistent_uids,
                intrinsic_scan=scan,
                recurrence_access_uids=tuple(sorted(recurrence_uids)),
                blockers=tuple(sorted(set(blockers))),
                note=(
                    "persistent-state component from V02342; StateDerivedMaps remain contracted"
                ),
            )
        )

    contracted_edges = _contracted_persistent_edges_from_structure(structure)
    pair_rows: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for source_uid, target_uid in contracted_edges:
        source_component = component_by_value.get(source_uid)
        target_component = component_by_value.get(target_uid)
        if source_component is None or target_component is None:
            continue
        if source_component == target_component:
            continue
        pair_rows.setdefault((source_component, target_component), []).append(
            (source_uid, target_uid)
        )

    constraints: list[ComponentScheduleConstraint] = []
    graph_edges: set[tuple[str, str]] = set(pair_rows)
    barrier_edges: set[tuple[str, str]] = set()
    global_blockers: list[str] = [
        blocker
        for component in components
        for blocker in component.blockers
    ]
    deferred_unknowns: list[str] = []

    for (source_component, target_component), rows in sorted(pair_rows.items()):
        source_scan = component_scan.get(source_component, "unknown")
        target_scan = component_scan.get(target_component, "unknown")
        scan_incompatible = (
            source_scan in {"ascending", "descending"}
            and target_scan in {"ascending", "descending"}
            and source_scan != target_scan
        )
        barrier = scan_incompatible
        if barrier:
            barrier_edges.add((source_component, target_component))
        constraints.append(
            ComponentScheduleConstraint(
                uid=_stable_id(
                    "component_constraint",
                    (source_component, target_component, "persistent_flow"),
                ),
                source_component_uid=source_component,
                target_component_uid=target_component,
                kind="persistent_flow",
                availability="CURRENT_OR_PREFIX",
                barrier_required=barrier,
                source_uids=tuple(sorted({x[0] for x in rows})),
                target_uids=tuple(sorted({x[1] for x in rows})),
                proof_kind="source_persistent_condensation_v1",
                note=(
                    "cross-component persistent dataflow; incompatible intrinsic scans require completion barrier"
                    if barrier else
                    "cross-component persistent dataflow can remain in one dependency stage"
                ),
            )
        )

    # Availability relations can strengthen the condensation graph with explicit
    # prefix/current requirements or turn a cross-component future requirement into
    # a completed-history barrier.  They can never weaken a same-component suffix.
    transition_rows = _transition_frontier_availability_rows(structure, availability)
    for row in transition_rows:
        if not row.transition_relevant:
            continue
        source_components = tuple(sorted({
            component_by_value[uid]
            for uid in row.persistent_source_uids
            if uid in component_by_value
        }))
        sink_components = tuple(sorted({
            component_by_value[uid]
            for uid in row.transition_sink_uids
            if uid in component_by_value
        }))

        if row.availability == "STATIC":
            continue
        if row.availability == "UNKNOWN":
            global_blockers.append(
                f"transition_availability_unknown:{row.target_uid}->{row.source_uid}:{row.callsite_uid}"
            )
            continue
        if not source_components:
            # CURRENT/PREFIX can be proven without a persistent leaf (for example a
            # static/runtime fact); no component ordering follows from such a row.
            if row.availability in {"SUFFIX", "COMPLETE"}:
                global_blockers.append(
                    f"future_transition_has_no_persistent_source_component:{row.callsite_uid}"
                )
            continue
        if not sink_components:
            global_blockers.append(
                f"transition_sink_component_unresolved:{row.callsite_uid}"
            )
            continue

        for source_component in source_components:
            for sink_component in sink_components:
                same_component = source_component == sink_component
                barrier = False
                blockers: list[str] = []
                if row.availability in {"SUFFIX", "COMPLETE"}:
                    if same_component:
                        blockers.append("same_component_future_state_requirement")
                        global_blockers.append(
                            f"same_component_future_state:{row.target_uid}->{row.source_uid}:{source_component}"
                        )
                    else:
                        barrier = True
                elif row.availability == "PREFIX":
                    sink_scan = component_scan.get(sink_component, "unknown")
                    if same_component and sink_scan == "descending":
                        blockers.append("prefix_requirement_conflicts_with_descending_component")
                        global_blockers.append(
                            f"prefix_in_descending_component:{row.callsite_uid}:{sink_component}"
                        )
                    elif not same_component and sink_scan == "descending":
                        barrier = True

                if not same_component:
                    graph_edges.add((source_component, sink_component))
                if barrier:
                    barrier_edges.add((source_component, sink_component))
                constraints.append(
                    ComponentScheduleConstraint(
                        uid=_stable_id(
                            "component_constraint",
                            (
                                source_component, sink_component, row.callsite_uid,
                                row.availability,
                            ),
                        ),
                        source_component_uid=source_component,
                        target_component_uid=sink_component,
                        kind="availability",
                        availability=row.availability,
                        barrier_required=barrier,
                        source_uids=tuple(sorted(row.persistent_source_uids)),
                        target_uids=tuple(sorted(row.transition_sink_uids)),
                        evidence_uids=(row.uid,),
                        proof_kind=row.proof_kind,
                        blockers=tuple(blockers),
                        note=(
                            "future requirement crosses persistent components and is reinterpretable only after producer completion"
                            if barrier else
                            "availability requirement is compatible with the component-local scan"
                        ),
                    )
                )

    # CURRENT/PREFIX dataflow may form a legitimate co-scheduled SCC (for example
    # mapped annual state inside a monthly projection).  Only a completion barrier
    # imposes a strict stage boundary.  Collapse the complete dependency graph first,
    # then carry barrier edges as weight-1 edges through the condensation DAG.
    component_uids = tuple(x.uid for x in components)
    stage_by_component, dependency_sccs, stage_blockers = _assign_component_stages(
        component_uids, graph_edges, barrier_edges
    )
    global_blockers.extend(stage_blockers)

    # Also report the exact barrier edges that oppose an already-proven reverse
    # dependency path.  This names the semantic conflict behind an SCC blocker.
    base_adjacency: dict[str, set[str]] = {uid: set() for uid in component_uids}
    for source_component, target_component in graph_edges:
        base_adjacency.setdefault(source_component, set()).add(target_component)
    for source_component, target_component in sorted(barrier_edges):
        reduced = {uid: set(children) for uid, children in base_adjacency.items()}
        reduced.setdefault(source_component, set()).discard(target_component)
        if _graph_path_exists(reduced, target_component, source_component):
            global_blockers.append(
                f"component_barrier_reverse_path:{source_component}->{target_component}"
            )

    # Non-transition StateDerivedMaps are scheduled from their completed footprint.
    # Unknown post-state work is retained as deferred evidence rather than poisoning
    # a sound persistent-state component schedule.
    derived_groups: dict[str, list[SourceCallSiteAvailability]] = {}
    for row in availability.evidence:
        if row.transition_relevant:
            continue
        if row.source_semantic_class not in {"state_derived_map", "persistent_state", "unknown"}:
            continue
        if not row.persistent_source_uids and row.availability != "UNKNOWN":
            continue
        derived_groups.setdefault(row.root_target_uid, []).append(row)

    derived_tasks: list[DerivedScheduleTask] = []
    for target_uid, rows in sorted(derived_groups.items()):
        aggregate = _aggregate_availability(row.availability for row in rows)
        persistent_sources = tuple(sorted({
            uid for row in rows for uid in row.persistent_source_uids
        }))
        producer_components = tuple(sorted({
            component_by_value[uid]
            for uid in persistent_sources
            if uid in component_by_value
        }))
        blockers = tuple(sorted({
            blocker for row in rows for blocker in row.blockers
            if blocker != "future_state_required_inside_active_persistent_transition"
        }))
        if aggregate == "UNKNOWN":
            stage_index = None
            deferred_unknowns.append(f"derived_task_unknown:{target_uid}")
        elif producer_components:
            base_stage = max(stage_by_component.get(uid, 0) for uid in producer_components)
            stage_index = base_stage + (1 if aggregate in {"COMPLETE", "SUFFIX"} else 0)
        else:
            stage_index = 0
        materialized = (
            persistent_sources if aggregate in {"COMPLETE", "SUFFIX"} else ()
        )
        derived_tasks.append(
            DerivedScheduleTask(
                uid=_stable_id("derived_schedule_task", (target_uid, aggregate)),
                target_uid=target_uid,
                availability=aggregate,
                producer_component_uids=producer_components,
                stage_index=stage_index,
                persistent_source_uids=persistent_sources,
                evidence_uids=tuple(sorted(row.uid for row in rows)),
                materialized_source_uids=materialized,
                blockers=blockers,
                note=(
                    "completed-history StateDerivedMap" if aggregate in {"COMPLETE", "SUFFIX"}
                    else "state-derived work placed from persistent availability evidence"
                ),
            )
        )

    # Materialization evidence belongs to the producer stage, not the consumer.
    materialized_by_stage: dict[int, set[str]] = {}
    for constraint in constraints:
        if not constraint.barrier_required:
            continue
        source_stage = stage_by_component.get(constraint.source_component_uid, 0)
        sources = constraint.source_uids or next(
            (
                component.persistent_uids
                for component in components
                if component.uid == constraint.source_component_uid
            ),
            (),
        )
        materialized_by_stage.setdefault(source_stage, set()).update(sources)
    for task in derived_tasks:
        if task.stage_index is None:
            continue
        for producer in task.producer_component_uids:
            producer_stage = stage_by_component.get(producer, 0)
            if task.stage_index > producer_stage:
                materialized_by_stage.setdefault(producer_stage, set()).update(
                    task.materialized_source_uids
                )

    stage_indices = set(stage_by_component.values())
    stage_indices.update(
        task.stage_index for task in derived_tasks if task.stage_index is not None
    )
    stages: list[ComponentScheduleStage] = []
    for index in sorted(stage_indices or {0}):
        members = tuple(sorted(
            uid for uid, stage_index in stage_by_component.items() if stage_index == index
        ))
        directions = tuple(sorted({
            component_scan[uid]
            for uid in members
            if component_scan.get(uid) not in {None, "any"}
        }))
        tasks = tuple(sorted(
            task.uid for task in derived_tasks if task.stage_index == index
        ))
        stages.append(
            ComponentScheduleStage(
                index=index,
                component_uids=members,
                scan_directions=directions,
                derived_task_uids=tasks,
                materialized_source_uids=tuple(sorted(materialized_by_stage.get(index, ()))),
                note="barrier-count stage; components retain their own intrinsic scan proof",
            )
        )

    blockers = tuple(sorted(set(global_blockers)))
    deferred = tuple(sorted(set(deferred_unknowns)))
    persistent_proven = not blockers
    full_proven = persistent_proven and not deferred and all(
        not task.blockers for task in derived_tasks if task.availability != "UNKNOWN"
    )
    return ComponentSchedulePlan(
        schema="modelx_graph.domain_graph.source_component_schedule.v1",
        axis=axis,
        components=tuple(sorted(components, key=lambda x: x.uid)),
        constraints=tuple(sorted(constraints, key=lambda x: x.uid)),
        derived_tasks=tuple(sorted(derived_tasks, key=lambda x: x.uid)),
        stages=tuple(stages),
        persistent_schedule_proven=persistent_proven,
        full_schedule_proven=full_proven,
        blockers=blockers,
        deferred_unknowns=deferred,
        validation_notes=(
            "proof-only: ExecutableGraph/storage/codegen/native dispatch remain authoritative and unchanged",
            "persistent components come only from V02342 semantics; StateDerivedMaps are never promoted by the scheduler",
            "cross-component SUFFIX is legal only as producer-before-consumer completed-history materialization and only in an acyclic component DAG",
            "same-component SUFFIX and transition-relevant UNKNOWN remain fail-closed",
        ),
    )


_CANONICAL_STATE_INDEPENDENT_CALLS = {
    # Lowered immutable/runtime inputs.  They may vary by run key or coordinate,
    # but they do not read persistent projection state.
    "global_input", "point_input", "array_input", "table_input", "lookup_input",
    "__interval_lookup_1d__", "__step_lookup_1d__", "__guard_fail__",
    "__exact_axis_code__", "__sparse_table_lookup__", "__prepared_point_lookup__",
    # Numeric/control helpers retained by canonical lowering.
    "abs", "all", "any", "bool", "enumerate", "float", "int", "len",
    "max", "min", "range", "round", "sum", "tuple", "zip",
}


def _canonical_variant_nodes(cv: Any) -> tuple[ast.AST, ...]:
    nodes: list[ast.AST] = []
    fn = getattr(cv, "function", None)
    if fn is not None:
        nodes.extend(fn.body)
    reduction = getattr(cv, "reduction", None)
    if reduction is not None:
        nodes.extend([
            reduction.init, reduction.body_expr,
            *tuple(reduction.filters), *tuple(reduction.range_args),
        ])
    return tuple(nodes)


def _canonical_direct_calls(cv: Any, variants: Mapping[str, Any]) -> tuple[tuple[ast.Call, Any], ...]:
    out: list[tuple[ast.Call, Any]] = []
    for root in _canonical_variant_nodes(cv):
        for node in ast.walk(root):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in variants
            ):
                out.append((node, variants[node.func.id]))
    return tuple(out)


def _canonical_external_calls(cv: Any, variants: Mapping[str, Any]) -> tuple[str, ...]:
    names: set[str] = set()
    for root in _canonical_variant_nodes(cv):
        for node in ast.walk(root):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name not in variants and name not in _CANONICAL_STATE_INDEPENDENT_CALLS:
                    names.add(name)
            elif isinstance(node.func, ast.Attribute):
                root_name = _root_name(node.func)
                if root_name not in _PURE_MODULE_ROOTS:
                    names.add(ast.unparse(node.func))
            else:
                names.add(ast.unparse(node.func))
    return tuple(sorted(names))


def _canonical_intrinsic_state_footprints(
    snapshot: Any,
    *,
    persistent_source_uids: set[str],
) -> dict[str, tuple[tuple[tuple[str, int], ...], tuple[str, ...]]]:
    """Return persistent leaves of each canonical variant relative to its own axis.

    Canonical variants are already branch/aux/fixed-coordinate specialized by the
    production frontend.  The proof therefore walks the *actual specialized formula*
    rather than raw source.  Persistent source names are inherited unchanged from the
    V02342 state-semantics proof.  This function never promotes or demotes state.
    """
    variants = getattr(snapshot, "variants", {})
    canonical_static_source_names = {
        str(row.source_name) for row in getattr(snapshot, "static_facts", ())
    } | {
        str(row.source_name) for row in getattr(snapshot, "validated_static_facts", ())
    }
    memo: dict[str, tuple[tuple[tuple[str, int], ...], tuple[str, ...]]] = {}
    in_progress: set[str] = set()

    def visit(uid: str) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
        if uid in memo:
            return memo[uid]
        cv = variants.get(uid)
        if cv is None:
            return (), (f"canonical_variant_missing:{uid}",)
        if uid in in_progress:
            return (), (f"canonical_state_footprint_cycle:{uid}",)

        # The production frontend has already proved these source Cells invariant
        # over the entire compiled run-key domain.  Their canonical body may still
        # contain a pruned helper UID that is no longer reachable; the explicit
        # StaticScalarFact is stronger evidence than re-walking that dead source.
        if str(getattr(cv, "source_name", "")) in canonical_static_source_names:
            result = ((), ())
            memo[uid] = result
            return result

        # A carried-state canonical coordinate is the leaf.  Do not recursively
        # inspect its definition and accidentally turn transition logic into another
        # persistence classification.
        if (
            getattr(cv, "role", None) == "coordinate"
            and getattr(cv, "source_name", None) in persistent_source_uids
        ):
            result = (((str(cv.source_name), 0),), ())
            memo[uid] = result
            return result

        in_progress.add(uid)
        leaves: set[tuple[str, int]] = set()
        blockers: set[str] = set()
        blockers.update(
            f"canonical_unproved_external_call:{name}"
            for name in _canonical_external_calls(cv, variants)
        )
        parent_axis = getattr(cv, "time_param_source", None)
        for call, child in _canonical_direct_calls(cv, variants):
            child_leaves, child_blockers = visit(str(child.uid))
            blockers.update(child_blockers)
            if not child_leaves:
                # A state-independent child imposes no scheduling availability,
                # even when its coordinate is a derived/non-affine expression.
                continue
            if getattr(child, "role", None) != "coordinate":
                blockers.add(
                    f"canonical_stateful_scalar_dependency:{cv.source_name}->{child.source_name}"
                )
                continue
            if not call.args or parent_axis is None:
                blockers.add(
                    f"canonical_coordinate_context_unproved:{cv.source_name}->{child.source_name}"
                )
                continue
            offset = _affine_offset(call.args[0], str(parent_axis))
            if offset is None:
                blockers.add(
                    "canonical_coordinate_relation_unproved:"
                    f"{cv.source_name}->{child.source_name}:{ast.unparse(call.args[0])}"
                )
                continue
            for source_name, child_offset in child_leaves:
                leaves.add((source_name, int(offset) + int(child_offset)))

        in_progress.discard(uid)
        result = (tuple(sorted(leaves)), tuple(sorted(blockers)))
        memo[uid] = result
        return result

    for uid in sorted(variants):
        visit(uid)
    return memo


def _snapshot_overlay_inventory_facts(
    source_structure: SourceStateStructure,
    snapshot: Any,
) -> list[CanonicalOverlayFact]:
    """Expose existing compiler proof objects as typed overlay inventory."""
    proof_size = int(getattr(snapshot, "run_key_count", 0) or 0)
    variants = getattr(snapshot, "variants", {})
    facts: list[CanonicalOverlayFact] = []

    for row in source_structure.semantics:
        if row.semantic_class != "pure_map":
            continue
        facts.append(CanonicalOverlayFact(
            uid=_stable_id("overlay_fact", ("source_pure_map", row.uid)),
            fact_kind="source_pure_map",
            source_uid=row.uid,
            availability="STATIC",
            proof_domain_size=proof_size,
            provenance=(row.proof_kind,),
            note="V02340/V02342 source PureMap evidence carried into the overlay inventory",
        ))

    for row in getattr(snapshot, "static_facts", ()):
        facts.append(CanonicalOverlayFact(
            uid=_stable_id("overlay_fact", ("static_scalar", row.source_kind, row.source_name)),
            fact_kind="static_scalar",
            source_uid=str(row.source_name),
            availability="STATIC",
            proof_domain_size=int(getattr(row, "proof_domain_size", proof_size)),
            provenance=(f"{row.source_kind}:{row.source_name}={row.value!r}",),
            note="frontend-proven scalar invariant over the compiled run-key domain",
        ))
    for row in getattr(snapshot, "validated_static_facts", ()):
        facts.append(CanonicalOverlayFact(
            uid=_stable_id("overlay_fact", ("validated_static", row.source_name)),
            fact_kind="validated_static_scalar",
            source_uid=str(row.source_name),
            availability="STATIC",
            proof_domain_size=int(getattr(row, "proof_domain_size", proof_size)),
            provenance=(f"validated:{row.source_name}={row.value!r}",),
            note="runtime-validated static selector used by canonical branch specialization",
        ))
    for row in getattr(snapshot, "finite_domain_facts", ()):
        facts.append(CanonicalOverlayFact(
            uid=_stable_id("overlay_fact", ("finite_domain", row.source_kind, row.source_name)),
            fact_kind="finite_domain",
            source_uid=str(row.source_name),
            proof_domain_size=int(getattr(row, "proof_domain_size", proof_size)),
            provenance=(f"values={tuple(row.values)!r}",),
            note="frontend-proven finite domain",
        ))
    for row in getattr(snapshot, "fixed_coordinate_facts", ()):
        facts.append(CanonicalOverlayFact(
            uid=_stable_id("overlay_fact", ("fixed_coordinate", row.source_name, str(row.coordinate))),
            fact_kind="fixed_coordinate",
            source_uid=str(row.source_name),
            canonical_uids=(str(row.synthetic_uid),),
            coordinate=int(row.coordinate),
            proof_domain_size=proof_size,
            provenance=(f"source_uid:{row.source_uid}", f"synthetic_uid:{row.synthetic_uid}"),
            note="frontend specialized this exact source coordinate to a canonical scalar",
        ))
    for row in getattr(snapshot, "runtime_scalar_helper_facts", ()):
        facts.append(CanonicalOverlayFact(
            uid=_stable_id("overlay_fact", ("runtime_scalar_helper", row.source_name)),
            fact_kind="runtime_scalar_helper",
            source_uid=str(row.source_name),
            availability="STATIC",
            proof_domain_size=proof_size,
            provenance=("frontend_runtime_scalar_helper",),
            note="pure scalar helper already removed from scheduled coordinate state by the frontend",
        ))
    for row in getattr(snapshot, "source_backed_scheduled_facts", ()):
        facts.append(CanonicalOverlayFact(
            uid=_stable_id("overlay_fact", ("source_backed_scheduled", row.synthetic_uid)),
            fact_kind="source_backed_scheduled",
            source_uid=str(row.source_name),
            canonical_uids=(str(row.synthetic_uid),),
            proof_domain_size=proof_size,
            provenance=(f"time_param:{row.time_param}", f"aux_values:{tuple(row.aux_values)!r}"),
            note="canonical scheduled value reconstructed from source rather than representative trace",
        ))
    for row in getattr(snapshot, "step_lookup_domain_facts", ()):
        facts.append(CanonicalOverlayFact(
            uid=_stable_id("overlay_fact", ("step_lookup_domain", row.source_uid, row.query_expr)),
            fact_kind="step_lookup_domain",
            source_uid=str(row.source_name),
            canonical_uids=(str(row.source_uid),),
            proof_domain_size=proof_size,
            provenance=(
                f"query:{row.query_expr}",
                f"bounds:{row.proven_lower_bound!r}..{row.proven_upper_bound!r}",
                str(row.proof_kind),
            ),
            note="proven coordinate/query domain for normalized step lookup",
        ))
    for uid, lo, hi in getattr(snapshot, "coordinate_domains", ()):
        cv = variants.get(uid)
        facts.append(CanonicalOverlayFact(
            uid=_stable_id("overlay_fact", ("coordinate_domain", str(uid))),
            fact_kind="coordinate_domain",
            source_uid=(None if cv is None else str(cv.source_name)),
            canonical_uids=(str(uid),),
            proof_domain_size=proof_size,
            provenance=(f"inclusive_domain:{int(lo)}..{int(hi)}",),
            note="exact specialized canonical coordinate domain",
        ))
    return facts


def apply_canonical_evidence_overlay(
    source: str,
    snapshot: Any,
    *,
    axis: str = "t",
    caller_domains: Mapping[str, tuple[int, int]] | None = None,
    max_expanded_calls: int = 32,
    max_expansion_depth: int = 6,
    raw_availability: SourceAvailabilityPlan | None = None,
) -> tuple[SourceAvailabilityPlan, tuple[CanonicalOverlayFact, ...], tuple[CanonicalOverlayApplication, ...]]:
    """Strengthen raw UNKNOWN call-site evidence using existing canonical proofs.

    The operation is monotone by construction: only ``UNKNOWN`` rows may change.
    Already-proven PREFIX/CURRENT/SUFFIX/COMPLETE/STATIC relations are copied byte-for-
    byte at the semantic field level.  Canonical specialization can therefore resolve
    uncertainty, but cannot erase a previously proved future-state requirement.
    """
    proof_size = int(getattr(snapshot, "run_key_count", 0) or 0)
    if proof_size <= 0:
        raise DomainGraphError("canonical overlay requires a non-empty proven run-key domain")
    variants: Mapping[str, Any] = getattr(snapshot, "variants", {})
    if not variants:
        raise DomainGraphError("canonical overlay snapshot contains no variants")

    structure = analyze_source_state_structure(source, axis=axis)
    raw = raw_availability or analyze_source_callsite_availability(
        source,
        axis=axis,
        caller_domains=caller_domains,
        max_expanded_calls=max_expanded_calls,
        max_expansion_depth=max_expansion_depth,
    )
    persistent_names = {
        row.uid for row in structure.semantics if row.persistence == "persistent"
    }
    footprints = _canonical_intrinsic_state_footprints(
        snapshot, persistent_source_uids=persistent_names
    )
    target_variants: dict[str, list[Any]] = {}
    for cv in variants.values():
        target_variants.setdefault(str(cv.source_name), []).append(cv)

    facts = _snapshot_overlay_inventory_facts(structure, snapshot)
    applications: list[CanonicalOverlayApplication] = []
    replacements: dict[str, SourceCallSiteAvailability] = {}
    fixed_by_source_coordinate: dict[tuple[str, int], set[str]] = {}
    fixed_synthetic_uids: set[str] = set()
    for fixed in getattr(snapshot, "fixed_coordinate_facts", ()):
        key = (str(fixed.source_name), int(fixed.coordinate))
        fixed_by_source_coordinate.setdefault(key, set()).add(str(fixed.synthetic_uid))
        fixed_synthetic_uids.add(str(fixed.synthetic_uid))

    for row in raw.evidence:
        # Monotonicity guard: canonical evidence is an UNKNOWN resolver only in this
        # foundation.  A proved suffix/complete relation remains exactly as proved.
        if row.availability != "UNKNOWN":
            continue
        matching_calls: list[tuple[Any, ast.Call, Any]] = []
        for target_cv in target_variants.get(row.target_uid, ()):
            for call, child_cv in _canonical_direct_calls(target_cv, variants):
                if str(child_cv.source_name) == row.source_uid:
                    matching_calls.append((target_cv, call, child_cv))
        # Exact fixed-coordinate evidence outranks another ordinary call to the
        # same source family elsewhere in the target formula.  Conversely, a raw
        # non-literal call must not accidentally inherit a synthetic fixed branch.
        raw_fixed_coordinate: int | None = None
        try:
            parsed_index = ast.parse(str(row.index_expr), mode="eval").body
            if (
                isinstance(parsed_index, ast.Constant)
                and isinstance(parsed_index.value, int)
                and not isinstance(parsed_index.value, bool)
            ):
                raw_fixed_coordinate = int(parsed_index.value)
        except (SyntaxError, TypeError, ValueError):
            raw_fixed_coordinate = None
        if raw_fixed_coordinate is not None:
            exact_fixed = fixed_by_source_coordinate.get(
                (row.source_uid, raw_fixed_coordinate), set()
            )
            if exact_fixed:
                matching_calls = [
                    triple for triple in matching_calls
                    if str(triple[2].uid) in exact_fixed
                ]
        else:
            matching_calls = [
                triple for triple in matching_calls
                if str(triple[2].uid) not in fixed_synthetic_uids
            ]

        if not matching_calls:
            # Absence may mean dead-branch specialization, but it may also mean an
            # inlined helper.  Do not infer unreachability from absence alone.
            continue

        leaves: set[tuple[str, int]] = set()
        blockers: set[str] = set()
        root_offsets: list[int] = []
        canonical_uids: set[str] = set()
        provenance: set[str] = set()
        for target_cv, call, child_cv in matching_calls:
            canonical_uids.update((str(target_cv.uid), str(child_cv.uid)))
            child_leaves, child_blockers = footprints.get(
                str(child_cv.uid), ((), (f"canonical_footprint_missing:{child_cv.uid}",))
            )
            blockers.update(child_blockers)
            if not child_leaves:
                provenance.add(f"state_independent_variant:{child_cv.uid}")
                continue
            if getattr(child_cv, "role", None) != "coordinate":
                blockers.add(
                    f"canonical_root_stateful_scalar:{target_cv.source_name}->{child_cv.source_name}"
                )
                continue
            target_axis = getattr(target_cv, "time_param_source", None)
            if not call.args or target_axis is None:
                blockers.add(
                    f"canonical_root_coordinate_context_unproved:{target_cv.source_name}->{child_cv.source_name}"
                )
                continue
            offset = _affine_offset(call.args[0], str(target_axis))
            if offset is None:
                blockers.add(
                    "canonical_root_coordinate_relation_unproved:"
                    f"{target_cv.source_name}->{child_cv.source_name}:{ast.unparse(call.args[0])}"
                )
                continue
            root_offsets.append(int(offset))
            for source_name, child_offset in child_leaves:
                leaves.add((source_name, int(offset) + int(child_offset)))
            provenance.add(
                f"specialized_call:{target_cv.uid}->{child_cv.uid}:{ast.unparse(call.args[0])}"
            )

        if blockers:
            fact = CanonicalOverlayFact(
                uid=_stable_id("overlay_callsite_fact", (row.callsite_uid, "UNKNOWN")),
                fact_kind="canonical_callsite_state_footprint",
                source_uid=row.source_uid,
                target_uid=row.target_uid,
                canonical_uids=tuple(sorted(canonical_uids)),
                availability="UNKNOWN",
                proof_domain_size=proof_size,
                provenance=tuple(sorted(provenance)),
                blockers=tuple(sorted(blockers)),
                note="canonical formula was inspected but did not prove a complete persistent-state footprint",
            )
            facts.append(fact)
            continue

        if leaves:
            min_offset = min(offset for _uid, offset in leaves)
            max_offset = max(offset for _uid, offset in leaves)
            availability = _availability_for_state_bounds(
                min_offset, max_offset, transition_relevant=row.transition_relevant
            )
            persistent_sources = tuple(sorted({uid for uid, _off in leaves}))
        else:
            min_offset = max_offset = None
            availability = "STATIC"
            persistent_sources = ()
        if availability == "UNKNOWN":
            continue
        relation_min = min(root_offsets) if root_offsets else None
        relation_max = max(root_offsets) if root_offsets else None
        fact = CanonicalOverlayFact(
            uid=_stable_id(
                "overlay_callsite_fact",
                (row.callsite_uid, availability, str(min_offset), str(max_offset)),
            ),
            fact_kind="canonical_specialized_callsite_footprint",
            source_uid=row.source_uid,
            target_uid=row.target_uid,
            canonical_uids=tuple(sorted(canonical_uids)),
            availability=availability,
            relation_min_offset=relation_min,
            relation_max_offset=relation_max,
            min_offset=min_offset,
            max_offset=max_offset,
            persistent_source_uids=persistent_sources,
            proof_domain_size=proof_size,
            provenance=tuple(sorted(provenance)) or ("canonical_state_independent_formula",),
            note=(
                "specialized canonical formula supersedes raw-source UNKNOWN geometry within this exact run-key proof domain"
            ),
        )
        facts.append(fact)
        replacement = replace(
            row,
            availability=availability,
            relation_min_offset=relation_min,
            relation_max_offset=relation_max,
            min_offset=min_offset,
            max_offset=max_offset,
            proof_kind="canonical_specialized_callsite_footprint_v1",
            persistent_source_uids=persistent_sources,
            blockers=(),
            note=(
                row.note
                + "; raw UNKNOWN strengthened by exact canonical specialization over the compiled run-key domain"
            ),
        )
        replacements[row.callsite_uid] = replacement
        applications.append(CanonicalOverlayApplication(
            callsite_uid=row.callsite_uid,
            target_uid=row.target_uid,
            source_uid=row.source_uid,
            fact_uid=fact.uid,
            before_availability=row.availability,
            after_availability=availability,
            before_min_offset=row.min_offset,
            before_max_offset=row.max_offset,
            after_min_offset=min_offset,
            after_max_offset=max_offset,
            persistent_source_uids=persistent_sources,
            changed=True,
            reason="raw UNKNOWN replaced only because the production frontend produced an exact specialized canonical call/footprint",
        ))

    overlaid_evidence = tuple(
        replacements.get(row.callsite_uid, row) for row in raw.evidence
    )
    replacements_by_callsite = {
        row.callsite_uid: row for row in overlaid_evidence
    }
    overlaid_summaries: list[SourceCallSiteSummary] = []
    for summary in raw.summaries:
        evidence_row = replacements_by_callsite.get(summary.callsite_uid)
        if evidence_row is None or summary.availability != "UNKNOWN":
            overlaid_summaries.append(summary)
            continue
        overlaid_summaries.append(replace(
            summary,
            availability=evidence_row.availability,
            min_offset=evidence_row.min_offset,
            max_offset=evidence_row.max_offset,
            persistent_source_uids=evidence_row.persistent_source_uids,
            blockers=evidence_row.blockers,
            note=(
                summary.note
                + "; strengthened by canonical specialized call-site footprint"
            ),
        ))

    overlaid = SourceAvailabilityPlan(
        schema="modelx_graph.domain_graph.source_callsite_availability.canonical_overlay.v1",
        axis=raw.axis,
        evidence=tuple(overlaid_evidence),
        summaries=tuple(overlaid_summaries),
        transition_cone_uids=raw.transition_cone_uids,
        validation_notes=raw.validation_notes + (
            "canonical overlay is monotone: only raw UNKNOWN rows are eligible for strengthening",
            "overlay proof scope is the exact canonical compiled run-key domain and is not generalized beyond it",
        ),
    )
    return (
        overlaid,
        tuple(sorted(facts, key=lambda x: x.uid)),
        tuple(sorted(applications, key=lambda x: x.callsite_uid)),
    )


def analyze_source_component_schedule_with_canonical_overlay(
    source: str,
    snapshot: Any,
    *,
    axis: str = "t",
    caller_domains: Mapping[str, tuple[int, int]] | None = None,
    max_expanded_calls: int = 32,
    max_expansion_depth: int = 6,
) -> CanonicalOverlaySchedulePlan:
    """Overlay production canonical proofs, then rebuild the V02344 shadow plan."""
    raw = analyze_source_callsite_availability(
        source,
        axis=axis,
        caller_domains=caller_domains,
        max_expanded_calls=max_expanded_calls,
        max_expansion_depth=max_expansion_depth,
    )
    overlaid, facts, applications = apply_canonical_evidence_overlay(
        source,
        snapshot,
        axis=axis,
        caller_domains=caller_domains,
        max_expanded_calls=max_expanded_calls,
        max_expansion_depth=max_expansion_depth,
        raw_availability=raw,
    )
    schedule = analyze_source_component_schedule(
        source,
        axis=axis,
        caller_domains=caller_domains,
        max_expanded_calls=max_expanded_calls,
        max_expansion_depth=max_expansion_depth,
        availability_plan=overlaid,
    )
    return CanonicalOverlaySchedulePlan(
        schema="modelx_graph.domain_graph.canonical_overlay_schedule.v1",
        axis=axis,
        proof_scope="canonical_run_key_domain",
        proof_domain_size=int(getattr(snapshot, "run_key_count", 0) or 0),
        raw_availability=raw,
        overlaid_availability=overlaid,
        facts=facts,
        applications=applications,
        component_schedule=schedule,
        validation_notes=(
            "proof-only: production frontend/scheduler/storage/codegen/native dispatch are unchanged",
            "canonical variants and proof facts come from the normal frontend immediately before executable scheduling",
            "V02342 persistence is held fixed; canonical overlay only resolves raw call-site availability UNKNOWNs",
            "existing proved SUFFIX/PREFIX/CURRENT/COMPLETE/STATIC relations are never weakened or rewritten",
        ),
    )

def analyze_source_phase_constraints(source: str, *, axis: str = "t") -> SourcePhasePlan:
    """Derive proof-only scan/pass constraints from a source formula graph.

    Only functions explicitly parameterized by ``axis`` participate.  Calls from a
    proven pure map remain in the evidence but are excluded from SCC and direction
    analysis.  Stateful accesses are decomposed into SCCs; an SCC with both past and
    future edges, or with an unresolved/window access inside the SCC, fails closed.

    Cross-SCC accesses may instead require a phase barrier.  The returned phase plan
    says only that the *scan-direction problem* can be decomposed.  It is not execution
    permission and deliberately says nothing about storage/codegen support for the
    resulting cross-phase values.
    """
    module = ast.parse(source)
    axis_functions = _source_axis_functions(module, axis)
    purity = {x.uid: x for x in classify_source_purity(source)}
    accesses = list(_collect_source_axis_accesses(
        module,
        axis_functions,
        {uid: row.classification for uid, row in purity.items()},
        axis=axis,
    ))

    scheduled_nodes = {
        uid for uid in axis_functions
        if purity[uid].classification != "pure_map"
    }
    relevant_accesses = [
        a for a in accesses
        if a.scheduling_relevant
        and a.source_uid in scheduled_nodes
        and a.target_uid in scheduled_nodes
    ]
    components_raw = _tarjan_source_components(
        scheduled_nodes,
        ((a.source_uid, a.target_uid) for a in relevant_accesses),
    )
    component_uid_by_member: dict[str, str] = {}
    raw_by_uid: dict[str, tuple[str, ...]] = {}
    for members in components_raw:
        uid = _stable_id("source_scc", members)
        raw_by_uid[uid] = members
        for member in members:
            component_uid_by_member[member] = uid

    internal: dict[str, list[SourceAccessEvidence]] = {uid: [] for uid in raw_by_uid}
    cross: list[SourceAccessEvidence] = []
    for access in relevant_accesses:
        source_component = component_uid_by_member[access.source_uid]
        target_component = component_uid_by_member[access.target_uid]
        if source_component == target_component:
            internal[source_component].append(access)
        else:
            cross.append(access)

    components: list[SourcePhaseComponent] = []
    required_by_component: dict[str, str] = {}
    for uid, members in sorted(raw_by_uid.items()):
        rows = internal[uid]
        orders = {x.order for x in rows}
        blockers: list[str] = []
        has_before = "strict_before" in orders
        has_after = "strict_after" in orders
        has_dynamic = bool(orders.intersection({"window", "unknown"}))
        if has_dynamic:
            required = "unknown"
            for row in rows:
                if row.order in {"window", "unknown"}:
                    blockers.append(
                        f"intra_scc_{row.order}:{row.source_uid}->{row.target_uid}"
                    )
        elif has_before and has_after:
            required = "mixed"
            blockers.append("intra_scc_mixed_scan_direction")
        elif has_before:
            required = "ascending"
        elif has_after:
            required = "descending"
        else:
            required = "any"
        same_cycles = [
            x for x in rows
            if x.order == "same" and x.source_uid == x.target_uid
        ]
        note = ""
        if same_cycles:
            note = (
                "contains same-coordinate source recursion; this analysis does not "
                "grant local recursion execution permission"
            )
        components.append(
            SourcePhaseComponent(
                uid=uid,
                member_uids=members,
                internal_access_uids=tuple(sorted(x.uid for x in rows)),
                required_scan=required,
                blockers=tuple(sorted(set(blockers))),
                note=note,
            )
        )
        required_by_component[uid] = required

    barrier_reason: dict[str, str] = {}
    for access in cross:
        source_component = component_uid_by_member[access.source_uid]
        target_component = component_uid_by_member[access.target_uid]
        compatible = (
            _scan_set(required_by_component[source_component])
            & _scan_set(required_by_component[target_component])
            & _access_scan_set(access.order)
        )
        if not compatible:
            barrier_reason[access.uid] = (
                "cross_scc_relation_requires_materialized_source"
                if access.order in {"window", "unknown"}
                else "cross_scc_scan_direction_incompatible"
            )

    def phase_assignment() -> tuple[dict[str, int], list[str]]:
        pair_weight: dict[tuple[str, str], int] = {}
        adjacency: dict[str, set[str]] = {uid: set() for uid in raw_by_uid}
        indegree: dict[str, int] = {uid: 0 for uid in raw_by_uid}
        for access in cross:
            source_component = component_uid_by_member[access.source_uid]
            target_component = component_uid_by_member[access.target_uid]
            pair = (source_component, target_component)
            pair_weight[pair] = max(
                pair_weight.get(pair, 0),
                1 if access.uid in barrier_reason else 0,
            )
            if target_component not in adjacency[source_component]:
                adjacency[source_component].add(target_component)
                indegree[target_component] += 1
        ready = sorted(uid for uid, deg in indegree.items() if deg == 0)
        topo: list[str] = []
        while ready:
            uid = ready.pop(0)
            topo.append(uid)
            for child in sorted(adjacency[uid]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(topo) != len(raw_by_uid):
            return {}, ["component_condensation_cycle"]
        phase = {uid: 0 for uid in raw_by_uid}
        for uid in topo:
            for child in sorted(adjacency[uid]):
                phase[child] = max(
                    phase[child], phase[uid] + pair_weight[(uid, child)]
                )
        return phase, []

    # Resolve a common real-product shape conservatively: an intrinsically ascending
    # state SCC can feed a stateless coordinate helper, after which a t+1 read would
    # otherwise force descending within the same physical pass.  Promote that opposing
    # cross-SCC access to a barrier and recompute.  Ambiguous no-intrinsic conflicts
    # remain fail-closed rather than choosing a preferred direction arbitrarily.
    phase_blockers: list[str] = []
    phase: dict[str, int] = {}
    for _ in range(max(4, 2 * max(1, len(raw_by_uid)))):
        phase, topo_blockers = phase_assignment()
        phase_blockers.extend(topo_blockers)
        if topo_blockers:
            break
        changed = False
        phase_ids = sorted(set(phase.values()))
        for phase_index in phase_ids:
            phase_components = {uid for uid, p in phase.items() if p == phase_index}
            same_phase_cross = [
                a for a in cross
                if component_uid_by_member[a.source_uid] in phase_components
                and component_uid_by_member[a.target_uid] in phase_components
                and a.uid not in barrier_reason
            ]
            # Connected groups matter; unrelated ascending and descending SCCs need not
            # share one eventual executable pass.
            undirected: dict[str, set[str]] = {uid: set() for uid in phase_components}
            for access in same_phase_cross:
                s = component_uid_by_member[access.source_uid]
                t = component_uid_by_member[access.target_uid]
                undirected[s].add(t)
                undirected[t].add(s)
            unseen = set(phase_components)
            while unseen:
                seed = min(unseen)
                stack = [seed]
                group: set[str] = set()
                while stack:
                    uid = stack.pop()
                    if uid in group:
                        continue
                    group.add(uid)
                    unseen.discard(uid)
                    stack.extend(sorted(undirected[uid] - group))
                intrinsic = {
                    required_by_component[uid]
                    for uid in group
                    if required_by_component[uid] in {"ascending", "descending"}
                }
                group_accesses = [
                    a for a in same_phase_cross
                    if component_uid_by_member[a.source_uid] in group
                    and component_uid_by_member[a.target_uid] in group
                ]
                access_dirs = {
                    "ascending" if a.order == "strict_before" else "descending"
                    for a in group_accesses
                    if a.order in {"strict_before", "strict_after"}
                }
                directions = intrinsic | access_dirs
                if len(directions) <= 1:
                    continue
                if len(intrinsic) == 1:
                    preferred = next(iter(intrinsic))
                    opposite_order = (
                        "strict_after" if preferred == "ascending" else "strict_before"
                    )
                    conflicting = [a for a in group_accesses if a.order == opposite_order]
                    if conflicting:
                        for access in conflicting:
                            barrier_reason.setdefault(
                                access.uid,
                                f"same_phase_conflict_with_intrinsic_{preferred}",
                            )
                        changed = True
                        break
                phase_blockers.append(
                    "unresolved_same_phase_direction_conflict:"
                    + ",".join(sorted(group))
                )
            if changed:
                break
        if not changed:
            break
    else:
        phase_blockers.append("phase_barrier_analysis_did_not_converge")

    if not phase:
        phase, topo_blockers = phase_assignment()
        phase_blockers.extend(topo_blockers)

    constraints: list[SourcePhaseConstraint] = []
    for access in sorted(cross, key=lambda x: x.uid):
        source_component = component_uid_by_member[access.source_uid]
        target_component = component_uid_by_member[access.target_uid]
        compatible = tuple(sorted(
            _scan_set(required_by_component[source_component])
            & _scan_set(required_by_component[target_component])
            & _access_scan_set(access.order)
        ))
        is_barrier = (
            access.uid in barrier_reason
            or phase.get(source_component, 0) < phase.get(target_component, 0)
        )
        reason = barrier_reason.get(access.uid)
        if reason is None and is_barrier:
            reason = "crosses_phase_due_to_other_dependency_barrier"
        if reason is None:
            reason = "same_phase_compatible"
        constraints.append(
            SourcePhaseConstraint(
                access_uid=access.uid,
                source_component_uid=source_component,
                target_component_uid=target_component,
                barrier_required=is_barrier,
                compatible_same_pass_scans=compatible,
                reason=reason,
            )
        )

    phase_rows: list[SourcePhase] = []
    if phase:
        for phase_index in sorted(set(phase.values())):
            phase_components = sorted(uid for uid, p in phase.items() if p == phase_index)
            scan_dirs = {
                required_by_component[uid]
                for uid in phase_components
                if required_by_component[uid] in {"ascending", "descending"}
            }
            for access in cross:
                s = component_uid_by_member[access.source_uid]
                t = component_uid_by_member[access.target_uid]
                if phase.get(s) != phase_index or phase.get(t) != phase_index:
                    continue
                if access.order == "strict_before":
                    scan_dirs.add("ascending")
                elif access.order == "strict_after":
                    scan_dirs.add("descending")
            cross_sources = sorted({
                access.source_uid
                for access in relevant_accesses
                if phase.get(component_uid_by_member[access.source_uid], 0) < phase.get(
                    component_uid_by_member[access.target_uid], 0
                )
                and phase.get(component_uid_by_member[access.source_uid], 0) == phase_index
            })
            phase_rows.append(
                SourcePhase(
                    index=phase_index,
                    component_uids=tuple(phase_components),
                    scan_directions=tuple(sorted(scan_dirs)),
                    cross_phase_source_uids=tuple(cross_sources),
                    note=(
                        "phase index is a proof-only dependency barrier layer; disconnected "
                        "components may later become separate physical passes"
                    ),
                )
            )

    component_blockers = sorted({
        blocker for component in components for blocker in component.blockers
    })
    blockers = tuple(sorted(set(component_blockers + phase_blockers)))
    scan_decomposable = not blockers
    pure_future = [
        a for a in accesses
        if not a.scheduling_relevant and a.order == "strict_after"
    ]
    barrier_count = sum(1 for x in constraints if x.barrier_required)
    validation_notes = (
        "only axis-parameterized source functions participate in this proof; scalar/reduction scheduling remains outside this source shadow",
        "pure-map accesses are retained as evidence but excluded from SCC and scan-direction constraints",
        f"identified {len(pure_future)} pure future access(es) and {barrier_count} cross-SCC phase-barrier access(es)",
        "scan_decomposable means only that scan-direction conflicts can be separated structurally; storage/codegen remain authoritative elsewhere",
    )
    return SourcePhasePlan(
        schema="modelx_graph.domain_graph.source_phase_constraints.v1",
        axis=axis,
        accesses=tuple(accesses),
        components=tuple(sorted(components, key=lambda x: x.uid)),
        constraints=tuple(constraints),
        phases=tuple(phase_rows),
        scan_decomposable=scan_decomposable,
        blockers=blockers,
        validation_notes=validation_notes,
    )

def classify_domain_graph_purity(
    graph: DomainGraph, variants: dict[str, Any]
) -> tuple[PurityEvidence, ...]:
    """Classify canonical DomainGraph values with scheduling/storage evidence.

    Canonical ASTs have already normalized inputs/tables and numeric control flow.
    DomainGraph persistence and non-same self accesses therefore provide the decisive
    statefulness evidence; dependency purity is closed transitively.
    """
    incoming: dict[str, set[str]] = {}
    for access in graph.accesses:
        incoming.setdefault(access.target_uid, set()).add(access.source_uid)
    persistent = {uid for k in graph.kernels for uid in k.persistent_uids}
    self_state = {
        a.source_uid for a in graph.accesses
        if a.source_uid == a.target_uid and a.order in {"strict_before", "before_or_same", "strict_after", "mixed", "unknown"}
    }
    result: dict[str, PurityEvidence] = {}
    pending = set(dict(graph.uid_domain)) | set(variants)
    while pending:
        progressed = False
        for uid in sorted(tuple(pending)):
            deps = tuple(sorted(d for d in incoming.get(uid, ()) if d != uid))
            blockers: list[str] = []
            if uid in persistent:
                blockers.append("persistent_storage")
            if uid in self_state:
                blockers.append("delayed_or_persistent_self_access")
            cv = variants.get(uid)
            if cv is not None and getattr(cv, "role", None) == "reduction":
                blockers.append("reduction_kernel")
            unresolved = [d for d in deps if d in pending]
            if blockers:
                state = "stateful"
            elif unresolved:
                continue
            else:
                dep_states = [result[d].classification for d in deps if d in result]
                if any(x == "stateful" for x in dep_states):
                    state = "stateful"
                    blockers.extend(f"dependency_stateful:{d}" for d in deps if d in result and result[d].classification == "stateful")
                elif any(x == "unknown" for x in dep_states):
                    state = "unknown"
                    blockers.extend(f"dependency_unknown:{d}" for d in deps if d in result and result[d].classification == "unknown")
                else:
                    state = "pure_map"
            result[uid] = PurityEvidence(
                uid=uid, classification=state, proof_kind="domain_graph_transitive_purity_v1",
                dependencies=deps, blockers=tuple(sorted(set(blockers))),
                note="canonical DomainGraph purity evidence; execution authority unchanged",
            )
            pending.remove(uid); progressed = True
        if not progressed:
            for uid in sorted(pending):
                result[uid] = PurityEvidence(
                    uid=uid, classification="stateful", proof_kind="domain_graph_transitive_purity_v1",
                    dependencies=tuple(sorted(incoming.get(uid, ()))), blockers=("dependency_cycle",),
                    note="fail-closed cyclic canonical dependency",
                )
            break
    return tuple(result[k] for k in sorted(result))


def classify_domain_graph_value_semantics(
    graph: DomainGraph,
    variants: dict[str, Any],
) -> tuple[ValueSemanticsEvidence, ...]:
    """Split canonical DomainGraph values into persistent and state-derived roles.

    Existing ``Kernel.persistent_uids`` are treated as the stronger canonical proof
    of carried state.  This is deliberately not re-inferred from source syntax, so
    every currently native graph preserves exactly the persistent set already proved
    by the authoritative executable/template scheduler.  State dependence then
    propagates through canonical access relations; non-persistent dependents become
    ``state_derived_map`` values.

    Like the rest of DomainGraph in this foundation, the result is evidence-only.
    """
    purity = {x.uid: x for x in classify_domain_graph_purity(graph, variants)}
    persistent = {uid for kernel in graph.kernels for uid in kernel.persistent_uids}
    incoming: dict[str, set[str]] = {}
    outgoing: dict[str, list[AccessRelation]] = {}
    for access in graph.accesses:
        incoming.setdefault(access.target_uid, set()).add(access.source_uid)
        outgoing.setdefault(access.source_uid, []).append(access)

    all_uids = set(dict(graph.uid_domain)) | set(variants) | set(purity)
    reads_state = set(persistent)
    changed = True
    while changed:
        changed = False
        for uid in sorted(all_uids):
            if uid in reads_state:
                continue
            if any(dep in reads_state for dep in incoming.get(uid, ())):
                reads_state.add(uid)
                changed = True

    rows: list[ValueSemanticsEvidence] = []
    for uid in sorted(all_uids):
        purity_row = purity.get(uid)
        purity_class = "unknown" if purity_row is None else purity_row.classification
        if uid in persistent:
            persistence = "persistent"
        elif purity_row is not None and "dependency_cycle" in purity_row.blockers:
            persistence = "unknown"
        else:
            persistence = "ephemeral"

        if purity_class == "pure_map":
            state_dependency = "static_only"
        elif uid in reads_state:
            state_dependency = "reads_state"
        else:
            state_dependency = "unknown"

        if persistence == "persistent":
            semantic_class = "persistent_state"
            note = "carried state is inherited exactly from canonical Kernel.persistent_uids"
        elif purity_class == "pure_map" and persistence == "ephemeral":
            semantic_class = "pure_map"
            note = "canonical transitive closure is state-free"
        elif state_dependency == "reads_state" and persistence == "ephemeral":
            semantic_class = "state_derived_map"
            note = "canonical value reads persistent state but is not itself a carried slot"
        else:
            semantic_class = "unknown"
            note = "canonical evidence is insufficient for a stronger semantic class"

        traits: set[str] = set()
        cv = variants.get(uid)
        if cv is not None and getattr(cv, "role", None) == "reduction":
            traits.add("reduction_kernel")
        recurrence_accesses = tuple(sorted(
            access.uid
            for access in outgoing.get(uid, ())
            if uid in persistent
            and access.order not in {"same"}
        ))
        blockers = () if purity_row is None else purity_row.blockers
        rows.append(
            ValueSemanticsEvidence(
                uid=uid,
                semantic_class=semantic_class,
                state_dependency=state_dependency,
                persistence=persistence,
                proof_kind="domain_graph_value_semantics_v1",
                dependencies=tuple(sorted(incoming.get(uid, ()))),
                recurrence_access_uids=recurrence_accesses,
                operator_traits=tuple(sorted(traits)),
                blockers=tuple(blockers),
                note=note,
            )
        )
    return tuple(rows)



def classify_domain_graph_access_availability(
    graph: DomainGraph,
    variants: dict[str, Any],
) -> tuple[DomainAccessAvailabilityEvidence, ...]:
    """Classify canonical access availability while preserving stronger graph proofs.

    Existing DomainGraph access relations already carry normalized history/event
    provenance.  This adapter deliberately consumes those relations rather than
    reconstructing them from raw source, so a proven DerivedHistoryRead does not
    regress to source-level uncertainty.
    """
    semantics = {row.uid: row for row in classify_domain_graph_value_semantics(graph, variants)}
    persistent = {uid for uid, row in semantics.items() if row.persistence == "persistent"}

    predecessors: dict[str, set[str]] = {}
    for access in graph.accesses:
        if access.order == "same":
            predecessors.setdefault(access.target_uid, set()).add(access.source_uid)
    transition_cone = set(persistent)
    pending = list(sorted(persistent))
    while pending:
        target = pending.pop(0)
        for source in sorted(predecessors.get(target, ())):
            if source not in transition_cone:
                transition_cone.add(source)
                pending.append(source)

    rows: list[DomainAccessAvailabilityEvidence] = []
    for access in sorted(graph.accesses, key=lambda x: x.uid):
        source = semantics.get(access.source_uid)
        target = semantics.get(access.target_uid)
        source_class = source.semantic_class if source is not None else "unknown"
        target_class = target.semantic_class if target is not None else "unknown"
        transition_relevant = access.target_uid in transition_cone
        blockers: list[str] = []

        if access.min_offset is not None and access.max_offset is not None:
            min_offset = int(access.min_offset)
            max_offset = int(access.max_offset)
        elif access.order == "strict_before":
            min_offset = -int(access.max_lag) if access.max_lag is not None else None
            max_offset = -int(access.min_lag) if access.min_lag is not None else -1
        elif access.order == "before_or_same":
            min_offset = -int(access.max_lag) if access.max_lag is not None else None
            max_offset = 0
        elif access.order == "same":
            min_offset = max_offset = 0
        else:
            min_offset = max_offset = None

        if source_class == "pure_map":
            availability = "STATIC"
        elif source_class == "persistent_state":
            if access.order == "strict_before":
                availability = "PREFIX"
            elif access.order == "before_or_same":
                availability = "PREFIX"
            elif access.order == "same":
                availability = "CURRENT"
            elif access.order == "strict_after":
                availability = "SUFFIX" if transition_relevant else "COMPLETE"
                if availability == "SUFFIX":
                    blockers.append("future_state_required_inside_active_persistent_transition")
            elif min_offset is not None and max_offset is not None:
                availability = _availability_for_state_bounds(
                    min_offset, max_offset, transition_relevant=transition_relevant
                )
            else:
                availability = "UNKNOWN"
                blockers.append(f"canonical_relation_not_availability_proven:{access.order}")
        elif source_class == "state_derived_map":
            # Canonical transitive StateDerivedMap footprints are not yet lowered here.
            # Keep direct geometry but fail closed on availability until the next
            # shadow SchedulePlan consumes call-site/source footprint evidence.
            availability = "UNKNOWN"
            blockers.append("canonical_state_derived_footprint_not_lowered")
        else:
            availability = "UNKNOWN"
            blockers.append(f"unproved_source_semantics:{source_class}")

        rows.append(
            DomainAccessAvailabilityEvidence(
                access_uid=access.uid,
                source_uid=access.source_uid,
                target_uid=access.target_uid,
                source_semantic_class=source_class,
                target_semantic_class=target_class,
                transition_relevant=transition_relevant,
                availability=availability,
                min_offset=min_offset,
                max_offset=max_offset,
                proof_kind=access.proof_kind or "domain_graph_access_relation_v1",
                blockers=tuple(sorted(set(blockers))),
                note=(
                    "canonical relation/provenance reused directly; execution authority unchanged"
                ),
            )
        )
    return tuple(rows)


def _domain_scan_direction(
    domain_uid: str,
    domains: Mapping[str, IterationDomain],
    *,
    _seen: set[str] | None = None,
) -> str:
    domain = domains.get(domain_uid)
    if domain is None:
        return "unknown"
    if domain.order in {"ascending", "descending"}:
        return domain.order
    if domain.kind == "event" and domain.parent_uid is not None:
        seen = set() if _seen is None else set(_seen)
        if domain_uid in seen:
            return "unknown"
        seen.add(domain_uid)
        return _domain_scan_direction(domain.parent_uid, domains, _seen=seen)
    if domain.kind == "scalar":
        return "any"
    return "unknown"


def analyze_domain_graph_component_schedule(
    graph: DomainGraph,
    variants: dict[str, Any],
) -> ComponentSchedulePlan:
    """Build the canonical/native positive-control component schedule.

    Unlike the source candidate path, canonical DomainGraph already contains the
    stronger scheduler proofs that made the current native roots executable.  Each
    kernel carrying ``persistent_uids`` is therefore treated as one established
    persistent component and inherits its domain order.  Direct canonical access
    relations then prove whether cross-kernel state can share a stage.

    The function remains shadow-only: it compares the new component vocabulary to
    authoritative evidence but is not consulted by ``ExecutableGraph``.
    """
    semantics = {row.uid: row for row in classify_domain_graph_value_semantics(graph, variants)}
    domains = {domain.uid: domain for domain in graph.domains}
    purity = {uid: row.semantic_class for uid, row in semantics.items()}

    components: list[ComponentScheduleComponent] = []
    component_by_member: dict[str, str] = {}
    component_scan: dict[str, str] = {}
    kernel_component: dict[str, str] = {}
    blockers: list[str] = []
    for kernel in graph.kernels:
        if not kernel.persistent_uids:
            continue
        component_uid = _stable_id("canonical_component", (kernel.uid, kernel.domain_uid))
        scan = _domain_scan_direction(kernel.domain_uid, domains)
        component_blockers: list[str] = []
        if scan == "unknown":
            component_blockers.append(f"canonical_component_scan_unknown:{kernel.uid}")
            blockers.extend(component_blockers)
        components.append(
            ComponentScheduleComponent(
                uid=component_uid,
                persistent_uids=tuple(sorted(kernel.persistent_uids)),
                intrinsic_scan=scan,
                domain_uids=(kernel.domain_uid,),
                blockers=tuple(component_blockers),
                note=(
                    "canonical component inherits carried slots from Kernel.persistent_uids and scan order from its proven iteration domain"
                ),
            )
        )
        component_scan[component_uid] = scan
        kernel_component[kernel.uid] = component_uid
        for uid in kernel.member_uids:
            component_by_member[uid] = component_uid

    constraints: list[ComponentScheduleConstraint] = []
    graph_edges: set[tuple[str, str]] = set()
    barrier_edges: set[tuple[str, str]] = set()
    derived_tasks_by_target: dict[str, list[AccessRelation]] = {}

    for access in sorted(graph.accesses, key=lambda x: x.uid):
        if purity.get(access.source_uid) == "pure_map":
            continue
        source_component = component_by_member.get(access.source_uid)
        target_component = component_by_member.get(access.target_uid)

        if access.order in {"strict_before", "before_or_same", "strict_past", "past_or_same"}:
            availability = "PREFIX"
        elif access.order == "same":
            availability = "CURRENT"
        elif access.order == "strict_after":
            availability = "SUFFIX" if target_component is not None else "COMPLETE"
        elif access.min_offset is not None and access.max_offset is not None:
            availability = _availability_for_state_bounds(
                int(access.min_offset), int(access.max_offset),
                transition_relevant=target_component is not None,
            )
        else:
            availability = "UNKNOWN"

        if source_component is not None and target_component is None:
            if availability in {"COMPLETE", "SUFFIX", "UNKNOWN"}:
                derived_tasks_by_target.setdefault(access.target_uid, []).append(access)
            continue
        if source_component is None or target_component is None:
            continue

        same_component = source_component == target_component
        barrier = False
        row_blockers: list[str] = []
        if availability == "UNKNOWN":
            row_blockers.append("canonical_transition_availability_unknown")
            blockers.append(
                f"canonical_transition_unknown:{access.source_uid}->{access.target_uid}:{access.uid}"
            )
        elif availability in {"SUFFIX", "COMPLETE"}:
            if same_component:
                row_blockers.append("same_component_future_state_requirement")
                blockers.append(
                    f"canonical_same_component_future_state:{access.source_uid}->{access.target_uid}"
                )
            else:
                barrier = True
        elif availability == "PREFIX" and component_scan.get(target_component) == "descending":
            if same_component:
                row_blockers.append("prefix_requirement_conflicts_with_descending_component")
                blockers.append(
                    f"canonical_prefix_in_descending_component:{access.uid}:{target_component}"
                )
            else:
                barrier = True

        if not same_component:
            graph_edges.add((source_component, target_component))
            source_scan = component_scan.get(source_component, "unknown")
            target_scan = component_scan.get(target_component, "unknown")
            if (
                source_scan in {"ascending", "descending"}
                and target_scan in {"ascending", "descending"}
                and source_scan != target_scan
            ):
                barrier = True
        if barrier:
            barrier_edges.add((source_component, target_component))
        constraints.append(
            ComponentScheduleConstraint(
                uid=_stable_id("canonical_constraint", (access.uid, availability)),
                source_component_uid=source_component,
                target_component_uid=target_component,
                kind="availability",
                availability=availability,
                barrier_required=barrier,
                source_uids=(access.source_uid,),
                target_uids=(access.target_uid,),
                evidence_uids=(access.uid,),
                proof_kind=access.proof_kind or "domain_graph_access_relation_v1",
                blockers=tuple(row_blockers),
                note="direct canonical access relation; stronger native proof provenance retained",
            )
        )

    component_uids = tuple(x.uid for x in components)
    stage_by_component, dependency_sccs, stage_blockers = _assign_component_stages(
        component_uids, graph_edges, barrier_edges
    )
    blockers.extend(f"canonical_{row}" for row in stage_blockers)
    adjacency: dict[str, set[str]] = {uid: set() for uid in component_uids}
    for source_component, target_component in graph_edges:
        adjacency.setdefault(source_component, set()).add(target_component)
    for source_component, target_component in sorted(barrier_edges):
        reduced = {uid: set(children) for uid, children in adjacency.items()}
        reduced.setdefault(source_component, set()).discard(target_component)
        if _graph_path_exists(reduced, target_component, source_component):
            blockers.append(
                f"canonical_component_barrier_reverse_path:{source_component}->{target_component}"
            )

    derived_tasks: list[DerivedScheduleTask] = []
    deferred_unknowns: list[str] = []
    for target_uid, accesses in sorted(derived_tasks_by_target.items()):
        availabilities: list[str] = []
        producers: set[str] = set()
        source_uids: set[str] = set()
        for access in accesses:
            source_component = component_by_member.get(access.source_uid)
            if source_component is not None:
                producers.add(source_component)
                source_uids.add(access.source_uid)
            if access.order == "strict_after":
                availabilities.append("COMPLETE")
            elif access.order in {"strict_before", "before_or_same", "strict_past", "past_or_same"}:
                availabilities.append("PREFIX")
            elif access.order == "same":
                availabilities.append("CURRENT")
            else:
                availabilities.append("UNKNOWN")
        aggregate = _aggregate_availability(availabilities)
        if aggregate == "UNKNOWN":
            stage_index = None
            deferred_unknowns.append(f"canonical_derived_task_unknown:{target_uid}")
        elif producers:
            base_stage = max(stage_by_component.get(uid, 0) for uid in producers)
            stage_index = base_stage + (1 if aggregate == "COMPLETE" else 0)
        else:
            stage_index = 0
        derived_tasks.append(
            DerivedScheduleTask(
                uid=_stable_id("canonical_derived_task", (target_uid, aggregate)),
                target_uid=target_uid,
                availability=aggregate,
                producer_component_uids=tuple(sorted(producers)),
                stage_index=stage_index,
                persistent_source_uids=tuple(sorted(source_uids)),
                evidence_uids=tuple(sorted(access.uid for access in accesses)),
                materialized_source_uids=(
                    tuple(sorted(source_uids)) if aggregate == "COMPLETE" else ()
                ),
                note="canonical state-derived/reduction work downstream of carried state",
            )
        )

    materialized_by_stage: dict[int, set[str]] = {}
    for constraint in constraints:
        if not constraint.barrier_required:
            continue
        stage = stage_by_component.get(constraint.source_component_uid, 0)
        materialized_by_stage.setdefault(stage, set()).update(constraint.source_uids)
    for task in derived_tasks:
        if task.stage_index is None:
            continue
        for producer in task.producer_component_uids:
            producer_stage = stage_by_component.get(producer, 0)
            if task.stage_index > producer_stage:
                materialized_by_stage.setdefault(producer_stage, set()).update(
                    task.materialized_source_uids
                )

    stage_indices = set(stage_by_component.values())
    stage_indices.update(task.stage_index for task in derived_tasks if task.stage_index is not None)
    stages: list[ComponentScheduleStage] = []
    for index in sorted(stage_indices or {0}):
        members = tuple(sorted(
            uid for uid, stage_index in stage_by_component.items() if stage_index == index
        ))
        stages.append(
            ComponentScheduleStage(
                index=index,
                component_uids=members,
                scan_directions=tuple(sorted({
                    component_scan[uid]
                    for uid in members
                    if component_scan.get(uid) not in {None, "any"}
                })),
                derived_task_uids=tuple(sorted(
                    task.uid for task in derived_tasks if task.stage_index == index
                )),
                materialized_source_uids=tuple(sorted(materialized_by_stage.get(index, ()))),
                note="canonical shadow component stage; authoritative ExecutableGraph remains unchanged",
            )
        )

    blocker_rows = tuple(sorted(set(blockers)))
    deferred_rows = tuple(sorted(set(deferred_unknowns)))
    return ComponentSchedulePlan(
        schema="modelx_graph.domain_graph.canonical_component_schedule.v1",
        axis="t",
        components=tuple(sorted(components, key=lambda x: x.uid)),
        constraints=tuple(sorted(constraints, key=lambda x: x.uid)),
        derived_tasks=tuple(sorted(derived_tasks, key=lambda x: x.uid)),
        stages=tuple(stages),
        persistent_schedule_proven=not blocker_rows,
        full_schedule_proven=not blocker_rows and not deferred_rows,
        blockers=blocker_rows,
        deferred_unknowns=deferred_rows,
        validation_notes=(
            "proof-only canonical positive control; no execution consumer changed",
            "persistent components inherit exact Kernel.persistent_uids and proven domain ordering",
            "canonical access/history/event provenance is consumed directly instead of raw-source reconstruction",
        ),
    )


def with_domain_graph_purity(graph: DomainGraph, variants: dict[str, Any]) -> DomainGraph:
    """Return an evidence-enriched shadow graph; never changes executable authority."""
    purity = classify_domain_graph_purity(graph, variants)
    pure = {x.uid for x in purity if x.is_pure}
    accesses = tuple(
        AccessRelation(
            **{**a.__dict__,
               "scheduling_relevant": a.source_uid not in pure,
               "scheduling_reason": (
                   "pure_map_call" if a.source_uid in pure else "stateful_or_unproven_source"
               )}
        )
        for a in graph.accesses
    )
    return DomainGraph(
        schema="modelx_graph.domain_graph.shadow.v2", domains=graph.domains,
        kernels=graph.kernels, accesses=accesses, initial_conditions=graph.initial_conditions,
        uid_domain=graph.uid_domain, legacy_summary=graph.legacy_summary,
        purity_evidence=purity,
        validation_notes=graph.validation_notes + (
            "PureMap/statefulness evidence is proof-only; accesses from proven pure maps are scheduling-irrelevant in the shadow graph",
        ),
    )


def _candidate(kind: str, function: str, **payload: Any) -> SourceGraphCandidate:
    return SourceGraphCandidate(kind, function, tuple(sorted(payload.items())))


def scan_source_graph_candidates(source: str) -> tuple[SourceGraphCandidate, ...]:
    """Recognize generic unsupported graph shapes without granting execution.

    The scanner is deliberately syntactic.  It records candidate domain/access
    structures for architecture work; it does not prove them executable.
    """
    module = ast.parse(source)
    functions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    known = set(functions)
    out: list[SourceGraphCandidate] = []

    for name, fn in functions.items():
        params = [a.arg for a in fn.args.args]
        primary = params[0] if params else None

        # Finite-window reductions such as any(source(u) for u in range(lo, hi)).
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"any", "all", "sum"}
                and len(node.args) == 1
                and isinstance(node.args[0], ast.GeneratorExp)
            ):
                continue
            gen = node.args[0]
            if len(gen.generators) != 1:
                continue
            comp = gen.generators[0]
            if comp.is_async or comp.ifs or not isinstance(comp.target, ast.Name):
                continue
            iterator = comp.target.id
            if not (
                isinstance(comp.iter, ast.Call)
                and isinstance(comp.iter.func, ast.Name)
                and comp.iter.func.id == "range"
                and 1 <= len(comp.iter.args) <= 3
            ):
                continue
            body_call = gen.elt
            if not (
                isinstance(body_call, ast.Call)
                and isinstance(body_call.func, ast.Name)
                and body_call.func.id in known
                and body_call.args
                and isinstance(body_call.args[0], ast.Name)
                and body_call.args[0].id == iterator
            ):
                continue
            lo, hi, step = _range_bounds(tuple(comp.iter.args))
            out.append(
                _candidate(
                    "finite_window_reduction",
                    name,
                    op=node.func.id,
                    iterator=iterator,
                    source=body_call.func.id,
                    start_expr=lo,
                    stop_expr=hi,
                    step=step,
                    caller_param=primary,
                )
            )

        # First-hit reductions over a statically expressed range.  The exact
        # source-local semantic recognizer is shared with the recursive backend;
        # this scanner remains evidence-only.
        for stmt in fn.body:
            spec = recognize_first_hit_range_stmt(stmt)
            if spec is None:
                continue
            calls = [
                c.func.id for c in ast.walk(spec.predicate)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in known
            ]
            lo, hi, step = _range_bounds(tuple(spec.range_args))
            out.append(
                _candidate(
                    "first_hit_reduction",
                    name,
                    iterator=spec.iterator,
                    range_start=lo,
                    range_stop=hi,
                    step=step,
                    predicate=ast.unparse(spec.predicate),
                    sources=tuple(sorted(set(calls))),
                )
            )

        # Multi-axis recursive state candidates.  Keep the original source axes;
        # later proof may transform a vintage axis to rolling age.
        if primary is not None and len(params) >= 2:
            for call in ast.walk(fn):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == name
                    and len(call.args) >= len(params)
                ):
                    continue
                off = _affine_offset(call.args[0], primary)
                if off is None or off >= 0:
                    continue
                out.append(
                    _candidate(
                        "recursive_indexed_state",
                        name,
                        axes=tuple(params),
                        primary_offset=off,
                        source_indices=tuple(ast.unparse(x) for x in call.args[: len(params)]),
                    )
                )

        # Vector-shift state: prev = self(t-1), followed by append(prev[z-1] * ...).
        if primary is not None and len(params) == 1:
            prev_names: set[str] = set()
            for stmt in fn.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id == name
                    and stmt.value.args
                    and _affine_offset(stmt.value.args[0], primary) == -1
                ):
                    prev_names.add(stmt.targets[0].id)
            for stmt in fn.body:
                if not isinstance(stmt, ast.For) or not isinstance(stmt.target, ast.Name):
                    continue
                iterator = stmt.target.id
                for call in ast.walk(stmt):
                    if not (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "append"
                        and call.args
                    ):
                        continue
                    for sub in ast.walk(call.args[0]):
                        if not (
                            isinstance(sub, ast.Subscript)
                            and isinstance(sub.value, ast.Name)
                            and sub.value.id in prev_names
                        ):
                            continue
                        idx = ast.unparse(sub.slice)
                        out.append(
                            _candidate(
                                "vector_shift_state",
                                name,
                                primary_param=primary,
                                iterator=iterator,
                                previous_vector=sub.value.id,
                                previous_index=idx,
                                range_expr=ast.unparse(stmt.iter),
                            )
                        )
                        break

        # Positive point accesses are evidence for a future/lookahead relation.
        if primary is not None:
            for call in ast.walk(fn):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in known
                    and call.args
                ):
                    continue
                off = _affine_offset(call.args[0], primary)
                if off is None or off <= 0:
                    continue
                out.append(
                    _candidate(
                        "future_point_access",
                        name,
                        source=call.func.id,
                        offset=off,
                        expr=ast.unparse(call.args[0]),
                    )
                )

    # Deduplicate AST-walk duplicates deterministically.
    uniq: dict[tuple[str, str, tuple[tuple[str, Any], ...]], SourceGraphCandidate] = {}
    for row in out:
        key = (row.kind, row.function, row.payload)
        uniq[key] = row
    return tuple(sorted(uniq.values(), key=lambda x: (x.kind, x.function, repr(x.payload))))


def scan_source_graph_candidates_path(path: str | Path) -> tuple[SourceGraphCandidate, ...]:
    return scan_source_graph_candidates(Path(path).read_text())
