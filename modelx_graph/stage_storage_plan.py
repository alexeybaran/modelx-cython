from __future__ import annotations

"""Semantic stage-aware storage/lowering plan over :mod:`stage_execution_plan`.

This layer is deliberately between canonical execution semantics and physical
backend layout.  It owns *why* a value must survive, for how long, and over which
coordinate envelope.  It does not assign C/Python slot numbers.  Unsupported
backend shapes remain typed blockers; contradictory canonical evidence raises.
"""

import ast
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical_semantic_graph import CanonicalSemanticGraph
from .canonical_schedule import CanonicalComponentScheduleAnalysis
from .stage_execution_plan import StageExecutionPlan


class StageStoragePlanError(RuntimeError):
    """Stage execution/storage evidence is internally contradictory."""


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class StageStorageValue:
    value_uid: str
    state_semantic: str
    execution_semantic: str
    canonical_role: str
    producer_stage_index: int | None
    consumer_stage_indices: tuple[int, ...]
    domain_uid: str | None
    working_storage_kind: str
    ring_depth: int | None
    working_offset_min: int | None
    working_offset_max: int | None
    cross_stage_storage_kind: str
    materialize_after_stage: int | None
    retained_offset_min: int | None
    retained_offset_max: int | None
    retained_bounds: tuple[int | None, int | None] | None
    evidence_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "value_uid": self.value_uid,
            "state_semantic": self.state_semantic,
            "execution_semantic": self.execution_semantic,
            "canonical_role": self.canonical_role,
            "producer_stage_index": self.producer_stage_index,
            "consumer_stage_indices": list(self.consumer_stage_indices),
            "domain_uid": self.domain_uid,
            "working_storage_kind": self.working_storage_kind,
            "ring_depth": self.ring_depth,
            "working_offset_min": self.working_offset_min,
            "working_offset_max": self.working_offset_max,
            "cross_stage_storage_kind": self.cross_stage_storage_kind,
            "materialize_after_stage": self.materialize_after_stage,
            "retained_offset_min": self.retained_offset_min,
            "retained_offset_max": self.retained_offset_max,
            "retained_bounds": None if self.retained_bounds is None else {
                "lo": self.retained_bounds[0], "hi": self.retained_bounds[1]
            },
            "evidence_uids": list(self.evidence_uids),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageStorageFenceValue:
    """Exact storage row required to satisfy one semantic stage fence."""

    value_uid: str
    storage_identity_uid: str
    producer_stage_index: int | None
    consumer_stage_indices: tuple[int, ...]
    materialize_after_stage: int | None
    cross_stage_storage_kind: str
    retained_bounds: tuple[int, int] | None
    evidence_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "value_uid": self.value_uid,
            "storage_identity_uid": self.storage_identity_uid,
            "producer_stage_index": self.producer_stage_index,
            "consumer_stage_indices": list(self.consumer_stage_indices),
            "materialize_after_stage": self.materialize_after_stage,
            "cross_stage_storage_kind": self.cross_stage_storage_kind,
            "retained_bounds": None if self.retained_bounds is None else {
                "lo": self.retained_bounds[0], "hi": self.retained_bounds[1]
            },
            "evidence_uids": list(self.evidence_uids),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageStorageFence:
    """Storage-owned completion contract for one semantic execution barrier.

    The execution plan owns *that* a fence exists.  This record owns exactly
    which materialized histories and retained coordinate bounds make that fence
    physically satisfiable.  Availability is preserved as semantic evidence and
    is deliberately not interpreted as a backend fence kind here.
    """

    uid: str
    semantic_barrier_uid: str
    source_component_uid: str
    target_component_uid: str
    source_stage_index: int | None
    target_stage_index: int | None
    availability: str
    semantic_proof_kind: str | None
    semantic_evidence_uids: tuple[str, ...]
    required_values: tuple[StageStorageFenceValue, ...]
    execution_mode: str = "unproved"
    execution_mode_proof_kind: str | None = None
    proof_kind: str = "stage_storage_fence_contract_v1"
    blockers: tuple[str, ...] = ()

    @property
    def required_value_uids(self) -> tuple[str, ...]:
        return tuple(row.value_uid for row in self.required_values)

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "semantic_barrier_uid": self.semantic_barrier_uid,
            "source_component_uid": self.source_component_uid,
            "target_component_uid": self.target_component_uid,
            "source_stage_index": self.source_stage_index,
            "target_stage_index": self.target_stage_index,
            "availability": self.availability,
            "semantic_proof_kind": self.semantic_proof_kind,
            "semantic_evidence_uids": list(self.semantic_evidence_uids),
            "required_value_uids": list(self.required_value_uids),
            "required_values": [row.manifest() for row in self.required_values],
            "execution_mode": self.execution_mode,
            "execution_mode_proof_kind": self.execution_mode_proof_kind,
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageStorageKernel:
    uid: str
    task_uid: str
    stage_index: int
    kind: str
    component_uid: str | None
    canonical_uids: tuple[str, ...]
    domain_uids: tuple[str, ...]
    scan_direction: str
    reduction_range_asts: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "task_uid": self.task_uid,
            "stage_index": int(self.stage_index),
            "kind": self.kind,
            "component_uid": self.component_uid,
            "canonical_uids": list(self.canonical_uids),
            "domain_uids": list(self.domain_uids),
            "scan_direction": self.scan_direction,
            "reduction_range_asts": list(self.reduction_range_asts),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageStorageStage:
    index: int
    kernel_uids: tuple[str, ...]
    materialized_value_uids: tuple[str, ...]
    stage_carried_scalar_uids: tuple[str, ...]
    incoming_barrier_uids: tuple[str, ...]
    outgoing_barrier_uids: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "kernel_uids": list(self.kernel_uids),
            "materialized_value_uids": list(self.materialized_value_uids),
            "stage_carried_scalar_uids": list(self.stage_carried_scalar_uids),
            "incoming_barrier_uids": list(self.incoming_barrier_uids),
            "outgoing_barrier_uids": list(self.outgoing_barrier_uids),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageStoragePlan:
    schema: str
    uid: str
    execution_plan_uid: str
    graph_schema: str
    stages: tuple[StageStorageStage, ...]
    kernels: tuple[StageStorageKernel, ...]
    values: tuple[StageStorageValue, ...]
    fences: tuple[StageStorageFence, ...]
    barrier_uids: tuple[str, ...]
    geometry_proven: bool
    capability_blockers: tuple[str, ...] = ()
    optimization_blockers: tuple[str, ...] = ()
    validation_notes: tuple[str, ...] = ()

    def value(self, uid: str) -> StageStorageValue:
        return next(x for x in self.values if x.value_uid == uid)

    def fence(self, semantic_barrier_uid: str) -> StageStorageFence:
        return next(x for x in self.fences if x.semantic_barrier_uid == semantic_barrier_uid)

    @property
    def materialized_value_uids(self) -> tuple[str, ...]:
        return tuple(sorted(
            x.value_uid for x in self.values
            if x.cross_stage_storage_kind == "materialized_history"
        ))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "uid": self.uid,
            "execution_plan_uid": self.execution_plan_uid,
            "graph_schema": self.graph_schema,
            "stages": [x.manifest() for x in self.stages],
            "kernels": [x.manifest() for x in self.kernels],
            "values": [x.manifest() for x in self.values],
            "fences": [x.manifest() for x in self.fences],
            "barrier_uids": list(self.barrier_uids),
            "geometry_proven": bool(self.geometry_proven),
            "capability_blockers": list(self.capability_blockers),
            "optimization_blockers": list(self.optimization_blockers),
            "metrics": {
                "stage_count": len(self.stages),
                "kernel_count": len(self.kernels),
                "value_count": len(self.values),
                "materialized_value_count": len(self.materialized_value_uids),
                "ring_value_count": sum(x.working_storage_kind == "ring" for x in self.values),
                "history_value_count": sum(x.working_storage_kind == "history" for x in self.values),
                "stage_carried_scalar_count": sum(x.cross_stage_storage_kind == "stage_carried_scalar" for x in self.values),
                "storage_fence_count": len(self.fences),
            },
            "validation_notes": list(self.validation_notes),
        }


def _storage_value_identity(
    execution_plan_uid: str,
    row: StageStorageValue,
) -> str:
    """Stable semantic identity for the storage fields a fence relies upon."""
    return _stable_id("stage_storage_value_identity", (
        execution_plan_uid,
        row.value_uid,
        row.producer_stage_index,
        row.consumer_stage_indices,
        row.cross_stage_storage_kind,
        row.materialize_after_stage,
        row.retained_bounds,
        row.evidence_uids,
        row.blockers,
    ))


def _build_storage_fences(
    execution_plan: StageExecutionPlan,
    values: tuple[StageStorageValue, ...],
    stages: tuple[StageStorageStage, ...],
) -> tuple[tuple[StageStorageFence, ...], tuple[str, ...]]:
    """Resolve semantic barriers to exact storage-owned materialization facts.

    This intentionally mirrors the historical StageProgram selection rule so
    ownership can move upstream without changing physical behavior: every
    history materialized after the barrier's source stage and consumed at the
    target stage or later is part of the fence contract.  The semantic barrier's
    direct ``source_uids`` are additionally required to be covered by those rows.
    """
    storage_by_uid = {row.value_uid: row for row in values}
    stage_by_index = {int(row.index): row for row in stages}
    fence_rows: list[StageStorageFence] = []
    all_blockers: set[str] = set()

    for barrier in sorted(execution_plan.barriers, key=lambda row: row.uid):
        row_blockers: set[str] = set(barrier.blockers)
        src = barrier.source_stage_index
        dst = barrier.target_stage_index
        required: list[StageStorageFenceValue] = []

        if src is None or dst is None:
            row_blockers.add(f"storage_fence_stage_unproved:{barrier.uid}")
        else:
            src = int(src)
            dst = int(dst)
            if src >= dst:
                row_blockers.add(
                    f"storage_fence_stage_order_invalid:{barrier.uid}:{src}:{dst}"
                )
            source_stage = stage_by_index.get(src)
            target_stage = stage_by_index.get(dst)
            if source_stage is None or barrier.uid not in source_stage.outgoing_barrier_uids:
                row_blockers.add(
                    f"storage_fence_source_stage_link_missing:{barrier.uid}:{src}"
                )
            if target_stage is None or barrier.uid not in target_stage.incoming_barrier_uids:
                row_blockers.add(
                    f"storage_fence_target_stage_link_missing:{barrier.uid}:{dst}"
                )

            required_values = [
                value for value in sorted(values, key=lambda row: row.value_uid)
                if value.cross_stage_storage_kind == "materialized_history"
                and value.materialize_after_stage == src
                and any(int(consumer) >= dst for consumer in value.consumer_stage_indices)
            ]

            for source_uid in barrier.source_uids:
                value = storage_by_uid.get(source_uid)
                if value is None:
                    row_blockers.add(
                        f"storage_fence_source_value_missing:{barrier.uid}:{source_uid}"
                    )
                    continue
                if value.cross_stage_storage_kind != "materialized_history":
                    row_blockers.add(
                        f"storage_fence_source_value_not_materialized:{barrier.uid}:"
                        f"{source_uid}:{value.cross_stage_storage_kind}"
                    )
                if value.producer_stage_index != src:
                    row_blockers.add(
                        f"storage_fence_source_producer_stage_mismatch:{barrier.uid}:"
                        f"{source_uid}:{value.producer_stage_index}:{src}"
                    )
                if value.materialize_after_stage != src:
                    row_blockers.add(
                        f"storage_fence_source_materialization_stage_mismatch:{barrier.uid}:"
                        f"{source_uid}:{value.materialize_after_stage}:{src}"
                    )
                if not any(int(consumer) >= dst for consumer in value.consumer_stage_indices):
                    row_blockers.add(
                        f"storage_fence_source_not_consumed_through_target:{barrier.uid}:"
                        f"{source_uid}:{dst}"
                    )

            for value in required_values:
                # Fence completeness cares about materialized history geometry,
                # not optional working-storage compaction.  A value may soundly
                # fall back from a ring to full stage-local history while still
                # providing an exact retained cross-stage envelope.
                value_blockers: set[str] = {
                    blocker for blocker in value.blockers
                    if not blocker.startswith(
                        "physical_iteration_step_unproved_for_ring_compaction:"
                    )
                }
                bounds: tuple[int, int] | None = None
                if (
                    value.retained_bounds is None
                    or value.retained_bounds[0] is None
                    or value.retained_bounds[1] is None
                ):
                    value_blockers.add(
                        f"storage_fence_required_value_bounds_unproved:{barrier.uid}:{value.value_uid}"
                    )
                else:
                    bounds = (
                        int(value.retained_bounds[0]),
                        int(value.retained_bounds[1]),
                    )
                for blocker in value_blockers:
                    row_blockers.add(blocker)
                required.append(StageStorageFenceValue(
                    value_uid=value.value_uid,
                    storage_identity_uid=_storage_value_identity(execution_plan.uid, value),
                    producer_stage_index=value.producer_stage_index,
                    consumer_stage_indices=tuple(value.consumer_stage_indices),
                    materialize_after_stage=value.materialize_after_stage,
                    cross_stage_storage_kind=value.cross_stage_storage_kind,
                    retained_bounds=bounds,
                    evidence_uids=tuple(value.evidence_uids),
                    blockers=tuple(sorted(value_blockers)),
                ))

            required_uids = {row.value_uid for row in required}
            missing_semantic_sources = sorted(set(barrier.source_uids) - required_uids)
            for uid in missing_semantic_sources:
                row_blockers.add(
                    f"storage_fence_semantic_source_not_covered:{barrier.uid}:{uid}"
                )
            if not required:
                row_blockers.add(f"storage_fence_required_values_empty:{barrier.uid}")

        execution_mode = "unproved"
        execution_mode_proof_kind = None
        if not row_blockers and required:
            execution_mode = "full_stage_materialization"
            execution_mode_proof_kind = "stage_storage_full_stage_materialization_v1"

        fence_uid = _stable_id("stage_storage_fence", (
            execution_plan.uid,
            barrier.uid,
            barrier.source_component_uid,
            barrier.target_component_uid,
            src,
            dst,
            barrier.availability,
            barrier.proof_kind,
            barrier.evidence_uids,
            tuple(
                (row.value_uid, row.storage_identity_uid, row.retained_bounds)
                for row in required
            ),
            execution_mode,
            execution_mode_proof_kind,
            tuple(sorted(row_blockers)),
        ))
        fence_rows.append(StageStorageFence(
            uid=fence_uid,
            semantic_barrier_uid=barrier.uid,
            source_component_uid=barrier.source_component_uid,
            target_component_uid=barrier.target_component_uid,
            source_stage_index=barrier.source_stage_index,
            target_stage_index=barrier.target_stage_index,
            availability=barrier.availability,
            semantic_proof_kind=barrier.proof_kind,
            semantic_evidence_uids=tuple(barrier.evidence_uids),
            required_values=tuple(required),
            execution_mode=execution_mode,
            execution_mode_proof_kind=execution_mode_proof_kind,
            blockers=tuple(sorted(row_blockers)),
        ))
        all_blockers.update(row_blockers)

    return tuple(fence_rows), tuple(sorted(all_blockers))


def _task_stages(plan: StageExecutionPlan) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for task in plan.tasks:
        for uid in task.canonical_uids:
            prior = out.get(uid)
            stages = tuple(task.stage_indices)
            if prior is not None and prior != stages:
                raise StageStoragePlanError(
                    f"canonical UID {uid} has inconsistent task stage placement: {prior} vs {stages}"
                )
            out[uid] = stages
    return out


def _component_scan_map(
    plan: StageExecutionPlan,
    analysis: CanonicalComponentScheduleAnalysis | None,
) -> dict[str, str]:
    if analysis is not None:
        return {x.uid: x.intrinsic_scan for x in analysis.component_schedule.components}
    stage_scan = {x.index: tuple(x.scan_directions) for x in plan.stages}
    out: dict[str, str] = {}
    for task in plan.tasks:
        if task.kind != "persistent_component" or task.component_uid is None:
            continue
        if len(task.stage_indices) != 1:
            out[task.component_uid] = "unknown"
            continue
        dirs = set(stage_scan.get(task.stage_indices[0], ()))
        out[task.component_uid] = next(iter(dirs)) if len(dirs) == 1 else "unknown"
    return out


def _transition_rows(
    graph: CanonicalSemanticGraph,
    analysis: CanonicalComponentScheduleAnalysis | None,
) -> tuple[tuple[str, str, int | None, int | None, str, tuple[str, ...], tuple[str, ...]], ...]:
    if analysis is not None and analysis.transition_evidence:
        return tuple(
            (x.source_uid, x.target_uid, x.min_offset, x.max_offset, x.availability,
             tuple(x.access_uids), tuple(x.blockers))
            for x in analysis.transition_evidence
        )
    persistent = {x.uid for x in graph.nodes if x.state_semantic == "persistent_state"}
    return tuple(
        (a.source_uid, a.target_uid, a.min_offset, a.max_offset,
         "UNKNOWN" if a.relation == "unknown" else "CURRENT", (a.uid,), tuple(a.blockers))
        for a in graph.accesses
        if a.source_uid in persistent and a.target_uid in persistent
    )


def _literal_reduction_range(variant: Any | None) -> tuple[str, ...]:
    if variant is None or getattr(variant, "reduction", None) is None:
        return ()
    return tuple(ast.dump(x, include_attributes=False) for x in variant.reduction.range_args)


def _proven_physical_iteration_step(
    graph: CanonicalSemanticGraph,
    uid: str,
) -> int | None:
    """Return an exact physical coordinate stride when canonical topology proves it.

    ``CanonicalLoopTopologyFact.coordinate_stride`` is frozen before executable
    lowering and therefore may be used by semantic storage without consulting the
    legacy loop object.  Ambiguous/missing topology stays ``None``.
    """

    steps: set[int] = set()
    for fact in graph.loop_topology_facts:
        positions = dict(fact.coordinate_positions)
        if uid not in positions:
            continue
        step = abs(int(fact.coordinate_stride))
        if step > 0:
            steps.add(step)
    return next(iter(steps)) if len(steps) == 1 else None


def build_stage_storage_plan(
    graph: CanonicalSemanticGraph,
    execution_plan: StageExecutionPlan,
    *,
    analysis: CanonicalComponentScheduleAnalysis | None = None,
    variants: Mapping[str, Any] | None = None,
) -> StageStoragePlan:
    """Lower semantic lifetimes into backend-neutral storage requirements.

    Construction is total for a structurally valid stage plan.  Missing geometry
    or unsupported tensor/barrier shapes become capability blockers; only dangling
    or contradictory exact-UID evidence raises.
    """

    nodes = {x.uid: x for x in graph.nodes}
    if graph.output_uid not in nodes:
        raise StageStoragePlanError(f"output UID absent from canonical graph: {graph.output_uid}")
    domains = {x.owner_uid: x for x in execution_plan.domains}
    lifetimes = {x.value_uid: x for x in execution_plan.lifetimes}
    if set(nodes) != set(lifetimes):
        missing = sorted(set(nodes) - set(lifetimes))
        extra = sorted(set(lifetimes) - set(nodes))
        raise StageStoragePlanError(f"lifetime coverage mismatch; missing={missing!r} extra={extra!r}")

    task_stages = _task_stages(execution_plan)
    persistent_component: dict[str, str] = {}
    for task in execution_plan.tasks:
        if task.kind != "persistent_component":
            continue
        if task.component_uid is None:
            raise StageStoragePlanError(f"persistent task {task.uid} lacks component UID")
        for uid in task.canonical_uids:
            prior = persistent_component.get(uid)
            if prior is not None and prior != task.component_uid:
                raise StageStoragePlanError(f"persistent UID {uid} belongs to two components")
            persistent_component[uid] = task.component_uid

    persistent_stage: dict[str, int] = {}
    for task in execution_plan.tasks:
        if task.kind != "persistent_component" or len(task.stage_indices) != 1:
            continue
        for uid in task.canonical_uids:
            persistent_stage[uid] = int(task.stage_indices[0])

    component_scan = _component_scan_map(execution_plan, analysis)
    transitions = _transition_rows(graph, analysis)
    variants = dict(variants or {})
    blockers: set[str] = set()

    # Exact stage-to-stage coordinate envelope evidence.  Only accesses whose
    # consumer actually executes after the persistent producer contribute.
    cross_accesses: dict[str, list[Any]] = {uid: [] for uid in nodes}
    for access in graph.accesses:
        lifetime = lifetimes[access.source_uid]
        producer = lifetime.producer_stage_index
        if producer is None:
            continue
        if any(stage > producer for stage in task_stages.get(access.target_uid, ())):
            cross_accesses[access.source_uid].append(access)

    values: list[StageStorageValue] = []
    for uid, node in sorted(nodes.items()):
        lifetime = lifetimes[uid]
        domain = domains.get(uid)
        row_blockers: list[str] = list(lifetime.blockers)
        working_kind = "none"
        ring_depth: int | None = None
        working_min: int | None = 0
        working_max: int | None = 0
        cross_kind = "none"
        retained_min: int | None = 0
        retained_max: int | None = 0
        retained_bounds: tuple[int | None, int | None] | None = None

        if node.state_semantic == "auxiliary_recurrence":
            recurrence_reads = [
                access for access in graph.accesses
                if access.source_uid == uid
                and nodes[access.target_uid].state_semantic == "auxiliary_recurrence"
                and access.scheduling_relevant is False
                and access.proof_kind == "canonical_state_free_causal_recurrence_v1"
            ]
            # The semantic access is policy-state-irrelevant, but its exact offset
            # still defines the helper's private recurrence storage geometry.
            # If access refinement cleared offsets, recover them from the canonical
            # source call shape relative to the target helper parameter.
            offsets: list[int] = []
            for access in recurrence_reads:
                target = nodes[access.target_uid]
                if len(target.parameters) != 1 or len(access.argument_asts) != 1:
                    row_blockers.append(f"auxiliary_recurrence_axis_unproved:{access.uid}")
                    continue
                try:
                    arg = ast.parse(access.argument_exprs[0], mode="eval").body
                except Exception:
                    row_blockers.append(f"auxiliary_recurrence_argument_unproved:{access.uid}")
                    continue
                variable = target.parameters[0]
                off = None
                if isinstance(arg, ast.Name) and arg.id == variable:
                    off = 0
                elif (
                    isinstance(arg, ast.BinOp) and isinstance(arg.left, ast.Name)
                    and arg.left.id == variable and isinstance(arg.right, ast.Constant)
                    and isinstance(arg.right.value, int) and not isinstance(arg.right.value, bool)
                ):
                    if isinstance(arg.op, ast.Sub): off = -int(arg.right.value)
                    elif isinstance(arg.op, ast.Add): off = int(arg.right.value)
                if off is None or off == 0:
                    row_blockers.append(f"auxiliary_recurrence_offset_unproved:{access.uid}")
                else:
                    offsets.append(int(off))
            if not offsets:
                working_kind = "history"
                row_blockers.append(f"auxiliary_recurrence_storage_geometry_unproved:{uid}")
                working_min = working_max = None
            elif all(x < 0 for x in offsets):
                working_min, working_max = min(offsets), 0
                ring_depth = max(abs(x) for x in offsets) + 1
                working_kind = "ring"
            elif all(x > 0 for x in offsets):
                working_min, working_max = 0, max(offsets)
                ring_depth = max(abs(x) for x in offsets) + 1
                working_kind = "ring"
            else:
                working_kind = "history"
                working_min, working_max = min(offsets), max(offsets)
                row_blockers.append(f"auxiliary_recurrence_mixed_direction:{uid}")
            cross_kind = "recompute"

        elif node.state_semantic == "persistent_state":
            component_uid = persistent_component.get(uid)
            if component_uid is None:
                raise StageStoragePlanError(f"persistent value {uid} lacks a persistent component")
            scan = component_scan.get(component_uid, "unknown")
            producer_stage = lifetime.producer_stage_index
            local_rows = [
                x for x in transitions
                if x[0] == uid
                and persistent_stage.get(x[1]) == producer_stage
            ]
            finite_mins = [int(x[2]) for x in local_rows if x[2] is not None]
            finite_maxs = [int(x[3]) for x in local_rows if x[3] is not None]
            unknown_bounds = any(x[2] is None or x[3] is None or x[4] == "UNKNOWN" or x[6] for x in local_rows)
            working_min = min([0, *finite_mins]) if finite_mins else 0
            working_max = max([0, *finite_maxs]) if finite_maxs else 0
            rank = None if domain is None else domain.rank
            if rank is not None and rank > 1:
                working_kind = "tensor"
                row_blockers.append(f"multiaxis_persistent_storage_not_supported:{uid}:{rank}")
            elif scan == "ascending" and not unknown_bounds and working_max <= 0:
                lag = max(0, -int(working_min or 0))
                physical_step = _proven_physical_iteration_step(graph, uid)
                exact_offsets = [
                    abs(int(x[2])) for x in local_rows
                    if x[2] is not None and x[3] is not None and int(x[2]) == int(x[3]) and int(x[2]) != 0
                ]
                all_nonzero_rows_exact = all(
                    x[2] is not None and x[3] is not None and int(x[2]) == int(x[3])
                    for x in local_rows if (x[2] or x[3])
                )
                if lag == 0:
                    working_kind = "current"
                elif lag == 1:
                    # One coordinate lag is one predecessor slot for the canonical
                    # unit-step subset even when explicit loop-topology evidence is absent.
                    working_kind = "ring"
                    ring_depth = 2
                elif (
                    physical_step is not None
                    and all_nonzero_rows_exact
                    and exact_offsets
                    and all(offset % physical_step == 0 for offset in exact_offsets)
                ):
                    working_kind = "ring"
                    ring_depth = max(offset // physical_step for offset in exact_offsets) + 1
                else:
                    working_kind = "history"
                    row_blockers.append(
                        f"physical_iteration_step_unproved_for_ring_compaction:{uid}:{lag}"
                    )
            elif scan == "descending" and not unknown_bounds and working_min >= 0:
                lag = max(0, int(working_max or 0))
                physical_step = _proven_physical_iteration_step(graph, uid)
                exact_offsets = [
                    abs(int(x[3])) for x in local_rows
                    if x[2] is not None and x[3] is not None and int(x[2]) == int(x[3]) and int(x[3]) != 0
                ]
                all_nonzero_rows_exact = all(
                    x[2] is not None and x[3] is not None and int(x[2]) == int(x[3])
                    for x in local_rows if (x[2] or x[3])
                )
                if lag == 0:
                    working_kind = "current"
                elif lag == 1:
                    working_kind = "ring"
                    ring_depth = 2
                elif (
                    physical_step is not None
                    and all_nonzero_rows_exact
                    and exact_offsets
                    and all(offset % physical_step == 0 for offset in exact_offsets)
                ):
                    working_kind = "ring"
                    ring_depth = max(offset // physical_step for offset in exact_offsets) + 1
                else:
                    working_kind = "history"
                    row_blockers.append(
                        f"physical_iteration_step_unproved_for_ring_compaction:{uid}:{lag}"
                    )
            elif scan == "any" and not unknown_bounds and working_min == 0 and working_max == 0:
                working_kind = "current"
            else:
                # Full stage-local history is a sound generic representation when
                # recurrence geometry is not compact enough for a ring proof.
                working_kind = "history"
                if scan in {"mixed", "unknown"}:
                    row_blockers.append(f"persistent_scan_storage_compaction_unproved:{uid}:{scan}")

            if lifetime.preservation_kind == "materialized_cross_stage":
                cross_kind = "materialized_history"
                rows = cross_accesses[uid]
                mins: list[int] = []
                maxs: list[int] = []
                fixed_coordinates: list[int] = []
                unknown_cross: list[str] = []
                for access in rows:
                    if (
                        access.relation == "fixed"
                        and access.fixed_coordinate is not None
                    ):
                        fixed_coordinates.append(int(access.fixed_coordinate))
                    elif access.min_offset is None or access.max_offset is None:
                        unknown_cross.append(access.uid)
                    else:
                        mins.append(int(access.min_offset))
                        maxs.append(int(access.max_offset))
                if unknown_cross:
                    retained_min = None
                    retained_max = None
                    row_blockers.append(
                        "cross_stage_coordinate_envelope_unproved:" + ",".join(sorted(unknown_cross))
                    )
                else:
                    # Relative accesses retain an envelope around the producer's
                    # declared coordinate domain.  A fixed OutputInvocation access
                    # is absolute geometry instead, so keep it as an exact
                    # singleton (or finite union) rather than fabricating a
                    # caller-relative offset.
                    retained_min = min([0, *mins]) if mins else None
                    retained_max = max([0, *maxs]) if maxs else None
                    absolute_bounds: list[tuple[int, int]] = []
                    if fixed_coordinates:
                        absolute_bounds.append((
                            min(fixed_coordinates), max(fixed_coordinates)
                        ))
                if domain is not None and domain.rank == 1 and domain.bounds:
                    lo, hi = domain.bounds[0]
                    if lo is not None and hi is not None and retained_min is not None and retained_max is not None:
                        absolute_bounds.append((
                            int(lo) + int(retained_min), int(hi) + int(retained_max)
                        ))
                if not unknown_cross and absolute_bounds:
                    retained_bounds = (
                        min(lo for lo, _ in absolute_bounds),
                        max(hi for _, hi in absolute_bounds),
                    )
                if lifetime.materialize_after_stage is None:
                    raise StageStoragePlanError(f"materialized lifetime {uid} lacks boundary stage")
            elif lifetime.preservation_kind == "persistent_cross_stage_unmaterialized":
                cross_kind = "required_unproved"
                row_blockers.append(f"persistent_cross_stage_storage_unproved:{uid}")

        elif node.canonical_role in {"scalar", "reduction"}:
            working_kind = "scalar"
            producer = lifetime.producer_stage_index
            if producer is not None and any(x > producer for x in lifetime.consumer_stage_indices):
                cross_kind = "stage_carried_scalar"
        elif node.state_semantic == "state_derived_map":
            working_kind = "none"
            cross_kind = "recompute"
        elif node.state_semantic == "state_free":
            working_kind = "none"
            cross_kind = "recompute"
        else:
            # Unknown/nonpersistent execution semantics do not by themselves
            # require storage.  They remain execution/operator blockers in the
            # StageExecutionPlan; storage geometry stays independent.
            working_kind = "none"
            cross_kind = "recompute"

        if domain is not None:
            for blocker in domain.blockers:
                if (
                    blocker == "reduction_domain_geometry_not_lifted_yet"
                    and node.canonical_role == "reduction"
                    and _literal_reduction_range(variants.get(uid))
                ):
                    continue
                row_blockers.append(blocker)
        for blocker in row_blockers:
            blockers.add(blocker)

        values.append(StageStorageValue(
            value_uid=uid,
            state_semantic=node.state_semantic,
            execution_semantic=node.execution_semantic,
            canonical_role=node.canonical_role,
            producer_stage_index=lifetime.producer_stage_index,
            consumer_stage_indices=tuple(lifetime.consumer_stage_indices),
            domain_uid=None if domain is None else domain.uid,
            working_storage_kind=working_kind,
            ring_depth=ring_depth,
            working_offset_min=working_min,
            working_offset_max=working_max,
            cross_stage_storage_kind=cross_kind,
            materialize_after_stage=lifetime.materialize_after_stage,
            retained_offset_min=retained_min,
            retained_offset_max=retained_max,
            retained_bounds=retained_bounds,
            evidence_uids=tuple(lifetime.evidence_uids),
            blockers=tuple(sorted(set(row_blockers))),
        ))

    # Execution kernels are semantic task/stage instances.  A canonical stage can
    # lower to many physical loops; the IR deliberately does not resurrect
    # legacy coord_phase as a semantic concept.
    component_scan_by_uid = component_scan
    kernels: list[StageStorageKernel] = []
    for task in sorted(execution_plan.tasks, key=lambda x: x.uid):
        for stage_index in task.stage_indices:
            scan = "any"
            if task.kind == "persistent_component" and task.component_uid is not None:
                scan = component_scan_by_uid.get(task.component_uid, "unknown")
            reduction_ranges: set[str] = set()
            kernel_blockers = list(task.blockers)
            for uid in task.canonical_uids:
                node = nodes[uid]
                if node.canonical_role == "reduction":
                    row = _literal_reduction_range(variants.get(uid))
                    if row:
                        reduction_ranges.update(row)
                    else:
                        kernel_blockers.append(f"reduction_domain_variant_unavailable:{uid}")
            kernel_uid = _stable_id("stage_storage_kernel", (
                execution_plan.uid, task.uid, stage_index, task.kind,
                task.canonical_uids, scan, tuple(sorted(reduction_ranges)),
            ))
            kernels.append(StageStorageKernel(
                uid=kernel_uid,
                task_uid=task.uid,
                stage_index=int(stage_index),
                kind=task.kind,
                component_uid=task.component_uid,
                canonical_uids=tuple(task.canonical_uids),
                domain_uids=tuple(task.domain_uids),
                scan_direction=scan,
                reduction_range_asts=tuple(sorted(reduction_ranges)),
                blockers=tuple(sorted(set(kernel_blockers))),
            ))
            # Task/operator blockers remain attached to the kernel but do not
            # invalidate storage geometry.  Storage and execution capability are
            # intentionally separate proof dimensions.  Only absence of a
            # reduction-domain representation is a storage/lowering blocker here.
            blockers.update(
                x for x in kernel_blockers
                if x.startswith("reduction_domain_variant_unavailable:")
            )

    kernel_by_stage: dict[int, list[str]] = {x.index: [] for x in execution_plan.stages}
    for kernel in kernels:
        kernel_by_stage.setdefault(kernel.stage_index, []).append(kernel.uid)

    storage_values = {x.value_uid: x for x in values}
    stage_rows: list[StageStorageStage] = []
    for stage in execution_plan.stages:
        mats = tuple(sorted(
            uid for uid, value in storage_values.items()
            if value.materialize_after_stage == stage.index
            and value.cross_stage_storage_kind == "materialized_history"
        ))
        carried = tuple(sorted(
            uid for uid, value in storage_values.items()
            if value.producer_stage_index == stage.index
            and value.cross_stage_storage_kind == "stage_carried_scalar"
        ))
        stage_rows.append(StageStorageStage(
            index=int(stage.index),
            kernel_uids=tuple(sorted(kernel_by_stage.get(stage.index, ()))),
            materialized_value_uids=mats,
            stage_carried_scalar_uids=carried,
            incoming_barrier_uids=tuple(stage.incoming_barrier_uids),
            outgoing_barrier_uids=tuple(stage.outgoing_barrier_uids),
            blockers=tuple(stage.blockers),
        ))

    storage_fences, storage_fence_blockers = _build_storage_fences(
        execution_plan,
        tuple(values),
        tuple(stage_rows),
    )
    blockers.update(storage_fence_blockers)

    plan_uid = _stable_id("stage_storage_plan", (
        execution_plan.uid,
        *(f"{x.value_uid}:{x.working_storage_kind}:{x.cross_stage_storage_kind}:{x.retained_offset_min}:{x.retained_offset_max}" for x in values),
        *(f"kernel:{x.uid}" for x in kernels),
        *(f"fence:{x.uid}:{x.semantic_barrier_uid}:{x.execution_mode}:{x.execution_mode_proof_kind}:{x.required_value_uids}:{x.blockers}" for x in storage_fences),
        *sorted(blockers),
    ))
    # Ring compaction is an optimization proof, not a prerequisite for sound
    # storage geometry.  When a coordinate lag cannot yet be normalized to the
    # physical iteration step, full stage-local history is a valid conservative
    # representation.  Keep the compaction gap visible without pretending the
    # semantic storage plan itself is unknown.
    optimization_only_prefixes = (
        "physical_iteration_step_unproved_for_ring_compaction:",
    )
    geometry_blockers = tuple(sorted(
        blocker for blocker in blockers
        if not blocker.startswith(optimization_only_prefixes)
    ))
    optimization_blockers = tuple(sorted(
        blocker for blocker in blockers
        if blocker.startswith(optimization_only_prefixes)
    ))

    return StageStoragePlan(
        schema="modelx_graph.stage_storage_plan.v3",
        uid=plan_uid,
        execution_plan_uid=execution_plan.uid,
        graph_schema=graph.schema,
        stages=tuple(stage_rows),
        kernels=tuple(kernels),
        values=tuple(values),
        fences=tuple(storage_fences),
        barrier_uids=tuple(sorted(x.uid for x in execution_plan.barriers)),
        geometry_proven=not geometry_blockers,
        capability_blockers=geometry_blockers,
        optimization_blockers=optimization_blockers,
        validation_notes=(
            "semantic lifetimes, not legacy coord_phase/DerivedHistoryRead, own cross-stage preservation",
            "persistent working storage is classified independently from cross-stage materialization",
            "StateDerivedMap/PureMap values are not allocated persistent storage",
            "materialized coordinate envelopes derive from exact canonical cross-stage access offsets",
            "storage-owned fence contracts freeze exact materialized histories and retained bounds per semantic barrier",
            "unproved ring compaction falls back to sound stage-local history without invalidating geometry",
            "physical backend slot numbering remains a downstream implementation detail",
        ),
    )

@dataclass(frozen=True)
class StageStorageCompatibility:
    """Compatibility of a physical backend layout with semantic storage facts."""

    compatible: bool
    reasons: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {"compatible": bool(self.compatible), "reasons": list(self.reasons)}


def validate_template_storage_compatibility(
    stage_storage: StageStoragePlan,
    layout: Any,
) -> StageStorageCompatibility:
    """Check that the legacy physical layout is a sound implementation.

    This does not require identical slot numbering or force the semantic planner
    to mimic legacy optimization.  A more conservative full history may implement
    a semantic ring/current requirement, but it may not omit required persistent
    state, undersize a proven ring, or store an execution PureMap as state.
    Multi-stage materialization is intentionally rejected by the legacy ABI.
    """

    reasons: list[str] = []
    current = set(getattr(layout, "current_slots", {}))
    history = set(getattr(layout, "history_slots", {}))
    ring_depths = dict(getattr(layout, "ring_depths", {}))
    derived = set(getattr(layout, "derived_state_slots", {}))
    derived_region = set(getattr(layout, "derived_region_state_slots", {}))
    available_state = current | history | set(ring_depths) | derived | derived_region

    for row in stage_storage.values:
        uid = row.value_uid
        if row.cross_stage_storage_kind == "materialized_history":
            reasons.append(f"legacy_layout_has_no_semantic_stage_materialization:{uid}")
            continue
        if row.state_semantic == "persistent_state":
            if uid not in available_state:
                reasons.append(f"legacy_layout_omits_persistent_value:{uid}")
                continue
            if row.working_storage_kind == "ring" and uid in ring_depths:
                required = int(row.ring_depth or 0)
                if int(ring_depths[uid]) < required:
                    reasons.append(
                        f"legacy_ring_too_shallow:{uid}:{ring_depths[uid]}<{required}"
                    )
        if row.execution_semantic == "pure_map" and row.canonical_role == "coordinate":
            if uid in available_state and row.working_storage_kind == "none":
                reasons.append(f"legacy_layout_stores_pure_map_coordinate:{uid}")
        if row.state_semantic == "auxiliary_recurrence":
            if uid not in available_state:
                reasons.append(f"legacy_layout_omits_auxiliary_recurrence:{uid}")

    return StageStorageCompatibility(
        compatible=not reasons,
        reasons=tuple(sorted(set(reasons))),
    )
