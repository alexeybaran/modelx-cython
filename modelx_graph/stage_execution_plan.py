from __future__ import annotations

"""Backend-neutral stage/lifetime execution plan over canonical graph semantics.

V02350A is intentionally observational.  The plan is total over a structurally
valid ``CanonicalSemanticGraph`` + ``CanonicalComponentScheduleAnalysis`` pair:
unsupported execution shapes become typed capability blockers instead of plan
construction failures.  Only contradictory/dangling canonical evidence raises
``StageExecutionPlanError``.

The plan owns *semantic* completion stages and cross-stage value lifetimes.  It
does not choose rings, full-history arrays, local variable names, Cython storage,
or legacy ``coord_phase`` values.  Those are downstream lowering decisions.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical_semantic_graph import CanonicalSemanticGraph, CanonicalSemanticNode
from .canonical_schedule import CanonicalComponentScheduleAnalysis


class StageExecutionPlanError(RuntimeError):
    """The canonical graph/schedule pair is internally inconsistent."""


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class ExecutionDomain:
    """Backend-neutral domain evidence for one exact canonical value.

    ``rank=None`` means the canonical program contains iteration semantics whose
    geometry is not yet expressed by the current semantic graph vocabulary (the
    main current example is a reduction whose loop domain remains in ReductionSpec).
    That is representable evidence, not a graph-construction error.
    """

    uid: str
    owner_uid: str
    canonical_role: str
    rank: int | None
    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[int | None, int | None], ...]
    proof_kind: str
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "owner_uid": self.owner_uid,
            "canonical_role": self.canonical_role,
            "rank": self.rank,
            "parameter_names": list(self.parameter_names),
            "bounds": [
                {"lo": lo, "hi": hi} for lo, hi in self.bounds
            ],
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class ExecutionTask:
    """One semantic unit of work in the canonical execution plan."""

    uid: str
    kind: str  # persistent_component | derived_task | pure_map | value | unknown
    canonical_uids: tuple[str, ...]
    stage_indices: tuple[int, ...]
    domain_uids: tuple[str, ...]
    dependency_uids: tuple[str, ...]
    state_semantic: str
    execution_semantic: str
    availability: str | None = None
    component_uid: str | None = None
    schedule_task_uid: str | None = None
    producer_component_uids: tuple[str, ...] = ()
    persistent_source_uids: tuple[str, ...] = ()
    evidence_uids: tuple[str, ...] = ()
    materialized_source_uids: tuple[str, ...] = ()
    access_uids: tuple[str, ...] = ()
    operator_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    note: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "kind": self.kind,
            "canonical_uids": list(self.canonical_uids),
            "stage_indices": list(self.stage_indices),
            "domain_uids": list(self.domain_uids),
            "dependency_uids": list(self.dependency_uids),
            "state_semantic": self.state_semantic,
            "execution_semantic": self.execution_semantic,
            "availability": self.availability,
            "component_uid": self.component_uid,
            "schedule_task_uid": self.schedule_task_uid,
            "producer_component_uids": list(self.producer_component_uids),
            "persistent_source_uids": list(self.persistent_source_uids),
            "evidence_uids": list(self.evidence_uids),
            "materialized_source_uids": list(self.materialized_source_uids),
            "access_uids": list(self.access_uids),
            "operator_uids": list(self.operator_uids),
            "blockers": list(self.blockers),
            "note": self.note,
        }


@dataclass(frozen=True)
class ExecutionBarrier:
    """Explicit semantic completion barrier between canonical components/stages."""

    uid: str
    source_component_uid: str
    target_component_uid: str
    source_stage_index: int | None
    target_stage_index: int | None
    availability: str
    source_uids: tuple[str, ...]
    target_uids: tuple[str, ...]
    evidence_uids: tuple[str, ...]
    proof_kind: str | None
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_component_uid": self.source_component_uid,
            "target_component_uid": self.target_component_uid,
            "source_stage_index": self.source_stage_index,
            "target_stage_index": self.target_stage_index,
            "availability": self.availability,
            "source_uids": list(self.source_uids),
            "target_uids": list(self.target_uids),
            "evidence_uids": list(self.evidence_uids),
            "proof_kind": self.proof_kind,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class ValueLifetime:
    """Semantic lifetime evidence, deliberately before concrete storage choice."""

    uid: str
    value_uid: str
    state_semantic: str
    producer_stage_index: int | None
    consumer_stage_indices: tuple[int, ...]
    preservation_kind: str
    materialize_after_stage: int | None
    availabilities: tuple[str, ...] = ()
    evidence_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "value_uid": self.value_uid,
            "state_semantic": self.state_semantic,
            "producer_stage_index": self.producer_stage_index,
            "consumer_stage_indices": list(self.consumer_stage_indices),
            "preservation_kind": self.preservation_kind,
            "materialize_after_stage": self.materialize_after_stage,
            "availabilities": list(self.availabilities),
            "evidence_uids": list(self.evidence_uids),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class ExecutionStage:
    """A semantic completion stage, not a legacy executable phase."""

    index: int
    component_task_uids: tuple[str, ...]
    derived_task_uids: tuple[str, ...]
    value_task_uids: tuple[str, ...]
    scan_directions: tuple[str, ...]
    incoming_barrier_uids: tuple[str, ...] = ()
    outgoing_barrier_uids: tuple[str, ...] = ()
    materialized_source_uids: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "component_task_uids": list(self.component_task_uids),
            "derived_task_uids": list(self.derived_task_uids),
            "value_task_uids": list(self.value_task_uids),
            "scan_directions": list(self.scan_directions),
            "incoming_barrier_uids": list(self.incoming_barrier_uids),
            "outgoing_barrier_uids": list(self.outgoing_barrier_uids),
            "materialized_source_uids": list(self.materialized_source_uids),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageExecutionPlan:
    schema: str
    uid: str
    graph_schema: str
    schedule_schema: str
    output_uid: str
    formula_hash: str
    structural_hash: str
    stages: tuple[ExecutionStage, ...]
    tasks: tuple[ExecutionTask, ...]
    domains: tuple[ExecutionDomain, ...]
    lifetimes: tuple[ValueLifetime, ...]
    barriers: tuple[ExecutionBarrier, ...]
    unplaced_task_uids: tuple[str, ...]
    backend_eligible: bool
    capability_blockers: tuple[str, ...] = ()
    validation_notes: tuple[str, ...] = ()

    def task(self, uid: str) -> ExecutionTask:
        return next(x for x in self.tasks if x.uid == uid)

    def lifetime(self, value_uid: str) -> ValueLifetime:
        return next(x for x in self.lifetimes if x.value_uid == value_uid)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "uid": self.uid,
            "graph_schema": self.graph_schema,
            "schedule_schema": self.schedule_schema,
            "output_uid": self.output_uid,
            "formula_hash": self.formula_hash,
            "structural_hash": self.structural_hash,
            "stages": [x.manifest() for x in self.stages],
            "tasks": [x.manifest() for x in self.tasks],
            "domains": [x.manifest() for x in self.domains],
            "lifetimes": [x.manifest() for x in self.lifetimes],
            "barriers": [x.manifest() for x in self.barriers],
            "unplaced_task_uids": list(self.unplaced_task_uids),
            "backend_eligible": bool(self.backend_eligible),
            "capability_blockers": list(self.capability_blockers),
            "metrics": {
                "stage_count": len(self.stages),
                "task_count": len(self.tasks),
                "domain_count": len(self.domains),
                "lifetime_count": len(self.lifetimes),
                "barrier_count": len(self.barriers),
                "unplaced_task_count": len(self.unplaced_task_uids),
                "materialized_value_count": sum(
                    x.materialize_after_stage is not None for x in self.lifetimes
                ),
            },
            "validation_notes": list(self.validation_notes),
        }


def _domain_for_node(node: CanonicalSemanticNode) -> ExecutionDomain:
    params = tuple(node.parameters)
    blockers: list[str] = []
    if params:
        rank: int | None = len(params)
        if rank == 1 and node.domain is not None:
            bounds = ((int(node.domain[0]), int(node.domain[1])),)
            proof_kind = "canonical_parameter_domain_v1"
        else:
            bounds = tuple((None, None) for _ in params)
            proof_kind = "canonical_parameter_shape_v1"
            if rank > 1:
                blockers.append("axis_bounds_unproved_for_multiaxis_value")
    elif node.domain is not None:
        rank = 1
        bounds = ((int(node.domain[0]), int(node.domain[1])),)
        proof_kind = "canonical_coordinate_domain_v1"
    elif node.canonical_role == "scalar":
        rank = 0
        bounds = ()
        proof_kind = "scalar_domain_v1"
    elif node.canonical_role == "reduction":
        rank = None
        bounds = ()
        proof_kind = "reduction_domain_preserved_in_canonical_spec_v1"
        blockers.append("reduction_domain_geometry_not_lifted_yet")
    elif node.canonical_role == "vector":
        rank = None
        bounds = ()
        proof_kind = "vector_domain_shape_unproved_v1"
        blockers.append("vector_domain_geometry_unproved")
    else:
        rank = None
        bounds = ()
        proof_kind = "canonical_domain_unknown_v1"
        blockers.append("domain_geometry_unproved")
    return ExecutionDomain(
        uid=_stable_id("execution_domain", (node.uid, node.canonical_role, rank, params, bounds)),
        owner_uid=node.uid,
        canonical_role=node.canonical_role,
        rank=rank,
        parameter_names=params,
        bounds=bounds,
        proof_kind=proof_kind,
        blockers=tuple(blockers),
    )

def _backend_shape_blockers(node: CanonicalSemanticNode, domain: ExecutionDomain) -> tuple[str, ...]:
    rows: list[str] = []
    if domain.rank is not None and domain.rank > 1:
        rows.append(f"backend_multiaxis_domain_not_supported:{node.uid}:{domain.rank}")
    if node.canonical_role == "vector":
        rows.append(f"backend_vector_value_not_supported:{node.uid}")
    return tuple(rows)


def build_stage_execution_plan(
    graph: CanonicalSemanticGraph,
    analysis: CanonicalComponentScheduleAnalysis,
) -> StageExecutionPlan:
    """Build a total stage/lifetime plan without changing production execution.

    Unsupported backend forms are encoded in ``capability_blockers``.  Contradictory
    exact-UID evidence raises because continuing would corrupt semantic identity.
    """

    schedule = analysis.component_schedule
    nodes = {x.uid: x for x in graph.nodes}
    if len(nodes) != len(graph.nodes):
        raise StageExecutionPlanError("duplicate canonical node UID")
    if graph.output_uid not in nodes:
        raise StageExecutionPlanError(f"output UID is absent from canonical graph: {graph.output_uid}")

    for access in graph.accesses:
        if access.source_uid not in nodes or access.target_uid not in nodes:
            raise StageExecutionPlanError(
                f"dangling canonical access {access.uid}: {access.source_uid}->{access.target_uid}"
            )

    components = {x.uid: x for x in schedule.components}
    if len(components) != len(schedule.components):
        raise StageExecutionPlanError("duplicate persistent component UID")

    persistent_to_component: dict[str, str] = {}
    for component in schedule.components:
        for uid in component.persistent_uids:
            if uid not in nodes:
                raise StageExecutionPlanError(
                    f"persistent component {component.uid} references unknown canonical UID {uid}"
                )
            prior = persistent_to_component.get(uid)
            if prior is not None and prior != component.uid:
                raise StageExecutionPlanError(
                    f"persistent canonical UID {uid} appears in multiple components: {prior}, {component.uid}"
                )
            if nodes[uid].state_semantic != "persistent_state":
                raise StageExecutionPlanError(
                    f"component {component.uid} contains non-persistent value {uid}: {nodes[uid].state_semantic}"
                )
            persistent_to_component[uid] = component.uid

    stage_by_component: dict[str, int] = {}
    stage_rows = {int(x.index): x for x in schedule.stages}
    if len(stage_rows) != len(schedule.stages):
        raise StageExecutionPlanError("duplicate canonical stage index")
    for stage in schedule.stages:
        for component_uid in stage.component_uids:
            if component_uid not in components:
                raise StageExecutionPlanError(
                    f"stage {stage.index} references unknown component {component_uid}"
                )
            prior = stage_by_component.get(component_uid)
            if prior is not None and prior != stage.index:
                raise StageExecutionPlanError(
                    f"component {component_uid} appears in multiple canonical stages: {prior}, {stage.index}"
                )
            stage_by_component[component_uid] = int(stage.index)

    derived_by_target: dict[str, Any] = {}
    derived_by_uid: dict[str, Any] = {}
    for task in schedule.derived_tasks:
        if task.uid in derived_by_uid:
            raise StageExecutionPlanError(f"duplicate derived schedule task UID {task.uid}")
        if task.target_uid not in nodes:
            raise StageExecutionPlanError(
                f"derived task {task.uid} references unknown target {task.target_uid}"
            )
        if task.target_uid in persistent_to_component:
            raise StageExecutionPlanError(
                f"StateDerivedMap target {task.target_uid} is also persistent state"
            )
        for component_uid in task.producer_component_uids:
            if component_uid not in components:
                raise StageExecutionPlanError(
                    f"derived task {task.uid} references unknown producer component {component_uid}"
                )
        derived_by_uid[task.uid] = task
        prior = derived_by_target.get(task.target_uid)
        if prior is not None and prior.uid != task.uid:
            raise StageExecutionPlanError(
                f"canonical target {task.target_uid} has multiple derived schedule tasks"
            )
        derived_by_target[task.target_uid] = task

    domains = tuple(_domain_for_node(node) for node in sorted(graph.nodes, key=lambda x: x.uid))
    domain_by_owner = {x.owner_uid: x for x in domains}

    deps_by_target: dict[str, set[str]] = {uid: set() for uid in nodes}
    access_uids_by_target: dict[str, set[str]] = {uid: set() for uid in nodes}
    consumers_by_source: dict[str, set[str]] = {uid: set() for uid in nodes}
    for access in graph.accesses:
        deps_by_target[access.target_uid].add(access.source_uid)
        access_uids_by_target[access.target_uid].add(access.uid)
        consumers_by_source[access.source_uid].add(access.target_uid)
    operators_by_target: dict[str, tuple[str, ...]] = {}
    for uid in nodes:
        operators_by_target[uid] = tuple(sorted(
            x.uid for x in graph.operators if x.target_uid == uid
        ))

    # Semantic *execution* placement.  A derived schedule ``stage_index`` is a
    # lower bound on availability, not necessarily the stage in which a value must
    # execute.  Scalar/reduction results can be computed once at their earliest
    # valid stage and retained cheaply; coordinate-valued maps must execute in the
    # stage(s) that request their coordinates.  Keeping these cases distinct is
    # what lets completed-history work move to a later stage without dragging all
    # ordinary reductions, and their persistent inputs, across the boundary.
    stage_sets: dict[str, set[int]] = {uid: set() for uid in nodes}
    minimum_stage: dict[str, int] = {}
    unresolved_derived_targets = {
        task.target_uid for task in schedule.derived_tasks if task.stage_index is None
    }
    for uid, component_uid in persistent_to_component.items():
        if component_uid in stage_by_component:
            stage_sets[uid].add(stage_by_component[component_uid])
            minimum_stage[uid] = stage_by_component[component_uid]
    for target_uid, task in derived_by_target.items():
        if task.stage_index is not None:
            minimum_stage[target_uid] = int(task.stage_index)

    known_stage_indices = sorted(stage_rows)
    first_stage = min(known_stage_indices) if known_stage_indices else 0

    # Scalar/reduction values have one execution stage.  Start at their proven
    # minimum, then propagate later scalar/reduction prerequisites forward.  This
    # moves the final scalar output behind stage-1 reductions while leaving ordinary
    # stage-0 reductions in stage 0.
    singleton_stage: dict[str, int] = {}
    for uid, node in nodes.items():
        if uid in unresolved_derived_targets:
            continue
        if node.canonical_role in {"scalar", "reduction"}:
            singleton_stage[uid] = int(minimum_stage.get(uid, first_stage))
    changed = True
    while changed:
        changed = False
        for access in graph.accesses:
            target_uid = access.target_uid
            source_uid = access.source_uid
            if target_uid not in singleton_stage:
                continue
            if source_uid not in singleton_stage:
                continue
            required = max(
                singleton_stage[target_uid],
                singleton_stage[source_uid],
                int(minimum_stage.get(target_uid, first_stage)),
            )
            if required != singleton_stage[target_uid]:
                singleton_stage[target_uid] = required
                changed = True
    for uid, stage_index in singleton_stage.items():
        stage_sets[uid].add(stage_index)

    # Coordinate-valued nonpersistent work is recomputed at the coordinates of its
    # consumers.  Seed demands from already-placed persistent/scalar/reduction
    # consumers and propagate them backwards through coordinate helper chains.
    # Persistent values never move; a later consumer becomes lifetime evidence.
    changed = True
    while changed:
        changed = False
        for access in graph.accesses:
            source_uid = access.source_uid
            target_uid = access.target_uid
            if source_uid in persistent_to_component or source_uid in unresolved_derived_targets:
                continue
            source_node = nodes[source_uid]
            if source_node.canonical_role in {"scalar", "reduction"}:
                continue
            demanded = set(stage_sets[target_uid])
            if not demanded:
                continue
            lower = int(minimum_stage.get(source_uid, first_stage))
            demanded = {x for x in demanded if x >= lower}
            before = len(stage_sets[source_uid])
            stage_sets[source_uid].update(demanded)
            if len(stage_sets[source_uid]) != before:
                changed = True

    # Non-derived coordinate output is rare, but if present it must have a concrete
    # final-stage demand.  UNKNOWN derived outputs deliberately remain unplaced.
    if (
        graph.output_uid not in unresolved_derived_targets
        and not stage_sets[graph.output_uid]
        and known_stage_indices
    ):
        stage_sets[graph.output_uid].add(max(known_stage_indices))

    capability_blockers: set[str] = set()
    if not schedule.persistent_schedule_proven:
        capability_blockers.add("persistent_schedule_not_proven")
    if not schedule.full_schedule_proven:
        capability_blockers.add("full_schedule_not_proven")
    if schedule.deferred_unknowns:
        capability_blockers.add(f"deferred_unknown_tasks:{len(schedule.deferred_unknowns)}")
    capability_blockers.update(f"schedule_blocker:{x}" for x in schedule.blockers)

    task_rows: list[ExecutionTask] = []
    component_task_uid: dict[str, str] = {}
    for component in sorted(schedule.components, key=lambda x: x.uid):
        stages = () if component.uid not in stage_by_component else (stage_by_component[component.uid],)
        blockers = list(component.blockers)
        for member_uid in component.persistent_uids:
            for blocker in _backend_shape_blockers(nodes[member_uid], domain_by_owner[member_uid]):
                blockers.append(blocker)
                capability_blockers.add(blocker)
        if component.intrinsic_scan in {"mixed", "unknown"}:
            blockers.append(f"component_scan_unproved:{component.intrinsic_scan}")
        if not stages:
            blockers.append("persistent_component_stage_unproved")
        for blocker in blockers:
            capability_blockers.add(f"component:{component.uid}:{blocker}")
        dependency_uids = tuple(sorted({
            dep
            for member_uid in component.persistent_uids
            for dep in deps_by_target.get(member_uid, ())
        }))
        operator_uids = tuple(sorted({
            op
            for member_uid in component.persistent_uids
            for op in operators_by_target.get(member_uid, ())
        }))
        task_uid = _stable_id("execution_task_component", (
            component.uid, component.persistent_uids, stages, component.intrinsic_scan
        ))
        component_task_uid[component.uid] = task_uid
        task_rows.append(ExecutionTask(
            uid=task_uid,
            kind="persistent_component",
            canonical_uids=tuple(component.persistent_uids),
            stage_indices=stages,
            domain_uids=tuple(sorted(
                domain_by_owner[uid].uid for uid in component.domain_uids if uid in domain_by_owner
            )),
            dependency_uids=dependency_uids,
            state_semantic="persistent_state",
            execution_semantic="scheduled_state",
            availability=None,
            component_uid=component.uid,
            access_uids=tuple(sorted({
                access_uid
                for member_uid in component.persistent_uids
                for access_uid in access_uids_by_target.get(member_uid, ())
            })),
            operator_uids=operator_uids,
            blockers=tuple(sorted(set(blockers))),
            note="exact canonical persistent SCC/component task",
        ))

    derived_task_uid: dict[str, str] = {}
    for task in sorted(schedule.derived_tasks, key=lambda x: x.uid):
        node = nodes[task.target_uid]
        stages = tuple(sorted(stage_sets[task.target_uid]))
        blockers = list(task.blockers)
        for blocker in _backend_shape_blockers(node, domain_by_owner[task.target_uid]):
            blockers.append(blocker)
            capability_blockers.add(blocker)
        if task.availability == "UNKNOWN":
            blockers.append("derived_task_availability_unknown")
        if task.stage_index is None:
            blockers.append("derived_task_stage_unproved")
        for blocker in blockers:
            capability_blockers.add(f"derived_task:{task.uid}:{blocker}")
        uid = _stable_id("execution_task_derived", (task.uid, task.target_uid, stages, task.availability))
        derived_task_uid[task.uid] = uid
        task_rows.append(ExecutionTask(
            uid=uid,
            kind="derived_task",
            canonical_uids=(task.target_uid,),
            stage_indices=stages,
            domain_uids=(domain_by_owner[task.target_uid].uid,),
            dependency_uids=tuple(sorted(deps_by_target.get(task.target_uid, ()))),
            state_semantic=node.state_semantic,
            execution_semantic=node.execution_semantic,
            availability=task.availability,
            schedule_task_uid=task.uid,
            producer_component_uids=tuple(task.producer_component_uids),
            persistent_source_uids=tuple(task.persistent_source_uids),
            evidence_uids=tuple(task.evidence_uids),
            materialized_source_uids=tuple(task.materialized_source_uids),
            access_uids=tuple(sorted(access_uids_by_target.get(task.target_uid, ()))),
            operator_uids=operators_by_target[task.target_uid],
            blockers=tuple(sorted(set(blockers))),
            note=task.note,
        ))

    covered = set(persistent_to_component) | set(derived_by_target)
    for node in sorted(graph.nodes, key=lambda x: x.uid):
        if node.uid in covered:
            continue
        if node.execution_semantic == "pure_map":
            kind = "pure_map"
        elif node.execution_semantic == "unknown":
            kind = "unknown"
        else:
            kind = "value"
        blockers = list(node.semantic_blockers)
        domain = domain_by_owner[node.uid]
        if (
            node.execution_semantic == "unknown"
            and node.state_semantic == "state_free"
            and node.canonical_role in {"coordinate", "reduction", "vector"}
        ):
            blocker = "state_free_execution_operator_unapproved"
            blockers.append(blocker)
            capability_blockers.add(f"execution_semantic_unapproved:{node.uid}")
        for blocker in _backend_shape_blockers(node, domain):
            blockers.append(blocker)
            capability_blockers.add(blocker)
        stages = tuple(sorted(stage_sets[node.uid]))
        # Unplaced non-output values are still represented.  If they are reachable
        # only through an UNKNOWN task, lack of placement is evidence, not failure.
        if not stages and node.uid == graph.output_uid:
            blockers.append("output_stage_unproved")
            capability_blockers.add(f"output_stage_unproved:{node.uid}")
        uid = _stable_id("execution_task_value", (node.uid, kind, stages))
        task_rows.append(ExecutionTask(
            uid=uid,
            kind=kind,
            canonical_uids=(node.uid,),
            stage_indices=stages,
            domain_uids=(domain.uid,),
            dependency_uids=tuple(sorted(deps_by_target.get(node.uid, ()))),
            state_semantic=node.state_semantic,
            execution_semantic=node.execution_semantic,
            access_uids=tuple(sorted(access_uids_by_target.get(node.uid, ()))),
            operator_uids=operators_by_target[node.uid],
            blockers=tuple(sorted(set(blockers))),
            note="exact canonical non-carried value task",
        ))

    # Build explicit barrier rows from scheduler evidence.  Unsupported barrier
    # execution is a capability blocker, never a plan-construction exception.
    barriers: list[ExecutionBarrier] = []
    for constraint in sorted(schedule.constraints, key=lambda x: x.uid):
        if not constraint.barrier_required:
            continue
        if constraint.source_component_uid not in components or constraint.target_component_uid not in components:
            raise StageExecutionPlanError(
                f"barrier {constraint.uid} references an unknown component"
            )
        row = ExecutionBarrier(
            uid=constraint.uid,
            source_component_uid=constraint.source_component_uid,
            target_component_uid=constraint.target_component_uid,
            source_stage_index=stage_by_component.get(constraint.source_component_uid),
            target_stage_index=stage_by_component.get(constraint.target_component_uid),
            availability=constraint.availability,
            source_uids=tuple(constraint.source_uids),
            target_uids=tuple(constraint.target_uids),
            evidence_uids=tuple(constraint.evidence_uids),
            proof_kind=constraint.proof_kind,
            blockers=tuple(constraint.blockers),
        )
        barriers.append(row)

    task_by_canonical: dict[str, list[ExecutionTask]] = {uid: [] for uid in nodes}
    for task in task_rows:
        for uid in task.canonical_uids:
            task_by_canonical[uid].append(task)

    # Lifetime evidence.  Materialization is derived only from explicit schedule
    # task/barrier evidence, never from legacy history allocation.
    materialization_evidence: dict[str, set[str]] = {uid: set() for uid in nodes}
    materialization_availability: dict[str, set[str]] = {uid: set() for uid in nodes}
    materialize_after: dict[str, set[int]] = {uid: set() for uid in nodes}
    for task in schedule.derived_tasks:
        if task.stage_index is None:
            continue
        for source_uid in task.materialized_source_uids:
            if source_uid not in nodes:
                raise StageExecutionPlanError(
                    f"derived task {task.uid} materializes unknown value {source_uid}"
                )
            component_uid = persistent_to_component.get(source_uid)
            source_stage = None if component_uid is None else stage_by_component.get(component_uid)
            if source_stage is not None:
                materialize_after[source_uid].add(source_stage)
            materialization_evidence[source_uid].update(task.evidence_uids or (task.uid,))
            materialization_availability[source_uid].add(task.availability)
    for barrier in barriers:
        for source_uid in barrier.source_uids:
            if source_uid not in nodes:
                raise StageExecutionPlanError(
                    f"barrier {barrier.uid} materializes unknown value {source_uid}"
                )
            if barrier.source_stage_index is not None:
                materialize_after[source_uid].add(barrier.source_stage_index)
            materialization_evidence[source_uid].update(barrier.evidence_uids or (barrier.uid,))
            materialization_availability[source_uid].add(barrier.availability)

    consumer_stages: dict[str, set[int]] = {uid: set() for uid in nodes}
    for access in graph.accesses:
        for stage_index in stage_sets[access.target_uid]:
            consumer_stages[access.source_uid].add(stage_index)
    for task in schedule.derived_tasks:
        for stage_index in stage_sets[task.target_uid]:
            for source_uid in task.persistent_source_uids:
                if source_uid in consumer_stages:
                    consumer_stages[source_uid].add(int(stage_index))
    for barrier in barriers:
        if barrier.target_stage_index is not None:
            for source_uid in barrier.source_uids:
                consumer_stages[source_uid].add(barrier.target_stage_index)

    lifetime_rows: list[ValueLifetime] = []
    for node in sorted(graph.nodes, key=lambda x: x.uid):
        producer: int | None
        if node.uid in persistent_to_component:
            producer = stage_by_component.get(persistent_to_component[node.uid])
        elif node.uid in derived_by_target:
            producer = min(stage_sets[node.uid]) if stage_sets[node.uid] else None
        elif stage_sets[node.uid]:
            producer = min(stage_sets[node.uid])
        else:
            producer = None
        consumers = tuple(sorted(consumer_stages[node.uid]))
        mats = sorted(materialize_after[node.uid])
        blockers: list[str] = []
        materialize_stage = mats[0] if mats else None
        if len(mats) > 1:
            blockers.append("multiple_materialization_boundaries_for_value")
            capability_blockers.add(f"multiple_materialization_boundaries:{node.uid}")
        if node.state_semantic == "state_free":
            preservation = "recompute"
            materialize_stage = None
            if mats:
                blockers.append("state_free_value_requested_for_materialization")
                capability_blockers.add(f"invalid_state_free_materialization:{node.uid}")
        elif node.state_semantic == "persistent_state":
            if materialize_stage is not None:
                preservation = "materialized_cross_stage"
            elif producer is not None and any(x > producer for x in consumers):
                # Only carried persistent state requires preservation across a
                # semantic stage boundary. Derived/scalar values may be recomputed
                # in the consumer stage from their exact canonical dependencies.
                preservation = "persistent_cross_stage_unmaterialized"
                blockers.append("persistent_cross_stage_consumer_without_materialization_evidence")
                capability_blockers.add(f"persistent_cross_stage_unmaterialized:{node.uid}")
            else:
                preservation = "stage_local_persistent"
        elif node.state_semantic == "state_derived_map":
            preservation = "derived_recompute"
            materialize_stage = None
            if mats:
                blockers.append("state_derived_value_requested_for_materialization")
                capability_blockers.add(f"invalid_state_derived_materialization:{node.uid}")
        else:
            preservation = "value_recompute_or_stage_local"
            materialize_stage = None
            if mats:
                blockers.append("nonpersistent_value_requested_for_materialization")
                capability_blockers.add(f"invalid_nonpersistent_materialization:{node.uid}")
        lifetime_rows.append(ValueLifetime(
            uid=_stable_id("value_lifetime", (
                node.uid, producer, consumers, preservation, materialize_stage,
                tuple(sorted(materialization_availability[node.uid])),
            )),
            value_uid=node.uid,
            state_semantic=node.state_semantic,
            producer_stage_index=None if producer is None else int(producer),
            consumer_stage_indices=consumers,
            preservation_kind=preservation,
            materialize_after_stage=materialize_stage,
            availabilities=tuple(sorted(materialization_availability[node.uid])),
            evidence_uids=tuple(sorted(materialization_evidence[node.uid])),
            blockers=tuple(sorted(set(blockers))),
        ))

    task_rows = sorted(task_rows, key=lambda x: x.uid)
    task_map = {x.uid: x for x in task_rows}
    stage_execution_rows: list[ExecutionStage] = []
    incoming: dict[int, set[str]] = {idx: set() for idx in stage_rows}
    outgoing: dict[int, set[str]] = {idx: set() for idx in stage_rows}
    for barrier in barriers:
        if barrier.source_stage_index is not None:
            outgoing.setdefault(barrier.source_stage_index, set()).add(barrier.uid)
        if barrier.target_stage_index is not None:
            incoming.setdefault(barrier.target_stage_index, set()).add(barrier.uid)

    for index, stage in sorted(stage_rows.items()):
        comp_tasks = tuple(sorted(
            component_task_uid[uid] for uid in stage.component_uids
        ))
        deriv_tasks = tuple(sorted(
            task.uid for task in task_rows
            if task.kind == "derived_task" and index in task.stage_indices
        ))
        value_tasks = tuple(sorted(
            task.uid for task in task_rows
            if task.kind not in {"persistent_component", "derived_task"}
            and index in task.stage_indices
        ))
        stage_blockers: list[str] = []
        if len(set(stage.scan_directions)) > 1:
            stage_blockers.append("multiple_component_scan_directions")
        stage_execution_rows.append(ExecutionStage(
            index=index,
            component_task_uids=comp_tasks,
            derived_task_uids=deriv_tasks,
            value_task_uids=value_tasks,
            scan_directions=tuple(stage.scan_directions),
            incoming_barrier_uids=tuple(sorted(incoming.get(index, ()))),
            outgoing_barrier_uids=tuple(sorted(outgoing.get(index, ()))),
            materialized_source_uids=tuple(stage.materialized_source_uids),
            blockers=tuple(stage_blockers),
        ))

    unplaced = tuple(sorted(
        task.uid for task in task_rows if not task.stage_indices
    ))

    # The current backend-compatible subset is intentionally the same semantic
    # capability boundary as V02349, independently reconstructed here.  The audit
    # compares this boolean to CanonicalExecutionSchedule.eligible.
    backend_eligible = not capability_blockers

    plan_uid = _stable_id("stage_execution_plan", (
        graph.formula_hash,
        graph.structural_hash,
        schedule.schema,
        *(f"stage:{x.index}:{x.component_task_uids}:{x.derived_task_uids}" for x in stage_execution_rows),
        *(f"barrier:{x.uid}" for x in barriers),
        *sorted(capability_blockers),
    ))
    return StageExecutionPlan(
        schema="modelx_graph.stage_execution_plan.v1",
        uid=plan_uid,
        graph_schema=graph.schema,
        schedule_schema=schedule.schema,
        output_uid=graph.output_uid,
        formula_hash=graph.formula_hash,
        structural_hash=graph.structural_hash,
        stages=tuple(stage_execution_rows),
        tasks=tuple(task_rows),
        domains=domains,
        lifetimes=tuple(lifetime_rows),
        barriers=tuple(barriers),
        unplaced_task_uids=unplaced,
        backend_eligible=backend_eligible,
        capability_blockers=tuple(sorted(capability_blockers)),
        validation_notes=(
            "canonical stage is a semantic completion boundary and is not legacy coord_phase",
            "every canonical node is represented by a task or persistent component task",
            "unsupported execution/domain/barrier shapes are typed capability blockers",
            "materialization derives only from canonical task/barrier evidence",
            "no concrete ring/history/storage representation is selected in this IR",
        ),
    )
