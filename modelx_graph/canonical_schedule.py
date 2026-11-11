from __future__ import annotations

"""Canonical state-footprint, component scheduling, and authority contracts.

This module is downstream of :mod:`canonical_semantic_graph` and upstream of the
existing executable/storage/codegen ABI. Exact canonical UIDs are the only
scheduling identities. All formula/source semantics needed for scheduling are frozen into the semantic
graph before this module runs.  This module is intentionally graph-only: it never
parses or inspects Python formula syntax.

V02348 introduced the observational component proof. V02349 adds a strict
``CanonicalExecutionSchedule`` capability gate: fully proven one-stage schedules
may own persistent scan/completion-stage semantics while the existing
``ExecutableGraph`` remains the lowering ABI. Ineligible schedules stay explicit
and use legacy authority; disagreement on an eligible schedule is a hard error.
"""

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .canonical_semantic_graph import (
    CanonicalFootprintPath,
    CanonicalSemanticGraph,
    CanonicalStateFootprintEvidence,
    CanonicalTransitionEvidence,
)
from .canonical_semantic_analysis import (
    _PartialPath,
    _compose_bounds,
    _dedupe_partial_paths,
    _is_geometry_blocker,
    _period_floor_plus_constant_bounds,
    _scan_requirement,
)
from .domain_graph import (
    ComponentScheduleComponent,
    ComponentScheduleConstraint,
    ComponentSchedulePlan,
    ComponentScheduleStage,
    CrossComponentOrderProof,
    DerivedScheduleTask,
    StageScanOrderProof,
    _aggregate_availability,
    _assign_component_stages,
    _graph_path_exists,
    _tarjan_source_components,
    _topological_component_order,
)


_MONOTONE_SCANS = frozenset({"ascending", "descending"})


def _component_scan_options(scan: str) -> frozenset[str]:
    if scan == "ascending":
        return frozenset({"ascending"})
    if scan == "descending":
        return frozenset({"descending"})
    if scan == "any":
        return _MONOTONE_SCANS
    return frozenset()


def _transition_scan_options(
    min_offset: int | None,
    max_offset: int | None,
) -> frozenset[str]:
    """Monotone orders in which one exact transition can be consumed inline."""
    if min_offset is None or max_offset is None:
        return frozenset()
    lo = int(min_offset)
    hi = int(max_offset)
    if lo > hi:
        return frozenset()
    if lo == 0 and hi == 0:
        return _MONOTONE_SCANS
    if hi <= 0:
        return frozenset({"ascending"})
    if lo >= 0:
        return frozenset({"descending"})
    # A dependency spanning both past and future coordinates cannot be
    # satisfied by a single monotone pass.
    return frozenset()


def _build_cross_component_order_proof(
    *,
    source_component: str,
    target_component: str,
    source_scan: str,
    target_scan: str,
    rows: list[CanonicalTransitionEvidence],
) -> CrossComponentOrderProof:
    source_options = _component_scan_options(source_scan)
    target_options = _component_scan_options(target_scan)
    blockers: list[str] = []
    if not source_options:
        blockers.append(
            f"canonical_cross_component_source_scan_unproved:{source_component}:{source_scan}"
        )
    if not target_options:
        blockers.append(
            f"canonical_cross_component_target_scan_unproved:{target_component}:{target_scan}"
        )

    dependency_options = set(_MONOTONE_SCANS)
    exact_mins: list[int] = []
    exact_maxs: list[int] = []
    row_options: dict[str, frozenset[str]] = {}
    for row in rows:
        options = _transition_scan_options(row.min_offset, row.max_offset)
        row_options[row.uid] = options
        if row.min_offset is None or row.max_offset is None:
            blockers.extend(row.blockers or (
                f"canonical_cross_component_transition_bounds_unproved:{row.uid}",
            ))
            continue
        exact_mins.append(int(row.min_offset))
        exact_maxs.append(int(row.max_offset))
        dependency_options.intersection_update(options)

    dependency_proven = len(exact_mins) == len(rows) and not blockers
    co_scan = (
        set(source_options) & set(target_options) & dependency_options
        if dependency_proven
        else set()
    )
    requires_fence = bool(dependency_proven and not co_scan)

    reasons: list[str] = []
    if requires_fence:
        component_common = set(source_options) & set(target_options)
        if not component_common:
            reasons.append("component_scan_order_incompatible")
        if not dependency_options:
            reasons.append("dependency_requires_full_history")
        elif component_common and not (component_common & dependency_options):
            reasons.append("dependency_scan_order_incompatible")

    forcing: list[str] = []
    if requires_fence:
        for row in rows:
            options = row_options[row.uid]
            if not (set(source_options) & set(target_options) & set(options)):
                forcing.append(row.uid)
        # Several individually co-scannable transitions can require opposite
        # orders in aggregate.  In that case the group, not one row, forces the
        # fence, so retain every transition as forcing evidence.
        if not forcing:
            forcing = [row.uid for row in rows]

    uid = _stable_id("canonical_cross_component_order", (
        source_component,
        target_component,
        tuple(sorted(row.uid for row in rows)),
        tuple(sorted(source_options)),
        tuple(sorted(target_options)),
        tuple(sorted(dependency_options)) if dependency_proven else (),
    ))
    return CrossComponentOrderProof(
        uid=uid,
        source_component_uid=source_component,
        target_component_uid=target_component,
        source_scan_options=tuple(sorted(source_options)),
        target_scan_options=tuple(sorted(target_options)),
        dependency_scan_options=(
            tuple(sorted(dependency_options)) if dependency_proven else ()
        ),
        co_scan_options=tuple(sorted(co_scan)),
        required_offset_min=min(exact_mins) if dependency_proven else None,
        required_offset_max=max(exact_maxs) if dependency_proven else None,
        requires_stage_fence=requires_fence,
        fence_reasons=tuple(reasons),
        evidence_uids=tuple(sorted(row.uid for row in rows)),
        forcing_evidence_uids=tuple(sorted(forcing)),
        blockers=tuple(sorted(set(blockers))),
    )


def _stage_scan_options(
    *,
    members: Iterable[str],
    component_scan: dict[str, str],
    order_proofs_by_edge: dict[tuple[str, str], CrossComponentOrderProof],
    barrier_edges: set[tuple[str, str]],
) -> tuple[frozenset[str], tuple[str, ...], tuple[str, ...]]:
    """Return exact global scan options for one candidate persistent stage.

    The proof is intentionally set-theoretic.  Component intrinsic options and
    every non-fence dependency that remains *inside* the stage must share at
    least one monotone direction.  Unknown edge geometry remains a blocker and
    is never repaired by inserting an implicit fence.
    """
    member_set = set(members)
    options = set(_MONOTONE_SCANS)
    blockers: list[str] = []
    proof_uids: list[str] = []
    for uid in sorted(member_set):
        component_options = _component_scan_options(component_scan.get(uid, "unknown"))
        if not component_options:
            blockers.append(
                f"canonical_stage_component_scan_unproved:{uid}:{component_scan.get(uid, 'unknown')}"
            )
        else:
            options.intersection_update(component_options)

    for edge, proof in sorted(order_proofs_by_edge.items()):
        source, target = edge
        if source not in member_set or target not in member_set or edge in barrier_edges:
            continue
        proof_uids.append(proof.uid)
        if proof.blockers:
            blockers.append(f"canonical_stage_order_proof_unproved:{proof.uid}")
            continue
        if proof.requires_stage_fence:
            blockers.append(f"canonical_stage_required_fence_missing:{proof.uid}")
            continue
        if not proof.co_scan_options:
            blockers.append(f"canonical_stage_order_options_empty:{proof.uid}")
            continue
        options.intersection_update(proof.co_scan_options)

    if blockers:
        return frozenset(), tuple(sorted(set(proof_uids))), tuple(sorted(set(blockers)))
    return frozenset(options), tuple(sorted(set(proof_uids))), ()


def _weak_component_groups(
    members: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    members = tuple(sorted(set(members)))
    adjacency: dict[str, set[str]] = {uid: set() for uid in members}
    member_set = set(members)
    for source, target in edges:
        if source in member_set and target in member_set:
            adjacency[source].add(target)
            adjacency[target].add(source)
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for root in members:
        if root in seen:
            continue
        pending = [root]
        group: list[str] = []
        while pending:
            uid = pending.pop(0)
            if uid in seen:
                continue
            seen.add(uid)
            group.append(uid)
            pending.extend(sorted(adjacency.get(uid, ()) - seen))
        groups.append(tuple(sorted(group)))
    return tuple(sorted(groups))


def _refine_disconnected_stage_partitions(
    *,
    stage_by_component: dict[str, int],
    component_scan: dict[str, str],
    graph_edges: set[tuple[str, str]],
    barrier_edges: set[tuple[str, str]],
    order_proofs_by_edge: dict[tuple[str, str], CrossComponentOrderProof],
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Split only disconnected incompatible blocks without inventing fences."""
    blockers: list[str] = []
    by_stage: dict[int, list[str]] = {}
    for uid, stage in stage_by_component.items():
        by_stage.setdefault(int(stage), []).append(uid)

    staged_blocks: list[list[tuple[tuple[str, ...], frozenset[str]]]] = []
    for base_stage in sorted(by_stage):
        members = tuple(sorted(by_stage[base_stage]))
        options, _proofs, stage_blockers = _stage_scan_options(
            members=members,
            component_scan=component_scan,
            order_proofs_by_edge=order_proofs_by_edge,
            barrier_edges=barrier_edges,
        )
        if stage_blockers:
            blockers.extend(stage_blockers)
            staged_blocks.append([[(members, frozenset())]])
            continue
        if options:
            staged_blocks.append([[(members, options)]])
            continue

        weak_groups = _weak_component_groups(members, graph_edges)
        if len(weak_groups) <= 1:
            blockers.append(
                "canonical_stage_scan_partition_unresolved:" + ",".join(members)
            )
            staged_blocks.append([[(members, frozenset())]])
            continue

        group_rows: list[tuple[tuple[str, ...], frozenset[str]]] = []
        valid = True
        for group in weak_groups:
            group_options, _group_proofs, group_blockers = _stage_scan_options(
                members=group,
                component_scan=component_scan,
                order_proofs_by_edge=order_proofs_by_edge,
                barrier_edges=barrier_edges,
            )
            if group_blockers or not group_options:
                blockers.extend(group_blockers or (
                    "canonical_stage_scan_partition_block_unresolved:" + ",".join(group),
                ))
                valid = False
            group_rows.append((group, group_options))
        if not valid:
            staged_blocks.append([group_rows])
            continue

        local_stages: list[list[tuple[tuple[str, ...], frozenset[str]]]] = []
        local_options: list[set[str]] = []
        for group, group_options in sorted(group_rows):
            placed = False
            for index in range(len(local_stages)):
                common = local_options[index] & set(group_options)
                if common:
                    local_stages[index].append((group, group_options))
                    local_options[index] = common
                    placed = True
                    break
            if not placed:
                local_stages.append([(group, group_options)])
                local_options.append(set(group_options))
        staged_blocks.append(local_stages)

    if blockers:
        return dict(stage_by_component), tuple(sorted(set(blockers)))

    refined: dict[str, int] = {}
    next_stage = 0
    for local_stages in staged_blocks:
        for local_stage in local_stages:
            for group, _options in local_stage:
                for uid in group:
                    refined[uid] = next_stage
            next_stage += 1
    return refined, ()


def _solve_stage_scan_constraints(
    *,
    component_uids: Iterable[str],
    component_scan: dict[str, str],
    graph_edges: set[tuple[str, str]],
    barrier_edges: set[tuple[str, str]],
    cross_component_order_proofs: Iterable[CrossComponentOrderProof],
) -> tuple[
    dict[str, int],
    set[tuple[str, str]],
    tuple[StageScanOrderProof, ...],
    tuple[str, ...],
]:
    """Close pairwise co-scan facts into globally executable stage proofs.

    The existing weighted-SCC scheduler remains the stage-order primitive.  This
    closure promotes the smallest deterministic non-fence edge that improves an
    otherwise-empty stage-wide scan intersection, recomputing weighted stages
    after each promotion.  If conflicting components are dependency-disconnected,
    they may be serialized without inventing a semantic dependency fence.
    """
    component_uids = tuple(sorted(set(component_uids)))
    proofs_by_edge = {
        (proof.source_component_uid, proof.target_component_uid): proof
        for proof in cross_component_order_proofs
    }
    final_barriers = set(barrier_edges)
    blockers: list[str] = []
    promoted_order_proof_uids: set[str] = set()

    def assignment_and_conflicts(current_barriers: set[tuple[str, str]]):
        stage_by_component, _sccs, stage_blockers = _assign_component_stages(
            component_uids, graph_edges, current_barriers
        )
        conflicts: list[tuple[int, tuple[str, ...]]] = []
        unknowns: list[str] = list(stage_blockers)
        by_stage: dict[int, list[str]] = {}
        for uid, stage in stage_by_component.items():
            by_stage.setdefault(int(stage), []).append(uid)
        for stage, members in sorted(by_stage.items()):
            options, _proof_uids, stage_unknowns = _stage_scan_options(
                members=members,
                component_scan=component_scan,
                order_proofs_by_edge=proofs_by_edge,
                barrier_edges=current_barriers,
            )
            if stage_unknowns:
                unknowns.extend(stage_unknowns)
            elif not options:
                conflicts.append((stage, tuple(sorted(members))))
        return stage_by_component, conflicts, tuple(sorted(set(unknowns)))

    stage_by_component, conflicts, unknowns = assignment_and_conflicts(final_barriers)
    if unknowns:
        blockers.extend(unknowns)

    # Unknown geometry is never "fixed" by a partition.  Only fully proven
    # empty intersections participate in deterministic fence promotion.
    if not blockers:
        max_promotions = len(graph_edges)
        promotions = 0
        while conflicts and promotions < max_promotions:
            baseline_metric = (
                len(conflicts),
                sum(len(members) for _stage, members in conflicts),
            )
            conflict_members = set().union(*(set(members) for _stage, members in conflicts))
            candidates = [
                edge for edge in sorted(graph_edges - final_barriers)
                if edge[0] in conflict_members and edge[1] in conflict_members
                and edge in proofs_by_edge
                and not proofs_by_edge[edge].blockers
                and not proofs_by_edge[edge].requires_stage_fence
            ]
            best = None
            best_result = None
            for edge in candidates:
                trial_barriers = set(final_barriers)
                trial_barriers.add(edge)
                trial_stage, trial_conflicts, trial_unknowns = assignment_and_conflicts(
                    trial_barriers
                )
                if trial_unknowns:
                    continue
                metric = (
                    len(trial_conflicts),
                    sum(len(members) for _stage, members in trial_conflicts),
                    proofs_by_edge[edge].uid,
                )
                if metric[:2] >= baseline_metric:
                    continue
                if best is None or metric < best:
                    best = metric
                    best_result = (edge, trial_stage, trial_conflicts)
            if best_result is None:
                break
            edge, stage_by_component, conflicts = best_result
            final_barriers.add(edge)
            promoted_order_proof_uids.add(proofs_by_edge[edge].uid)
            promotions += 1

    if not blockers and conflicts:
        refined, partition_blockers = _refine_disconnected_stage_partitions(
            stage_by_component=stage_by_component,
            component_scan=component_scan,
            graph_edges=graph_edges,
            barrier_edges=final_barriers,
            order_proofs_by_edge=proofs_by_edge,
        )
        if partition_blockers:
            blockers.extend(partition_blockers)
        else:
            stage_by_component = refined
            # Verify the refined stages rather than trusting the partition helper.
            by_stage: dict[int, list[str]] = {}
            for uid, stage in stage_by_component.items():
                by_stage.setdefault(int(stage), []).append(uid)
            for stage, members in sorted(by_stage.items()):
                options, _proofs, stage_blockers = _stage_scan_options(
                    members=members,
                    component_scan=component_scan,
                    order_proofs_by_edge=proofs_by_edge,
                    barrier_edges=final_barriers,
                )
                if stage_blockers:
                    blockers.extend(stage_blockers)
                elif not options:
                    blockers.append(
                        f"canonical_stage_scan_options_empty:{stage}:" + ",".join(sorted(members))
                    )

    stage_proofs: list[StageScanOrderProof] = []
    by_stage: dict[int, list[str]] = {}
    for uid, stage in stage_by_component.items():
        by_stage.setdefault(int(stage), []).append(uid)
    for stage, members in sorted(by_stage.items()):
        members = tuple(sorted(members))
        initial = set(_MONOTONE_SCANS)
        for uid in members:
            initial.intersection_update(_component_scan_options(component_scan.get(uid, "unknown")))
        final_options, proof_uids, stage_blockers = _stage_scan_options(
            members=members,
            component_scan=component_scan,
            order_proofs_by_edge=proofs_by_edge,
            barrier_edges=final_barriers,
        )
        selected = next(iter(final_options)) if len(final_options) == 1 else None
        outgoing_partitions = tuple(sorted(
            proof.uid for edge, proof in proofs_by_edge.items()
            if proof.uid in promoted_order_proof_uids
            and stage_by_component.get(edge[0]) == stage
            and stage_by_component.get(edge[1]) != stage
        ))
        proof_blockers = tuple(sorted(set(stage_blockers)))
        if not proof_blockers and not final_options:
            proof_blockers = (f"canonical_stage_scan_options_empty:{stage}",)
        stage_proofs.append(StageScanOrderProof(
            uid=_stable_id("canonical_stage_scan_order", (
                stage, members, tuple(sorted(proof_uids)),
                tuple(sorted(initial)), tuple(sorted(final_options)), outgoing_partitions,
            )),
            stage_index=int(stage),
            component_uids=members,
            dependency_order_proof_uids=tuple(sorted(proof_uids)),
            initial_scan_options=tuple(sorted(initial)),
            final_scan_options=tuple(sorted(final_options)),
            selected_scan_direction=selected,
            partition_order_proof_uids=outgoing_partitions,
            blockers=proof_blockers,
        ))
        blockers.extend(proof_blockers)

    return (
        stage_by_component,
        final_barriers,
        tuple(stage_proofs),
        tuple(sorted(set(blockers))),
    )


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class CanonicalExecutionNodeFact:
    """Execution-capability facts frozen at the canonical scheduling boundary."""

    uid: str
    canonical_role: str
    state_semantic: str
    execution_semantic: str
    domain: tuple[int, int] | None
    semantic_blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "canonical_role": self.canonical_role,
            "state_semantic": self.state_semantic,
            "execution_semantic": self.execution_semantic,
            "domain": None if self.domain is None else {"lo": int(self.domain[0]), "hi": int(self.domain[1])},
            "semantic_blockers": list(self.semantic_blockers),
        }


@dataclass(frozen=True)
class CanonicalExecutionComponent:
    uid: str
    persistent_uids: tuple[str, ...]
    scan_direction: str
    domain_uids: tuple[str, ...] = ()
    negative_offset_transition_uids: tuple[str, ...] = ()
    positive_offset_transition_uids: tuple[str, ...] = ()
    zero_offset_transition_uids: tuple[str, ...] = ()
    unresolved_transition_uids: tuple[str, ...] = ()
    scan_proof_kind: str = "canonical_signed_transition_evidence_v1"

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "persistent_uids": list(self.persistent_uids),
            "scan_direction": self.scan_direction,
            "domain_uids": list(self.domain_uids),
            "negative_offset_transition_uids": list(self.negative_offset_transition_uids),
            "positive_offset_transition_uids": list(self.positive_offset_transition_uids),
            "zero_offset_transition_uids": list(self.zero_offset_transition_uids),
            "unresolved_transition_uids": list(self.unresolved_transition_uids),
            "scan_proof_kind": self.scan_proof_kind,
        }


@dataclass(frozen=True)
class CanonicalExecutionStage:
    index: int
    component_uids: tuple[str, ...]
    scan_direction: str
    derived_task_uids: tuple[str, ...] = ()
    materialized_source_uids: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "component_uids": list(self.component_uids),
            "scan_direction": self.scan_direction,
            "derived_task_uids": list(self.derived_task_uids),
            "materialized_source_uids": list(self.materialized_source_uids),
        }


@dataclass(frozen=True)
class CanonicalExecutionSchedule:
    """Authority-capable schedule contract over frozen canonical graph facts.

    Multiple ordered stages may have different scan directions.  Direction
    consistency is an SCC/stage-local invariant; a model-global "one direction"
    restriction would incorrectly reject a valid forward stage followed by a
    backward stage.  Ineligible plans remain explicit evidence and the lowerer
    never catches a canonical error and silently substitutes legacy meaning.
    """

    schema: str
    uid: str
    analysis_schema: str
    eligible: bool
    axis: str
    components: tuple[CanonicalExecutionComponent, ...]
    stages: tuple[CanonicalExecutionStage, ...]
    persistent_uids: tuple[str, ...]
    derived_task_uids: tuple[str, ...]
    barrier_uids: tuple[str, ...]
    materialized_source_uids: tuple[str, ...]
    pure_map_uids: tuple[str, ...]
    canonical_domains: tuple[tuple[str, int, int], ...]
    loop_topology_fact_uids: tuple[str, ...]
    recovered_loop_topology_fact_uids: tuple[str, ...]
    capability_blockers: tuple[str, ...] = ()
    provenance: tuple[str, ...] = (
        "canonical_component_schedule_v1",
        "exact_canonical_uid_authority_v1",
    )

    @property
    def effective_scan_direction(self) -> str:
        directions = {
            component.scan_direction for component in self.components
            if component.scan_direction != "any"
        }
        if not directions:
            directions = {
                stage.scan_direction for stage in self.stages
                if stage.scan_direction != "any"
            }
        if not directions:
            return "any"
        if len(directions) != 1:
            return "mixed"
        return next(iter(directions))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "uid": self.uid,
            "analysis_schema": self.analysis_schema,
            "eligible": bool(self.eligible),
            "axis": self.axis,
            "components": [x.manifest() for x in self.components],
            "stages": [x.manifest() for x in self.stages],
            "persistent_uids": list(self.persistent_uids),
            "derived_task_uids": list(self.derived_task_uids),
            "barrier_uids": list(self.barrier_uids),
            "materialized_source_uids": list(self.materialized_source_uids),
            "pure_map_uids": list(self.pure_map_uids),
            "canonical_domains": [
                {"uid": uid, "lo": int(lo), "hi": int(hi)}
                for uid, lo, hi in self.canonical_domains
            ],
            "loop_topology_fact_uids": list(self.loop_topology_fact_uids),
            "recovered_loop_topology_fact_uids": list(self.recovered_loop_topology_fact_uids),
            "effective_scan_direction": self.effective_scan_direction,
            "capability_blockers": list(self.capability_blockers),
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class CanonicalComponentScheduleAnalysis:
    schema: str
    graph_schema: str
    state_footprints: tuple[CanonicalStateFootprintEvidence, ...]
    transition_evidence: tuple[CanonicalTransitionEvidence, ...]
    component_schedule: ComponentSchedulePlan
    validation_notes: tuple[str, ...] = ()
    execution_node_facts: tuple[CanonicalExecutionNodeFact, ...] = ()
    canonical_domains: tuple[tuple[str, int, int], ...] = ()
    pure_map_uids: tuple[str, ...] = ()
    loop_topology_fact_uids: tuple[str, ...] = ()
    recovered_loop_topology_fact_uids: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "graph_schema": self.graph_schema,
            "state_footprints": [x.manifest() for x in self.state_footprints],
            "transition_evidence": [x.manifest() for x in self.transition_evidence],
            "component_schedule": self.component_schedule.manifest(),
            "execution_node_facts": [x.manifest() for x in self.execution_node_facts],
            "canonical_domains": [
                {"uid": uid, "lo": int(lo), "hi": int(hi)}
                for uid, lo, hi in self.canonical_domains
            ],
            "pure_map_uids": list(self.pure_map_uids),
            "loop_topology_fact_uids": list(self.loop_topology_fact_uids),
            "recovered_loop_topology_fact_uids": list(self.recovered_loop_topology_fact_uids),
            "metrics": {
                "footprint_count": len(self.state_footprints),
                "transition_evidence_count": len(self.transition_evidence),
                "unknown_footprint_count": sum(x.availability == "UNKNOWN" for x in self.state_footprints),
                "unknown_transition_count": sum(x.availability == "UNKNOWN" for x in self.transition_evidence),
            },
            "validation_notes": list(self.validation_notes),
        }


def build_canonical_execution_schedule(
    analysis: CanonicalComponentScheduleAnalysis,
) -> CanonicalExecutionSchedule:
    """Lower an observational component proof into a strict authority contract.

    This function consumes only the canonical schedule analysis.  Backend codegen
    details are intentionally *not* re-proved here; unsupported legacy lowering
    still fails normally.  The contract authorizes only the schedule dimensions
    that V02349 can drive without changing the ExecutableGraph ABI.
    """

    plan = analysis.component_schedule
    blockers: list[str] = []
    barriers = [row for row in plan.constraints if row.barrier_required]

    if not plan.persistent_schedule_proven:
        blockers.append("persistent_schedule_not_proven")
    if not plan.full_schedule_proven:
        blockers.append("full_schedule_not_proven")
    if plan.deferred_unknowns:
        blockers.append(f"deferred_unknown_tasks:{len(plan.deferred_unknowns)}")

    component_by_value: dict[str, str] = {}
    for component in plan.components:
        for uid in component.persistent_uids:
            component_by_value[uid] = component.uid

    signed_transition_evidence: dict[str, dict[str, list[str]]] = {
        component.uid: {
            "negative": [], "positive": [], "zero": [], "unresolved": [],
        }
        for component in plan.components
    }
    for transition in analysis.transition_evidence:
        source_component = component_by_value.get(transition.source_uid)
        target_component = component_by_value.get(transition.target_uid)
        if source_component is None or source_component != target_component:
            continue
        requirement = _scan_requirement(transition.min_offset, transition.max_offset)
        bucket = signed_transition_evidence[source_component]
        if requirement == "ascending":
            bucket["negative"].append(transition.uid)
        elif requirement == "descending":
            bucket["positive"].append(transition.uid)
        elif requirement == "any":
            bucket["zero"].append(transition.uid)
        else:
            bucket["unresolved"].append(transition.uid)

    component_rows: list[CanonicalExecutionComponent] = []
    for component in plan.components:
        if component.blockers:
            blockers.append(f"component_blocked:{component.uid}")
        if component.intrinsic_scan in {"mixed", "unknown"}:
            blockers.append(f"component_scan_unproved:{component.uid}:{component.intrinsic_scan}")
        signed = signed_transition_evidence[component.uid]
        evidence_directions = {
            direction
            for direction, rows in (
                ("ascending", signed["negative"]),
                ("descending", signed["positive"]),
            )
            if rows
        }
        if len(evidence_directions) > 1:
            blockers.append(f"component_signed_transition_conflict:{component.uid}")
        elif evidence_directions and component.intrinsic_scan not in {
            "any", next(iter(evidence_directions))
        }:
            blockers.append(
                f"component_scan_evidence_disagreement:{component.uid}:"
                f"{component.intrinsic_scan}:{next(iter(evidence_directions))}"
            )
        if signed["unresolved"] and component.intrinsic_scan not in {"mixed", "unknown"}:
            blockers.append(f"component_signed_transition_unresolved:{component.uid}")
        component_rows.append(CanonicalExecutionComponent(
            uid=component.uid,
            persistent_uids=tuple(component.persistent_uids),
            scan_direction=component.intrinsic_scan,
            domain_uids=tuple(component.domain_uids),
            negative_offset_transition_uids=tuple(sorted(signed["negative"])),
            positive_offset_transition_uids=tuple(sorted(signed["positive"])),
            zero_offset_transition_uids=tuple(sorted(signed["zero"])),
            unresolved_transition_uids=tuple(sorted(signed["unresolved"])),
        ))

    stage_rows: list[CanonicalExecutionStage] = []
    for stage in plan.stages:
        directions = tuple(sorted(set(stage.scan_directions)))
        if len(directions) > 1:
            blockers.append(f"stage_has_multiple_scan_directions:{stage.index}:{','.join(directions)}")
            direction = "mixed"
        elif directions:
            direction = directions[0]
        else:
            direction = "any"
        stage_rows.append(CanonicalExecutionStage(
            index=int(stage.index),
            component_uids=tuple(stage.component_uids),
            scan_direction=direction,
            derived_task_uids=tuple(stage.derived_task_uids),
            materialized_source_uids=tuple(stage.materialized_source_uids),
        ))

    for task in plan.derived_tasks:
        if task.availability == "UNKNOWN" or task.stage_index is None or task.blockers:
            blockers.append(f"derived_task_unproved:{task.uid}")

    # V02347 deliberately permits state-free expressions whose exact backend
    # execution form is not yet admitted.  That is valid for state scheduling but
    # must block *execution schedule authority* (WP is the canonical example).
    for fact in analysis.execution_node_facts:
        if (
            fact.execution_semantic == "unknown"
            and fact.state_semantic == "state_free"
            and fact.canonical_role in {"coordinate", "reduction", "vector"}
        ):
            blockers.append(f"execution_semantic_unapproved:{fact.uid}")

    persistent_uids = tuple(sorted({
        uid for component in component_rows for uid in component.persistent_uids
    }))
    contract_uid = _stable_id(
        "canonical_execution_schedule",
        [
            analysis.schema, plan.schema,
            *(component.uid for component in component_rows),
            *(f"{stage.index}:{stage.scan_direction}" for stage in stage_rows),
            *sorted(set(blockers)),
        ],
    )
    return CanonicalExecutionSchedule(
        schema="modelx_graph.canonical_execution_schedule.v2",
        uid=contract_uid,
        analysis_schema=analysis.schema,
        eligible=not blockers,
        axis=plan.axis,
        components=tuple(component_rows),
        stages=tuple(stage_rows),
        persistent_uids=persistent_uids,
        derived_task_uids=tuple(sorted(task.uid for task in plan.derived_tasks)),
        barrier_uids=tuple(sorted(row.uid for row in barriers)),
        materialized_source_uids=tuple(sorted({
            uid
            for stage in plan.stages for uid in stage.materialized_source_uids
        } | {
            uid
            for task in plan.derived_tasks for uid in task.materialized_source_uids
        })),
        pure_map_uids=tuple(sorted(analysis.pure_map_uids)),
        canonical_domains=tuple(analysis.canonical_domains),
        loop_topology_fact_uids=tuple(analysis.loop_topology_fact_uids),
        recovered_loop_topology_fact_uids=tuple(analysis.recovered_loop_topology_fact_uids),
        capability_blockers=tuple(sorted(set(blockers))),
    )



def analyze_canonical_component_schedule(
    graph: CanonicalSemanticGraph,
) -> CanonicalComponentScheduleAnalysis:
    """Build an exact-UID observational component schedule.

    Graph construction stays total.  Unknown state geometry becomes footprint or
    schedule evidence and only blocks persistent authorization when it lies on a
    material persistent transition path.  StateDerivedMaps are never promoted into
    carried state.
    """

    nodes = {x.uid: x for x in graph.nodes}
    persistent_uids = {
        uid for uid, node in nodes.items() if node.state_semantic == "persistent_state"
    }
    footprints = list(graph.state_footprints)
    footprint_by_uid = {row.node_uid: row for row in footprints}
    transitions = list(graph.transition_evidence)

    # Stage 0 ownership invariant: all formula/source interpretation must have
    # completed while CanonicalSemanticGraph was built.  The scheduler is now a
    # pure graph consumer and deliberately has no fallback that reparses source.
    missing = tuple(sorted(set(nodes) - set(footprint_by_uid)))
    if missing:
        raise ValueError(
            "canonical semantic graph is missing pre-scheduler footprint evidence: "
            + ",".join(missing[:8])
        )

    # Persistent components are SCCs of exact persistent canonical UIDs only.
    recurrence_edges = {(x.source_uid, x.target_uid) for x in transitions}
    persistent_components_raw = _tarjan_source_components(persistent_uids, recurrence_edges)
    component_by_value: dict[str, str] = {}
    component_members: dict[str, tuple[str, ...]] = {}
    for members in persistent_components_raw:
        component_uid = _stable_id("canonical_persistent_component", members)
        component_members[component_uid] = members
        for uid in members:
            component_by_value[uid] = component_uid

    component_transitions: dict[str, list[CanonicalTransitionEvidence]] = {
        uid: [] for uid in component_members
    }
    for row in transitions:
        source_component = component_by_value.get(row.source_uid)
        target_component = component_by_value.get(row.target_uid)
        if source_component is not None and source_component == target_component:
            component_transitions[source_component].append(row)

    components: list[ComponentScheduleComponent] = []
    component_scan: dict[str, str] = {}
    global_blockers: list[str] = []
    for component_uid, members in sorted(component_members.items()):
        requirements: set[str] = set()
        blockers: list[str] = []
        recurrence_uids: list[str] = []
        for row in component_transitions.get(component_uid, ()):
            recurrence_uids.append(row.uid)
            requirement = _scan_requirement(row.min_offset, row.max_offset)
            if requirement == "any":
                continue
            if requirement in {"ascending", "descending"}:
                requirements.add(requirement)
            else:
                blockers.append(
                    f"canonical_persistent_recurrence_scan_{requirement}:{row.source_uid}->{row.target_uid}"
                )
            if row.availability == "UNKNOWN":
                blockers.extend(row.blockers or (f"canonical_transition_unknown:{row.uid}",))
        if len(requirements) > 1:
            scan = "mixed"
            blockers.append("canonical_persistent_component_mixed_scan_direction")
        elif requirements:
            scan = next(iter(requirements))
        elif recurrence_uids:
            scan = "any"
        else:
            scan = "any"
        component_scan[component_uid] = scan
        component = ComponentScheduleComponent(
            uid=component_uid,
            persistent_uids=members,
            intrinsic_scan=scan,
            recurrence_access_uids=tuple(sorted(recurrence_uids)),
            domain_uids=tuple(sorted(uid for uid in members if nodes[uid].domain is not None)),
            blockers=tuple(sorted(set(blockers))),
            note="exact-canonical persistent SCC; StateDerivedMaps are contracted",
        )
        components.append(component)
        global_blockers.extend(component.blockers)

    constraints: list[ComponentScheduleConstraint] = []
    cross_component_order_proofs: list[CrossComponentOrderProof] = []
    graph_edges: set[tuple[str, str]] = set()
    barrier_edges: set[tuple[str, str]] = set()

    cross_groups: dict[tuple[str, str], list[CanonicalTransitionEvidence]] = {}
    for row in transitions:
        source_component = component_by_value.get(row.source_uid)
        target_component = component_by_value.get(row.target_uid)
        if source_component is None or target_component is None or source_component == target_component:
            continue
        cross_groups.setdefault((source_component, target_component), []).append(row)

    for (source_component, target_component), rows in sorted(cross_groups.items()):
        graph_edges.add((source_component, target_component))
        availability = _aggregate_availability(row.availability for row in rows)
        source_scan = component_scan.get(source_component, "unknown")
        target_scan = component_scan.get(target_component, "unknown")
        blockers = sorted({b for row in rows for b in row.blockers})
        order_proof = _build_cross_component_order_proof(
            source_component=source_component,
            target_component=target_component,
            source_scan=source_scan,
            target_scan=target_scan,
            rows=rows,
        )
        cross_component_order_proofs.append(order_proof)
        barrier = bool(order_proof.requires_stage_fence)
        if availability == "UNKNOWN":
            global_blockers.extend(blockers or [
                f"canonical_cross_component_availability_unknown:{source_component}->{target_component}"
            ])
        if order_proof.blockers:
            global_blockers.extend(order_proof.blockers)
        if barrier:
            barrier_edges.add((source_component, target_component))
        # Materialization evidence contains only transitions that individually
        # make a common monotone pass impossible. If incompatibility emerges only
        # from the intersection of several otherwise-compatible transitions, the
        # whole group is forcing evidence.
        forcing_uids = set(order_proof.forcing_evidence_uids)
        barrier_rows = (
            [x for x in rows if x.uid in forcing_uids]
            if barrier and forcing_uids else list(rows)
        )
        constraints.append(ComponentScheduleConstraint(
            uid=_stable_id("canonical_component_constraint", (
                source_component, target_component, availability,
                tuple(sorted(x.uid for x in rows)), order_proof.uid,
            )),
            source_component_uid=source_component,
            target_component_uid=target_component,
            kind="availability",
            availability=availability,
            barrier_required=barrier,
            order_proof_uid=order_proof.uid,
            source_uids=tuple(sorted({x.source_uid for x in barrier_rows})),
            target_uids=tuple(sorted({x.target_uid for x in barrier_rows})),
            evidence_uids=tuple(sorted(x.uid for x in rows)),
            proof_kind="canonical_cross_component_monotone_order_v1",
            blockers=tuple(blockers),
            note=(
                "cross-component monotone order requires full-stage history fence"
                if barrier else "co-schedulable canonical persistent dependency"
            ),
        ))

    component_uids = tuple(sorted(component_members))
    original_barrier_edges = set(barrier_edges)
    (
        stage_by_component,
        barrier_edges,
        stage_scan_order_proofs,
        stage_blockers,
    ) = _solve_stage_scan_constraints(
        component_uids=component_uids,
        component_scan=component_scan,
        graph_edges=graph_edges,
        barrier_edges=barrier_edges,
        cross_component_order_proofs=cross_component_order_proofs,
    )
    global_blockers.extend(stage_blockers)

    promoted_partition_edges = set(barrier_edges) - original_barrier_edges
    if promoted_partition_edges:
        constraints = [
            replace(
                constraint,
                kind="stage_scan_partition",
                barrier_required=True,
                proof_kind="canonical_stage_global_monotone_order_v1",
                note=(
                    "pairwise dependency is co-scannable in isolation but crosses a "
                    "stage boundary required by the global scan-order proof"
                ),
            )
            if (
                constraint.source_component_uid,
                constraint.target_component_uid,
            ) in promoted_partition_edges
            else constraint
            for constraint in constraints
        ]

    base_adjacency: dict[str, set[str]] = {uid: set() for uid in component_uids}
    for source_component, target_component in graph_edges:
        base_adjacency.setdefault(source_component, set()).add(target_component)
    for source_component, target_component in sorted(barrier_edges):
        reduced = {uid: set(children) for uid, children in base_adjacency.items()}
        reduced.setdefault(source_component, set()).discard(target_component)
        if _graph_path_exists(reduced, target_component, source_component):
            global_blockers.append(
                f"canonical_component_barrier_reverse_path:{source_component}->{target_component}"
            )

    # Exact StateDerivedMaps are tasks, not carried-state members.  A coordinate
    # helper can have an UNKNOWN *standalone* frame while every use site is fully
    # proven after path composition (for example an annual helper called from a
    # monthly projection).  Such a helper is contextual: it executes on demand in
    # its consumers' proven stage and never invents a cross-frame numeric offset.
    derived_uids = {
        uid for uid, node in nodes.items() if node.execution_semantic == "derived_task"
    }
    consumers_by_source: dict[str, set[str]] = {uid: set() for uid in nodes}
    for access in graph.accesses:
        if access.scheduling_relevant:
            consumers_by_source.setdefault(access.source_uid, set()).add(access.target_uid)

    base_stage_by_derived: dict[str, int] = {}
    base_blockers: dict[str, tuple[str, ...]] = {}
    for uid in sorted(derived_uids):
        footprint = footprint_by_uid[uid]
        availability = footprint.availability
        blockers = tuple(sorted(set(footprint.blockers) | {
            b for path in footprint.paths for b in path.blockers
        }))
        base_blockers[uid] = blockers
        if availability != "UNKNOWN":
            producers = {
                component_by_value[source_uid]
                for source_uid in footprint.persistent_source_uids
                if source_uid in component_by_value
            }
            if producers:
                stage = max(stage_by_component.get(x, 0) for x in producers)
                stage += 1 if availability in {"COMPLETE", "SUFFIX"} else 0
            else:
                stage = 0
            base_stage_by_derived[uid] = stage

    contextual_stage: dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for uid in sorted(derived_uids):
            if uid in base_stage_by_derived or uid in contextual_stage:
                continue
            blockers = base_blockers[uid]
            if not blockers or any(not _is_geometry_blocker(b) for b in blockers):
                continue
            consumers = consumers_by_source.get(uid, set())
            if not consumers:
                continue
            consumer_stages: list[int] = []
            resolvable = True
            for consumer_uid in consumers:
                if consumer_uid in component_by_value:
                    component_uid = component_by_value[consumer_uid]
                    consumer_stages.append(stage_by_component.get(component_uid, 0))
                elif consumer_uid in base_stage_by_derived:
                    consumer_stages.append(base_stage_by_derived[consumer_uid])
                elif consumer_uid in contextual_stage:
                    consumer_stages.append(contextual_stage[consumer_uid])
                else:
                    resolvable = False
                    break
            if resolvable and consumer_stages:
                contextual_stage[uid] = min(consumer_stages)
                changed = True

    derived_tasks: list[DerivedScheduleTask] = []
    deferred_unknowns: list[str] = []
    for uid, node in sorted(nodes.items()):
        if node.execution_semantic != "derived_task":
            continue
        footprint = footprint_by_uid[uid]
        contextual = uid in contextual_stage
        availability = "CONTEXTUAL" if contextual else footprint.availability
        producers = tuple(sorted({
            component_by_value[source_uid]
            for source_uid in footprint.persistent_source_uids
            if source_uid in component_by_value
        }))
        task_blockers = () if contextual else base_blockers[uid]
        if contextual:
            stage_index = contextual_stage[uid]
        elif availability == "UNKNOWN":
            stage_index = None
            deferred_unknowns.append(f"canonical_derived_task_unknown:{uid}")
        elif uid in base_stage_by_derived:
            stage_index = base_stage_by_derived[uid]
        else:
            stage_index = 0
        materialized = (
            footprint.persistent_source_uids
            if availability in {"COMPLETE", "SUFFIX"}
            else ()
        )
        derived_tasks.append(DerivedScheduleTask(
            uid=_stable_id("canonical_derived_task", (uid, availability)),
            target_uid=uid,
            availability=availability,
            producer_component_uids=producers,
            stage_index=stage_index,
            persistent_source_uids=footprint.persistent_source_uids,
            evidence_uids=tuple(path.uid for path in footprint.paths),
            materialized_source_uids=materialized,
            blockers=tuple(task_blockers),
            note=(
                "contextual coordinate helper; standalone frame is intentionally not compared"
                if contextual else
                "exact canonical completed-history StateDerivedMap"
                if availability in {"COMPLETE", "SUFFIX"}
                else "exact canonical state-derived task"
            ),
        ))

    materialized_by_stage: dict[int, set[str]] = {}
    for constraint in constraints:
        if not constraint.barrier_required:
            continue
        source_stage = stage_by_component.get(constraint.source_component_uid, 0)
        materialized_by_stage.setdefault(source_stage, set()).update(constraint.source_uids)
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
    scan_proof_by_stage = {proof.stage_index: proof for proof in stage_scan_order_proofs}
    complete_stage_scan_proofs = list(stage_scan_order_proofs)
    stages: list[ComponentScheduleStage] = []
    for index in sorted(stage_indices or {0}):
        members = tuple(sorted(
            uid for uid, stage in stage_by_component.items() if stage == index
        ))
        stage_scan_proof = scan_proof_by_stage.get(index)
        if stage_scan_proof is not None and len(stage_scan_proof.final_scan_options) == 1:
            directions = (stage_scan_proof.final_scan_options[0],)
        elif stage_scan_proof is not None and len(stage_scan_proof.final_scan_options) == 2:
            directions = ()
        else:
            directions = tuple(sorted({
                component_scan[uid]
                for uid in members
                if component_scan.get(uid) not in {None, "any"}
            }))
        tasks = tuple(sorted(task.uid for task in derived_tasks if task.stage_index == index))
        if stage_scan_proof is None:
            stage_scan_proof = StageScanOrderProof(
                uid=_stable_id("canonical_stage_scan_order", (index, members, "derived_only")),
                stage_index=int(index),
                component_uids=members,
                dependency_order_proof_uids=(),
                initial_scan_options=tuple(sorted(_MONOTONE_SCANS)),
                final_scan_options=tuple(sorted(_MONOTONE_SCANS)),
                selected_scan_direction=None,
                proof_kind="canonical_stage_global_monotone_order_v1",
            )
            complete_stage_scan_proofs.append(stage_scan_proof)
        stages.append(ComponentScheduleStage(
            index=index,
            component_uids=members,
            scan_directions=directions,
            derived_task_uids=tasks,
            materialized_source_uids=tuple(sorted(materialized_by_stage.get(index, ()))),
            note="weighted-SCC stage over exact canonical persistent components",
        ))

    blockers = tuple(sorted(set(global_blockers)))
    deferred = tuple(sorted(set(deferred_unknowns)))
    persistent_proven = not blockers
    full_proven = persistent_proven and not deferred and all(
        not task.blockers for task in derived_tasks if task.availability != "UNKNOWN"
    )
    schedule = ComponentSchedulePlan(
        schema="modelx_graph.canonical_component_schedule.v3",
        axis="canonical",
        components=tuple(components),
        constraints=tuple(sorted(constraints, key=lambda x: x.uid)),
        derived_tasks=tuple(derived_tasks),
        stages=tuple(stages),
        persistent_schedule_proven=persistent_proven,
        full_schedule_proven=full_proven,
        cross_component_order_proofs=tuple(sorted(
            cross_component_order_proofs, key=lambda x: x.uid
        )),
        stage_scan_order_proofs=tuple(sorted(
            complete_stage_scan_proofs, key=lambda x: (x.stage_index, x.uid)
        )),
        blockers=blockers,
        deferred_unknowns=deferred,
        validation_notes=(
            "backend-independent canonical proof; execution authority is granted only by CanonicalExecutionSchedule capability checks",
            "exact specialized canonical UIDs are the only scheduling identities",
            "state-free values contribute no persistent causality even when backend execution is unapproved",
            "StateDerivedMaps are expanded to persistent leaves and remain derived tasks",
            "cross-frame coordinate helpers may use consumer-context placement only when all use sites have proven stages",
            "cross-component stage fences are derived from exact monotone co-scan feasibility rather than availability labels or product rules",
            "stage-global scan feasibility closes pairwise order proofs before physical authority is granted",
            "weighted-SCC assignment remains the stage-order primitive; globally incompatible zero-fence edges may become typed partition fences",
        ),
    )

    return CanonicalComponentScheduleAnalysis(
        schema="modelx_graph.canonical_component_schedule_analysis.v3",
        graph_schema=graph.schema,
        state_footprints=tuple(footprints),
        transition_evidence=tuple(sorted(transitions, key=lambda x: x.uid)),
        component_schedule=schedule,
        validation_notes=(
            "canonical graph construction remains total; UNKNOWN becomes schedule evidence",
            "no product, Cell, or loop-variable names participate in scheduling rules",
        ),
        execution_node_facts=tuple(
            CanonicalExecutionNodeFact(
                uid=node.uid,
                canonical_role=node.canonical_role,
                state_semantic=node.state_semantic,
                execution_semantic=node.execution_semantic,
                domain=node.domain,
                semantic_blockers=tuple(node.semantic_blockers),
            )
            for node in sorted(graph.nodes, key=lambda x: x.uid)
        ),
        canonical_domains=tuple(sorted(
            (node.uid, int(node.domain[0]), int(node.domain[1]))
            for node in graph.nodes if node.domain is not None
        )),
        pure_map_uids=tuple(sorted(
            node.uid for node in graph.nodes if node.execution_semantic == "pure_map"
        )),
        loop_topology_fact_uids=tuple(sorted(x.uid for x in graph.loop_topology_facts)),
        recovered_loop_topology_fact_uids=tuple(sorted(x.uid for x in graph.recovered_loop_topology_facts)),
    )
