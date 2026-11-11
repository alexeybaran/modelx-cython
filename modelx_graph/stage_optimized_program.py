from __future__ import annotations

"""Stage-aware backend-neutral optimized program.

Semantic placement belongs to :class:`StageExecutionPlan`, lifetimes belong to
:class:`StageStoragePlan`, and this object owns the frozen backend/ABI contract.
Mature ``ExecutableGraph``/``OptimizedProgram`` objects are optional comparison
evidence only and are never required by direct Stage emission.
"""

import ast
import base64
import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .canonical_semantic_graph import CanonicalSemanticGraph
from .normalized_operator import NormalizedOperator
from .invocation import OutputInvocation, ResultDomain
from .program_inputs import OptimizedInputSpec
from .program_semantics import GuardSpec, ValidatedStaticFact
from .stage_physical_lowering import (
    StageAuxiliaryRecurrenceABI, StageBoundarySeed, StageDirectContract,
    StageExecutionBlock, StageIterationDomain, StagePureMapABI,
    build_one_stage_direct_contract, build_stage_direct_contract,
    build_zero_barrier_direct_contract,
)
from .stage_execution_plan import StageExecutionPlan
from .stage_storage_plan import (
    StageStoragePlan, StageStorageValue, _storage_value_identity,
)


class StageOptimizedProgramError(RuntimeError):
    pass


def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class StagePhysicalValue:
    value_uid: str
    working_kind: str
    current_slot: int | None = None
    ring_base: int | None = None
    ring_depth: int | None = None
    history_slot: int | None = None
    materialized_slot: int | None = None
    scalar_slot: int | None = None
    working_scalar_slot: int | None = None
    retained_bounds: tuple[int | None, int | None] | None = None
    retained_offset_min: int | None = None
    retained_offset_max: int | None = None
    producer_stage_index: int | None = None
    consumer_stage_indices: tuple[int, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "value_uid": self.value_uid,
            "working_kind": self.working_kind,
            "current_slot": self.current_slot,
            "ring_base": self.ring_base,
            "ring_depth": self.ring_depth,
            "history_slot": self.history_slot,
            "materialized_slot": self.materialized_slot,
            "scalar_slot": self.scalar_slot,
            "working_scalar_slot": self.working_scalar_slot,
            "retained_bounds": None if self.retained_bounds is None else {
                "lo": self.retained_bounds[0], "hi": self.retained_bounds[1]
            },
            "retained_offset_min": self.retained_offset_min,
            "retained_offset_max": self.retained_offset_max,
            "producer_stage_index": self.producer_stage_index,
            "consumer_stage_indices": list(self.consumer_stage_indices),
        }


@dataclass(frozen=True)
class StageProgramKernel:
    uid: str
    stage_index: int
    task_uid: str
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
            "stage_index": self.stage_index,
            "task_uid": self.task_uid,
            "kind": self.kind,
            "component_uid": self.component_uid,
            "canonical_uids": list(self.canonical_uids),
            "domain_uids": list(self.domain_uids),
            "scan_direction": self.scan_direction,
            "reduction_range_asts": list(self.reduction_range_asts),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class StageProgramStage:
    index: int
    kernel_uids: tuple[str, ...]
    materialized_value_uids: tuple[str, ...]
    stage_carried_scalar_uids: tuple[str, ...]
    incoming_barrier_uids: tuple[str, ...]
    outgoing_barrier_uids: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kernel_uids": list(self.kernel_uids),
            "materialized_value_uids": list(self.materialized_value_uids),
            "stage_carried_scalar_uids": list(self.stage_carried_scalar_uids),
            "incoming_barrier_uids": list(self.incoming_barrier_uids),
            "outgoing_barrier_uids": list(self.outgoing_barrier_uids),
        }


@dataclass(frozen=True)
class StageProgramBarrier:
    uid: str
    source_component_uid: str
    target_component_uid: str
    source_stage_index: int | None
    target_stage_index: int | None
    availability: str
    source_uids: tuple[str, ...]
    target_uids: tuple[str, ...]
    proof_kind: str | None
    evidence_uids: tuple[str, ...]
    storage_fence_uid: str
    execution_mode: str
    execution_mode_proof_kind: str | None
    required_materialized_value_uids: tuple[str, ...]
    required_materialized_slots: tuple[int, ...]
    backend_support: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.blockers and bool(self.backend_support)

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
            "proof_kind": self.proof_kind,
            "evidence_uids": list(self.evidence_uids),
            "storage_fence_uid": self.storage_fence_uid,
            "execution_mode": self.execution_mode,
            "execution_mode_proof_kind": self.execution_mode_proof_kind,
            "required_materialized_value_uids": list(self.required_materialized_value_uids),
            "required_materialized_slots": list(self.required_materialized_slots),
            "backend_support": list(self.backend_support),
            "blockers": list(self.blockers),
            "supported": self.supported,
        }


@dataclass(frozen=True)
class StageOperatorRequirement:
    uid: str
    kind: str  # external_state_free_value | unresolved_external_call
    state_semantic: str
    backend_support: tuple[str, ...]
    blocker: str | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "kind": self.kind,
            "state_semantic": self.state_semantic,
            "backend_support": list(self.backend_support),
            "blocker": self.blocker,
        }


@dataclass(frozen=True)
class StageRunDomain:
    uid: str
    point_count: int
    run_key_tokens: tuple[str, ...]
    proof_kind: str = "finite_native_batch_run_domain_v1"
    semantic_run_domain_uid: str | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "point_count": int(self.point_count),
            "run_key_tokens": list(self.run_key_tokens),
            "proof_kind": self.proof_kind,
            "semantic_run_domain_uid": self.semantic_run_domain_uid,
        }


@dataclass(frozen=True)
class StageRuntimePayload:
    program_uid: str
    variants: Mapping[str, Any]
    graph: CanonicalSemanticGraph
    execution_plan: StageExecutionPlan
    storage_plan: StageStoragePlan
    physical_values: tuple[StagePhysicalValue, ...]


@dataclass(frozen=True)
class StageOptimizedProgram:
    schema: str
    uid: str
    output_uid: str
    run_domain: StageRunDomain
    execution_plan: StageExecutionPlan
    storage_plan: StageStoragePlan
    graph: CanonicalSemanticGraph = field(compare=False, repr=False)
    variants: Mapping[str, Any] = field(compare=False, repr=False)
    output_invocation: OutputInvocation | None = None
    result_domain: ResultDomain | None = None
    stages: tuple[StageProgramStage, ...] = ()
    kernels: tuple[StageProgramKernel, ...] = ()
    physical_values: tuple[StagePhysicalValue, ...] = ()
    barriers: tuple[StageProgramBarrier, ...] = ()
    operators: tuple[StageOperatorRequirement, ...] = ()
    normalized_operators: tuple[NormalizedOperator, ...] = ()
    input_order: tuple[str, ...] = ()
    inputs: tuple[OptimizedInputSpec, ...] = ()
    guards: tuple[GuardSpec, ...] = ()
    validated_static_facts: tuple[ValidatedStaticFact, ...] = ()
    iteration_domains: tuple[StageIterationDomain, ...] = ()
    execution_blocks: tuple[StageExecutionBlock, ...] = ()
    boundary_seeds: tuple[StageBoundarySeed, ...] = ()
    pure_maps: tuple[StagePureMapABI, ...] = ()
    auxiliary_recurrences: tuple[StageAuxiliaryRecurrenceABI, ...] = ()
    direct_python_supported: bool = False
    direct_cython_supported: bool = False
    direct_python_blockers: tuple[str, ...] = ()
    direct_cython_blockers: tuple[str, ...] = ()
    python_supported: bool = False
    cython_supported: bool = False
    capability_blockers: tuple[str, ...] = ()
    legacy_program: Any | None = field(default=None, compare=False, repr=False)
    validation_notes: tuple[str, ...] = ()

    def physical_value(self, uid: str) -> StagePhysicalValue:
        return next(x for x in self.physical_values if x.value_uid == uid)

    @property
    def stage_indices(self) -> tuple[int, ...]:
        return tuple(x.index for x in self.stages)

    @property
    def materialized_value_uids(self) -> tuple[str, ...]:
        return tuple(sorted(
            x.value_uid for x in self.physical_values if x.materialized_slot is not None
        ))

    def input_spec(self, key: str) -> OptimizedInputSpec:
        for spec in self.inputs:
            if spec.key == key:
                return spec
        raise StageOptimizedProgramError(f"unknown Stage input key {key!r}")

    def build_fingerprint(
        self,
        *,
        backend: str,
        optimization_level: str = "O2",
        emission_mode: str = "array_abi",
        c_optimization: str = "O3",
        native_arch: bool = True,
    ) -> str:
        """Stable production build key owned exclusively by canonical Stage IR."""
        return hashlib.sha256("|".join((
            self.schema, self.uid, self.graph.formula_hash, self.run_domain.uid, self.execution_plan.uid,
            self.storage_plan.uid, str(backend).lower(), str(optimization_level).upper(),
            str(emission_mode).lower(), str(c_optimization).upper(),
            str(int(bool(native_arch))),
            f"output_invocation:{self.output_invocation.uid if self.output_invocation is not None else '<none>'}",
            f"result_domain:{self.result_domain.uid if self.result_domain is not None else '<none>'}",
            *(f"{x.key}:{x.dtype}:{x.ndim}:{x.scope}:{x.expected_shape}:{x.domain_token}:{x.enum_labels}" for x in self.inputs),
            *(f"{x.uid}:{x.kind}:{x.result_dtype}:{x.result_shape}:{x.static_inputs}:{x.dynamic_inputs}" for x in self.normalized_operators),
        )).encode("utf-8", "backslashreplace")).hexdigest()

    def runtime_payload(self) -> StageRuntimePayload:
        return StageRuntimePayload(
            program_uid=self.uid,
            variants=self.variants,
            graph=self.graph,
            execution_plan=self.execution_plan,
            storage_plan=self.storage_plan,
            physical_values=self.physical_values,
        )

    def encoded_runtime_payload(self) -> str:
        raw = pickle.dumps(self.runtime_payload(), protocol=5)
        return base64.b85encode(raw).decode("ascii")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "uid": self.uid,
            "output_uid": self.output_uid,
            "run_domain": self.run_domain.manifest(),
            "output_invocation": (
                self.output_invocation.manifest()
                if self.output_invocation is not None else None
            ),
            "result_domain": (
                self.result_domain.manifest()
                if self.result_domain is not None else None
            ),
            "execution_plan_uid": self.execution_plan.uid,
            "storage_plan_uid": self.storage_plan.uid,
            "stages": [x.manifest() for x in self.stages],
            "kernels": [x.manifest() for x in self.kernels],
            "physical_values": [x.manifest() for x in self.physical_values],
            "barriers": [x.manifest() for x in self.barriers],
            "operators": [x.manifest() for x in self.operators],
            "normalized_operators": [x.manifest() for x in self.normalized_operators],
            "input_order": list(self.input_order),
            "inputs": {x.key: x.manifest() for x in self.inputs},
            "guards": [x.manifest() for x in self.guards],
            "validated_static_facts": [x.manifest() for x in self.validated_static_facts],
            "iteration_domains": [x.manifest() for x in self.iteration_domains],
            "execution_blocks": [x.manifest() for x in self.execution_blocks],
            "boundary_seeds": [x.manifest() for x in self.boundary_seeds],
            "pure_maps": [x.manifest() for x in self.pure_maps],
            "auxiliary_recurrences": [x.manifest() for x in self.auxiliary_recurrences],
            "direct_python_supported": self.direct_python_supported,
            "direct_cython_supported": self.direct_cython_supported,
            "direct_python_blockers": list(self.direct_python_blockers),
            "direct_cython_blockers": list(self.direct_cython_blockers),
            "python_supported": self.python_supported,
            "cython_supported": self.cython_supported,
            "capability_blockers": list(self.capability_blockers),
            "metrics": {
                "stage_count": len(self.stages),
                "kernel_count": len(self.kernels),
                "physical_value_count": len(self.physical_values),
                "materialized_value_count": len(self.materialized_value_uids),
                "barrier_count": len(self.barriers),
                "operator_requirement_count": len(self.operators),
                "normalized_operator_count": len(self.normalized_operators),
                "execution_block_count": len(self.execution_blocks),
                "boundary_seed_count": len(self.boundary_seeds),
                "pure_map_abi_count": len(self.pure_maps),
                "auxiliary_recurrence_count": len(self.auxiliary_recurrences),
            },
            "validation_notes": list(self.validation_notes),
        }


def _physical_values(storage: StageStoragePlan) -> tuple[StagePhysicalValue, ...]:
    current_slot = 0
    ring_base = 0
    history_slot = 0
    materialized_slot = 0
    scalar_slot = 0
    working_scalar_slot = 0
    rows: list[StagePhysicalValue] = []
    for row in sorted(storage.values, key=lambda x: x.value_uid):
        cur = ring = hist = mat = scalar = working_scalar = None
        if row.working_storage_kind == "current":
            cur = current_slot
            current_slot += 1
        elif row.working_storage_kind == "ring":
            ring = ring_base
            depth = int(row.ring_depth or 0)
            if depth <= 0:
                raise StageOptimizedProgramError(
                    f"ring storage has invalid depth for {row.value_uid}: {row.ring_depth}"
                )
            ring_base += depth
        elif row.working_storage_kind == "history":
            hist = history_slot
            history_slot += 1
        elif row.working_storage_kind == "scalar":
            working_scalar = working_scalar_slot
            working_scalar_slot += 1
        elif row.working_storage_kind in {"none", "tensor"}:
            pass
        else:
            raise StageOptimizedProgramError(
                f"unknown StageStoragePlan working storage kind {row.working_storage_kind!r}"
            )

        if row.cross_stage_storage_kind == "materialized_history":
            mat = materialized_slot
            materialized_slot += 1
        elif row.cross_stage_storage_kind == "stage_carried_scalar":
            scalar = scalar_slot
            scalar_slot += 1
        elif row.cross_stage_storage_kind in {"none", "recompute"}:
            pass
        else:
            raise StageOptimizedProgramError(
                f"unknown StageStoragePlan cross-stage kind {row.cross_stage_storage_kind!r}"
            )

        rows.append(StagePhysicalValue(
            value_uid=row.value_uid,
            working_kind=row.working_storage_kind,
            current_slot=cur,
            ring_base=ring,
            ring_depth=row.ring_depth if ring is not None else None,
            history_slot=hist,
            materialized_slot=mat,
            scalar_slot=scalar,
            working_scalar_slot=working_scalar,
            retained_bounds=row.retained_bounds,
            retained_offset_min=row.retained_offset_min,
            retained_offset_max=row.retained_offset_max,
            producer_stage_index=row.producer_stage_index,
            consumer_stage_indices=tuple(row.consumer_stage_indices),
        ))
    return tuple(rows)


def _program_barriers(
    execution_plan: StageExecutionPlan,
    storage_plan: StageStoragePlan,
    physical: tuple[StagePhysicalValue, ...],
) -> tuple[StageProgramBarrier, ...]:
    """Map storage-owned fence contracts to physical slots.

    StageStoragePlan is the sole authority for *which* materialized histories and
    retained bounds satisfy a semantic barrier.  This layer verifies that frozen
    contract and assigns physical slots only; it never scans unrelated storage
    values to reconstruct fence requirements.
    """
    physical_by_uid = {x.value_uid: x for x in physical}
    storage_by_uid = {x.value_uid: x for x in storage_plan.values}
    stage_by_index = {int(x.index): x for x in storage_plan.stages}

    fences_by_barrier = {}
    for fence in storage_plan.fences:
        prior = fences_by_barrier.get(fence.semantic_barrier_uid)
        if prior is not None:
            raise StageOptimizedProgramError(
                f"duplicate StageStorageFence for semantic barrier {fence.semantic_barrier_uid}"
            )
        fences_by_barrier[fence.semantic_barrier_uid] = fence
    expected = {row.uid for row in execution_plan.barriers}
    actual = set(fences_by_barrier)
    if expected != actual:
        raise StageOptimizedProgramError(
            "StageStorageFence coverage mismatch; "
            f"missing={sorted(expected-actual)!r} extra={sorted(actual-expected)!r}"
        )

    rows: list[StageProgramBarrier] = []
    for barrier in sorted(execution_plan.barriers, key=lambda x: x.uid):
        fence = fences_by_barrier[barrier.uid]
        semantic_identity = (
            barrier.source_component_uid, barrier.target_component_uid,
            barrier.source_stage_index, barrier.target_stage_index,
            barrier.availability, barrier.proof_kind, tuple(barrier.evidence_uids),
        )
        storage_identity = (
            fence.source_component_uid, fence.target_component_uid,
            fence.source_stage_index, fence.target_stage_index,
            fence.availability, fence.semantic_proof_kind,
            tuple(fence.semantic_evidence_uids),
        )
        if semantic_identity != storage_identity:
            raise StageOptimizedProgramError(
                f"StageStorageFence semantic identity mismatch for {barrier.uid}"
            )

        blockers: set[str] = set(barrier.blockers) | set(fence.blockers)
        src = barrier.source_stage_index
        dst = barrier.target_stage_index
        if src is None or dst is None:
            blockers.add("barrier_stage_unproved")
        else:
            src = int(src); dst = int(dst)
            if fence.execution_mode != "full_stage_materialization":
                blockers.add(
                    f"barrier_execution_mode_unsupported:{fence.execution_mode}"
                )
            if src >= dst:
                blockers.add(f"barrier_stage_order_unsupported:{src}:{dst}")
            src_stage = stage_by_index.get(src)
            dst_stage = stage_by_index.get(dst)
            if src_stage is None or barrier.uid not in src_stage.outgoing_barrier_uids:
                blockers.add(f"barrier_source_stage_link_missing:{src}")
            if dst_stage is None or barrier.uid not in dst_stage.incoming_barrier_uids:
                blockers.add(f"barrier_target_stage_link_missing:{dst}")

        required_uids = fence.required_value_uids
        required_slots: list[int] = []
        for contract in fence.required_values:
            value = storage_by_uid.get(contract.value_uid)
            if value is None:
                raise StageOptimizedProgramError(
                    f"StageStorageFence {fence.uid} references missing storage value {contract.value_uid}"
                )
            identity = _storage_value_identity(execution_plan.uid, value)
            if identity != contract.storage_identity_uid:
                raise StageOptimizedProgramError(
                    f"StageStorageFence value identity mismatch for {fence.uid}:{contract.value_uid}"
                )
            if (
                value.producer_stage_index != contract.producer_stage_index
                or tuple(value.consumer_stage_indices) != tuple(contract.consumer_stage_indices)
                or value.materialize_after_stage != contract.materialize_after_stage
                or value.cross_stage_storage_kind != contract.cross_stage_storage_kind
                or value.retained_bounds != contract.retained_bounds
            ):
                raise StageOptimizedProgramError(
                    f"StageStorageFence storage geometry mismatch for {fence.uid}:{contract.value_uid}"
                )
            physical_row = physical_by_uid.get(contract.value_uid)
            if physical_row is None or physical_row.materialized_slot is None:
                blockers.add(f"barrier_materialized_slot_missing:{contract.value_uid}")
                continue
            required_slots.append(int(physical_row.materialized_slot))
        required_slots_tuple = tuple(required_slots)
        if len(required_slots_tuple) != len(required_uids):
            blockers.add("barrier_materialized_slot_coverage_incomplete")

        support = ("python", "cython") if not blockers else ()
        rows.append(StageProgramBarrier(
            uid=barrier.uid,
            source_component_uid=barrier.source_component_uid,
            target_component_uid=barrier.target_component_uid,
            source_stage_index=barrier.source_stage_index,
            target_stage_index=barrier.target_stage_index,
            availability=barrier.availability,
            source_uids=tuple(barrier.source_uids),
            target_uids=tuple(barrier.target_uids),
            proof_kind=barrier.proof_kind,
            evidence_uids=tuple(barrier.evidence_uids),
            storage_fence_uid=fence.uid,
            execution_mode=fence.execution_mode,
            execution_mode_proof_kind=fence.execution_mode_proof_kind,
            required_materialized_value_uids=tuple(required_uids),
            required_materialized_slots=required_slots_tuple,
            backend_support=support,
            blockers=tuple(sorted(blockers)),
        ))
    return tuple(rows)


def _operator_requirements(
    graph: CanonicalSemanticGraph,
    execution_plan: StageExecutionPlan,
    variants: Mapping[str, Any],
    normalized_operators: tuple[NormalizedOperator, ...] = (),
) -> tuple[StageOperatorRequirement, ...]:
    nodes = {x.uid: x for x in graph.nodes}
    rows: list[StageOperatorRequirement] = []
    approved = {op.uid for op in normalized_operators if op.python_supported or op.cython_supported}
    for blocker in execution_plan.capability_blockers:
        if not blocker.startswith("execution_semantic_unapproved:"):
            continue
        uid = blocker.split(":", 1)[1]
        if uid in approved:
            continue
        node = nodes.get(uid)
        if node is None:
            raise StageOptimizedProgramError(f"operator blocker references unknown UID {uid}")
        rows.append(StageOperatorRequirement(
            uid=uid,
            kind="external_state_free_value",
            state_semantic=node.state_semantic,
            backend_support=("python_external",),
            blocker=blocker,
        ))

    known = set(variants)
    builtins = {
        "abs", "bool", "enumerate", "float", "int", "len", "list", "max",
        "min", "range", "round", "sum", "tuple", "zip",
        "array_input", "table_input", "global_input", "point_input",
        "__is_missing__", "__is_inf__", "__erf__", "__step_lookup_1d__",
        "__interval_lookup_1d__", "__interp_lookup_1d__", "__guard_fail__",
        "__exact_axis_code__", "__sparse_table_lookup__", "__prepared_point_lookup__",
    }
    external: set[str] = set()
    unresolved_attributes: set[str] = set()
    unresolved_static_attributes: set[str] = set()
    unresolved_space_selections: set[str] = set()
    runtime_module_roots = {"math", "np", "numpy", "pd", "pandas"}
    for cv in variants.values():
        roots: list[ast.AST] = []
        if getattr(cv, "function", None) is not None:
            roots.append(cv.function)
        reduction = getattr(cv, "reduction", None)
        if reduction is not None:
            roots.extend([reduction.init, reduction.body_expr, *reduction.filters, *reduction.range_args])
        for root in roots:
            local_names: set[str] = set()
            if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local_names.update(arg.arg for arg in root.args.posonlyargs)
                local_names.update(arg.arg for arg in root.args.args)
                local_names.update(arg.arg for arg in root.args.kwonlyargs)
                if root.args.vararg is not None:
                    local_names.add(root.args.vararg.arg)
                if root.args.kwarg is not None:
                    local_names.add(root.args.kwarg.arg)
            local_names.update(
                node.id for node in ast.walk(root)
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param))
            )
            parents: dict[int, ast.AST] = {}
            for parent in ast.walk(root):
                for child in ast.iter_child_nodes(parent):
                    parents[id(child)] = parent
            for node in ast.walk(root):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                        if name in known or name in builtins:
                            continue
                        external.add(name)
                        continue
                    if (
                        isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                    ):
                        root_name = node.func.value.id
                        if root_name in local_names or root_name in runtime_module_roots:
                            continue
                        unresolved_attributes.add(f"{root_name}.{node.func.attr}")
                        continue
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                    root_name = node.value.id
                    if (
                        root_name not in local_names
                        and root_name not in runtime_module_roots
                        and root_name not in known
                        and root_name not in builtins
                    ):
                        unresolved_space_selections.add(root_name)
                    continue
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    root_name = node.value.id
                    if root_name in local_names or root_name in runtime_module_roots:
                        continue
                    parent = parents.get(id(node))
                    if isinstance(parent, ast.Call) and parent.func is node:
                        continue
                    unresolved_static_attributes.add(f"{root_name}.{node.attr}")
    for name in sorted(external):
        rows.append(StageOperatorRequirement(
            uid=name,
            kind="unresolved_external_call",
            state_semantic="external",
            backend_support=("python_external",),
            blocker=f"external_call_requires_binding:{name}",
        ))
    for name in sorted(unresolved_attributes):
        rows.append(StageOperatorRequirement(
            uid=name,
            kind="unresolved_model_bound_accessor",
            state_semantic="external",
            backend_support=(),
            blocker=f"direct_unresolved_model_bound_accessor:{name}",
        ))
    for name in sorted(unresolved_static_attributes):
        rows.append(StageOperatorRequirement(
            uid=name,
            kind="unresolved_model_bound_static_attribute",
            state_semantic="external",
            backend_support=(),
            blocker=f"direct_unresolved_model_bound_static_attribute:{name}",
        ))
    for name in sorted(unresolved_space_selections):
        rows.append(StageOperatorRequirement(
            uid=name,
            kind="unresolved_model_bound_space_selection",
            state_semantic="external",
            backend_support=(),
            blocker=f"direct_unresolved_model_bound_space_selection:{name}",
        ))
    return tuple(sorted(rows, key=lambda x: (x.kind, x.uid)))


def build_stage_optimized_program(
    graph: CanonicalSemanticGraph,
    execution_plan: StageExecutionPlan,
    storage_plan: StageStoragePlan,
    *,
    variants: Mapping[str, Any],
    output_uid: str | None = None,
    legacy_program: Any | None = None,
    normalized_operators: tuple[NormalizedOperator, ...] = (),
    input_order: tuple[str, ...] = (),
    inputs: tuple[OptimizedInputSpec, ...] = (),
    guards: tuple[GuardSpec, ...] = (),
    validated_static_facts: tuple[ValidatedStaticFact, ...] = (),
    proof_input_rows: tuple[Mapping[str, Any], ...] = (),
    run_domain: StageRunDomain | None = None,
    output_invocation: OutputInvocation | None = None,
    result_domain: ResultDomain | None = None,
) -> StageOptimizedProgram:
    """Lower canonical stage/storage semantics into one backend-neutral program.

    Construction is total over sound StageStoragePlan shapes. Unsupported backend
    features remain capability blockers. A legacy OptimizedProgram may be attached
    only as a one-stage parity/emission adapter; it does not supply stage/storage
    semantics to this program.
    """
    if storage_plan.execution_plan_uid != execution_plan.uid:
        raise StageOptimizedProgramError("StageStoragePlan belongs to a different execution plan")
    if storage_plan.graph_schema != graph.schema:
        raise StageOptimizedProgramError("StageStoragePlan graph schema does not match canonical graph")
    if output_uid is None:
        output_uid = execution_plan.output_uid
    if output_uid not in variants:
        raise StageOptimizedProgramError(f"output UID {output_uid} is absent from variants")
    if run_domain is None:
        run_domain = StageRunDomain(
            uid=_stable_id("stage_run_domain", (execution_plan.uid, 1, "default_single_point")),
            point_count=1,
            run_key_tokens=("<default>",),
            proof_kind="implicit_single_point_v1",
        )
    if run_domain.point_count <= 0:
        raise StageOptimizedProgramError("Stage run domain must contain at least one point")

    physical = _physical_values(storage_plan)
    kernels = tuple(StageProgramKernel(
        uid=x.uid,
        stage_index=x.stage_index,
        task_uid=x.task_uid,
        kind=x.kind,
        component_uid=x.component_uid,
        canonical_uids=x.canonical_uids,
        domain_uids=x.domain_uids,
        scan_direction=x.scan_direction,
        reduction_range_asts=x.reduction_range_asts,
        blockers=x.blockers,
    ) for x in storage_plan.kernels)
    stages = tuple(StageProgramStage(
        index=x.index,
        kernel_uids=x.kernel_uids,
        materialized_value_uids=x.materialized_value_uids,
        stage_carried_scalar_uids=x.stage_carried_scalar_uids,
        incoming_barrier_uids=x.incoming_barrier_uids,
        outgoing_barrier_uids=x.outgoing_barrier_uids,
    ) for x in storage_plan.stages)
    barriers = _program_barriers(execution_plan, storage_plan, physical)
    normalized_operators = tuple(sorted(normalized_operators, key=lambda x: (x.uid, x.kind)))
    for op in normalized_operators:
        if op.uid not in variants:
            raise StageOptimizedProgramError(
                f"normalized operator references unknown canonical UID {op.uid}"
            )
    operators = _operator_requirements(
        graph, execution_plan, variants, normalized_operators
    )

    stage_indices = tuple(x.index for x in stages)
    input_specs = {x.key: x for x in inputs}
    static_input_values: dict[str, Any] = {}
    for fact in validated_static_facts:
        spec = input_specs.get(fact.validation_input_key)
        if spec is None:
            continue
        value = fact.value
        if spec.enum_labels is not None:
            if value not in spec.enum_labels:
                continue
            value = spec.enum_labels.index(value)
        if isinstance(value, (bool, int, float)):
            static_input_values[fact.validation_input_key] = value
    direct = build_stage_direct_contract(
        graph, execution_plan, storage_plan, variants,
        static_input_values=static_input_values,
        proof_input_rows=proof_input_rows,
    )

    blockers: set[str] = set(storage_plan.capability_blockers)
    for op in normalized_operators:
        blockers.update(op.capability_blockers)
    approved_operator_uids = {op.uid for op in normalized_operators if op.python_supported or op.cython_supported}
    blockers.update(
        x for x in execution_plan.capability_blockers
        if not x.startswith("backend_multistage_execution_not_supported:")
        and not x.startswith("backend_cross_stage_materialization_not_supported:")
        and not (
            x.startswith("execution_semantic_unapproved:")
            and x.split(":", 1)[1] in approved_operator_uids
        )
    )
    if execution_plan.unplaced_task_uids:
        blockers.add(f"stage_program_unplaced_tasks:{len(execution_plan.unplaced_task_uids)}")
    for barrier in barriers:
        for blocker in barrier.blockers:
            blockers.add(f"stage_program_barrier_unsupported:{barrier.uid}:{blocker}")

    one_stage_adapter = stage_indices == (0,) and legacy_program is not None
    normalized_python_ok = all(op.python_supported for op in normalized_operators)
    normalized_cython_ok = all(op.cython_supported for op in normalized_operators)
    if not normalized_python_ok:
        blockers.add("stage_normalized_operator_python_unsupported")
    if not normalized_cython_ok:
        blockers.add("stage_normalized_operator_cython_unsupported")

    direct_input_contract = tuple(x.key for x in inputs) == tuple(input_order)
    direct_python_blockers: set[str] = set(direct.blockers)
    for barrier in barriers:
        if "python" not in barrier.backend_support:
            direct_python_blockers.add(f"direct_barrier_python_unsupported:{barrier.uid}")
        direct_python_blockers.update(
            f"direct_barrier:{barrier.uid}:{blocker}" for blocker in barrier.blockers
        )
    if not direct_input_contract:
        direct_python_blockers.add("direct_input_contract_incomplete")
    if not normalized_python_ok:
        direct_python_blockers.add("direct_normalized_operator_python_unsupported")
    direct_python_blockers.update(
        op.blocker for op in operators
        if op.kind in {
            "unresolved_model_bound_accessor",
            "unresolved_model_bound_static_attribute",
            "unresolved_model_bound_space_selection",
        }
    )
    direct_python_supported = bool(
        not execution_plan.unplaced_task_uids
        and direct.supported
        and direct_input_contract
        and normalized_python_ok
        and not direct_python_blockers
    )
    direct_cython_blockers: set[str] = set(direct_python_blockers)
    for barrier in barriers:
        if "cython" not in barrier.backend_support:
            direct_cython_blockers.add(f"direct_barrier_cython_unsupported:{barrier.uid}")
    if not normalized_cython_ok:
        direct_cython_blockers.add("direct_normalized_operator_cython_unsupported")
    for spec in inputs:
        supported_layout = (
            (spec.ndim == 0 and spec.dtype in {"float64", "int64", "bool"})
            or (spec.ndim == 1 and spec.dtype in {"float64", "int64", "bool"})
            or (spec.ndim == 2 and spec.dtype == "float64")
            or (spec.scope == "global" and spec.ndim >= 2 and spec.dtype in {"float64", "int64", "bool"})
        )
        if not supported_layout:
            direct_cython_blockers.add(
                f"direct_cython_input_layout_unsupported:{spec.key}:{spec.dtype}:{spec.ndim}"
            )
    direct_cython_supported = bool(
        direct_python_supported and normalized_cython_ok and not direct_cython_blockers
    )
    # External state-free/operator bindings remain explicit capability records.
    # Direct Stage execution itself is permitted only when the normalized operator
    # contract is complete; mature fallback is never consulted.
    python_blocking = tuple(sorted(
        x for x in blockers
        if not x.startswith("execution_semantic_unapproved:")
        and not x.startswith("external_call_requires_binding:")
        and not x.startswith("physical_iteration_step_unproved_for_ring_compaction:")
    ))
    python_supported = bool(
        (direct_python_supported and normalized_python_ok and not python_blocking)
        or (one_stage_adapter and normalized_python_ok)
    )
    cython_supported = bool(
        (direct_cython_supported and normalized_cython_ok and not python_blocking)
        or (one_stage_adapter and normalized_cython_ok)
    )

    uid = _stable_id("stage_optimized_program", (
        execution_plan.uid, storage_plan.uid, output_uid, run_domain.uid,
        f"output_invocation:{output_invocation.uid if output_invocation is not None else '<none>'}",
        f"result_domain:{result_domain.uid if result_domain is not None else '<none>'}",
        *(f"{x.value_uid}:{x.working_kind}:{x.current_slot}:{x.ring_base}:{x.history_slot}:{x.materialized_slot}:{x.scalar_slot}:{x.producer_stage_index}:{x.consumer_stage_indices}" for x in physical),
        *(x.uid for x in kernels),
        *(f"barrier:{x.uid}:{x.source_stage_index}:{x.target_stage_index}:{x.availability}:{x.proof_kind}:{x.evidence_uids}:{x.storage_fence_uid}:{x.execution_mode}:{x.execution_mode_proof_kind}:{x.required_materialized_value_uids}:{x.required_materialized_slots}:{x.backend_support}:{x.blockers}" for x in barriers),
        *(f"op:{x.uid}:{x.kind}:{x.result_dtype}:{x.result_shape}" for x in normalized_operators),
        *(f"iter:{x.uid}:{x.coordinate_step}:{x.scan_direction}" for x in direct.iteration_domains),
        *(f"block:{x.uid}" for x in direct.execution_blocks),
        *(f"seed:{x.uid}:{x.coordinate}:{x.active_boundary_coordinates}:{x.replay_start_coordinate}:{x.proof_kind}" for x in direct.boundary_seeds),
        *(f"aux:{x.uid}:{x.scan_direction}:{x.ring_depth}" for x in direct.auxiliary_recurrences),
        *(f"input:{x.key}:{x.dtype}:{x.ndim}:{x.scope}:{x.expected_shape}:{x.domain_token}:{x.enum_labels}" for x in inputs),
        *(f"guard:{x.code}:{x.exception_type}:{x.message}" for x in guards),
        *(f"static:{x.source_name}:{x.value}:{x.validation_input_key}" for x in validated_static_facts),
    ))
    return StageOptimizedProgram(
        schema="modelx_graph.stage_optimized_program.v11",
        uid=uid,
        output_uid=output_uid,
        run_domain=run_domain,
        execution_plan=execution_plan,
        storage_plan=storage_plan,
        graph=graph,
        variants=variants,
        output_invocation=output_invocation,
        result_domain=result_domain,
        stages=stages,
        kernels=kernels,
        physical_values=physical,
        barriers=barriers,
        operators=operators,
        normalized_operators=normalized_operators,
        input_order=tuple(input_order),
        inputs=tuple(inputs),
        guards=tuple(guards),
        validated_static_facts=tuple(validated_static_facts),
        iteration_domains=direct.iteration_domains,
        execution_blocks=direct.execution_blocks,
        boundary_seeds=direct.boundary_seeds,
        pure_maps=direct.pure_maps,
        auxiliary_recurrences=direct.auxiliary_recurrences,
        direct_python_supported=direct_python_supported,
        direct_cython_supported=direct_cython_supported,
        direct_python_blockers=tuple(sorted(direct_python_blockers)),
        direct_cython_blockers=tuple(sorted(direct_cython_blockers)),
        python_supported=python_supported,
        cython_supported=cython_supported,
        capability_blockers=tuple(sorted(blockers)),
        legacy_program=legacy_program,
        validation_notes=(
            "semantic stage and lifetime authority comes only from StageExecutionPlan/StageStoragePlan",
            "physical slots are assigned deterministically without consulting legacy slot counts",
            "one-stage execution blocks and auxiliary recurrence ABI are frozen without legacy coord_phase",
            "legacy OptimizedProgram is optional comparison evidence only and is never an emission fallback",
            "full-stage fences consume exact StageStorageFence history/bounds/mode contracts and map them only to physical slots",
            "multi-stage Python may use explicit external operator bindings; Cython remains fail-closed",
        ),
    )


def decode_runtime_payload(encoded: str) -> StageRuntimePayload:
    value = pickle.loads(base64.b85decode(encoded.encode("ascii")))
    if not isinstance(value, StageRuntimePayload):
        raise StageOptimizedProgramError("serialized stage runtime payload has unexpected type")
    return value


def build_stage_optimized_program_from_model(
    model: Any,
    *,
    optimization_level: str = "O2",
    legacy_comparison_program: Any | None = None,
) -> StageOptimizedProgram:
    """Build the production StageOptimizedProgram from canonical Stage facts.

    Mature ``OptimizedProgram`` construction is never performed here.  An already
    constructed comparison program may be attached explicitly for audit tooling,
    but production callers should leave ``legacy_comparison_program`` as ``None``.
    """
    graph = getattr(model, "canonical_semantic_graph", None)
    execution = getattr(model, "stage_execution_plan", None)
    storage = getattr(model, "stage_storage_plan", None)
    if graph is None or execution is None or storage is None:
        raise StageOptimizedProgramError(
            "CanonicalModel lacks canonical graph/stage execution/storage plans"
        )
    optimized_inputs = tuple(
        OptimizedInputSpec(
            key=key, dtype=spec.dtype, ndim=int(spec.ndim), scope=spec.scope,
            description=spec.description, expected_shape=spec.expected_shape,
            domain_token=spec.domain_token, normalized_table=spec.normalized_table,
            point_row_source=spec.point_row_source, point_field=spec.point_field,
            normalized_structured_column=spec.normalized_structured_column,
            enum_labels=spec.enum_labels, category_validity_key=spec.category_validity_key,
            normalized_lookup_1d=spec.normalized_lookup_1d,
            lookup_1d_role=spec.lookup_1d_role,
        )
        for key, spec in model.inputs.items()
    )
    # Preserve exact run-domain correlation for direct physical proofs that may
    # depend on point-varying scalar inputs, most notably iteration boundaries.
    # These rows are compile-time proof evidence only; Stage runtime inputs remain
    # the ordinary columnar numeric ABI.
    proof_input_rows: tuple[Mapping[str, Any], ...] = ()
    try:
        bound_for_proof = model.bind_inputs()
        run_count = len(tuple(model.run_keys))
        rows: list[dict[str, Any]] = []
        for i in range(run_count):
            row: dict[str, Any] = {}
            for key, spec in model.inputs.items():
                if spec.ndim == 0:
                    value = bound_for_proof[key]
                elif spec.scope == "point" and spec.ndim == 1:
                    value = bound_for_proof[key][i]
                else:
                    continue
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except Exception:
                        pass
                if isinstance(value, (bool, int, float)):
                    row[key] = value
            rows.append(row)
        proof_input_rows = tuple(rows)
    except Exception:
        # Direct lowering remains fail-closed if the exact boundary cannot be
        # proven without this optional finite run-domain evidence.
        proof_input_rows = ()
    run_keys = tuple(getattr(model, "run_keys", ()))
    if not run_keys:
        raise StageOptimizedProgramError("CanonicalModel lacks a finite production run domain")
    run_key_tokens = tuple(repr(x) for x in run_keys)
    run_domain = StageRunDomain(
        uid=_stable_id("stage_run_domain", (
            graph.formula_hash,
            getattr(getattr(model, "run_domain", None), "uid", "<legacy>"),
            len(run_keys),
            *run_key_tokens,
        )),
        point_count=len(run_keys),
        run_key_tokens=run_key_tokens,
        semantic_run_domain_uid=getattr(getattr(model, "run_domain", None), "uid", None),
    )

    guard_by_code: dict[int, GuardSpec] = {}
    for cv in model.variants.values():
        for guard in cv.guards:
            prior = guard_by_code.get(int(guard.code))
            if prior is not None and prior != guard:
                raise StageOptimizedProgramError(f"conflicting stage guard code {guard.code}")
            guard_by_code[int(guard.code)] = guard
    return build_stage_optimized_program(
        graph,
        execution,
        storage,
        variants=model.variants,
        output_uid=model.output_uid,
        legacy_program=legacy_comparison_program,
        normalized_operators=tuple(getattr(model, "normalized_operators", ())),
        input_order=tuple(model.inputs),
        inputs=optimized_inputs,
        guards=tuple(guard_by_code[k] for k in sorted(guard_by_code)),
        validated_static_facts=tuple(getattr(model, "validated_static_facts", ())),
        proof_input_rows=proof_input_rows,
        run_domain=run_domain,
        output_invocation=getattr(model, "output_invocation", None),
        result_domain=getattr(model, "result_domain", None),
    )
