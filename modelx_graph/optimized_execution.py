from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .loop_recovery import LiteralBlock
from .native_bindings import RuntimeArgSite, build_runtime_arg_sites
from .native_family import (
    _exact_native_formula_ok,
    _family_output_structural_kinds,
    _typed_slot_plan,
)
from .native_plan import (
    NativeFamilyBackendPlan,
    build_native_family_backend_plan,
    expand_native_family_instance,
)
from .native_references import ReferenceBindingPlan, build_reference_binding_plan


class OptimizedPlanError(RuntimeError):
    """Fail-closed error while deriving the optional optimized backend IR."""


class RoleBackend(str, Enum):
    """Execution ownership for one canonical family role.

    ``REGISTER_NATIVE`` means the role is eligible for the future local/register
    backend. ``SLOT_PYTHON`` means the canonical slot/Python path remains the
    owner.  The first architecture slice deliberately has no semantic middle
    ground: it proves ownership and boundaries before adding more codegen modes.
    """

    REGISTER_NATIVE = "register_native"
    SLOT_PYTHON = "slot_python"


@dataclass(frozen=True)
class BindingSitePlan:
    site_id: int
    arg_index: int
    cython_type: str
    value_kind: str
    structural_kind: str


@dataclass(frozen=True)
class ReferenceSiteSummary:
    site_id: int
    name: str
    pool: str
    arg_count: int
    structural_kind: str


@dataclass(frozen=True)
class RoleExecutionPlan:
    role: int
    formula_op_id: int
    backend: RoleBackend
    output_pool: str | None
    output_structural_kind: int
    binding_sites: tuple[BindingSitePlan, ...]
    reference_sites: tuple[ReferenceSiteSummary, ...]
    reasons: tuple[str, ...] = ()

    @property
    def register_native(self) -> bool:
        return self.backend is RoleBackend.REGISTER_NATIVE


@dataclass(frozen=True)
class OptimizedFamilyPlan:
    """Code-level optimized ABI shared by every instance of one loop family."""

    family_id: int
    code_signature: str
    formula_op_ids: tuple[int, ...]
    variants: tuple[tuple[int, ...], ...]
    role_plans: tuple[RoleExecutionPlan, ...]

    @property
    def register_roles(self) -> tuple[int, ...]:
        return tuple(x.role for x in self.role_plans if x.register_native)

    @property
    def register_role_count(self) -> int:
        return len(self.register_roles)


@dataclass(frozen=True)
class RegisterValuePlan:
    node_id: int
    role: int
    formula_op_id: int
    pool: str
    slot_offset: int
    spill_required: bool
    spill_reasons: tuple[str, ...]
    internal_register_consumers: int
    boundary_consumers: int
    external_consumers: int


@dataclass(frozen=True)
class ReloadValuePlan:
    node_id: int
    pool: str
    slot_offset: int
    register_consumer_count: int
    source_kind: str  # live_in | slot_boundary


@dataclass(frozen=True)
class PythonBoundaryOccurrence:
    position_in_region: int
    node_id: int
    role: int
    formula_op_id: int
    spill_before: tuple[int, ...]
    register_consumers_after: tuple[int, ...]


@dataclass(frozen=True)
class RegisterReferenceReadPlan:
    """Instance-level source contract for one direct scheduled Cell call site.

    ``register_ring`` is intentionally structural rather than temporal. For every
    executed occurrence the referenced producer is the same source role and the
    same finite distance behind the latest source-role occurrence already produced
    at that point. A future backend can therefore retain a ring of C locals without
    interpreting any coordinate as time. Other shapes remain ``slot_reference``.
    """

    consumer_role: int
    formula_op_id: int
    site_id: int
    source_kind: str  # register_ring | slot_reference | dormant
    source_role: int | None
    lag_from_latest_produced: int | None
    executed_occurrences: int
    dormant_occurrences: int
    slot_source_node_ids: tuple[int, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def register_local(self) -> bool:
        return self.source_kind == "register_ring"


@dataclass(frozen=True)
class LiteralRegionPlan:
    block_index: int
    node_ids: tuple[int, ...]

    @property
    def kind(self) -> str:
        return "literal"

    @property
    def operation_count(self) -> int:
        return len(self.node_ids)

    @property
    def register_operation_count(self) -> int:
        return 0

    @property
    def slot_operation_count(self) -> int:
        return self.operation_count


@dataclass(frozen=True)
class SlotRegionPlan:
    block_index: int
    family_id: int
    instance_ordinal: int
    variant_ids: tuple[int, ...]
    node_ids: tuple[int, ...]
    reasons: tuple[str, ...]

    @property
    def kind(self) -> str:
        return "slot"

    @property
    def operation_count(self) -> int:
        return len(self.node_ids)

    @property
    def register_operation_count(self) -> int:
        return 0

    @property
    def slot_operation_count(self) -> int:
        return self.operation_count


@dataclass(frozen=True)
class RegisterRegionPlan:
    """One exact canonical loop instance with optional register-owned roles.

    The region retains the family's compact grammar and bindings; node-sized
    tuples below are proof/debug metadata only.  Future generated source must use
    the compact role/variant/binding representation rather than unrolling these
    node streams.
    """

    block_index: int
    family_id: int
    instance_ordinal: int
    variant_ids: tuple[int, ...]
    node_ids: tuple[int, ...]
    register_roles: tuple[int, ...]
    register_values: tuple[RegisterValuePlan, ...]
    reload_values: tuple[ReloadValuePlan, ...]
    boundaries: tuple[PythonBoundaryOccurrence, ...]
    reference_reads: tuple[RegisterReferenceReadPlan, ...]
    live_in_node_ids: tuple[int, ...]
    live_out_node_ids: tuple[int, ...]

    @property
    def kind(self) -> str:
        return "register"

    @property
    def operation_count(self) -> int:
        return len(self.node_ids)

    @property
    def register_operation_count(self) -> int:
        roles = set(self.register_roles)
        if not roles:
            return 0
        # role occurrence counts are encoded by register_values exactly once per
        # native result, so this does not need operation-stream re-expansion.
        return len(self.register_values)

    @property
    def slot_operation_count(self) -> int:
        return self.operation_count - self.register_operation_count

    @property
    def spill_count(self) -> int:
        return sum(x.spill_required for x in self.register_values)

    @property
    def register_reference_site_count(self) -> int:
        return sum(x.register_local for x in self.reference_reads)

    @property
    def slot_reference_site_count(self) -> int:
        return sum(x.source_kind == "slot_reference" for x in self.reference_reads)

    @property
    def register_ring_depth_by_role(self) -> dict[int, int]:
        depth: dict[int, int] = {}
        for read in self.reference_reads:
            if not read.register_local or read.source_role is None:
                continue
            lag = int(read.lag_from_latest_produced or 0)
            depth[read.source_role] = max(depth.get(read.source_role, 0), lag + 1)
        return depth


OptimizedRegion = LiteralRegionPlan | SlotRegionPlan | RegisterRegionPlan


@dataclass(frozen=True)
class OptimizedExecutionPlan:
    """Backend-only optimization plan layered below CanonicalExecutionPlan.

    This object never changes canonical semantics. It records which operations a
    future backend may own in C locals and the exact slot boundaries required to
    preserve interoperability with existing Python/object execution.
    """

    families: tuple[OptimizedFamilyPlan, ...]
    regions: tuple[OptimizedRegion, ...]
    canonical_operation_count: int
    register_operation_count: int
    slot_operation_count: int
    python_boundary_count: int
    register_value_count: int
    spill_value_count: int
    reload_value_count: int
    register_reference_site_count: int
    slot_reference_site_count: int

    @property
    def register_fraction(self) -> float:
        return (
            self.register_operation_count / self.canonical_operation_count
            if self.canonical_operation_count else 0.0
        )

    @property
    def region_counts(self) -> dict[str, int]:
        out = {"register": 0, "slot": 0, "literal": 0}
        for region in self.regions:
            out[region.kind] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_operation_count": self.canonical_operation_count,
            "register_operation_count": self.register_operation_count,
            "slot_operation_count": self.slot_operation_count,
            "register_fraction": self.register_fraction,
            "python_boundary_count": self.python_boundary_count,
            "register_value_count": self.register_value_count,
            "spill_value_count": self.spill_value_count,
            "reload_value_count": self.reload_value_count,
            "register_reference_site_count": self.register_reference_site_count,
            "slot_reference_site_count": self.slot_reference_site_count,
            "region_counts": self.region_counts,
            "families": [
                {
                    "family_id": f.family_id,
                    "code_signature": f.code_signature,
                    "formula_op_ids": list(f.formula_op_ids),
                    "variant_count": len(f.variants),
                    "register_roles": list(f.register_roles),
                    "roles": [
                        {
                            "role": rp.role,
                            "formula_op_id": rp.formula_op_id,
                            "backend": rp.backend.value,
                            "output_pool": rp.output_pool,
                            "output_structural_kind": rp.output_structural_kind,
                            "binding_kinds": [x.structural_kind for x in rp.binding_sites],
                            "reference_kinds": [x.structural_kind for x in rp.reference_sites],
                            "reasons": list(rp.reasons),
                        }
                        for rp in f.role_plans
                    ],
                }
                for f in self.families
            ],
            "regions": [
                {
                    "kind": r.kind,
                    "block_index": r.block_index,
                    "operation_count": r.operation_count,
                    "register_operation_count": r.register_operation_count,
                    "slot_operation_count": r.slot_operation_count,
                    **(
                        {
                            "family_id": r.family_id,
                            "instance_ordinal": r.instance_ordinal,
                            "register_roles": list(r.register_roles),
                            "spills": r.spill_count,
                            "reloads": len(r.reload_values),
                            "boundaries": len(r.boundaries),
                            "register_reference_sites": r.register_reference_site_count,
                            "slot_reference_sites": r.slot_reference_site_count,
                            "register_ring_depth_by_role": r.register_ring_depth_by_role,
                            "live_in_count": len(r.live_in_node_ids),
                            "live_out_count": len(r.live_out_node_ids),
                            "reference_reads": [
                                {
                                    "consumer_role": read.consumer_role,
                                    "formula_op_id": read.formula_op_id,
                                    "site_id": read.site_id,
                                    "source_kind": read.source_kind,
                                    "source_role": read.source_role,
                                    "lag_from_latest_produced": read.lag_from_latest_produced,
                                    "executed_occurrences": read.executed_occurrences,
                                    "dormant_occurrences": read.dormant_occurrences,
                                    "slot_source_node_ids": list(read.slot_source_node_ids),
                                    "reasons": list(read.reasons),
                                }
                                for read in r.reference_reads
                            ],
                        }
                        if isinstance(r, RegisterRegionPlan)
                        else {}
                    ),
                    **(
                        {
                            "family_id": r.family_id,
                            "instance_ordinal": r.instance_ordinal,
                            "reasons": list(r.reasons),
                        }
                        if isinstance(r, SlotRegionPlan)
                        else {}
                    ),
                }
                for r in self.regions
            ],
        }

    def validate(self, compiler: Any, *, deep_reference_check: bool = True) -> None:
        """Prove exact canonical coverage and local/slot boundary completeness.

        ``deep_reference_check`` independently rebuilds reference-site evidence and
        is intended for tests/debug validation. Plan construction already derived
        that evidence once, so the normal build path skips the duplicate expensive
        pass while retaining all schedule, instance, spill and reload invariants.
        """
        if compiler.structured is None or compiler.structured.canonical_plan is None:
            raise OptimizedPlanError("canonical execution plan is required")
        seq_ids = tuple(op.node_id for op in compiler.sequential.ops)
        planned_ids = tuple(nid for region in self.regions for nid in region.node_ids)
        if planned_ids != seq_ids:
            raise OptimizedPlanError("optimized region stream differs from canonical schedule")
        if sum(r.operation_count for r in self.regions) != self.canonical_operation_count:
            raise OptimizedPlanError("optimized region operation accounting mismatch")
        if self.register_operation_count + self.slot_operation_count != self.canonical_operation_count:
            raise OptimizedPlanError("optimized backend operation accounting mismatch")
        computed_register = sum(r.register_operation_count for r in self.regions)
        computed_slot = sum(r.slot_operation_count for r in self.regions)
        computed_boundaries = sum(
            len(r.boundaries) for r in self.regions if isinstance(r, RegisterRegionPlan)
        )
        computed_values = sum(
            len(r.register_values) for r in self.regions if isinstance(r, RegisterRegionPlan)
        )
        computed_spills = sum(
            r.spill_count for r in self.regions if isinstance(r, RegisterRegionPlan)
        )
        computed_reloads = sum(
            len(r.reload_values) for r in self.regions if isinstance(r, RegisterRegionPlan)
        )
        computed_register_ref_sites = sum(
            r.register_reference_site_count for r in self.regions if isinstance(r, RegisterRegionPlan)
        )
        computed_slot_ref_sites = sum(
            r.slot_reference_site_count for r in self.regions if isinstance(r, RegisterRegionPlan)
        )
        if (computed_register, computed_slot) != (self.register_operation_count, self.slot_operation_count):
            raise OptimizedPlanError("optimized region/backend summary mismatch")
        if computed_boundaries != self.python_boundary_count:
            raise OptimizedPlanError("optimized boundary summary mismatch")
        if computed_values != self.register_value_count:
            raise OptimizedPlanError("optimized register-value summary mismatch")
        if computed_spills != self.spill_value_count:
            raise OptimizedPlanError("optimized spill summary mismatch")
        if computed_reloads != self.reload_value_count:
            raise OptimizedPlanError("optimized reload summary mismatch")
        if computed_register_ref_sites != self.register_reference_site_count:
            raise OptimizedPlanError("optimized register-reference summary mismatch")
        if computed_slot_ref_sites != self.slot_reference_site_count:
            raise OptimizedPlanError("optimized slot-reference summary mismatch")

        canonical = compiler.structured.canonical_plan
        if len(self.regions) != len(compiler.structured.blocks):
            raise OptimizedPlanError("optimized region count differs from canonical block count")
        family_ids = [f.family_id for f in self.families]
        if len(family_ids) != len(set(family_ids)):
            raise OptimizedPlanError("optimized family ids are not unique")
        if set(family_ids) != {f.family_id for f in canonical.code_families}:
            raise OptimizedPlanError("optimized family set differs from canonical family set")
        ordinal_by_block: dict[tuple[int, int], int] = {}
        for family_id in family_ids:
            backend = build_native_family_backend_plan(compiler, family_id)
            for ordinal, inst in enumerate(backend.instances):
                ordinal_by_block[(family_id, inst.block_index)] = ordinal
        for block_index, region in enumerate(self.regions):
            if region.block_index != block_index:
                raise OptimizedPlanError("optimized region block ordering mismatch")
            instance_index = canonical.block_to_instance[block_index]
            if instance_index is None:
                if not isinstance(region, LiteralRegionPlan):
                    raise OptimizedPlanError(f"canonical literal block {block_index} changed backend kind")
                continue
            if isinstance(region, LiteralRegionPlan):
                raise OptimizedPlanError(f"canonical family block {block_index} changed to literal")
            inst = canonical.loop_instances[instance_index]
            if region.family_id != inst.family_id:
                raise OptimizedPlanError(f"block {block_index} family identity mismatch")
            if region.variant_ids != tuple(inst.variant_ids):
                raise OptimizedPlanError(f"block {block_index} variant stream mismatch")
            expected_ordinal = ordinal_by_block[(inst.family_id, block_index)]
            if region.instance_ordinal != expected_ordinal:
                raise OptimizedPlanError(f"block {block_index} family instance ordinal mismatch")

        deps_by_src: dict[int, list[int]] = defaultdict(list)
        deps_by_dst: dict[int, list[int]] = defaultdict(list)
        for src, dst in compiler.trace.dependencies:
            deps_by_src[src].append(dst)
            deps_by_dst[dst].append(src)
        target_ids = {x for x in compiler.trace.target_node_ids if x is not None}

        family_by_id = {f.family_id: f for f in self.families}
        if deep_reference_check:
            analysis = compiler.native_plan or compiler.analyze_native()
            typed = _typed_slot_plan(compiler, analysis, include_objects=True)
            analysis_by_id = {x.formula_op_id: x for x in analysis.formulae}
        else:
            typed = analysis_by_id = None
        validation_family_context: dict[int, Any] = {}
        for region in self.regions:
            if not isinstance(region, RegisterRegionPlan):
                continue
            if region.family_id not in family_by_id:
                raise OptimizedPlanError(f"register region references unknown family {region.family_id}")
            region_nodes = set(region.node_ids)
            role_by_node = _role_by_node_for_region(compiler, region)
            register_roles = set(region.register_roles)
            if deep_reference_check:
                if region.family_id not in validation_family_context:
                    validation_family_context[region.family_id] = _build_family_context(
                        compiler, region.family_id, typed, analysis_by_id
                    )
                backend, _sites, refs, _native, _out_kinds = validation_family_context[region.family_id]
                expansion = expand_native_family_instance(compiler, backend, region.instance_ordinal)
                expected_reference_reads = _register_reference_read_plans(
                    compiler, backend, refs, region.instance_ordinal, expansion,
                    register_roles, deps_by_dst,
                )
                if expected_reference_reads != region.reference_reads:
                    raise OptimizedPlanError(
                        f"register region {region.block_index} reference-source metadata mismatch"
                    )
            register_nodes = {nid for nid, role in role_by_node.items() if role in register_roles}
            spill_ids = {x.node_id for x in region.register_values if x.spill_required}
            reload_ids = {x.node_id for x in region.reload_values}
            required_slot_reference_spills = {
                nid for read in region.reference_reads if read.source_kind == "slot_reference"
                for nid in read.slot_source_node_ids
            }
            missing_slot_ref_spills = required_slot_reference_spills - spill_ids
            if missing_slot_ref_spills:
                raise OptimizedPlanError(
                    f"register direct-slot reference producers escape without a spill: {sorted(missing_slot_ref_spills)}"
                )

            for nid in register_nodes:
                outside = any(dst not in region_nodes for dst in deps_by_src.get(nid, ()))
                boundary = any(
                    dst in region_nodes and role_by_node.get(dst) not in register_roles
                    for dst in deps_by_src.get(nid, ())
                )
                if (outside or boundary or nid in target_ids) and nid not in spill_ids:
                    raise OptimizedPlanError(f"register value {nid} escapes without a spill")

            for dst in register_nodes:
                for src in deps_by_dst.get(dst, ()):
                    local = src in register_nodes
                    if not local and src not in reload_ids:
                        raise OptimizedPlanError(
                            f"register consumer {dst} reads nonlocal value {src} without reload"
                        )

            boundary_by_node = {b.node_id: b for b in region.boundaries}
            for dst in region_nodes - register_nodes:
                b = boundary_by_node.get(dst)
                if b is None:
                    raise OptimizedPlanError(f"slot/Python node {dst} has no boundary occurrence")
                required = {
                    src for src in deps_by_dst.get(dst, ())
                    if src in register_nodes
                }
                if not required.issubset(set(b.spill_before)):
                    raise OptimizedPlanError(
                        f"boundary {dst} misses register spills {sorted(required - set(b.spill_before))}"
                    )
                expected_register_consumers = {
                    consumer for consumer in deps_by_src.get(dst, ()) if consumer in register_nodes
                }
                if expected_register_consumers != set(b.register_consumers_after):
                    raise OptimizedPlanError(
                        f"boundary {dst} register-consumer metadata mismatch"
                    )


def _reference_kind_name(kind: int) -> str:
    return {0: "constant", 1: "affine", 2: "modulo", 3: "table", -1: "missing"}.get(
        int(kind), "dynamic"
    )


def _reference_site_summaries(
    refs: ReferenceBindingPlan, formula_op_id: int
) -> tuple[ReferenceSiteSummary, ...]:
    out = []
    bindings_by_site: dict[int, set[int]] = defaultdict(set)
    for inst in refs.instances:
        for item in inst.sites:
            bindings_by_site[item.site_id].add(int(item.kind))
    for site in refs.sites:
        if site.formula_op_id != formula_op_id:
            continue
        kinds = bindings_by_site.get(site.site_id, set())
        structural = _reference_kind_name(next(iter(kinds))) if len(kinds) == 1 else "dynamic"
        out.append(
            ReferenceSiteSummary(
                site_id=site.site_id,
                name=site.name,
                pool=site.pool,
                arg_count=site.arg_count,
                structural_kind=structural,
            )
        )
    return tuple(out)


def _build_family_context(compiler: Any, family_id: int, typed: Any, analysis_by_id: dict[int, Any]):
    backend = build_native_family_backend_plan(compiler, family_id)
    sites = build_runtime_arg_sites(backend)
    from .whole_artifact import _native_scalar_environment_stable

    plan = compiler.structured.canonical_plan
    assert plan is not None
    nodes_by_role: dict[Any, list[Any]] = defaultdict(list)
    for node in compiler.trace.nodes:
        nodes_by_role[node.shape_token].append(node)

    native = {
        fid: (
            _exact_native_formula_ok(compiler, fid, analysis_by_id[fid])
            and _native_scalar_environment_stable(
                compiler, fid, nodes_by_role.get(plan.formula_ops[fid].role, [])
            )
        )
        for fid in set(backend.kernel.formula_op_ids)
    }
    frozen_plan = getattr(compiler, "frozen_reference_plan", None)
    frozen_candidates: set[int] = set()
    if frozen_plan is not None and frozen_plan.site_count:
        from .frozen_references import frozen_formula_ast_exact_ok
        for fid in set(backend.kernel.formula_op_ids):
            if not frozen_plan.sites_for_formula(fid):
                continue
            if not frozen_formula_ast_exact_ok(compiler, fid, analysis_by_id[fid]):
                continue
            if not _native_scalar_environment_stable(
                compiler, fid, nodes_by_role.get(plan.formula_ops[fid].role, [])
            ):
                continue
            native[fid] = True
            frozen_candidates.add(fid)
    for role, fid in enumerate(backend.kernel.formula_op_ids):
        if fid not in typed.formula_by_id:
            native[fid] = False
        if any(site.cython_type == "object" for site in sites if site.role == role):
            native[fid] = False
    refs = build_reference_binding_plan(compiler, typed, backend, native)
    for fid in refs.unsupported_formula_ops:
        native[fid] = False
    if frozen_candidates:
        from .frozen_references import FrozenRuntimeFormulaEmitter
        sites_by_role: dict[int, list[Any]] = defaultdict(list)
        for site in sites:
            sites_by_role[site.role].append(site)
        for fid in frozen_candidates:
            if not native.get(fid, False):
                continue
            roles = [role for role, ffid in enumerate(backend.kernel.formula_op_ids) if ffid == fid]
            try:
                for role in roles:
                    emitter = FrozenRuntimeFormulaEmitter(
                        compiler, typed, fid, refs.sites,
                        frozen_sites=frozen_plan.sites_for_formula(fid),
                    )
                    formal_names = [a.arg for a in emitter.fn.args.args]
                    rows = sorted(sites_by_role.get(role, ()), key=lambda x: x.arg_index)
                    if len(formal_names) != len(rows):
                        raise OptimizedPlanError("frozen FormulaOp arity differs from runtime binding ABI")
                    arg_types = [(name, site.cython_type) for name, site in zip(formal_names, rows)]
                    emitter.emit_function(arg_types, analysis_by_id[fid].observed_return_type.dtype)
            except Exception:
                native[fid] = False
    output_kinds = _family_output_structural_kinds(compiler, typed, backend)
    return backend, sites, refs, native, output_kinds


def _role_by_node_for_region(compiler: Any, region: RegisterRegionPlan) -> dict[int, int]:
    backend = build_native_family_backend_plan(compiler, region.family_id)
    exp = expand_native_family_instance(compiler, backend, region.instance_ordinal)
    return {nid: role for role, nid in exp.operation_sequence}


def _register_reference_read_plans(
    compiler: Any,
    backend: NativeFamilyBackendPlan,
    refs: ReferenceBindingPlan,
    instance_ordinal: int,
    expansion: Any,
    register_roles: set[int],
    deps_by_dst: dict[int, list[int]],
) -> tuple[RegisterReferenceReadPlan, ...]:
    """Fit exact register-source relations for direct scheduled Cell call sites.

    The implementation deliberately walks realized dependency edges once per
    register role. The earlier laboratory version scanned every reference site
    across every destination occurrence, which becomes needlessly expensive on
    large phase families. Concrete Cell implementation identity assigns an edge
    to its reference site; only actually executed site/destination pairs are then
    analyzed.
    """
    byid = compiler.sequential.node_by_id
    position = {nid: i for i, (_role, nid) in enumerate(expansion.operation_sequence)}
    occurrence_by_node: dict[int, tuple[int, int]] = {}
    positions_by_role: dict[int, list[int]] = defaultdict(list)
    for role, node_ids in enumerate(expansion.role_node_ids):
        for occurrence, nid in enumerate(node_ids):
            occurrence_by_node[nid] = (role, occurrence)
            positions_by_role[role].append(position[nid])

    binding_rows = {
        row.site_id: row for row in refs.instances[instance_ordinal].sites
    } if instance_ordinal < len(refs.instances) else {}
    sites_by_fid: dict[int, list[Any]] = defaultdict(list)
    for site in refs.sites:
        sites_by_fid[site.formula_op_id].append(site)

    out: list[RegisterReferenceReadPlan] = []
    for consumer_role in sorted(register_roles):
        fid = backend.kernel.formula_op_ids[consumer_role]
        dest_ids = expansion.role_node_ids[consumer_role]
        role_sites = list(sites_by_fid.get(fid, ()))
        if not role_sites:
            continue

        token_to_site_ids: dict[int, list[int]] = defaultdict(list)
        fixed_reasons: dict[int, list[str]] = defaultdict(list)
        for site in role_sites:
            row = binding_rows.get(site.site_id)
            token = None if row is None else row.concrete_impl_token
            if token is None:
                fixed_reasons[site.site_id].append("reference:no_concrete_binding")
            else:
                token_to_site_ids[int(token)].append(site.site_id)
        for token, site_ids in token_to_site_ids.items():
            if len(site_ids) > 1:
                for site_id in site_ids:
                    fixed_reasons[site_id].append("reference:aliased_concrete_binding")

        # site_id -> destination node -> exact realized producer nodes. A single
        # pass over dependency edges replaces O(site * occurrence) dependency scans.
        source_by_site_dest: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for dst in dest_ids:
            for src in deps_by_dst.get(dst, ()):
                for site_id in token_to_site_ids.get(id(byid[src].obj), ()):
                    source_by_site_dest[site_id][dst].append(src)

        for site in role_sites:
            site_id = site.site_id
            reasons = list(fixed_reasons.get(site_id, ()))
            executed_rows = source_by_site_dest.get(site_id, {})
            executed = len(executed_rows)
            dormant = len(dest_ids) - executed
            observations: list[tuple[int, int]] = []  # source role, lag from latest produced

            for dst, matching in executed_rows.items():
                if len(matching) != 1:
                    reasons.append("reference:multiple_realized_sources")
                    continue
                src = matching[0]
                src_info = occurrence_by_node.get(src)
                if src_info is None or src_info[0] not in register_roles:
                    reasons.append("reference:slot_or_external_source")
                    continue
                source_role, source_occurrence = src_info
                source_positions = positions_by_role[source_role]
                latest_before = bisect_left(source_positions, position[dst]) - 1
                if latest_before < 0 or source_occurrence > latest_before:
                    reasons.append("reference:source_not_previously_produced")
                    continue
                observations.append((source_role, latest_before - source_occurrence))

            if executed == 0:
                kind = "dormant"; source_role = None; lag = None
            elif reasons or len(observations) != executed:
                kind = "slot_reference"; source_role = None; lag = None
            else:
                roles = {x[0] for x in observations}; lags = {x[1] for x in observations}
                if len(roles) == 1 and len(lags) == 1:
                    kind = "register_ring"
                    source_role = next(iter(roles)); lag = next(iter(lags))
                else:
                    kind = "slot_reference"; source_role = None; lag = None
                    if len(roles) != 1:
                        reasons.append("reference:source_role_changes")
                    if len(lags) != 1:
                        reasons.append("reference:ring_lag_changes")
            slot_source_node_ids: tuple[int, ...] = ()
            if kind == "slot_reference":
                slot_source_node_ids = tuple(sorted({
                    src
                    for matching in executed_rows.values()
                    for src in matching
                    if (src in occurrence_by_node and occurrence_by_node[src][0] in register_roles)
                }))
            out.append(RegisterReferenceReadPlan(
                consumer_role=consumer_role, formula_op_id=fid, site_id=site_id,
                source_kind=kind, source_role=source_role,
                lag_from_latest_produced=lag, executed_occurrences=executed,
                dormant_occurrences=dormant, slot_source_node_ids=slot_source_node_ids,
                reasons=tuple(dict.fromkeys(reasons)),
            ))
    return tuple(out)


def build_optimized_execution_plan(compiler: Any) -> OptimizedExecutionPlan:
    """Derive the first canonical register/slot ownership plan.

    This pass is intentionally non-executable. It establishes the exact ownership
    and spill/reload contract that subsequent Cython lowering will consume. A plan
    construction failure leaves all existing compiler backends untouched.
    """
    if compiler.structured is None:
        compiler.recover_loops()
    if compiler.slots is None:
        compiler.optimize_storage(); compiler.lower_slots()
    analysis = compiler.native_plan or compiler.analyze_native()
    plan = compiler.structured.canonical_plan
    if plan is None:
        raise OptimizedPlanError("canonical execution plan is required")

    typed = _typed_slot_plan(compiler, analysis, include_objects=True)
    analysis_by_id = {x.formula_op_id: x for x in analysis.formulae}
    deps_by_src: dict[int, list[int]] = defaultdict(list)
    deps_by_dst: dict[int, list[int]] = defaultdict(list)
    for src, dst in compiler.trace.dependencies:
        deps_by_src[src].append(dst)
        deps_by_dst[dst].append(src)
    target_ids = {x for x in compiler.trace.target_node_ids if x is not None}

    family_context: dict[int, Any] = {}
    optimized_families: list[OptimizedFamilyPlan] = []
    family_role_plans: dict[int, tuple[RoleExecutionPlan, ...]] = {}
    ordinal_by_block: dict[tuple[int, int], int] = {}
    for family in plan.code_families:
        ctx = _build_family_context(compiler, family.family_id, typed, analysis_by_id)
        family_context[family.family_id] = ctx
        backend, sites, refs, native, output_kinds = ctx
        sites_by_role: dict[int, list[RuntimeArgSite]] = defaultdict(list)
        for site in sites:
            sites_by_role[site.role].append(site)
        role_plans: list[RoleExecutionPlan] = []
        for role, fid in enumerate(backend.kernel.formula_op_ids):
            row = analysis_by_id[fid]
            reasons = []
            if not native.get(fid, False):
                reasons.extend(row.reasons or ("not_native",))
                reasons.extend(reason for ffid, reason in refs.fallback_reasons if ffid == fid)
            pool = typed.formula_by_id[fid].pool if fid in typed.formula_by_id else None
            role_plans.append(
                RoleExecutionPlan(
                    role=role, formula_op_id=fid,
                    backend=(RoleBackend.REGISTER_NATIVE if native.get(fid, False) else RoleBackend.SLOT_PYTHON),
                    output_pool=pool, output_structural_kind=output_kinds[role],
                    binding_sites=tuple(
                        BindingSitePlan(x.site_id, x.arg_index, x.cython_type, x.value_kind, x.binding_kind)
                        for x in sorted(sites_by_role.get(role, ()), key=lambda x: x.arg_index)
                    ),
                    reference_sites=_reference_site_summaries(refs, fid),
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
        family_role_plans[family.family_id] = tuple(role_plans)
        optimized_families.append(
            OptimizedFamilyPlan(
                family_id=family.family_id,
                code_signature=backend.kernel.code_signature,
                formula_op_ids=tuple(backend.kernel.formula_op_ids),
                variants=tuple(backend.kernel.variants),
                role_plans=tuple(role_plans),
            )
        )
        for ordinal, inst in enumerate(backend.instances):
            ordinal_by_block[(family.family_id, inst.block_index)] = ordinal

    regions: list[OptimizedRegion] = []
    for block_index, block in enumerate(compiler.structured.blocks):
        instance_index = plan.block_to_instance[block_index]
        if isinstance(block, LiteralBlock) or instance_index is None:
            regions.append(LiteralRegionPlan(block_index, tuple(op.node_id for op in block.ops)))
            continue

        inst = plan.loop_instances[instance_index]
        backend, sites, refs, native, output_kinds = family_context[inst.family_id]
        ordinal = ordinal_by_block[(inst.family_id, block_index)]
        expansion = expand_native_family_instance(compiler, backend, ordinal)
        role_by_node = {nid: role for role, nid in expansion.operation_sequence}
        region_node_ids = tuple(nid for _role, nid in expansion.operation_sequence)
        region_nodes = set(region_node_ids)

        role_plans = list(family_role_plans[inst.family_id])
        register_roles = {x.role for x in role_plans if x.register_native}

        # If a supposedly native role needs a non-numeric/non-addressable slot
        # value from a non-register producer, demote it. Iterate because one
        # demotion may turn a previously local dependency into a reload.
        changed = True
        while changed:
            changed = False
            for dst in region_node_ids:
                role = role_by_node[dst]
                if role not in register_roles:
                    continue
                for src in deps_by_dst.get(dst, ()):
                    if src in region_nodes and role_by_node.get(src) in register_roles:
                        continue
                    addr = typed.node_by_id.get(src)
                    if addr is None or addr.pool == "object":
                        register_roles.remove(role)
                        changed = True
                        break
                if changed:
                    break
        if register_roles != {x.role for x in role_plans if x.register_native}:
            revised = []
            for rp in role_plans:
                if rp.register_native and rp.role not in register_roles:
                    revised.append(
                        RoleExecutionPlan(
                            **{**rp.__dict__, "backend": RoleBackend.SLOT_PYTHON,
                               "reasons": rp.reasons + ("register:dependency_not_numeric_slot_backed",)}
                        )
                    )
                else:
                    revised.append(rp)
            role_plans = revised

        register_nodes = {
            nid for nid, role in role_by_node.items() if role in register_roles
        }
        live_in = sorted({
            src for dst in region_nodes for src in deps_by_dst.get(dst, ()) if src not in region_nodes
        })
        live_out = sorted({
            src for src in region_nodes
            if src in target_ids or any(dst not in region_nodes for dst in deps_by_src.get(src, ()))
        })

        reference_reads = _register_reference_read_plans(
            compiler, backend, refs, ordinal, expansion, register_roles, deps_by_dst
        )
        # A direct Cell reference can be proven to need a physical-slot read even
        # when its producer is register-owned (for example when the source role or
        # finite ring lag changes across occurrences). Such producer occurrences
        # must therefore be materialized just like Python-boundary/external uses.
        slot_reference_spill_nodes = {
            nid for read in reference_reads if read.source_kind == "slot_reference"
            for nid in read.slot_source_node_ids
        }

        register_values: list[RegisterValuePlan] = []
        for nid in region_node_ids:
            if nid not in register_nodes:
                continue
            role = role_by_node[nid]
            fid = backend.kernel.formula_op_ids[role]
            addr = typed.node_by_id[nid]
            consumers = deps_by_src.get(nid, ())
            reg_c = sum(dst in register_nodes for dst in consumers)
            boundary_c = sum(dst in region_nodes and dst not in register_nodes for dst in consumers)
            external_c = sum(dst not in region_nodes for dst in consumers)
            reasons = []
            if boundary_c: reasons.append("python_or_slot_consumer")
            if external_c: reasons.append("outside_region_consumer")
            if nid in target_ids: reasons.append("target")
            if nid in slot_reference_spill_nodes: reasons.append("slot_reference_consumer")
            register_values.append(
                RegisterValuePlan(
                    node_id=nid,
                    role=role,
                    formula_op_id=fid,
                    pool=addr.pool,
                    slot_offset=addr.offset,
                    spill_required=bool(reasons),
                    spill_reasons=tuple(reasons),
                    internal_register_consumers=reg_c,
                    boundary_consumers=boundary_c,
                    external_consumers=external_c,
                )
            )

        reload_counts: dict[int, int] = defaultdict(int)
        reload_source_kind: dict[int, str] = {}
        for dst in register_nodes:
            for src in deps_by_dst.get(dst, ()):
                if src in register_nodes:
                    continue
                addr = typed.node_by_id.get(src)
                if addr is None or addr.pool == "object":
                    # This should already have demoted the role above.
                    raise OptimizedPlanError(
                        f"register node {dst} depends on non-numeric/non-slot value {src}"
                    )
                reload_counts[src] += 1
                reload_source_kind[src] = "live_in" if src not in region_nodes else "slot_boundary"
        reload_values = tuple(
            ReloadValuePlan(
                node_id=nid,
                pool=typed.node_by_id[nid].pool,
                slot_offset=typed.node_by_id[nid].offset,
                register_consumer_count=count,
                source_kind=reload_source_kind[nid],
            )
            for nid, count in sorted(reload_counts.items())
        )

        position = {nid: i for i, nid in enumerate(region_node_ids)}
        boundaries: list[PythonBoundaryOccurrence] = []
        for nid in region_node_ids:
            role = role_by_node[nid]
            if role in register_roles:
                continue
            spills = tuple(sorted(
                src for src in deps_by_dst.get(nid, ()) if src in register_nodes
            ))
            # Keep exact downstream register consumers as proof/debug metadata.
            # The corresponding ReloadValuePlan says which physical slot holds
            # this boundary-produced value. Future codegen may compact consumers
            # by role/occurrence stream; it must not infer them from source names.
            register_consumers_after = tuple(sorted(
                dst for dst in deps_by_src.get(nid, ()) if dst in register_nodes
            ))
            boundaries.append(
                PythonBoundaryOccurrence(
                    position_in_region=position[nid],
                    node_id=nid,
                    role=role,
                    formula_op_id=backend.kernel.formula_op_ids[role],
                    spill_before=spills,
                    register_consumers_after=register_consumers_after,
                )
            )

        if register_nodes:
            regions.append(
                RegisterRegionPlan(
                    block_index=block_index,
                    family_id=inst.family_id,
                    instance_ordinal=ordinal,
                    variant_ids=tuple(inst.variant_ids),
                    node_ids=region_node_ids,
                    register_roles=tuple(sorted(register_roles)),
                    register_values=tuple(register_values),
                    reload_values=reload_values,
                    boundaries=tuple(boundaries),
                    reference_reads=reference_reads,
                    live_in_node_ids=tuple(live_in),
                    live_out_node_ids=tuple(live_out),
                )
            )
        else:
            reasons = tuple(sorted({
                reason for rp in role_plans for reason in rp.reasons
            })) or ("no_register_native_roles",)
            regions.append(
                SlotRegionPlan(
                    block_index=block_index,
                    family_id=inst.family_id,
                    instance_ordinal=ordinal,
                    variant_ids=tuple(inst.variant_ids),
                    node_ids=region_node_ids,
                    reasons=reasons,
                )
            )

    result = OptimizedExecutionPlan(
        families=tuple(optimized_families),
        regions=tuple(regions),
        canonical_operation_count=compiler.sequential.operation_count,
        register_operation_count=sum(r.register_operation_count for r in regions),
        slot_operation_count=sum(r.slot_operation_count for r in regions),
        python_boundary_count=sum(
            len(r.boundaries) for r in regions if isinstance(r, RegisterRegionPlan)
        ),
        register_value_count=sum(
            len(r.register_values) for r in regions if isinstance(r, RegisterRegionPlan)
        ),
        spill_value_count=sum(
            r.spill_count for r in regions if isinstance(r, RegisterRegionPlan)
        ),
        reload_value_count=sum(
            len(r.reload_values) for r in regions if isinstance(r, RegisterRegionPlan)
        ),
        register_reference_site_count=sum(
            r.register_reference_site_count for r in regions if isinstance(r, RegisterRegionPlan)
        ),
        slot_reference_site_count=sum(
            r.slot_reference_site_count for r in regions if isinstance(r, RegisterRegionPlan)
        ),
    )
    result.validate(compiler, deep_reference_check=False)
    return result
