from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .loop_recovery import LiteralBlock
from .native_bindings import NativeBindingError, build_runtime_arg_sites
from .native_family import _exact_native_formula_ok, _typed_slot_plan
from .native_plan import (
    NativeFamilyBackendPlan,
    NativePlanError,
    build_native_family_backend_plan,
    expand_native_family_instance,
)
from .native_references import NativeReferenceError, build_reference_binding_plan


@dataclass(frozen=True)
class ArtifactFamilyPlan:
    family_id: int
    code_signature: str | None
    variant_count: int
    instance_count: int
    native_formula_ops: int
    python_formula_ops: int
    reference_site_count: int
    fallback_reasons: tuple[str, ...]
    role_native: tuple[bool, ...]


@dataclass(frozen=True)
class ArtifactRegionPlan:
    block_index: int
    kind: str
    backend: str
    operation_count: int
    native_operations: int
    python_operations: int
    family_id: int | None = None
    family_instance_ordinal: int | None = None
    fallback_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class WholeArtifactBackendPlan:
    regions: tuple[ArtifactRegionPlan, ...]
    families: tuple[ArtifactFamilyPlan, ...]
    total_operations: int
    native_operations: int
    python_operations: int
    python_callback_operations: int
    double_slots: int
    int_slots: int
    bool_slots: int
    object_slots: int

    @property
    def weighted_native_fraction(self) -> float:
        return self.native_operations / self.total_operations if self.total_operations else 0.0

    @property
    def fallback_region_count(self) -> int:
        return sum(region.backend == "python" for region in self.regions)

    @property
    def mixed_region_count(self) -> int:
        return sum(region.backend == "mixed" for region in self.regions)

    @property
    def native_region_count(self) -> int:
        return sum(region.backend == "native" for region in self.regions)


def _family_plan(
    compiler: Any,
    family_id: int,
    typed: Any,
    analysis_by_id: dict[int, Any],
) -> tuple[ArtifactFamilyPlan, NativeFamilyBackendPlan | None]:
    try:
        backend = build_native_family_backend_plan(compiler, family_id)
        sites = build_runtime_arg_sites(backend)
        formula_ids = tuple(backend.kernel.formula_op_ids)
        native = {
            fid: _exact_native_formula_ok(compiler, fid, analysis_by_id[fid])
            for fid in set(formula_ids)
        }
        # A FormulaOp shared by several roles must obey the strictest formal ABI.
        for role, fid in enumerate(formula_ids):
            if any(site.role == role and site.cython_type == "object" for site in sites):
                native[fid] = False
        references = build_reference_binding_plan(compiler, typed, backend, native)
        for fid in references.unsupported_formula_ops:
            native[fid] = False
        role_native = tuple(bool(native[fid]) for fid in formula_ids)
        reasons = tuple(reason for _fid, reason in references.fallback_reasons)
        return (
            ArtifactFamilyPlan(
                family_id=family_id,
                code_signature=backend.kernel.code_signature,
                variant_count=backend.kernel.variant_count,
                instance_count=len(backend.instances),
                native_formula_ops=sum(native[fid] for fid in set(formula_ids)),
                python_formula_ops=sum(not native[fid] for fid in set(formula_ids)),
                reference_site_count=len(references.sites),
                fallback_reasons=reasons,
                role_native=role_native,
            ),
            backend,
        )
    except (NativePlanError, NativeBindingError, NativeReferenceError, KeyError, ValueError) as exc:
        # Planning is deliberately fail-closed at the family boundary. A whole
        # artifact may still execute this region through DirectSlot-compatible
        # Python rather than rejecting an otherwise valid model.
        plan = compiler.structured.canonical_plan
        assert plan is not None
        family = next(f for f in plan.code_families if f.family_id == family_id)
        role_native = (False,) * len(family.formula_op_ids)
        return (
            ArtifactFamilyPlan(
                family_id=family_id,
                code_signature=None,
                variant_count=len(family.grammar.variant_programs),
                instance_count=sum(i.family_id == family_id for i in plan.loop_instances),
                native_formula_ops=0,
                python_formula_ops=len(set(family.formula_op_ids)),
                reference_site_count=0,
                fallback_reasons=(f"family_planning:{type(exc).__name__}:{exc}",),
                role_native=role_native,
            ),
            None,
        )


def build_whole_artifact_backend_plan(compiler: Any) -> WholeArtifactBackendPlan:
    """Plan complete canonical execution without yet generating a whole extension.

    This is intentionally a composition foundation. Every canonical block is
    represented in exact execution order. Recovered family blocks are classified
    using the same runtime-binding/reference guards as the Stage 4b.2 kernel;
    literal blocks remain explicit Python fallback until literal lowering is
    implemented. One unsupported family never invalidates the whole plan.
    """
    if compiler.structured is None:
        compiler.recover_loops()
    if compiler.slots is None:
        compiler.optimize_storage(); compiler.lower_slots()
    analysis = compiler.native_plan or compiler.analyze_native()
    typed = _typed_slot_plan(compiler, analysis, include_objects=True)
    structured = compiler.structured
    plan = structured.canonical_plan
    assert plan is not None
    analysis_by_id = {row.formula_op_id: row for row in analysis.formulae}

    family_rows: dict[int, ArtifactFamilyPlan] = {}
    family_backends: dict[int, NativeFamilyBackendPlan | None] = {}
    for family in plan.code_families:
        row, backend = _family_plan(compiler, family.family_id, typed, analysis_by_id)
        family_rows[family.family_id] = row
        family_backends[family.family_id] = backend

    # The canonical plan stores block -> global LoopInstancePlan index. Convert to
    # the ordinal used by each shared FamilyKernelPlan exactly once.
    family_instance_ordinal: dict[tuple[int, int], int] = {}
    for family_id, backend in family_backends.items():
        if backend is None:
            continue
        for ordinal, inst in enumerate(backend.instances):
            family_instance_ordinal[(family_id, inst.block_index)] = ordinal

    regions: list[ArtifactRegionPlan] = []
    for block_index, block in enumerate(structured.blocks):
        instance_index = plan.block_to_instance[block_index]
        if instance_index is None or isinstance(block, LiteralBlock):
            regions.append(
                ArtifactRegionPlan(
                    block_index=block_index,
                    kind="literal",
                    backend="python",
                    operation_count=block.operation_count,
                    native_operations=0,
                    python_operations=block.operation_count,
                    fallback_reasons=("literal_lowering_pending",),
                )
            )
            continue

        inst = plan.loop_instances[instance_index]
        fplan = family_rows[inst.family_id]
        backend = family_backends[inst.family_id]
        if backend is None:
            regions.append(
                ArtifactRegionPlan(
                    block_index=block_index,
                    kind="family",
                    backend="python",
                    operation_count=block.operation_count,
                    native_operations=0,
                    python_operations=block.operation_count,
                    family_id=inst.family_id,
                    fallback_reasons=fplan.fallback_reasons,
                )
            )
            continue

        ordinal = family_instance_ordinal[(inst.family_id, block_index)]
        expansion = expand_native_family_instance(compiler, backend, ordinal)
        native_ops = sum(
            1 for role, _nid in expansion.operation_sequence if fplan.role_native[role]
        )
        python_ops = expansion.operation_count - native_ops
        if native_ops == expansion.operation_count:
            region_backend = "native"
        elif native_ops:
            region_backend = "mixed"
        else:
            region_backend = "python"
        regions.append(
            ArtifactRegionPlan(
                block_index=block_index,
                kind="family",
                backend=region_backend,
                operation_count=expansion.operation_count,
                native_operations=native_ops,
                python_operations=python_ops,
                family_id=inst.family_id,
                family_instance_ordinal=ordinal,
                fallback_reasons=fplan.fallback_reasons if python_ops else (),
            )
        )

    total = sum(row.operation_count for row in regions)
    native = sum(row.native_operations for row in regions)
    python = sum(row.python_operations for row in regions)
    if total != structured.operation_count or native + python != total:
        raise RuntimeError("whole-artifact backend plan does not cover canonical execution exactly")

    return WholeArtifactBackendPlan(
        regions=tuple(regions),
        families=tuple(family_rows[k] for k in sorted(family_rows)),
        total_operations=total,
        native_operations=native,
        python_operations=python,
        python_callback_operations=python,
        double_slots=typed.double_count,
        int_slots=typed.int_count,
        bool_slots=typed.bool_count,
        object_slots=typed.object_count,
    )
