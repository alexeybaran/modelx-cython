from __future__ import annotations

"""Compact native kernel contract derived from canonical Stage authority.

V02392 deliberately adds no semantic coverage.  It proves that an already-admitted
one-stage StageOptimizedProgram can be physicalized as a compact direct-locals Cython
kernel instead of executing the generic ``_StageKernel`` runtime.  The mature
``OptimizedProgram``/direct-locals emitter is reused *only* as an expression/storage
lowering implementation after the canonical StageExecutionPlan and StageStoragePlan
have granted admission and compatibility has been checked.
"""

from dataclasses import dataclass, field
from typing import Any

from .codegen import CythonGenerator
from .codegen_common import CodegenError
from .optimized_program import OptimizedProgram, build_optimized_program
from .stage_optimized_program import StageOptimizedProgram


class CompactKernelIRError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompactKernelIR:
    schema: str
    uid: str
    stage_program_uid: str
    execution_plan_uid: str
    storage_plan_uid: str
    output_uid: str
    stage_indices: tuple[int, ...]
    scan_directions: tuple[str, ...]
    persistent_value_uids: tuple[str, ...]
    scalar_value_uids: tuple[str, ...]
    reduction_uids: tuple[str, ...]
    input_order: tuple[str, ...]
    blockers: tuple[str, ...] = ()
    optimized_program: OptimizedProgram | None = field(default=None, compare=False, repr=False)

    @property
    def supported(self) -> bool:
        return not self.blockers and self.optimized_program is not None

    def manifest(self) -> dict[str, Any]:
        compatibility = None
        if self.optimized_program is not None and self.optimized_program.stage_storage_compatibility is not None:
            compatibility = self.optimized_program.stage_storage_compatibility.manifest()
        return {
            "schema": self.schema,
            "uid": self.uid,
            "stage_program_uid": self.stage_program_uid,
            "execution_plan_uid": self.execution_plan_uid,
            "storage_plan_uid": self.storage_plan_uid,
            "output_uid": self.output_uid,
            "stage_indices": list(self.stage_indices),
            "scan_directions": list(self.scan_directions),
            "persistent_value_uids": list(self.persistent_value_uids),
            "scalar_value_uids": list(self.scalar_value_uids),
            "reduction_uids": list(self.reduction_uids),
            "input_order": list(self.input_order),
            "blockers": list(self.blockers),
            "supported": self.supported,
            "stage_storage_compatibility": compatibility,
            "physical_backend": "direct_locals_compatibility_bridge_v1",
        }



def build_compact_kernel_ir(program: StageOptimizedProgram, canonical_model) -> CompactKernelIR:
    """Build the V02392 compact-kernel foundation for the proven one-stage subset.

    Admission is intentionally narrow and fail-closed.  This stage is about proving
    the physical architecture, not broadening semantics.
    """
    blockers: list[str] = []
    if not program.direct_cython_supported:
        blockers.append("stage_direct_cython_not_supported")
    if len(program.stages) != 1:
        blockers.append(f"compact_v1_requires_one_stage:{len(program.stages)}")
    if program.barriers:
        blockers.append(f"compact_v1_rejects_barriers:{len(program.barriers)}")
    coordinate_blocks = tuple(x for x in program.execution_blocks if x.kind == "coordinate_scan")
    other_blocks = tuple(x for x in program.execution_blocks if x.kind != "coordinate_scan")
    if len(coordinate_blocks) != 1 or other_blocks:
        blockers.append(
            "compact_v1_requires_one_coordinate_scan:" +
            f"coordinate={len(coordinate_blocks)} other={len(other_blocks)}"
        )
    if coordinate_blocks and coordinate_blocks[0].scan_direction not in {"ascending", "descending"}:
        blockers.append(f"compact_v1_unknown_scan_direction:{coordinate_blocks[0].scan_direction}")
    if any(x.materialized_slot is not None for x in program.physical_values):
        blockers.append("compact_v1_rejects_cross_stage_materialization")
    allowed_working = {"none", "scalar", "current", "ring", "history"}
    bad_working = sorted({x.working_kind for x in program.physical_values if x.working_kind not in allowed_working})
    if bad_working:
        blockers.append(f"compact_v1_unknown_storage_kind:{','.join(bad_working)}")

    optimized: OptimizedProgram | None = None
    if not blockers:
        canonical = canonical_model
        if canonical is None:
            blockers.append("compact_v1_missing_canonical_model_authority")
        elif canonical.stage_execution_plan is not program.execution_plan or canonical.stage_storage_plan is not program.storage_plan:
            blockers.append("compact_v1_canonical_stage_authority_mismatch")
        else:
            try:
                optimized = build_optimized_program(canonical, optimization_level="O2")
            except Exception as exc:  # normalize legacy lowering failures to typed IR blocker
                blockers.append(f"compact_v1_physical_bridge_failed:{type(exc).__name__}:{exc}")
            else:
                if tuple(optimized.input_order) != tuple(program.input_order):
                    blockers.append("compact_v1_input_order_mismatch")
                    optimized = None
                else:
                    compat = optimized.stage_storage_compatibility
                    if compat is None or not compat.compatible:
                        reasons = () if compat is None else compat.reasons
                        blockers.append("compact_v1_stage_storage_incompatible:" + ";".join(reasons))
                        optimized = None

    block = coordinate_blocks[0] if coordinate_blocks else None
    uid = "compact_kernel_" + program.uid.split("_")[-1]
    return CompactKernelIR(
        schema="modelx_graph.compact_kernel_ir.v1",
        uid=uid,
        stage_program_uid=program.uid,
        execution_plan_uid=program.execution_plan.uid,
        storage_plan_uid=program.storage_plan.uid,
        output_uid=program.output_uid,
        stage_indices=tuple(program.stage_indices),
        scan_directions=(() if block is None else (block.scan_direction,)),
        persistent_value_uids=(() if block is None else tuple(block.persistent_driver_uids)),
        scalar_value_uids=(() if block is None else tuple(block.pre_scalar_uids)),
        reduction_uids=(() if block is None else tuple(block.reduction_uids)),
        input_order=tuple(program.input_order),
        blockers=tuple(blockers),
        optimized_program=optimized,
    )


class CompactKernelCythonGenerator:
    """Emit a compact direct-locals Cython kernel from :class:`CompactKernelIR`."""

    def __init__(self, ir: CompactKernelIR):
        self.ir = ir
        if not ir.supported or ir.optimized_program is None:
            raise CodegenError("CompactKernelIR is unsupported: " + "; ".join(ir.blockers))

    def source(self, module_name: str = "compact_stage_kernel") -> str:
        source = CythonGenerator(
            self.ir.optimized_program, emission_mode="direct_locals"
        ).emit(module_name)
        forbidden = ("class _StageKernel", "_IMPLS", "_PHYSICAL", "globals()", "_ACTIVE")
        hits = tuple(token for token in forbidden if token in source)
        if hits:
            raise CodegenError(f"compact kernel source leaked Stage runtime tokens: {hits!r}")
        if "cdef inline double run_one(" not in source or "noexcept nogil" not in source:
            raise CodegenError("compact kernel source lacks native direct-locals point kernel")
        return source

    def write(self, path, module_name: str = "compact_stage_kernel") -> str:
        from pathlib import Path
        text = self.source(module_name)
        Path(path).write_text(text)
        return text
