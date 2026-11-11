from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from .loop_recovery import LoopBlock, PhaseLoopBlock, StructuredSequentialProgram, recover_loops
from .realized_trace import RealizedTrace, capture_realized_trace
from .sequential_ir import SequentialProgram, build_sequential_program
from .storage import StorageOptimizedProgram, optimize_storage
from .slot_storage import SlotStorageProgram, DirectSlotProgram, lower_to_slots, lower_to_direct_slots
from .native_analysis import NativePlanAnalysis, analyze_native_capability


@dataclass
class RealizedTraceCompiler:
    """Exact realized Stage-1/2 compiler with hierarchical Stage-2 recovery."""

    trace: RealizedTrace
    sequential: SequentialProgram
    structured: StructuredSequentialProgram | None = None
    storage: StorageOptimizedProgram | None = None
    slots: SlotStorageProgram | None = None
    direct_slots: DirectSlotProgram | None = None
    native_plan: NativePlanAnalysis | None = None
    optimized_plan: Any | None = None
    frozen_reference_plan: Any | None = None
    frozen_reference_prepare_seconds: float = 0.0
    final_profile: Any | None = None

    @classmethod
    def from_target(cls, model: Any, targets: Any, *, clear_first: bool = True) -> "RealizedTraceCompiler":
        trace = capture_realized_trace(model, targets, clear_first=clear_first)
        sequential = build_sequential_program(trace)
        return cls(trace=trace, sequential=sequential)

    def recover_loops(self, **kwargs) -> StructuredSequentialProgram:
        self.structured = recover_loops(self.sequential, **kwargs)
        return self.structured

    def run_sequential(self, *, validate: bool = True, direct: bool = True):
        return self.sequential.execute(clear_first=True, validate=validate, direct=direct)

    def run_structured(self, *, validate: bool = True, direct: bool = True):
        if self.structured is None:
            self.recover_loops()
        assert self.structured is not None
        return self.structured.execute(clear_first=True, validate=validate, direct=direct)

    def optimize_storage(self) -> StorageOptimizedProgram:
        if self.structured is None:
            self.recover_loops()
        assert self.structured is not None
        self.storage = optimize_storage(self.structured)
        return self.storage

    def run_storage(self, *, validate: bool = True, direct: bool = True):
        if self.storage is None:
            self.optimize_storage()
        assert self.storage is not None
        return self.storage.execute(clear_first=True, validate=validate, direct=direct)

    def lower_slots(self) -> SlotStorageProgram:
        if self.storage is None:
            self.optimize_storage()
        assert self.storage is not None
        self.slots = lower_to_slots(self.storage.structured, self.storage.plan)
        return self.slots

    def run_slots(self, *, validate: bool = True):
        if self.slots is None:
            self.lower_slots()
        assert self.slots is not None
        return self.slots.execute(validate=validate)

    def lower_direct_slots(self) -> DirectSlotProgram:
        if self.slots is None:
            self.lower_slots()
        assert self.slots is not None
        self.direct_slots = lower_to_direct_slots(self.slots)
        return self.direct_slots

    def run_direct_slots(self, *, validate: bool = True):
        if self.direct_slots is None:
            self.lower_direct_slots()
        assert self.direct_slots is not None
        return self.direct_slots.execute(validate=validate)

    def analyze_native(self) -> NativePlanAnalysis:
        self.native_plan = analyze_native_capability(self)
        return self.native_plan

    def build_native_family(self, family_id: int, build_dir: Any):
        """Compile one fully native recovered loop family against physical slots."""
        from .native_family import build_native_family_cython
        return build_native_family_cython(self, family_id, build_dir)

    def build_mixed_native_family(self, family_id: int, build_dir: Any):
        """Compile one recovered family with explicit numeric Python FormulaOp islands."""
        from .native_family import build_mixed_native_family_cython
        return build_mixed_native_family_cython(self, family_id, build_dir)

    def build_whole_artifact(
        self, build_dir: Any, *, optimization: str = "O1", experimental_allow_pow: bool = False,
        experimental_register_regions: bool = False,
        experimental_register_region_blocks: tuple[int, ...] | None = None,
        experimental_frozen_references: bool = False,
        experimental_literal_lowering: bool = True,
        execution_profile: str = "manual",
    ):
        """Compile one non-recursive sequential whole-artifact Cython driver.

        ``experimental_allow_pow`` exists only for the Part-1 exactness study;
        default behavior retains the proven conservative power guard.
        """
        from .whole_artifact import build_whole_artifact_cython
        if experimental_frozen_references:
            self.plan_frozen_numeric_references()
            # Register ownership depends on which Python FormulaOps were promoted.
            self.optimized_plan = None
        return build_whole_artifact_cython(
            self, build_dir, optimization=optimization,
            experimental_allow_pow=experimental_allow_pow,
            experimental_register_regions=experimental_register_regions,
            experimental_register_region_blocks=experimental_register_region_blocks,
            experimental_frozen_references=experimental_frozen_references,
            experimental_literal_lowering=experimental_literal_lowering,
            execution_profile=execution_profile,
        )

    def profile_final_backend(self):
        """Choose the frozen pre-benchmark backend profile from realized target shape only."""
        from .final_backend import choose_final_backend_profile
        self.final_profile = choose_final_backend_profile(self.trace.target_values)
        return self.final_profile

    def build_final_artifact(
        self, build_dir: Any, *, optimization: str = "O1", prove: bool = True,
        detach: bool = False, trim_allocator: bool = False,
    ):
        """Build the final pre-benchmark artifact using structural profile selection.

        Vector targets preserve NumPy/pandas vector FormulaOps and optimize the
        canonical schedule/cache ownership only. Scalar targets enable only the
        already-proven frozen-reference and conservative RegisterRegion passes.
        Power remains disabled.  Optional proof and detachment make the split
        between correctness evidence and production memory ownership explicit.
        """
        profile = self.profile_final_backend()
        program = self.build_whole_artifact(
            build_dir,
            optimization=optimization,
            experimental_allow_pow=profile.allow_pow,
            experimental_register_regions=profile.use_register_regions,
            experimental_frozen_references=profile.use_frozen_numeric_references,
            execution_profile=profile.execution_profile.value,
        )
        if prove:
            program.execute(validate=True, prove_no_modelx=True)
        if detach:
            program.detach_for_production(
                require_proof=True, trim_allocator=trim_allocator
            )
        return program

    def plan_frozen_numeric_references(self):
        """Freeze exact numeric ndarray/pandas lookup references for the current realized artifact."""
        from .frozen_references import build_frozen_numeric_reference_plan
        t0 = time.perf_counter()
        self.frozen_reference_plan = build_frozen_numeric_reference_plan(self)
        self.frozen_reference_prepare_seconds = time.perf_counter() - t0
        return self.frozen_reference_plan

    def plan_optimized_execution(self):
        """Derive and validate the canonical register/slot ownership IR.

        This is an optimization plan only. It never replaces or mutates the
        canonical execution plan or existing fallback backends.
        """
        from .optimized_execution import build_optimized_execution_plan
        self.optimized_plan = build_optimized_execution_plan(self)
        return self.optimized_plan

    def build_register_region(
        self, block_index: int, build_dir: Any, *, optimization: str = "O2"
    ):
        """Build a canonical RegisterRegion proof; mixed numeric Python boundaries are supported when proven."""
        from .register_region import build_register_region_cython
        return build_register_region_cython(
            self, block_index, build_dir, optimization=optimization
        )

    def plan_fast_foundation(self):
        """Analyze the experimental fixed-graph register/snapshot fast path."""
        from .fast_foundation import plan_fast_foundation
        return plan_fast_foundation(self)

    def build_fast_foundation(self, build_dir: Any, *, native_arch: bool = False):
        """Build the Part-1 experimental serial batch kernel.

        Failure means only that the fast optimization is inapplicable; the normal
        whole-artifact path remains the correctness backend.
        """
        from .fast_foundation import build_fast_foundation
        return build_fast_foundation(self, build_dir, native_arch=native_arch)

    def prepare_optimized_scalar(self, **kwargs):
        """Compatibility entry point for the mature comparison scalar program."""
        from .native_batch import prepare_optimized_scalar
        return prepare_optimized_scalar(self, **kwargs)

    def prepare_legacy_comparison_scalar(self, **kwargs):
        """Explicitly build the mature ExecutableGraph/OptimizedProgram comparison oracle."""
        from .native_batch import prepare_legacy_comparison_scalar
        return prepare_legacy_comparison_scalar(self, **kwargs)

    def prepare_stage_scalar(self, **kwargs):
        """Build and modelx-validate the production StageOptimizedProgram."""
        from .native_batch import prepare_stage_scalar
        return prepare_stage_scalar(self, **kwargs)

    def prepare_stage_scalar_auto_trace(self, **kwargs):
        """Close representative trace coverage automatically over a finite run domain."""
        from .native_batch import prepare_stage_scalar_auto_trace
        return prepare_stage_scalar_auto_trace(self, **kwargs)

    def plan_native_batch(self, **kwargs):
        """Plan the detached StageOptimizedProgram production NativeBatch backend."""
        from .native_batch import plan_native_batch
        return plan_native_batch(self, **kwargs)

    def build_optimized_python_batch(self, build_dir: Any, **kwargs):
        """Emit the standalone optimized-Python artifact before Cython compilation."""
        from .native_batch import build_optimized_python_batch
        return build_optimized_python_batch(self, build_dir, **kwargs)

    def build_native_batch(self, build_dir: Any, **kwargs):
        """Build the direct serial portfolio kernel without modelx/modelx-cython runtime."""
        from .native_batch import build_native_batch
        return build_native_batch(self, build_dir, **kwargs)

    def build_native_batch_auto_trace(self, build_dir: Any, **kwargs):
        """Build a complete-domain native batch after automatic trace closure."""
        from .native_batch import build_native_batch_auto_trace
        return build_native_batch_auto_trace(self, build_dir, **kwargs)

    def build_performance_artifact(self, build_dir: Any, **kwargs):
        """Build a measured exact backend with modelx-cython as the floor/fallback."""
        from .performance_backend import build_performance_artifact
        return build_performance_artifact(self, build_dir, **kwargs)

    def manifest(self) -> dict[str, Any]:
        out = {
            "mode": "realized_trace_only",
            "calculated_nodes": self.trace.calculated_count,
            "dependency_edges": len(self.trace.dependencies),
            "targets": len(self.trace.target_runtime_nodes),
            "diagnostics": list(self.trace.diagnostics),
        }
        if self.structured is not None:
            local = [b for b in self.structured.blocks if isinstance(b, LoopBlock)]
            phases = [b for b in self.structured.blocks if isinstance(b, PhaseLoopBlock)]
            out.update(
                recovery_mode=self.structured.recovery_mode,
                block_count=len(self.structured.blocks),
                loop_count=self.structured.loop_count,
                local_loop_count=self.structured.local_loop_count,
                phase_loop_count=self.structured.phase_loop_count,
                phase_variant_count=self.structured.phase_variant_count,
                stored_body_ops=self.structured.stored_body_ops,
                compression_ratio=self.structured.compression_ratio,
                control_items=self.structured.control_items,
                binding_payload_items=self.structured.binding_payload_items,
                binding_kinds=dict(self.structured.binding_kind_counts()),
                representation_items=self.structured.representation_items,
                representation_compression_ratio=self.structured.representation_compression_ratio,
                loop_family_count=len(self.structured.loop_families),
                family_code_ops=self.structured.family_code_ops,
                family_code_compression_ratio=self.structured.family_code_compression_ratio,
                local_loops=[
                    {
                        "start": b.start,
                        "body_size": b.body_size,
                        "repetitions": b.repetitions,
                        "saved_ops": b.saved_ops,
                        "dependency_variant_count": b.dependency_variant_count,
                        "bindings": [
                            {
                                "object": op.obj_binding.kind,
                                "args": [x.kind for x in op.arg_bindings],
                            }
                            for op in b.body
                        ],
                    }
                    for b in local
                ],
                phase_loops=[
                    {
                        "start": b.start,
                        "stop": b.stop,
                        "repetitions": b.repetitions,
                        "phase_variants": b.phase_count,
                        "terminal_roles": b.grammar.terminal_count,
                        "grammar_rules": len(b.grammar.rules),
                        "grammar_symbols": b.grammar.stored_symbols,
                        "stored_body_ops": b.stored_body_ops,
                        "terminal_callsite_max": b.grammar_terminal_callsite_max,
                    }
                    for b in phases
                ],
            )
        if self.storage is not None:
            sp = self.storage.plan
            out.update(
                storage_peak_live_values=sp.peak_live_values,
                storage_baseline_values=sp.baseline_scheduled_values,
                storage_cache_entry_reduction=sp.exact_cache_entry_reduction,
                storage_baseline_shallow_bytes=sp.baseline_shallow_value_bytes,
                storage_peak_shallow_bytes=sp.peak_live_shallow_value_bytes,
                storage_shallow_value_reduction=sp.exact_shallow_value_reduction,
                storage_classes=dict(sp.storage_class_counts()),
                storage_formula_specs=[
                    {
                        "formula_op_id": x.formula_op_id,
                        "name": x.name,
                        "produced": x.produced_count,
                        "required_slots": x.required_slots,
                        "storage_class": x.storage_class,
                        "ring_compatible": x.ring_compatible,
                        "max_future_span_ops": x.max_future_span_ops,
                    }
                    for x in sp.formula_specs
                ],
            )
        if self.slots is not None:
            out.update(
                physical_slot_count=self.slots.layout.total_slots,
                physical_slot_reduction=self.slots.layout.physical_slot_reduction,
                physical_slot_formulas=[
                    {
                        "formula_op_id": x.formula_op_id,
                        "name": x.name,
                        "storage_class": x.storage_class,
                        "produced": x.produced_count,
                        "slots": x.slot_count,
                    }
                    for x in self.slots.layout.formula_layouts
                ],
            )
        if self.direct_slots is not None:
            out.update(direct_slot_lowering=True)
        if self.native_plan is not None:
            out.update(
                native_formula_ops=self.native_plan.native_formula_ops,
                native_executed_ops=self.native_plan.native_executed_ops,
                native_weighted_fraction=self.native_plan.weighted_native_fraction,
                native_reason_counts=dict(self.native_plan.reason_counts),
            )
        if self.optimized_plan is not None:
            out.update(optimized_execution=self.optimized_plan.to_dict())
        return out
