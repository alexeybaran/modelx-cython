from .normalized_operator import NormalizedOperator
from .semantic_ir import AsofLookupIR, SemanticFormulaIR, SemanticPassRecord, FormulaNormalizationResult
from .lookup_semantics import recognize_asof_lookups
from .semantic_passes import AstSemanticPass, SemanticPassPipeline, SemanticPassError
from .invocation import OutputInvocation, RunDomain, ResultDomain
from ._version import __version__
from .trace import capture_trace, TraceCapture, TraceError
from .ir import build_graph_ir, GraphIR, ConcreteValue, IRDependency, LoopTemplate, GuardSpec, IRRegion
from .frontend import GraphModelCompiler, CanonicalModel, CanonicalProofSnapshot, capture_canonical_proof_snapshot, FrontendError
from .codegen import CythonGenerator, CodegenError
from .optimized_program import OptimizedProgram, OptimizedProgramError, build_optimized_program
from .program_inputs import NormalizedPointRowSource, NormalizedLookup1D
from .program_semantics import GuardSpec, StaticScalarFact, ValidatedStaticFact, FiniteDomainFact, FixedCoordinateFact, RuntimeScalarHelperFact, ProgramSemanticError
from .optimized_python_artifact import (
    OptimizedPythonArtifact, OptimizedPythonArtifactError, build_optimized_python_artifact,
)
from .build import build_extension
from .safety import validate_formula_control_flow, SemanticSafetyError
from .python_codegen import PythonLoopGenerator
from .template_ir import ExecutableGraph, ExecutableLoopTemplate, TemplateStoragePlan, TemplateError
from .canonical_semantic_graph import (
    CanonicalLoopTopologyFact, RecoveredLoopTopologyFact, CanonicalSemanticNode,
    CanonicalSemanticAccess, CanonicalOperatorUse, CanonicalSemanticGraph,
    StateFreeCoordinateExecutionProof, CanonicalRecurrenceDomainFact,
    FormulaSchema, DomainFact, CoordinateRelation, SemanticOperator,
    CanonicalFootprintPath, CanonicalStateFootprintEvidence, CanonicalTransitionEvidence,
    freeze_graph_ir_loop_topology, freeze_recovered_loop_topology,
    build_canonical_semantic_graph,
)
from .canonical_schedule import (
    CanonicalExecutionNodeFact, CanonicalExecutionComponent, CanonicalExecutionStage,
    CanonicalExecutionSchedule, CanonicalComponentScheduleAnalysis,
    analyze_canonical_component_schedule, build_canonical_execution_schedule,
)
from .stage_execution_plan import (
    StageExecutionPlanError, ExecutionDomain, ExecutionTask, ExecutionBarrier,
    ValueLifetime, ExecutionStage, StageExecutionPlan, build_stage_execution_plan,
)
from .stage_storage_plan import (
    StageStoragePlanError, StageStorageValue, StageStorageFenceValue, StageStorageFence,
    StageStorageKernel, StageStorageStage,
    StageStoragePlan, StageStorageCompatibility, build_stage_storage_plan,
    validate_template_storage_compatibility,
)
from .stage_optimized_program import (
    StageOptimizedProgramError, StagePhysicalValue, StageProgramKernel, StageProgramStage,
    StageProgramBarrier, StageOperatorRequirement, StageRuntimePayload, StageOptimizedProgram,
    build_stage_optimized_program, build_stage_optimized_program_from_model, decode_runtime_payload,
)
from .stage_physical_lowering import (
    StageIterationDomain, StageBoundarySeed, StagePureMapABI, StageAuxiliaryRecurrenceABI,
    StageExecutionBlock, StageDirectContract, build_one_stage_direct_contract, build_stage_direct_contract,
)
from .stage_codegen import StagePythonGenerator, StageCythonGenerator
from .compact_kernel_ir import (
    CompactKernelIRError, CompactKernelIR, build_compact_kernel_ir,
    CompactKernelCythonGenerator,
)
from .domain_graph import (
    DomainGraph, DomainGraphError, DomainAxis, IterationDomain, Kernel,
    AccessRelation, InitialCondition, PurityEvidence, ValueSemanticsEvidence, SourceGraphCandidate,
    SourceAccessEvidence, SourcePersistentComponent, SourceStateStructure,
    SourceCallSiteAvailability, SourceCallSiteSummary, SourceAvailabilityPlan,
    DomainAccessAvailabilityEvidence,
    ComponentScheduleComponent, CrossComponentOrderProof, StageScanOrderProof, ComponentScheduleConstraint, DerivedScheduleTask,
    ComponentScheduleStage, ComponentSchedulePlan,
    CanonicalOverlayFact, CanonicalOverlayApplication, CanonicalOverlaySchedulePlan,
    SourcePhaseComponent, SourcePhaseConstraint, SourcePhase, SourcePhasePlan,
    build_shadow_domain_graph, classify_domain_graph_purity, classify_domain_graph_value_semantics, with_domain_graph_purity,
    classify_source_purity, classify_source_value_semantics, analyze_source_state_structure,
    analyze_source_callsite_availability, classify_domain_graph_access_availability,
    analyze_source_component_schedule, analyze_domain_graph_component_schedule,
    apply_canonical_evidence_overlay, analyze_source_component_schedule_with_canonical_overlay,
    analyze_source_phase_constraints, scan_source_graph_candidates,
    scan_source_graph_candidates_path,
)
from .passes import PassReport, ReductionFusionPass, reports_for_level
from .dispatcher import (
    GuardDecision, GuardStats, GuardedExecutor, ArtifactRecord,
    ArtifactManagerStats, VariantArtifactManager, artifact_cache_key,
)
from .numeric_view import (
    NumericViewError, NumericViewReport, formula_fingerprint, formula_fingerprints,
    verify_numeric_view, require_numeric_view,
)
from .mixed import (
    AutomaticMixedCompiler, AutomaticMixedRuntime, MixedPartitionPlan, MixedRegion,
    BoundaryFieldSpec, BoundaryBatch, PartitionError, build_mixed_partition,
)

__all__ = [
    '__version__',
    'capture_trace','TraceCapture','TraceError','build_graph_ir','GraphIR','ConcreteValue','IRDependency','LoopTemplate','GuardSpec','IRRegion',
    'GraphModelCompiler','CanonicalModel','CanonicalProofSnapshot','capture_canonical_proof_snapshot','FrontendError','OutputInvocation','RunDomain','ResultDomain','CythonGenerator','CodegenError','build_extension',
    'validate_formula_control_flow','SemanticSafetyError','PythonLoopGenerator',
    'OptimizedProgram','OptimizedProgramError','build_optimized_program','NormalizedPointRowSource','NormalizedLookup1D','GuardSpec','StaticScalarFact','ValidatedStaticFact','FiniteDomainFact','FixedCoordinateFact','RuntimeScalarHelperFact','ProgramSemanticError',
    'OptimizedPythonArtifact','OptimizedPythonArtifactError','build_optimized_python_artifact',
    'ExecutableGraph','ExecutableLoopTemplate','TemplateStoragePlan','TemplateError',
    'CanonicalLoopTopologyFact','RecoveredLoopTopologyFact','CanonicalSemanticNode','CanonicalSemanticAccess','CanonicalOperatorUse','CanonicalSemanticGraph','StateFreeCoordinateExecutionProof','CanonicalRecurrenceDomainFact','FormulaSchema','DomainFact','CoordinateRelation','SemanticOperator','freeze_graph_ir_loop_topology','freeze_recovered_loop_topology','build_canonical_semantic_graph',
    'CanonicalFootprintPath','CanonicalStateFootprintEvidence','CanonicalTransitionEvidence','CanonicalExecutionNodeFact','CanonicalExecutionComponent','CanonicalExecutionStage','CanonicalExecutionSchedule','CanonicalComponentScheduleAnalysis','analyze_canonical_component_schedule','build_canonical_execution_schedule',
    'StageExecutionPlanError','ExecutionDomain','ExecutionTask','ExecutionBarrier','ValueLifetime','ExecutionStage','StageExecutionPlan','build_stage_execution_plan','StageStoragePlanError','StageStorageValue','StageStorageFenceValue','StageStorageFence','StageStorageKernel','StageStorageStage','StageStoragePlan','StageStorageCompatibility','build_stage_storage_plan','validate_template_storage_compatibility','StageOptimizedProgramError','StagePhysicalValue','StageProgramKernel','StageProgramStage','StageProgramBarrier','StageOperatorRequirement','StageRuntimePayload','StageOptimizedProgram','build_stage_optimized_program','build_stage_optimized_program_from_model','decode_runtime_payload','StageIterationDomain','StageBoundarySeed','StagePureMapABI','StageAuxiliaryRecurrenceABI','StageExecutionBlock','StageDirectContract','build_one_stage_direct_contract','build_stage_direct_contract','StagePythonGenerator','StageCythonGenerator','CompactKernelIRError','CompactKernelIR','build_compact_kernel_ir','CompactKernelCythonGenerator',
    'DomainGraph','DomainGraphError','DomainAxis','IterationDomain','Kernel','AccessRelation','InitialCondition','PurityEvidence','ValueSemanticsEvidence','SourceGraphCandidate','SourceAccessEvidence','SourcePersistentComponent','SourceStateStructure','SourceCallSiteAvailability','SourceCallSiteSummary','SourceAvailabilityPlan','DomainAccessAvailabilityEvidence','ComponentScheduleComponent','CrossComponentOrderProof','StageScanOrderProof','ComponentScheduleConstraint','DerivedScheduleTask','ComponentScheduleStage','ComponentSchedulePlan','CanonicalOverlayFact','CanonicalOverlayApplication','CanonicalOverlaySchedulePlan','SourcePhaseComponent','SourcePhaseConstraint','SourcePhase','SourcePhasePlan',
    'build_shadow_domain_graph','classify_domain_graph_purity','classify_domain_graph_value_semantics','with_domain_graph_purity','classify_source_purity','classify_source_value_semantics','analyze_source_state_structure','analyze_source_callsite_availability','classify_domain_graph_access_availability','analyze_source_component_schedule','analyze_domain_graph_component_schedule','apply_canonical_evidence_overlay','analyze_source_component_schedule_with_canonical_overlay','analyze_source_phase_constraints','scan_source_graph_candidates','scan_source_graph_candidates_path',
    'PassReport','ReductionFusionPass','reports_for_level',
    'NumericViewError','NumericViewReport','formula_fingerprint','formula_fingerprints','verify_numeric_view','require_numeric_view',
    'GuardDecision','GuardStats','GuardedExecutor','ArtifactRecord','ArtifactManagerStats','VariantArtifactManager','artifact_cache_key',
    'AutomaticMixedCompiler','AutomaticMixedRuntime','MixedPartitionPlan','MixedRegion','BoundaryFieldSpec','BoundaryBatch','PartitionError','build_mixed_partition',
    "LegacyComparisonError", "LegacyComparisonBundle", "build_legacy_comparison_bundle",
]

# v0.13 exact realized-trace Stage-1/2 path with executable canonical loop families.  Kept separate from the legacy
# semantic/native frontend so arbitrary Python/object formulas can participate in
# de-recursion and loop recovery before any lowering decision is made.
from .realized_trace import RealizedTrace, TraceEvent, ConcreteNode, RealizedTraceError, capture_realized_trace
from .sequential_ir import SequentialProgram, EvalOp, ReplayValidation, SequentialError, build_sequential_program
from .loop_recovery import (
    LoopRecoveryError, ConstantBinding, AffineIntBinding, ArithmeticRunsIntBinding, PeriodicBinding, InterleavedAffineIntBinding, RunLengthBinding, TableBinding,
    EvalTemplate, LiteralBlock, LoopBlock, PhaseGrammar, PhaseLoopBlock, LoopFamilyTemplate, FormulaOp, RoleBindingTemplate, LoopCodeFamily, LoopInstancePlan, CanonicalExecutionPlan, StructuredSequentialProgram,
    infer_binding, binding_payload_items, recover_loops,
)
from .realized_compiler import RealizedTraceCompiler

__all__ += [
    'RealizedTrace','TraceEvent','ConcreteNode','RealizedTraceError','capture_realized_trace',
    'SequentialProgram','EvalOp','ReplayValidation','SequentialError','build_sequential_program',
    'LoopRecoveryError','ConstantBinding','AffineIntBinding','ArithmeticRunsIntBinding','PeriodicBinding','InterleavedAffineIntBinding','RunLengthBinding','TableBinding','EvalTemplate',
    'LiteralBlock','LoopBlock','PhaseGrammar','PhaseLoopBlock','LoopFamilyTemplate','FormulaOp','RoleBindingTemplate','LoopCodeFamily','LoopInstancePlan','CanonicalExecutionPlan','StructuredSequentialProgram','infer_binding','binding_payload_items','recover_loops',
    'RealizedTraceCompiler',
]

from .storage import (
    StoragePlanError, NodeLifetime, FormulaStorageSpec, StorageExecutionStats,
    StoragePlan, StorageOptimizedProgram, build_storage_plan, optimize_storage,
)

__all__ += [
    'StoragePlanError','NodeLifetime','FormulaStorageSpec','StorageExecutionStats',
    'StoragePlan','StorageOptimizedProgram','build_storage_plan','optimize_storage',
]

from .slot_storage import (
    SlotStorageError, NodeSlot, FormulaSlotLayout, PhysicalSlotLayout, SlotExecutionStats,
    SlotStorageProgram, build_physical_slot_layout, lower_to_slots,
)

__all__ += [
    'SlotStorageError','NodeSlot','FormulaSlotLayout','PhysicalSlotLayout','SlotExecutionStats',
    'SlotStorageProgram','build_physical_slot_layout','lower_to_slots',
]

from .slot_storage import DirectSlotExecutionStats, DirectSlotProgram, lower_to_direct_slots
__all__ += ['DirectSlotExecutionStats','DirectSlotProgram','lower_to_direct_slots']

from .native_analysis import (
    NativeAnalysisError, NativeTypeEvidence, FormulaNativeAnalysis,
    NativePlanAnalysis, analyze_native_capability,
)
from .native_proof import (
    NativeProofError, NativeSingleRecurrenceProof, build_single_formula_recurrence_cython,
)

__all__ += [
    'NativeAnalysisError','NativeTypeEvidence','FormulaNativeAnalysis',
    'NativePlanAnalysis','analyze_native_capability',
    'NativeProofError','NativeSingleRecurrenceProof','build_single_formula_recurrence_cython',
]

from .native_family import (
    NativeFamilyError, TypedFormulaSlots, TypedNodeAddress, TypedSlotPlan,
    NativeFamilyProof, build_native_family_cython,
)
__all__ += [
    'NativeFamilyError','TypedFormulaSlots','TypedNodeAddress','TypedSlotPlan',
    'NativeFamilyProof','build_native_family_cython',
]

from .native_family import MixedNativeFamilyProof, build_mixed_native_family_cython
__all__ += ['MixedNativeFamilyProof','build_mixed_native_family_cython']

from .native_plan import (
    NativePlanError, NativeFamilyKernelPlan, NativeFamilyInstanceRuntimePlan,
    NativeFamilyBackendPlan, NativeInstanceExpansion,
    build_native_family_backend_plan, expand_native_family_instance,
)
__all__ += [
    'NativePlanError','NativeFamilyKernelPlan','NativeFamilyInstanceRuntimePlan',
    'NativeFamilyBackendPlan','NativeInstanceExpansion',
    'build_native_family_backend_plan','expand_native_family_instance',
]

from .native_family import (
    RuntimeArgSite, MixedNativeFamilyInstanceProof, MixedNativeFamilyKernelProof,
    build_mixed_native_family_kernel_cython,
)
__all__ += [
    'RuntimeArgSite','MixedNativeFamilyInstanceProof','MixedNativeFamilyKernelProof',
    'build_mixed_native_family_kernel_cython',
]

from .native_artifact import (
    ArtifactFamilyPlan, ArtifactRegionPlan, WholeArtifactBackendPlan,
    build_whole_artifact_backend_plan,
)
__all__ += [
    'ArtifactFamilyPlan','ArtifactRegionPlan','WholeArtifactBackendPlan',
    'build_whole_artifact_backend_plan',
]

from .whole_artifact import (
    WholeArtifactError, WholeArtifactExecutionStats, WholeArtifactBuildReport,
    WholeArtifactProgram, build_whole_artifact_cython,
)
__all__ += [
    'WholeArtifactError','WholeArtifactExecutionStats','WholeArtifactBuildReport',
    'WholeArtifactProgram','build_whole_artifact_cython',
]

from .fast_foundation import (
    FastRegionPlan, FastFoundationPlan, FastFoundationBuildReport,
    FastFoundationBenchmark, FastFoundationProgram, FastFoundationError,
    plan_fast_foundation, build_fast_foundation,
)
__all__ += [
    'FastRegionPlan','FastFoundationPlan','FastFoundationBuildReport',
    'FastFoundationBenchmark','FastFoundationProgram','FastFoundationError',
    'plan_fast_foundation','build_fast_foundation',
]

from .optimized_execution import (
    OptimizedPlanError, RoleBackend, BindingSitePlan, ReferenceSiteSummary,
    RoleExecutionPlan, RegisterValuePlan, ReloadValuePlan, PythonBoundaryOccurrence,
    RegisterReferenceReadPlan, LiteralRegionPlan, SlotRegionPlan, RegisterRegionPlan, OptimizedExecutionPlan,
    build_optimized_execution_plan,
)
__all__ += [
    'OptimizedPlanError','RoleBackend','BindingSitePlan','ReferenceSiteSummary',
    'RoleExecutionPlan','RegisterValuePlan','ReloadValuePlan','PythonBoundaryOccurrence',
    'RegisterReferenceReadPlan','LiteralRegionPlan','SlotRegionPlan','RegisterRegionPlan','OptimizedExecutionPlan',
    'build_optimized_execution_plan',
]

from .register_region import (
    RegisterRegionError, RegisterReferenceKernelSite, RegisterFamilyKernelCode,
    RegisterRegionBuildReport, RegisterRegionValidation, RegisterRegionBenchmark,
    RegisterRegionKernelBenchmark, RegisterRegionProgram,
    MixedRegisterFamilyKernelCode, MixedRegisterRegionBuildReport, MixedRegisterRegionProgram,
    build_register_region_cython,
)
__all__ += [
    'RegisterRegionError','RegisterReferenceKernelSite','RegisterFamilyKernelCode',
    'RegisterRegionBuildReport','RegisterRegionValidation','RegisterRegionBenchmark',
    'RegisterRegionKernelBenchmark','RegisterRegionProgram',
    'MixedRegisterFamilyKernelCode','MixedRegisterRegionBuildReport','MixedRegisterRegionProgram',
    'build_register_region_cython',
]

from .frozen_references import (
    FrozenAxisPlan, FrozenNumericReferenceSite, FrozenNumericReferencePlan,
    FrozenReferenceError, build_frozen_numeric_reference_plan,
)
__all__ += [
    "FrozenAxisPlan", "FrozenNumericReferenceSite", "FrozenNumericReferencePlan",
    "FrozenReferenceError", "build_frozen_numeric_reference_plan",
]

from .final_backend import (
    FinalBackendError, ExecutionProfile, FinalBackendProfile,
    choose_final_backend_profile,
)
__all__ += [
    "FinalBackendError", "ExecutionProfile", "FinalBackendProfile",
    "choose_final_backend_profile",
]


from .modelx_cython_bridge import (
    ModelxCythonBridgeError, ModelxCythonTarget, ModelxCythonBridgeBuildReport,
    ModelxCythonBridgeProgram, infer_modelx_cython_target,
    derive_modelx_cython_trace_sample, build_modelx_cython_bridge,
)
from .performance_backend import (
    PerformanceBackendError, PerformanceBackendKind, PerformanceBuildReport,
    PerformanceArtifactProgram, select_against_modelx_cython_floor,
    build_performance_artifact,
)
__all__ += [
    "ModelxCythonBridgeError", "ModelxCythonTarget", "ModelxCythonBridgeBuildReport",
    "ModelxCythonBridgeProgram", "infer_modelx_cython_target",
    "derive_modelx_cython_trace_sample", "build_modelx_cython_bridge",
    "PerformanceBackendError", "PerformanceBackendKind", "PerformanceBuildReport",
    "PerformanceArtifactProgram", "select_against_modelx_cython_floor",
    "build_performance_artifact",
]

from .native_batch import (
    NativeBatchError, NativeBatchValidationError, NativeBatchPlan, NativeBatchBuildReport, NativeBatchBenchmark,
    RunDomainCoverageProof, TraceClosureRound, StageTraceClosurePreparation,
    OptimizedScalarPreparation, StageScalarPreparation, NativeBatchProgram, prepare_optimized_scalar, prepare_legacy_comparison_scalar, prepare_stage_scalar, prepare_stage_scalar_auto_trace, plan_cython_backend, plan_legacy_comparison_cython, plan_stage_cython_backend,
    plan_native_batch, build_optimized_python_batch, build_native_batch, build_native_batch_auto_trace,
)
__all__ += [
    "NativeBatchError", "NativeBatchValidationError", "NativeBatchPlan", "NativeBatchBuildReport", "NativeBatchBenchmark",
    "RunDomainCoverageProof", "TraceClosureRound", "StageTraceClosurePreparation",
    "OptimizedScalarPreparation", "StageScalarPreparation", "NativeBatchProgram", "prepare_optimized_scalar", "prepare_legacy_comparison_scalar", "prepare_stage_scalar", "prepare_stage_scalar_auto_trace", "plan_cython_backend", "plan_legacy_comparison_cython", "plan_stage_cython_backend",
    "plan_native_batch", "build_optimized_python_batch", "build_native_batch", "build_native_batch_auto_trace",
]

from .prepared_batch import (
    PreparedBatchError, PreparedBatchFoundation, PreparedBatchExecution,
    CanonicalBindingFeasibility, analyze_canonical_batch_bindings,
    generated_source_fingerprint,
)
__all__ += [
    "PreparedBatchError", "PreparedBatchFoundation", "PreparedBatchExecution",
    "CanonicalBindingFeasibility", "analyze_canonical_batch_bindings",
    "generated_source_fingerprint",
]

from .staged_python import (
    StagedPythonError, StagedPythonExecutionResult, StagedPythonExecutor,
)
__all__ += [
    'StagedPythonError','StagedPythonExecutionResult','StagedPythonExecutor',
]

from .legacy_comparison import LegacyComparisonError, LegacyComparisonBundle, build_legacy_comparison_bundle


# V02394: schedule-independent native formula IR + typed recursive backend.
from .native_formula import (
    NativeFormulaError, NativeFormulaCell, NativeFormulaProgram,
    build_native_formula_program,
)
from .recursive_native import (
    RecursiveNativeError, RecursiveMemoCell, RecursiveMemoPlan,
    RecursiveNativeProgram, build_recursive_memo_plan,
    build_recursive_native_program, RecursiveCythonGenerator,
)
from .fast_recursive_graph import (
    FastRecursiveGraphError, FastRecursiveCell, FastRecursiveCallSite,
    FastRecursiveProgram, build_fast_recursive_program,
)
from .fast_recursive_python import (
    emit_fast_recursive_python, write_fast_recursive_python,
)
from .fast_recursive_cython import (
    emit_fast_recursive_cython_object, write_fast_recursive_cython_object,
)
__all__ += [
    "NativeFormulaError", "NativeFormulaCell", "NativeFormulaProgram",
    "build_native_formula_program",
    "RecursiveNativeError", "RecursiveMemoCell", "RecursiveMemoPlan",
    "RecursiveNativeProgram", "build_recursive_memo_plan",
    "build_recursive_native_program", "RecursiveCythonGenerator",
    "FastRecursiveGraphError", "FastRecursiveCell", "FastRecursiveCallSite",
    "FastRecursiveProgram", "build_fast_recursive_program",
    "emit_fast_recursive_python", "write_fast_recursive_python",
    "emit_fast_recursive_cython_object", "write_fast_recursive_cython_object",
]

# V02395F: static recursive graph-ownership foundation.
from .static_recursive_frontend import (
    StaticRecursiveFrontendError,
    QualifiedCellKey,
    StaticRecursiveFormulaCall,
    StaticRecursiveFormulaCell,
    StaticRecursiveFormulaSeed,
    StaticRecursiveBindingContext,
    build_static_recursive_formula_seed,
    build_native_formula_program_from_static,
)
__all__ += [
    "StaticRecursiveFrontendError",
    "QualifiedCellKey",
    "StaticRecursiveFormulaCall",
    "StaticRecursiveFormulaCell",
    "StaticRecursiveFormulaSeed",
    "StaticRecursiveBindingContext",
    "build_static_recursive_formula_seed",
    "build_native_formula_program_from_static",
]


# V02396/V127: static graph-owned provider binding adapter.
from .static_provider_binding import (
    StaticProviderBindingError, StaticCellBindingRecord, StaticProviderBindingPlan,
    build_static_provider_binding_plan,
)
__all__ += [
    "StaticProviderBindingError", "StaticCellBindingRecord",
    "StaticProviderBindingPlan", "build_static_provider_binding_plan",
]


# V02397/V128: backend-neutral static type/domain proof.
from .static_type_domain import (
    StaticTypeDomainError, StaticValueDomain, StaticCellTypeProof,
    StaticTypeDomainProof, unknown_domain, literal_domain, numeric_domain,
    infer_expression_domain, build_static_type_domain_proof,
)
__all__ += [
    "StaticTypeDomainError", "StaticValueDomain", "StaticCellTypeProof",
    "StaticTypeDomainProof", "unknown_domain", "literal_domain",
    "numeric_domain", "infer_expression_domain",
    "build_static_type_domain_proof",
]


# V02398/V129: deterministic finite-category semantic/native ABI.
from .static_category_abi import (
    StaticCategoryABIError, StaticCategoryDomain, StaticCategoryEncodingPlan,
    build_static_category_encoding_plan,
)
__all__ += [
    "StaticCategoryABIError", "StaticCategoryDomain",
    "StaticCategoryEncodingPlan", "build_static_category_encoding_plan",
]


# Production recursive memo backend. Legacy policy helpers below are comparison-only.
from .dense_recursive_native import (
    DenseRecursiveNativeError, DenseMemoCell, DenseMemoPlan, DenseRecursiveProgram,
    build_dense_memo_plan, build_dense_recursive_program,
    build_legacy_comparison_recursive_program, DenseRecursiveCythonGenerator,
    size_native_hash_capacities_from_peaks,
    PRODUCTION_MEMOIZATION_POLICY, PRODUCTION_OVERFLOW_POLICY,
    DEPRECATED_COMPARISON_MEMO_POLICIES, LEGACY_COMPARISON_MEMO_POLICIES,
)
from .source_preserving_native import (
    SourcePreservingNativeError, CompleteNumericBound, NumericInterval,
    CythonDirectiveEvidence, CellCompilePlan, SourcePreservingCompilePlan,
    SourcePreservingDenseProgram, derive_complete_input_numeric_bounds,
    build_source_preserving_compile_plan, build_source_preserving_dense_program,
    build_source_preserving_program_from_fast_plan,
    SourcePreservingDenseCythonGenerator,
)
from .execution_classes import (
    N0, N1, P, ExecutionClassError, NativeMemoContract,
    NATIVE_SCALAR_VALUE_ABI, PYTHON_OBJECT_VALUE_ABI, RANKED_ARRAY_VALUE_ABI,
    UNSUPPORTED_VALUE_ABI,
    ExecutionClassCell, FallbackPolicyContextContract, FallbackBridgeSpec,
    ExecutionClassPlan, build_execution_class_plan,
)
from .object_memo_native import (
    ObjectMemoNativeError, ObjectMemoCell, ObjectMemoPlan,
    build_object_memo_plan, ObjectMemoCythonEmitter,
)
from .python_key_memo_native import (
    PythonKeyMemoNativeError, PythonKeyMemoCell, PythonKeyMemoPlan,
    build_python_key_memo_plan, PythonKeyMemoCythonEmitter,
)
from .mixed_recursive_native import (
    MixedRecursiveNativeError, FallbackTargetResolutionError,
    FallbackSpaceHop, FallbackTargetSpec, FallbackTargetBlocker, FallbackTargetPlan,
    build_fallback_target_plan, FallbackPolicyRuntime, MixedRecursiveProgram,
    build_mixed_recursive_program_from_fast_plan, MixedRecursiveCythonGenerator,
)
from .native_hash_capacity_evidence import (
    NativeHashCapacityEvidenceError,
    NativeHashCapacityEvidence,
    profile_native_hash_capacity_evidence,
)
from .semantic_parameter_policy import (
    NativeCallSemantics,
    SemanticParameterCellPolicy,
    SemanticParameterPolicy,
    native_call_semantics,
    classify_semantic_parameter_policy,
)
__all__ += [
    "DenseRecursiveNativeError", "DenseMemoCell", "DenseMemoPlan",
    "DenseRecursiveProgram", "build_dense_memo_plan",
    "build_dense_recursive_program", "build_legacy_comparison_recursive_program",
    "DenseRecursiveCythonGenerator",
    "size_native_hash_capacities_from_peaks",
    "PRODUCTION_MEMOIZATION_POLICY", "PRODUCTION_OVERFLOW_POLICY",
    "DEPRECATED_COMPARISON_MEMO_POLICIES", "LEGACY_COMPARISON_MEMO_POLICIES",
    "SourcePreservingNativeError", "CompleteNumericBound", "NumericInterval",
    "CythonDirectiveEvidence", "CellCompilePlan", "SourcePreservingCompilePlan",
    "SourcePreservingDenseProgram", "derive_complete_input_numeric_bounds",
    "build_source_preserving_compile_plan", "build_source_preserving_dense_program",
    "build_source_preserving_program_from_fast_plan",
    "SourcePreservingDenseCythonGenerator",
    "N0", "N1", "P", "ExecutionClassError", "NativeMemoContract",
    "NATIVE_SCALAR_VALUE_ABI", "PYTHON_OBJECT_VALUE_ABI", "RANKED_ARRAY_VALUE_ABI",
    "UNSUPPORTED_VALUE_ABI",
    "ExecutionClassCell", "FallbackPolicyContextContract", "FallbackBridgeSpec",
    "ExecutionClassPlan", "build_execution_class_plan",
    "ObjectMemoNativeError", "ObjectMemoCell", "ObjectMemoPlan",
    "build_object_memo_plan", "ObjectMemoCythonEmitter",
    "PythonKeyMemoNativeError", "PythonKeyMemoCell", "PythonKeyMemoPlan",
    "build_python_key_memo_plan", "PythonKeyMemoCythonEmitter",
    "MixedRecursiveNativeError", "FallbackTargetResolutionError",
    "FallbackSpaceHop", "FallbackTargetSpec", "FallbackTargetBlocker", "FallbackTargetPlan",
    "build_fallback_target_plan", "FallbackPolicyRuntime", "MixedRecursiveProgram",
    "build_mixed_recursive_program_from_fast_plan", "MixedRecursiveCythonGenerator",
    "NativeHashCapacityEvidenceError", "NativeHashCapacityEvidence",
    "profile_native_hash_capacity_evidence",
    "NativeCallSemantics", "SemanticParameterCellPolicy", "SemanticParameterPolicy",
    "native_call_semantics", "classify_semantic_parameter_policy",
]

__all__ += [
    "AsofLookupIR", "SemanticFormulaIR", "SemanticPassRecord", "FormulaNormalizationResult",
    "recognize_asof_lookups",
    "AstSemanticPass", "SemanticPassPipeline", "SemanticPassError",
]

# M2/dev82: modelx-cython-owned finite-category/provider fast path.
from .category_abi import (
    CategoryABIError, CategoryDomain, CategoryTranslation, CategoryEncodingPlan,
    attach_parameter_positions,
)
from .modelx_cython_category_abi import (
    ModelxCythonCategoryABIError, FastCategoryEvidence,
    ParameterizedPointCategoryCase, ParameterizedPointCategoryBinding,
    ModelxCythonCategoryPlan, build_modelx_cython_category_plan,
)
from .modelx_cython_fast_backend import (
    ModelxCythonFastBackendError, FastParameterEvidence,
    FastScalarizedVectorHelper, FastFrontendCell, ModelxCythonFastPlan,
    build_modelx_cython_fast_plan,
)
__all__ += [
    "CategoryABIError", "CategoryDomain", "CategoryTranslation", "CategoryEncodingPlan",
    "attach_parameter_positions",
    "ModelxCythonCategoryABIError", "FastCategoryEvidence",
    "ParameterizedPointCategoryCase", "ParameterizedPointCategoryBinding",
    "ModelxCythonCategoryPlan", "build_modelx_cython_category_plan",
    "ModelxCythonFastBackendError", "FastParameterEvidence",
    "FastScalarizedVectorHelper", "FastFrontendCell", "ModelxCythonFastPlan",
    "build_modelx_cython_fast_plan",
]

from .semantic_contract import (
    SemanticContractError, AxisDomain, ValueSpec, ParameterSpec, CellSignature,
    LoweringPlan, SemanticFacadePlan, build_semantic_facade,
)
__all__ += [
    'SemanticContractError','AxisDomain','ValueSpec','ParameterSpec','CellSignature',
    'LoweringPlan','SemanticFacadePlan','build_semantic_facade',
]
