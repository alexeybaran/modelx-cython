from __future__ import annotations

"""Direct native portfolio batch kernel built from the generic graph backend.

This module deliberately adds no second modelx graph trace.  ``NativeBatch`` now
reuses the concrete dependency evidence already captured by
``RealizedTraceCompiler`` and lowers that evidence through the shared
``GraphModelCompiler -> StageOptimizedProgram -> direct Python/Cython emitters``
stages. Mature OptimizedProgram support remains only in explicitly named comparison
helpers and is not imported by the production planning/build path.

The generated hot loop contains no modelx/modelx-cython runtime, role dispatcher,
``_ref_index`` lookup, binding descriptor interpreter, or physical-slot VM.  Those
abstractions exist only while the compiler constructs the direct Cython program.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import hashlib
import statistics
import tempfile
import time

import numpy as np

from .build import build_extension
from .codegen_common import CodegenError
from .frontend import GraphModelCompiler, FrontendError
from .invocation import OutputInvocation, RunDomain, ResultDomain
from .trace_bridge import trace_capture_from_realized
from .optimized_python_artifact import OptimizedPythonArtifact, build_optimized_python_artifact
from .loop_evidence import build_recovered_loop_evidence
from .stage_optimized_program import StageOptimizedProgram, build_stage_optimized_program_from_model
from .stage_codegen import StagePythonGenerator, StageCythonGenerator


class NativeBatchError(RuntimeError):
    """Fail-closed native batch planning/build error."""


class NativeBatchValidationError(NativeBatchError):
    """Full/sampled modelx parity failure with machine-readable evidence."""

    def __init__(self, message: str, validation: dict[str, Any]):
        super().__init__(message)
        self.validation = validation


@dataclass(frozen=True)
class RunDomainCoverageProof:
    """Compile-time evidence that one artifact covers its frozen run domain.

    This is deliberately separate from the Stage scheduler/program.  Scheduling
    proves *how* the graph executes; this record proves that the selected realized
    trace was semantically sufficient for the declared finite NativeBatch domain.
    """

    proof_kind: str
    run_domain_count: int
    compared_count: int
    sample_count: int
    complete: bool
    exact: bool
    allclose_1e12: bool
    max_abs_error: float
    max_scaled_relative_error: float
    bad_count: int
    bad_keys: tuple[Any, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "proof_kind": self.proof_kind,
            "run_domain_count": int(self.run_domain_count),
            "compared_count": int(self.compared_count),
            "sample_count": int(self.sample_count),
            "complete": bool(self.complete),
            "exact": bool(self.exact),
            "allclose_1e12": bool(self.allclose_1e12),
            "max_abs_error": float(self.max_abs_error),
            "max_scaled_relative_error": float(self.max_scaled_relative_error),
            "bad_count": int(self.bad_count),
            "bad_keys": [repr(x) for x in self.bad_keys],
        }


@dataclass(frozen=True)
class TraceClosureRound:
    round_index: int
    sample_keys: tuple[Any, ...]
    outcome: str
    reason: str | None
    added_keys: tuple[Any, ...] = ()


@dataclass
class StageTraceClosurePreparation:
    """Production Stage preparation after automatic representative-trace closure."""

    compiler: Any
    preparation: "StageScalarPreparation"
    coverage_proof: RunDomainCoverageProof
    rounds: tuple[TraceClosureRound, ...]


@dataclass(frozen=True)
class NativeBatchPlan:
    available: bool
    reason: str | None
    output_name: str | None
    base_space_name: str | None
    sample_keys: tuple[Any, ...]
    point_count: int
    optimization_level: str
    emission_mode: str = "array_abi"
    input_names: tuple[str, ...] = ()
    history_count: int = 0
    ring_count: int = 0
    scalar_count: int = 0
    fused_reduction_count: int = 0
    previous_scalar_count: int = 0
    trace_source: str = "realized_trace"
    optimized_program_ready: bool = False
    optimized_python_ready: bool = False
    optimized_python_correct: bool = False
    cython_ready: bool = False
    python_reason: str | None = None
    cython_reason: str | None = None
    guard_count: int = 0
    stage_program_ready: bool = False
    stage_program_uid: str | None = None
    execution_plan_uid: str | None = None
    storage_plan_uid: str | None = None


@dataclass
class OptimizedScalarPreparation:
    frontend: GraphModelCompiler
    base: Any
    output_name: str
    samples: tuple[Any, ...]
    optimized: Any
    bound_inputs: dict[str, Any]
    point_count: int
    python_source: str
    python_validation: dict[str, Any]
    frontend_seconds: float
    input_prepare_seconds: float
    optimized_program_seconds: float
    optimized_python_seconds: float


@dataclass
class StageScalarPreparation:
    frontend: GraphModelCompiler
    base: Any
    output_name: str
    samples: tuple[Any, ...]
    stage_program: StageOptimizedProgram
    bound_inputs: dict[str, Any]
    point_count: int
    python_source: str
    python_validation: dict[str, Any]
    frontend_seconds: float
    input_prepare_seconds: float
    stage_program_seconds: float
    stage_python_seconds: float


@dataclass(frozen=True)
class NativeBatchBuildReport:
    point_count: int
    output_name: str
    optimization_level: str
    emission_mode: str
    c_optimization: str
    native_arch: bool
    frontend_seconds: float
    input_prepare_seconds: float
    optimized_program_seconds: float
    optimized_python_seconds: float
    codegen_seconds: float
    build_seconds: float
    source_lines: int
    source_bytes: int
    optimized_python_path: str
    optimized_program_manifest_path: str
    optimized_inputs_path: str
    pyx_path: str
    so_path: str
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    history_count: int
    ring_count: int
    scalar_count: int
    fused_reduction_count: int
    previous_scalar_count: int
    trace_source: str
    stage_program_uid: str | None = None
    execution_plan_uid: str | None = None
    storage_plan_uid: str | None = None
    build_fingerprint: str | None = None
    sample_count: int = 0
    run_domain_coverage_proof: dict[str, Any] | None = None
    trace_closure_rounds: int = 0


@dataclass(frozen=True)
class NativeBatchBenchmark:
    point_count: int
    median_seconds: float
    p10_seconds: float
    p90_seconds: float
    min_seconds: float
    max_seconds: float
    per_point_seconds: float


@dataclass(frozen=True)
class NativeTargetContext:
    base_space: Any
    output_invocation: OutputInvocation
    sample_run_domain: RunDomain
    sample_keys: tuple[Any, ...]

    @property
    def output_name(self) -> str:
        return self.output_invocation.output_name

    # Private compatibility for older tests/scripts that unpacked
    # ``_target_context`` into (base, output_name, sample_keys).
    def __iter__(self):
        yield self.base_space
        yield self.output_name
        yield self.sample_keys


def _target_context(compiler) -> NativeTargetContext:
    targets = tuple(compiler.trace.target_runtime_nodes)
    if not targets:
        raise NativeBatchError("native batch requires at least one realized target")

    base0 = None
    output0 = None
    output_arg_names0: tuple[str, ...] | None = None
    output_arg_values0: tuple[Any, ...] | None = None
    samples: list[Any] = []
    space_params0: tuple[str, ...] | None = None

    for obj, cell_key in targets:
        output_name = getattr(obj, "name", None)
        if not output_name:
            raise NativeBatchError("could not determine output Cell name")
        try:
            cell_parameters = tuple(getattr(obj.interface, "parameters", ()) or ())
        except Exception as exc:
            raise NativeBatchError("could not resolve output Cell parameters") from exc
        raw_args = tuple(cell_key)
        if len(raw_args) != len(cell_parameters):
            raise NativeBatchError(
                f"output Cell {output_name!r} target key {raw_args!r} does not match "
                f"parameters {cell_parameters!r}"
            )
        try:
            invocation = OutputInvocation.fixed(
                str(output_name), cell_parameters, raw_args
            )
        except (TypeError, ValueError) as exc:
            raise NativeBatchError(str(exc)) from exc

        containing_impl = getattr(obj, "parent", None)
        if containing_impl is None:
            raise NativeBatchError("target has no containing Space")
        try:
            containing = containing_impl.interface
        except Exception as exc:  # pragma: no cover
            raise NativeBatchError("could not resolve target containing Space interface") from exc

        # Dynamic Cells live under an ItemSpace whose parent is the structural
        # parameterized Space.  UserCells in a non-parameterized Space already
        # live directly under that structural Space.
        if containing.__class__.__name__ == "ItemSpace":
            base = getattr(containing, "parent", None)
            if base is None:
                raise NativeBatchError("target ItemSpace has no base Space")
            params = tuple(getattr(base, "parameters", ()) or ())
            values = tuple(getattr(containing, name) for name in params)
            sample_key = values[0] if len(values) == 1 else values
        else:
            base = containing
            params = tuple(getattr(base, "parameters", ()) or ())
            if params:
                raise NativeBatchError(
                    "parameterized structural Space target did not resolve through an ItemSpace"
                )
            sample_key = ()

        if base0 is None:
            base0 = base
            output0 = invocation
            output_arg_names0 = invocation.argument_names
            output_arg_values0 = invocation.argument_values
            space_params0 = params
        else:
            if getattr(base, "fullname", None) != getattr(base0, "fullname", None):
                raise NativeBatchError(
                    "all realized batch targets must belong to one structural Space"
                )
            if str(output_name) != output0.output_name:
                raise NativeBatchError(
                    "all realized batch targets must be the same output Cell"
                )
            if invocation.argument_names != output_arg_names0 or invocation.argument_values != output_arg_values0:
                raise NativeBatchError(
                    "all realized batch targets must use one fixed output invocation tuple"
                )
            if params != space_params0:
                raise NativeBatchError(
                    "all realized batch targets must share one ItemSpace parameter schema"
                )
        samples.append(sample_key)

    assert base0 is not None and output0 is not None and space_params0 is not None
    try:
        sample_domain = RunDomain.finite(
            space_fullname=str(getattr(base0, "fullname", None) or getattr(base0, "name", "<space>")),
            itemspace_parameter_names=space_params0,
            run_keys=tuple(samples),
            proof_kind="realized_target_sample_domain_v1",
        )
    except (TypeError, ValueError) as exc:
        raise NativeBatchError(str(exc)) from exc
    return NativeTargetContext(
        base_space=base0,
        output_invocation=output0,
        sample_run_domain=sample_domain,
        sample_keys=tuple(samples),
    )


def _build_frontend(
    compiler,
    *,
    sample_keys: Sequence[Any] | None,
    run_keys: Sequence[Any] | None,
    canonical_schedule_authority: bool = True,
    legacy_comparison: bool | None = None,
):
    context = _target_context(compiler)
    base = context.base_space
    output_name = context.output_name
    realized_samples = context.sample_keys
    samples = tuple(realized_samples if sample_keys is None else sample_keys)
    if not samples:
        raise NativeBatchError("sample_keys must not be empty")
    if tuple(samples) != tuple(realized_samples):
        raise NativeBatchError(
            "NativeBatch sample_keys must match the already-realized compiler targets; "
            "build RealizedTraceCompiler with all representative targets instead of tracing twice"
        )
    capture = trace_capture_from_realized(compiler.trace, sample_keys=samples)
    production_run_domain: RunDomain | None = None
    params = tuple(context.sample_run_domain.itemspace_parameter_names)
    if run_keys is not None:
        production_keys = tuple(run_keys)
    elif len(params) != 1:
        # Multi-parameter and scalar/no-parameter targets cannot use the old
        # one-column model-point inference.  The already-realized structural
        # instances are therefore the explicit finite RunDomain.
        production_keys = samples
    else:
        production_keys = None
    if production_keys is not None:
        try:
            production_run_domain = RunDomain.finite(
                space_fullname=context.sample_run_domain.space_fullname,
                itemspace_parameter_names=params,
                run_keys=production_keys,
                proof_kind=(
                    "explicit_native_batch_run_domain_v1"
                    if run_keys is not None
                    else "realized_target_run_domain_v1"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise NativeBatchError(str(exc)) from exc
    kwargs = dict(
        model=compiler.trace.model,
        space=base,
        output=output_name,
        output_invocation=context.output_invocation,
        sample_keys=samples,
        trace_capture=capture,
        realized_graph_only=True,
        canonical_schedule_authority=bool(canonical_schedule_authority),
    )
    if legacy_comparison is not None:
        kwargs["legacy_comparison"] = bool(legacy_comparison)
    if production_run_domain is not None:
        kwargs["run_domain"] = production_run_domain

    def recovered_evidence_provider():
        # Invoked by template scheduling only after every source/static semantic
        # check has succeeded and the sole missing proof is repeated-loop evidence.
        structured = compiler.structured if compiler.structured is not None else compiler.recover_loops()
        recovered = build_recovered_loop_evidence(structured)
        if not recovered.exact_expansion_validated or not recovered.witnesses:
            return None
        return recovered

    kwargs["recovered_loop_evidence_provider"] = recovered_evidence_provider
    try:
        frontend = GraphModelCompiler(**kwargs)
    except (FrontendError, ValueError, TypeError) as exc:
        raise NativeBatchError(str(exc)) from exc
    return frontend, base, output_name, samples


def _validate_python_source_against_modelx(
    frontend: GraphModelCompiler,
    source: str,
    *,
    max_points: int = 16,
    execute_full_run_domain: bool = False,
    compare_full_run_domain: bool = False,
) -> dict[str, Any]:
    compare_count = max(1, int(max_points))
    execution_keys = tuple(
        frontend.run_keys if execute_full_run_domain
        else frontend.run_keys[:compare_count]
    )
    if not execution_keys:
        raise NativeBatchError("optimized Python validation requires at least one run key")
    inputs = frontend.canonical.bind_inputs(execution_keys)
    ns: dict[str, Any] = {}
    exec(compile(source, "<optimized-python-plan>", "exec"), ns)
    try:
        got_all = np.asarray(ns["run"](inputs), dtype=np.float64)
    except Exception as exc:
        raise NativeBatchError(
            "optimized Python validation execution failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if got_all.size != len(execution_keys):
        raise NativeBatchError(
            f"optimized Python validation produced {got_all.size} points for "
            f"{len(execution_keys)} execution keys"
        )
    keys = (
        execution_keys
        if compare_full_run_domain
        else execution_keys[: min(compare_count, len(execution_keys))]
    )
    got = got_all[: len(keys)]
    expected = np.asarray(
        [float(frontend.modelx_output_value(key)) for key in keys],
        dtype=np.float64,
    )
    if got.shape != expected.shape:
        raise NativeBatchError(
            f"optimized Python validation shape mismatch: {got.shape} vs {expected.shape}"
        )
    diff = np.abs(got - expected)
    scale = np.maximum(1.0, np.abs(expected))
    close = np.isclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    bad_positions = np.where(~close)[0]
    bad_keys = tuple(keys[int(i)] for i in bad_positions[:64])
    result = {
        "point_count": len(keys),
        "execution_point_count": len(execution_keys),
        "complete_run_domain_validated": bool(len(keys) == len(frontend.run_keys)),
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "max_scaled_relative_error": float((diff / scale).max()) if diff.size else 0.0,
        "exact": bool(np.array_equal(got, expected)),
        "allclose_1e12": bool(np.all(close)),
        "bad_count": int(len(bad_positions)),
        "bad_positions": [int(i) for i in bad_positions[:64]],
        "bad_keys": bad_keys,
        "checksum": float(np.sum(got, dtype=np.float64)),
    }
    if not result["allclose_1e12"]:
        first_bad = bad_keys[0] if bad_keys else None
        raise NativeBatchValidationError(
            "optimized Python validation failed the rtol=atol=1e-12 modelx gate: "
            f"max_abs={result['max_abs_error']:.6g}; bad_count={result['bad_count']}; "
            f"first_bad_key={first_bad!r}",
            result,
        )
    return result


def prepare_optimized_scalar(
    compiler,
    *,
    sample_keys: Sequence[Any] | None = None,
    run_keys: Sequence[Any] | None = None,
    optimization_level: str = "O2",
    validation_points: int = 16,
) -> OptimizedScalarPreparation:
    """Compatibility alias for the explicit mature comparison preparation.

    Production NativeBatch does not call this function.  New callers should use
    :func:`prepare_stage_scalar` for production or
    :func:`prepare_legacy_comparison_scalar` when they intentionally need the
    mature ExecutableGraph/OptimizedProgram oracle.
    """
    from .optimized_program import build_optimized_program
    from .python_codegen import PythonLoopGenerator

    level = optimization_level.upper()
    t0 = time.perf_counter()
    frontend, base, output_name, samples = _build_frontend(
        compiler, sample_keys=sample_keys, run_keys=run_keys,
        canonical_schedule_authority=True, legacy_comparison=True,
    )
    frontend_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    bound = frontend.canonical.bind_inputs()
    input_prepare_seconds = time.perf_counter() - t0
    point_count = _point_count(frontend, bound)

    t0 = time.perf_counter()
    optimized = build_optimized_program(frontend.canonical, optimization_level=level)
    optimized_program_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    python_source = PythonLoopGenerator(optimized).emit("optimized_model")
    python_validation = _validate_python_source_against_modelx(
        frontend, python_source, max_points=min(validation_points, point_count)
    )
    optimized_python_seconds = time.perf_counter() - t0

    return OptimizedScalarPreparation(
        frontend=frontend,
        base=base,
        output_name=output_name,
        samples=samples,
        optimized=optimized,
        bound_inputs=bound,
        point_count=point_count,
        python_source=python_source,
        python_validation=python_validation,
        frontend_seconds=frontend_seconds,
        input_prepare_seconds=input_prepare_seconds,
        optimized_program_seconds=optimized_program_seconds,
        optimized_python_seconds=optimized_python_seconds,
    )


def prepare_legacy_comparison_scalar(compiler, **kwargs) -> OptimizedScalarPreparation:
    """Explicitly build and validate the mature comparison-only scalar program."""
    return prepare_optimized_scalar(compiler, **kwargs)


def plan_cython_backend(
    preparation: OptimizedScalarPreparation,
    *,
    emission_mode: str = "direct_locals",
) -> NativeBatchPlan:
    """Compatibility probe for the mature comparison-only Cython representation."""
    from .codegen import CythonGenerator
    optimized = preparation.optimized
    level = optimized.optimization_level
    mode = str(emission_mode).lower()
    base_name = getattr(preparation.base, "fullname", None) or getattr(preparation.base, "name", None)
    try:
        gen = CythonGenerator(optimized, emission_mode=mode)
        # Force complete source lowering.  Constructor success alone is not a
        # capability proof for statement/control-flow constructs.
        gen.emit("mxg_native_plan_probe")
        return NativeBatchPlan(
            available=True,
            reason=None,
            output_name=preparation.output_name,
            base_space_name=base_name,
            sample_keys=preparation.samples,
            point_count=preparation.point_count,
            optimization_level=level,
            emission_mode=mode,
            input_names=tuple(optimized.input_order),
            history_count=len(gen.layout.history_slots),
            ring_count=len(gen.layout.ring_bases),
            scalar_count=len(gen.layout.scalar_slots),
            fused_reduction_count=len(gen.layout.fused_reductions),
            previous_scalar_count=len(gen.direct_prev_uids),
            trace_source=preparation.frontend.trace_source,
            optimized_program_ready=True,
            optimized_python_ready=True,
            optimized_python_correct=bool(preparation.python_validation["allclose_1e12"]),
            cython_ready=True,
            guard_count=len(optimized.guards),
        )
    except CodegenError as exc:
        return NativeBatchPlan(
            available=False,
            reason=str(exc),
            output_name=preparation.output_name,
            base_space_name=base_name,
            sample_keys=preparation.samples,
            point_count=preparation.point_count,
            optimization_level=level,
            emission_mode=mode,
            input_names=tuple(optimized.input_order),
            history_count=len(optimized.layout.history_slots),
            ring_count=len(optimized.layout.ring_bases),
            scalar_count=len(optimized.layout.scalar_slots),
            fused_reduction_count=len(optimized.layout.fused_reductions),
            previous_scalar_count=0,
            trace_source=preparation.frontend.trace_source,
            optimized_program_ready=True,
            optimized_python_ready=True,
            optimized_python_correct=bool(preparation.python_validation["allclose_1e12"]),
            cython_ready=False,
            cython_reason=str(exc),
            guard_count=len(optimized.guards),
        )


def plan_legacy_comparison_cython(
    preparation: OptimizedScalarPreparation,
    *,
    emission_mode: str = "direct_locals",
) -> NativeBatchPlan:
    """Explicit mature comparison-only Cython capability probe."""
    return plan_cython_backend(preparation, emission_mode=emission_mode)


def _stage_storage_metrics(program: StageOptimizedProgram) -> tuple[int, int, int, int]:
    history_count = sum(x.history_slot is not None for x in program.physical_values)
    ring_count = sum(x.ring_base is not None for x in program.physical_values)
    scalar_count = sum(
        x.working_scalar_slot is not None or x.scalar_slot is not None
        for x in program.physical_values
    )
    reduction_count = sum(len(x.reduction_uids) for x in program.execution_blocks)
    return int(history_count), int(ring_count), int(scalar_count), int(reduction_count)


def prepare_stage_scalar(
    compiler,
    *,
    sample_keys: Sequence[Any] | None = None,
    run_keys: Sequence[Any] | None = None,
    optimization_level: str = "O2",
    validation_points: int = 16,
    full_run_domain_validation: bool = False,
) -> StageScalarPreparation:
    """Prepare the canonical Stage program without mature IR construction.

    Structural planning keeps the historical bounded modelx comparison unless
    ``full_run_domain_validation`` is requested.  Production build/promotion and
    automatic trace closure request complete-domain validation explicitly.
    """
    level = optimization_level.upper()
    t0 = time.perf_counter()
    frontend, base, output_name, samples = _build_frontend(
        compiler, sample_keys=sample_keys, run_keys=run_keys,
        canonical_schedule_authority=True, legacy_comparison=False,
    )
    frontend_seconds = time.perf_counter() - t0
    canonical = frontend.canonical
    if canonical.executable is not None:
        raise NativeBatchError("production Stage frontend unexpectedly constructed ExecutableGraph")
    authority = canonical.canonical_execution_schedule
    if authority is None or not bool(getattr(authority, "eligible", False)):
        blockers = () if authority is None else tuple(getattr(authority, "capability_blockers", ()))
        raise NativeBatchError(
            "canonical Stage authority is unavailable"
            + (": " + "; ".join(blockers) if blockers else "")
        )

    t0 = time.perf_counter()
    bound = canonical.bind_inputs()
    input_prepare_seconds = time.perf_counter() - t0
    point_count = _point_count(frontend, bound)

    t0 = time.perf_counter()
    stage_program = build_stage_optimized_program_from_model(
        canonical, optimization_level=level
    )
    stage_program_seconds = time.perf_counter() - t0
    if stage_program.legacy_program is not None:
        raise NativeBatchError("production Stage program unexpectedly carries mature OptimizedProgram")
    if not stage_program.direct_python_supported:
        raise NativeBatchError(
            "direct Stage Python unavailable: " + "; ".join(stage_program.direct_python_blockers)
        )

    t0 = time.perf_counter()
    python_source = StagePythonGenerator(stage_program).source("stage_optimized_model")
    python_validation = _validate_python_source_against_modelx(
        frontend, python_source, max_points=min(validation_points, point_count),
        execute_full_run_domain=True,
        compare_full_run_domain=bool(full_run_domain_validation),
    )
    stage_python_seconds = time.perf_counter() - t0
    return StageScalarPreparation(
        frontend=frontend, base=base, output_name=output_name, samples=samples,
        stage_program=stage_program, bound_inputs=bound, point_count=point_count,
        python_source=python_source, python_validation=python_validation,
        frontend_seconds=frontend_seconds, input_prepare_seconds=input_prepare_seconds,
        stage_program_seconds=stage_program_seconds, stage_python_seconds=stage_python_seconds,
    )


def _coverage_proof_for_preparation(preparation: StageScalarPreparation) -> RunDomainCoverageProof:
    row = preparation.python_validation
    run_count = len(tuple(preparation.frontend.run_keys))
    return RunDomainCoverageProof(
        proof_kind="modelx_full_run_domain_parity_v1",
        run_domain_count=run_count,
        compared_count=int(row.get("point_count", 0)),
        sample_count=len(preparation.samples),
        complete=bool(row.get("complete_run_domain_validated", False)),
        exact=bool(row.get("exact", False)),
        allclose_1e12=bool(row.get("allclose_1e12", False)),
        max_abs_error=float(row.get("max_abs_error", float("inf"))),
        max_scaled_relative_error=float(row.get("max_scaled_relative_error", float("inf"))),
        bad_count=int(row.get("bad_count", 0)),
        bad_keys=tuple(row.get("bad_keys", ())),
    )


def _trace_incompleteness_reason(reason: str) -> bool:
    text = reason.lower()
    markers = (
        "not observed in the representative trace",
        "not observed in sample trace",
        "broader representative sampling",
        "was not observed in the representative trace",
    )
    return any(marker in text for marker in markers)


def _target_node_for_run_key(
    base: Any, output_invocation: OutputInvocation, key: Any
) -> Any:
    params = tuple(getattr(base, "parameters", ()) or ())
    if not params:
        if key != ():
            raise NativeBatchError(
                f"non-parameterized Space expects RunDomain key (), got {key!r}"
            )
        item = base
    elif len(params) == 1:
        item = base[key]
    else:
        if not isinstance(key, tuple):
            try:
                key = tuple(key)
            except TypeError as exc:
                raise NativeBatchError(
                    f"run key {key!r} does not provide {len(params)} Space coordinates"
                ) from exc
        if len(key) != len(params):
            raise NativeBatchError(
                f"run key {key!r} does not match {len(params)} Space parameters"
            )
        item = base[key]
    try:
        return getattr(item, output_invocation.output_name).node(
            *output_invocation.argument_values
        )
    except Exception as exc:
        raise NativeBatchError(
            "could not construct automatic trace target "
            f"{output_invocation.output_name}{output_invocation.argument_values!r} "
            f"for run key {key!r}"
        ) from exc


def _retrace_for_sample_keys(compiler: Any, sample_keys: Sequence[Any]) -> Any:
    context = _target_context(compiler)
    targets = tuple(
        _target_node_for_run_key(
            context.base_space, context.output_invocation, key
        )
        for key in sample_keys
    )
    from .realized_compiler import RealizedTraceCompiler
    return RealizedTraceCompiler.from_target(compiler.trace.model, targets)


def _stratified_unseen_keys(
    run_keys: tuple[Any, ...], sample_keys: tuple[Any, ...], limit: int,
) -> tuple[Any, ...]:
    sampled = set(sample_keys)
    unseen = tuple(key for key in run_keys if key not in sampled)
    n = min(max(0, int(limit)), len(unseen))
    if n <= 0:
        return ()
    if n == len(unseen):
        return unseen
    # Deterministic domain-wide exploration.  This is a compiler search policy,
    # not model configuration: large portfolios are sampled across their entire
    # declared order instead of assuming the first rows are representative.
    indices = []
    used: set[int] = set()
    for i in range(n):
        pos = min(len(unseen) - 1, int(((i + 0.5) * len(unseen)) / n))
        if pos not in used:
            indices.append(pos)
            used.add(pos)
    if len(indices) < n:
        for pos in range(len(unseen)):
            if pos not in used:
                indices.append(pos)
                used.add(pos)
                if len(indices) == n:
                    break
    return tuple(unseen[pos] for pos in indices)


def prepare_stage_scalar_auto_trace(
    compiler,
    *,
    run_keys: Sequence[Any],
    optimization_level: str = "O2",
    validation_points: int = 16,
    expansion_batch: int = 16,
    max_rounds: int = 8,
    max_sample_keys: int = 128,
) -> StageTraceClosurePreparation:
    """Automatically close representative trace coverage over a finite RunDomain.

    The closure loop has two generic witnesses:

    * a full-domain numerical mismatch contributes the failing run keys;
    * an explicit *unobserved representative trace* frontend failure contributes
      deterministic stratified unseen run keys.

    Other semantic/frontend blockers fail immediately.  In particular this does
    not hide dynamic-parameter, table, control-flow, or graph-shape gaps by blindly
    tracing more model points.
    """
    domain = tuple(run_keys)
    if not domain:
        raise NativeBatchError("automatic trace closure requires a non-empty run domain")
    if int(expansion_batch) <= 0 or int(max_rounds) <= 0 or int(max_sample_keys) <= 0:
        raise NativeBatchError("automatic trace closure limits must be positive")

    current = compiler
    _, _, initial_samples = _target_context(current)
    samples = tuple(initial_samples)
    if not samples:
        raise NativeBatchError("automatic trace closure requires at least one realized sample")
    domain_set = set(domain)
    outside = tuple(key for key in samples if key not in domain_set)
    if outside:
        raise NativeBatchError(
            "automatic trace closure requires every realized sample to belong to the "
            f"declared run domain; outside={outside!r}"
        )
    rounds: list[TraceClosureRound] = []

    for round_index in range(int(max_rounds)):
        try:
            preparation = prepare_stage_scalar(
                current,
                run_keys=domain,
                optimization_level=optimization_level,
                validation_points=validation_points,
                full_run_domain_validation=True,
            )
        except NativeBatchValidationError as exc:
            bad = tuple(exc.validation.get("bad_keys", ()))
            sampled = set(samples)
            additions = tuple(key for key in bad if key not in sampled)[: int(expansion_batch)]
            if not additions:
                additions = _stratified_unseen_keys(domain, samples, int(expansion_batch))
            additions = additions[: max(0, int(max_sample_keys) - len(samples))]
            rounds.append(TraceClosureRound(
                round_index=round_index,
                sample_keys=samples,
                outcome="validation_mismatch",
                reason=str(exc),
                added_keys=additions,
            ))
            if not additions:
                raise NativeBatchError(
                    "automatic trace closure found a full-domain numerical mismatch but "
                    "could not add any new representative run keys"
                ) from exc
        except NativeBatchError as exc:
            if not _trace_incompleteness_reason(str(exc)):
                raise
            additions = _stratified_unseen_keys(domain, samples, int(expansion_batch))
            additions = additions[: max(0, int(max_sample_keys) - len(samples))]
            rounds.append(TraceClosureRound(
                round_index=round_index,
                sample_keys=samples,
                outcome="trace_incomplete",
                reason=str(exc),
                added_keys=additions,
            ))
            if not additions:
                raise NativeBatchError(
                    "automatic trace closure exhausted the declared run domain without "
                    "resolving representative-trace incompleteness"
                ) from exc
        else:
            proof = _coverage_proof_for_preparation(preparation)
            if not proof.complete or not proof.allclose_1e12:
                raise NativeBatchError(
                    "automatic trace closure requires complete full-domain parity evidence"
                )
            rounds.append(TraceClosureRound(
                round_index=round_index,
                sample_keys=samples,
                outcome="complete",
                reason=None,
                added_keys=(),
            ))
            return StageTraceClosurePreparation(
                compiler=current,
                preparation=preparation,
                coverage_proof=proof,
                rounds=tuple(rounds),
            )

        sampled = set(samples)
        samples = samples + tuple(key for key in additions if key not in sampled)
        if len(samples) > int(max_sample_keys):
            raise NativeBatchError(
                f"automatic trace closure exceeds the generic cap of {int(max_sample_keys)} sample keys"
            )
        current = _retrace_for_sample_keys(current, samples)

    raise NativeBatchError(
        f"automatic trace closure did not converge within {int(max_rounds)} rounds"
    )


def plan_stage_cython_backend(
    preparation: StageScalarPreparation,
    *,
    emission_mode: str = "direct_locals",
) -> NativeBatchPlan:
    program = preparation.stage_program
    mode = str(emission_mode).lower()
    base_name = getattr(preparation.base, "fullname", None) or getattr(preparation.base, "name", None)
    history_count, ring_count, scalar_count, reduction_count = _stage_storage_metrics(program)
    try:
        with tempfile.TemporaryDirectory(prefix="mxg_stage_plan_") as td:
            StageCythonGenerator(program, emission_mode=mode).write(
                Path(td) / "stage_plan.pyx", "mxg_stage_plan_probe"
            )
        return NativeBatchPlan(
            available=True, reason=None, output_name=preparation.output_name,
            base_space_name=base_name, sample_keys=preparation.samples,
            point_count=preparation.point_count, optimization_level="O2",
            emission_mode=mode, input_names=tuple(program.input_order),
            history_count=history_count, ring_count=ring_count, scalar_count=scalar_count,
            fused_reduction_count=reduction_count, previous_scalar_count=0,
            trace_source=preparation.frontend.trace_source, optimized_program_ready=True,
            optimized_python_ready=True,
            optimized_python_correct=bool(preparation.python_validation["allclose_1e12"]),
            cython_ready=True, guard_count=len(program.guards), stage_program_ready=True,
            stage_program_uid=program.uid, execution_plan_uid=program.execution_plan.uid,
            storage_plan_uid=program.storage_plan.uid,
        )
    except CodegenError as exc:
        return NativeBatchPlan(
            available=False, reason=str(exc), output_name=preparation.output_name,
            base_space_name=base_name, sample_keys=preparation.samples,
            point_count=preparation.point_count, optimization_level="O2",
            emission_mode=mode, input_names=tuple(program.input_order),
            history_count=history_count, ring_count=ring_count, scalar_count=scalar_count,
            fused_reduction_count=reduction_count, previous_scalar_count=0,
            trace_source=preparation.frontend.trace_source, optimized_program_ready=True,
            optimized_python_ready=True,
            optimized_python_correct=bool(preparation.python_validation["allclose_1e12"]),
            cython_ready=False, cython_reason=str(exc), guard_count=len(program.guards),
            stage_program_ready=True, stage_program_uid=program.uid,
            execution_plan_uid=program.execution_plan.uid, storage_plan_uid=program.storage_plan.uid,
        )


def plan_native_batch(
    compiler,
    *,
    sample_keys: Sequence[Any] | None = None,
    run_keys: Sequence[Any] | None = None,
    optimization_level: str = "O2",
    emission_mode: str = "direct_locals",
) -> NativeBatchPlan:
    """Plan NativeBatch exclusively from canonical Stage production IR."""
    level = optimization_level.upper()
    mode = str(emission_mode).lower()
    try:
        preparation = prepare_stage_scalar(
            compiler, sample_keys=sample_keys, run_keys=run_keys,
            optimization_level=level,
        )
    except (NativeBatchError, CodegenError, FrontendError) as exc:
        return NativeBatchPlan(
            available=False, reason=str(exc), output_name=None, base_space_name=None,
            sample_keys=tuple(sample_keys or ()), point_count=0,
            optimization_level=level, emission_mode=mode, optimized_program_ready=False,
            optimized_python_ready=False, optimized_python_correct=False, cython_ready=False,
            python_reason=str(exc), stage_program_ready=False,
        )
    plan = plan_stage_cython_backend(preparation, emission_mode=mode)
    # Preserve the caller-selected optimization label in the public plan.
    return NativeBatchPlan(**{**plan.__dict__, "optimization_level": level})

def _point_count(frontend: GraphModelCompiler, values: dict[str, Any]) -> int:
    counts: list[int] = []
    for name, spec in frontend.canonical.inputs.items():
        if spec.scope != "point" or spec.ndim != 1:
            continue
        counts.append(int(np.asarray(values[name]).shape[0]))
    frozen = len(tuple(frontend.canonical.run_keys))
    if not counts:
        if frozen <= 0:
            raise NativeBatchError("native batch requires a non-empty frozen run domain")
        return frozen
    if len(set(counts)) != 1:
        raise NativeBatchError(f"point-scoped input lengths differ: {sorted(set(counts))}")
    if counts[0] != frozen:
        raise NativeBatchError(
            f"point-scoped input length {counts[0]} differs from frozen run domain {frozen}"
        )
    return counts[0]


class NativeBatchProgram:
    def __init__(
        self,
        *,
        frontend: GraphModelCompiler,
        module: Any,
        bound_inputs: dict[str, Any],
        build_report: NativeBatchBuildReport,
        source: str,
        optimized_python_artifact: OptimizedPythonArtifact | None = None,
    ):
        self._frontend = frontend
        self.module = module
        self.bound_inputs = bound_inputs
        self.build_report = build_report
        self.source = source
        self.optimized_python_artifact = optimized_python_artifact
        self.input_order = tuple(frontend.canonical.inputs)
        self._input_specs = {
            name: (spec.scope, spec.ndim)
            for name, spec in frontend.canonical.inputs.items()
        }
        self.point_count = int(build_report.point_count)
        self._has_point_axis = any(
            scope == "point" and ndim == 1 for scope, ndim in self._input_specs.values()
        )
        self.detached = False

    def _args(self) -> list[Any]:
        """Return inputs for the exact frozen Stage run domain."""
        return [self.bound_inputs[name] for name in self.input_order]

    def run_batch(self, limit: int | None = None, *, threads: int = 1) -> np.ndarray:
        if int(threads) != 1:
            raise NativeBatchError(
                "NativeBatch is serial-only until the direct kernel shape is frozen; threads must be 1"
            )
        # StageRunDomain is part of the compiled physical contract.  A convenience
        # result limit must therefore not mutate the input domain seen by the
        # extension; execute that frozen domain and slice only the returned vector.
        got = np.asarray(self.module.run(*self._args(), 1), dtype=np.float64)
        if limit is not None:
            n = max(0, min(int(limit), self.point_count))
            got = got[:n]
        return got

    def benchmark(self, *, limit: int | None = None, repeats: int = 11) -> NativeBatchBenchmark:
        requested = self.point_count if limit is None else max(0, min(int(limit), self.point_count))
        if requested <= 0:
            raise NativeBatchError("benchmark point count must be positive")
        if requested != self.point_count:
            raise NativeBatchError(
                "benchmark limit cannot reduce a frozen Stage run domain; build a smaller artifact instead"
            )
        self.run_batch()
        samples: list[float] = []
        for _ in range(max(1, int(repeats))):
            t0 = time.perf_counter()
            self.run_batch()
            samples.append(time.perf_counter() - t0)
        med = statistics.median(samples)
        return NativeBatchBenchmark(
            point_count=self.point_count,
            median_seconds=med,
            p10_seconds=float(np.percentile(samples, 10)),
            p90_seconds=float(np.percentile(samples, 90)),
            min_seconds=min(samples),
            max_seconds=max(samples),
            per_point_seconds=med / self.point_count,
        )

    def validate_against_modelx(self, *, keys: Sequence[Any] | None = None) -> dict[str, Any]:
        """Compile-time/reference validation only; production execution never calls modelx."""
        if self.detached or self._frontend is None:
            raise NativeBatchError("reference validation is unavailable after detachment")
        full_keys = tuple(self._frontend.run_keys)
        chosen = full_keys if keys is None else tuple(keys)
        if not chosen:
            raise NativeBatchError("validation keys must not be empty")
        positions: list[int] = []
        for key in chosen:
            try:
                positions.append(full_keys.index(key))
            except ValueError as exc:
                raise NativeBatchError(
                    f"validation key {key!r} is outside the frozen Stage run domain"
                ) from exc
        got_full = np.asarray(self.module.run(*self._args(), 1), dtype=np.float64)
        got = got_full[np.asarray(positions, dtype=np.intp)]
        expected = np.asarray(
            [float(self._frontend.modelx_output_value(key)) for key in chosen],
            dtype=np.float64,
        )
        diff = np.abs(got - expected)
        scale = np.maximum(1.0, np.abs(expected))
        return {
            "point_count": len(chosen),
            "max_abs_error": float(diff.max()) if diff.size else 0.0,
            "max_scaled_relative_error": float((diff / scale).max()) if diff.size else 0.0,
            "exact": bool(np.array_equal(got, expected)),
            "allclose_1e12": bool(np.allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)),
            "checksum": float(np.sum(got, dtype=np.float64)),
        }

    def detach_for_production(self) -> None:
        """Drop live model/frontend ownership after inputs and extension are prepared."""
        if self.detached:
            return
        self._frontend = None
        self.detached = True

    def generated_runtime_is_direct(self) -> bool:
        forbidden = (
            "_ref_index(", "_out_index(", "_bind_i64(", "_bind_f64(",
            "modelx.", "self._v_", "self._has_", "._v_", "._has_",
        )
        return not any(token in self.source for token in forbidden)

    def generated_runtime_uses_direct_locals(self) -> bool:
        if self.build_report.emission_mode != "direct_locals":
            return False
        forbidden = ("eval_v_", "double cur[", "double state[", "double svals[")
        return not any(token in self.source for token in forbidden)


def build_optimized_python_batch(
    compiler,
    build_dir: str | Path,
    *,
    sample_keys: Sequence[Any] | None = None,
    run_keys: Sequence[Any] | None = None,
    optimization_level: str = "O2",
    validation_points: int = 16,
) -> OptimizedPythonArtifact:
    """Build the standalone canonical Stage Python artifact."""
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    preparation = prepare_stage_scalar(
        compiler, sample_keys=sample_keys, run_keys=run_keys,
        optimization_level=optimization_level.upper(), validation_points=validation_points,
    )
    artifact = build_optimized_python_artifact(
        preparation.stage_program, build_dir, bound_inputs=preparation.bound_inputs,
        module_name="stage_optimized_model", validation_points=validation_points,
    )
    (build_dir / "modelx_parity.json").write_text(
        __import__("json").dumps(preparation.python_validation, indent=2, sort_keys=True) + "\n"
    )
    return artifact


def _build_native_batch_from_preparation(
    preparation: StageScalarPreparation,
    build_dir: str | Path,
    *,
    optimization_level: str = "O2",
    emission_mode: str = "direct_locals",
    c_optimization: str = "O3",
    native_arch: bool = True,
    coverage_proof: RunDomainCoverageProof | None = None,
    trace_closure_rounds: int = 0,
) -> NativeBatchProgram:
    """Compile an already validated Stage preparation without rebuilding its trace."""
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    level = optimization_level.upper()
    frontend = preparation.frontend
    output_name = preparation.output_name
    bound = preparation.bound_inputs
    point_count = preparation.point_count
    program = preparation.stage_program

    t0 = time.perf_counter()
    optimized_python = build_optimized_python_artifact(
        program, build_dir / "optimized_python", bound_inputs=bound,
        module_name="stage_optimized_model", validation_points=min(16, point_count),
    )
    (optimized_python.directory / "modelx_parity.json").write_text(
        __import__("json").dumps(preparation.python_validation, indent=2, sort_keys=True) + "\n"
    )
    stage_python_seconds = preparation.stage_python_seconds + (time.perf_counter() - t0)

    build_fingerprint = program.build_fingerprint(
        backend="cython", optimization_level=level, emission_mode=emission_mode,
        c_optimization=c_optimization, native_arch=native_arch,
    )
    module_name = f"mxg_stage_batch_{build_fingerprint[:12]}"
    pyx_path = build_dir / f"{module_name}.pyx"
    t0 = time.perf_counter()
    source = StageCythonGenerator(program, emission_mode=emission_mode).write(pyx_path, module_name)
    codegen_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    module, so_path, _proc = build_extension(
        pyx_path, module_name, build_dir, openmp=False,
        optimization=c_optimization, native_arch=native_arch,
    )
    build_seconds = time.perf_counter() - t0

    input_shapes = tuple(tuple(np.asarray(bound[name]).shape) for name in program.input_order)
    history_count, ring_count, scalar_count, reduction_count = _stage_storage_metrics(program)
    report = NativeBatchBuildReport(
        point_count=point_count, output_name=output_name, optimization_level=level,
        emission_mode=str(emission_mode).lower(), c_optimization=str(c_optimization).upper(),
        native_arch=bool(native_arch), frontend_seconds=preparation.frontend_seconds,
        input_prepare_seconds=preparation.input_prepare_seconds,
        optimized_program_seconds=preparation.stage_program_seconds,
        optimized_python_seconds=stage_python_seconds, codegen_seconds=codegen_seconds,
        build_seconds=build_seconds, source_lines=source.count("\n") + 1,
        source_bytes=len(source.encode("utf-8")),
        optimized_python_path=str(optimized_python.source_path),
        optimized_program_manifest_path=str(optimized_python.manifest_path),
        optimized_inputs_path=str(optimized_python.inputs_path), pyx_path=str(pyx_path),
        so_path=str(so_path), input_names=tuple(program.input_order),
        input_shapes=input_shapes, history_count=history_count, ring_count=ring_count,
        scalar_count=scalar_count, fused_reduction_count=reduction_count,
        previous_scalar_count=0, trace_source=frontend.trace_source,
        stage_program_uid=program.uid, execution_plan_uid=program.execution_plan.uid,
        storage_plan_uid=program.storage_plan.uid, build_fingerprint=build_fingerprint,
        sample_count=len(preparation.samples),
        run_domain_coverage_proof=(
            None if coverage_proof is None else coverage_proof.manifest()
        ),
        trace_closure_rounds=int(trace_closure_rounds),
    )
    return NativeBatchProgram(
        frontend=frontend, module=module, bound_inputs=bound, build_report=report,
        source=source, optimized_python_artifact=optimized_python,
    )


def build_native_batch(
    compiler,
    build_dir: str | Path,
    *,
    sample_keys: Sequence[Any] | None = None,
    run_keys: Sequence[Any] | None = None,
    optimization_level: str = "O2",
    emission_mode: str = "direct_locals",
    c_optimization: str = "O3",
    native_arch: bool = True,
) -> NativeBatchProgram:
    """Build NativeBatch exclusively from canonical StageOptimizedProgram."""
    level = optimization_level.upper()
    preparation = prepare_stage_scalar(
        compiler, sample_keys=sample_keys, run_keys=run_keys,
        optimization_level=level, validation_points=16,
        full_run_domain_validation=True,
    )
    proof = _coverage_proof_for_preparation(preparation)
    return _build_native_batch_from_preparation(
        preparation, build_dir,
        optimization_level=level, emission_mode=emission_mode,
        c_optimization=c_optimization, native_arch=native_arch,
        coverage_proof=proof,
    )


def build_native_batch_auto_trace(
    compiler,
    build_dir: str | Path,
    *,
    run_keys: Sequence[Any],
    optimization_level: str = "O2",
    emission_mode: str = "direct_locals",
    c_optimization: str = "O3",
    native_arch: bool = True,
    validation_points: int = 16,
    expansion_batch: int = 16,
    max_rounds: int = 8,
    max_sample_keys: int = 128,
) -> NativeBatchProgram:
    """Build a complete-domain artifact after automatic representative trace closure."""
    closure = prepare_stage_scalar_auto_trace(
        compiler,
        run_keys=run_keys,
        optimization_level=optimization_level,
        validation_points=validation_points,
        expansion_batch=expansion_batch,
        max_rounds=max_rounds,
        max_sample_keys=max_sample_keys,
    )
    return _build_native_batch_from_preparation(
        closure.preparation,
        build_dir,
        optimization_level=optimization_level,
        emission_mode=emission_mode,
        c_optimization=c_optimization,
        native_arch=native_arch,
        coverage_proof=closure.coverage_proof,
        trace_closure_rounds=len(closure.rounds),
    )

