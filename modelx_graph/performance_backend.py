from __future__ import annotations

"""Measured performance selector with modelx-cython as the required floor.

This stage intentionally adds no broad new formula semantics.  It reuses the
existing exact native batch fast path when available and reuses modelx-cython
unchanged as the performance floor/fallback.  A native candidate is selected
only after fresh-process execution, full-portfolio numerical comparison and an
end-to-end parity gate that includes candidate input preparation.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import importlib.util
import json
import math
import os
import pickle
import statistics
import struct
import subprocess
import sys
import time

import numpy as np

from .fast_foundation import FastFoundationError, FastFoundationProgram
from .modelx_cython_bridge import (
    ModelxCythonBridgeError,
    ModelxCythonBridgeProgram,
    ModelxCythonTarget,
    build_modelx_cython_bridge,
    derive_modelx_cython_trace_sample,
    infer_modelx_cython_target,
)


class PerformanceBackendError(RuntimeError):
    pass


class PerformanceBackendKind(str, Enum):
    NATIVE_BATCH = "native_batch"
    MIXED_RECURSIVE = "mixed_recursive"
    MODELX_CYTHON_REUSE = "modelx_cython_reuse"


@dataclass(frozen=True)
class PerformanceBuildReport:
    backend: str
    fast_plan_available: bool
    fast_plan_reason: str | None
    proof_points: int
    proof_exact: bool
    backend_report: dict[str, Any]


@dataclass
class PerformanceArtifactProgram:
    backend_kind: PerformanceBackendKind
    target: ModelxCythonTarget
    portfolio_points: tuple[tuple[Any, ...], ...]
    build_report: PerformanceBuildReport
    native_program: FastFoundationProgram | None = None
    modelx_cython_program: ModelxCythonBridgeProgram | None = None

    def execute_prefix(self, count: int) -> np.ndarray:
        count = min(int(count), len(self.portfolio_points))
        if self.backend_kind is PerformanceBackendKind.NATIVE_BATCH:
            if self.native_program is None:
                raise PerformanceBackendError("selected native program is missing")
            return np.asarray(self.native_program.execute_prefix(count))
        if self.modelx_cython_program is None:
            raise PerformanceBackendError("selected modelx-cython program is missing")
        return np.asarray(self.modelx_cython_program.execute_points(self.portfolio_points[:count]))

    def execute(self) -> np.ndarray:
        return self.execute_prefix(len(self.portfolio_points))

    def detach_for_production(self, *, trim_allocator: bool = False) -> dict[str, Any]:
        if self.backend_kind is not PerformanceBackendKind.NATIVE_BATCH:
            return {
                "detached": False,
                "reason": "modelx-cython owns its generated model/cache runtime",
                "point_count": len(self.portfolio_points),
            }
        assert self.native_program is not None
        return self.native_program.detach_for_production(
            len(self.portfolio_points), trim_allocator=trim_allocator
        )


def _normalize_shapes(candidate: np.ndarray, floor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(candidate)
    b = np.asarray(floor)
    if a.size == b.size:
        return a.reshape(-1), b.reshape(-1)
    return a, b


def _ulp_distance_scalar(a: float, b: float) -> int:
    if math.isnan(a) and math.isnan(b):
        return 0
    if a == b:
        return 0
    if not math.isfinite(a) or not math.isfinite(b):
        return 2**64 - 1
    ia = struct.unpack(">q", struct.pack(">d", float(a)))[0]
    ib = struct.unpack(">q", struct.pack(">d", float(b)))[0]
    if ia < 0:
        ia = 0x8000000000000000 - ia
    if ib < 0:
        ib = 0x8000000000000000 - ib
    return abs(ia - ib)


def _exact_comparison(
    candidate: Any, floor: Any, *, ulp_tolerance: float = 4.0
) -> dict[str, Any]:
    raw_a = np.asarray(candidate)
    raw_b = np.asarray(floor)
    a, b = _normalize_shapes(raw_a, raw_b)
    same_size = a.size == b.size
    exact = bool(same_size and np.array_equal(a, b, equal_nan=True))
    max_abs_error = float("inf")
    max_ulp_error = float("inf")
    if same_size:
        af = np.asarray(a, dtype=np.float64).reshape(-1)
        bf = np.asarray(b, dtype=np.float64).reshape(-1)
        finite = np.isfinite(af) & np.isfinite(bf)
        nan_equal = np.isnan(af) & np.isnan(bf)
        incompatible = ~(finite | nan_equal | (af == bf))
        if np.any(incompatible):
            max_abs_error = float("inf")
            max_ulp_error = float("inf")
        else:
            if np.any(finite):
                max_abs_error = float(np.max(np.abs(af[finite] - bf[finite])))
                mismatched = finite & (af != bf)
                max_ulp_error = float(max(
                    (_ulp_distance_scalar(x, y) for x, y in zip(af[mismatched], bf[mismatched])),
                    default=0,
                ))
            else:
                max_abs_error = 0.0
                max_ulp_error = 0.0
    within = bool(same_size and max_ulp_error <= float(ulp_tolerance))
    return {
        "exact": exact,
        "accepted": bool(exact or within),
        "max_abs_error": max_abs_error,
        "max_ulp_error": max_ulp_error,
        "ulp_tolerance": float(ulp_tolerance),
        "within_ulp_tolerance": within,
        "candidate_shape": list(raw_a.shape),
        "floor_shape": list(raw_b.shape),
        "candidate_size": int(raw_a.size),
        "floor_size": int(raw_b.size),
        "candidate_checksum": float(np.nansum(raw_a, dtype=np.float64)),
        "floor_checksum": float(np.nansum(raw_b, dtype=np.float64)),
    }


def _selection_record(
    *,
    candidate_kind: PerformanceBackendKind,
    candidate_benchmark: dict[str, Any],
    floor_benchmark: dict[str, Any],
    exact_comparison: dict[str, Any],
    max_candidate_ratio: float,
) -> dict[str, Any]:
    c = float(candidate_benchmark["median_seconds"])
    f = float(floor_benchmark["median_seconds"])
    ce = float(candidate_benchmark.get("end_to_end_seconds", c))
    fe = float(floor_benchmark.get("end_to_end_seconds", f))
    ratio = c / f if f else float("inf")
    end_ratio = ce / fe if fe else float("inf")
    passed = bool(
        exact_comparison.get("accepted")
        and ratio <= float(max_candidate_ratio)
        and end_ratio <= float(max_candidate_ratio)
    )
    return {
        "candidate_backend": candidate_kind.value,
        "candidate": candidate_benchmark,
        "modelx_cython_floor": floor_benchmark,
        "exact_comparison": exact_comparison,
        "candidate_to_floor_ratio": ratio,
        "candidate_end_to_end_seconds": ce,
        "floor_end_to_end_seconds": fe,
        "candidate_to_floor_end_to_end_ratio": end_ratio,
        "max_candidate_ratio": float(max_candidate_ratio),
        "parity_gate_passed": passed,
        "selected": (
            candidate_kind.value if passed else PerformanceBackendKind.MODELX_CYTHON_REUSE.value
        ),
    }


def select_against_modelx_cython_floor(
    *,
    candidate_kind: PerformanceBackendKind,
    candidate_benchmark: Mapping[str, Any] | dict[str, Any],
    floor_benchmark: Mapping[str, Any] | dict[str, Any],
    candidate_values: Any,
    floor_values: Any,
    max_candidate_ratio: float = 0.95,
    ulp_tolerance: float = 8.0,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Apply the measured modelx-cython performance floor to a candidate.

    Runtime selection uses the compiler's established numerical parity contract
    (``allclose`` with explicit rtol/atol) rather than requiring bitwise identity
    across different generated arithmetic schedules.  Exact/ULP comparison is
    still retained as a diagnostic and can independently satisfy the gate.
    A candidate must then beat modelx-cython by the configured safety margin;
    otherwise modelx-cython is selected.
    """

    if candidate_kind is PerformanceBackendKind.MODELX_CYTHON_REUSE:
        raise PerformanceBackendError("modelx-cython cannot be its own candidate")
    comparison = _exact_comparison(
        candidate_values, floor_values, ulp_tolerance=ulp_tolerance
    )
    raw_candidate = np.asarray(candidate_values)
    raw_floor = np.asarray(floor_values)
    same_size = raw_candidate.size == raw_floor.size
    strict_allclose = bool(
        same_size
        and np.allclose(
            raw_candidate.reshape(-1), raw_floor.reshape(-1),
            rtol=float(rtol), atol=float(atol), equal_nan=True,
        )
    )
    comparison.update({
        "strict_allclose": strict_allclose,
        "rtol": float(rtol),
        "atol": float(atol),
        "accepted_by": (
            "exact_or_ulp" if comparison.get("accepted")
            else "rtol_atol" if strict_allclose
            else "rejected"
        ),
    })
    comparison["accepted"] = bool(comparison.get("accepted") or strict_allclose)
    return _selection_record(
        candidate_kind=candidate_kind,
        candidate_benchmark=dict(candidate_benchmark),
        floor_benchmark=dict(floor_benchmark),
        exact_comparison=comparison,
        max_candidate_ratio=float(max_candidate_ratio),
    )


def _subprocess_environment(*prepend: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(Path(p).resolve()) for p in prepend]
    paths.extend(p for p in sys.path if p)
    if env.get("PYTHONPATH"):
        paths.extend(x for x in env["PYTHONPATH"].split(os.pathsep) if x)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


_NATIVE_WORKER = r'''
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np

def status_kb():
    out = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value = line.split(':', 1)
            out[key] = int(value.strip().split()[0])
    return out

ap = argparse.ArgumentParser()
ap.add_argument('--so', required=True)
ap.add_argument('--inputs', required=True)
ap.add_argument('--count', required=True, type=int)
ap.add_argument('--point-inputs', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--values')
args = ap.parse_args()
so = Path(args.so)
name = so.name.split('.cpython')[0].split('.pypy')[0]
t0 = time.perf_counter()
spec = importlib.util.spec_from_file_location(name, so)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module_load = time.perf_counter() - t0
t0 = time.perf_counter()
raw = np.load(args.inputs)
point_inputs = set(json.loads(args.point_inputs))
inputs = {name: (raw[name][:args.count] if name in point_inputs else raw[name]) for name in raw.files}
input_load = time.perf_counter() - t0
before = status_kb()
t0 = time.perf_counter()
values = np.asarray(module.run(**inputs, threads=1))
seconds = time.perf_counter() - t0
after = status_kb()
if args.values:
    np.save(args.values, values)
result = {
    'seconds': seconds,
    'module_load_seconds': module_load,
    'input_load_seconds': input_load,
    'checksum': float(np.nansum(values, dtype=np.float64)),
    'shape': list(values.shape),
    'rss_before_kb': before.get('VmRSS'),
    'rss_after_kb': after.get('VmRSS'),
    'vmhwm_kb': after.get('VmHWM'),
}
Path(args.out).write_text(json.dumps(result, indent=2))
'''


_FLOOR_WORKER = r'''
from __future__ import annotations
import argparse, importlib, json, pickle, time
from pathlib import Path
import numpy as np

def status_kb():
    out = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, value = line.split(':', 1)
            out[key] = int(value.strip().split()[0])
    return out

ap = argparse.ArgumentParser()
ap.add_argument('--payload', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--values')
args = ap.parse_args()
payload = pickle.loads(Path(args.payload).read_bytes())
t0 = time.perf_counter()
module = importlib.import_module(payload['package'])
model = module.mx_model
load_seconds = time.perf_counter() - t0
space = model
for name in payload['space_path']:
    space = getattr(space, name)

def get_cell(point):
    if payload['point_parameter_count'] == 0:
        item = space
    elif payload['point_parameter_count'] == 1:
        item = space[point[0]]
    else:
        item = space[tuple(point)]
    return getattr(item, payload['cell_name'])

before = status_kb()
t0 = time.perf_counter()
values = np.asarray([get_cell(point)(*payload['cell_args']) for point in payload['points']])
seconds = time.perf_counter() - t0
after = status_kb()
t0 = time.perf_counter()
warm = np.asarray([get_cell(point)(*payload['cell_args']) for point in payload['points']])
warm_seconds = time.perf_counter() - t0
if not np.array_equal(values, warm, equal_nan=True):
    raise RuntimeError('modelx-cython warm cache changed values')
if args.values:
    np.save(args.values, values)
result = {
    'seconds': seconds,
    'warm_cache_seconds': warm_seconds,
    'load_seconds': load_seconds,
    'checksum': float(np.nansum(values, dtype=np.float64)),
    'shape': list(values.shape),
    'rss_before_kb': before.get('VmRSS'),
    'rss_after_kb': after.get('VmRSS'),
    'vmhwm_kb': after.get('VmHWM'),
}
Path(args.out).write_text(json.dumps(result, indent=2))
'''


def _run_worker(cmd: list[str], *, env: dict[str, str], timeout_seconds: float) -> dict[str, Any]:
    cp = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout_seconds
    )
    if cp.returncode:
        raise PerformanceBackendError(
            f"benchmark worker failed ({cp.returncode}): {cp.stderr[-4000:]}"
        )
    out_index = cmd.index("--out") + 1
    return json.loads(Path(cmd[out_index]).read_text())


def _aggregate_samples(samples: list[dict[str, Any]], *, prepare_seconds: float = 0.0) -> dict[str, Any]:
    med = float(statistics.median(x["seconds"] for x in samples))
    result = {
        "median_seconds": med,
        "min_seconds": float(min(x["seconds"] for x in samples)),
        "max_seconds": float(max(x["seconds"] for x in samples)),
        "prepare_seconds": float(prepare_seconds),
        "end_to_end_seconds": float(prepare_seconds + med),
        "samples": samples,
        "checksum": float(samples[0]["checksum"]),
        "shape": list(samples[0]["shape"]),
        "median_rss_before_kb": float(statistics.median(x["rss_before_kb"] for x in samples)),
        "median_rss_after_kb": float(statistics.median(x["rss_after_kb"] for x in samples)),
        "median_peak_rss_kb": float(statistics.median(x["vmhwm_kb"] for x in samples)),
    }
    if "warm_cache_seconds" in samples[0]:
        result["median_warm_cache_seconds"] = float(
            statistics.median(x["warm_cache_seconds"] for x in samples)
        )
    return result


def _benchmark_native_cold(
    program: FastFoundationProgram,
    build_dir: Path,
    count: int,
    *,
    repeats: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], np.ndarray]:
    build_dir.mkdir(parents=True, exist_ok=True)
    program._prepared_cache.pop(count, None)
    t0 = time.perf_counter()
    inputs = program.prepare_inputs(count)
    prepare_seconds = time.perf_counter() - t0
    point_inputs = tuple(
        name for name, value in inputs.items()
        if name.startswith("mp__") and value.ndim and value.shape[0] == count
    )
    inputs_path = build_dir / "native_inputs.npz"
    np.savez(inputs_path, **inputs)
    worker = build_dir / "native_worker.py"
    worker.write_text(_NATIVE_WORKER)
    samples = []
    values_path = build_dir / "native_values.npy"
    env = _subprocess_environment(Path(program.build_report.so_path).parent)
    for i in range(repeats):
        out = build_dir / f"native_sample_{i}.json"
        cmd = [
            sys.executable, str(worker),
            "--so", program.build_report.so_path,
            "--inputs", str(inputs_path),
            "--count", str(count),
            "--point-inputs", json.dumps(point_inputs),
            "--out", str(out),
        ]
        if i == 0:
            cmd += ["--values", str(values_path)]
        samples.append(_run_worker(cmd, env=env, timeout_seconds=timeout_seconds))
    result = _aggregate_samples(samples, prepare_seconds=prepare_seconds)
    result.update({
        "point_count": count,
        "input_bytes": int(sum(value.nbytes for value in inputs.values())),
        "point_input_names": list(point_inputs),
        "values_path": str(values_path),
        "inputs_path": str(inputs_path),
        "measurement": "first native calculation in a fresh process; module/input load excluded",
    })
    return result, np.load(values_path)


def _benchmark_floor_cold(
    program: ModelxCythonBridgeProgram,
    points: Sequence[tuple[Any, ...]],
    build_dir: Path,
    *,
    repeats: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], np.ndarray]:
    build_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "package": program.package_name,
        "space_path": program.target.space_path,
        "point_parameter_count": program.target.point_parameter_count,
        "cell_name": program.target.cell_name,
        "cell_args": program.target.cell_args,
        "points": tuple(points),
    }
    payload_path = build_dir / "floor_payload.pkl"
    payload_path.write_bytes(pickle.dumps(payload, protocol=5))
    worker = build_dir / "floor_worker.py"
    worker.write_text(_FLOOR_WORKER)
    values_path = build_dir / "floor_values.npy"
    samples = []
    env = _subprocess_environment(program.root)
    for i in range(repeats):
        out = build_dir / f"floor_sample_{i}.json"
        cmd = [
            sys.executable, str(worker),
            "--payload", str(payload_path),
            "--out", str(out),
        ]
        if i == 0:
            cmd += ["--values", str(values_path)]
        samples.append(_run_worker(cmd, env=env, timeout_seconds=timeout_seconds))
    result = _aggregate_samples(samples)
    result.update({
        "point_count": len(points),
        "values_path": str(values_path),
        "measurement": "first calculation in a fresh process; package import excluded",
    })
    return result, np.load(values_path)


def _normalize_portfolio_points(
    target: ModelxCythonTarget, point_keys: Iterable[Any]
) -> tuple[tuple[Any, ...], ...]:
    return tuple(target.normalize_point(p) for p in point_keys)


def build_performance_artifact(
    compiler: Any,
    build_dir: str | Path,
    *,
    portfolio_point_keys: Iterable[Any],
    modelx_cython_trace_point_keys: Iterable[Any] | None = None,
    proof_points: int = 100,
    selection_repeats: int = 3,
    max_candidate_ratio: float = 0.95,
    max_trace_points: int = 128,
    floor_ulp_tolerance: float = 8.0,
    modelx_cython_build_timeout_seconds: float | None = 600.0,
    execution_timeout_seconds: float = 180.0,
    force_modelx_cython_rebuild: bool = False,
    trim_allocator: bool = True,
) -> PerformanceArtifactProgram:
    """Build the fastest exact backend, with modelx-cython as measured floor."""
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    progress_path = build_dir / "performance_progress.log"
    def progress(message: str) -> None:
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.time():.6f} {message}\n")
    progress("start")
    target = infer_modelx_cython_target(compiler)
    progress("target_inferred")
    portfolio = _normalize_portfolio_points(target, portfolio_point_keys)
    if not portfolio:
        raise PerformanceBackendError("portfolio_point_keys must not be empty")

    progress("portfolio_normalized")
    fast_plan = compiler.plan_fast_foundation()
    progress(f"fast_plan available={fast_plan.available}")
    native: FastFoundationProgram | None = None
    native_validation: dict[str, Any] | None = None
    fast_reason = fast_plan.reference_frontend_reason
    if fast_plan.available:
        try:
            progress("native_build_start")
            native = compiler.build_fast_foundation(build_dir / "native_batch", native_arch=False)
            progress("native_build_done")
            native_validation = native.validate_points(min(proof_points, len(portfolio)))
            progress(f"native_validation exact={native_validation['exact']}")
            if not native_validation["exact"]:
                native = None
                fast_reason = "native proof was not bitwise exact"
        except (FastFoundationError, Exception) as exc:
            native = None
            fast_reason = f"native build failed: {type(exc).__name__}: {exc}"

    progress("trace_sample_start")
    trace_points, trace_profile = derive_modelx_cython_trace_sample(
        compiler,
        portfolio,
        explicit_point_keys=tuple(modelx_cython_trace_point_keys or ()),
        max_points=max_trace_points,
    )
    progress(f"trace_sample_done count={len(trace_points)}")
    progress("floor_build_start")
    floor = build_modelx_cython_bridge(
        compiler,
        build_dir / "modelx_cython",
        sample_point_keys=trace_points,
        force_rebuild=force_modelx_cython_rebuild,
        build_timeout_seconds=modelx_cython_build_timeout_seconds,
    )
    progress("floor_build_done")
    proof_subset = portfolio[: min(proof_points, len(portfolio))]
    progress("floor_validation_start")
    floor_validation = floor.validate_against(compiler, proof_subset)
    floor_comparison = _exact_comparison(
        floor_validation["actual"],
        floor_validation["expected"],
        ulp_tolerance=floor_ulp_tolerance,
    )
    floor_validation.update(floor_comparison)
    progress(
        "floor_validation_done "
        f"exact={floor_validation['exact']} accepted={floor_validation['accepted']} "
        f"max_ulp={floor_validation['max_ulp_error']}"
    )
    if not floor_validation["accepted"]:
        raise PerformanceBackendError(
            "modelx-cython floor failed live-model numerical proof: "
            f"max_ulp={floor_validation['max_ulp_error']} "
            f"tolerance={floor_ulp_tolerance}"
        )

    progress("floor_benchmark_start")
    floor_bench, floor_values = _benchmark_floor_cold(
        floor,
        portfolio,
        build_dir / "selection_floor",
        repeats=selection_repeats,
        timeout_seconds=execution_timeout_seconds,
    )
    progress("floor_benchmark_done")
    if native is None:
        report = PerformanceBuildReport(
            backend=PerformanceBackendKind.MODELX_CYTHON_REUSE.value,
            fast_plan_available=bool(fast_plan.available),
            fast_plan_reason=fast_reason,
            proof_points=len(proof_subset),
            proof_exact=True,
            backend_report={
                "selection": {
                    "selected": PerformanceBackendKind.MODELX_CYTHON_REUSE.value,
                    "parity_gate_passed": True,
                    "selection_reason": fast_reason or "native candidate unavailable",
                    "modelx_cython_floor": floor_bench,
                },
                "modelx_cython_build": asdict(floor.build_report),
                "modelx_cython_trace_profile": trace_profile,
                "validation": {"modelx_cython_vs_live_model": _strip_arrays(floor_validation)},
            },
        )
        return PerformanceArtifactProgram(
            PerformanceBackendKind.MODELX_CYTHON_REUSE,
            target,
            portfolio,
            report,
            modelx_cython_program=floor,
        )

    progress("native_benchmark_start")
    native_bench, native_values = _benchmark_native_cold(
        native,
        build_dir / "selection_native",
        len(portfolio),
        repeats=selection_repeats,
        timeout_seconds=execution_timeout_seconds,
    )
    progress("native_benchmark_done")
    comparison = _exact_comparison(native_values, floor_values)
    progress(f"comparison exact={comparison['exact']}")
    selection = _selection_record(
        candidate_kind=PerformanceBackendKind.NATIVE_BATCH,
        candidate_benchmark=native_bench,
        floor_benchmark=floor_bench,
        exact_comparison=comparison,
        max_candidate_ratio=max_candidate_ratio,
    )
    selected = PerformanceBackendKind(selection["selected"])
    detach = None
    if selected is PerformanceBackendKind.NATIVE_BATCH:
        progress("detach_start")
        detach = native.detach_for_production(len(portfolio), trim_allocator=trim_allocator)
        progress("detach_done")
    report = PerformanceBuildReport(
        backend=selected.value,
        fast_plan_available=True,
        fast_plan_reason=fast_reason,
        proof_points=len(proof_subset),
        proof_exact=bool(native_validation and native_validation["exact"]),
        backend_report={
            "operation_count": native.build_report.operation_count,
            "point_count": len(portfolio),
            "output_count": native.build_report.output_count,
            "source_lines": native.build_report.source_lines,
            "source_bytes": native.build_report.source_bytes,
            "pyx_path": native.build_report.pyx_path,
            "so_path": native.build_report.so_path,
            "build_seconds": native.build_report.build_seconds,
            "input_names": list(native.build_report.input_names),
            "input_shapes": [list(x) for x in native.build_report.input_shapes],
            "validation": {
                "live_model": _strip_arrays(native_validation),
                "portfolio_vs_modelx_cython": comparison,
                "modelx_cython_vs_live_model": _strip_arrays(floor_validation),
            },
            "selection": selection,
            "modelx_cython_build": asdict(floor.build_report),
            "modelx_cython_trace_profile": trace_profile,
            "production_detach": detach,
        },
    )
    progress(f"done selected={selected.value}")
    return PerformanceArtifactProgram(
        selected,
        target,
        portfolio,
        report,
        native_program=native if selected is PerformanceBackendKind.NATIVE_BATCH else None,
        modelx_cython_program=floor if selected is PerformanceBackendKind.MODELX_CYTHON_REUSE else None,
    )


def _strip_arrays(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {k: v for k, v in value.items() if k not in {"actual", "expected"}}
