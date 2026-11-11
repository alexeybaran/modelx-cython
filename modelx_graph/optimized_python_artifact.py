from __future__ import annotations

"""Standalone optimized-Python artifact writer.

The emitted directory requires Python + NumPy only.  It does not import modelx,
modelx_graph or Cython at runtime.
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .stage_optimized_program import StageOptimizedProgram
from .stage_codegen import StagePythonGenerator


class OptimizedPythonArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizedPythonArtifact:
    program: Any
    directory: Path
    source_path: Path
    manifest_path: Path
    inputs_path: Path
    runner_path: Path
    validation_path: Path
    source: str

    def load_inputs(self) -> dict[str, Any]:
        with np.load(self.inputs_path, allow_pickle=False) as data:
            out: dict[str, Any] = {}
            for key in self.program.input_order:
                value = np.asarray(data[key])
                out[key] = value.item() if value.ndim == 0 else value
            return out

    def execute(self) -> np.ndarray:
        spec = importlib.util.spec_from_file_location("_mxg_optimized_python_artifact", self.source_path)
        if spec is None or spec.loader is None:
            raise OptimizedPythonArtifactError("cannot load generated optimized Python module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return np.asarray(mod.run(self.load_inputs()), dtype=np.float64)


def _save_inputs(path: Path, program: Any, bound_inputs: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    arrays = {}
    shapes = []
    for key in program.input_order:
        if key not in bound_inputs:
            raise OptimizedPythonArtifactError(f"missing bound input {key!r}")
        arr = np.asarray(bound_inputs[key])
        if arr.dtype == object:
            raise OptimizedPythonArtifactError(
                f"standalone optimized Python artifact requires numeric input {key!r}"
            )
        arrays[key] = arr
        shapes.append(tuple(int(x) for x in arr.shape))
    np.savez(path, **arrays)
    return tuple(shapes)


def _validation_inputs(
    program: Any, bound_inputs: dict[str, Any], limit: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in program.input_order:
        spec = program.input_spec(key)
        value = bound_inputs[key]
        if spec.scope == "point" and spec.ndim == 1:
            out[key] = value[:limit]
        else:
            out[key] = value
    return out


def _execute_source(source: str, inputs: dict[str, Any]) -> np.ndarray:
    ns: dict[str, Any] = {}
    exec(compile(source, "<optimized-python-artifact>", "exec"), ns)
    return np.asarray(ns["run"](inputs), dtype=np.float64)


def build_optimized_python_artifact(
    program: Any,
    build_dir: str | Path,
    *,
    bound_inputs: dict[str, Any] | None = None,
    module_name: str = "optimized_model",
    validation_points: int = 16,
) -> OptimizedPythonArtifact:
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    if bound_inputs is None:
        canonical = getattr(program, "canonical", None)
        if canonical is None:
            raise OptimizedPythonArtifactError(
                "bound_inputs are required for standalone StageOptimizedProgram artifacts"
            )
        bound_inputs = canonical.bind_inputs()

    source_path = build_dir / f"{module_name}.py"
    manifest_path = build_dir / "program_manifest.json"
    inputs_path = build_dir / "inputs.npz"
    runner_path = build_dir / "run_artifact.py"
    validation_path = build_dir / "validation.json"

    if isinstance(program, StageOptimizedProgram):
        source = StagePythonGenerator(program).write(source_path, module_name)
    else:
        # Explicit legacy artifact compatibility; production NativeBatch reaches
        # only the Stage branch above and therefore never imports mature emitters.
        from .python_codegen import PythonLoopGenerator
        source = PythonLoopGenerator(program).write(source_path, module_name)
    input_shapes = _save_inputs(inputs_path, program, bound_inputs)

    manifest = program.manifest()
    manifest.update({
        "runtime_requirements": ["python", "numpy"],
        "source_file": source_path.name,
        "inputs_file": inputs_path.name,
        "runner_file": runner_path.name,
        "input_shapes": {k: list(shape) for k, shape in zip(program.input_order, input_shapes)},
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    runner_path.write_text(
        "from pathlib import Path\n"
        "import numpy as np\n"
        f"from {module_name} import run\n\n"
        "def load_inputs():\n"
        "    path = Path(__file__).with_name('inputs.npz')\n"
        "    with np.load(path, allow_pickle=False) as data:\n"
        "        return {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}\n\n"
        "if __name__ == '__main__':\n"
        "    result = np.asarray(run(load_inputs()), dtype=np.float64)\n"
        "    print(result)\n"
    )

    artifact = OptimizedPythonArtifact(
        program=program,
        directory=build_dir,
        source_path=source_path,
        manifest_path=manifest_path,
        inputs_path=inputs_path,
        runner_path=runner_path,
        validation_path=validation_path,
        source=source,
    )
    limit = max(1, int(validation_points))
    if isinstance(program, StageOptimizedProgram):
        # StageRunDomain cardinality is part of the frozen program contract.  A
        # validation convenience slice must not mutate point-scoped input lengths;
        # execute the complete frozen domain and slice only the returned vector.
        got_all = _execute_source(source, bound_inputs)
        got = got_all[: min(limit, got_all.size)]
        validation_scope = "frozen_run_domain_then_leading_result_slice"
        execution_point_count = int(got_all.size)
    else:
        check_inputs = _validation_inputs(program, bound_inputs, limit)
        got = _execute_source(source, check_inputs)
        validation_scope = "leading_point_slice"
        execution_point_count = int(got.size)
    validation_path.write_text(json.dumps({
        "execution_ok": True,
        "validation_scope": validation_scope,
        "point_count": int(got.size),
        "execution_point_count": execution_point_count,
        "requested_validation_points": limit,
        "checksum": float(np.sum(got, dtype=np.float64)),
        "min": None if not got.size else float(np.min(got)),
        "max": None if not got.size else float(np.max(got)),
    }, indent=2, sort_keys=True) + "\n")
    return artifact
