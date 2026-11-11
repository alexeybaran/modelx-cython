from __future__ import annotations

"""Python/Cython emission from :class:`StageOptimizedProgram`.

One-stage programs emit only from the frozen Stage physical contract.  Mature
emitters are comparison tools and are intentionally absent from this production
module.  The production path prefers the frozen direct physical contract, including
admitted completion barriers; the serialized two-stage runtime remains comparison-only
compatibility for older non-direct shapes.
"""

from pathlib import Path
from .codegen_common import CodegenError
from .stage_optimized_program import StageOptimizedProgram
from .stage_direct_codegen import StageDirectSourceBuilder


class StagePythonGenerator:
    def __init__(self, program: StageOptimizedProgram):
        self.program = program
        if not program.python_supported:
            raise CodegenError(
                "stage optimized program is not supported by Python backend: "
                + "; ".join(program.capability_blockers)
            )

    def source(self, module_name: str = "stage_graph_python") -> str:
        p = self.program
        if p.direct_python_supported:
            return StageDirectSourceBuilder(p).python_source(module_name)
        if p.stage_indices != (0, 1):
            raise CodegenError(
                f"stage Python emitter cannot lower stages {p.stage_indices!r}: "
                + "; ".join(p.direct_python_blockers)
            )
        encoded = p.encoded_runtime_payload()
        return (
            "from __future__ import annotations\n"
            "from modelx_graph.stage_runtime import run_serialized_stage_program\n\n"
            f"STAGE_PROGRAM_UID = {p.uid!r}\n"
            f"STAGE_EXECUTION_PLAN_UID = {p.execution_plan.uid!r}\n"
            f"STAGE_STORAGE_PLAN_UID = {p.storage_plan.uid!r}\n"
            f"_STAGE_PAYLOAD = {encoded!r}\n\n"
            "def run(inputs, external_value=None, external_call=None, return_details=False):\n"
            "    result = run_serialized_stage_program(\n"
            "        _STAGE_PAYLOAD, inputs, external_call=external_call, external_value=external_value\n"
            "    )\n"
            "    return result if return_details else result.output\n"
        )

    def write(self, path: str | Path, module_name: str = "stage_graph_python") -> str:
        path = Path(path)
        if self.program.direct_python_supported:
            source = StageDirectSourceBuilder(self.program).python_source(module_name)
            path.write_text(source)
            return source
        source = self.source(module_name)
        path.write_text(source)
        return source


class StageCythonGenerator:
    def __init__(self, program: StageOptimizedProgram, *, emission_mode: str = "array_abi"):
        self.program = program
        self.emission_mode = emission_mode
        if not program.cython_supported:
            raise CodegenError(
                "stage optimized program is not supported by Cython backend: "
                + "; ".join(program.capability_blockers)
            )

    def write(self, path: str | Path, module_name: str = "stage_graph_template") -> str:
        path = Path(path)
        if not self.program.direct_cython_supported:
            raise CodegenError(
                "canonical Stage program has no direct Cython lowering: "
                + "; ".join(self.program.direct_cython_blockers)
            )
        source = StageDirectSourceBuilder(self.program).cython_source(module_name)
        path.write_text(source)
        return source
