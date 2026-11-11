from __future__ import annotations

"""Runtime support for generated stage-aware Python programs."""

from types import SimpleNamespace
from typing import Any, Mapping

from .stage_optimized_program import decode_runtime_payload
from .staged_python import ExternalCallResolver, ExternalValueResolver, StagedPythonExecutionResult, StagedPythonExecutor


def run_serialized_stage_program(
    encoded_payload: str,
    input_values: Mapping[str, Any],
    *,
    external_call: ExternalCallResolver | None = None,
    external_value: ExternalValueResolver | None = None,
) -> StagedPythonExecutionResult:
    payload = decode_runtime_payload(encoded_payload)
    snapshot = SimpleNamespace(variants=payload.variants)
    executor = StagedPythonExecutor(
        snapshot,
        payload.graph,
        payload.execution_plan,
        storage_plan=payload.storage_plan,
        physical_values=payload.physical_values,
        stage_program_uid=payload.program_uid,
        input_values=input_values,
        external_call=external_call,
        external_value=external_value,
    )
    return executor.run()
