from __future__ import annotations

"""Same-graph PreparedBatch foundation for the canonical whole-artifact backend.

This is deliberately a bridge over the existing canonical graph/binding/storage
machinery, not a second compiler.  The current foundation accepts already-built
point programs in order to prove that one generated module can execute distinct
point payloads.  The next step is to prepare subsequent payloads without another
trace/build/compile.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .native_bindings import build_runtime_arg_sites, encode_runtime_bindings
from .native_family import _hybrid_value_equal
from .native_plan import build_native_family_backend_plan
from .whole_artifact import WholeArtifactProgram, _read_address


class PreparedBatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class CanonicalBindingFeasibility:
    operation_count: int
    family_count: int
    runtime_arg_sites: int
    dynamic_sites: int
    binding_kinds: dict[str, int]
    value_kinds: dict[str, int]
    encoded_instance_payloads: int
    errors: tuple[str, ...]

    @property
    def encodable(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class PreparedBatchExecution:
    point_count: int
    exact: bool | None
    values: tuple[Any, ...]
    max_abs_error: float | None


def analyze_canonical_batch_bindings(compiler) -> CanonicalBindingFeasibility:
    """Check whether realized dynamic call arguments fit the existing native ABI."""
    if compiler.structured is None:
        compiler.recover_loops()
    plan = compiler.structured.canonical_plan
    if plan is None:
        raise PreparedBatchError("canonical execution plan is unavailable")
    kinds: Counter[str] = Counter()
    values: Counter[str] = Counter()
    sites_total = dynamic = encoded = 0
    errors: list[str] = []
    for family in plan.code_families:
        backend = build_native_family_backend_plan(compiler, family.family_id)
        sites = build_runtime_arg_sites(backend)
        sites_total += len(sites)
        dynamic += sum(site.binding_kind == "dynamic" for site in sites)
        kinds.update(site.binding_kind for site in sites)
        values.update(site.value_kind for site in sites)
        for inst in backend.instances:
            try:
                encode_runtime_bindings(inst, sites)
                encoded += 1
            except Exception as exc:
                errors.append(
                    f"family={family.family_id} block={inst.block_index}: "
                    f"{type(exc).__name__}: {exc}"
                )
    return CanonicalBindingFeasibility(
        operation_count=compiler.sequential.operation_count,
        family_count=len(plan.code_families),
        runtime_arg_sites=sites_total,
        dynamic_sites=dynamic,
        binding_kinds=dict(kinds),
        value_kinds=dict(values),
        encoded_instance_payloads=encoded,
        errors=tuple(errors),
    )


def generated_source_fingerprint(program: WholeArtifactProgram) -> str:
    return hashlib.sha256(Path(program.build_report.pyx_path).read_bytes()).hexdigest()


@dataclass
class PreparedBatchFoundation:
    """Proof object for one whole-artifact module reused by distinct point states."""

    template: WholeArtifactProgram
    points: tuple[WholeArtifactProgram, ...]
    source_fingerprint: str

    @classmethod
    def from_programs(cls, programs: Iterable[WholeArtifactProgram]) -> "PreparedBatchFoundation":
        points = tuple(programs)
        if not points:
            raise PreparedBatchError("PreparedBatch requires at least one point program")
        fp = generated_source_fingerprint(points[0])
        if not hasattr(points[0].module, "run_batch"):
            raise PreparedBatchError("template module has no generated run_batch entry")
        for i, program in enumerate(points[1:], start=1):
            other = generated_source_fingerprint(program)
            if other != fp:
                raise PreparedBatchError(
                    f"point {i} generated source is not same-kernel compatible: {other} != {fp}"
                )
        return cls(points[0], points, fp)

    @property
    def point_count(self) -> int:
        return len(self.points)

    def reset(self) -> None:
        for program in self.points:
            program.reset(production=False)

    def run(self, *, validate: bool = True) -> PreparedBatchExecution:
        if validate and any(not p.expected_targets for p in self.points):
            raise PreparedBatchError("validation reference values are unavailable")
        self.reset()
        # Exactly one Python -> Cython batch entry for all point payloads.
        self.template.module.run_batch(
            [p.runtime_payload() for p in self.points],
            True,
        )
        values: list[Any] = []
        errors: list[float] = []
        ok = True
        for program in self.points:
            vals = tuple(
                _read_address(a, program.dslots, program.islots, program.bslots, program.oslots)
                for a in program.targets
            )
            values.append(vals[0] if len(vals) == 1 else vals)
            if validate:
                ok = ok and all(
                    _hybrid_value_equal(a, e)
                    for a, e in zip(vals, program.expected_targets)
                )
                for a, e in zip(vals, program.expected_targets):
                    try:
                        errors.append(abs(float(a) - float(e)))
                    except Exception:
                        pass
        exact = bool(ok) if validate else None
        max_abs = max(errors, default=0.0) if validate else None
        if validate and not exact:
            raise PreparedBatchError(
                f"shared-module PreparedBatch mismatch; max_abs_error={max_abs}"
            )
        return PreparedBatchExecution(len(values), exact, tuple(values), max_abs)
