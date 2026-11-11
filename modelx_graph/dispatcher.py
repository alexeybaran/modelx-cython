from __future__ import annotations

"""Runtime guards and finite-variant dispatch for compiled graph templates.

The dispatcher is intentionally outside the hot numerical kernel.  It validates
artifact assumptions and either calls the compiled variant or falls back to
modelx.  Optional trace validation is expensive and is intended for first-use,
audit, or speculative variant selection rather than every portfolio run.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .ir import build_graph_ir
from .trace import capture_trace


CompiledCallable = Callable[[dict[str, Any], int], np.ndarray]
FallbackCallable = Callable[[tuple[Any, ...]], np.ndarray]


@dataclass(frozen=True)
class GuardDecision:
    ok: bool
    reasons: tuple[str, ...]
    keys_checked: int
    trace_checked: bool
    runtime_variants: int = 0
    runtime_edges: int = 0


@dataclass
class GuardStats:
    compiled_runs: int = 0
    fallback_runs: int = 0
    checks: int = 0
    trace_checks: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)

    def manifest(self) -> dict[str, Any]:
        total = self.compiled_runs + self.fallback_runs
        return {
            "compiled_runs": self.compiled_runs,
            "fallback_runs": self.fallback_runs,
            "checks": self.checks,
            "trace_checks": self.trace_checks,
            "native_coverage": (self.compiled_runs / total if total else None),
            "fallback_reasons": dict(self.fallback_reasons),
        }


def _iter_spaces(model):
    stack = list(getattr(model, "spaces", {}).values())
    seen: set[int] = set()
    while stack:
        space = stack.pop()
        marker = id(getattr(space, "_impl", space))
        if marker in seen:
            continue
        seen.add(marker)
        yield space
        stack.extend(getattr(space, "spaces", {}).values())


def _current_formula_sources(model) -> dict[str, str]:
    result: dict[str, str] = {}
    for space in _iter_spaces(model):
        for cell in getattr(space, "cells", {}).values():
            formula = getattr(cell, "formula", None)
            if formula is None:
                continue
            try:
                result[cell.fullname] = formula.source
            except Exception:
                continue
    return result


def _edge_tokens(capture) -> set[tuple[str, str, int | None]]:
    out: set[tuple[str, str, int | None]] = set()
    for edge in capture.edges:
        source = capture.variants[edge.source].uid
        target = capture.variants[edge.target].uid
        if edge.offsets:
            for off in edge.offsets:
                out.add((source, target, int(off)))
        else:
            out.add((source, target, None))
    return out


class GuardedExecutor:
    """One compiled variant plus a fail-closed modelx fallback."""

    def __init__(
        self,
        canonical,
        compiled: CompiledCallable,
        *,
        name: str = "compiled_variant",
        fallback: FallbackCallable | None = None,
        optimization_level: str | None = None,
        actual_build_fingerprint: str | None = None,
    ):
        self.canonical = canonical
        self.compiled = compiled
        self.name = name
        self.fallback = fallback or self._modelx_fallback
        self.stats = GuardStats()
        self.optimization_level = optimization_level.upper() if optimization_level else None
        self.expected_build_fingerprint = (
            canonical.executable.build_fingerprint(self.optimization_level)
            if self.optimization_level else None
        )
        self.actual_build_fingerprint = actual_build_fingerprint
        self.expected_sources = {
            schema.fullname: schema.source for schema in canonical.trace.schemas.values()
        }
        self.allowed_variants = {
            tr.uid: tr.dtype for tr in canonical.trace.variants.values()
        }
        self.allowed_edges = _edge_tokens(canonical.trace)

    @classmethod
    def for_python_module(cls, canonical, module, **kwargs):
        kwargs.setdefault("optimization_level", getattr(module, "OPTIMIZATION_LEVEL", None))
        kwargs.setdefault("actual_build_fingerprint", getattr(module, "BUILD_FINGERPRINT", None))
        return cls(canonical, lambda inputs, threads: module.run(inputs), **kwargs)

    @classmethod
    def for_native_module(cls, canonical, module, **kwargs):
        order = list(canonical.inputs)
        kwargs.setdefault("optimization_level", getattr(module, "OPTIMIZATION_LEVEL", None))
        kwargs.setdefault("actual_build_fingerprint", getattr(module, "BUILD_FINGERPRINT", None))
        return cls(
            canonical,
            lambda inputs, threads: module.run(*(inputs[k] for k in order), threads),
            **kwargs,
        )

    def _modelx_fallback(self, keys: tuple[Any, ...]) -> np.ndarray:
        return np.asarray(
            [float(getattr(self.canonical.space[key], self.canonical.output_name)()) for key in keys],
            dtype=np.float64,
        )

    def _formula_reasons(self) -> list[str]:
        current = _current_formula_sources(self.canonical.model)
        reasons = []
        if self.expected_build_fingerprint is not None:
            if self.actual_build_fingerprint is None:
                reasons.append("build_fingerprint_missing")
            elif self.actual_build_fingerprint != self.expected_build_fingerprint:
                reasons.append(
                    f"build_fingerprint_mismatch:{self.actual_build_fingerprint}!={self.expected_build_fingerprint}"
                )
        for fullname, expected in self.expected_sources.items():
            actual = current.get(fullname)
            if actual is None:
                reasons.append(f"formula_missing:{fullname}")
            elif actual != expected:
                reasons.append(f"formula_changed:{fullname}")
        return reasons

    def _input_reasons(self, keys: tuple[Any, ...], raw_inputs: dict[str, Any] | None = None) -> list[str]:
        reasons: list[str] = []
        for name, spec in self.canonical.inputs.items():
            if spec.domain_guard is not None:
                try:
                    domain_reason = spec.domain_guard()
                except Exception as exc:
                    domain_reason = f"table_domain_guard_failed:{name}:{type(exc).__name__}"
                if domain_reason:
                    reasons.append(domain_reason)
            try:
                raw = raw_inputs[name] if raw_inputs is not None else spec.getter(keys)
            except Exception as exc:
                reasons.append(f"input_getter_failed:{name}:{type(exc).__name__}")
                continue
            arr = np.asarray(raw)
            if spec.expected_shape is not None and tuple(arr.shape) != tuple(spec.expected_shape):
                reasons.append(f"input_shape:{name}:{tuple(arr.shape)}!={tuple(spec.expected_shape)}")
                continue
            if spec.ndim == 0:
                if arr.ndim != 0:
                    reasons.append(f"input_ndim:{name}:{arr.ndim}!={spec.ndim}")
                    continue
            elif arr.ndim != spec.ndim:
                reasons.append(f"input_ndim:{name}:{arr.ndim}!={spec.ndim}")
                continue
            if spec.scope == "point" and spec.ndim >= 1 and arr.shape[0] != len(keys):
                reasons.append(f"point_extent:{name}:{arr.shape[0]}!={len(keys)}")
            kind = arr.dtype.kind
            if spec.dtype == "int64" and kind not in "iub":
                reasons.append(f"input_dtype:{name}:{arr.dtype}->int64")
            elif spec.dtype == "bool" and kind not in "biu":
                reasons.append(f"input_dtype:{name}:{arr.dtype}->bool")
            elif spec.dtype == "float64" and kind not in "iufb":
                reasons.append(f"input_dtype:{name}:{arr.dtype}->float64")
        return reasons

    def _trace_reasons(self, keys: tuple[Any, ...]) -> tuple[list[str], int, int]:
        runtime = capture_trace(
            self.canonical.model,
            lambda key: getattr(self.canonical.space[key], self.canonical.output_name).node(),
            keys,
        )
        runtime_variants = {tr.uid: tr.dtype for tr in runtime.variants.values()}
        reasons: list[str] = []
        for uid, dtype in runtime_variants.items():
            if uid not in self.allowed_variants:
                reasons.append(f"unseen_variant:{uid}")
            elif self.allowed_variants[uid] != dtype:
                reasons.append(f"variant_dtype:{uid}:{dtype}!={self.allowed_variants[uid]}")
        runtime_edges = _edge_tokens(runtime)
        for token in sorted(runtime_edges - self.allowed_edges):
            reasons.append(f"unseen_edge:{token[0]}->{token[1]}:{token[2]}")

        # If a template is observed-only, do not silently extrapolate beyond the
        # concrete coordinate support.  Static-formula-guarded templates may grow.
        if any(loop.generalization == "observed_only" for loop in self.canonical.executable.loops):
            expected_times = {
                tr.uid: set(tr.observed_times) for tr in self.canonical.trace.variants.values()
                if tr.observed_times
            }
            for tr in runtime.variants.values():
                if tr.uid in expected_times and not set(tr.observed_times).issubset(expected_times[tr.uid]):
                    reasons.append(f"observed_only_extent:{tr.uid}")
        return reasons, len(runtime_variants), len(runtime_edges)

    def prepare(self, keys: Sequence[Any] | None = None, *, trace_validation: bool = False):
        """Validate one workload and bind its inputs exactly once.

        The returned input mapping is the one consumed by the kernel.  This keeps
        steady-state safety checks from repeating pandas/index extraction.
        """
        keys = tuple(self.canonical.run_keys if keys is None else keys)
        self.stats.checks += 1
        reasons = self._formula_reasons()
        raw_inputs = None
        inputs = None
        if not reasons:
            try:
                raw_inputs = self.canonical.raw_inputs(keys)
            except Exception as exc:
                reasons.append(f"input_getter_failed:batch:{type(exc).__name__}")
        if raw_inputs is not None:
            reasons.extend(self._input_reasons(keys, raw_inputs))
        if raw_inputs is not None and not reasons:
            try:
                inputs = self.canonical.coerce_inputs(raw_inputs)
            except Exception as exc:
                reasons.append(f"input_bind_failed:{type(exc).__name__}")
        rv = re = 0
        if trace_validation and not reasons:
            self.stats.trace_checks += 1
            trace_reasons, rv, re = self._trace_reasons(keys)
            reasons.extend(trace_reasons)
        decision = GuardDecision(
            ok=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            keys_checked=len(keys),
            trace_checked=trace_validation,
            runtime_variants=rv,
            runtime_edges=re,
        )
        return decision, inputs

    def check(self, keys: Sequence[Any] | None = None, *, trace_validation: bool = False) -> GuardDecision:
        return self.prepare(keys, trace_validation=trace_validation)[0]

    def execute_after_check(self, decision: GuardDecision, keys: Sequence[Any] | None = None, *, threads: int = 1):
        keys = tuple(self.canonical.run_keys if keys is None else keys)
        if not decision.ok:
            self.stats.fallback_runs += 1
            for reason in decision.reasons:
                self.stats.fallback_reasons[reason.split(":", 1)[0]] += 1
            return self.fallback(keys), "fallback", decision
        inputs = self.canonical.bind_inputs(keys)
        self.stats.compiled_runs += 1
        return np.asarray(self.compiled(inputs, int(threads)), dtype=np.float64), self.name, decision

    def run(
        self,
        keys: Sequence[Any] | None = None,
        *,
        threads: int = 1,
        trace_validation: bool = False,
    ):
        keys = tuple(self.canonical.run_keys if keys is None else keys)
        decision, inputs = self.prepare(keys, trace_validation=trace_validation)
        if not decision.ok:
            self.stats.fallback_runs += 1
            for reason in decision.reasons:
                self.stats.fallback_reasons[reason.split(":", 1)[0]] += 1
            return self.fallback(keys), "fallback", decision
        assert inputs is not None
        self.stats.compiled_runs += 1
        return np.asarray(self.compiled(inputs, int(threads)), dtype=np.float64), self.name, decision


# ---------------------------------------------------------------------------
# Deterministic artifact registry
# ---------------------------------------------------------------------------
#
# v0.7 inferred sample-uniform branch predicates and used them as artifact
# selectors even though the generated kernel retained the dynamic branch.  That
# rejected valid workloads and added a
# Python loop over every model point.  v0.8 deliberately guards only assumptions
# that code generation actually relies on.  If a future optimization specializes
# a branch, that pass must attach its own explicit proof/guard to the build
# fingerprint rather than reusing trace profile as semantics.

import hashlib as _hashlib
import json as _json


@dataclass(frozen=True)
class ArtifactRecord:
    cache_key: str
    name: str
    executor: GuardedExecutor
    backend: str = "unknown"
    artifact_path: str | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "name": self.name,
            "backend": self.backend,
            "artifact_path": self.artifact_path,
            "expected_build_fingerprint": self.executor.expected_build_fingerprint,
            "actual_build_fingerprint": self.executor.actual_build_fingerprint,
        }


@dataclass
class ArtifactManagerStats:
    requests: int = 0
    compiled_hits: int = 0
    fallback_misses: int = 0
    learned_variants: int = 0
    stale_rejections: int = 0
    miss_reasons: Counter[str] = field(default_factory=Counter)
    artifact_hits: Counter[str] = field(default_factory=Counter)

    def manifest(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "compiled_hits": self.compiled_hits,
            "fallback_misses": self.fallback_misses,
            "learned_variants": self.learned_variants,
            "stale_rejections": self.stale_rejections,
            "native_coverage": self.compiled_hits / self.requests if self.requests else None,
            "miss_reasons": dict(self.miss_reasons),
            "artifact_hits": dict(self.artifact_hits),
        }


def artifact_cache_key(executor: GuardedExecutor, *, backend: str = "unknown") -> str:
    """Identity for a compiled artifact's *actual* executable assumptions.

    Observed branch outcomes are intentionally absent.  Dynamic branches remain
    in the generated code unless a transformation explicitly specializes them.
    """
    payload = {
        "backend": backend,
        "build": executor.expected_build_fingerprint,
        "sources": sorted(executor.expected_sources.items()),
        "inputs": sorted(
            (k, v.dtype, v.ndim, v.scope, v.description, v.domain_token, v.expected_shape)
            for k, v in executor.canonical.inputs.items()
        ),
    }
    raw = _json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "artifact_" + _hashlib.sha256(raw.encode()).hexdigest()[:20]


class VariantArtifactManager:
    """Finite guarded artifact registry with immediate fallback on a miss.

    ``run`` performs no tracing or compilation.  ``learn`` is explicit and
    synchronous.  Selection is based only on assumptions that the generated
    artifact actually depends on: build/formula fingerprints, input contracts and
    structural domains.
    """

    def __init__(self, *, fallback: FallbackCallable):
        self.fallback = fallback
        self._records: dict[str, ArtifactRecord] = {}
        self._order: list[str] = []
        self.stats = ArtifactManagerStats()

    @property
    def records(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._records[k] for k in self._order)

    def register(
        self,
        executor: GuardedExecutor,
        *,
        backend: str = "unknown",
        artifact_path: str | None = None,
        name: str | None = None,
    ) -> ArtifactRecord:
        key = artifact_cache_key(executor, backend=backend)
        record = ArtifactRecord(key, name or executor.name, executor, backend, artifact_path)
        if key not in self._records:
            self._order.append(key)
        self._records[key] = record
        return record

    def _record_reasons(self, record: ArtifactRecord, keys: tuple[Any, ...]):
        decision, inputs = record.executor.prepare(keys)
        return decision.reasons, inputs

    def run(self, keys: Sequence[Any], *, threads: int = 1):
        keys = tuple(keys)
        self.stats.requests += 1
        decisions = []
        for record in self.records:
            reasons, inputs = self._record_reasons(record, keys)
            decisions.append((record.cache_key, reasons))
            if reasons:
                heads = {r.split(":", 1)[0] for r in reasons}
                for head in heads:
                    self.stats.miss_reasons[head] += 1
                if heads.intersection({"build_fingerprint_missing", "build_fingerprint_mismatch", "formula_missing", "formula_changed"}):
                    self.stats.stale_rejections += 1
                continue
            assert inputs is not None
            out = np.asarray(record.executor.compiled(inputs, int(threads)), dtype=np.float64)
            self.stats.compiled_hits += 1
            self.stats.artifact_hits[record.cache_key] += 1
            record.executor.stats.compiled_runs += 1
            return out, record.name, tuple(decisions)

        self.stats.fallback_misses += 1
        self.stats.miss_reasons["no_compatible_artifact"] += 1
        return np.asarray(self.fallback(keys), dtype=np.float64), "fallback", tuple(decisions)

    def learn(self, keys: Sequence[Any], builder: Callable[[tuple[Any, ...]], GuardedExecutor], *, backend: str = "unknown", name: str | None = None) -> ArtifactRecord:
        """Explicitly build/register a new structural variant outside ``run``."""
        keys = tuple(keys)
        executor = builder(keys)
        record = self.register(executor, backend=backend, name=name)
        self.stats.learned_variants += 1
        return record

    def manifest(self) -> dict[str, Any]:
        return {
            "artifacts": [r.manifest() for r in self.records],
            "stats": self.stats.manifest(),
            "miss_policy": "immediate fallback; learning/build only via explicit learn() outside hot path",
            "selector_policy": "guard only assumptions used by generated code; no sample-only branch profile",
        }
