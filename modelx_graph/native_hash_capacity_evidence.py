from __future__ import annotations

"""Complete-RunDomain evidence for fixed native exact-hash memo capacity.

The native hash runtime deliberately does not resize or fall back to Python. A
compiled artifact therefore needs a fixed capacity for each hash-memo Cell. This
module obtains that capacity from the already-exported modelx-cython model:

* only Cells that the native memo planner classifies as exact-hash candidates are
  retained as key sets;
* the selected output is evaluated once for every key in the frozen RunDomain;
* unique semantic argument tuples are counted independently for each policy;
* integral parameter extrema are aggregated online from the same complete trace;
* the maximum per-policy cardinality is converted to a power-of-two capacity by
  the generic load-factor rule in :mod:`dense_recursive_native`.

This is build-time evidence. It is not a runtime Python memo and it does not
participate in native valuation after code generation.

Unlike the earlier implementation, the production logger is streaming: it does
not retain every ``CallTrace`` and does not invoke modelx-cython's general
``RuntimeCellsInfo`` aggregation/model walk at ``flush()``. Proof authority is
unchanged; only evidence storage and aggregation are narrower.
"""

import importlib
import math
import numbers
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from modelx_cython.tracer import MxCallTraceLogger, MxCodeFilter, trace_calls

from .dense_recursive_native import size_native_hash_capacities_from_peaks
from .native_formula import NativeFormulaCell


class NativeHashCapacityEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeHashCapacityEvidence:
    schema: str
    run_population: int
    completed_policies: int
    peak_entries_by_uid: Mapping[str, int]
    capacities_by_uid: Mapping[str, int]
    nan_uids: tuple[str, ...]
    target_load_factor: float
    min_capacity: int
    max_capacity: int
    complete_integral_ranges_by_uid: Mapping[str, tuple[tuple[int, int] | None, ...]] = field(default_factory=dict)
    authority: str = "complete_exported_model_trace"
    trace_call_count: int = 0
    policy_metrics: tuple[Mapping[str, Any], ...] = ()
    collector: str = "streaming_targeted_formula_trace"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_population": self.run_population,
            "completed_policies": self.completed_policies,
            "peak_entries_by_uid": dict(self.peak_entries_by_uid),
            "capacities_by_uid": dict(self.capacities_by_uid),
            "complete_integral_ranges_by_uid": {
                uid: [None if item is None else list(item) for item in ranges]
                for uid, ranges in self.complete_integral_ranges_by_uid.items()
            },
            "nan_uids": list(self.nan_uids),
            "target_load_factor": self.target_load_factor,
            "min_capacity": self.min_capacity,
            "max_capacity": self.max_capacity,
            "authority": self.authority,
            "runtime_dependency": False,
            "python_memo_storage": False,
            "collector": self.collector,
            "trace_call_count": int(self.trace_call_count),
            "policy_metrics": [dict(row) for row in self.policy_metrics],
        }


def _canonical_trace_value(dtype: str, value: Any) -> tuple[Any, bool]:
    """Return a hashable cardinality key using native-cache equality semantics."""
    if dtype == "float64":
        x = float(value)
        if math.isnan(x):
            # Native execution deliberately fails closed on NaN. Keep a stable
            # marker for cardinality accounting and report the Cell separately.
            return ("__native_hash_nan__",), True
        if x == 0.0:
            return 0.0, False
        return x, False
    if dtype == "bool":
        return bool(value), False
    if dtype == "int64":
        # Exported category arguments are still semantic labels (often strings)
        # while the native ABI uses int64 category codes. Cardinality is
        # preserved by exact semantic label equality, so retain hashable labels.
        if isinstance(value, numbers.Integral):
            return int(value), False
        try:
            hash(value)
            return (type(value).__name__, value), False
        except Exception:
            return (type(value).__name__, repr(value)), False
    try:
        hash(value)
        return value, False
    except Exception:
        return repr(value), False


def _formula_code_name(cell: NativeFormulaCell) -> str:
    """Return the exported modelx-cython method name for a Cell formula."""
    tail = str(cell.source_fullname).rsplit(".", 1)[-1]
    if tail:
        return tail
    name = str(cell.source_name)
    return name if name.startswith("__") else "_f_" + name


class _TargetedFormulaCodeFilter:
    """Trace only formula methods needed by capacity/range evidence.

    ``MxCodeFilter`` admits every exported formula, ``__call__`` and ref-assignment
    method. Capacity evidence needs only the selected output plus the hash/range
    Cells. The logger still validates full qualified names, so a same-named Cell in
    another Space can at worst generate an ignored callback; it cannot contaminate
    evidence.
    """

    def __init__(self, cells: Sequence[NativeFormulaCell]) -> None:
        self._base = MxCodeFilter()
        self._allowed_names = frozenset(_formula_code_name(cell) for cell in cells)

    @property
    def allowed_names(self) -> frozenset[str]:
        return self._allowed_names

    def __call__(self, code) -> bool:
        return code.co_name in self._allowed_names and self._base(code)


def _current_rss_bytes() -> int | None:
    """Best-effort current process RSS without adding a runtime dependency."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return None


def _peak_rss_bytes() -> int | None:
    """Best-effort process peak RSS normalized to bytes."""
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux and the BSDs used by our CI report KiB.
        if sys.platform == "darwin":
            return value
        return value * 1024
    except Exception:
        return None


@dataclass
class _IntegralExtrema:
    seen: bool = False
    non_integral_seen: bool = False
    lower: int | None = None
    upper: int | None = None

    def observe(self, value: Any) -> None:
        self.seen = True
        if not isinstance(value, numbers.Integral):
            # RuntimeCellsInfo exposes min/max only when every observed value for
            # the argument belongs to the integral family. Preserve that rule.
            self.non_integral_seen = True
            return
        ival = int(value)
        self.lower = ival if self.lower is None else min(self.lower, ival)
        self.upper = ival if self.upper is None else max(self.upper, ival)

    def interval(self) -> tuple[int, int] | None:
        if self.non_integral_seen or not self.seen or self.lower is None or self.upper is None:
            return None
        return self.lower, self.upper


class _StreamingCompleteRunEvidenceLogger:
    """Aggregate complete-run hash/range evidence without retaining CallTrace rows."""

    def __init__(
        self,
        *,
        output_fqname: str,
        hash_cells_by_fqname: Mapping[str, NativeFormulaCell],
        range_cells_by_fqname: Mapping[str, NativeFormulaCell],
    ) -> None:
        self.output_fqname = str(output_fqname)
        self.hash_cells_by_fqname = dict(hash_cells_by_fqname)
        self.range_cells_by_fqname = dict(range_cells_by_fqname)
        self.current: dict[str, set[tuple[Any, ...]]] = {
            cell.uid: set() for cell in self.hash_cells_by_fqname.values()
        }
        self.peaks: dict[str, int] = {uid: 0 for uid in self.current}
        self.nan_uids: set[str] = set()
        self.completed_policies = 0
        self.trace_call_count = 0
        self._policy_trace_calls = 0
        self._active_policy_index: int | None = None
        self._completed_at_policy_start = 0
        self._rss_before_bytes: int | None = None
        self.policy_metrics: list[dict[str, Any]] = []
        self._integral_extrema_by_uid: dict[str, list[_IntegralExtrema | None]] = {}
        for cell in self.range_cells_by_fqname.values():
            self._integral_extrema_by_uid[cell.uid] = [
                _IntegralExtrema() if dtype in {"int64", "bool"} else None
                for dtype in cell.parameter_dtypes
            ]

    def begin_policy(self, index: int) -> None:
        if self._active_policy_index is not None:
            raise NativeHashCapacityEvidenceError("streaming capacity logger already has an active policy")
        self._active_policy_index = int(index)
        self._completed_at_policy_start = self.completed_policies
        self._policy_trace_calls = 0
        self._rss_before_bytes = _current_rss_bytes()

    def finish_policy(self, *, key: Any, elapsed_s: float) -> None:
        index = self._active_policy_index
        if index is None:
            raise NativeHashCapacityEvidenceError("streaming capacity logger has no active policy")
        if self.completed_policies != self._completed_at_policy_start + 1:
            raise NativeHashCapacityEvidenceError(
                f"policy {index}: selected output completed "
                f"{self.completed_policies - self._completed_at_policy_start} times; expected exactly once"
            )
        rss_after = _current_rss_bytes()
        self.policy_metrics.append(
            {
                "policy_index": index,
                "run_key_repr": repr(key),
                "elapsed_s": float(elapsed_s),
                "trace_call_count": int(self._policy_trace_calls),
                "rss_before_bytes": self._rss_before_bytes,
                "rss_after_bytes": rss_after,
                "rss_delta_bytes": (
                    None
                    if rss_after is None or self._rss_before_bytes is None
                    else int(rss_after - self._rss_before_bytes)
                ),
                "process_peak_rss_bytes": _peak_rss_bytes(),
            }
        )
        self._active_policy_index = None
        self._rss_before_bytes = None

    @staticmethod
    def _raw_args(trace) -> tuple[Any, ...]:
        return tuple(value for _name, value in list(trace.arg_vals.items())[1:])

    def _observe_hash_cell(self, cell: NativeFormulaCell, trace) -> None:
        raw_args = self._raw_args(trace)
        if len(raw_args) != len(cell.parameter_dtypes):
            raise NativeHashCapacityEvidenceError(
                f"{cell.uid}: trace arity {len(raw_args)} != native arity {len(cell.parameter_dtypes)}"
            )
        key: list[Any] = []
        for dtype, value in zip(cell.parameter_dtypes, raw_args):
            item, is_nan = _canonical_trace_value(dtype, value)
            key.append(item)
            if is_nan:
                self.nan_uids.add(cell.uid)
        self.current[cell.uid].add(tuple(key))

    def _observe_range_cell(self, cell: NativeFormulaCell, trace) -> None:
        raw_items = list(trace.arg_vals.items())[1:]
        if len(raw_items) != len(cell.parameter_dtypes):
            raise NativeHashCapacityEvidenceError(
                f"{cell.uid}: trace arity {len(raw_items)} != native arity {len(cell.parameter_dtypes)}"
            )
        states = self._integral_extrema_by_uid[cell.uid]
        for state, (_name, value) in zip(states, raw_items):
            if state is not None:
                state.observe(value)

    def log(self, trace) -> None:  # modelx-cython tracer callback
        self.trace_call_count += 1
        if self._active_policy_index is not None:
            self._policy_trace_calls += 1

        hash_cell = self.hash_cells_by_fqname.get(trace.funcname)
        if hash_cell is not None:
            self._observe_hash_cell(hash_cell, trace)

        range_cell = self.range_cells_by_fqname.get(trace.funcname)
        if range_cell is not None:
            self._observe_range_cell(range_cell, trace)

        # The selected output's return trace occurs after all nested Cell calls for
        # one policy. Snapshot and clear here so capacity reflects the maximum
        # *one-generation* occupancy, matching the compiled memo epoch semantics.
        if trace.funcname == self.output_fqname:
            for uid, values in self.current.items():
                self.peaks[uid] = max(self.peaks[uid], len(values))
                values.clear()
            self.completed_policies += 1

    def flush(self) -> None:
        """Required tracer hook; all evidence is already aggregated online."""
        return None

    def complete_integral_ranges(
        self,
        range_cells: Mapping[str, NativeFormulaCell],
    ) -> dict[str, tuple[tuple[int, int] | None, ...]]:
        out: dict[str, tuple[tuple[int, int] | None, ...]] = {}
        for uid, cell in range_cells.items():
            states = self._integral_extrema_by_uid.get(str(uid))
            if states is None:
                out[str(uid)] = tuple(None for _ in cell.parameter_dtypes)
                continue
            ranges: list[tuple[int, int] | None] = []
            for dtype, state in zip(cell.parameter_dtypes, states):
                interval = None if dtype not in {"int64", "bool"} or state is None else state.interval()
                if interval is not None:
                    lo, hi = interval
                    if lo < -(1 << 63) or hi > (1 << 63) - 1:
                        raise NativeHashCapacityEvidenceError(
                            f"{uid}: complete-run integer range {(lo, hi)!r} exceeds signed int64"
                        )
                ranges.append(interval)
            out[str(uid)] = tuple(ranges)
        return out


def _space_path(model_name: str, space_fullname: str, fallback_name: str) -> tuple[str, ...]:
    parts = tuple(part for part in str(space_fullname).split(".") if part)
    if parts and parts[0] == model_name:
        parts = parts[1:]
    if not parts:
        parts = (fallback_name,)
    if any("[" in part or "]" in part for part in parts):
        raise NativeHashCapacityEvidenceError(
            f"capacity tracing requires a canonical top-level/static Space path, got {space_fullname!r}"
        )
    return parts


def _complete_integral_ranges_from_flushed_trace(
    logger: MxCallTraceLogger,
    range_cells: Mapping[str, NativeFormulaCell],
) -> dict[str, tuple[tuple[int, int] | None, ...]]:
    """Legacy-equivalence helper retained for regression/reference tests.

    The production path no longer calls ``MxCallTraceLogger.flush``. This helper
    documents the exact prior extrema semantics and lets regression tests compare
    the streaming collector against modelx-cython's retained-trace aggregation.
    """
    out: dict[str, tuple[tuple[int, int] | None, ...]] = {}
    for uid, cell in range_cells.items():
        info = logger.cells_info.get(cell.source_fullname)
        ranges: list[tuple[int, int] | None] = []
        for name, dtype in zip(cell.parameters, cell.parameter_dtypes):
            if (
                info is not None
                and dtype in {"int64", "bool"}
                and name in getattr(info, "min_args", {})
                and name in getattr(info, "max_args", {})
            ):
                lo = int(info.min_args[name])
                hi = int(info.max_args[name])
                if lo < -(1 << 63) or hi > (1 << 63) - 1:
                    raise NativeHashCapacityEvidenceError(
                        f"{uid}: complete-run integer range {(lo, hi)!r} exceeds signed int64"
                    )
                ranges.append((lo, hi))
            else:
                ranges.append(None)
        out[str(uid)] = tuple(ranges)
    return out


def _cells_by_fqname(
    cells: Mapping[str, NativeFormulaCell],
    *,
    purpose: str,
) -> dict[str, NativeFormulaCell]:
    out: dict[str, NativeFormulaCell] = {}
    for uid, cell in cells.items():
        if cell.source_fullname in out:
            other = out[cell.source_fullname]
            raise NativeHashCapacityEvidenceError(
                f"duplicate {purpose} traced Cell identity {cell.source_fullname!r}: {other.uid}, {uid}"
            )
        out[cell.source_fullname] = cell
    return out


def profile_native_hash_capacity_evidence(
    *,
    model_path: str | Path,
    space: Any,
    output_name: str,
    run_keys: Sequence[Any],
    output_cell: NativeFormulaCell,
    hash_cells: Mapping[str, NativeFormulaCell],
    range_cells: Mapping[str, NativeFormulaCell] | None = None,
    target_load_factor: float = 0.50,
    min_capacity: int = 16,
    max_capacity: int = 1 << 20,
) -> NativeHashCapacityEvidence:
    """Profile complete-domain unique semantic keys and size fixed native tables."""
    model_path = Path(model_path)
    run_keys = tuple(run_keys)
    if not hash_cells:
        return NativeHashCapacityEvidence(
            schema="modelx_graph.native_hash_capacity_evidence.v3_streaming_complete_ranges",
            run_population=len(run_keys),
            completed_policies=len(run_keys),
            peak_entries_by_uid={},
            capacities_by_uid={},
            nan_uids=(),
            target_load_factor=target_load_factor,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            complete_integral_ranges_by_uid={},
        )

    hash_cells_by_fqname = _cells_by_fqname(hash_cells, purpose="hash")
    effective_range_cells = hash_cells if range_cells is None else range_cells
    range_cells_by_fqname = _cells_by_fqname(effective_range_cells, purpose="range")

    module = model_path.name
    source_model_name = getattr(getattr(space, "model", None), "name", module)
    path = _space_path(source_model_name, getattr(space, "fullname", ""), getattr(space, "name", "Projection"))
    parameterized = bool(tuple(getattr(space, "parameters", ()) or ()))

    for loaded in tuple(sys.modules):
        if loaded == module or loaded.startswith(module + "."):
            sys.modules.pop(loaded, None)

    parent = str(model_path.parent)
    sys.path.insert(0, parent)
    logger = _StreamingCompleteRunEvidenceLogger(
        output_fqname=output_cell.source_fullname,
        hash_cells_by_fqname=hash_cells_by_fqname,
        range_cells_by_fqname=range_cells_by_fqname,
    )
    traced_cells: dict[str, NativeFormulaCell] = {
        output_cell.source_fullname: output_cell,
        **hash_cells_by_fqname,
        **range_cells_by_fqname,
    }
    code_filter = _TargetedFormulaCodeFilter(tuple(traced_cells.values()))
    try:
        with trace_calls(
            module=module,
            logger=logger,
            max_typed_dict_size=0,
            code_filter=code_filter,
        ):
            package = importlib.import_module(module)
            # Current modelx exports expose mx_model independently of the
            # destination package name. Preserve older named exports too.
            root = getattr(package, "mx_model", None)
            if root is None:
                root = getattr(package, module)
            target_space = root
            for component in path:
                target_space = getattr(target_space, component)
            for index, key in enumerate(run_keys):
                logger.begin_policy(index)
                started = perf_counter()
                instance = target_space[key] if parameterized else target_space
                getattr(instance, output_name)()
                elapsed_s = perf_counter() - started
                logger.finish_policy(key=key, elapsed_s=elapsed_s)
    finally:
        assert sys.path.pop(0) == parent

    if logger.completed_policies != len(run_keys):
        raise NativeHashCapacityEvidenceError(
            f"complete-domain hash trace finished {logger.completed_policies} policies; expected {len(run_keys)}"
        )

    peaks = {uid: int(logger.peaks.get(uid, 0)) for uid in sorted(hash_cells)}
    capacities = size_native_hash_capacities_from_peaks(
        peaks,
        target_load_factor=target_load_factor,
        min_capacity=min_capacity,
        max_capacity=max_capacity,
    )
    return NativeHashCapacityEvidence(
        schema="modelx_graph.native_hash_capacity_evidence.v3_streaming_complete_ranges",
        run_population=len(run_keys),
        completed_policies=logger.completed_policies,
        peak_entries_by_uid=peaks,
        capacities_by_uid=capacities,
        nan_uids=tuple(sorted(logger.nan_uids)),
        complete_integral_ranges_by_uid=logger.complete_integral_ranges(effective_range_cells),
        target_load_factor=target_load_factor,
        min_capacity=min_capacity,
        max_capacity=max_capacity,
        trace_call_count=logger.trace_call_count,
        policy_metrics=tuple(logger.policy_metrics),
    )


__all__ = [
    "NativeHashCapacityEvidenceError",
    "NativeHashCapacityEvidence",
    "profile_native_hash_capacity_evidence",
]
