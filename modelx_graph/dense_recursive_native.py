from __future__ import annotations

"""Dense generation-tagged recursive memoization for the fast backend.

This module deliberately keeps recursive execution. It changes only memo cost and
physical representation. Scalar Cells receive typed slots and bounded integral or
category coordinates receive dense generation-tagged slots. Stage-A ``native_cache``
adds signature-specialized native exact hash tables for non-dense keys, including
float64 keys represented by canonicalized IEEE-754 bits. Historical cache/direct policies are quarantined behind the explicit
``build_legacy_comparison_recursive_program`` entry point. Production is native
caching by default with no Python memo objects in admitted execution. Scalar/dense
values are stored in result-dtype-specific compact arenas. A local native call relation is used only for cycle
safety, never as an execution schedule. No Stage or scan conversion participates.
"""

import ast
import copy
import hashlib
import math
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .native_formula import NativeFormulaCell, NativeFormulaProgram
from .recursive_native import RecursiveCythonGenerator, RecursiveNativeError


PRODUCTION_MEMOIZATION_POLICY = "native_cache"
PRODUCTION_OVERFLOW_POLICY = "fail_closed"
DEPRECATED_COMPARISON_MEMO_POLICIES = frozenset({"pure_map_direct", "semantic_parameter"})
LEGACY_COMPARISON_MEMO_POLICIES = frozenset({"cache_all", "pure_map_direct", "semantic_parameter"})


class DenseRecursiveNativeError(RecursiveNativeError):
    pass


@dataclass(frozen=True)
class DenseMemoCell:
    uid: str
    arity: int
    kind: str  # scalar | dense | hash | dict | direct
    bounds: tuple[tuple[int, int], ...]
    strides: tuple[int, ...]
    size: int
    decision_reason: str = "cache_all"
    arena_offset: int = -1
    cycle_guard: bool = False
    hash_capacity: int = 0
    value_offset: int = -1

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "arity": self.arity,
            "kind": self.kind,
            "bounds": [list(x) for x in self.bounds],
            "strides": list(self.strides),
            "size": self.size,
            "decision_reason": self.decision_reason,
            "arena_offset": self.arena_offset,
            "cycle_guard": self.cycle_guard,
            "hash_capacity": self.hash_capacity,
            "value_offset": self.value_offset,
            "overflow_fallback": self.kind in {"dense", "dict"},
        }


@dataclass(frozen=True)
class DenseMemoPlan:
    schema: str
    cells: tuple[DenseMemoCell, ...]

    def cell(self, uid: str) -> DenseMemoCell:
        for row in self.cells:
            if row.uid == uid:
                return row
        raise DenseRecursiveNativeError(f"dense memo cell {uid!r} not found")

    @property
    def dense_slots(self) -> int:
        return sum(x.size for x in self.cells if x.kind in {"scalar", "dense"})

    @property
    def arena_slots(self) -> int:
        return self.dense_slots

    @property
    def hash_slots(self) -> int:
        return sum(x.hash_capacity for x in self.cells if x.kind == "hash")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dense_slots": self.dense_slots,
            "hash_slots": self.hash_slots,
            "cells": [x.manifest() for x in self.cells],
        }


@dataclass(frozen=True)
class DenseRecursiveProgram:
    schema: str
    formula_program: NativeFormulaProgram
    memo_plan: DenseMemoPlan
    supported: bool
    blockers: tuple[str, ...] = ()
    memoization_policy: str = PRODUCTION_MEMOIZATION_POLICY
    overflow_policy: str = PRODUCTION_OVERFLOW_POLICY
    native_hash_debug: bool = False

    @property
    def output_uid(self) -> str:
        return self.formula_program.output_uid

    def manifest(self) -> dict[str, Any]:
        output = self.formula_program.cells.get(self.output_uid)
        hash_storage: dict[str, dict[str, int]] = {}
        hash_bytes_total = 0
        for row in self.memo_plan.cells:
            if row.kind != "hash":
                continue
            cell = self.formula_program.cells[row.uid]
            key_bytes_per_slot = sum(_native_dtype_bytes(x) for x in cell.parameter_dtypes)
            value_bytes_per_slot = _native_dtype_bytes(cell.dtype)
            metadata_bytes_per_slot = 5  # uint32 generation + uint8 busy
            bytes_total = row.hash_capacity * (
                key_bytes_per_slot + value_bytes_per_slot + metadata_bytes_per_slot
            )
            hash_bytes_total += bytes_total
            hash_storage[row.uid] = {
                "capacity": row.hash_capacity,
                "key_bytes_per_slot": key_bytes_per_slot,
                "value_bytes_per_slot": value_bytes_per_slot,
                "metadata_bytes_per_slot": metadata_bytes_per_slot,
                "bytes_total": bytes_total,
            }
        dense_slots_by_dtype = {"float64": 0, "int64": 0, "bool": 0}
        dense_slots_by_kind = {"scalar": 0, "dense": 0}
        dense_bytes_by_kind_dtype: dict[str, dict[str, int]] = {
            "scalar": {"float64": 0, "int64": 0, "bool": 0},
            "dense": {"float64": 0, "int64": 0, "bool": 0},
        }
        dense_slots_by_kind_dtype: dict[str, dict[str, int]] = {
            "scalar": {"float64": 0, "int64": 0, "bool": 0},
            "dense": {"float64": 0, "int64": 0, "bool": 0},
        }
        for row in self.memo_plan.cells:
            if row.kind not in {"scalar", "dense"}:
                continue
            dtype = self.formula_program.cells[row.uid].dtype
            dense_slots_by_dtype[dtype] += row.size
            dense_slots_by_kind[row.kind] += row.size
            dense_slots_by_kind_dtype[row.kind][dtype] += row.size
            dense_bytes_by_kind_dtype[row.kind][dtype] += row.size * _native_dtype_bytes(dtype)
        dense_value_bytes = sum(
            slots * _native_dtype_bytes(dtype)
            for dtype, slots in dense_slots_by_dtype.items()
        )
        # Dense/scalar metadata packs 31-bit generation + busy flag into one uint32.
        dense_metadata_bytes = self.memo_plan.arena_slots * 4
        dense_arena_bytes = dense_value_bytes + dense_metadata_bytes
        legacy_dense_arena_bytes = self.memo_plan.arena_slots * (8 + 8 + 1 + 4 + 1)
        dense_storage_by_kind_dtype: dict[str, dict[str, dict[str, int]]] = {}
        for kind in ("scalar", "dense"):
            dense_storage_by_kind_dtype[kind] = {}
            for dtype in ("float64", "int64", "bool"):
                slots = dense_slots_by_kind_dtype[kind][dtype]
                value_bytes = dense_bytes_by_kind_dtype[kind][dtype]
                metadata_bytes = slots * 4
                dense_storage_by_kind_dtype[kind][dtype] = {
                    "slots": slots,
                    "value_bytes": value_bytes,
                    "metadata_bytes": metadata_bytes,
                    "total_bytes": value_bytes + metadata_bytes,
                    "legacy_layout_bytes": slots * 22,
                }
        dense_bytes_by_kind = {
            kind: sum(row["total_bytes"] for row in dense_storage_by_kind_dtype[kind].values())
            for kind in ("scalar", "dense")
        }
        return {
            "schema": self.schema,
            "formula_program_uid": self.formula_program.uid,
            "output_uid": self.output_uid,
            "supported": self.supported,
            "blockers": list(self.blockers),
            "memoization_policy": self.memoization_policy,
            "overflow_policy": self.overflow_policy,
            "memo_policy_role": (
                "production_default"
                if self.memoization_policy == "native_cache"
                else "deprecated_comparison_only"
                if self.memoization_policy in {"semantic_parameter", "pure_map_direct"}
                else "legacy_compatibility"
            ),
            "native_hash_debug": self.native_hash_debug,
            "native_hash_key_semantics": {
                "int64": "exact_signed_64_bit_value",
                "category": "exact_category_abi_int64_code",
                "bool": "exact_0_or_1",
                "float64": "ieee754_bits_with_negative_zero_canonicalized_to_positive_zero",
                "nan": "fail_closed_native_memo_error_5",
            } if any(x.kind == "hash" for x in self.memo_plan.cells) else None,
            "python_memo_objects_required": any(
                x.kind == "dict" or (x.kind == "dense" and self.overflow_policy == "python_exact")
                for x in self.memo_plan.cells
            ),
            "memo_storage_estimate": {
                "dense_arena_bytes": dense_arena_bytes,
                "dense_arena_bytes_legacy_layout": legacy_dense_arena_bytes,
                "dense_value_bytes": dense_value_bytes,
                "dense_metadata_bytes": dense_metadata_bytes,
                "dense_metadata_layout": "uint32_packed_generation31_plus_busy_high_bit",
                "dense_slots_by_dtype": dict(dense_slots_by_dtype),
                "dense_slots_by_kind": dict(dense_slots_by_kind),
                "dense_value_bytes_by_kind_dtype": dense_bytes_by_kind_dtype,
                "dense_storage_by_kind_dtype": dense_storage_by_kind_dtype,
                "dense_bytes_by_kind": dense_bytes_by_kind,
                "dense_compaction_ratio_vs_legacy": (
                    float(dense_arena_bytes) / float(legacy_dense_arena_bytes)
                    if legacy_dense_arena_bytes else 1.0
                ),
                "hash_bytes": hash_bytes_total,
                "total_bytes": dense_arena_bytes + hash_bytes_total,
                "total_bytes_legacy_dense_layout": legacy_dense_arena_bytes + hash_bytes_total,
                "hash_cells": hash_storage,
                "estimate_scope": "generated_fixed_arrays_excluding_module/runtime_overhead",
            },
            "memo_plan": self.memo_plan.manifest(),
            "stage_dependency": False,
            "graph_dependency": False,
            "selected_output_abi": None if output is None else {
                "dtype": output.dtype,
                "ndim": output.logical_result_ndim,
                "shape": (
                    None if output.logical_result_shape is None
                    else list(output.logical_result_shape)
                ),
                "array_semantics": (
                    "preserved_ranked_formula_memoryview"
                    if output.ranked_function is not None else "scalar_recursive_driver"
                ),
            },
        }


def _dense_strides(bounds: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], int]:
    if not bounds:
        return (), 1
    widths = [hi - lo + 1 for lo, hi in bounds]
    if any(width <= 0 for width in widths):
        raise DenseRecursiveNativeError(f"invalid dense memo bounds {bounds!r}")
    strides = [1] * len(widths)
    for i in range(len(widths) - 2, -1, -1):
        strides[i] = strides[i + 1] * widths[i + 1]
    size = 1
    for width in widths:
        size *= width
    return tuple(strides), size



def _cell_call_uids(cell: NativeFormulaCell, known: set[str]) -> set[str]:
    """Return direct native Cell calls from one already-normalized formula.

    This is deliberately a local call scan used only for memo-cost classification.
    It does not construct an execution schedule or alter formula closure.
    """
    nodes: list[ast.AST] = []
    if cell.function is not None:
        nodes.append(cell.function)
    if cell.reduction is not None:
        r = cell.reduction
        nodes.extend([r.init, *r.range_args, r.body_expr, *r.filters])
    out: set[str] = set()
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in known:
                out.add(node.func.id)
    return out


def _memo_semantics(
    formula_program: NativeFormulaProgram, *, include_legacy_pure_maps: bool = False,
) -> tuple[dict[str, set[str]], set[str], set[str]]:
    """Return native call relation and recursive SCC membership.

    Production native-cache planning needs only recursion membership for busy/cycle
    semantics. The historical pure-map/no-cache reachability proof is computed only
    when the explicit legacy comparison builder requests it.
    """
    active = {
        uid: cell for uid, cell in formula_program.cells.items()
        if cell.execution_role == "formula"
    }
    known = set(active)
    graph = {uid: _cell_call_uids(cell, known) for uid, cell in active.items()}

    # Tarjan SCC over the native call relation.  This is a memo-cost proof only,
    # never an execution/scheduling authority.
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    recursive: set[str] = set()

    def visit(uid: str) -> None:
        nonlocal index
        indices[uid] = low[uid] = index
        index += 1
        stack.append(uid)
        on_stack.add(uid)
        for dep in graph[uid]:
            if dep not in indices:
                visit(dep)
                low[uid] = min(low[uid], low[dep])
            elif dep in on_stack:
                low[uid] = min(low[uid], indices[dep])
        if low[uid] != indices[uid]:
            return
        comp: list[str] = []
        while True:
            item = stack.pop()
            on_stack.remove(item)
            comp.append(item)
            if item == uid:
                break
        if len(comp) > 1 or (len(comp) == 1 and uid in graph[uid]):
            recursive.update(comp)

    for uid in sorted(active):
        if uid not in indices:
            visit(uid)

    if not include_legacy_pure_maps:
        return graph, recursive, set()

    reaches_recursive: dict[str, bool] = {}
    visiting: set[str] = set()

    def reaches(uid: str) -> bool:
        if uid in reaches_recursive:
            return reaches_recursive[uid]
        if uid in recursive:
            reaches_recursive[uid] = True
            return True
        if uid in visiting:
            # A cycle would already have been marked recursive above.  This guard
            # only keeps the proof fail-closed if the relation is malformed.
            reaches_recursive[uid] = True
            return True
        visiting.add(uid)
        result = any(reaches(dep) for dep in graph[uid])
        visiting.remove(uid)
        reaches_recursive[uid] = result
        return result

    pure_maps = {
        uid for uid, cell in active.items()
        if cell.arity > 0 and not reaches(uid)
    }
    return graph, recursive, pure_maps

def _native_hash_capacity(value: int) -> int:
    """Return a power-of-two open-addressed capacity, minimum four slots."""
    n = max(4, int(value))
    out = 1
    while out < n:
        out <<= 1
    return out


def size_native_hash_capacities_from_peaks(
    peak_entries_by_uid: Mapping[str, int],
    *,
    target_load_factor: float = 0.50,
    min_capacity: int = 16,
    max_capacity: int = 1 << 20,
) -> dict[str, int]:
    """Size per-Cell native hash tables from observed/authoritative peak entries.

    The rule is deliberately generic and Cell-name agnostic.  It is intended for
    Stage-B profiling where the complete frozen RunDomain has been executed with
    debug counters enabled.  The result is the smallest power-of-two capacity
    whose peak load is no greater than ``target_load_factor``.

    Capacity overflow fails closed rather than silently clipping to
    ``max_capacity``.  That keeps the compiled artifact honest about the domain it
    was sized for.
    """
    if not (0.0 < float(target_load_factor) < 1.0):
        raise DenseRecursiveNativeError(
            f"target_load_factor must be between 0 and 1, got {target_load_factor!r}"
        )
    lo = _native_hash_capacity(min_capacity)
    max_raw = int(max_capacity)
    if max_raw < 4:
        raise DenseRecursiveNativeError(
            f"max_capacity must permit at least four slots, got {max_capacity!r}"
        )
    hi = 1 << (max_raw.bit_length() - 1)
    if lo > hi:
        raise DenseRecursiveNativeError(
            f"min_capacity {min_capacity} exceeds max_capacity {max_capacity}"
        )
    out: dict[str, int] = {}
    for uid, peak in peak_entries_by_uid.items():
        n = int(peak)
        if n < 0:
            raise DenseRecursiveNativeError(f"{uid}: negative hash peak {peak!r}")
        required = max(lo, math.ceil(n / float(target_load_factor)))
        cap = _native_hash_capacity(required)
        if cap > hi:
            raise DenseRecursiveNativeError(
                f"{uid}: required native hash capacity {cap} exceeds maximum {hi}"
            )
        out[str(uid)] = cap
    return out


def _native_dtype_bytes(dtype: str) -> int:
    if dtype in {"float64", "int64"}:
        return 8
    if dtype == "bool":
        return 1
    raise DenseRecursiveNativeError(dtype)


def _native_hash_seed(uid: str) -> int:
    """Stable per-Cell 64-bit seed; never use Python's randomized hash()."""
    raw = hashlib.sha256(uid.encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "little") | 1


def build_dense_memo_plan(
    formula_program: NativeFormulaProgram,
    *,
    parameter_ranges: Mapping[str, Sequence[tuple[int, int] | None]] | None = None,
    max_dense_slots_per_cell: int = 1_000_000,
    memoization_policy: str = PRODUCTION_MEMOIZATION_POLICY,
    native_hash_capacity: int | None = None,
    native_hash_capacity_by_uid: Mapping[str, int] | None = None,
    _allow_legacy_comparison: bool = False,
) -> DenseMemoPlan:
    """Build the per-Cell native memo plan from frontend-proven ABI evidence.

    Production ``native_cache`` uses scalar storage for zero-arity state, dense
    generation-tagged storage for bounded integer/category tuples, and exact
    native hash storage for every other typed key. Fixed hash capacity must be
    supplied by complete-RunDomain evidence; there is no guessed capacity or
    Python fallback in the production profile.

    Only ``native_cache`` is accepted through this production builder. Historical
    ``cache_all``, ``pure_map_direct`` and ``semantic_parameter`` planning is
    reachable only through the explicit legacy comparison builder.
    """

    if memoization_policy not in {"cache_all", "pure_map_direct", "semantic_parameter", "native_cache"}:
        raise DenseRecursiveNativeError(f"unknown memoization policy {memoization_policy!r}")
    if memoization_policy != PRODUCTION_MEMOIZATION_POLICY and not _allow_legacy_comparison:
        raise DenseRecursiveNativeError(
            f"memoization policy {memoization_policy!r} is comparison-only; "
            "use build_legacy_comparison_recursive_program()"
        )
    parameter_ranges = parameter_ranges or {}
    native_hash_capacity_by_uid = native_hash_capacity_by_uid or {}
    _graph, recursive_cells, pure_maps = _memo_semantics(
        formula_program,
        include_legacy_pure_maps=(memoization_policy == "pure_map_direct"),
    )
    semantic = None
    if memoization_policy == "semantic_parameter":
        from .semantic_parameter_policy import classify_semantic_parameter_policy
        semantic = classify_semantic_parameter_policy(
            formula_program.cells,
            parameter_ranges=parameter_ranges,
            max_dense_slots_per_cell=max_dense_slots_per_cell,
        )
    rows: list[DenseMemoCell] = []
    arena_offset = 0
    for uid in sorted(formula_program.cells):
        cell = formula_program.cells[uid]
        if cell.execution_role != "formula":
            continue
        if memoization_policy == "native_cache":
            if cell.arity == 0:
                rows.append(DenseMemoCell(
                    uid, 0, "scalar", (), (), 1, "zero_arity_scalar",
                    arena_offset, uid in recursive_cells,
                ))
                arena_offset += 1
                continue
            requested = tuple(parameter_ranges.get(uid, ()))
            if len(requested) != cell.arity:
                requested = tuple(None for _ in range(cell.arity))
            dense_bounds: list[tuple[int, int]] = []
            dense_safe = all(dtype in {"int64", "bool"} for dtype in cell.parameter_dtypes)
            if dense_safe:
                for dtype, item in zip(cell.parameter_dtypes, requested):
                    if item is None and dtype == "bool":
                        item = (0, 1)
                    if item is None:
                        dense_safe = False
                        break
                    lo, hi = int(item[0]), int(item[1])
                    if lo > hi:
                        raise DenseRecursiveNativeError(f"{uid}: invalid parameter range {(lo, hi)!r}")
                    dense_bounds.append((lo, hi))
            if dense_safe and len(dense_bounds) == cell.arity:
                strides, size = _dense_strides(dense_bounds)
                if size <= max_dense_slots_per_cell:
                    rows.append(DenseMemoCell(
                        uid, cell.arity, "dense", tuple(dense_bounds), strides, size,
                        "native_cache_bounded_dense", arena_offset, uid in recursive_cells,
                    ))
                    arena_offset += size
                    continue
            requested_capacity = native_hash_capacity_by_uid.get(uid, native_hash_capacity)
            cap = 0 if requested_capacity is None else _native_hash_capacity(requested_capacity)
            reason = (
                "native_exact_hash_float_or_mixed_key"
                if any(dtype == "float64" for dtype in cell.parameter_dtypes)
                else "native_exact_hash_non_dense_key"
            )
            if cap == 0:
                reason += ":complete_run_capacity_evidence_required"
            rows.append(DenseMemoCell(
                uid, cell.arity, "hash", tuple(dense_bounds), (), 0, reason,
                -1, True, cap,
            ))
            continue
        if semantic is not None:
            policy = semantic.cell(uid)
            if policy.mode == "direct":
                rows.append(DenseMemoCell(
                    uid, cell.arity, "direct", (), (), 0,
                    policy.reason, -1, False,
                ))
                continue
            if policy.mode == "scalar":
                rows.append(DenseMemoCell(
                    uid, 0, "scalar", (), (), 1, policy.reason,
                    arena_offset, uid in recursive_cells,
                ))
                arena_offset += 1
                continue
            if policy.mode == "dense":
                bounds = tuple(item for item in policy.bounds if item is not None)
                strides, size = _dense_strides(bounds)
                rows.append(DenseMemoCell(
                    uid, cell.arity, "dense", bounds, strides, size,
                    policy.reason, arena_offset, uid in recursive_cells,
                ))
                arena_offset += size
                continue
            rows.append(DenseMemoCell(
                uid, cell.arity, "dict", (), (), 0, policy.reason,
                -1, uid in recursive_cells,
            ))
            continue
        if memoization_policy == "pure_map_direct" and uid in pure_maps:
            rows.append(DenseMemoCell(
                uid, cell.arity, "direct", (), (), 0,
                "cycle_free_parameterized_pure_map", -1, False,
            ))
            continue
        if cell.arity == 0:
            rows.append(DenseMemoCell(
                uid, 0, "scalar", (), (), 1, "zero_arity_scalar",
                arena_offset, uid in recursive_cells,
            ))
            arena_offset += 1
            continue
        requested = tuple(parameter_ranges.get(uid, ()))
        if len(requested) != cell.arity:
            requested = tuple(None for _ in range(cell.arity))
        bounds: list[tuple[int, int]] = []
        safe = True
        for dtype, item in zip(cell.parameter_dtypes, requested):
            if dtype == "bool":
                if item is None:
                    item = (0, 1)
            elif dtype != "int64":
                safe = False
            if item is None:
                safe = False
            else:
                lo, hi = int(item[0]), int(item[1])
                if lo > hi:
                    raise DenseRecursiveNativeError(f"{uid}: invalid parameter range {(lo, hi)!r}")
                bounds.append((lo, hi))
        if not safe or len(bounds) != cell.arity:
            rows.append(DenseMemoCell(uid, cell.arity, "dict", (), (), 0, "unbounded_parameter_domain", -1, uid in recursive_cells))
            continue
        strides, size = _dense_strides(bounds)
        if size > max_dense_slots_per_cell:
            rows.append(DenseMemoCell(uid, cell.arity, "dict", tuple(bounds), strides, 0, "dense_slot_limit", -1, uid in recursive_cells))
        else:
            reason = "recursive_or_state_derived" if uid in recursive_cells else "cache_all_or_state_derived"
            rows.append(DenseMemoCell(
                uid, cell.arity, "dense", tuple(bounds), strides, size, reason,
                arena_offset, uid in recursive_cells,
            ))
            arena_offset += size
    value_offsets = {"float64": 0, "int64": 0, "bool": 0}
    compact_rows: list[DenseMemoCell] = []
    for row in rows:
        if row.kind in {"scalar", "dense"}:
            dtype = formula_program.cells[row.uid].dtype
            compact_rows.append(replace(row, value_offset=value_offsets[dtype]))
            value_offsets[dtype] += row.size
        else:
            compact_rows.append(row)
    memo_schema = (
        "modelx_graph.dense_recursive_memo.v6_compact_values_packed_state_native_exact_hash"
        if memoization_policy == "native_cache" else "modelx_graph.dense_recursive_memo.v6_compact_values_packed_state_legacy"
    )
    return DenseMemoPlan(memo_schema, tuple(compact_rows))


def build_dense_recursive_program(
    formula_program: NativeFormulaProgram,
    *,
    parameter_ranges: Mapping[str, Sequence[tuple[int, int] | None]] | None = None,
    max_dense_slots_per_cell: int = 1_000_000,
    memoization_policy: str = PRODUCTION_MEMOIZATION_POLICY,
    overflow_policy: str = PRODUCTION_OVERFLOW_POLICY,
    native_hash_capacity: int | None = None,
    native_hash_capacity_by_uid: Mapping[str, int] | None = None,
    native_hash_debug: bool = False,
    _allow_legacy_comparison: bool = False,
) -> DenseRecursiveProgram:
    if overflow_policy not in {"python_exact", "fail_closed"}:
        raise DenseRecursiveNativeError(f"unknown overflow policy {overflow_policy!r}")
    if not _allow_legacy_comparison and (
        memoization_policy != PRODUCTION_MEMOIZATION_POLICY
        or overflow_policy != PRODUCTION_OVERFLOW_POLICY
    ):
        raise DenseRecursiveNativeError(
            "non-production memo/overflow policies are comparison-only; "
            "use build_legacy_comparison_recursive_program()"
        )
    blockers: list[str] = []
    active = {
        uid: cell for uid, cell in formula_program.cells.items()
        if cell.execution_role == "formula"
    }
    for uid, cell in active.items():
        if cell.dtype not in {"float64", "int64", "bool"}:
            blockers.append(f"unsupported_result_dtype:{uid}:{cell.dtype}")
        if any(dtype not in {"int64", "bool", "float64"} for dtype in cell.parameter_dtypes):
            blockers.append(f"unsupported_parameter_dtype:{uid}:{cell.parameter_dtypes}")
        if cell.reduction is None and cell.function is None and cell.ranked_function is None:
            blockers.append(f"missing_formula:{uid}")
    out = formula_program.cells.get(formula_program.output_uid)
    if out is None:
        blockers.append("output_missing")
    elif out.execution_role != "formula":
        blockers.append("output_is_not_fast_formula")
    elif out.arity != 0:
        blockers.append("run_all_requires_zero_arity_output")
    elif out.logical_result_ndim not in {0, 1}:
        blockers.append(
            f"run_all_unsupported_output_rank:{out.logical_result_ndim}"
        )
    elif out.logical_result_ndim == 1 and (
        out.logical_result_shape is None
        or len(out.logical_result_shape) != 1
        or out.logical_result_shape[0] <= 0
    ):
        blockers.append("run_all_requires_fixed_positive_rank1_shape")
    memo = build_dense_memo_plan(
        formula_program,
        parameter_ranges=parameter_ranges,
        max_dense_slots_per_cell=max_dense_slots_per_cell,
        memoization_policy=memoization_policy,
        native_hash_capacity=native_hash_capacity,
        native_hash_capacity_by_uid=native_hash_capacity_by_uid,
        _allow_legacy_comparison=_allow_legacy_comparison,
    )
    if overflow_policy == "fail_closed":
        for row in memo.cells:
            if row.kind == "dict":
                blockers.append(f"fail_closed_requires_bounded_parameter_domain:{row.uid}")
    if memoization_policy == "native_cache" and overflow_policy != "fail_closed":
        blockers.append("native_cache_requires_no_python_overflow")
    if memoization_policy == "native_cache":
        for row in memo.cells:
            if row.kind == "dict":
                blockers.append(f"native_cache_python_memo_forbidden:{row.uid}")
            if row.kind == "hash" and row.hash_capacity <= 0:
                blockers.append(f"native_hash_capacity_evidence_required:{row.uid}")
    if memoization_policy == "semantic_parameter":
        for row in memo.cells:
            if row.kind == "dict":
                blockers.append(f"semantic_parameter_policy_blocked:{row.uid}:{row.decision_reason}")
    return DenseRecursiveProgram(
        schema=(
            "modelx_graph.dense_recursive_program.v6_compact_values_packed_state_native_exact_hash"
            if memoization_policy == "native_cache" else "modelx_graph.dense_recursive_program.v6_compact_values_packed_state_legacy"
        ),
        formula_program=formula_program,
        memo_plan=memo,
        supported=not blockers,
        blockers=tuple(sorted(set(blockers))),
        memoization_policy=memoization_policy,
        overflow_policy=overflow_policy,
        native_hash_debug=bool(native_hash_debug),
    )


def build_legacy_comparison_recursive_program(
    formula_program: NativeFormulaProgram,
    *,
    parameter_ranges: Mapping[str, Sequence[tuple[int, int] | None]] | None = None,
    max_dense_slots_per_cell: int = 1_000_000,
    memoization_policy: str,
    overflow_policy: str,
    native_hash_capacity: int | None = None,
    native_hash_capacity_by_uid: Mapping[str, int] | None = None,
    native_hash_debug: bool = False,
) -> DenseRecursiveProgram:
    """Build an explicit historical memo-policy comparison program.

    This API exists only to reproduce cache-all/Python-dict and direct-call
    benchmarks during the transition. Normal compiler callers must use
    ``build_dense_recursive_program`` / ``ModelxCythonFastPlan.build_recursive_program``
    and therefore receive native-cache + fail-closed semantics.
    """
    if memoization_policy not in LEGACY_COMPARISON_MEMO_POLICIES:
        raise DenseRecursiveNativeError(
            f"legacy comparison policy required, got {memoization_policy!r}"
        )
    return build_dense_recursive_program(
        formula_program,
        parameter_ranges=parameter_ranges,
        max_dense_slots_per_cell=max_dense_slots_per_cell,
        memoization_policy=memoization_policy,
        overflow_policy=overflow_policy,
        native_hash_capacity=native_hash_capacity,
        native_hash_capacity_by_uid=native_hash_capacity_by_uid,
        native_hash_debug=native_hash_debug,
        _allow_legacy_comparison=True,
    )


class DenseRecursiveCythonGenerator(RecursiveCythonGenerator):

    def _ranked_type(self, cell: NativeFormulaCell, *, output: bool = False) -> str:
        if cell.logical_result_ndim == 1:
            if output:
                ctype = "double" if cell.dtype == "float64" else (
                    "long long" if cell.dtype == "int64" else "cnp.npy_bool"
                )
                return f"const {ctype}[:]"
            return "object"
        return self.ctype(cell.dtype)

    def _ranked_function_source(self, cell: NativeFormulaCell) -> str:
        fn = cell.ranked_function if cell.ranked_function is not None else cell.function
        if fn is None:
            raise DenseRecursiveNativeError(
                f"ranked formula {cell.source_fullname} has no preserved function"
            )

        known = set(self.v)
        input_specs = self.input_specs

        class Lower(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call):  # noqa: N802
                node = self.generic_visit(node)
                if not isinstance(node.func, ast.Name):
                    return node
                name = node.func.id
                if name in known:
                    return ast.copy_location(
                        ast.Call(ast.Name("r_" + name, ast.Load()), node.args, node.keywords),
                        node,
                    )
                if name in {"point_input", "ranked_input", "global_input"}:
                    if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
                        raise DenseRecursiveNativeError(f"{name} requires a literal ABI key")
                    key = node.args[0].value
                    if key not in input_specs:
                        raise DenseRecursiveNativeError(f"unknown ranked ABI key {key!r}")
                    return ast.copy_location(ast.Name(str(key), ast.Load()), node)
                if name in {"array_input", "table_input"}:
                    if not node.args or not isinstance(node.args[0], ast.Constant):
                        raise DenseRecursiveNativeError(f"{name} requires a literal ABI key")
                    key = node.args[0].value
                    if key not in input_specs:
                        raise DenseRecursiveNativeError(f"unknown ranked ABI key {key!r}")
                    indices = node.args[1:]
                    if len(indices) != input_specs[key].ndim:
                        raise DenseRecursiveNativeError(
                            f"{name} {key!r} rank mismatch in ranked formula"
                        )
                    slice_node: ast.expr
                    if len(indices) == 1:
                        slice_node = indices[0]
                    else:
                        slice_node = ast.Tuple(indices, ast.Load())
                    return ast.copy_location(
                        ast.Subscript(ast.Name(str(key), ast.Load()), slice_node, ast.Load()),
                        node,
                    )
                if name == "ranked_where":
                    return ast.copy_location(
                        ast.Call(
                            ast.Attribute(ast.Name("np", ast.Load()), "where", ast.Load()),
                            node.args,
                            node.keywords,
                        ),
                        node,
                    )
                if name == "ranked_series":
                    if len(node.args) != 2:
                        raise DenseRecursiveNativeError("ranked_series requires value and size")
                    return ast.copy_location(
                        ast.Call(
                            ast.Attribute(ast.Name("np", ast.Load()), "broadcast_to", ast.Load()),
                            [
                                ast.Call(
                                    ast.Attribute(ast.Name("np", ast.Load()), "asarray", ast.Load()),
                                    [node.args[0]],
                                    [],
                                ),
                                ast.Tuple([node.args[1]], ast.Load()),
                            ],
                            [],
                        ),
                        node,
                    )
                return node

        lowered = Lower().visit(copy.deepcopy(fn))
        assert isinstance(lowered, ast.FunctionDef)
        lowered = ast.fix_missing_locations(lowered)
        params = []
        for name, dtype in zip(cell.parameters, cell.parameter_dtypes):
            params.append(f"{self.ctype(dtype)} {name}")
        return_type = self._ranked_type(cell)
        lines = [f"cdef {return_type} calc_{cell.uid}({', '.join(params)}):"]
        for statement in lowered.body:
            rendered = ast.unparse(statement)
            lines.append(textwrap.indent(rendered, "    "))
        if not lowered.body:
            lines.append("    pass")
        lines.append("")
        return "\n".join(lines)

    def _ranked_source(self, module_name: str) -> str:
        out = self.v[self.formula.output_uid]
        if out.logical_result_ndim != 1 or out.logical_result_shape is None:
            raise DenseRecursiveNativeError("ranked source requires a fixed rank-1 output")
        lines = [
            "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False",
            "cimport numpy as cnp",
            "import numpy as np",
            "",
            "cdef object __guard_fail__(long long code):",
            "    raise ValueError(f'native formula guard failed with code {code}')",
            "",
        ]
        if self.has_valid_exact_axis_intrinsic:
            lines += [
                "cdef long long __exact_axis_invalid__(long long code):",
                "    raise ValueError(f'native exact category axis invalid with code {code}')",
                "",
            ]
        for spec in self.formula.inputs:
            declaration = self.ctype(spec.dtype) if spec.ndim == 0 else "object"
            lines.append(f"cdef {declaration} {spec.key}")
        if self.formula.inputs:
            lines.append("")

        for uid in sorted(self.v):
            cell = self.v[uid]
            if cell.arity == 0:
                lines += [
                    f"cdef object _ranked_value_{uid}",
                    f"cdef bint _ranked_has_{uid} = False",
                ]
            else:
                lines.append(f"cdef dict _ranked_cache_{uid} = {{}}")
        lines.append("")

        for uid in sorted(self.v):
            lines.append(self._ranked_function_source(self.v[uid]))

        # Emit wrappers after all calc declarations.  Cython resolves these
        # module-local symbols statically while the formula bodies keep the
        # original array operations and recursive call structure.
        for uid in sorted(self.v):
            cell = self.v[uid]
            params = [
                f"{self.ctype(dtype)} {name}"
                for name, dtype in zip(cell.parameters, cell.parameter_dtypes)
            ]
            args = ", ".join(cell.parameters)
            return_type = self._ranked_type(cell, output=uid == out.uid)
            lines.append(f"cdef {return_type} r_{uid}({', '.join(params)}):")
            if cell.arity == 0:
                lines += [
                    f"    global _ranked_value_{uid}, _ranked_has_{uid}",
                    f"    if _ranked_has_{uid}:",
                    f"        return _ranked_value_{uid}",
                    f"    _ranked_value_{uid} = calc_{uid}()",
                    f"    _ranked_has_{uid} = True",
                    f"    return _ranked_value_{uid}",
                ]
            else:
                key = cell.parameters[0] if cell.arity == 1 else "(" + args + ")"
                lines += [
                    f"    cdef object key = {key}",
                    f"    cdef object value",
                    f"    if key in _ranked_cache_{uid}:",
                    f"        return _ranked_cache_{uid}[key]",
                    f"    value = calc_{uid}({args})",
                    f"    _ranked_cache_{uid}[key] = value",
                    f"    return value",
                ]
            lines.append("")

        globals_ = [spec.key for spec in self.formula.inputs]
        lines.append("cpdef init_inputs(dict d):")
        if globals_:
            lines.append("    global " + ", ".join(globals_))
        for spec in self.formula.inputs:
            if spec.ndim == 0:
                lines += self._scalar_input_init_lines(spec)
            else:
                npdt = "np.float64" if spec.dtype == "float64" else (
                    "np.bool_" if spec.dtype == "bool" else "np.int64"
                )
                lines.append(
                    f"    {spec.key} = np.ascontiguousarray(d[{spec.key!r}], dtype={npdt})"
                )
                if spec.expected_shape is not None:
                    shape = tuple(int(x) for x in spec.expected_shape)
                    lines += [
                        f"    if {spec.key}.shape != {shape!r}:",
                        f"        raise ValueError('input {spec.key} shape mismatch')",
                    ]
        for uid in sorted(self.v):
            cell = self.v[uid]
            if cell.arity == 0:
                lines += [
                    f"    global _ranked_has_{uid}, _ranked_value_{uid}",
                    f"    _ranked_has_{uid} = False",
                    f"    _ranked_value_{uid} = None",
                ]
            else:
                lines.append(f"    _ranked_cache_{uid}.clear()")
        lines += ["    return None", ""]
        expected = tuple(int(x) for x in out.logical_result_shape)
        npdt = "np.float64" if out.dtype == "float64" else (
            "np.bool_" if out.dtype == "bool" else "np.int64"
        )
        lines += [
            "cpdef cnp.ndarray run_all():",
            f"    cdef cnp.ndarray result = np.asarray(r_{out.uid}(), dtype={npdt})",
            f"    if result.ndim != 1 or result.shape[0] != {expected[0]}:",
            "        raise ValueError('rank-1 output shape mismatch')",
            "    return result",
            "",
        ]
        return "\n".join(lines)
    """Recursive Cython generator with typed dense generation-tag caches."""

    def __init__(self, program: DenseRecursiveProgram):
        if not program.supported:
            raise DenseRecursiveNativeError(
                "dense recursive program is blocked: " + "; ".join(program.blockers)
            )
        # Do not call RecursiveCythonGenerator.__init__: its nominal program type
        # carries a hash memo plan, but the formula-side setup is identical.
        self.program = program
        self.formula = program.formula_program
        self.all_cells = self.formula.cells
        self.v = {
            uid: cell for uid, cell in self.formula.cells.items()
            if cell.execution_role == "formula"
        }
        self.input_specs = {x.key: x for x in self.formula.inputs}
        self.input_order = list(self.formula.input_order)
        self.module_refs = dict(self.formula.module_refs)
        self._detect_intrinsics()

    @staticmethod
    def _zero(dtype: str) -> str:
        return "NAN" if dtype == "float64" else "0"

    @staticmethod
    def _array_value_type(dtype: str) -> str:
        if dtype == "float64":
            return "double"
        if dtype == "int64":
            return "long long"
        if dtype == "bool":
            return "unsigned char"
        raise DenseRecursiveNativeError(dtype)

    @staticmethod
    def _safe_uid(uid: str) -> str:
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in uid)

    def _memo_names(self, uid: str) -> tuple[str, str, str]:
        safe = self._safe_uid(uid)
        return f"_mv_{safe}", f"_mg_{safe}", f"_mb_{safe}"

    def _overflow_name(self, uid: str) -> str:
        return "_overflow_" + self._safe_uid(uid)

    def _hash_prefix(self, uid: str) -> str:
        return "_nh_" + self._safe_uid(uid)

    @staticmethod
    def _hash_key_ctype(dtype: str) -> str:
        if dtype == "float64":
            return "uint64_t"
        if dtype == "int64":
            return "int64_t"
        if dtype == "bool":
            return "unsigned char"
        raise DenseRecursiveNativeError(dtype)

    def _hash_value_name(self, uid: str) -> str:
        return self._hash_prefix(uid) + "_value"

    def _hash_gen_name(self, uid: str) -> str:
        return self._hash_prefix(uid) + "_gen"

    def _hash_busy_name(self, uid: str) -> str:
        return self._hash_prefix(uid) + "_busy"

    def _hash_key_name(self, uid: str, index: int) -> str:
        return self._hash_prefix(uid) + f"_key{index}"

    def _hash_stat_name(self, uid: str, suffix: str) -> str:
        return self._hash_prefix(uid) + "_" + suffix

    @staticmethod
    def _arena_value_name(dtype: str) -> str:
        if dtype == "float64":
            return "_memo_f64"
        if dtype == "int64":
            return "_memo_i64"
        if dtype == "bool":
            return "_memo_bool"
        raise DenseRecursiveNativeError(dtype)

    @staticmethod
    def _arena_state_name(dtype: str) -> str:
        if dtype == "float64":
            return "_memo_f64_state"
        if dtype == "int64":
            return "_memo_i64_state"
        if dtype == "bool":
            return "_memo_bool_state"
        raise DenseRecursiveNativeError(dtype)

    def _overflow_signature(self, cell: NativeFormulaCell) -> str:
        args = [f"Py_ssize_t {self._policy_index_name(cell)}"]
        for name, dtype in zip(cell.parameters, cell.parameter_dtypes):
            args.append(f"{self.ctype(dtype)} {name}")
        return f"cdef {self.ctype(cell.dtype)} {self._overflow_name(cell.uid)}({', '.join(args)})"

    def _emit_overflow(self, cell: NativeFormulaCell) -> list[str]:
        # The overflow path deliberately requires the GIL because its exact
        # compatibility fallback is a Python dict.  Dense in-range recursion
        # never enters this function and remains fully nogil.
        sig = self._overflow_signature(cell) + ":"
        policy_i = self._policy_index_name(cell)
        ct = self.ctype(cell.dtype)
        zero = self._zero(cell.dtype)
        args = ", ".join(cell.parameters)
        calc_suffix = "" if not args else ", " + args
        key_items = [repr(cell.uid), "int(_epoch)"]
        for name, dtype in zip(cell.parameters, cell.parameter_dtypes):
            key_items.append(f"float({name})" if dtype == "float64" else f"int({name})")
        key_expr = "(" + ", ".join(key_items) + ("," if len(key_items) == 1 else "") + ")"
        return [
            sig,
            f"    cdef {ct} val",
            "    cdef object key",
            "    cdef object cached",
            "    global _memo_error",
            f"    key = {key_expr}",
            "    cached = _overflow_cache.get(key, _OVERFLOW_MISSING)",
            "    if cached is _OVERFLOW_BUSY:",
            "        _memo_error = 2",
            f"        return <{ct}>{zero}",
            "    if cached is not _OVERFLOW_MISSING:",
            f"        return <{ct}>cached",
            "    _overflow_cache[key] = _OVERFLOW_BUSY",
            f"    val = calc_{cell.uid}({policy_i}{calc_suffix})",
            "    _overflow_cache[key] = val",
            "    return val",
            "",
        ]

    def _emit_hash_wrapper(self, cell: NativeFormulaCell, memo: DenseMemoCell) -> list[str]:
        """Emit one signature-specialized native exact hash memo wrapper."""
        sig = self._function_signature(cell, "r") + ":"
        policy_i = self._policy_index_name(cell)
        ct = self.ctype(cell.dtype)
        zero = self._zero(cell.dtype)
        cap = int(memo.hash_capacity)
        if cap <= 0 or cap & (cap - 1):
            raise DenseRecursiveNativeError(f"{cell.uid}: native hash capacity must be power of two")
        mask = cap - 1
        prefix = self._hash_prefix(cell.uid)
        values = self._hash_value_name(cell.uid)
        gens = self._hash_gen_name(cell.uid)
        busy = self._hash_busy_name(cell.uid)
        callargs = ", ".join(cell.parameters)
        suffix = "" if not callargs else ", " + callargs
        seed = _native_hash_seed(cell.uid)
        debug_globals: list[str] = []
        if self.program.native_hash_debug:
            debug_globals = [
                self._hash_stat_name(cell.uid, suffix)
                for suffix in ("hits", "inserts", "collisions", "current", "peak", "stats_epoch")
            ]
        lines = [
            sig,
            f"    cdef uint64_t h = <uint64_t>{seed}ULL",
            "    cdef Py_ssize_t pos",
            "    cdef Py_ssize_t probe",
            f"    cdef {ct} val",
            "    cdef bint match",
        ]
        for j, dtype in enumerate(cell.parameter_dtypes):
            lines.append(f"    cdef {self._hash_key_ctype(dtype)} key{j}")
        lines.append("    global " + ", ".join(["_memo_error", *debug_globals]))
        for j, (name, dtype) in enumerate(zip(cell.parameters, cell.parameter_dtypes)):
            if dtype == "float64":
                lines += [
                    f"    if {name} != {name}:",
                    "        _memo_error = 5",
                    f"        return <{ct}>{zero}",
                    f"    key{j} = __memo_f64_bits({name})",
                    f"    h = __memo_mix64(h ^ __memo_mix64(key{j} + <uint64_t>{j + 1}ULL))",
                ]
            elif dtype == "int64":
                lines += [
                    f"    key{j} = <int64_t>{name}",
                    f"    h = __memo_mix64(h ^ __memo_mix64(<uint64_t>key{j} + <uint64_t>{j + 1}ULL))",
                ]
            elif dtype == "bool":
                lines += [
                    f"    key{j} = <unsigned char>{name}",
                    f"    h = __memo_mix64(h ^ __memo_mix64(<uint64_t>key{j} + <uint64_t>{j + 1}ULL))",
                ]
            else:
                raise DenseRecursiveNativeError(dtype)
        if self.program.native_hash_debug:
            se = self._hash_stat_name(cell.uid, "stats_epoch")
            cur = self._hash_stat_name(cell.uid, "current")
            peak = self._hash_stat_name(cell.uid, "peak")
            lines += [
                f"    if {se} != _epoch:",
                f"        {se} = _epoch",
                f"        {cur} = 0",
            ]
        lines += [
            f"    pos = <Py_ssize_t>(h & <uint64_t>{mask})",
            f"    for probe in range({cap}):",
            f"        if {gens}[pos] != _epoch:",
            f"            {gens}[pos] = _epoch",
            f"            {busy}[pos] = 1",
        ]
        for j in range(cell.arity):
            lines.append(f"            {self._hash_key_name(cell.uid, j)}[pos] = key{j}")
        if self.program.native_hash_debug:
            inserts = self._hash_stat_name(cell.uid, "inserts")
            cur = self._hash_stat_name(cell.uid, "current")
            peak = self._hash_stat_name(cell.uid, "peak")
            lines += [
                f"            {inserts} += 1",
                f"            {cur} += 1",
                f"            if {cur} > {peak}:",
                f"                {peak} = {cur}",
            ]
        lines += [
            f"            val = calc_{cell.uid}({policy_i}{suffix})",
            f"            {values}[pos] = <{self._array_value_type(cell.dtype)}>val",
            f"            {busy}[pos] = 0",
            "            return val",
            "        match = True",
        ]
        for j in range(cell.arity):
            lines.append(f"        match = match and ({self._hash_key_name(cell.uid, j)}[pos] == key{j})")
        lines += [
            "        if match:",
            f"            if {busy}[pos]:",
            "                _memo_error = 2",
            f"                return <{ct}>{zero}",
        ]
        if self.program.native_hash_debug:
            hits = self._hash_stat_name(cell.uid, "hits")
            lines.append(f"            {hits} += 1")
        lines += [
            f"            return <{ct}>{values}[pos]",
        ]
        if self.program.native_hash_debug:
            collisions = self._hash_stat_name(cell.uid, "collisions")
            lines.append(f"        {collisions} += 1")
        lines += [
            f"        pos = (pos + 1) & {mask}",
            "    _memo_error = 1",
            f"    return <{ct}>{zero}",
            "",
        ]
        return lines

    def _emit_wrapper(self, cell: NativeFormulaCell) -> list[str]:
        memo = self.program.memo_plan.cell(cell.uid)
        sig = self._function_signature(cell, "r") + ":"
        policy_i = self._policy_index_name(cell)
        ct = self.ctype(cell.dtype)
        zero = self._zero(cell.dtype)
        values = self._arena_value_name(cell.dtype)
        states = self._arena_state_name(cell.dtype)
        callargs = ", ".join(cell.parameters)
        suffix = "" if not callargs else ", " + callargs

        if memo.kind == "direct":
            return [sig, f"    return calc_{cell.uid}({policy_i}{suffix})", ""]

        if memo.kind == "dict":
            return [
                sig,
                "    with gil:",
                f"        return {self._overflow_name(cell.uid)}({policy_i}{suffix})",
                "",
            ]

        if memo.kind == "hash":
            return self._emit_hash_wrapper(cell, memo)

        lines = [
            sig,
            f"    cdef Py_ssize_t idx = {memo.value_offset}",
            f"    cdef {ct} val",
            "    global _memo_error",
        ]
        if memo.kind == "dense":
            checks: list[str] = []
            terms: list[str] = []
            for name, (lo, hi), stride in zip(cell.parameters, memo.bounds, memo.strides):
                checks.extend((f"{name} < {lo}", f"{name} > {hi}"))
                term = f"({name} - ({lo}))"
                if stride != 1:
                    term = f"({term}) * {stride}"
                terms.append(term)
            lines.append("    if " + " or ".join(checks) + ":")
            if self.program.overflow_policy == "python_exact":
                lines += [
                    "        with gil:",
                    f"            return {self._overflow_name(cell.uid)}({policy_i}{suffix})",
                ]
            else:
                lines += [
                    "        _memo_error = 4",
                    f"        return <{ct}>{zero}",
                ]
            offset_expr = " + ".join(terms)
            lines.append(f"    idx = {memo.value_offset} + " + offset_expr)

        lines.append(
            f"    if ({states}[idx] & <unsigned int>0x7fffffff) == _epoch:"
        )
        if memo.cycle_guard:
            lines += [
                f"        if ({states}[idx] & <unsigned int>0x80000000) != 0:",
                "            _memo_error = 2",
                f"            return <{ct}>{zero}",
            ]
        lines.append(f"        return <{ct}>{values}[idx]")
        if memo.cycle_guard:
            lines.append(f"    {states}[idx] = _epoch | <unsigned int>0x80000000")
        lines += [
            f"    val = calc_{cell.uid}({policy_i}{suffix})",
            f"    {values}[idx] = <{self._array_value_type(cell.dtype)}>val",
            f"    {states}[idx] = _epoch",
        ]
        lines += ["    return val", ""]
        return lines

    def _hash_declarations(self) -> list[str]:
        lines: list[str] = []
        for row in self.program.memo_plan.cells:
            if row.kind != "hash":
                continue
            cell = self.v[row.uid]
            cap = int(row.hash_capacity)
            lines += [
                f"cdef {self._array_value_type(cell.dtype)} {self._hash_value_name(row.uid)}[{cap}]",
                f"cdef unsigned int {self._hash_gen_name(row.uid)}[{cap}]",
                f"cdef unsigned char {self._hash_busy_name(row.uid)}[{cap}]",
            ]
            for j, dtype in enumerate(cell.parameter_dtypes):
                lines.append(f"cdef {self._hash_key_ctype(dtype)} {self._hash_key_name(row.uid, j)}[{cap}]")
            if self.program.native_hash_debug:
                for suffix in ("hits", "inserts", "collisions", "current", "peak"):
                    lines.append(f"cdef unsigned long long {self._hash_stat_name(row.uid, suffix)} = 0")
                lines.append(f"cdef unsigned int {self._hash_stat_name(row.uid, 'stats_epoch')} = 0")
        return lines

    def _hash_clear_generation_lines(self, indent: str = "    ") -> list[str]:
        lines: list[str] = []
        for row in self.program.memo_plan.cells:
            if row.kind == "hash":
                lines.append(
                    f"{indent}memset(&{self._hash_gen_name(row.uid)}[0], 0, "
                    f"{row.hash_capacity} * sizeof(unsigned int))"
                )
        return lines

    def _hash_debug_reset_lines(self, indent: str = "    ") -> list[str]:
        if not self.program.native_hash_debug:
            return []
        lines: list[str] = []
        for row in self.program.memo_plan.cells:
            if row.kind != "hash":
                continue
            for suffix in ("hits", "inserts", "collisions", "current", "peak", "stats_epoch"):
                lines.append(f"{indent}{self._hash_stat_name(row.uid, suffix)} = 0")
        return lines

    def _hash_debug_stats_source(self) -> list[str]:
        if not self.program.native_hash_debug:
            return []
        hash_rows = [x for x in self.program.memo_plan.cells if x.kind == "hash"]
        globals_ = []
        for row in hash_rows:
            globals_.extend(self._hash_stat_name(row.uid, s) for s in ("hits", "inserts", "collisions", "peak"))
        lines = ["cpdef dict memo_hash_stats():"]
        if globals_:
            lines.append("    global " + ", ".join(globals_))
        lines.append("    return {")
        for row in hash_rows:
            cap = int(row.hash_capacity)
            lines.append(
                f"        {row.uid!r}: {{'capacity': {cap}, "
                f"'hits': int({self._hash_stat_name(row.uid, 'hits')}), "
                f"'inserts': int({self._hash_stat_name(row.uid, 'inserts')}), "
                f"'collisions': int({self._hash_stat_name(row.uid, 'collisions')}), "
                f"'peak_entries': int({self._hash_stat_name(row.uid, 'peak')}), "
                f"'peak_load_factor': float({self._hash_stat_name(row.uid, 'peak')}) / {cap}.0}},"
            )
        lines += ["    }", ""]
        return lines

    def _memo_value_slots(self) -> dict[str, int]:
        slots = {"float64": 0, "int64": 0, "bool": 0}
        for row in self.program.memo_plan.cells:
            if row.kind in {"scalar", "dense"}:
                slots[self.v[row.uid].dtype] += row.size
        return slots

    def _memo_declarations(self) -> list[str]:
        slots = self._memo_value_slots()
        lines: list[str] = []
        for dtype in ("float64", "int64", "bool"):
            n = max(1, slots[dtype])
            lines += [
                f"cdef {self._array_value_type(dtype)} {self._arena_value_name(dtype)}[{n}]",
                f"cdef unsigned int {self._arena_state_name(dtype)}[{n}]",
            ]
        return lines

    def _clear_generation_lines(self, indent: str = "    ") -> list[str]:
        slots = self._memo_value_slots()
        lines: list[str] = []
        for dtype in ("float64", "int64", "bool"):
            n = max(1, slots[dtype])
            lines.append(
                f"{indent}memset(&{self._arena_state_name(dtype)}[0], 0, "
                f"{n} * sizeof(unsigned int))"
            )
        return lines + self._hash_clear_generation_lines(indent)

    def source(self, module_name: str = "dense_recursive_native") -> str:
        output = self.v[self.formula.output_uid]
        if output.logical_result_ndim == 1 and output.ranked_function is not None:
            return self._ranked_source(module_name)
        needs_overflow = any(
            row.kind == "dict"
            or (row.kind == "dense" and self.program.overflow_policy == "python_exact")
            for row in self.program.memo_plan.cells
        )
        has_native_hash = any(row.kind == "hash" for row in self.program.memo_plan.cells)
        lines = [
            "# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False",
            self._math_cimport_line(),
            "from libc.string cimport memset, memcpy" if has_native_hash else "from libc.string cimport memset",
        ]
        if has_native_hash:
            lines.append("from libc.stdint cimport uint64_t, int64_t")
        lines += [
            "cimport numpy as cnp",
            "import numpy as np",
            "",
            "cdef unsigned int _epoch = 1",
            "cdef long long _memo_error = 0",
            "cdef long long _guard_error = 0",
            "cdef Py_ssize_t _n = 0",
        ]
        if has_native_hash:
            lines += [
                "",
                "cdef inline uint64_t __memo_mix64(uint64_t x) noexcept nogil:",
                "    x ^= x >> 30",
                "    x *= <uint64_t>0xbf58476d1ce4e5b9ULL",
                "    x ^= x >> 27",
                "    x *= <uint64_t>0x94d049bb133111ebULL",
                "    x ^= x >> 31",
                "    return x",
                "",
                "cdef inline uint64_t __memo_f64_bits(double x) noexcept nogil:",
                "    cdef uint64_t bits",
                "    if x == 0.0:",
                "        return <uint64_t>0",
                "    memcpy(&bits, &x, sizeof(double))",
                "    return bits",
            ]
        if self.has_valid_exact_axis_intrinsic:
            lines += [
                "cdef inline long long __exact_axis_invalid__(long long code) noexcept nogil:",
                "    global _guard_error",
                "    _guard_error = code",
                "    return 0",
            ]
        if needs_overflow:
            lines += [
                "cdef dict _overflow_cache = {}",
                "cdef object _OVERFLOW_BUSY = object()",
                "cdef object _OVERFLOW_MISSING = object()",
            ]
        lines += [""]
        lines += self._memo_declarations()
        lines += self._hash_declarations()
        lines.append("")
        for spec in self.formula.inputs:
            lines.append(f"cdef {self._memoryview_type(spec)} {spec.key}")
        lines.append("")
        lines += self._lookup_helpers()

        for uid in sorted(self.v):
            cell = self.v[uid]
            lines += self._emit_reduction(cell) if cell.reduction is not None else self._emit_formula(cell)
        for uid in sorted(self.v):
            memo = self.program.memo_plan.cell(uid)
            if memo.kind == "dict" or (memo.kind == "dense" and self.program.overflow_policy == "python_exact"):
                lines += self._emit_overflow(self.v[uid])
        for uid in sorted(self.v):
            lines += self._emit_wrapper(self.v[uid])

        point_specs = [x for x in self.formula.inputs if x.scope == "point"]
        out = self.v[self.formula.output_uid]
        ranked_size = (
            int(out.logical_result_shape[0])
            if out.logical_result_ndim == 1 and out.logical_result_shape is not None
            else None
        )
        # A model with only global inputs and a scalar selected output is a
        # perfectly valid one-item run.  Older runners inferred batch length
        # exclusively from point-scoped arrays and therefore rejected models
        # such as smithwilson even though no point axis is semantically needed.
        # Keep the existing point/ranked authorities when present; otherwise
        # execute the scalar root once at policy index 0.
        implicit_scalar_run = not point_specs and ranked_size is None
        global_names = ["_epoch", "_memo_error", "_guard_error", "_n"] + [x.key for x in self.formula.inputs]
        if self.program.native_hash_debug:
            for row in self.program.memo_plan.cells:
                if row.kind == "hash":
                    global_names.extend(
                        self._hash_stat_name(row.uid, suffix)
                        for suffix in ("hits", "inserts", "collisions", "current", "peak", "stats_epoch")
                    )
        lines += ["cpdef init_inputs(dict d):", "    global " + ", ".join(global_names)]
        for spec in self.formula.inputs:
            key = spec.key
            if spec.ndim == 0:
                lines += self._scalar_input_init_lines(spec)
            else:
                npdt = "np.float64" if spec.dtype == "float64" else ("np.bool_" if spec.dtype == "bool" else "np.int64")
                lines.append(f"    {key} = np.ascontiguousarray(d[{key!r}], dtype={npdt})")
                if spec.expected_shape is not None:
                    shape = tuple(int(x) for x in spec.expected_shape)
                    checks = " or ".join(f"{key}.shape[{j}] != {dim}" for j, dim in enumerate(shape))
                    if checks:
                        lines += [f"    if {checks}:", f"        raise ValueError('input {key} shape mismatch')"]
        if point_specs:
            first = point_specs[0].key
            lines.append(f"    _n = {first}.shape[0]")
            for spec in point_specs[1:]:
                lines += [f"    if {spec.key}.shape[0] != _n:", f"        raise ValueError('point input length mismatch: {spec.key}')"]
            if ranked_size is not None:
                lines += [
                    f"    if _n != {ranked_size}:",
                    "        raise ValueError('rank-1 output axis length mismatch')",
                ]
        else:
            lines.append(f"    _n = {1 if implicit_scalar_run else ranked_size}")
        lines += ["    _epoch = 1"]
        if needs_overflow:
            lines.append("    _overflow_cache.clear()")
        lines += self._clear_generation_lines()
        lines += self._hash_debug_reset_lines()
        lines += [
            "    _memo_error = 0",
            "    _guard_error = 0",
            "    return None",
            "",
            "cdef inline void _next_epoch() noexcept nogil:",
            "    global _epoch",
            "    _epoch += 1",
            "    if _epoch > <unsigned int>0x7fffffff:",
        ]
        clear = self._clear_generation_lines(indent="        ")
        lines += clear or ["        pass"]
        lines += ["        _epoch = 1", ""]

        lines += [
            "cdef inline double _eval_one(Py_ssize_t i) noexcept nogil:",
            "    _next_epoch()",
            f"    return <double>r_{out.uid}(i)",
            "",
            "cpdef cnp.ndarray run_all():",
            "    cdef cnp.ndarray arr = np.empty(_n, dtype=np.float64)",
            "    cdef double[::1] outv = arr",
            "    cdef Py_ssize_t i",
            "    global _memo_error, _guard_error",
            "    _memo_error = 0",
            "    _guard_error = 0",
        ]
        if needs_overflow:
            lines.append("    _overflow_cache.clear()")
        lines += [
            "    with nogil:",
            "        for i in range(_n):",
            "            outv[i] = _eval_one(i)",
            "            if _memo_error != 0 or _guard_error != 0:",
            "                break",
            "    if _memo_error != 0:",
            "        raise RuntimeError(f'recursive memo failure code {_memo_error}')",
            "    if _guard_error != 0:",
            "        raise ValueError(f'native formula guard failed with code {_guard_error}')",
            "    return arr",
            "",
            "cpdef double bench(long long reps):",
            "    cdef long long j",
            "    cdef Py_ssize_t i",
            "    cdef double acc = 0.0",
            "    global _memo_error, _guard_error",
            "    _memo_error = 0",
            "    _guard_error = 0",
        ]
        if needs_overflow:
            lines.append("    _overflow_cache.clear()")
        lines += [
            "    with nogil:",
            "        for j in range(reps):",
            "            for i in range(_n):",
            "                acc += _eval_one(i)",
            "                if _memo_error != 0 or _guard_error != 0:",
            "                    break",
            "    if _memo_error != 0:",
            "        raise RuntimeError(f'recursive memo failure code {_memo_error}')",
            "    if _guard_error != 0:",
            "        raise ValueError(f'native formula guard failed with code {_guard_error}')",
            "    return acc",
            "",
        ]
        lines += self._hash_debug_stats_source()
        return "\n".join(lines) + "\n"

    def declaration_source(self) -> str:
        output = self.v[self.formula.output_uid]
        if output.logical_result_ndim == 1 and output.ranked_function is not None:
            return "# cython: language_level=3\n"
        lines = ["# cython: language_level=3"]
        for uid in sorted(self.v):
            cell = self.v[uid]
            for prefix in ("r", "calc"):
                sig = self._function_signature(cell, prefix).replace("cdef inline ", "cdef ", 1)
                lines.append(sig)
            memo = self.program.memo_plan.cell(uid)
            if memo.kind == "dict" or (memo.kind == "dense" and self.program.overflow_policy == "python_exact"):
                lines.append(self._overflow_signature(cell))
        return "\n".join(lines) + "\n"

    def write(self, path: str | Path, module_name: str = "dense_recursive_native") -> str:
        source = self.source(module_name)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(source, encoding="utf-8")
        out.with_suffix(".pxd").write_text(self.declaration_source(), encoding="utf-8")
        return source


__all__ = [
    "DenseRecursiveNativeError",
    "DenseMemoCell",
    "DenseMemoPlan",
    "DenseRecursiveProgram",
    "build_dense_memo_plan",
    "build_dense_recursive_program",
    "build_legacy_comparison_recursive_program",
    "DenseRecursiveCythonGenerator",
]
