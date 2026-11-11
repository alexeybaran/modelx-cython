from __future__ import annotations

"""Deprecated semantic-parameter comparison policy for fast recursion.

This module preserves the pre-native-hash local cache/direct classifier through
Stage C so old performance behavior can be reproduced. It consumes normalized
native formulas, modelx-cython/TransSpec parameter dtypes/ranges, and the direct
native call relation. The relation is diagnostic/comparison evidence only: it is
not an execution schedule and it no longer decides production native admission.

Production non-dense/continuous typed keys use native exact hash memoization.
"""



import ast
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .native_formula import NativeFormulaCell


@dataclass(frozen=True)
class NativeCallSemantics:
    calls: Mapping[str, frozenset[str]]
    recursive_uids: frozenset[str]
    reaches_recursive_uids: frozenset[str]

    def manifest(self) -> dict[str, Any]:
        return {
            "calls": {uid: sorted(values) for uid, values in sorted(self.calls.items())},
            "recursive_uids": sorted(self.recursive_uids),
            "reaches_recursive_uids": sorted(self.reaches_recursive_uids),
            "execution_schedule": False,
        }


@dataclass(frozen=True)
class SemanticParameterCellPolicy:
    uid: str
    mode: str  # scalar | dense | direct | blocked
    reason: str
    parameter_dtypes: tuple[str, ...]
    bounds: tuple[tuple[int, int] | None, ...]
    recursive: bool
    reaches_recursive: bool
    dense_size: int | None = None

    @property
    def supported_fast(self) -> bool:
        return self.mode != "blocked"

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "mode": self.mode,
            "reason": self.reason,
            "parameter_dtypes": list(self.parameter_dtypes),
            "bounds": [None if item is None else list(item) for item in self.bounds],
            "recursive": self.recursive,
            "reaches_recursive": self.reaches_recursive,
            "dense_size": self.dense_size,
        }


@dataclass(frozen=True)
class SemanticParameterPolicy:
    schema: str
    cells: tuple[SemanticParameterCellPolicy, ...]
    call_semantics: NativeCallSemantics

    def cell(self, uid: str) -> SemanticParameterCellPolicy:
        for row in self.cells:
            if row.uid == uid:
                return row
        raise KeyError(uid)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cells": [row.manifest() for row in self.cells],
            "call_semantics": self.call_semantics.manifest(),
            "float_grid_approximation": False,
            "python_hash_memo_fast_profile": False,
            "stage_or_scc_scheduler": False,
        }


def _cell_call_uids(cell: NativeFormulaCell, known: set[str]) -> frozenset[str]:
    roots: list[ast.AST] = []
    if cell.function is not None:
        roots.append(cell.function)
    if cell.ranked_function is not None:
        roots.append(cell.ranked_function)
    if cell.reduction is not None:
        r = cell.reduction
        roots.extend([r.init, *r.range_args, r.body_expr, *r.filters])
    out: set[str] = set()
    for root in roots:
        for node in ast.walk(root):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in known
            ):
                out.add(node.func.id)
    return frozenset(out)


def native_call_semantics(cells: Mapping[str, NativeFormulaCell]) -> NativeCallSemantics:
    active = {uid: cell for uid, cell in cells.items() if cell.execution_role == "formula"}
    known = set(active)
    graph = {uid: _cell_call_uids(cell, known) for uid, cell in active.items()}

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

    reaches_cache: dict[str, bool] = {}
    visiting: set[str] = set()

    def reaches(uid: str) -> bool:
        if uid in reaches_cache:
            return reaches_cache[uid]
        if uid in recursive:
            reaches_cache[uid] = True
            return True
        if uid in visiting:
            reaches_cache[uid] = True
            return True
        visiting.add(uid)
        result = any(reaches(dep) for dep in graph[uid])
        visiting.remove(uid)
        reaches_cache[uid] = result
        return result

    for uid in sorted(active):
        reaches(uid)
    return NativeCallSemantics(
        calls=graph,
        recursive_uids=frozenset(recursive),
        reaches_recursive_uids=frozenset(uid for uid, yes in reaches_cache.items() if yes),
    )


def _dense_size(bounds: Sequence[tuple[int, int]]) -> int:
    size = 1
    for lo, hi in bounds:
        if lo > hi:
            return -1
        size *= hi - lo + 1
    return size


def classify_semantic_parameter_policy(
    cells: Mapping[str, NativeFormulaCell],
    *,
    parameter_ranges: Mapping[str, Sequence[tuple[int, int] | None]] | None = None,
    max_dense_slots_per_cell: int = 1_000_000,
) -> SemanticParameterPolicy:
    ranges = parameter_ranges or {}
    calls = native_call_semantics(cells)
    rows: list[SemanticParameterCellPolicy] = []
    for uid in sorted(cells):
        cell = cells[uid]
        if cell.execution_role != "formula":
            continue
        requested = tuple(ranges.get(uid, ()))
        if len(requested) != cell.arity:
            requested = tuple(None for _ in range(cell.arity))
        recursive = uid in calls.recursive_uids
        reaches_recursive = uid in calls.reaches_recursive_uids

        if cell.arity == 0:
            rows.append(SemanticParameterCellPolicy(
                uid, "scalar", "zero_arity_generation_tag_memo", (), (),
                recursive, reaches_recursive, 1,
            ))
            continue

        unsupported = [dtype for dtype in cell.parameter_dtypes if dtype not in {"bool", "int64", "float64"}]
        if unsupported:
            rows.append(SemanticParameterCellPolicy(
                uid, "blocked", "unsupported_parameter_dtype", tuple(cell.parameter_dtypes),
                requested, recursive, reaches_recursive, None,
            ))
            continue

        has_continuous = any(dtype == "float64" for dtype in cell.parameter_dtypes)
        if has_continuous:
            # No float grid and no hash memo in the fast profile. Direct evaluation
            # is safe when this Cell itself is not in a recursive native SCC.
            # Descendants may still be recursive: they retain their own bounded
            # memo policy, so direct-calling this acyclic wrapper does not erase
            # descendant memoization. NativeFormulaCell already denotes normalized
            # read-only numeric code over semantic parameters/prepared inputs.
            if recursive:
                rows.append(SemanticParameterCellPolicy(
                    uid, "blocked", "continuous_parameter_is_recursive_native_cycle",
                    tuple(cell.parameter_dtypes), requested, recursive, reaches_recursive, None,
                ))
            else:
                rows.append(SemanticParameterCellPolicy(
                    uid, "direct", "continuous_parameter_cycle_free_pure_native_call",
                    tuple(cell.parameter_dtypes), requested, recursive, reaches_recursive, None,
                ))
            continue

        dense_bounds: list[tuple[int, int]] = []
        proven = True
        for dtype, item in zip(cell.parameter_dtypes, requested):
            if item is None and dtype == "bool":
                item = (0, 1)
            if item is None:
                proven = False
                break
            lo, hi = int(item[0]), int(item[1])
            if lo > hi:
                proven = False
                break
            dense_bounds.append((lo, hi))
        if not proven or len(dense_bounds) != cell.arity:
            rows.append(SemanticParameterCellPolicy(
                uid, "blocked", "bounded_integral_domain_unproven",
                tuple(cell.parameter_dtypes), requested, recursive, reaches_recursive, None,
            ))
            continue
        size = _dense_size(dense_bounds)
        if size < 0 or size > max_dense_slots_per_cell:
            rows.append(SemanticParameterCellPolicy(
                uid, "blocked", "dense_semantic_parameter_slot_limit",
                tuple(cell.parameter_dtypes), tuple(dense_bounds), recursive, reaches_recursive, size,
            ))
            continue
        rows.append(SemanticParameterCellPolicy(
            uid, "dense", "bounded_integral_or_category_semantic_tuple",
            tuple(cell.parameter_dtypes), tuple(dense_bounds), recursive, reaches_recursive, size,
        ))

    return SemanticParameterPolicy(
        schema="modelx_graph.semantic_parameter_policy.v1",
        cells=tuple(rows),
        call_semantics=calls,
    )


__all__ = [
    "NativeCallSemantics",
    "SemanticParameterCellPolicy",
    "SemanticParameterPolicy",
    "native_call_semantics",
    "classify_semantic_parameter_policy",
]
