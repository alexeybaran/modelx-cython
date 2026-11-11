from __future__ import annotations

"""Reusable evidence-sampling protocol for modelx-cython-fast corpus censuses.

This module deliberately owns *sampling policy*, not compiler semantics.  The
weighted feature sampler is the V02375 product-independent set-cover algorithm
that proved useful across lifelib.  A compiled artifact, when available, is
still validated over its complete RunDomain.  Numerical mismatches may buy
additional trace witnesses; structural compiler blockers may not.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


def _is_missing(value: Any) -> bool:
    try:
        return bool(value is None or (isinstance(value, float) and np.isnan(value)))
    except Exception:
        return value is None


def _norm(value: Any) -> str:
    if _is_missing(value):
        return "<NA>"
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return repr(value)


def policy_row_features(frame) -> tuple[frozenset[tuple[Any, ...]], ...]:
    """Return the V02375 semantic-feature set for every policy row."""

    n = len(frame)
    features: list[set[tuple[Any, ...]]] = [set() for _ in range(n)]
    low_cardinality: list[Any] = []
    for column in frame.columns:
        series = frame[column]
        try:
            unique = int(series.nunique(dropna=False))
        except Exception:
            continue
        lname = str(column).lower()
        identifier = any(
            token in lname for token in ("policy_id", "point_id", "spec_id", "provenance")
        )
        values = series.to_numpy()
        if unique <= 12 and not identifier:
            low_cardinality.append(column)
            counts = Counter(_norm(value) for value in values)
            for i, value in enumerate(values):
                normalized = _norm(value)
                features[i].add(("cat", str(column), normalized))
                features[i].add(("rare", str(column), normalized, int(counts[normalized])))

        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
            try:
                numeric = np.asarray(series, dtype=float)
            except Exception:
                continue
            finite = np.isfinite(numeric)
            if not finite.any():
                continue
            lo = float(np.nanmin(numeric))
            hi = float(np.nanmax(numeric))
            median = float(np.nanmedian(numeric))
            quantiles = np.unique(np.nanquantile(numeric[finite], [0, .1, .25, .5, .75, .9, 1]))
            for i, value in enumerate(numeric):
                if not np.isfinite(value):
                    features[i].add(("missing", str(column)))
                    continue
                if value == lo:
                    features[i].add(("extreme", str(column), "min"))
                if value == hi:
                    features[i].add(("extreme", str(column), "max"))
                if value == median:
                    features[i].add(("extreme", str(column), "median"))
                if len(quantiles) > 1:
                    bucket = min(
                        len(quantiles) - 2,
                        max(0, int(np.searchsorted(quantiles, value, side="right") - 1)),
                    )
                    features[i].add(("qbin", str(column), bucket))

    pair_columns = [
        column for column in low_cardinality
        if int(frame[column].nunique(dropna=False)) <= 6
    ]
    for ai, left in enumerate(pair_columns):
        for right in pair_columns[ai + 1:]:
            if (
                int(frame[left].nunique(dropna=False))
                * int(frame[right].nunique(dropna=False))
                > 24
            ):
                continue
            left_values = frame[left].to_numpy()
            right_values = frame[right].to_numpy()
            for i in range(n):
                features[i].add(
                    ("pair", str(left), _norm(left_values[i]), str(right), _norm(right_values[i]))
                )
    return tuple(frozenset(row) for row in features)


def smart_policy_keys(frame, keys: Sequence[Any], budget: int = 8):
    """Select a small feature-diverse trace set using V02375 weighted set cover.

    The historical first key remains the baseline for continuity with old census
    evidence.  This function is intentionally product-name agnostic.
    """

    keys = tuple(keys)
    budget = min(len(keys), max(1, int(budget)))
    features = policy_row_features(frame)
    index_by_key = {key: i for i, key in enumerate(keys)}
    frequency = Counter(token for row in features for token in row)
    selected: list[Any] = []
    covered: set[tuple[Any, ...]] = set()
    if keys:
        selected = [keys[0]]
        covered |= set(features[0])
    weights = {"pair": 7.0, "cat": 5.0, "rare": 5.0, "extreme": 4.0, "missing": 4.0, "qbin": 1.0}
    while len(selected) < budget:
        best = None
        best_score = -1.0
        for key in keys:
            if key in selected:
                continue
            row = set(features[index_by_key[key]])
            new = row - covered
            score = 0.0
            for token in new:
                score += weights.get(token[0], 1.0) / (max(1, frequency[token]) ** 0.35)
            score += 0.05 * len(row.symmetric_difference(covered))
            if score > best_score + 1e-12:
                best, best_score = key, score
        if best is None:
            break
        selected.append(best)
        covered |= set(features[index_by_key[best]])
    universe = set().union(*features) if features else set()
    return tuple(selected), {
        "feature_universe_count": len(universe),
        "feature_covered_count": len(covered),
        "coverage_fraction": len(covered) / len(universe) if universe else 1.0,
    }


def _spread_keys(keys: Sequence[Any], count: int, *, exclude: Iterable[Any] = ()) -> tuple[Any, ...]:
    excluded = set(exclude)
    pool = [key for key in keys if key not in excluded]
    n = min(max(0, int(count)), len(pool))
    if n <= 0:
        return ()
    if n >= len(pool):
        return tuple(pool)
    positions: list[int] = []
    seen: set[int] = set()
    for i in range(n):
        position = min(len(pool) - 1, int(round(i * (len(pool) - 1) / max(1, n - 1))))
        if position not in seen:
            positions.append(position)
            seen.add(position)
    for position in range(len(pool)):
        if len(positions) >= n:
            break
        if position not in seen:
            positions.append(position)
            seen.add(position)
    return tuple(pool[position] for position in positions[:n])


def mismatch_witness_keys(
    frame,
    full_keys: Sequence[Any],
    bad_keys: Sequence[Any],
    *,
    limit: int = 8,
) -> tuple[Any, ...]:
    """Select feature-diverse *actual failing* policies for retracing.

    This is the V02363/V02375 closure rule: numerical mismatches may expand the
    trace, while structural syntax/provider blockers stop instead of triggering
    blind sampling.
    """

    bad = tuple(dict.fromkeys(bad_keys))
    if len(bad) <= limit:
        return bad
    all_keys = tuple(full_keys)
    features = policy_row_features(frame)
    index_by_key = {key: i for i, key in enumerate(all_keys)}
    candidates = [key for key in all_keys if key in set(bad)]
    uncovered = set().union(*(features[index_by_key[key]] for key in candidates)) if candidates else set()
    weights = {"pair": 5.0, "cat": 4.0, "extreme": 4.0, "missing": 4.0, "qbin": 1.0}
    selected: list[Any] = []
    while uncovered and len(selected) < limit:
        best = None
        best_score = 0.0
        for key in candidates:
            if key in selected:
                continue
            score = sum(
                weights.get(token[0], 1.0)
                for token in features[index_by_key[key]]
                if token in uncovered
            )
            if score > best_score + 1e-12:
                best, best_score = key, score
        if best is None or best_score <= 0:
            break
        selected.append(best)
        uncovered -= set(features[index_by_key[best]])
    if len(selected) < limit:
        selected.extend(_spread_keys(bad, limit - len(selected), exclude=selected))
    return tuple(dict.fromkeys(selected))[:limit]


@dataclass(frozen=True)
class CensusProtocol:
    """Frozen reusable policy for future modelx-fast censuses."""

    normal_trace_budget: int = 8
    adaptive_trace_budgets: tuple[int, ...] = (8, 4, 2, 1)
    mismatch_witness_limit: int = 8
    full_domain_after_compile: bool = True
    structural_blockers_expand_trace: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "modelx_graph.census_protocol.v1",
            "sampler": "v02375_weighted_feature_set_cover",
            "normal_trace_budget": self.normal_trace_budget,
            "adaptive_trace_budgets": list(self.adaptive_trace_budgets),
            "mismatch_witness_limit": self.mismatch_witness_limit,
            "full_domain_after_compile": self.full_domain_after_compile,
            "structural_blockers_expand_trace": self.structural_blockers_expand_trace,
            "coverage_rule": "compiled_artifact_must_run_complete_RunDomain",
        }


def fast_plan_reachable_cells(plan) -> tuple[Any, ...]:
    """Reachable frontend rows including canonical cross-Space targets.

    ``fallback_dependency`` details are diagnostic names, while cross-Space
    identity is authoritative only through ``FastCrossSpaceCall.callee_uid``.
    This helper follows both without requiring a graph/execution schedule.
    """

    local_rows = tuple(plan.cells)
    cross_rows = tuple(getattr(plan, "cross_space_cells", ()) or ())
    rows = local_rows + cross_rows
    by_uid = {row.uid: row for row in rows}
    # Unqualified fallback_dependency names are lexical references in the caller
    # class.  Prefer local rows before considering canonical cross-Space rows;
    # otherwise a local proxy and its canonical target (both named disc_rate,
    # for example) make the diagnostic edge falsely ambiguous.
    local_by_name: dict[str, list[str]] = {}
    all_by_fullname: dict[str, list[str]] = {}
    all_by_name: dict[str, list[str]] = {}
    for row in local_rows:
        local_by_name.setdefault(row.source_name, []).append(row.uid)
    for row in rows:
        all_by_name.setdefault(row.source_name, []).append(row.uid)
        all_by_fullname.setdefault(row.source_fullname, []).append(row.uid)
    adjacency: dict[str, set[str]] = {uid: set() for uid in by_uid}
    for row in rows:
        for kind, detail in row.blocker_items:
            if kind != "fallback_dependency":
                continue
            for raw in str(detail).split(","):
                name = raw.strip()
                matches = local_by_name.get(name, ())
                if len(matches) != 1:
                    matches = all_by_fullname.get(name, ())
                if len(matches) != 1:
                    matches = all_by_name.get(name, ())
                if len(matches) == 1:
                    adjacency[row.uid].add(matches[0])
    for call in getattr(plan, "cross_space_calls", ()) or ():
        if call.caller_uid in adjacency and call.callee_uid in by_uid:
            adjacency[call.caller_uid].add(call.callee_uid)

    out: list[Any] = []
    seen: set[str] = set()
    stack = [plan.output_uid]
    while stack:
        uid = stack.pop()
        if uid in seen or uid not in by_uid:
            continue
        seen.add(uid)
        out.append(by_uid[uid])
        stack.extend(sorted(adjacency.get(uid, ()), reverse=True))
    return tuple(out)


def fast_plan_census_v3(plan) -> dict[str, Any]:
    """Execution-independent selected-closure + semantic-oracle census view."""

    rows = fast_plan_reachable_cells(plan)
    cross_targets = {call.callee_uid for call in getattr(plan, "cross_space_calls", ()) or ()}
    primitive: list[dict[str, str]] = []
    for row in rows:
        for kind, detail in row.blocker_items:
            if kind != "fallback_dependency":
                primitive.append({
                    "uid": row.uid,
                    "cell": row.source_fullname,
                    "kind": kind,
                    "detail": detail,
                })
    oracle = getattr(plan, "semantic_oracle", {}) or {}
    return {
        "schema": "modelx_graph.census_view.v3",
        "output_uid": plan.output_uid,
        "fast_admitted": bool(plan.fast_admitted),
        "reachable_uids": [row.uid for row in rows],
        "reachable_cells": [row.source_fullname for row in rows],
        "canonical_cross_space_target_uids": sorted(
            uid for uid in cross_targets if any(row.uid == uid for row in rows)
        ),
        "primitive_blockers": primitive,
        "semantic_oracle": {
            row.uid: oracle[row.uid]
            for row in rows
            if row.uid in oracle and oracle[row.uid].get("recognized")
        },
    }
