from __future__ import annotations

"""Verification boundary for user-supplied numeric modelx views.

A numeric view is an ordinary modelx model/Space whose formulas may be rewritten
into the compiler's numeric subset.  The reference model remains authoritative.
This module deliberately does *not* rewrite formulas or interpret domain names;
it only fingerprints explicitly acknowledged reference formulas and compares
observable numeric outputs on caller-selected representative keys.
"""

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


class NumericViewError(RuntimeError):
    pass


def formula_fingerprint(cell) -> str:
    try:
        source = cell.formula.source
    except Exception as exc:
        raise NumericViewError(f"cannot read formula source for {getattr(cell, 'fullname', cell)!r}") from exc
    if not isinstance(source, str):
        raise NumericViewError(f"formula source is unavailable for {getattr(cell, 'fullname', cell)!r}")
    return hashlib.sha256(source.encode("utf-8", "backslashreplace")).hexdigest()


def formula_fingerprints(space, names: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        if name not in space.cells:
            raise NumericViewError(f"reference Space has no Cell {name!r}")
        result[name] = formula_fingerprint(space.cells[name])
    return result


@dataclass(frozen=True)
class NumericViewReport:
    ok: bool
    output: str
    key_count: int
    value_count: int
    exact: bool
    max_abs: float
    max_rel: float
    mismatch_count: int
    reference_checksum: float
    numeric_checksum: float
    hash_mismatches: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "key_count": self.key_count,
            "value_count": self.value_count,
            "exact": self.exact,
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "mismatch_count": self.mismatch_count,
            "reference_checksum": self.reference_checksum,
            "numeric_checksum": self.numeric_checksum,
            "hash_mismatches": list(self.hash_mismatches),
            "shape_mismatches": list(self.shape_mismatches),
        }


def _target(space, key: Any):
    return space if not getattr(space, "parameters", None) else space[key]


def _numeric_array(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf":
        raise NumericViewError(f"{label} returned non-numeric dtype {arr.dtype}")
    return arr


def verify_numeric_view(
    reference_space,
    numeric_space,
    *,
    output: str,
    keys: Sequence[Any],
    output_args: Sequence[Any] = (),
    expected_reference_hashes: Mapping[str, str] | None = None,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> NumericViewReport:
    """Compare a numeric view with the authoritative modelx Space.

    ``expected_reference_hashes`` is optional but recommended for every Cell
    whose semantics the view replaces.  A stale adapter therefore fails before
    numerical sampling rather than silently trusting a view written for older
    source formulas.
    """
    keys = tuple(keys)
    hash_mismatches: list[str] = []
    if expected_reference_hashes:
        for name, expected in expected_reference_hashes.items():
            if name not in reference_space.cells:
                hash_mismatches.append(f"{name}:missing")
                continue
            actual = formula_fingerprint(reference_space.cells[name])
            if actual != expected:
                hash_mismatches.append(f"{name}:{actual}")

    # A source-hash miss means the adapter was written for different reference
    # semantics.  Do not execute it merely to discover whether a few samples
    # happen to agree.
    if hash_mismatches:
        return NumericViewReport(
            ok=False, output=output, key_count=len(keys), value_count=0, exact=False,
            max_abs=float("inf"), max_rel=float("inf"), mismatch_count=0,
            reference_checksum=0.0, numeric_checksum=0.0,
            hash_mismatches=tuple(hash_mismatches), shape_mismatches=(),
        )

    ref_parts: list[np.ndarray] = []
    num_parts: list[np.ndarray] = []
    shape_mismatches: list[str] = []
    for pos, key in enumerate(keys):
        ref_obj = _target(reference_space, key)
        num_obj = _target(numeric_space, key)
        ref = _numeric_array(getattr(ref_obj, output)(*tuple(output_args)), f"reference[{key!r}].{output}")
        num = _numeric_array(getattr(num_obj, output)(*tuple(output_args)), f"numeric[{key!r}].{output}")
        if ref.shape != num.shape:
            shape_mismatches.append(f"key[{pos}]={key!r}: {ref.shape!r} != {num.shape!r}")
            continue
        ref_parts.append(ref.reshape(-1).astype(np.float64, copy=False))
        num_parts.append(num.reshape(-1).astype(np.float64, copy=False))

    if ref_parts:
        ref_all = np.concatenate(ref_parts)
        num_all = np.concatenate(num_parts)
        exact = bool(np.array_equal(ref_all, num_all, equal_nan=True))
        finite = np.isfinite(ref_all) & np.isfinite(num_all)
        if finite.any():
            diff = np.abs(ref_all[finite] - num_all[finite])
            max_abs = float(diff.max(initial=0.0))
            denom = np.maximum(np.abs(ref_all[finite]), np.finfo(np.float64).tiny)
            max_rel = float((diff / denom).max(initial=0.0))
        else:
            max_abs = max_rel = 0.0
        close = np.isclose(ref_all, num_all, rtol=rtol, atol=atol, equal_nan=True)
        mismatch_count = int((~close).sum())
        ref_sum = float(np.nansum(ref_all))
        num_sum = float(np.nansum(num_all))
        value_count = int(ref_all.size)
    else:
        exact = False
        max_abs = max_rel = float("inf")
        mismatch_count = 0
        ref_sum = num_sum = 0.0
        value_count = 0

    ok = not hash_mismatches and not shape_mismatches and mismatch_count == 0
    return NumericViewReport(
        ok=ok,
        output=output,
        key_count=len(keys),
        value_count=value_count,
        exact=exact,
        max_abs=max_abs,
        max_rel=max_rel,
        mismatch_count=mismatch_count,
        reference_checksum=ref_sum,
        numeric_checksum=num_sum,
        hash_mismatches=tuple(hash_mismatches),
        shape_mismatches=tuple(shape_mismatches),
    )


def require_numeric_view(*args, **kwargs) -> NumericViewReport:
    report = verify_numeric_view(*args, **kwargs)
    if not report.ok:
        raise NumericViewError(
            "numeric view verification failed: "
            f"hash_mismatches={report.hash_mismatches!r}, "
            f"shape_mismatches={report.shape_mismatches!r}, "
            f"mismatch_count={report.mismatch_count}, max_abs={report.max_abs:g}, max_rel={report.max_rel:g}"
        )
    return report
