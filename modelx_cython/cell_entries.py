"""Shared layout and generated call contract for private Cell entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .consts import MX_SELF


class DenseAxis(NamedTuple):
    key: str
    lower: int
    stop: int
    c_type: str


@dataclass(frozen=True)
class DenseEntry:
    owner: str
    formula: str
    axes: tuple[DenseAxis, ...]
    exact_spill: bool
    logical_axes: tuple[DenseAxis, ...] = ()
    spill_entry: str | None = None


@dataclass(frozen=True)
class FormulaEntry:
    """A proven private numeric call target, independent of cache geometry."""
    owner: str
    formula: str
    keys: tuple[str, ...]
    guard_safe: bool


def dense_condition(axes, proof=None, proved_axes=()):
    checks = [f"({axis.lower} <= {axis.key} < {axis.stop})" for axis in axes]
    if proof is not None:
        checks = [f"(({proof} & {1 << i}) or {check})"
                  if not proved_axes or axes[i] == proved_axes[i] else check
                  for i, check in enumerate(checks)]
    return " and ".join(checks)


def exact_receiver(owner):
    return f"type({MX_SELF}) is {owner}"
