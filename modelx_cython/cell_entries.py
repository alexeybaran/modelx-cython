"""Shared layout and generated call contract for private Cell entries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import libcst as cst

from .consts import MX_SELF
from .scopes import CstFacts, Names


@dataclass(frozen=True)
class EntryModule:
    """One immutable syntax tree and its lazily derived entry facts.

    Each lowering returns a new snapshot, so facts cannot survive a rewrite
    accidentally. The pipeline keeps CST ownership and writes a module once.
    """
    syntax: cst.Module

    @cached_property
    def facts(self):
        return CstFacts(self.syntax)

    @cached_property
    def functions(self):
        return {node.name.value: node for node in self.syntax.body if isinstance(node, cst.FunctionDef)}

    @cached_property
    def methods(self):
        return {(cls.name.value, node.name.value): node for cls in self.syntax.body if isinstance(cls, cst.ClassDef)
                for node in cls.body.body if isinstance(node, cst.FunctionDef)}

    @cached_property
    def names(self):
        return Names(self.syntax)

    def replace(self, syntax):
        return self if syntax is self.syntax else EntryModule(syntax)


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
