"""Shared layout and generated call contract for private Cell entries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import NamedTuple

import libcst as cst

from .consts import MX_SELF
from .scopes import CstFacts, Names
from .syntax import same_expression


@dataclass(frozen=True)
class EntryModule:
    """One immutable syntax tree and its lazily derived entry facts.

    Each lowering returns a new snapshot, so facts cannot survive a rewrite
    accidentally. The pipeline keeps CST ownership and writes a module once.
    """
    syntax: cst.Module
    function_origins: dict[str, tuple[str, str]] = field(default_factory=dict)
    inbounds_entries: dict[str, str] = field(default_factory=dict)

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

    def replace(self, syntax, function_origins=None):
        if function_origins is not None:
            return replace(self, syntax=syntax, function_origins=function_origins)
        return self if syntax is self.syntax else replace(self, syntax=syntax)


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
    checks_spill_in_helper: bool = False


@dataclass(frozen=True)
class FormulaEntry:
    """A proven private numeric call target, independent of cache geometry."""
    owner: str
    formula: str
    keys: tuple[str, ...]
    guard_safe: bool
    scalar_view: bool


def dense_condition(axes, proof=None, proved_axes=()):
    checks = [f"({axis.lower} <= {axis.key} < {axis.stop})" for axis in axes]
    if proof is not None:
        checks = [f"(({proof} & {1 << i}) or {check})"
                  if not proved_axes or axes[i] == proved_axes[i] else check
                  for i, check in enumerate(checks)]
    return " and ".join(checks)


def checked_dense_guard(body, layout, proof=None):
    """Find the sole guard for the generated dense layout."""
    if len(body) != 1 or not isinstance(guard := body[0], cst.If):
        return None
    expected = cst.parse_expression(dense_condition(layout.axes, proof))
    return guard if same_expression(guard.test, expected) else None


def exact_receiver(owner):
    return f"type({MX_SELF}) is {owner}"
