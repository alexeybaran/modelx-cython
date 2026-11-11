from __future__ import annotations

"""Typed coordinate relations used by the executable NativeBatch scheduler.

The scheduler distinguishes an ordinary affine coordinate dependency from a
history/snapshot read that has been separately proven causal.  Keeping the proof
inside the relation prevents codegen/storage from interpreting ``offset=None`` as
an implicit permission to index arbitrary state history.
"""

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CausalHistoryProof:
    proof_kind: str
    relation: str
    period: int
    range_start: int
    coordinate_step: int
    normalized_expr: str
    note: str
    min_lag: int | None = None
    max_lag: int | None = None
    caller_lo: int | None = None
    caller_hi: int | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "proof_kind": self.proof_kind,
            "relation": self.relation,
            "period": int(self.period),
            "range_start": int(self.range_start),
            "coordinate_step": int(self.coordinate_step),
            "normalized_expr": self.normalized_expr,
            "note": self.note,
            "min_lag": None if self.min_lag is None else int(self.min_lag),
            "max_lag": None if self.max_lag is None else int(self.max_lag),
            "caller_lo": None if self.caller_lo is None else int(self.caller_lo),
            "caller_hi": None if self.caller_hi is None else int(self.caller_hi),
        }


@dataclass(frozen=True)
class AffineOffset:
    offset: int

    @property
    def kind(self) -> str:
        return "affine_offset"

    def manifest(self) -> dict[str, Any]:
        return {"kind": self.kind, "offset": int(self.offset)}


@dataclass(frozen=True)
class AffineAliasRead:
    """A source expression proven equivalent to an ordinary affine coordinate.

    The original expression is retained so code generation can authorize a call
    such as ``state(period_alias(t))`` after proof-only helper/local expansion has
    reduced it to ``t`` or ``t-1``.  Generated formula source stays canonical.
    """

    expr: ast.AST
    offset: int
    normalized_expr: str

    @property
    def kind(self) -> str:
        return "affine_alias_read"

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "expr": ast.unparse(self.expr),
            "offset": int(self.offset),
            "normalized_expr": self.normalized_expr,
        }


@dataclass(frozen=True)
class DerivedHistoryRead:
    expr: ast.AST
    proof: CausalHistoryProof | None = None

    @property
    def kind(self) -> str:
        return "derived_history_read"

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "expr": ast.unparse(self.expr),
            "proof": None if self.proof is None else self.proof.manifest(),
        }


@dataclass(frozen=True)
class DerivedClockProof:
    """Proof that a secondary integer clock is a monotone bucket of the primary clock.

    The first implementation is deliberately tiny: one bucket mapping advances by
    zero or one for every physical primary-coordinate step, starts at bucket one,
    and is consumed only as the current bucket.  This is enough for annual state
    carried by a monthly loop without pretending that an independent second
    scheduler dimension exists.
    """

    proof_kind: str
    period: int
    shift: int
    bias: int
    range_start: int
    coordinate_step: int
    initial_bucket: int
    normalized_expr: str
    note: str

    def manifest(self) -> dict[str, Any]:
        return {
            "proof_kind": self.proof_kind,
            "period": int(self.period),
            "shift": int(self.shift),
            "bias": int(self.bias),
            "range_start": int(self.range_start),
            "coordinate_step": int(self.coordinate_step),
            "initial_bucket": int(self.initial_bucket),
            "normalized_expr": self.normalized_expr,
            "note": self.note,
        }


@dataclass(frozen=True)
class DerivedClockRead:
    expr: ast.AST
    proof: DerivedClockProof

    @property
    def kind(self) -> str:
        return "derived_clock_read"

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "expr": ast.unparse(self.expr),
            "proof": self.proof.manifest(),
        }


@dataclass(frozen=True)
class TransitionLagProof:
    """Proof for a primary-state read performed only on derived-clock transitions.

    If a derived clock is ``y = floor((t + shift) / period) + bias``, then the
    physical coordinate at which bucket ``y`` first becomes current is

        ``t_transition = period * (y - bias) - shift``.

    A derived-region formula may therefore read a primary coordinate expression
    ``x(y)`` without creating a same-period cycle when the graph analyzer proves
    ``x(y) = t_transition + primary_offset`` with a strictly negative offset.

    The first executable target is deliberately narrower still: exact one-step
    lag.  Keeping the more explicit algebraic proof in the IR lets later codegen
    consume evidence rather than rediscovering source syntax.
    """

    proof_kind: str
    period: int
    shift: int
    bias: int
    source_slope: int
    source_intercept: int
    primary_offset: int
    normalized_expr: str
    note: str

    def manifest(self) -> dict[str, Any]:
        return {
            "proof_kind": self.proof_kind,
            "period": int(self.period),
            "shift": int(self.shift),
            "bias": int(self.bias),
            "source_slope": int(self.source_slope),
            "source_intercept": int(self.source_intercept),
            "primary_offset": int(self.primary_offset),
            "normalized_expr": self.normalized_expr,
            "note": self.note,
        }


CoordinateRelation = AffineOffset | AffineAliasRead | DerivedHistoryRead | DerivedClockRead
