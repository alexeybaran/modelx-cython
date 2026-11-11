from __future__ import annotations

"""Bounded path-sensitive finite-domain evaluation for category control flow.

This module deliberately implements a small abstract interpreter rather than Python.
It reasons only about exact finite ``str``/``None`` category values and boolean
control derived from them.  Everything else is opaque.  Opaque *non-category*
control is represented as UNKNOWN so callers can conservatively explore both
branches; an unsupported operation that actually consumes a tracked category is
reported separately and remains fail-closed.

The evaluator is intended to be shared by category demand/completion analyses.  It
has no knowledge of products, modelx Cells, category codes, or physical ABIs.
"""

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, MutableMapping, Sequence


class TruthValue(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FiniteValue:
    """An exact bounded set of semantic category labels."""

    values: frozenset[Any]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("finite category value set must be non-empty")
        if any(value is not None and not isinstance(value, str) for value in self.values):
            raise ValueError("finite category values must be str/None")

    @classmethod
    def one(cls, value: Any) -> "FiniteValue":
        return cls(frozenset((value,)))


@dataclass(frozen=True)
class PredicateResult:
    truth: TruthValue
    category_related: bool = False
    unsupported_category: bool = False
    reason: str = ""

    @property
    def exact_bool(self) -> bool | None:
        if self.truth == TruthValue.TRUE:
            return True
        if self.truth == TruthValue.FALSE:
            return False
        return None


Environment = Mapping[str, FiniteValue]
MutableEnvironment = MutableMapping[str, FiniteValue]


def _category_names(node: ast.AST | None, env: Environment) -> set[str]:
    if node is None:
        return set()
    return {
        child.id for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id in env
    }


def _literal_collection(node: ast.AST) -> FiniteValue | None:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    values: list[Any] = []
    for elt in node.elts:
        if not isinstance(elt, ast.Constant):
            return None
        value = elt.value
        if value is not None and not isinstance(value, str):
            return None
        values.append(value)
    if not values:
        return None
    return FiniteValue(frozenset(values))


def finite_value(node: ast.AST | None, env: Environment, *, max_values: int = 16) -> FiniteValue | None:
    """Evaluate an exact finite category expression when possible.

    Supported value forms are category constants/names and conditional expressions.
    An unknown condition unions the exact finite values from both arms.  The result
    remains exact because both arms are explicit complete alternatives.
    """

    if node is None:
        return None
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, str):
            return FiniteValue.one(value)
        return None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.IfExp):
        test = evaluate_predicate(node.test, env)
        if test.unsupported_category:
            return None
        if test.truth == TruthValue.TRUE:
            return finite_value(node.body, env, max_values=max_values)
        if test.truth == TruthValue.FALSE:
            return finite_value(node.orelse, env, max_values=max_values)
        left = finite_value(node.body, env, max_values=max_values)
        right = finite_value(node.orelse, env, max_values=max_values)
        if left is None or right is None:
            return None
        values = left.values | right.values
        if len(values) > max_values:
            return None
        return FiniteValue(frozenset(values))
    return None


def _relation_from_pairs(left: FiniteValue, right: FiniteValue, *, equal: bool) -> TruthValue:
    outcomes = {
        ((a == b) if equal else (a != b))
        for a in left.values for b in right.values
    }
    if outcomes == {True}:
        return TruthValue.TRUE
    if outcomes == {False}:
        return TruthValue.FALSE
    return TruthValue.UNKNOWN


def _negate(value: TruthValue) -> TruthValue:
    if value == TruthValue.TRUE:
        return TruthValue.FALSE
    if value == TruthValue.FALSE:
        return TruthValue.TRUE
    return TruthValue.UNKNOWN


def evaluate_predicate(node: ast.AST, env: Environment) -> PredicateResult:
    """Evaluate a bounded category predicate with three-valued truth semantics.

    Unknown unrelated control is safe evidence for branch exploration.  Unknown
    control caused by an unsupported operation *on a tracked category* is marked
    ``unsupported_category=True`` so admission remains fail-closed.
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return PredicateResult(TruthValue.TRUE if node.value else TruthValue.FALSE)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = evaluate_predicate(node.operand, env)
        return PredicateResult(
            _negate(inner.truth), inner.category_related,
            inner.unsupported_category, inner.reason,
        )

    if isinstance(node, ast.BoolOp):
        related = False
        unknown = False
        # Python short-circuit order matters.  Unsupported category operations are
        # errors only if their operand is actually evaluated on this abstract path.
        if isinstance(node.op, ast.And):
            for value in node.values:
                part = evaluate_predicate(value, env)
                related = related or part.category_related
                if part.unsupported_category:
                    return PredicateResult(
                        TruthValue.UNKNOWN, related, True,
                        part.reason or "unsupported_category_boolean_operand",
                    )
                if part.truth == TruthValue.FALSE:
                    return PredicateResult(TruthValue.FALSE, related)
                if part.truth == TruthValue.UNKNOWN:
                    unknown = True
            return PredicateResult(TruthValue.UNKNOWN if unknown else TruthValue.TRUE, related)
        if isinstance(node.op, ast.Or):
            for value in node.values:
                part = evaluate_predicate(value, env)
                related = related or part.category_related
                if part.unsupported_category:
                    return PredicateResult(
                        TruthValue.UNKNOWN, related, True,
                        part.reason or "unsupported_category_boolean_operand",
                    )
                if part.truth == TruthValue.TRUE:
                    return PredicateResult(TruthValue.TRUE, related)
                if part.truth == TruthValue.UNKNOWN:
                    unknown = True
            return PredicateResult(TruthValue.UNKNOWN if unknown else TruthValue.FALSE, related)

    if isinstance(node, ast.IfExp):
        test = evaluate_predicate(node.test, env)
        if test.unsupported_category:
            return test
        if test.truth == TruthValue.TRUE:
            branch = evaluate_predicate(node.body, env)
            return PredicateResult(
                branch.truth,
                test.category_related or branch.category_related,
                branch.unsupported_category,
                branch.reason,
            )
        if test.truth == TruthValue.FALSE:
            branch = evaluate_predicate(node.orelse, env)
            return PredicateResult(
                branch.truth,
                test.category_related or branch.category_related,
                branch.unsupported_category,
                branch.reason,
            )
        left = evaluate_predicate(node.body, env)
        right = evaluate_predicate(node.orelse, env)
        related = test.category_related or left.category_related or right.category_related
        if left.unsupported_category or right.unsupported_category:
            return PredicateResult(
                TruthValue.UNKNOWN, related, True,
                left.reason or right.reason or "unsupported_category_conditional",
            )
        if left.truth == right.truth:
            return PredicateResult(left.truth, related)
        return PredicateResult(TruthValue.UNKNOWN, related)

    if isinstance(node, ast.Compare):
        related_names = _category_names(node, env)
        related = bool(related_names)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return PredicateResult(
                TruthValue.UNKNOWN, related, related,
                "unsupported_category_comparison_chain" if related else "",
            )
        left_node, right_node = node.left, node.comparators[0]
        op = node.ops[0]
        left = finite_value(left_node, env)
        right = finite_value(right_node, env)
        if isinstance(op, (ast.Eq, ast.Is, ast.NotEq, ast.IsNot)):
            if left is not None and right is not None:
                truth = _relation_from_pairs(
                    left, right, equal=isinstance(op, (ast.Eq, ast.Is))
                )
                return PredicateResult(truth, related)
            return PredicateResult(
                TruthValue.UNKNOWN, related, related,
                "unsupported_category_comparison" if related else "",
            )
        if isinstance(op, (ast.In, ast.NotIn)):
            collection = _literal_collection(right_node)
            if left is not None and collection is not None:
                outcomes = {value in collection.values for value in left.values}
                if isinstance(op, ast.NotIn):
                    outcomes = {not value for value in outcomes}
                if outcomes == {True}:
                    truth = TruthValue.TRUE
                elif outcomes == {False}:
                    truth = TruthValue.FALSE
                else:
                    truth = TruthValue.UNKNOWN
                return PredicateResult(truth, related)
            return PredicateResult(
                TruthValue.UNKNOWN, related, related,
                "unsupported_category_membership" if related else "",
            )
        return PredicateResult(
            TruthValue.UNKNOWN, related, related,
            "unsupported_category_comparison_operator" if related else "",
        )

    related = bool(_category_names(node, env))
    return PredicateResult(
        TruthValue.UNKNOWN,
        category_related=related,
        unsupported_category=related,
        reason="unsupported_category_predicate" if related else "",
    )


def assign_finite_value(
    env: MutableEnvironment, name: str, expr: ast.AST | None, *, max_values: int = 16,
) -> None:
    """Update one local binding while preserving exact finite category information."""

    value = finite_value(expr, env, max_values=max_values)
    if value is None:
        env.pop(name, None)
    else:
        env[name] = value


def environment_for_label(
    names: Sequence[str], label: Any, *, other_labels: Mapping[str, Any] | None = None,
) -> dict[str, FiniteValue]:
    env = {name: FiniteValue.one(label) for name in names}
    for name, value in (other_labels or {}).items():
        env[name] = FiniteValue.one(value)
    return env


__all__ = [
    "FiniteValue",
    "PredicateResult",
    "TruthValue",
    "assign_finite_value",
    "environment_for_label",
    "evaluate_predicate",
    "finite_value",
]
