from __future__ import annotations

"""Fail-closed signed-int64 interval proof for source-local control.

The proof is deliberately independent of execution scheduling and products.  It
accepts only expressions whose complete value interval is supplied by source,
prepared-input, parameter, or native Cell-result authority.  Arithmetic that
could leave signed int64 is rejected instead of relying on C overflow behavior.
"""

import ast
from dataclasses import dataclass
from typing import Iterable, Mapping


INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class Int64Interval:
    lower: int
    upper: int
    authorities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.lower) is not int or type(self.upper) is not int:
            raise TypeError("int64 interval endpoints must be Python ints")
        if self.lower < INT64_MIN or self.upper > INT64_MAX or self.lower > self.upper:
            raise ValueError("invalid signed-int64 interval")

    @classmethod
    def exact(cls, value: int, authority: str) -> "Int64Interval":
        if type(value) is not int:
            raise TypeError("exact int64 interval excludes bool and non-int values")
        return cls(value, value, (authority,))

    @classmethod
    def typed(cls, authority: str) -> "Int64Interval":
        return cls(INT64_MIN, INT64_MAX, (authority,))

    def manifest(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "authorities": list(self.authorities),
        }


@dataclass(frozen=True)
class Int64RangeProof:
    arg_intervals: tuple[Int64Interval, ...]
    maximum_trip_count: int
    step_sign: str

    def manifest(self) -> dict[str, object]:
        return {
            "arg_intervals": [row.manifest() for row in self.arg_intervals],
            "maximum_trip_count": self.maximum_trip_count,
            "step_sign": self.step_sign,
            "authority": "complete_int64_interval_composition",
        }


def _authorities(kind: str, *domains: Int64Interval) -> tuple[str, ...]:
    return tuple(sorted({kind, *(x for d in domains for x in d.authorities)}))


def _checked_interval(lower: int, upper: int, kind: str, *parts: Int64Interval) -> Int64Interval | None:
    if lower < INT64_MIN or upper > INT64_MAX or lower > upper:
        return None
    return Int64Interval(lower, upper, _authorities(kind, *parts))


def _literal(node: ast.AST) -> int | None:
    sign = 1
    value_node = node
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        value_node = node.operand
    if not (isinstance(value_node, ast.Constant) and type(value_node.value) is int):
        return None
    value = sign * int(value_node.value)
    return value if INT64_MIN <= value <= INT64_MAX else None


def _binary_interval(
    left: Int64Interval, right: Int64Interval, op: ast.operator
) -> Int64Interval | None:
    if isinstance(op, ast.Add):
        return _checked_interval(
            left.lower + right.lower, left.upper + right.upper,
            "safe_add", left, right,
        )
    if isinstance(op, ast.Sub):
        return _checked_interval(
            left.lower - right.upper, left.upper - right.lower,
            "safe_sub", left, right,
        )
    if isinstance(op, ast.Mult):
        values = (
            left.lower * right.lower, left.lower * right.upper,
            left.upper * right.lower, left.upper * right.upper,
        )
        return _checked_interval(min(values), max(values), "safe_mul", left, right)
    if isinstance(op, ast.FloorDiv):
        if right.lower <= 0 <= right.upper:
            return None
        values = (
            left.lower // right.lower, left.lower // right.upper,
            left.upper // right.lower, left.upper // right.upper,
        )
        return _checked_interval(min(values), max(values), "safe_floor_div", left, right)
    if isinstance(op, ast.Mod):
        # A fixed non-zero divisor has a simple exact Python remainder enclosure.
        if right.lower != right.upper or right.lower == 0:
            return None
        divisor = right.lower
        lo, hi = (0, divisor - 1) if divisor > 0 else (divisor + 1, 0)
        return _checked_interval(lo, hi, "safe_fixed_mod", left, right)
    return None


def prove_int64_expression(
    node: ast.AST,
    *,
    name_intervals: Mapping[str, Int64Interval] | None = None,
    call_intervals: Mapping[str, Int64Interval] | None = None,
    input_intervals: Mapping[str, Int64Interval] | None = None,
) -> Int64Interval | None:
    """Return a complete signed-int64 interval, or ``None`` if unproven."""

    names = name_intervals or {}
    calls = call_intervals or {}
    inputs = input_intervals or {}
    literal = _literal(node)
    if literal is not None:
        return Int64Interval.exact(literal, "source_literal")
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        child = prove_int64_expression(
            node.operand, name_intervals=names, call_intervals=calls,
            input_intervals=inputs,
        )
        if child is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return Int64Interval(child.lower, child.upper, _authorities("safe_uplus", child))
        # Negating INT64_MIN cannot be represented by the native ABI.
        return _checked_interval(-child.upper, -child.lower, "safe_negate", child)
    if isinstance(node, ast.BinOp):
        left = prove_int64_expression(
            node.left, name_intervals=names, call_intervals=calls,
            input_intervals=inputs,
        )
        right = prove_int64_expression(
            node.right, name_intervals=names, call_intervals=calls,
            input_intervals=inputs,
        )
        if left is None or right is None:
            return None
        return _binary_interval(left, right, node.op)
    if isinstance(node, ast.IfExp):
        body = prove_int64_expression(
            node.body, name_intervals=names, call_intervals=calls,
            input_intervals=inputs,
        )
        other = prove_int64_expression(
            node.orelse, name_intervals=names, call_intervals=calls,
            input_intervals=inputs,
        )
        if body is None or other is None:
            return None
        return Int64Interval(
            min(body.lower, other.lower), max(body.upper, other.upper),
            _authorities("safe_conditional_union", body, other),
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in calls and not node.keywords and all(
            prove_int64_expression(
                arg, name_intervals=names, call_intervals=calls,
                input_intervals=inputs,
            ) is not None
            for arg in node.args
        ):
            # The mapping itself is the authoritative native result contract.
            return calls[name]
        if (
            name in {"point_input", "global_input"}
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return inputs.get(str(node.args[0].value))
        if name in {"min", "max"} and len(node.args) >= 2 and not node.keywords:
            domains = tuple(
                prove_int64_expression(
                    arg, name_intervals=names, call_intervals=calls,
                    input_intervals=inputs,
                )
                for arg in node.args
            )
            if any(domain is None for domain in domains):
                return None
            proven = tuple(domain for domain in domains if domain is not None)
            if name == "min":
                lower = min(x.lower for x in proven)
                upper = min(x.upper for x in proven)
            else:
                lower = max(x.lower for x in proven)
                upper = max(x.upper for x in proven)
            return Int64Interval(lower, upper, _authorities(f"safe_{name}", *proven))
    return None


def derive_local_int64_intervals(
    fn: ast.FunctionDef,
    *,
    parameter_intervals: Mapping[str, Int64Interval] | None = None,
    call_intervals: Mapping[str, Int64Interval] | None = None,
    input_intervals: Mapping[str, Int64Interval] | None = None,
) -> dict[str, Int64Interval]:
    """Propagate definitely assigned local intervals in source order.

    Branch outputs are intersected by name and unioned by interval.  Loops and
    augmented/multi-target assignments invalidate affected locals; the proof
    never treats Cython's zero initialization as Python definite assignment.
    """

    initial = dict(parameter_intervals or {})

    def scan(statements: Iterable[ast.stmt], incoming: dict[str, Int64Interval]) -> dict[str, Int64Interval]:
        current = dict(incoming)
        for stmt in statements:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                name = stmt.targets[0].id
                domain = prove_int64_expression(
                    stmt.value, name_intervals=current, call_intervals=call_intervals,
                    input_intervals=input_intervals,
                )
                if domain is None:
                    current.pop(name, None)
                else:
                    current[name] = domain
                continue
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                domain = None if stmt.value is None else prove_int64_expression(
                    stmt.value, name_intervals=current, call_intervals=call_intervals,
                    input_intervals=input_intervals,
                )
                if domain is None:
                    current.pop(name, None)
                else:
                    current[name] = domain
                continue
            if isinstance(stmt, ast.If):
                branch_input = dict(current)
                for child in ast.walk(stmt.test):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                        branch_input.pop(child.id, None)
                body = scan(stmt.body, branch_input)
                other = scan(stmt.orelse, branch_input) if stmt.orelse else dict(branch_input)
                joined: dict[str, Int64Interval] = {}
                for name in body.keys() & other.keys():
                    left, right = body[name], other[name]
                    joined[name] = Int64Interval(
                        min(left.lower, right.lower), max(left.upper, right.upper),
                        _authorities("safe_branch_union", left, right),
                    )
                current = joined
                continue
            if isinstance(stmt, (ast.For, ast.While)):
                assigned = {
                    child.id
                    for child in ast.walk(stmt)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))
                }
                for name in assigned:
                    current.pop(name, None)
                continue
            if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                current.pop(stmt.target.id, None)
                continue
            # Unknown compound statements must not preserve stale interval
            # facts for names they may assign on only some control-flow paths.
            assigned = {
                child.id
                for child in ast.walk(stmt)
                if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))
            }
            for name in assigned:
                current.pop(name, None)
        return current

    final = scan(fn.body, initial)
    return {name: domain for name, domain in final.items() if name not in initial}


def prove_int64_range(
    call: ast.Call,
    *,
    name_intervals: Mapping[str, Int64Interval] | None = None,
    call_intervals: Mapping[str, Int64Interval] | None = None,
    input_intervals: Mapping[str, Int64Interval] | None = None,
) -> Int64RangeProof | None:
    """Prove safe range argument evaluation and a finite int64 trip bound."""

    if not (
        isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and not call.keywords
        and 1 <= len(call.args) <= 3
    ):
        return None
    args = tuple(
        prove_int64_expression(
            arg, name_intervals=name_intervals, call_intervals=call_intervals,
            input_intervals=input_intervals,
        )
        for arg in call.args
    )
    if any(domain is None for domain in args):
        return None
    proven = tuple(domain for domain in args if domain is not None)
    if len(proven) == 1:
        start = Int64Interval.exact(0, "implicit_range_start")
        stop = proven[0]
        step = Int64Interval.exact(1, "implicit_range_step")
    elif len(proven) == 2:
        start, stop = proven
        step = Int64Interval.exact(1, "implicit_range_step")
    else:
        start, stop, step = proven

    if step.lower > 0:
        distance = max(0, stop.upper - start.lower)
        maximum = 0 if distance == 0 else (distance + step.lower - 1) // step.lower
        sign = "positive"
    elif step.upper < 0:
        magnitude = -step.upper
        distance = max(0, start.upper - stop.lower)
        maximum = 0 if distance == 0 else (distance + magnitude - 1) // magnitude
        sign = "negative"
    else:
        return None
    if maximum > INT64_MAX:
        return None
    return Int64RangeProof(proven, int(maximum), sign)


def intervals_from_rows(
    rows: Iterable[tuple[str, int, int, str]]
) -> dict[str, Int64Interval]:
    return {
        name: Int64Interval(lower, upper, (str(authority),))
        for name, lower, upper, authority in rows
    }


def interval_rows(
    domains: Mapping[str, Int64Interval]
) -> tuple[tuple[str, int, int, str], ...]:
    return tuple(
        (
            name, domain.lower, domain.upper,
            "+".join(domain.authorities) or "int64_interval",
        )
        for name, domain in sorted(domains.items())
    )


__all__ = [
    "INT64_MIN", "INT64_MAX", "Int64Interval", "Int64RangeProof",
    "prove_int64_expression", "derive_local_int64_intervals",
    "prove_int64_range", "intervals_from_rows", "interval_rows",
]
