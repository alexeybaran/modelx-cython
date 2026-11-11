from __future__ import annotations

"""Executable graph-template IR.

This module turns formula-specialized variants plus the concrete :class:`GraphIR`
into the schedule consumed by both generated Python and Cython.  It deliberately
uses structural roles (coordinate family, scalar, fold/reduction) and dependency
barriers rather than an actuarial notion of time or a project-specific stage.
"""

import ast
import copy
import hashlib
import heapq
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

import networkx as nx

from .coordinate_relation import (
    AffineAliasRead,
    AffineOffset,
    CausalHistoryProof,
    CoordinateRelation,
    DerivedClockProof,
    DerivedClockRead,
    DerivedHistoryRead,
    TransitionLagProof,
)


class TemplateError(RuntimeError):
    pass


def _digest(parts: Iterable[str], prefix: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


def _canonical_guarded_family_signature(variants: dict[str, Any], uids: Iterable[str]) -> str:
    parts: list[str] = []
    for uid in sorted(uids):
        cv = variants[uid]
        guards = tuple(
            (
                int(getattr(g, "code", 0)),
                str(getattr(g, "exception_type", "")),
                repr(getattr(g, "message", None)),
            )
            for g in getattr(cv, "guards", ())
        )
        parts.append(
            "%s|role=%s|aux=%r|guards=%r"
            % (
                getattr(cv, "source_fullname", uid),
                getattr(cv, "role", ""),
                getattr(cv, "aux_values", ()),
                guards,
            )
        )
    return _digest(parts, "guardfam")


def _offset(node: ast.AST, variable: str) -> int | None:
    if isinstance(node, ast.Name) and node.id == variable:
        return 0
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.left, ast.Name)
        and node.left.id == variable
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, int)
    ):
        if isinstance(node.op, ast.Sub):
            return -int(node.right.value)
        if isinstance(node.op, ast.Add):
            return int(node.right.value)
    return None


def _ast_sig(nodes: tuple[ast.AST, ...]) -> tuple[str, ...]:
    return tuple(ast.dump(x, include_attributes=False) for x in nodes)


def _range_signature(args: tuple[ast.AST, ...]) -> tuple[str, str, str]:
    if len(args) == 1:
        return ("0", ast.dump(args[0], include_attributes=False), "1")
    if len(args) == 2:
        return (
            ast.dump(args[0], include_attributes=False),
            ast.dump(args[1], include_attributes=False),
            "1",
        )
    if len(args) == 3:
        return tuple(ast.dump(x, include_attributes=False) for x in args)  # type: ignore[return-value]
    raise TemplateError(f"range() with {len(args)} arguments is outside the executable template subset")


def _range_parts(args: tuple[ast.AST, ...]) -> tuple[ast.AST, ast.AST, int]:
    """Return exact Python-range start/stop and a proven constant nonzero step."""
    _range_signature(args)
    if len(args) == 1:
        start, stop, step_node = ast.Constant(0), args[0], ast.Constant(1)
    elif len(args) == 2:
        start, stop, step_node = args[0], args[1], ast.Constant(1)
    else:
        start, stop, step_node = args
    try:
        step_value = ast.literal_eval(step_node)
    except Exception:
        step_value = None
    if not (
        isinstance(step_value, int)
        and not isinstance(step_value, bool)
        and int(step_value) != 0
    ):
        raise TemplateError("executable template requires a compile-time constant nonzero integer range step")
    return start, stop, int(step_value)


def _validate_range(args: tuple[ast.AST, ...]) -> int:
    """Validate the exact integer affine coordinate domain and return its step."""
    _, _, step = _range_parts(args)
    return step


def _literal_int(node: ast.AST) -> int | None:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _fixed_coordinate_expr_value(node: ast.AST, variable: str, coordinate: int) -> Any:
    """Evaluate the deliberately tiny pure expression subset used by seed guards."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id == variable:
        return int(coordinate)
    if isinstance(node, ast.UnaryOp):
        value = _fixed_coordinate_expr_value(node.operand, variable, coordinate)
        if isinstance(node.op, ast.Not):
            return not bool(value)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        raise ValueError(type(node.op).__name__)
    if isinstance(node, ast.BinOp):
        left = _fixed_coordinate_expr_value(node.left, variable, coordinate)
        right = _fixed_coordinate_expr_value(node.right, variable, coordinate)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError(type(node.op).__name__)
    if isinstance(node, ast.BoolOp):
        values = [_fixed_coordinate_expr_value(v, variable, coordinate) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(v) for v in values)
        if isinstance(node.op, ast.Or):
            return any(bool(v) for v in values)
        raise ValueError(type(node.op).__name__)
    if isinstance(node, ast.Compare):
        left = _fixed_coordinate_expr_value(node.left, variable, coordinate)
        for op, rhs_node in zip(node.ops, node.comparators):
            right = _fixed_coordinate_expr_value(rhs_node, variable, coordinate)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Is) and right is None:
                ok = left is None
            elif isinstance(op, ast.IsNot) and right is None:
                ok = left is not None
            else:
                raise ValueError(type(op).__name__)
            if not ok:
                return False
            left = right
        return True
    raise ValueError(type(node).__name__)


def _fixed_coordinate_seed_safe(
    function: ast.FunctionDef,
    coordinate: int,
    roles: dict[str, str],
) -> bool:
    """Prove that one pre-range evaluation needs no coordinate state.

    Known coordinate-only guards are followed exactly. Unknown guards remain
    conservative: both branches are considered reachable. A seed is admitted
    only when no reachable path contains a coordinate/derived-state Cell call.
    """
    if not function.args.args:
        return False
    variable = function.args.args[0].arg
    unavailable_roles = {"coordinate", "derived_state", "reduction", "vector"}

    def has_state_call(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and roles.get(child.func.id) in unavailable_roles
            for child in ast.walk(node)
        )

    def safe_sequence(stmts: list[ast.stmt]) -> tuple[bool, bool]:
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                try:
                    decision = bool(
                        _fixed_coordinate_expr_value(stmt.test, variable, coordinate)
                    )
                except Exception:
                    decision = None
                if decision is None and has_state_call(stmt.test):
                    return False, False
                if decision is True:
                    safe, terminated = safe_sequence(stmt.body)
                    if not safe:
                        return False, False
                    if terminated:
                        return True, True
                    continue
                if decision is False:
                    safe, terminated = safe_sequence(stmt.orelse)
                    if not safe:
                        return False, False
                    if terminated:
                        return True, True
                    continue
                body_safe, body_term = safe_sequence(stmt.body)
                else_safe, else_term = safe_sequence(stmt.orelse)
                if not body_safe or not else_safe:
                    return False, False
                if body_term and else_term:
                    return True, True
                continue
            if has_state_call(stmt):
                return False, False
            if isinstance(stmt, (ast.For, ast.While, ast.Try, ast.With, ast.Match)):
                return False, False
            if isinstance(stmt, (ast.Return, ast.Raise)):
                return True, True
        return True, False

    safe, _terminated = safe_sequence(function.body)
    return bool(safe)


def _period_floor_bucket(node: ast.AST, variable: str) -> int | None:
    """Recognize ``k * max(0, t // k)`` (commuted multiply accepted).

    This deliberately tiny proof form is enough to validate the architecture on
    anniversary/period-start reads without pretending to be a symbolic algebra
    system.  Unknown derived coordinates continue to fail closed.
    """
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return None
    if isinstance(node.left, ast.Constant) and isinstance(node.left.value, int) and not isinstance(node.left.value, bool):
        k = int(node.left.value)
        inner = node.right
    elif isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and not isinstance(node.right.value, bool):
        k = int(node.right.value)
        inner = node.left
    else:
        return None
    if k <= 0:
        return None

    if (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "max"
        and len(inner.args) == 2
        and not inner.keywords
    ):
        zero = None
        other = None
        for arg in inner.args:
            if isinstance(arg, ast.Constant) and arg.value == 0:
                zero = arg
            else:
                other = arg
        if zero is None or other is None:
            return None
        inner = other

    if not (
        isinstance(inner, ast.BinOp)
        and isinstance(inner.op, ast.FloorDiv)
        and isinstance(inner.left, ast.Name)
        and inner.left.id == variable
        and isinstance(inner.right, ast.Constant)
        and isinstance(inner.right.value, int)
        and not isinstance(inner.right.value, bool)
        and int(inner.right.value) == k
    ):
        return None
    return k


def _prove_past_or_same_derived_coordinate(
    node: ast.AST,
    variable: str,
    range_args: tuple[ast.AST, ...],
    coordinate_step: int,
) -> CausalHistoryProof:
    """Prove one bounded history-read form and return its executable proof.

    The first implementation intentionally supports only ascending integer grids
    starting at a nonnegative constant and a positive period-floor snapshot.  It
    proves three properties required by the history address calculation:

    * queried coordinate is never in the future;
    * queried coordinate is not before the materialized history domain;
    * queried coordinate is aligned to the source coordinate grid.
    """
    start, _stop, step = _range_parts(range_args)
    start_value = _literal_int(start)
    if step != int(coordinate_step) or step <= 0:
        raise TemplateError(
            "derived-coordinate history reads currently require an ascending positive-step executable range"
        )
    if start_value is None or start_value < 0:
        raise TemplateError(
            "derived-coordinate history read requires a proven nonnegative constant loop start"
        )
    period = _period_floor_bucket(node, variable)
    if period is None:
        raise TemplateError(
            f"derived coordinate is outside the proven causal snapshot subset: {ast.unparse(node)}"
        )
    if start_value != 0:
        raise TemplateError(
            "period-floor history proof currently requires a zero-based coordinate range"
        )
    if period % step != 0:
        raise TemplateError(
            f"derived coordinate period {period} is not aligned to executable range step {step}"
        )
    note = (
        f"proved 0 <= {ast.unparse(node)} <= {variable} on zero-based ascending integer range; "
        f"period {period} aligned to step {step}"
    )
    return CausalHistoryProof(
        proof_kind="period_floor_snapshot",
        relation="past_or_same",
        period=int(period),
        range_start=int(start_value),
        coordinate_step=int(step),
        normalized_expr=ast.unparse(node),
        note=note,
    )


def _eval_proof_int_expr(
    node: ast.AST,
    variable: str,
    coordinate: int,
    *,
    static_scalar_values: dict[str, int],
    static_scalar_resolver: Callable[[str], int | None] | None = None,
) -> int:
    """Evaluate the deliberately tiny integer algebra admitted by history proofs.

    Coordinate helper aliases and one-shot locals are expanded before this
    evaluator is called.  The remaining zero-argument canonical calls may only be
    full-run-proven integer scalar facts supplied by the frontend.  Nothing here
    executes model code or consults representative observed coordinates.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return int(node.value)
        raise TemplateError("bounded history proof requires integer constants")
    if isinstance(node, ast.Name):
        if node.id == variable:
            return int(coordinate)
        raise TemplateError(f"bounded history proof has unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        value = _eval_proof_int_expr(
            node.operand, variable, coordinate, static_scalar_values=static_scalar_values,
            static_scalar_resolver=static_scalar_resolver
        )
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise TemplateError("bounded history proof has unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _eval_proof_int_expr(
            node.left, variable, coordinate, static_scalar_values=static_scalar_values,
            static_scalar_resolver=static_scalar_resolver
        )
        right = _eval_proof_int_expr(
            node.right, variable, coordinate, static_scalar_values=static_scalar_values,
            static_scalar_resolver=static_scalar_resolver
        )
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise TemplateError("bounded history proof encountered division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise TemplateError("bounded history proof encountered modulo by zero")
            return left % right
        raise TemplateError("bounded history proof has unsupported binary operator")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if not node.args and not node.keywords:
            if name not in static_scalar_values and static_scalar_resolver is not None:
                resolved = static_scalar_resolver(name)
                if resolved is not None:
                    static_scalar_values[name] = int(resolved)
            if name in static_scalar_values:
                return int(static_scalar_values[name])
        if name in {"min", "max"} and node.args and not node.keywords:
            values = [
                _eval_proof_int_expr(
                    arg, variable, coordinate, static_scalar_values=static_scalar_values,
                    static_scalar_resolver=static_scalar_resolver
                )
                for arg in node.args
            ]
            return min(values) if name == "min" else max(values)
        if name == "int" and len(node.args) == 1 and not node.keywords:
            return int(
                _eval_proof_int_expr(
                    node.args[0], variable, coordinate,
                    static_scalar_values=static_scalar_values,
                    static_scalar_resolver=static_scalar_resolver,
                )
            )
    raise TemplateError(
        f"derived coordinate is outside the strict bounded-history algebra: {ast.unparse(node)}"
    )


def _prove_strict_bounded_past_coordinate(
    node: ast.AST,
    variable: str,
    caller_domain: tuple[int, int] | None,
    range_args: tuple[ast.AST, ...],
    coordinate_step: int,
    *,
    static_scalar_values: dict[str, int],
    static_scalar_resolver: Callable[[str], int | None] | None = None,
) -> CausalHistoryProof:
    """Prove an arbitrary *bounded* strict-past read over a complete caller domain.

    The proof is exact over the frontend's source-proven caller interval, not over
    representative trace coordinates.  We deliberately enumerate only a bounded
    integer interval after accepting a tiny source algebra.  This keeps the proof
    easy to audit and fail-closed while covering calendar snapshots such as a
    previous contract anniversary.
    """
    if caller_domain is None:
        raise TemplateError("strict bounded-history proof requires a source-proven caller domain")
    caller_lo, caller_hi = (int(caller_domain[0]), int(caller_domain[1]))
    if caller_lo > caller_hi:
        raise TemplateError("strict bounded-history proof received an empty caller domain")
    if caller_hi - caller_lo > 10000:
        raise TemplateError("strict bounded-history proof domain exceeds the generic proof cap")
    start, _stop, step = _range_parts(range_args)
    start_value = _literal_int(start)
    if step != int(coordinate_step) or step <= 0:
        raise TemplateError("strict bounded-history proof requires an ascending positive-step range")
    if start_value is None:
        raise TemplateError("strict bounded-history proof requires a constant physical range start")

    lags: list[int] = []
    for t in range(caller_lo, caller_hi + 1):
        if (t - start_value) % step != 0:
            # The caller-domain interval is an over-approximation.  Only physical
            # coordinates can execute, so off-grid integers need no proof.
            continue
        q = _eval_proof_int_expr(
            node, variable, t, static_scalar_values=static_scalar_values,
            static_scalar_resolver=static_scalar_resolver
        )
        if q < start_value:
            raise TemplateError(
                f"strict bounded-history coordinate {q} precedes physical history start {start_value}"
            )
        if (q - start_value) % step != 0:
            raise TemplateError(
                f"strict bounded-history coordinate {q} is not aligned to physical step {step}"
            )
        if q >= t:
            raise TemplateError(
                f"derived coordinate is not strict past over complete caller domain: q({t})={q}"
            )
        lag = t - q
        if lag % step != 0:
            raise TemplateError("strict bounded-history lag is not aligned to the physical step")
        lags.append(lag)
    if not lags:
        raise TemplateError("strict bounded-history proof has no executable caller coordinates")
    min_lag = min(lags)
    max_lag = max(lags)
    note = (
        f"proved {start_value} <= {ast.unparse(node)} < {variable} for every physical "
        f"caller coordinate in [{caller_lo}, {caller_hi}]; lag {min_lag}..{max_lag} "
        f"and step {step} are exact over the source-proven caller domain"
    )
    return CausalHistoryProof(
        proof_kind="strict_bounded_past",
        relation="strict_past",
        period=0,
        range_start=int(start_value),
        coordinate_step=int(step),
        normalized_expr=ast.unparse(node),
        note=note,
        min_lag=int(min_lag),
        max_lag=int(max_lag),
        caller_lo=int(caller_lo),
        caller_hi=int(caller_hi),
    )


def _expand_coordinate_proof_aliases(
    node: ast.AST,
    variants: dict[str, Any],
    *,
    stack: tuple[str, ...] = (),
) -> ast.AST:
    """Inline tiny single-return coordinate aliases for proof only.

    Generated code retains the canonical Cell calls.  This expansion exists only
    so a proof can see through harmless same-time helpers such as
    ``duration(t) -> max(0, t // 12)``. Statementful/stateful formulas are never
    expanded here.
    """

    class Expand(ast.NodeTransformer):
        def visit_Call(self, call: ast.Call):
            call = self.generic_visit(call)
            if not isinstance(call.func, ast.Name) or call.func.id not in variants:
                return call
            uid = call.func.id
            if uid in stack:
                return call
            cv = variants[uid]
            if getattr(cv, "role", None) != "coordinate" or len(call.args) != 1 or call.keywords:
                return call
            fn = getattr(cv, "function", None)
            if not (
                isinstance(fn, ast.FunctionDef)
                and len(fn.args.args) == 1
                and len(fn.body) == 1
                and isinstance(fn.body[0], ast.Return)
                and fn.body[0].value is not None
            ):
                return call
            param = fn.args.args[0].arg

            class Substitute(ast.NodeTransformer):
                def visit_Name(self, n: ast.Name):
                    if isinstance(n.ctx, ast.Load) and n.id == param:
                        return copy.deepcopy(call.args[0])
                    return n

            expanded = Substitute().visit(copy.deepcopy(fn.body[0].value))
            expanded = ast.fix_missing_locations(expanded)
            return _expand_coordinate_proof_aliases(
                expanded, variants, stack=stack + (uid,)
            )

    return ast.fix_missing_locations(Expand().visit(copy.deepcopy(node)))


def _expand_simple_coordinate_locals(node: ast.AST, target_variant: Any) -> ast.AST:
    """Expand one-shot unconditional locals for coordinate proof only.

    The generated formula remains unchanged.  Only top-level assignments whose
    target has exactly one AST Store in the whole function are eligible.  This is
    the same mutation-safety principle as dev29's literal-local propagation and
    deliberately excludes accumulators, loop variables and branch-local values.
    """
    fn = getattr(target_variant, "function", None)
    if not isinstance(fn, ast.FunctionDef):
        return copy.deepcopy(node)
    store_counts: dict[str, int] = defaultdict(int)
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            store_counts[n.id] += 1
    mapping: dict[str, ast.AST] = {}
    for st in fn.body:
        if (
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Name)
            and store_counts.get(st.targets[0].id) == 1
        ):
            mapping[st.targets[0].id] = copy.deepcopy(st.value)
        elif (
            isinstance(st, ast.AnnAssign)
            and isinstance(st.target, ast.Name)
            and st.value is not None
            and store_counts.get(st.target.id) == 1
        ):
            mapping[st.target.id] = copy.deepcopy(st.value)
    if not mapping:
        return copy.deepcopy(node)

    class ExpandLocal(ast.NodeTransformer):
        def __init__(self):
            self.stack: list[str] = []

        def visit_Name(self, n: ast.Name):
            if not (isinstance(n.ctx, ast.Load) and n.id in mapping):
                return n
            if n.id in self.stack:
                # Cyclic local aliases are not proof-expandable.  Retain the name
                # so the downstream proof evaluator fails closed.
                return n
            self.stack.append(n.id)
            try:
                return self.visit(copy.deepcopy(mapping[n.id]))
            finally:
                self.stack.pop()

    return ast.fix_missing_locations(ExpandLocal().visit(copy.deepcopy(node)))


def _expanded_coordinate_expr(use: "TemplateUse", variants: dict[str, Any]) -> ast.AST:
    rel = use.relation
    if not isinstance(rel, DerivedHistoryRead):
        raise TemplateError("coordinate proof expansion requires an unresolved derived expression")
    expr = _expand_simple_coordinate_locals(rel.expr, variants[use.target])
    return _expand_coordinate_proof_aliases(expr, variants)


def _var_shift(node: ast.AST, variable: str) -> int | None:
    if isinstance(node, ast.Name) and node.id == variable:
        return 0
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.left, ast.Name)
        and node.left.id == variable
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, int)
        and not isinstance(node.right.value, bool)
    ):
        if isinstance(node.op, ast.Add):
            return int(node.right.value)
        if isinstance(node.op, ast.Sub):
            return -int(node.right.value)
    return None


def _floor_bucket_mapping(node: ast.AST, variable: str) -> tuple[int, int, int] | None:
    """Recognize ``((t + shift) // period) + bias`` with literal integers."""
    bias = 0
    core = node
    if (
        isinstance(core, ast.BinOp)
        and isinstance(core.right, ast.Constant)
        and isinstance(core.right.value, int)
        and not isinstance(core.right.value, bool)
    ):
        if isinstance(core.op, ast.Add):
            bias = int(core.right.value)
            core = core.left
        elif isinstance(core.op, ast.Sub):
            bias = -int(core.right.value)
            core = core.left
    if not (
        isinstance(core, ast.BinOp)
        and isinstance(core.op, ast.FloorDiv)
        and isinstance(core.right, ast.Constant)
        and isinstance(core.right.value, int)
        and not isinstance(core.right.value, bool)
    ):
        return None
    period = int(core.right.value)
    if period <= 0:
        return None
    shift = _var_shift(core.left, variable)
    if shift is None:
        return None
    return period, int(shift), int(bias)


def _prove_derived_clock_mapping(
    node: ast.AST,
    variable: str,
    range_args: tuple[ast.AST, ...],
    coordinate_step: int,
    *,
    required_initial_bucket: int | None = 1,
) -> DerivedClockProof:
    start, _stop, step = _range_parts(range_args)
    start_value = _literal_int(start)
    mapping = _floor_bucket_mapping(node, variable)
    if mapping is None:
        raise TemplateError(
            f"derived clock is outside the proven floor-bucket subset: {ast.unparse(node)}"
        )
    period, shift, bias = mapping
    if start_value is None:
        raise TemplateError("derived clock requires a literal primary range start")
    if step != int(coordinate_step) or step <= 0 or step > period:
        raise TemplateError(
            "derived clock requires a positive primary step no larger than its bucket period"
        )
    initial = ((int(start_value) + shift) // period) + bias
    if required_initial_bucket is not None and initial != int(required_initial_bucket):
        raise TemplateError(
            f"derived clock must start at bucket {required_initial_bucket}, got {initial} from {ast.unparse(node)}"
        )
    note = (
        f"proved {ast.unparse(node)} starts at bucket {initial} and advances by 0 or 1 on each "
        f"primary step {step}; period={period} shift={shift} bias={bias}"
    )
    return DerivedClockProof(
        proof_kind="monotone_floor_bucket",
        period=period,
        shift=shift,
        bias=bias,
        range_start=int(start_value),
        coordinate_step=int(step),
        initial_bucket=int(initial),
        normalized_expr=ast.unparse(node),
        note=note,
    )


def _integer_affine_form(node: ast.AST, variable: str) -> tuple[int, int] | None:
    """Return integer ``(slope, intercept)`` for a tiny affine AST subset.

    This is intentionally structural proof support, not symbolic algebra.  It is
    sufficient for anniversary expressions such as ``12 * (y - 1)`` while
    rejecting nonlinear/history-shaped expressions rather than simplifying them
    optimistically.
    """
    if isinstance(node, ast.Name) and node.id == variable:
        return 1, 0
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return 0, int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _integer_affine_form(node.operand, variable)
        if inner is None:
            return None
        return inner if isinstance(node.op, ast.UAdd) else (-inner[0], -inner[1])
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _integer_affine_form(node.left, variable)
        right = _integer_affine_form(node.right, variable)
        if left is None or right is None:
            return None
        sign = 1 if isinstance(node.op, ast.Add) else -1
        return left[0] + sign * right[0], left[1] + sign * right[1]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _integer_affine_form(node.left, variable)
        right = _integer_affine_form(node.right, variable)
        if left is None or right is None:
            return None
        # At least one side must be a literal affine constant.
        if left[0] == 0:
            return right[0] * left[1], right[1] * left[1]
        if right[0] == 0:
            return left[0] * right[1], left[1] * right[1]
        return None
    return None


def _prove_transition_lag(
    node: ast.AST,
    variable: str,
    clock: DerivedClockProof,
    *,
    required_primary_offset: int | None = -1,
) -> TransitionLagProof:
    """Prove an affine primary-state read at derived-clock transition events.

    For ``y = floor((t+s)/P)+b`` the transition into bucket ``y`` occurs at
    ``t = P*(y-b)-s``.  Hence an affine source coordinate ``A*y+C`` is an
    exact primary lag iff ``A == P``; the lag is ``C + P*b + s``.
    """
    affine = _integer_affine_form(node, variable)
    if affine is None:
        raise TemplateError(
            f"derived-clock transition read is outside the affine proof subset: {ast.unparse(node)}"
        )
    slope, intercept = affine
    if slope != int(clock.period):
        raise TemplateError(
            f"derived-clock transition read slope {slope} does not match period {clock.period}: {ast.unparse(node)}"
        )
    offset = int(intercept + clock.period * clock.bias + clock.shift)
    if offset >= 0:
        raise TemplateError(
            f"derived-clock transition read is not strictly in the past; primary offset={offset}: {ast.unparse(node)}"
        )
    if required_primary_offset is not None and offset != int(required_primary_offset):
        raise TemplateError(
            f"derived-clock transition read requires primary offset {required_primary_offset}, got {offset}: {ast.unparse(node)}"
        )
    note = (
        f"proved {ast.unparse(node)} equals primary transition coordinate "
        f"t{offset:+d} for clock period={clock.period} shift={clock.shift} bias={clock.bias}"
    )
    return TransitionLagProof(
        proof_kind="derived_clock_transition_affine_lag",
        period=int(clock.period),
        shift=int(clock.shift),
        bias=int(clock.bias),
        source_slope=int(slope),
        source_intercept=int(intercept),
        primary_offset=offset,
        normalized_expr=ast.unparse(node),
        note=note,
    )


@dataclass(frozen=True)
class TemplateUse:
    source: str
    target: str
    context: str  # coordinate | scalar | reduction
    relation: CoordinateRelation | None = field(default=None, compare=False)

    @property
    def offset(self) -> int | None:
        """Compatibility view for affine-only callers."""
        return (
            int(self.relation.offset)
            if isinstance(self.relation, (AffineOffset, AffineAliasRead))
            else None
        )

    @property
    def coordinate_expr(self) -> ast.AST | None:
        """Compatibility view; permission lives in ``DerivedHistoryRead.proof``."""
        return self.relation.expr if isinstance(self.relation, DerivedHistoryRead) else None

    @property
    def derived_history(self) -> DerivedHistoryRead | None:
        return self.relation if isinstance(self.relation, DerivedHistoryRead) else None

    @property
    def affine_alias(self) -> AffineAliasRead | None:
        return self.relation if isinstance(self.relation, AffineAliasRead) else None

    @property
    def derived_clock(self) -> DerivedClockRead | None:
        return self.relation if isinstance(self.relation, DerivedClockRead) else None

    @property
    def has_proven_history(self) -> bool:
        return isinstance(self.relation, DerivedHistoryRead) and self.relation.proof is not None


@dataclass(frozen=True)
class DerivedClockState:
    uid: str
    phase: int
    mapping_expr: ast.AST
    proof: DerivedClockProof
    self_offset: int
    consumer_uids: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "phase": int(self.phase),
            "mapping_expr": ast.unparse(self.mapping_expr),
            "proof": self.proof.manifest(),
            "self_offset": int(self.self_offset),
            "consumer_uids": list(self.consumer_uids),
        }


@dataclass(frozen=True)
class DerivedClockTransitionRead:
    source_uid: str
    target_uid: str
    expr: ast.AST
    proof: TransitionLagProof

    def manifest(self) -> dict[str, Any]:
        return {
            "source_uid": self.source_uid,
            "target_uid": self.target_uid,
            "expr": ast.unparse(self.expr),
            "proof": self.proof.manifest(),
        }


@dataclass(frozen=True)
class DerivedClockRegionCandidate:
    """Structurally proven cross-clock region, not yet executable in dev32 foundation.

    A region is rooted at a coordinate formula consumed through one monotone
    derived-clock mapping.  Same-clock acyclic helpers are absorbed into the
    region.  Reads from primary coordinate state are allowed only when their
    transition relation is separately proven strictly lagged.

    Keeping this object non-executable is deliberate: it validates the graph
    classification and proof boundary before storage/codegen semantics are added.
    """

    root_uid: str
    member_uids: tuple[str, ...]
    mapping_expr: ast.AST
    clock_proof: DerivedClockProof
    consumer_uids: tuple[str, ...]
    transition_reads: tuple[DerivedClockTransitionRead, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "root_uid": self.root_uid,
            "member_uids": list(self.member_uids),
            "mapping_expr": ast.unparse(self.mapping_expr),
            "clock_proof": self.clock_proof.manifest(),
            "consumer_uids": list(self.consumer_uids),
            "transition_reads": [row.manifest() for row in self.transition_reads],
        }


@dataclass(frozen=True)
class DerivedClockRegion:
    """Executable acyclic region living on one proven derived clock.

    The first executable subset is intentionally narrow: region members are
    same-clock acyclic formulas, every primary-state ingress has an exact
    ``TransitionLagProof(-1)``, and the concrete initial bucket is proven safe
    for eager topological evaluation before the physical loop begins.
    """

    root_uid: str
    phase: int
    member_uids: tuple[str, ...]
    persistent_uids: tuple[str, ...]
    transient_uids: tuple[str, ...]
    mapping_expr: ast.AST
    clock_proof: DerivedClockProof
    consumer_uids: tuple[str, ...]
    transition_reads: tuple[DerivedClockTransitionRead, ...]
    initial_safety_notes: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "root_uid": self.root_uid,
            "phase": int(self.phase),
            "member_uids": list(self.member_uids),
            "persistent_uids": list(self.persistent_uids),
            "transient_uids": list(self.transient_uids),
            "mapping_expr": ast.unparse(self.mapping_expr),
            "clock_proof": self.clock_proof.manifest(),
            "consumer_uids": list(self.consumer_uids),
            "transition_reads": [row.manifest() for row in self.transition_reads],
            "initial_safety_notes": list(self.initial_safety_notes),
        }


@dataclass(frozen=True)
class BoundarySeed:
    uid: str
    phase: int
    coordinate: int
    note: str

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "phase": int(self.phase),
            "coordinate": int(self.coordinate),
            "note": self.note,
        }


@dataclass
class ExecutableLoopTemplate:
    uid: str
    index: int
    coordinate_symbol: str
    range_args: tuple[ast.AST, ...]
    coordinate_step: int  # signed Python range step; physical storage is 0..T-1
    scan_direction: int  # +1/-1 over physical iteration positions
    families: tuple[str, ...]
    reductions: tuple[str, ...]
    post_scalars: tuple[str, ...]
    evidence_loop_uids: tuple[str, ...]
    generalization: str
    guard_notes: tuple[str, ...]
    max_lag_by_family: dict[str, int]
    current_slots: dict[str, int]
    current_slot_count: int
    fusable_reductions: tuple[str, ...]
    fusion_reasons: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutablePureMapSpec:
    """State-free coordinate function callable at an arbitrary proven coordinate.

    The canonical frontend role remains ``coordinate``.  This descriptor is an
    execution-semantic refinement: the value is recomputed from its requested
    coordinate and read-only scalar/input dependencies instead of participating in
    causal scan order or coordinate storage.
    """

    uid: str
    coordinate_parameter: str
    scalar_dependencies: tuple[str, ...]
    pure_map_dependencies: tuple[str, ...]
    proof_scope: str = "specialized_canonical_closure"
    provenance: tuple[str, ...] = ("source_value_semantics_v1", "canonical_state_free_codegen_subset_v1")

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "coordinate_parameter": self.coordinate_parameter,
            "scalar_dependencies": list(self.scalar_dependencies),
            "pure_map_dependencies": list(self.pure_map_dependencies),
            "proof_scope": self.proof_scope,
            "provenance": list(self.provenance),
        }


@dataclass
class TemplateStoragePlan:
    optimization_level: str
    current_slots: dict[str, int]
    current_slot_count: int
    scalar_slots: dict[str, int]
    history_slots: dict[str, int]
    ring_bases: dict[str, int]
    ring_depths: dict[str, int]
    derived_state_slots: dict[str, int]
    derived_region_state_slots: dict[str, int]
    boundary_seed_slots: dict[str, int]
    state_slot_count: int
    fused_reductions: frozenset[str]
    history_reasons: dict[str, tuple[str, ...]]

    @property
    def time_slots(self) -> dict[str, int]:
        """Compatibility alias used by older benchmark/reporting code."""
        return self.current_slots

    @property
    def array_slots(self) -> dict[str, int]:
        """Compatibility alias: a materialized history is a full coordinate array."""
        return self.history_slots

    def estimated_bytes_per_point(self, horizon: int) -> int:
        return 8 * (self.current_slot_count + self.state_slot_count + len(self.scalar_slots) + len(self.history_slots) * max(1, int(horizon)))


@dataclass
class ExecutableGraph:
    pre_scalars: tuple[str, ...]
    loops: tuple[ExecutableLoopTemplate, ...]
    output_uid: str
    roles: dict[str, str]
    uses: tuple[TemplateUse, ...]
    coordinate_phase: dict[str, int]
    scalar_barrier: dict[str, int]
    reduction_phase: dict[str, int]
    derived_clock_states: dict[str, DerivedClockState]
    derived_clock_regions: dict[str, DerivedClockRegion]
    boundary_seeds: dict[str, BoundarySeed]
    formula_hash: str
    pure_maps: dict[str, ExecutablePureMapSpec] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    schedule_authority: str = "legacy"
    schedule_authority_contract_uid: str | None = None

    @property
    def common_range_args(self) -> tuple[ast.AST, ...]:
        if not self.loops:
            raise TemplateError("no executable loop template")
        sig = _range_signature(self.loops[0].range_args)
        for block in self.loops[1:]:
            if _range_signature(block.range_args) != sig:
                raise TemplateError("current native backend requires executable loop templates to share one coordinate range")
        return self.loops[0].range_args

    @property
    def family_uids(self) -> tuple[str, ...]:
        return tuple(uid for block in self.loops for uid in block.families)

    @property
    def reduction_uids(self) -> tuple[str, ...]:
        return tuple(uid for block in self.loops for uid in block.reductions)

    @property
    def scalar_uids(self) -> tuple[str, ...]:
        return tuple(sorted(uid for uid, role in self.roles.items() if role in ("scalar", "reduction")))

    @property
    def pure_map_uids(self) -> tuple[str, ...]:
        return tuple(sorted(self.pure_maps))

    @property
    def derived_region_transient_uids(self) -> tuple[str, ...]:
        return tuple(
            uid
            for region in self.derived_clock_regions.values()
            for uid in region.transient_uids
        )

    def derived_region_for_member(self, uid: str) -> DerivedClockRegion | None:
        for region in self.derived_clock_regions.values():
            if uid in region.member_uids:
                return region
        return None

    def transition_lag_relation(
        self, source: str, target: str, expr: ast.AST
    ) -> DerivedClockTransitionRead | None:
        sig = ast.dump(expr, include_attributes=False)
        for region in self.derived_clock_regions.values():
            for row in region.transition_reads:
                if (
                    row.source_uid == source
                    and row.target_uid == target
                    and ast.dump(row.expr, include_attributes=False) == sig
                ):
                    return row
        return None

    def derived_history_relation(
        self, source: str, target: str, expr: ast.AST
    ) -> DerivedHistoryRead | None:
        sig = ast.dump(expr, include_attributes=False)
        for use in self.uses:
            rel = use.derived_history
            if (
                use.source == source
                and use.target == target
                and rel is not None
                and ast.dump(rel.expr, include_attributes=False) == sig
            ):
                return rel
        return None

    def affine_alias_relation(
        self, source: str, target: str, expr: ast.AST
    ) -> AffineAliasRead | None:
        sig = ast.dump(expr, include_attributes=False)
        for use in self.uses:
            rel = use.affine_alias
            if (
                use.source == source
                and use.target == target
                and rel is not None
                and ast.dump(rel.expr, include_attributes=False) == sig
            ):
                return rel
        return None

    def derived_clock_relation(
        self, source: str, target: str, expr: ast.AST
    ) -> DerivedClockRead | None:
        sig = ast.dump(expr, include_attributes=False)
        for use in self.uses:
            rel = use.derived_clock
            if (
                use.source == source
                and use.target == target
                and rel is not None
                and ast.dump(rel.expr, include_attributes=False) == sig
            ):
                return rel
        return None

    def storage_plan(self, optimization_level: str) -> TemplateStoragePlan:
        level = optimization_level.upper()
        if level not in {"O0", "O1", "O2"}:
            raise TemplateError(
                f"optimization level {level!r} is not implemented; supported levels are O0, O1 and O2"
            )
        from .passes import fused_reductions_for_level
        fused = fused_reductions_for_level(self, level)
        families = list(self.family_uids)
        history: set[str] = set(families if level == "O0" else ())
        reasons: dict[str, list[str]] = defaultdict(list)
        if level == "O0":
            for uid in families:
                reasons[uid].append("O0 conservatively materializes every coordinate family")
        else:
            for use in self.uses:
                if self.roles.get(use.source) != "coordinate":
                    continue
                if (
                    use.context == "coordinate"
                    and self.transition_lag_relation(
                        use.source,
                        use.target,
                        use.coordinate_expr if use.coordinate_expr is not None else ast.Constant(0),
                    )
                    is not None
                ):
                    # Exact transition lag is served by the source family's normal
                    # rolling/history state; it does not require materializing the
                    # full primary history merely because source syntax uses a
                    # derived-coordinate expression.
                    continue
                source_phase = self.coordinate_phase[use.source]
                if use.context == "reduction":
                    if use.target in fused and source_phase >= self.reduction_phase[use.target]:
                        # Same-loop ordered reduction reads the just-produced current value.
                        continue
                    history.add(use.source)
                    reasons[use.source].append(
                        f"reduction {use.target} consumes coordinate history outside the producing loop"
                    )
                elif use.context == "scalar":
                    history.add(use.source)
                    reasons[use.source].append(
                        f"scalar operation {use.target} consumes a coordinate value outside its producing loop"
                    )
                elif use.context == "coordinate":
                    target_phase = self.coordinate_phase[use.target]
                    if use.coordinate_expr is not None:
                        history.add(use.source)
                        reasons[use.source].append(
                            f"coordinate {use.target} consumes a proven causal derived-coordinate history read"
                        )
                    elif target_phase > source_phase:
                        history.add(use.source)
                        reasons[use.source].append(
                            f"later executable loop {target_phase} consumes values from loop {source_phase}"
                        )

        # Coordinate deltas are normalized by the proven Python range step, so
        # ring depths are measured in physical iteration positions.
        max_lag: dict[str, int] = defaultdict(int)
        for block in self.loops:
            for uid, lag in block.max_lag_by_family.items():
                max_lag[uid] = max(max_lag[uid], int(lag))

        ring_depths: dict[str, int] = {}
        ring_bases: dict[str, int] = {}
        cursor = 0
        for uid in sorted(families):
            if uid in history or max_lag.get(uid, 0) <= 0:
                continue
            depth = max_lag[uid] + 1
            ring_bases[uid] = cursor
            ring_depths[uid] = depth
            cursor += depth

        current_slots: dict[str, int] = {}
        current_count = 0
        for block in self.loops:
            # Blocks execute serially, so their current-value slots are reusable.
            current_slots.update(block.current_slots)
            current_count = max(current_count, block.current_slot_count)

        scalar_slots = {uid: i for i, uid in enumerate(self.scalar_uids)}
        history_slots = {uid: i for i, uid in enumerate(sorted(history))}
        derived_state_slots: dict[str, int] = {}
        for uid in sorted(self.derived_clock_states):
            derived_state_slots[uid] = cursor
            cursor += 1
        derived_region_state_slots: dict[str, int] = {}
        for region in self.derived_clock_regions.values():
            for uid in sorted(region.persistent_uids):
                derived_region_state_slots[uid] = cursor
                cursor += 1
        boundary_seed_slots: dict[str, int] = {}
        for uid in sorted(self.boundary_seeds):
            boundary_seed_slots[uid] = cursor
            cursor += 1
        return TemplateStoragePlan(
            optimization_level=level,
            current_slots=current_slots,
            current_slot_count=max(1, current_count),
            scalar_slots=scalar_slots,
            history_slots=history_slots,
            ring_bases=ring_bases,
            ring_depths=ring_depths,
            derived_state_slots=derived_state_slots,
            derived_region_state_slots=derived_region_state_slots,
            boundary_seed_slots=boundary_seed_slots,
            state_slot_count=cursor,
            fused_reductions=fused,
            history_reasons={k: tuple(v) for k, v in sorted(reasons.items())},
        )

    def build_fingerprint(self, optimization_level: str) -> str:
        plan = self.storage_plan(optimization_level)
        parts = [
            self.formula_hash,
            optimization_level.upper(),
            self.output_uid,
            *[block.uid for block in self.loops],
            *[f"role:{uid}:{role}" for uid, role in sorted(self.roles.items())],
            *[
                "pure:%s:%s:%s:%s" % (
                    uid, spec.coordinate_parameter,
                    ",".join(spec.scalar_dependencies),
                    ",".join(spec.pure_map_dependencies),
                )
                for uid, spec in sorted(self.pure_maps.items())
            ],
            *[f"cur:{uid}:{slot}" for uid, slot in sorted(plan.current_slots.items())],
            *[f"hist:{uid}:{slot}" for uid, slot in sorted(plan.history_slots.items())],
            *[f"ring:{uid}:{plan.ring_bases[uid]}:{plan.ring_depths[uid]}" for uid in sorted(plan.ring_bases)],
            *[f"dstate:{uid}:{slot}" for uid, slot in sorted(plan.derived_state_slots.items())],
            *[f"dregion:{uid}:{slot}" for uid, slot in sorted(plan.derived_region_state_slots.items())],
            *[f"bseed:{uid}:{slot}" for uid, slot in sorted(plan.boundary_seed_slots.items())],
            *[f"fused:{uid}" for uid in sorted(plan.fused_reductions)],
        ]
        return _digest(parts, "build")

    def manifest(self) -> dict[str, Any]:
        return {
            "formula_hash": self.formula_hash,
            "schedule_authority": self.schedule_authority,
            "schedule_authority_contract_uid": self.schedule_authority_contract_uid,
            "pre_scalars": list(self.pre_scalars),
            "output_uid": self.output_uid,
            "roles": dict(sorted(self.roles.items())),
            "pure_maps": {
                uid: spec.manifest() for uid, spec in sorted(self.pure_maps.items())
            },
            "coordinate_phase": dict(sorted(self.coordinate_phase.items())),
            "scalar_barrier": dict(sorted(self.scalar_barrier.items())),
            "reduction_phase": dict(sorted(self.reduction_phase.items())),
            "derived_clock_states": {
                uid: state.manifest() for uid, state in sorted(self.derived_clock_states.items())
            },
            "derived_clock_regions": {
                uid: region.manifest() for uid, region in sorted(self.derived_clock_regions.items())
            },
            "boundary_seeds": {
                uid: seed.manifest() for uid, seed in sorted(self.boundary_seeds.items())
            },
            "uses": [
                {
                    "source": u.source,
                    "target": u.target,
                    "context": u.context,
                    "relation": None if u.relation is None else u.relation.manifest(),
                    # Compatibility fields for older audit consumers.
                    "offset": u.offset,
                    "coordinate_expr": (
                        ast.unparse(u.coordinate_expr) if u.coordinate_expr is not None else None
                    ),
                }
                for u in self.uses
            ],
            "loops": [
                {
                    "uid": b.uid,
                    "index": b.index,
                    "range": [ast.unparse(x) for x in b.range_args],
                    "coordinate_step": b.coordinate_step,
                    "scan_direction": b.scan_direction,
                    "families": list(b.families),
                    "reductions": list(b.reductions),
                    "post_scalars": list(b.post_scalars),
                    "evidence_loop_uids": list(b.evidence_loop_uids),
                    "generalization": b.generalization,
                    "guard_notes": list(b.guard_notes),
                    "max_lag_by_family": dict(sorted(b.max_lag_by_family.items())),
                    "current_slots": dict(sorted(b.current_slots.items())),
                    "current_slot_count": b.current_slot_count,
                    "fusable_reductions": list(b.fusable_reductions),
                    "fusion_reasons": dict(sorted(b.fusion_reasons.items())),
                }
                for b in self.loops
            ],
            "diagnostics": list(self.diagnostics),
        }


def _walk_calls(nodes: Iterable[ast.AST], known: set[str]) -> list[ast.Call]:
    out: list[ast.Call] = []
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in known:
                out.append(child)
    return out


def _topo_subset(names: Iterable[str], uses: Iterable[TemplateUse], *, contexts: set[str] | None = None, zero_only: bool = False) -> list[str]:
    names = list(dict.fromkeys(names))
    name_set = set(names)
    g = nx.DiGraph()
    g.add_nodes_from(names)
    for use in uses:
        if use.source not in name_set or use.target not in name_set:
            continue
        if contexts is not None and use.context not in contexts:
            continue
        if zero_only:
            if (
                isinstance(use.relation, DerivedHistoryRead)
                and use.relation.proof is not None
                and use.relation.proof.relation == "strict_past"
            ):
                # A source-proven strict-past dynamic read cannot impose a
                # same-iteration ordering edge.  Past-or-same snapshot reads keep
                # their conservative edge because they may equal the current
                # coordinate at a boundary.
                continue
            # A proven dynamic history read may equal the current coordinate at a
            # boundary (RILA anniversary months). Conservatively keep the
            # source->target edge in the same-iteration DAG so history is written
            # before the consumer executes. Strict-past reads lose nothing by the
            # stronger ordering.
            if use.offset != 0 and use.coordinate_expr is None:
                continue
        g.add_edge(use.source, use.target)
    if not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        raise TemplateError(f"same-coordinate/template dependency cycle requires fallback: {cycle[:8]!r}")
    return list(nx.topological_sort(g))


def _allocate_current_slots(order: list[str], uses: list[TemplateUse], fusable_sources: set[str]) -> tuple[dict[str, int], int]:
    """Color same-iteration live intervals for coordinate values.

    A value stays live through its last offset-0 consumer.  Potential O2-fused
    reductions execute at the end of the iteration, so their sources are kept
    live through the block end.  The value is written to history/ring storage
    immediately after production, allowing the current register to be reused.
    """
    pos = {uid: i for i, uid in enumerate(order)}
    end = len(order)
    last = {uid: pos[uid] for uid in order}
    for use in uses:
        if (
            use.context == "coordinate"
            and (use.offset == 0 or use.coordinate_expr is not None)
            and use.source in pos and use.target in pos
        ):
            last[use.source] = max(last[use.source], pos[use.target])
    for uid in fusable_sources:
        if uid in last:
            last[uid] = end

    active: list[tuple[int, int]] = []
    free: list[int] = []
    slots: dict[str, int] = {}
    next_slot = 0
    for uid in order:
        p = pos[uid]
        while active and active[0][0] < p:
            _, slot = heapq.heappop(active)
            heapq.heappush(free, slot)
        slot = heapq.heappop(free) if free else next_slot
        if slot == next_slot:
            next_slot += 1
        slots[uid] = slot
        heapq.heappush(active, (last[uid], slot))
    return slots, max(1, next_slot)


def analyze_derived_clock_regions(
    phase_families: Iterable[str],
    uses: list[TemplateUse],
    variants: dict[str, Any],
    range_args: tuple[ast.AST, ...],
    coordinate_step: int,
) -> tuple[DerivedClockRegionCandidate, ...]:
    """Find structurally safe cross-clock regions without making them executable.

    This is the dev32 foundation seam.  It deliberately reuses the graph analyzer's
    existing coordinate relations and topology instead of introducing a MYGA-shaped
    scheduler rule.

    A root is a coordinate family consumed through a monotone floor-bucket mapping
    of the primary loop.  Same-clock, offset-zero, acyclic helpers exclusively owned
    by that root are absorbed into the region.  Any coordinate dependency entering
    the region from outside must then prove to be an exact strict lag at *bucket
    transition* time.  The first accepted lag is one physical primary step.

    The returned candidates are evidence only.  Storage/codegen are intentionally
    unchanged; callers can use this function to distinguish a proven structural
    opportunity from an arbitrary dynamic history expression.
    """
    phase_set = set(phase_families)
    if not phase_set:
        return ()

    by_source: dict[str, list[int]] = defaultdict(list)
    for idx, use in enumerate(uses):
        if (
            use.context == "coordinate"
            and use.source in phase_set
            and use.target in phase_set
            and isinstance(use.relation, DerivedHistoryRead)
        ):
            try:
                expanded = _expanded_coordinate_expr(use, variants)
                _prove_derived_clock_mapping(
                    expanded,
                    "t",
                    range_args,
                    coordinate_step,
                    required_initial_bucket=None,
                )
            except TemplateError:
                continue
            by_source[use.source].append(idx)

    out: list[DerivedClockRegionCandidate] = []
    for root, root_mapping_indices in sorted(by_source.items()):
        mapping_expr: ast.AST | None = None
        clock: DerivedClockProof | None = None
        signature: tuple[int, int, int] | None = None
        valid = True
        for idx in root_mapping_indices:
            use = uses[idx]
            expanded = _expanded_coordinate_expr(use, variants)
            proof = _prove_derived_clock_mapping(
                expanded,
                "t",
                range_args,
                coordinate_step,
                required_initial_bucket=None,
            )
            sig = (proof.period, proof.shift, proof.bias)
            if signature is None:
                signature = sig
                mapping_expr = copy.deepcopy(expanded)
                clock = proof
            elif sig != signature:
                valid = False
                break
        if not valid or mapping_expr is None or clock is None:
            continue

        region: set[str] = {root}
        # Absorb pure same-clock helpers bottom-up.  Exclusive ownership prevents a
        # primary family from being relabeled merely because one consumer calls it
        # with the same variable name.
        changed = True
        while changed:
            changed = False
            for use in uses:
                if not (
                    use.context == "coordinate"
                    and use.target in region
                    and use.source in phase_set
                    and use.source not in region
                    and use.offset == 0
                ):
                    continue
                source = use.source
                self_uses = [
                    row
                    for row in uses
                    if row.context == "coordinate"
                    and row.source == source
                    and row.target == source
                ]
                if self_uses:
                    continue
                outgoing = [
                    row
                    for row in uses
                    if row.context == "coordinate"
                    and row.source == source
                    and row.target != source
                ]
                if outgoing and all(row.target in region for row in outgoing):
                    region.add(source)
                    changed = True

        # Every coordinate use leaving the region must be one of the proven clock
        # consumers.  This rejects a helper/value that is simultaneously used on
        # primary time with ordinary affine semantics.
        consumers: set[str] = set()
        for use in uses:
            if not (
                use.context == "coordinate"
                and use.source in region
                and use.target not in region
            ):
                continue
            if not isinstance(use.relation, DerivedHistoryRead):
                valid = False
                break
            expanded = _expanded_coordinate_expr(use, variants)
            try:
                proof = _prove_derived_clock_mapping(
                    expanded,
                    "t",
                    range_args,
                    coordinate_step,
                    required_initial_bucket=None,
                )
            except TemplateError:
                valid = False
                break
            if (proof.period, proof.shift, proof.bias) != signature:
                valid = False
                break
            consumers.add(use.target)
        if not valid or not consumers:
            continue

        transition_reads: list[DerivedClockTransitionRead] = []
        for use in uses:
            if not (
                use.context == "coordinate"
                and use.target in region
                and use.source not in region
                and use.source in phase_set
            ):
                continue
            if not isinstance(use.relation, DerivedHistoryRead):
                valid = False
                break
            expanded = _expanded_coordinate_expr(use, variants)
            try:
                lag = _prove_transition_lag(
                    expanded,
                    "t",
                    clock,
                    required_primary_offset=-abs(int(coordinate_step)),
                )
            except TemplateError:
                valid = False
                break
            transition_reads.append(
                DerivedClockTransitionRead(
                    source_uid=use.source,
                    target_uid=use.target,
                    expr=copy.deepcopy(use.relation.expr),
                    proof=lag,
                )
            )
        if not valid or not transition_reads:
            continue

        try:
            members = tuple(
                _topo_subset(region, uses, contexts={"coordinate"}, zero_only=True)
            )
        except TemplateError:
            continue
        out.append(
            DerivedClockRegionCandidate(
                root_uid=root,
                member_uids=members,
                mapping_expr=mapping_expr,
                clock_proof=clock,
                consumer_uids=tuple(sorted(consumers)),
                transition_reads=tuple(transition_reads),
            )
        )
    return tuple(out)


def _prove_initial_derived_region_safe(
    candidate: DerivedClockRegionCandidate,
    variants: dict[str, Any],
    roles: dict[str, str],
    scalar_barrier: dict[str, int],
    phase: int,
) -> tuple[str, ...]:
    """Prove eager evaluation of the concrete initial derived bucket is safe.

    Region execution is topological and eager on a bucket transition.  Therefore
    every member, not merely the root's currently selected call path, must have a
    safe initial evaluation.  Branches whose condition is decidable from the
    concrete derived coordinate are followed exactly.  Unknown branches are
    conservative and both sides must be safe.

    The proof never executes model code and never borrows observed values.  It
    only follows normalized AST control flow and existing scalar/coordinate roles.
    """

    member_set = set(candidate.member_uids)
    initial_bucket = int(candidate.clock_proof.initial_bucket)
    memo: dict[tuple[str, int], bool] = {}
    visiting: set[tuple[str, int]] = set()
    notes: list[str] = []

    def fixed(node: ast.AST, variable: str, coordinate: int) -> Any:
        return _fixed_coordinate_expr_value(node, variable, coordinate)

    def member_safe(uid: str, coordinate: int) -> bool:
        key = (uid, int(coordinate))
        if key in memo:
            return memo[key]
        if key in visiting:
            return False
        visiting.add(key)
        cv = variants[uid]
        fn = getattr(cv, "function", None)
        if not isinstance(fn, ast.FunctionDef) or len(fn.args.args) != 1:
            visiting.remove(key)
            memo[key] = False
            return False
        variable = fn.args.args[0].arg

        def expr_safe(node: ast.AST) -> bool:
            if isinstance(node, ast.IfExp):
                if not expr_safe(node.test):
                    return False
                try:
                    decision = bool(fixed(node.test, variable, coordinate))
                except Exception:
                    return expr_safe(node.body) and expr_safe(node.orelse)
                return expr_safe(node.body if decision else node.orelse)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                source = node.func.id
                if source in variants:
                    if source in member_set:
                        if len(node.args) != 1 or node.keywords:
                            return False
                        try:
                            called_coordinate = fixed(node.args[0], variable, coordinate)
                        except Exception:
                            return False
                        if (
                            isinstance(called_coordinate, bool)
                            or not isinstance(called_coordinate, int)
                        ):
                            return False
                        return member_safe(source, int(called_coordinate))
                    role = roles.get(source)
                    if role == "scalar":
                        if scalar_barrier.get(source, phase + 1) > phase:
                            return False
                    elif role in {"coordinate", "derived_state", "reduction", "vector"}:
                        return False
                return all(expr_safe(arg) for arg in node.args) and all(
                    expr_safe(kw.value) for kw in node.keywords
                )
            return all(expr_safe(child) for child in ast.iter_child_nodes(node))

        def seq_safe(stmts: list[ast.stmt]) -> tuple[bool, bool]:
            for stmt in stmts:
                if isinstance(stmt, ast.If):
                    if not expr_safe(stmt.test):
                        return False, False
                    try:
                        decision = bool(fixed(stmt.test, variable, coordinate))
                    except Exception:
                        decision = None
                    if decision is True:
                        safe, terminated = seq_safe(stmt.body)
                        if not safe:
                            return False, False
                        if terminated:
                            return True, True
                        continue
                    if decision is False:
                        safe, terminated = seq_safe(stmt.orelse)
                        if not safe:
                            return False, False
                        if terminated:
                            return True, True
                        continue
                    body_safe, body_term = seq_safe(stmt.body)
                    else_safe, else_term = seq_safe(stmt.orelse)
                    if not body_safe or not else_safe:
                        return False, False
                    if body_term and else_term:
                        return True, True
                    continue
                if isinstance(stmt, ast.Return):
                    if stmt.value is None or not expr_safe(stmt.value):
                        return False, False
                    return True, True
                if isinstance(stmt, ast.Assign):
                    if not expr_safe(stmt.value):
                        return False, False
                    continue
                if isinstance(stmt, ast.AnnAssign):
                    if stmt.value is not None and not expr_safe(stmt.value):
                        return False, False
                    continue
                if isinstance(stmt, ast.Expr):
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        continue
                    if not expr_safe(stmt.value):
                        return False, False
                    continue
                # Loops, mutation-heavy control flow and raises are deliberately
                # outside the initial-region proof subset.
                return False, False
            return True, False

        safe, _terminated = seq_safe(list(fn.body))
        visiting.remove(key)
        memo[key] = bool(safe)
        if safe:
            notes.append(
                f"proved initial derived-region member {cv.source_fullname} safe at bucket {coordinate}"
            )
        return bool(safe)

    # Eager topological initialization computes every member at the same initial
    # bucket, so each one must independently pass the branch-aware proof.
    for uid in candidate.member_uids:
        if not member_safe(uid, initial_bucket):
            raise TemplateError(
                "derived-clock region initial bucket is not source-proven safe for eager execution; "
                f"member={variants[uid].source_fullname}; bucket={initial_bucket}"
            )
    return tuple(dict.fromkeys(notes))


_PURE_MAP_SAFE_CALLS = frozenset({
    "abs", "min", "max", "round", "int", "float", "bool",
    "point_input", "global_input", "array_input", "table_input",
    "__is_missing__", "__is_inf__", "__erf__",
    "__step_lookup_1d__", "__interval_lookup_1d__", "__interp_lookup_1d__",
    "__exact_axis_code__", "__sparse_table_lookup__", "__prepared_point_lookup__",
})
_PURE_MAP_SAFE_MODULE_CALLS = frozenset({"exp", "log", "sqrt", "sin", "cos", "floor", "ceil"})
_PURE_MAP_REJECT_NODES = (
    ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match,
    ast.Delete, ast.AugAssign, ast.GeneratorExp, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr,
)


def _pure_map_external_call_supported(call: ast.Call) -> bool:
    if call.keywords:
        return False
    if isinstance(call.func, ast.Name):
        return call.func.id in _PURE_MAP_SAFE_CALLS
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return (
            call.func.value.id in {"math", "np", "numpy"}
            and call.func.attr in _PURE_MAP_SAFE_MODULE_CALLS
            and len(call.args) == 1
        )
    return False


def _prove_executable_pure_maps(
    variants: dict[str, Any],
    *,
    source_pure_map_names: frozenset[str] | set[str] = frozenset(),
    static_scalar_resolver: Callable[[str], int | None] | None = None,
) -> dict[str, ExecutablePureMapSpec]:
    """Prove state-free coordinate variants from the exact specialized closure.

    Admission has two independent gates.  First, the source Cell must already be
    proven ``pure_map`` by the source ValueSemantics analysis (except a scalar with
    a stronger run-key StaticScalarFact).  Second, the exact specialized canonical
    body must fit the backend-safe, state-free subset.  The source gate prevents an
    acyclic StateDerivedMap from being mistaken for a PureMap merely because its
    recurrence leaves sit elsewhere in the canonical graph.

    Cycles, reductions/vectors, unresolved helper calls, statement
    loops/comprehensions and any dependency on a non-state-free canonical Cell fail
    closed.  Scalar Cells become explicit read-only helper arguments in native
    direct-local mode.
    """
    known = set(variants)
    memo: dict[str, bool] = {}
    active: set[str] = set()
    scalar_closure: dict[str, set[str]] = {}
    pure_closure: dict[str, set[str]] = {}

    def prove(uid: str) -> bool:
        if uid in memo:
            return memo[uid]
        if uid in active:
            return False
        cv = variants[uid]
        role = getattr(cv, "role", getattr(cv, "kind", "scalar"))
        fn = getattr(cv, "function", None)
        if role not in {"scalar", "coordinate"} or fn is None:
            memo[uid] = False
            return False
        if role == "scalar" and fn.args.args:
            memo[uid] = False
            return False
        if role == "scalar" and static_scalar_resolver is not None:
            try:
                static_value = static_scalar_resolver(uid)
            except Exception:
                static_value = None
            if static_value is not None:
                memo[uid] = True
                scalar_closure[uid] = set()
                pure_closure[uid] = set()
                return True
        source_name = str(getattr(cv, "source_name", ""))
        if source_name not in source_pure_map_names:
            memo[uid] = False
            return False
        if role == "coordinate" and len(fn.args.args) != 1:
            memo[uid] = False
            return False

        active.add(uid)
        direct_scalars: set[str] = set()
        direct_pure: set[str] = set()
        ok = True
        for node in ast.walk(fn):
            if isinstance(node, _PURE_MAP_REJECT_NODES):
                ok = False
                break
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in known:
                dep = node.func.id
                if not prove(dep):
                    ok = False
                    break
                dep_role = getattr(variants[dep], "role", getattr(variants[dep], "kind", "scalar"))
                if dep_role == "scalar":
                    direct_scalars.add(dep)
                elif dep_role == "coordinate":
                    direct_pure.add(dep)
                else:
                    ok = False
                    break
                direct_scalars.update(scalar_closure.get(dep, ()))
                direct_pure.update(pure_closure.get(dep, ()))
            elif not _pure_map_external_call_supported(node):
                ok = False
                break
        active.remove(uid)
        memo[uid] = bool(ok)
        if ok:
            scalar_closure[uid] = direct_scalars
            pure_closure[uid] = direct_pure
        return bool(ok)

    specs: dict[str, ExecutablePureMapSpec] = {}
    for uid, cv in variants.items():
        if getattr(cv, "role", getattr(cv, "kind", "scalar")) != "coordinate":
            continue
        if not prove(uid):
            continue
        fn = cv.function
        assert fn is not None and len(fn.args.args) == 1
        specs[uid] = ExecutablePureMapSpec(
            uid=uid,
            coordinate_parameter=fn.args.args[0].arg,
            scalar_dependencies=tuple(sorted(scalar_closure.get(uid, ()))),
            pure_map_dependencies=tuple(sorted(pure_closure.get(uid, ()))),
        )
    return specs


def prove_executable_pure_maps(
    variants: dict[str, Any],
    *,
    source_pure_map_names: frozenset[str] | set[str] = frozenset(),
    static_scalar_resolver: Callable[[str], int | None] | None = None,
) -> dict[str, ExecutablePureMapSpec]:
    """Public proof entry point shared by the pre-scheduler semantic bridge.

    Keeping one implementation prevents the observational canonical graph from
    silently inventing a broader PureMap subset than the authoritative executable
    scheduler.  This function grants only the V02346 two-gate semantic proof; it
    does not itself authorize scheduling, storage or code generation.
    """
    return _prove_executable_pure_maps(
        variants,
        source_pure_map_names=source_pure_map_names,
        static_scalar_resolver=static_scalar_resolver,
    )


def build_executable_graph(
    ir,
    variants: dict[str, Any],
    output_uid: str,
    *,
    coordinate_domains: dict[str, tuple[int, int]] | None = None,
    proof_static_scalar_resolver: Callable[[str], int | None] | None = None,
    source_pure_map_names: frozenset[str] | set[str] = frozenset(),
    recovered_loop_evidence: Any | None = None,
    recovered_loop_evidence_provider: Callable[[], Any] | None = None,
    canonical_execution_schedule: Any | None = None,
) -> ExecutableGraph:
    """Build the executable schedule from GraphIR + statically lowered formulas.

    ``variants`` are frontend canonical variants.  Only the small structural
    interface (``role``, ``function``, ``reduction``) is used to avoid a module
    cycle and to keep the schedule independent from legacy planner stages.
    """
    canonical_authority = bool(
        canonical_execution_schedule is not None
        and bool(getattr(canonical_execution_schedule, "eligible", False))
    )
    authority_contract_uid = (
        str(getattr(canonical_execution_schedule, "uid", "")) or None
        if canonical_execution_schedule is not None else None
    )
    authority_note = None
    if canonical_execution_schedule is not None:
        if canonical_authority:
            authority_note = (
                "canonical schedule authority active: "
                f"{authority_contract_uid or '<unidentified-contract>'}"
            )
        else:
            blockers = tuple(getattr(canonical_execution_schedule, "capability_blockers", ()))
            authority_note = (
                "canonical schedule authority ineligible; legacy schedule retained: "
                + ("; ".join(str(x) for x in blockers) if blockers else "contract not eligible")
            )

    roles = {uid: getattr(cv, "role", getattr(cv, "kind", "scalar")) for uid, cv in variants.items()}
    pure_maps = _prove_executable_pure_maps(
        variants,
        source_pure_map_names=source_pure_map_names,
        static_scalar_resolver=proof_static_scalar_resolver,
    )
    for uid in pure_maps:
        roles[uid] = "pure_map"
    coordinate_domains = coordinate_domains or {}
    proof_static_scalar_values: dict[str, int] = {}
    known = set(variants)
    uses: list[TemplateUse] = []

    for uid, cv in variants.items():
        role = roles[uid]
        if role == "reduction":
            r = cv.reduction
            if r is None:
                raise TemplateError(f"reduction role {uid} has no normalized reduction specification")
            nodes = [r.init, r.body_expr, *r.filters, *r.range_args]
            for call in _walk_calls(nodes, known):
                source = call.func.id
                relation: CoordinateRelation | None = None
                if roles.get(source) == "coordinate":
                    if len(call.args) != 1:
                        raise TemplateError(f"coordinate family {source} must receive exactly one coordinate expression")
                    off = _offset(call.args[0], r.loop_var)
                    if off is None:
                        raise TemplateError(f"dynamic coordinate expression in reduction {uid}: {ast.unparse(call.args[0])}")
                    relation = AffineOffset(int(off))
                uses.append(TemplateUse(source, uid, "reduction", relation))
            continue

        fn = cv.function
        if fn is None or role == "vector":
            continue
        context = "coordinate" if role == "coordinate" else "scalar"
        for call in _walk_calls(fn.body, known):
            source = call.func.id
            relation: CoordinateRelation | None = None
            if roles.get(source) == "coordinate":
                if len(call.args) != 1:
                    raise TemplateError(f"coordinate family {source} must receive exactly one coordinate expression")
                if role == "coordinate":
                    off = _offset(call.args[0], "t")
                    if off is None:
                        # Presence is not permission: the loop-domain pass below
                        # must attach a CausalHistoryProof before this relation can
                        # affect storage or code generation.
                        relation = DerivedHistoryRead(copy.deepcopy(call.args[0]), proof=None)
                    else:
                        relation = AffineOffset(int(off))
                else:
                    # Fixed-coordinate calls should already have been inlined by
                    # the frontend.  A surviving dynamic coordinate read would
                    # need explicit history addressing and is conservatively rejected.
                    raise TemplateError(f"forward/backward executable template cannot schedule scalar operation {uid} with a non-inlined coordinate call to {source}")
            uses.append(TemplateUse(source, uid, context, relation))

    coord = sorted(uid for uid, role in roles.items() if role == "coordinate")
    reductions = sorted(uid for uid, role in roles.items() if role == "reduction")
    scalars = sorted(uid for uid, role in roles.items() if role == "scalar")
    coord_phase = {uid: 0 for uid in coord}
    red_phase = {uid: 0 for uid in reductions}
    scalar_barrier = {uid: 0 for uid in scalars}

    incoming: dict[str, list[TemplateUse]] = defaultdict(list)
    for use in uses:
        incoming[use.target].append(use)

    recovered_evidence_cache = recovered_loop_evidence
    recovered_evidence_loaded = recovered_loop_evidence is not None

    def recovered_evidence():
        nonlocal recovered_evidence_cache, recovered_evidence_loaded
        if not recovered_evidence_loaded and recovered_loop_evidence_provider is not None:
            recovered_evidence_cache = recovered_loop_evidence_provider()
            recovered_evidence_loaded = True
        return recovered_evidence_cache

    for _ in range(max(20, 8 * len(variants))):
        changed = False
        for uid in coord:
            required = 0
            for use in incoming.get(uid, ()):
                sr = roles.get(use.source)
                if sr == "scalar":
                    required = max(required, scalar_barrier[use.source])
                elif sr == "reduction":
                    required = max(required, red_phase[use.source] + 1)
                elif sr == "coordinate":
                    required = max(required, coord_phase[use.source])
            if required > coord_phase[uid]:
                coord_phase[uid] = required
                changed = True

        for uid in reductions:
            required = 0
            for use in incoming.get(uid, ()):
                sr = roles.get(use.source)
                if sr == "coordinate":
                    required = max(required, coord_phase[use.source])
                elif sr == "scalar":
                    required = max(required, scalar_barrier[use.source])
                elif sr == "reduction":
                    required = max(required, red_phase[use.source])
            if required > red_phase[uid]:
                red_phase[uid] = required
                changed = True

        for uid in scalars:
            required = 0
            for use in incoming.get(uid, ()):
                sr = roles.get(use.source)
                if sr == "coordinate":
                    required = max(required, coord_phase[use.source] + 1)
                elif sr == "reduction":
                    required = max(required, red_phase[use.source] + 1)
                elif sr == "scalar":
                    required = max(required, scalar_barrier[use.source])
            if required > scalar_barrier[uid]:
                scalar_barrier[uid] = required
                changed = True
        if not changed:
            break
    else:
        raise TemplateError("dependency-barrier analysis did not converge; graph requires fallback")

    pre_scalars = _topo_subset([u for u in scalars if scalar_barrier[u] == 0], uses, contexts={"scalar"})

    # A model whose coordinate-valued dependencies are all proven PureMaps has no
    # carried time state to schedule.  The reduction range itself is then the exact
    # executable iteration domain.  Keep one state-free loop scaffold so the shared
    # Python/Cython reduction machinery can execute without manufacturing a fake
    # coordinate family or storage slot.
    if not coord:
        if not reductions:
            raise TemplateError(
                "no coordinate families or reductions were proven for the executable template backend"
            )
        if any(red_phase[uid] != 0 for uid in reductions):
            raise TemplateError(
                "state-free PureMap reduction execution currently requires all reductions at barrier 0"
            )
        reduction_edges = [
            use for use in uses
            if use.context == "reduction" and roles.get(use.source) == "reduction"
        ]
        if reduction_edges:
            raise TemplateError(
                "state-free PureMap reduction execution does not yet schedule reduction-to-reduction dependencies"
            )
        reds = _topo_subset(reductions, uses, contexts={"reduction"})
        ranges = [variants[uid].reduction.range_args for uid in reds if variants[uid].reduction is not None]
        if len(ranges) != len(reds) or not ranges:
            raise TemplateError("state-free PureMap reduction has no exact normalized range")
        sig0 = _ast_sig(ranges[0])
        if any(_ast_sig(row) != sig0 for row in ranges[1:]):
            raise TemplateError(
                "state-free PureMap reductions currently require one shared exact range"
            )
        range_args = tuple(ranges[0])
        coordinate_step = _validate_range(range_args)
        post = [u for u in scalars if scalar_barrier[u] == 1]
        post = _topo_subset(post, uses, contexts={"scalar"}) if post else []
        if any(scalar_barrier[uid] > 1 for uid in scalars):
            raise TemplateError(
                "state-free PureMap reduction execution encountered a scalar barrier beyond the reduction loop"
            )
        fusable: list[str] = []
        fusion_reasons: dict[str, str] = {}
        for rid in reds:
            bad = []
            for use in uses:
                if use.target != rid:
                    continue
                sr = roles.get(use.source)
                if sr == "scalar" and scalar_barrier.get(use.source, 0) != 0:
                    bad.append(f"scalar source {use.source} is not available before the reduction")
                elif sr not in {"scalar", "pure_map"}:
                    bad.append(f"source {use.source} has execution role {sr}")
            if bad:
                fusion_reasons[rid] = "; ".join(bad)
            else:
                fusable.append(rid)
                fusion_reasons[rid] = (
                    "state-free PureMap reduction over its exact normalized range; "
                    "source traversal and reduction order are unchanged"
                )
        state_free_scan_direction = 1
        if canonical_authority:
            stages = tuple(getattr(canonical_execution_schedule, "stages", ()))
            if len(stages) != 1 or int(getattr(stages[0], "index", -1)) != 0:
                raise TemplateError(
                    "canonical schedule authority contract is outside the one-stage lowering subset"
                )
            semantic_scan = str(getattr(canonical_execution_schedule, "effective_scan_direction", "any"))
            if semantic_scan not in {"any", "ascending"}:
                raise TemplateError(
                    "canonical schedule authority disagreement: state-free reduction backend "
                    f"uses ascending traversal but canonical contract requires {semantic_scan}"
                )
        loop = ExecutableLoopTemplate(
            uid=_digest(["pure_map_reduction", *reds, *_range_signature(range_args)], "exec_loop"),
            index=0,
            coordinate_symbol="t",
            range_args=range_args,
            coordinate_step=coordinate_step,
            scan_direction=state_free_scan_direction,
            families=(),
            reductions=tuple(reds),
            post_scalars=tuple(post),
            evidence_loop_uids=(),
            generalization="state_free_pure_map_reduction",
            guard_notes=(
                "exact normalized reduction range; every coordinate-valued dependency is an executable PureMap",
            ),
            max_lag_by_family={},
            current_slots={},
            current_slot_count=0,
            fusable_reductions=tuple(fusable),
            fusion_reasons=fusion_reasons,
        )
        scheduled_scalars = set(pre_scalars) | set(post)
        missing_scalars = [u for u in scalars if u not in scheduled_scalars]
        if missing_scalars:
            raise TemplateError(
                "scalar operations were not assigned to the state-free PureMap reduction barrier: "
                f"{missing_scalars!r}"
            )
        if output_uid not in roles or roles[output_uid] not in ("scalar", "reduction"):
            raise TemplateError("current portfolio API requires a scalar/reduction output")
        graph = ExecutableGraph(
            pre_scalars=tuple(pre_scalars),
            loops=(loop,),
            output_uid=output_uid,
            roles=roles,
            pure_maps=pure_maps,
            uses=tuple(uses),
            coordinate_phase=coord_phase,
            scalar_barrier=scalar_barrier,
            reduction_phase=red_phase,
            derived_clock_states={},
            derived_clock_regions={},
            boundary_seeds={},
            formula_hash=ir.formula_hash,
            diagnostics=[
                "state-free PureMap reduction executed directly from its exact normalized range",
                *(() if authority_note is None else (authority_note,)),
            ],
            schedule_authority="canonical" if canonical_authority else "legacy",
            schedule_authority_contract_uid=authority_contract_uid if canonical_authority else None,
        )
        graph.common_range_args
        return graph

    max_phase = max(coord_phase.values(), default=0)
    if canonical_authority:
        stages = tuple(getattr(canonical_execution_schedule, "stages", ()))
        if len(stages) != 1 or int(getattr(stages[0], "index", -1)) != 0:
            raise TemplateError(
                "canonical schedule authority contract is outside the one-stage lowering subset"
            )
        contract_persistent = set(getattr(canonical_execution_schedule, "persistent_uids", ()))
        missing_persistent = sorted(contract_persistent - set(coord))
        if missing_persistent:
            raise TemplateError(
                "canonical schedule authority persistent members are absent from the backend coordinate set: "
                f"{missing_persistent!r}"
            )
        # Canonical stages express completion/materialization boundaries between
        # persistent components.  The legacy ABI may subdivide one canonical stage
        # into multiple coordinate phases for scalar/reduction ordering.  That is a
        # lowering detail, provided one canonical persistent SCC is never split
        # across those phases.
        for component in tuple(getattr(canonical_execution_schedule, "components", ())):
            phases = {
                int(coord_phase[uid]) for uid in getattr(component, "persistent_uids", ())
                if uid in coord_phase
            }
            if len(phases) > 1:
                raise TemplateError(
                    "canonical schedule authority component was split across legacy executable phases: "
                    f"{getattr(component, 'uid', '<component>')} -> {sorted(phases)!r}"
                )

    loops: list[ExecutableLoopTemplate] = []
    diagnostics: list[str] = []
    derived_clock_states: dict[str, DerivedClockState] = {}
    derived_clock_regions: dict[str, DerivedClockRegion] = {}
    boundary_seeds: dict[str, BoundarySeed] = {}
    derived_clock_signature: tuple[int, int, int] | None = None

    for phase in range(max_phase + 1):
        phase_fam = [u for u in coord if coord_phase[u] == phase]
        if not phase_fam:
            continue
        phase_famset = set(phase_fam)
        reds = [u for u in reductions if red_phase[u] == phase]
        # Reductions may depend on other reductions in the same barrier; preserve
        # that order for standalone execution.
        reds = _topo_subset(reds, uses, contexts={"reduction"}) if reds else []
        post = [u for u in scalars if scalar_barrier[u] == phase + 1]
        post = _topo_subset(post, uses, contexts={"scalar"}) if post else []

        ranges = [variants[u].reduction.range_args for u in reds if variants[u].reduction is not None]
        if not ranges:
            raise TemplateError(f"coordinate loop {phase} has no exact consumer range from which to prove its runtime domain")
        sig0 = _ast_sig(ranges[0])
        if any(_ast_sig(r) != sig0 for r in ranges[1:]):
            raise TemplateError(f"coordinate loop {phase} has incompatible reduction domains")
        range_args = tuple(ranges[0])
        coordinate_step = _validate_range(range_args)

        # Retry ordinary affine recognition after proof-only helper/local alias
        # expansion.  This turns calls such as ``state(alias(t))`` back into the
        # normal offset scheduler path without rewriting canonical formula source.
        for idx, use in enumerate(tuple(uses)):
            if not (
                use.context == "coordinate"
                and use.source in phase_famset
                and use.target in phase_famset
                and isinstance(use.relation, DerivedHistoryRead)
            ):
                continue
            proof_expr = _expanded_coordinate_expr(use, variants)
            alias_off = _offset(proof_expr, "t")
            if alias_off is not None:
                uses[idx] = replace(
                    use,
                    relation=AffineAliasRead(
                        expr=copy.deepcopy(use.relation.expr),
                        offset=int(alias_off),
                        normalized_expr=ast.unparse(proof_expr),
                    ),
                )

        # Detect one restricted secondary clock carried inside this primary loop.
        # A candidate must be a pure coordinate recurrence over its own y-1 value,
        # depend on no primary coordinate family, and be consumed only through one
        # monotone floor-bucket mapping of the primary coordinate.
        phase_candidates: list[tuple[str, list[int], ast.AST, DerivedClockProof]] = []
        for source in phase_fam:
            self_uses = [
                u for u in uses
                if u.context == "coordinate" and u.source == source and u.target == source
            ]
            if len(self_uses) != 1 or self_uses[0].offset != -1:
                continue
            incoming_coord = [
                u for u in uses
                if u.target == source
                and u.context == "coordinate"
                and u.source != source
                and roles.get(u.source) == "coordinate"
            ]
            if incoming_coord:
                continue
            incoming_nonself = [u for u in uses if u.target == source and u.source != source]
            if any(
                roles.get(u.source) != "scalar"
                or scalar_barrier.get(u.source, phase + 1) > phase
                for u in incoming_nonself
            ):
                continue
            external_indices = [
                i for i, u in enumerate(uses)
                if u.source == source and u.target != source
            ]
            if not external_indices:
                continue
            if any(
                uses[i].context != "coordinate"
                or not isinstance(uses[i].relation, DerivedHistoryRead)
                for i in external_indices
            ):
                continue
            mapping_expr: ast.AST | None = None
            proof: DerivedClockProof | None = None
            signature: tuple[int, int, int] | None = None
            valid = True
            for i in external_indices:
                use = uses[i]
                expanded = _expanded_coordinate_expr(use, variants)
                try:
                    candidate_proof = _prove_derived_clock_mapping(
                        expanded, "t", range_args, coordinate_step
                    )
                except TemplateError:
                    valid = False
                    break
                sig = (
                    int(candidate_proof.period),
                    int(candidate_proof.shift),
                    int(candidate_proof.bias),
                )
                if signature is None:
                    signature = sig
                    mapping_expr = copy.deepcopy(expanded)
                    proof = candidate_proof
                elif sig != signature:
                    valid = False
                    break
            if valid and mapping_expr is not None and proof is not None:
                phase_candidates.append((source, external_indices, mapping_expr, proof))

        if len(phase_candidates) > 1:
            candidate_sigs = {
                (p.period, p.shift, p.bias) for _u, _idx, _expr, p in phase_candidates
            }
            if len(candidate_sigs) > 1:
                raise TemplateError(
                    "multiple independent derived clocks are outside the loop-carried derived-state subset"
                )
        derived_in_phase: set[str] = set()
        for source, external_indices, mapping_expr, proof in phase_candidates:
            sig = (proof.period, proof.shift, proof.bias)
            if derived_clock_signature is None:
                derived_clock_signature = sig
            elif sig != derived_clock_signature:
                raise TemplateError(
                    "more than one derived clock mapping is outside the current scheduler subset"
                )
            consumer_uids = tuple(sorted({uses[i].target for i in external_indices}))
            for i in external_indices:
                use = uses[i]
                assert isinstance(use.relation, DerivedHistoryRead)
                uses[i] = replace(
                    use,
                    relation=DerivedClockRead(copy.deepcopy(use.relation.expr), proof),
                )
            roles[source] = "derived_state"
            derived_clock_states[source] = DerivedClockState(
                uid=source,
                phase=phase,
                mapping_expr=mapping_expr,
                proof=proof,
                self_offset=-1,
                consumer_uids=consumer_uids,
            )
            derived_in_phase.add(source)
            diagnostics.append(
                f"loop-carried derived clock {variants[source].source_fullname}: {proof.note}"
            )

        fam = [u for u in phase_fam if u not in derived_in_phase]
        if not fam:
            raise TemplateError(
                "derived-clock state currently requires at least one primary coordinate family in the same loop"
            )
        # Recognize one acyclic cross-clock region before the ordinary
        # dynamic-history pass sees its boundary mapping.  Ownership and
        # transition-lag algebra were proven in the preceding foundation; this
        # step promotes that evidence into an executable schedule descriptor.
        region_candidates = analyze_derived_clock_regions(
            fam,
            uses,
            variants,
            range_args,
            coordinate_step,
        )
        if region_candidates:
            if len(region_candidates) != 1:
                raise TemplateError(
                    "multiple derived-clock regions are outside the first executable cross-clock subset"
                )
            candidate = region_candidates[0]
            if derived_in_phase:
                raise TemplateError(
                    "loop-carried derived state and cross-clock region cannot share one phase in the first executable subset"
                )
            sig = (
                candidate.clock_proof.period,
                candidate.clock_proof.shift,
                candidate.clock_proof.bias,
            )
            if derived_clock_signature is None:
                derived_clock_signature = sig
            elif sig != derived_clock_signature:
                raise TemplateError(
                    "more than one derived clock mapping is outside the current scheduler subset"
                )

            member_set = set(candidate.member_uids)
            # The executable region is acyclic: every internal coordinate edge is
            # same-bucket offset zero.  Recursive annual state remains owned by the
            # older DerivedClockState path rather than being silently generalized.
            bad_internal = [
                use
                for use in uses
                if use.context == "coordinate"
                and use.source in member_set
                and use.target in member_set
                and use.offset != 0
            ]
            if bad_internal:
                raise TemplateError(
                    "cross-clock region contains nonzero/recursive same-clock dependency; fallback required"
                )

            initial_notes = _prove_initial_derived_region_safe(
                candidate, variants, roles, scalar_barrier, phase
            )
            persistent = tuple(
                uid
                for uid in candidate.member_uids
                if any(
                    use.context == "coordinate"
                    and use.source == uid
                    and use.target not in member_set
                    for use in uses
                )
            )
            if not persistent:
                raise TemplateError("cross-clock region has no persistent value consumed by the primary graph")
            transient = tuple(uid for uid in candidate.member_uids if uid not in persistent)

            # Outside consumers now read a carried current-bucket value.  The
            # source AST remains unchanged; only the typed relation is upgraded.
            for idx, use in enumerate(tuple(uses)):
                if not (
                    use.context == "coordinate"
                    and use.source in member_set
                    and use.target not in member_set
                ):
                    continue
                if use.source not in persistent or not isinstance(use.relation, DerivedHistoryRead):
                    raise TemplateError(
                        "derived-region value escapes without a persistent proven clock read"
                    )
                uses[idx] = replace(
                    use,
                    relation=DerivedClockRead(
                        copy.deepcopy(use.relation.expr), candidate.clock_proof
                    ),
                )

            for uid in candidate.member_uids:
                roles[uid] = "derived_region"
            region = DerivedClockRegion(
                root_uid=candidate.root_uid,
                phase=phase,
                member_uids=candidate.member_uids,
                persistent_uids=persistent,
                transient_uids=transient,
                mapping_expr=copy.deepcopy(candidate.mapping_expr),
                clock_proof=candidate.clock_proof,
                consumer_uids=candidate.consumer_uids,
                transition_reads=candidate.transition_reads,
                initial_safety_notes=initial_notes,
            )
            derived_clock_regions[region.root_uid] = region
            diagnostics.append(
                "executable derived-clock region %s: members=%s; %s"
                % (
                    variants[region.root_uid].source_fullname,
                    ",".join(variants[uid].source_name for uid in region.member_uids),
                    "; ".join(row.proof.note for row in region.transition_reads),
                )
            )
            fam = [uid for uid in fam if uid not in member_set]

        if not fam:
            raise TemplateError(
                "derived-clock execution currently requires a primary coordinate family in the same loop"
            )
        famset = set(fam)

        # A strict-past dynamic edge must be classified before same-iteration
        # topology is built, otherwise it can manufacture a false cycle.  This
        # pre-pass is opportunistic: expressions outside the strict bounded-past
        # subset retain their conservative same-iteration edge and are diagnosed by
        # the ordinary history-proof pass after topology succeeds.
        derived_notes: list[str] = []
        dynamic_history_indices = [
            idx
            for idx, use in enumerate(uses)
            if (
                use.context == "coordinate"
                and use.source in famset
                and use.target in famset
                and use.coordinate_expr is not None
            )
        ]
        for idx in dynamic_history_indices:
            use = uses[idx]
            rel = use.derived_history
            assert rel is not None
            proof_expr = _expanded_coordinate_expr(use, variants)
            try:
                proof = _prove_strict_bounded_past_coordinate(
                    proof_expr,
                    "t",
                    coordinate_domains.get(use.target),
                    range_args,
                    coordinate_step,
                    static_scalar_values=proof_static_scalar_values,
                    static_scalar_resolver=proof_static_scalar_resolver,
                )
            except TemplateError:
                continue
            uses[idx] = replace(
                use,
                relation=DerivedHistoryRead(copy.deepcopy(rel.expr), proof=proof),
            )
            derived_notes.append(
                f"strict bounded history read {variants[use.source].source_fullname} -> "
                f"{variants[use.target].source_fullname}: {proof.note}"
            )

        order = _topo_subset(fam, uses, contexts={"coordinate"}, zero_only=True)

        # All surviving dynamic reads still need a typed causal-history proof for
        # storage/codegen.  Strict proofs from the pre-pass are retained; unresolved
        # reads may use the older period-floor past-or-same snapshot proof.
        for idx in dynamic_history_indices:
            use = uses[idx]
            rel = use.derived_history
            assert rel is not None
            if rel.proof is not None:
                continue
            # Preserve the historical fallback proof and its diagnostics exactly.
            # The new local/caller-domain expansion is permission only for the
            # strict-past pre-pass; failing that proof must not silently broaden or
            # cosmetically rewrite the older causal-snapshot boundary.
            proof_expr = _expand_coordinate_proof_aliases(rel.expr, variants)
            try:
                proof = _prove_past_or_same_derived_coordinate(
                    proof_expr, "t", range_args, coordinate_step
                )
            except TemplateError as exc:
                raise TemplateError(
                    f"{exc}; read={variants[use.source].source_fullname} -> "
                    f"{variants[use.target].source_fullname}"
                ) from exc
            uses[idx] = replace(
                use,
                relation=DerivedHistoryRead(copy.deepcopy(rel.expr), proof=proof),
            )
            derived_notes.append(
                f"causal history read {variants[use.source].source_fullname} -> "
                f"{variants[use.target].source_fullname}: {proof.note}"
            )
        dynamic_history_uses = [uses[idx] for idx in dynamic_history_indices]

        # Dependencies are expressed in actual coordinate units by the formula AST.
        # Normalize them to physical iteration positions using the exact range step.
        iteration_shifts: set[int] = set()
        for use in uses:
            if not (
                use.context == "coordinate" and use.source in famset and use.target in famset
                and use.offset is not None and use.offset != 0
            ):
                continue
            if use.offset % coordinate_step != 0:
                raise TemplateError(
                    f"coordinate dependency offset {use.offset} is not aligned to range step {coordinate_step}; fallback required"
                )
            iteration_shifts.add(use.offset // coordinate_step)
        for region in derived_clock_regions.values():
            if region.phase != phase:
                continue
            for read in region.transition_reads:
                if read.proof.primary_offset % coordinate_step != 0:
                    raise TemplateError(
                        "derived-region transition lag is not aligned to physical coordinate step"
                    )
                iteration_shifts.add(read.proof.primary_offset // coordinate_step)
        has_negative = any(shift < 0 for shift in iteration_shifts)
        has_positive = any(shift > 0 for shift in iteration_shifts)
        if dynamic_history_uses and has_positive:
            raise TemplateError(
                f"executable loop {phase} mixes descending/future offset dependencies with an ascending causal history read"
            )
        if has_negative and has_positive:
            raise TemplateError(
                f"executable loop {phase} requires both ascending and descending causal scans; fallback required"
            )
        legacy_scan_direction = -1 if has_positive else 1
        scan_direction = legacy_scan_direction
        if canonical_authority:
            contract_persistent = set(getattr(canonical_execution_schedule, "persistent_uids", ()))
            persistent_in_phase = contract_persistent.intersection(famset)
            if persistent_in_phase:
                semantic_scan = str(getattr(canonical_execution_schedule, "effective_scan_direction", "any"))
                if semantic_scan == "ascending":
                    canonical_scan_direction = 1
                elif semantic_scan == "descending":
                    canonical_scan_direction = -1
                elif semantic_scan == "any":
                    # ``any`` proves that either traversal is valid; choose the
                    # backend's deterministic ascending policy explicitly.
                    canonical_scan_direction = 1
                else:
                    raise TemplateError(
                        f"canonical schedule authority has unsupported scan direction {semantic_scan!r}"
                    )
                if canonical_scan_direction != legacy_scan_direction:
                    raise TemplateError(
                        "canonical schedule authority disagreement: canonical persistent scan "
                        f"{canonical_scan_direction:+d} != legacy structural witness "
                        f"{legacy_scan_direction:+d} for executable loop {phase}"
                    )
                scan_direction = canonical_scan_direction

        evidence = []
        notes = list(derived_notes)
        policies = []
        for observed in ir.loop_templates:
            covered = famset.intersection(observed.coordinate_positions)
            if covered and int(getattr(observed, "coordinate_stride", 1)) == abs(coordinate_step):
                evidence.append(observed.uid)
                policies.append(observed.generalization)
                notes.extend(observed.proof_notes)
        recovered = recovered_evidence() if not evidence else None
        if not evidence and recovered is not None:
            candidate_fullnames = {variants[uid].source_fullname for uid in fam}
            canonical_family_signature = _canonical_guarded_family_signature(variants, fam)
            # If this loop contains a derived-coordinate history read, the recovered
            # witness must contain both the state being read and its consuming formula.
            # A random repeated formula elsewhere in the same source-level family is
            # not sufficient evidence for the new scheduling operation. The source
            # family still needs the requested stride match through ``matches``.
            required_history_fullnames = {
                variants[uid].source_fullname
                for use in dynamic_history_uses
                for uid in (use.source, use.target)
            }
            required_history_pairs = {
                (
                    variants[use.source].source_fullname,
                    variants[use.target].source_fullname,
                )
                for use in dynamic_history_uses
            }
            for match in recovered.matching_witnesses(
                candidate_fullnames,
                abs(coordinate_step),
                required_fullnames=required_history_fullnames,
                required_history_pairs=required_history_pairs,
                canonical_family_signature=canonical_family_signature,
            ):
                witness = match.witness
                matches = match.matched_fullnames
                evidence.append(witness.uid)
                policies.append("static_formula_guarded")
                notes.append(
                    "exact recovered %s loop evidence: block=%d span=%d:%d repetitions=%d "
                    "phases=%d matched=%s grammar=%s family=%s coverage=%s"
                    % (
                        witness.kind,
                        witness.block_index,
                        witness.operation_start,
                        witness.operation_stop,
                        witness.repetitions,
                        witness.phase_count,
                        ",".join(matches),
                        witness.grammar_signature,
                        match.canonical_family_signature,
                        match.range_summary,
                    )
                )
        if not evidence:
            raise TemplateError(f"coordinate loop {phase} has no repeated concrete-topology evidence")
        generalization = "static_formula_guarded" if policies and all(p == "static_formula_guarded" for p in policies) else "observed_only"

        max_lag: dict[str, int] = defaultdict(int)
        for use in uses:
            if (
                use.context == "coordinate"
                and use.source in famset
                and use.target in famset
                and use.offset is not None
                and use.offset != 0
            ):
                if use.offset % coordinate_step != 0:
                    raise TemplateError(
                        f"coordinate dependency offset {use.offset} is not aligned to range step {coordinate_step}"
                    )
                # Ring requirement is measured in physical iteration distance.
                max_lag[use.source] = max(max_lag[use.source], abs(use.offset // coordinate_step))
        for region in derived_clock_regions.values():
            if region.phase != phase:
                continue
            for read in region.transition_reads:
                if read.source_uid not in famset:
                    raise TemplateError(
                        "derived-region transition source is not produced by the primary loop phase"
                    )
                if read.proof.primary_offset % coordinate_step != 0:
                    raise TemplateError(
                        "derived-region transition lag is not aligned to range step"
                    )
                max_lag[read.source_uid] = max(
                    max_lag[read.source_uid],
                    abs(read.proof.primary_offset // coordinate_step),
                )

        # Prove one-step pre-range boundary values directly from source.  This
        # avoids the historical implicit-zero assumption for p=0 lag reads while
        # keeping arbitrary recursive prehistory outside the native subset.
        if scan_direction == 1:
            range_start, _range_stop, _range_step = _range_parts(range_args)
            start_value = _literal_int(range_start)
            if start_value is None:
                # The frontend may have proved a zero-argument integer scalar
                # invariant over the entire run-key domain.  Reuse that same proof
                # for boundary geometry rather than falling back to the historical
                # implicit-zero prehistory when the range starts at ``start()``.
                try:
                    start_value = _eval_proof_int_expr(
                        range_start,
                        "t",
                        0,
                        static_scalar_values=proof_static_scalar_values,
                        static_scalar_resolver=proof_static_scalar_resolver,
                    )
                except TemplateError:
                    start_value = None
            if start_value is not None:
                seed_coordinate = int(start_value - coordinate_step)
                for uid in fam:
                    if max_lag.get(uid, 0) != 1:
                        continue
                    fn = variants[uid].function
                    if fn is None:
                        continue
                    known_calls = _walk_calls((fn,), set(roles))
                    if any(
                        roles.get(call.func.id) == "scalar"
                        and scalar_barrier.get(call.func.id, phase + 1) > phase
                        for call in known_calls
                    ):
                        continue
                    if _fixed_coordinate_seed_safe(fn, seed_coordinate, roles):
                        boundary_seeds[uid] = BoundarySeed(
                            uid=uid,
                            phase=phase,
                            coordinate=seed_coordinate,
                            note=(
                                f"source-proven one-step boundary seed at coordinate "
                                f"{seed_coordinate}; reachable base path requires no coordinate state"
                            ),
                        )
                        diagnostics.append(
                            f"boundary seed {variants[uid].source_fullname}: "
                            f"coordinate {seed_coordinate} is source-state-free"
                        )

        fusable: list[str] = []
        fusion_reasons: dict[str, str] = {}
        fusable_sources: set[str] = set()
        for rid in reds:
            r = variants[rid].reduction
            if scan_direction != 1:
                fusion_reasons[rid] = (
                    "producing loop scans descending while the normalized reduction preserves an ascending range; "
                    "fusion would change reduction order"
                )
                continue
            if r is None:
                fusion_reasons[rid] = "missing normalized reduction"
                continue
            if _range_signature(r.range_args) != _range_signature(range_args):
                fusion_reasons[rid] = "reduction domain differs from producing loop"
                continue
            ruses = [u for u in uses if u.target == rid]
            bad = []
            for use in ruses:
                sr = roles.get(use.source)
                if sr == "coordinate":
                    if coord_phase[use.source] > phase or use.offset != 0:
                        bad.append(f"coordinate source {use.source} is not an offset-0 value available by this loop")
                elif sr == "reduction":
                    bad.append(f"depends on reduction {use.source}")
                elif sr == "scalar" and scalar_barrier[use.source] > phase:
                    bad.append(f"scalar source {use.source} is not available before loop")
            if bad:
                fusion_reasons[rid] = "; ".join(bad)
                continue
            fusable.append(rid)
            fusable_sources.update(
                u.source for u in ruses
                if roles.get(u.source) == "coordinate" and coord_phase[u.source] == phase
            )
            fusion_reasons[rid] = (
                "same ordered range; body/filter use offset-0 coordinate values available by this loop "
                "and scalars available before it"
            )

        slots, slot_count = _allocate_current_slots(order, uses, fusable_sources)
        # Region-local helpers are transient only during a bucket transition.  Give
        # them reusable current-value slots rather than persistent state/history.
        for region in derived_clock_regions.values():
            if region.phase != phase:
                continue
            for uid in region.transient_uids:
                if uid not in slots:
                    slots[uid] = slot_count
                    slot_count += 1
        loops.append(
            ExecutableLoopTemplate(
                uid=_digest([str(phase), *order, *_range_signature(range_args)], "exec_loop"),
                index=phase,
                coordinate_symbol="t",
                range_args=range_args,
                coordinate_step=coordinate_step,
                scan_direction=scan_direction,
                families=tuple(order),
                reductions=tuple(reds),
                post_scalars=tuple(post),
                evidence_loop_uids=tuple(sorted(set(evidence))),
                generalization=generalization,
                guard_notes=tuple(sorted(set(notes))),
                max_lag_by_family=dict(max_lag),
                current_slots=slots,
                current_slot_count=slot_count,
                fusable_reductions=tuple(fusable),
                fusion_reasons=fusion_reasons,
            )
        )

    # Any scalar barrier beyond the last loop must be part of that final loop's
    # post-scalar closure.  This catches an accidentally unscheduled operation.
    scheduled_scalars = set(pre_scalars)
    for block in loops:
        scheduled_scalars.update(block.post_scalars)
    missing_scalars = [u for u in scalars if u not in scheduled_scalars]
    if missing_scalars:
        raise TemplateError(f"scalar operations were not assigned to an executable dependency barrier: {missing_scalars!r}")

    if output_uid not in roles or roles[output_uid] not in ("scalar", "reduction"):
        raise TemplateError("current portfolio API requires a scalar/reduction output")

    if authority_note is not None:
        diagnostics.append(authority_note)

    if canonical_authority:
        backend_state_uids = {uid for loop in loops for uid in loop.families}
        backend_state_uids.update(derived_clock_states)
        for region in derived_clock_regions.values():
            backend_state_uids.update(region.persistent_uids)
        contract_persistent = set(getattr(canonical_execution_schedule, "persistent_uids", ()))
        missing_backend = sorted(contract_persistent - backend_state_uids)
        if missing_backend:
            raise TemplateError(
                "canonical schedule authority persistent membership disagreement; backend omitted: "
                f"{missing_backend!r}"
            )
        accidental_pure_state = sorted(
            set(getattr(canonical_execution_schedule, "pure_map_uids", ())) & backend_state_uids
        )
        if accidental_pure_state:
            raise TemplateError(
                "canonical schedule authority PureMap/storage disagreement: "
                f"{accidental_pure_state!r}"
            )

    graph = ExecutableGraph(
        pre_scalars=tuple(pre_scalars),
        loops=tuple(loops),
        output_uid=output_uid,
        roles=roles,
        pure_maps=pure_maps,
        uses=tuple(uses),
        coordinate_phase=coord_phase,
        scalar_barrier=scalar_barrier,
        reduction_phase=red_phase,
        derived_clock_states=derived_clock_states,
        derived_clock_regions=derived_clock_regions,
        boundary_seeds=boundary_seeds,
        formula_hash=ir.formula_hash,
        diagnostics=diagnostics,
        schedule_authority="canonical" if canonical_authority else "legacy",
        schedule_authority_contract_uid=authority_contract_uid if canonical_authority else None,
    )
    # Force validation of shared-domain restriction here so Python/native consume
    # exactly the same proven executable representation.
    graph.common_range_args
    return graph
