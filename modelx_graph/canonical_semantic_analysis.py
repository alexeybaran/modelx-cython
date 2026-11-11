from __future__ import annotations

"""Canonical semantic proof extraction.

This module owns source/AST interpretation needed to turn an exact canonical
semantic graph into scheduling-ready state-footprint and persistent-transition
facts.  The graph scheduler consumes only the frozen facts produced here and
never reparses formula source.

The split is intentionally architectural rather than behavioural: all proof
rules are migrated from the former scheduler implementation without changing
their admission policy.
"""

import ast
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical_semantic_graph import (
    CanonicalFootprintPath,
    CanonicalSemanticAccess,
    CanonicalSemanticGraph,
    CanonicalSemanticNode,
    CanonicalStateFootprintEvidence,
    CanonicalTransitionEvidence,
)
from .domain_graph import (
    _aggregate_availability,
    _availability_for_state_bounds,
    _eval_proof_affine,
    _expand_source_proof_expr,
    _exact_periodic_relative_bounds,
    _guards_axis_lower_bound,
    _normalized_guards,
    _proof_affine,
    _relative_lower_bound,
    _relative_upper_bound,
    _tarjan_source_components,
)
from .template_ir import _period_floor_bucket

def _stable_id(prefix: str, parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "backslashreplace"))
        h.update(b"\0")
    return f"{prefix}_{h.hexdigest()[:16]}"


def _parse_function(node: CanonicalSemanticNode) -> ast.FunctionDef | None:
    text = (node.canonical_source or "").strip()
    if not text.startswith("def "):
        return None
    try:
        module = ast.parse(text)
    except SyntaxError:
        return None
    return next((x for x in module.body if isinstance(x, ast.FunctionDef)), None)


@dataclass(frozen=True)
class _CanonicalCallsiteContext:
    """Lexical proof context for one exact canonical access.

    The semantic graph deliberately keeps the complete canonical AST.  Scheduling
    refines that lossless representation with conservative branch/short-circuit
    guards and generator-window metadata.  This mirrors the generic V02343
    call-site proof, but works on exact specialized UIDs and does not assume any
    particular loop-variable spelling.
    """

    guards: tuple[ast.AST, ...] = ()
    selector: str = "point"
    range_call: ast.Call | None = None
    generator_uid: str | None = None
    generator_filters: tuple[ast.AST, ...] = ()


def _copy_expr(node: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(ast.parse(ast.unparse(node), mode="eval").body)


def _negate_expr(node: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(ast.UnaryOp(op=ast.Not(), operand=_copy_expr(node)))


def _block_always_terminates(statements: list[ast.stmt]) -> bool:
    for statement in statements:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return True
        if isinstance(statement, ast.If) and statement.orelse:
            if _block_always_terminates(statement.body) and _block_always_terminates(statement.orelse):
                return True
    return False


def _canonical_callsite_contexts(
    graph: CanonicalSemanticGraph,
    functions: Mapping[str, ast.FunctionDef],
) -> dict[str, _CanonicalCallsiteContext]:
    """Recover path guards/window geometry for graph accesses without source names."""

    # ``canonical_source`` is reparsed from ast.unparse, so source locations are
    # intentionally not stable.  Match by exact structural call AST and consume
    # repeated identical calls in original source-location order.
    exact_access: dict[tuple[str, str, str], list[CanonicalSemanticAccess]] = {}
    for access in graph.accesses:
        exact_access.setdefault(
            (access.target_uid, access.source_uid, access.call_ast), []
        ).append(access)
    for rows in exact_access.values():
        rows.sort(key=lambda x: (int(x.lineno), int(x.col_offset), x.uid))
    consumed: dict[tuple[str, str, str], int] = {}

    result: dict[str, _CanonicalCallsiteContext] = {}

    def record(
        call: ast.Call,
        target_uid: str,
        guards: tuple[ast.AST, ...],
        *,
        selector: str = "point",
        range_call: ast.Call | None = None,
        generator_uid: str | None = None,
        generator_filters: tuple[ast.AST, ...] = (),
    ) -> None:
        if not isinstance(call.func, ast.Name):
            return
        call_ast = ast.dump(call, include_attributes=False)
        key = (target_uid, call.func.id, call_ast)
        rows = exact_access.get(key, ())
        offset = consumed.get(key, 0)
        if offset >= len(rows):
            return
        access = rows[offset]
        consumed[key] = offset + 1
        result[access.uid] = _CanonicalCallsiteContext(
            guards=tuple(_copy_expr(x) for x in guards),
            selector=selector,
            range_call=None if range_call is None else _copy_expr(range_call),
            generator_uid=generator_uid,
            generator_filters=tuple(_copy_expr(x) for x in generator_filters),
        )

    def walk_expr(node: ast.AST | None, target_uid: str, guards: tuple[ast.AST, ...]) -> None:
        if node is None:
            return
        if isinstance(node, ast.BoolOp):
            prior: list[ast.AST] = []
            for value in node.values:
                extra = (
                    tuple(_copy_expr(x) for x in prior)
                    if isinstance(node.op, ast.And)
                    else tuple(_negate_expr(x) for x in prior)
                )
                walk_expr(value, target_uid, guards + extra)
                prior.append(value)
            return
        if isinstance(node, ast.IfExp):
            walk_expr(node.test, target_uid, guards)
            walk_expr(node.body, target_uid, guards + (_copy_expr(node.test),))
            walk_expr(node.orelse, target_uid, guards + (_negate_expr(node.test),))
            return
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"any", "all", "sum", "min", "max"}
                and len(node.args) == 1
                and isinstance(node.args[0], ast.GeneratorExp)
                and len(node.args[0].generators) == 1
                and isinstance(node.args[0].elt, ast.Call)
            ):
                gen = node.args[0]
                comp = gen.generators[0]
                body_call = gen.elt
                if (
                    isinstance(body_call.func, ast.Name)
                    and body_call.func.id in functions
                    and isinstance(comp.target, ast.Name)
                    and isinstance(comp.iter, ast.Call)
                    and isinstance(comp.iter.func, ast.Name)
                    and comp.iter.func.id == "range"
                    and not comp.is_async
                ):
                    record(
                        body_call, target_uid, guards,
                        selector="window",
                        range_call=comp.iter,
                        generator_uid=comp.target.id,
                        generator_filters=tuple(comp.ifs),
                    )
                    walk_expr(comp.iter, target_uid, guards)
                    for filt in comp.ifs:
                        walk_expr(filt, target_uid, guards)
                    return
            record(node, target_uid, guards)
            for arg in node.args:
                walk_expr(arg, target_uid, guards)
            for keyword in node.keywords:
                walk_expr(keyword.value, target_uid, guards)
            return
        if isinstance(node, ast.GeneratorExp):
            walk_expr(node.elt, target_uid, guards)
            for comp in node.generators:
                walk_expr(comp.iter, target_uid, guards)
                for filt in comp.ifs:
                    walk_expr(filt, target_uid, guards)
            return
        for child in ast.iter_child_nodes(node):
            walk_expr(child, target_uid, guards)

    def walk_statements(
        statements: list[ast.stmt], target_uid: str, guards: tuple[ast.AST, ...]
    ) -> None:
        active = tuple(_copy_expr(x) for x in guards)
        for statement in statements:
            if isinstance(statement, ast.If):
                walk_expr(statement.test, target_uid, active)
                walk_statements(statement.body, target_uid, active + (_copy_expr(statement.test),))
                if statement.orelse:
                    walk_statements(
                        statement.orelse, target_uid, active + (_negate_expr(statement.test),)
                    )
                body_ends = _block_always_terminates(statement.body)
                else_ends = bool(statement.orelse) and _block_always_terminates(statement.orelse)
                if body_ends and else_ends:
                    return
                if body_ends and not statement.orelse:
                    active = active + (_negate_expr(statement.test),)
                elif body_ends and statement.orelse and not else_ends:
                    active = active + (_negate_expr(statement.test),)
                elif else_ends and not body_ends:
                    active = active + (_copy_expr(statement.test),)
                continue
            if isinstance(statement, (ast.Return, ast.Raise)):
                if isinstance(statement, ast.Return):
                    walk_expr(statement.value, target_uid, active)
                elif isinstance(statement, ast.Raise):
                    walk_expr(statement.exc, target_uid, active)
                return
            if isinstance(statement, ast.Assign):
                walk_expr(statement.value, target_uid, active)
                continue
            if isinstance(statement, ast.AnnAssign):
                walk_expr(statement.value, target_uid, active)
                continue
            if isinstance(statement, ast.AugAssign):
                walk_expr(statement.value, target_uid, active)
                continue
            if isinstance(statement, ast.Expr):
                walk_expr(statement.value, target_uid, active)
                continue
            if isinstance(statement, (ast.For, ast.While)):
                walk_expr(statement.iter if isinstance(statement, ast.For) else statement.test, target_uid, active)
                walk_statements(statement.body, target_uid, active)
                walk_statements(statement.orelse, target_uid, active)
                continue
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.expr):
                    walk_expr(child, target_uid, active)

    for target_uid, fn in sorted(functions.items()):
        walk_statements(fn.body, target_uid, ())
    return result


@dataclass(frozen=True)
class _PartialPath:
    persistent_uid: str
    min_offset: int | None
    max_offset: int | None
    access_uids: tuple[str, ...]
    proof_kinds: tuple[str, ...]
    blockers: tuple[str, ...] = ()


def _dedupe_partial_paths(paths: Iterable[_PartialPath]) -> tuple[_PartialPath, ...]:
    """Keep one exact witness for each scheduling-equivalent persistent path."""
    best: dict[tuple[Any, ...], _PartialPath] = {}
    for row in paths:
        key = (row.persistent_uid, row.min_offset, row.max_offset, row.blockers)
        prior = best.get(key)
        if prior is None or (len(row.access_uids), row.access_uids) < (len(prior.access_uids), prior.access_uids):
            best[key] = row
    return tuple(sorted(
        best.values(),
        key=lambda x: (x.persistent_uid, str(x.min_offset), str(x.max_offset), x.blockers, x.access_uids),
    ))


def _compose_bounds(
    outer_lo: int | None,
    outer_hi: int | None,
    inner_lo: int | None,
    inner_hi: int | None,
) -> tuple[int | None, int | None]:
    """Compose relative bounds without discarding one-sided causal proof.

    A known upper bound below zero is already sufficient to prove an ascending
    recurrence even when the lower extent is unbounded/unknown.  The legacy helper
    required both sides, which unnecessarily destroyed useful causal evidence.
    """
    lo = (
        int(outer_lo) + int(inner_lo)
        if outer_lo is not None and inner_lo is not None
        else None
    )
    hi = (
        int(outer_hi) + int(inner_hi)
        if outer_hi is not None and inner_hi is not None
        else None
    )
    return lo, hi


def _canonical_availability_for_bounds(
    lo: int | None,
    hi: int | None,
    *,
    transition_relevant: bool,
) -> str:
    """Canonical availability lattice with sound one-sided support."""
    if hi is not None and int(hi) < 0:
        return "PREFIX"
    if lo is not None and hi is not None:
        return _availability_for_state_bounds(
            lo, hi, transition_relevant=transition_relevant
        )
    if lo is not None and int(lo) > 0:
        return "SUFFIX" if transition_relevant else "COMPLETE"
    return "UNKNOWN"


def _path_availability(path: _PartialPath, *, transition_relevant: bool) -> str:
    availability = _canonical_availability_for_bounds(
        path.min_offset, path.max_offset, transition_relevant=transition_relevant
    )
    if availability == "UNKNOWN" and path.blockers:
        return "UNKNOWN"
    return availability


def _scan_requirement(lo: int | None, hi: int | None) -> str:
    if hi is not None and int(hi) < 0:
        return "ascending"
    if lo is not None and int(lo) > 0:
        return "descending"
    if lo is not None and hi is not None:
        if int(lo) == 0 and int(hi) == 0:
            return "any"
        return "mixed"
    return "unknown"


class _NameSubstituter(ast.NodeTransformer):
    def __init__(self, name: str, replacement: ast.AST) -> None:
        self.name = name
        self.replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            return ast.copy_location(_copy_expr(self.replacement), node)
        return node


def _substitute_name_expr(node: ast.AST, name: str, replacement: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(_NameSubstituter(name, replacement).visit(_copy_expr(node)))


def _period_floor_plus_constant_bounds(
    node: ast.AST,
    *,
    axis: str,
    domain: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Bound ``k*max(0,t//k)+c`` on a nonnegative integer domain.

    The snapshot coordinate lies between ``t-(k-1)`` and ``t``.  An additive
    integer lag therefore gives exact relative bounds ``[-(k-1)+c, c]``.
    This is the generic period-snapshot geometry already recognized by the legacy
    scheduler, now used after exact canonical path composition.
    """
    if domain is None or int(domain[0]) < 0:
        return None

    def literal_int(x: ast.AST) -> int | None:
        if isinstance(x, ast.Constant) and isinstance(x.value, int) and not isinstance(x.value, bool):
            return int(x.value)
        if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub):
            value = literal_int(x.operand)
            return None if value is None else -value
        return None

    base = node
    offset = 0
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        right = literal_int(node.right)
        left = literal_int(node.left)
        if right is not None:
            base = node.left
            offset = right if isinstance(node.op, ast.Add) else -right
        elif left is not None and isinstance(node.op, ast.Add):
            base = node.right
            offset = left
    period = _period_floor_bucket(base, axis)
    if period is None:
        return None
    return -(int(period) - 1) + int(offset), int(offset)


_FINITE_DOMAIN_RELATION_ENUM_CAP = 4096


def _finite_domain_relative_bounds(
    node: ast.AST,
    *,
    axis: str,
    domain: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Prove exact relative bounds by finite evaluation on a frozen coordinate domain.

    This is deliberately a semantic proof fallback, not a trace sampler.  Every
    integer coordinate in the declared canonical domain is evaluated through the
    same tiny integer proof algebra already used by periodic relation analysis.
    Expressions containing unresolved calls, non-integer/object operations, or a
    domain above the generic proof-work cap fail closed.

    The fallback is useful for composed transforms whose relation to the root
    coordinate is bounded on the *declared finite domain* but not translation
    invariant, e.g. year-to-month transforms such as ``12 * (t - 1)`` or clamped
    bucketing expressions.  It never changes physical iteration geometry.
    """
    if domain is None:
        return None
    lo, hi = int(domain[0]), int(domain[1])
    if hi < lo or hi - lo + 1 > _FINITE_DOMAIN_RELATION_ENUM_CAP:
        return None
    rows: list[int] = []
    for coordinate in range(lo, hi + 1):
        value = _eval_proof_affine(
            node,
            {axis: _proof_affine(const=int(coordinate))},
        )
        if value is None or value.coeffs:
            return None
        rows.append(int(value.const) - int(coordinate))
    return (min(rows), max(rows)) if rows else None


def _composed_path_bounds(
    *,
    root_uid: str,
    access_uids: tuple[str, ...],
    nodes: Mapping[str, CanonicalSemanticNode],
    accesses_by_uid: Mapping[str, CanonicalSemanticAccess],
    functions: Mapping[str, ast.FunctionDef],
    state_free_uids: set[str],
) -> tuple[int | None, int | None, str, tuple[str, ...]]:
    """Prove persistent-leaf geometry by composing exact canonical call ASTs.

    This is deliberately path based.  Intermediate event clocks and period-start
    helpers may each be awkward relative to their immediate caller while their
    composition is simple and causal in the root persistent coordinate.
    """
    root = nodes[root_uid]
    root_domain = root.domain
    if len(root.parameters) == 1:
        root_axis = root.parameters[0]
    elif root.canonical_role == "reduction" and access_uids:
        first = accesses_by_uid.get(access_uids[0])
        if (
            first is None or first.target_uid != root_uid
            or len(first.argument_exprs) != 1
        ):
            return None, None, "canonical_composed_path_unproved_v1", ("root_reduction_axis_unproved",)
        try:
            first_expr = ast.parse(first.argument_exprs[0], mode="eval").body
        except SyntaxError:
            return None, None, "canonical_composed_path_unproved_v1", ("root_reduction_axis_unproved",)
        if not isinstance(first_expr, ast.Name):
            return None, None, "canonical_composed_path_unproved_v1", ("root_reduction_axis_unproved",)
        root_axis = first_expr.id
        first_source = nodes.get(first.source_uid)
        if first_source is not None and first_source.domain is not None:
            root_domain = first_source.domain
    else:
        return None, None, "canonical_composed_path_unproved_v1", ("root_coordinate_axis_unproved",)
    current_expr: ast.AST = ast.Name(id=root_axis, ctx=ast.Load())
    current_uid = root_uid

    for access_uid in access_uids:
        access = accesses_by_uid.get(access_uid)
        if access is None or access.target_uid != current_uid:
            return None, None, "canonical_composed_path_unproved_v1", (
                f"canonical_path_discontinuity:{access_uid}",
            )
        target = nodes[access.target_uid]
        source = nodes[access.source_uid]
        target_fn = functions.get(target.uid)

        if source.canonical_role == "scalar":
            current_uid = source.uid
            continue
        if source.canonical_role != "coordinate" or len(access.argument_exprs) != 1:
            return None, None, "canonical_composed_path_unproved_v1", (
                f"canonical_path_coordinate_arity:{access_uid}",
            )
        if current_uid == root_uid and root.canonical_role == "reduction":
            target_axis = root_axis
        elif target_fn is not None and len(target.parameters) == 1:
            target_axis = target.parameters[0]
        else:
            return None, None, "canonical_composed_path_unproved_v1", (
                f"canonical_path_target_axis_unknown:{access_uid}",
            )
        try:
            expr = ast.parse(access.argument_exprs[0], mode="eval").body
        except SyntaxError:
            return None, None, "canonical_composed_path_unproved_v1", (
                f"canonical_path_argument_parse:{access_uid}",
            )
        if target_fn is not None:
            expr = _expand_source_proof_expr(expr, target_fn, dict(functions), state_free_uids)
        current_expr = _substitute_name_expr(expr, target_axis, current_expr)
        current_uid = source.uid

    if current_uid not in nodes:
        return None, None, "canonical_composed_path_unproved_v1", ("canonical_path_terminal_unknown",)

    exact = _exact_periodic_relative_bounds(current_expr, (), axis=root_axis)
    if exact is not None:
        return (
            int(exact[0]), int(exact[1]),
            "canonical_composed_periodic_path_v1", (),
        )
    floor_bounds = _period_floor_plus_constant_bounds(
        current_expr, axis=root_axis, domain=root_domain
    )
    if floor_bounds is not None:
        return (
            int(floor_bounds[0]), int(floor_bounds[1]),
            "canonical_composed_period_floor_path_v1", (),
        )
    finite_bounds = _finite_domain_relative_bounds(
        current_expr,
        axis=root_axis,
        domain=root_domain,
    )
    if finite_bounds is not None:
        return (
            int(finite_bounds[0]), int(finite_bounds[1]),
            "canonical_composed_finite_domain_path_v1", (),
        )
    return None, None, "canonical_composed_path_unproved_v1", (
        "canonical_composed_relation_unproved:" + ast.unparse(current_expr),
    )


def _is_geometry_blocker(value: str) -> bool:
    return (
        value.startswith("canonical_relation_unknown:")
        or value.startswith("canonical_composed_relation_unproved:")
        or value in {
            "non_affine_or_unproved_coordinate_relation",
            "state_dependent_scalar_requires_derived_footprint",
            "coordinate_arity_not_one",
            "target_has_no_iteration_axis",
            "root_coordinate_arity_not_one",
            "root_coordinate_axis_unproved",
            "root_reduction_axis_unproved",
        }
    )


def _periodic_proof_guard_supported(node: ast.AST) -> bool:
    """Whether V02343's finite-residue proof can make use of this guard.

    Dropping an unsupported guard is conservative: it widens the proven subdomain
    and can only weaken a bound.  Retaining opaque state calls, by contrast, creates
    thousands of memo-distinct contexts that the proof evaluator treats identically.
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if not isinstance(child.func, ast.Name):
            return False
        if not child.args and not child.keywords:
            continue
        if child.func.id in {"min", "max"} and not child.keywords:
            continue
        return False
    return True


def _usable_proof_guards(
    guards: tuple[ast.AST, ...],
    *,
    target_fn: ast.FunctionDef,
    functions: Mapping[str, ast.FunctionDef],
    state_free_uids: set[str],
) -> tuple[ast.AST, ...]:
    # Split conjunctions/negated disjunctions before dropping opaque pieces.
    # Otherwise a useful periodic guard such as ``is_anniv(t)`` would be lost merely
    # because it shares one branch predicate with a persistent-state condition.
    conjuncts = _normalized_guards(guards)
    expanded = tuple(
        _expand_source_proof_expr(g, target_fn, dict(functions), state_free_uids)
        for g in conjuncts
    )
    return _normalized_guards(tuple(g for g in expanded if _periodic_proof_guard_supported(g)))


def _canonical_window_bounds(
    *,
    target_fn: ast.FunctionDef,
    callsite: _CanonicalCallsiteContext,
    functions: Mapping[str, ast.FunctionDef],
    state_free_uids: set[str],
    axis: str,
    inherited_guards: tuple[ast.AST, ...] = (),
) -> tuple[int | None, int | None, str, tuple[str, ...]]:
    """Prove one exact canonical generator window relative to the caller axis."""

    call = callsite.range_call
    if callsite.selector != "window" or call is None or callsite.generator_uid is None:
        return None, None, "canonical_window_bounds_unknown_v1", ("missing_window_context",)
    if not (isinstance(call.func, ast.Name) and call.func.id == "range"):
        return None, None, "canonical_window_bounds_unknown_v1", ("canonical_window_iterator_not_range",)
    args = list(call.args)
    if len(args) == 1:
        start = ast.Constant(value=0)
        stop = args[0]
        step = 1
    elif len(args) in {2, 3}:
        start, stop = args[0], args[1]
        if len(args) == 3:
            if not (
                isinstance(args[2], ast.Constant)
                and isinstance(args[2].value, int)
                and not isinstance(args[2].value, bool)
            ):
                return None, None, "canonical_window_bounds_unknown_v1", ("canonical_window_dynamic_step",)
            step = int(args[2].value)
        else:
            step = 1
    else:
        return None, None, "canonical_window_bounds_unknown_v1", ("canonical_window_range_arity",)
    if step != 1:
        return None, None, "canonical_window_bounds_unknown_v1", ("canonical_window_nonunit_step",)

    expanded_start = _expand_source_proof_expr(start, target_fn, dict(functions), state_free_uids)
    expanded_stop = _expand_source_proof_expr(stop, target_fn, dict(functions), state_free_uids)
    stop_inclusive = ast.fix_missing_locations(
        ast.BinOp(left=expanded_stop, op=ast.Sub(), right=ast.Constant(value=1))
    )
    guards = _usable_proof_guards(
        inherited_guards + callsite.guards + callsite.generator_filters,
        target_fn=target_fn,
        functions=functions,
        state_free_uids=state_free_uids,
    )
    lo = _relative_lower_bound(expanded_start, guards, axis=axis)
    hi = _relative_upper_bound(stop_inclusive, guards, axis=axis)
    if lo is None or hi is None:
        return None, None, "canonical_window_bounds_unknown_v1", (
            "canonical_window_relative_bounds_unproved:"
            f"{ast.unparse(expanded_start)}..{ast.unparse(stop_inclusive)}",
        )
    return int(lo), int(hi), "canonical_window_relative_bounds_v1", ()



def _refined_access_bounds(
    access: CanonicalSemanticAccess,
    *,
    nodes: Mapping[str, CanonicalSemanticNode],
    functions: Mapping[str, ast.FunctionDef],
    state_free_uids: set[str],
    callsite: _CanonicalCallsiteContext | None = None,
    inherited_guards: tuple[ast.AST, ...] = (),
) -> tuple[int | None, int | None, str, tuple[str, ...], tuple[ast.AST, ...]]:
    """Return the best exact-callsite canonical relation proof.

    The returned guard tuple is the normalized proof subdomain and is propagated
    into a same-coordinate StateDerivedMap expansion.  This prevents a helper
    with a full-domain future window from being treated as full-domain when a
    particular transition uses it only on a narrower proven subdomain.
    """

    source = nodes[access.source_uid]
    target = nodes[access.target_uid]
    target_fn = functions.get(target.uid)
    local_guards = () if callsite is None else callsite.guards
    proof_guards: tuple[ast.AST, ...]
    if target_fn is not None:
        proof_guards = _usable_proof_guards(
            inherited_guards + local_guards,
            target_fn=target_fn,
            functions=functions,
            state_free_uids=state_free_uids,
        )
    else:
        proof_guards = ()

    # Direct affine geometry remains valid under a narrower subdomain.  Keep the
    # lexical guards for any nested state-derived expansion.
    if access.min_offset is not None and access.max_offset is not None:
        return (
            int(access.min_offset), int(access.max_offset), access.proof_kind,
            tuple(access.blockers), proof_guards,
        )

    # A scalar/helper invocation does not itself move the caller coordinate.  Any
    # state geometry lives in the helper's own canonical accesses.
    if source.canonical_role == "scalar":
        return 0, 0, "canonical_scalar_context_identity_v1", (), proof_guards

    if source.canonical_role != "coordinate" or len(access.argument_exprs) != 1:
        return None, None, access.proof_kind, tuple(access.blockers), proof_guards

    axis = target.parameters[0] if len(target.parameters) == 1 else None
    if target_fn is None or axis is None:
        return None, None, access.proof_kind, tuple(access.blockers), proof_guards

    try:
        expr = ast.parse(access.argument_exprs[0], mode="eval").body
    except SyntaxError:
        return None, None, access.proof_kind, tuple(access.blockers), proof_guards

    # A generator body coordinate needs its exact surrounding range.  Do this
    # before ordinary point algebra because the iterator name is intentionally not
    # the caller axis.
    if callsite is not None and callsite.selector == "window":
        wlo, whi, proof, blockers = _canonical_window_bounds(
            target_fn=target_fn,
            callsite=callsite,
            functions=functions,
            state_free_uids=state_free_uids,
            axis=axis,
            inherited_guards=inherited_guards,
        )
        if wlo is not None and whi is not None:
            return wlo, whi, proof, blockers, proof_guards

    expanded = _expand_source_proof_expr(expr, target_fn, dict(functions), state_free_uids)
    lo = _relative_lower_bound(expanded, proof_guards, axis=axis)
    hi = _relative_upper_bound(expanded, proof_guards, axis=axis)
    if lo is not None and hi is not None:
        return int(lo), int(hi), "canonical_periodic_relative_bounds_v1", (), proof_guards

    # Reuse V02343's generic zero-floor proof.  ``max(0, q(t))`` remains
    # strict-past when q(t) is already strict-past and the caller subdomain proves
    # t >= 1.  This is common for bounded prior-anniversary lookups and is not a
    # model-specific calendar exception.
    if (
        isinstance(expanded, ast.Call)
        and isinstance(expanded.func, ast.Name)
        and expanded.func.id == "max"
        and len(expanded.args) == 2
    ):
        nonzero = None
        if isinstance(expanded.args[0], ast.Constant) and expanded.args[0].value == 0:
            nonzero = expanded.args[1]
        elif isinstance(expanded.args[1], ast.Constant) and expanded.args[1].value == 0:
            nonzero = expanded.args[0]
        if nonzero is not None:
            raw = _exact_periodic_relative_bounds(nonzero, proof_guards, axis=axis)
            caller_lower = _guards_axis_lower_bound(proof_guards, axis=axis)
            if raw is not None and raw[1] < 0 and caller_lower is not None and caller_lower >= 1:
                return (
                    int(raw[0]), -1, "canonical_zero_floor_strict_past_v1", (), proof_guards
                )

    # A fixed canonical coordinate can still be related to the exact caller
    # interval without assuming a loop direction.
    if (
        isinstance(expanded, ast.Constant)
        and isinstance(expanded.value, int)
        and not isinstance(expanded.value, bool)
        and target.domain is not None
    ):
        q = int(expanded.value)
        target_lo, target_hi = target.domain
        return (
            q - int(target_hi), q - int(target_lo),
            "canonical_fixed_coordinate_relative_domain_v1", (), proof_guards,
        )

    return None, None, access.proof_kind, tuple(access.blockers), proof_guards





def build_canonical_semantic_evidence(
    graph: CanonicalSemanticGraph,
    *,
    functions: Mapping[str, ast.FunctionDef] | None = None,
) -> tuple[tuple[CanonicalStateFootprintEvidence, ...], tuple[CanonicalTransitionEvidence, ...]]:
    """Freeze all scheduling-relevant formula semantics onto the canonical graph.

    This is the last layer allowed to parse canonical formula source.  The result
    is total: unknown relations remain typed blockers in the returned evidence.
    """
    nodes = {x.uid: x for x in graph.nodes}
    accesses_by_target: dict[str, list[CanonicalSemanticAccess]] = {uid: [] for uid in nodes}
    accesses_by_uid = {access.uid: access for access in graph.accesses}
    for access in graph.accesses:
        accesses_by_target.setdefault(access.target_uid, []).append(access)
    for rows in accesses_by_target.values():
        rows.sort(key=lambda x: x.uid)

    if functions is None:
        # Compatibility path for externally constructed semantic graphs. Production
        # graph construction passes the original canonical FunctionDef objects and
        # therefore performs no source reparse at this boundary.
        functions = {
            uid: fn for uid, node in nodes.items()
            if (fn := _parse_function(node)) is not None
        }
    else:
        functions = {str(uid): fn for uid, fn in functions.items()}
    state_free_uids = {uid for uid, node in nodes.items() if node.state_semantic in {"state_free", "auxiliary_recurrence"}}
    persistent_uids = {uid for uid, node in nodes.items() if node.state_semantic == "persistent_state"}
    callsite_contexts = _canonical_callsite_contexts(graph, functions)

    # Only values whose internal relation proof can actually depend on a caller
    # subdomain need guard-specialized footprint memoization.  Propagating guards
    # through every same-coordinate helper creates a combinatorial set of equivalent
    # contexts on large models.  The fixed point below is semantic, not name based:
    # a value is guard-sensitive when it owns a window/unproved coordinate relation
    # or same-coordinate-calls another guard-sensitive value.
    guard_sensitive_uids: set[str] = {
        access.target_uid
        for access in graph.accesses
        if (
            callsite_contexts.get(access.uid, _CanonicalCallsiteContext()).selector == "window"
            or (
                access.scheduling_relevant
                and nodes[access.source_uid].canonical_role == "coordinate"
                and access.min_offset is None
                and access.max_offset is None
            )
        )
    }
    changed = True
    while changed:
        changed = False
        for access in graph.accesses:
            if access.source_uid not in guard_sensitive_uids or access.target_uid in guard_sensitive_uids:
                continue
            if access.min_offset == 0 and access.max_offset == 0:
                guard_sensitive_uids.add(access.target_uid)
                changed = True

    # Explicit SCC analysis for recursive state-derived/helper regions.  Cycles are
    # represented as typed evidence, never discovered accidentally by recursion.
    helper_nodes = {
        uid for uid, node in nodes.items()
        if node.state_semantic not in {"state_free", "auxiliary_recurrence", "persistent_state"}
    }
    helper_edges = {
        (access.target_uid, access.source_uid)
        for access in graph.accesses
        if access.scheduling_relevant
        and access.target_uid in helper_nodes
        and access.source_uid in helper_nodes
    }
    helper_sccs = _tarjan_source_components(helper_nodes, helper_edges)
    helper_group: dict[str, tuple[str, ...]] = {}
    cyclic_helpers: set[str] = set()
    for members in helper_sccs:
        for uid in members:
            helper_group[uid] = members
        if len(members) > 1 or any(source == target == members[0] for source, target in helper_edges):
            cyclic_helpers.update(members)

    memo: dict[
        tuple[str, tuple[str, ...]],
        tuple[tuple[_PartialPath, ...], tuple[str, ...]],
    ] = {}

    def resolve_value(
        uid: str,
        inherited_guards: tuple[ast.AST, ...] = (),
        active: tuple[str, ...] = (),
    ) -> tuple[tuple[_PartialPath, ...], tuple[str, ...]]:
        node = nodes[uid]
        if node.state_semantic in {"state_free", "auxiliary_recurrence"}:
            return (), ()
        if node.state_semantic == "persistent_state":
            return (
                (_PartialPath(uid, 0, 0, (), ("canonical_persistent_leaf_v1",), ()),),
                (),
            )

        normalized_inherited = (
            _normalized_guards(inherited_guards) if uid in guard_sensitive_uids else ()
        )
        memo_key = (uid, tuple(ast.unparse(x) for x in normalized_inherited))
        if memo_key in memo:
            return memo[memo_key]

        paths: list[_PartialPath] = []
        blockers: set[str] = set()
        if uid in cyclic_helpers:
            blockers.add(
                "canonical_state_helper_scc:" + ",".join(helper_group.get(uid, (uid,)))
            )

        for access in accesses_by_target.get(uid, ()):
            if not access.scheduling_relevant:
                continue
            source_uid = access.source_uid
            # An SCC-internal helper edge cannot be expanded finitely without a
            # recurrence proof. Other outgoing branches may still expose useful
            # persistent leaves and are retained.
            if uid in cyclic_helpers and source_uid in helper_group.get(uid, ()):
                continue

            alo, ahi, proof, relation_blockers, proof_guards = _refined_access_bounds(
                access,
                nodes=nodes,
                functions=functions,
                state_free_uids=state_free_uids,
                callsite=callsite_contexts.get(access.uid),
                inherited_guards=normalized_inherited,
            )
            # A caller subdomain is sound to propagate only through a same-coordinate
            # helper call. Once the coordinate moves, the guard belongs to the outer
            # coordinate and must not be reinterpreted at the child coordinate.
            child_guards = (
                proof_guards
                if (
                    source_uid in guard_sensitive_uids
                    and alo is not None and ahi is not None
                    and int(alo) == 0 and int(ahi) == 0
                )
                else ()
            )
            child_paths, child_blockers = resolve_value(
                source_uid, child_guards, active + (uid,)
            )
            blockers.update(child_blockers)
            if not child_paths:
                continue
            for child in child_paths:
                lo, hi = _compose_bounds(alo, ahi, child.min_offset, child.max_offset)
                row_blockers = tuple(sorted(set(child.blockers) | set(relation_blockers)))
                if alo is None or ahi is None:
                    row_blockers = tuple(sorted(set(row_blockers) | {
                        f"canonical_relation_unknown:{access.uid}"
                    }))
                paths.append(_PartialPath(
                    persistent_uid=child.persistent_uid,
                    min_offset=lo,
                    max_offset=hi,
                    access_uids=(access.uid,) + child.access_uids,
                    proof_kinds=(proof,) + child.proof_kinds,
                    blockers=row_blockers,
                ))

        # Exact helper with no persistent leaves is state-independent for scheduling
        # even when the source classifier was conservative. Backend executability is
        # kept separate by CanonicalSemanticGraph.execution_semantic.
        result = (
            _dedupe_partial_paths(paths),
            tuple(sorted(blockers)),
        )
        memo[memo_key] = result
        return result

    footprints: list[CanonicalStateFootprintEvidence] = []
    footprint_by_uid: dict[str, CanonicalStateFootprintEvidence] = {}
    for uid, node in sorted(nodes.items()):
        if node.state_semantic == "persistent_state":
            partials = (_PartialPath(uid, 0, 0, (), ("canonical_persistent_leaf_v1",), ()),)
            blockers: tuple[str, ...] = ()
        elif node.state_semantic in {"state_free", "auxiliary_recurrence"}:
            partials = ()
            blockers = ()
        else:
            partials, blockers = resolve_value(uid)
        paths: list[CanonicalFootprintPath] = []
        for row in partials:
            row_lo, row_hi = row.min_offset, row.max_offset
            row_proofs = row.proof_kinds
            row_blockers = row.blockers
            if row.access_uids and (
                row_lo is None or row_hi is None or any(_is_geometry_blocker(x) for x in row_blockers)
            ):
                plo, phi, pproof, pblockers = _composed_path_bounds(
                    root_uid=uid,
                    access_uids=row.access_uids,
                    nodes=nodes,
                    accesses_by_uid=accesses_by_uid,
                    functions=functions,
                    state_free_uids=state_free_uids,
                )
                if plo is not None or phi is not None:
                    row_lo, row_hi = plo, phi
                    row_proofs = row_proofs + (pproof,)
                    row_blockers = tuple(x for x in row_blockers if not _is_geometry_blocker(x))
                elif pblockers:
                    row_blockers = tuple(sorted(set(row_blockers) | set(pblockers)))
            refined_row = _PartialPath(
                persistent_uid=row.persistent_uid,
                min_offset=row_lo,
                max_offset=row_hi,
                access_uids=row.access_uids,
                proof_kinds=row_proofs,
                blockers=row_blockers,
            )
            availability = _path_availability(refined_row, transition_relevant=False)

            # A fixed scalar OutputInvocation has no caller iteration axis by
            # design.  When its first state-bearing coordinate call has already
            # been frozen as an exact ``relation='fixed'`` access, relative
            # offset geometry is unnecessary: the scalar may be evaluated once
            # after the required producer histories complete.  This is a root
            # OutputInvocation proof only; arbitrary axis-less helpers keep the
            # ordinary UNKNOWN geometry and fail closed.
            if uid == graph.output_uid and not node.parameters and row.access_uids:
                first = accesses_by_uid[row.access_uids[0]]
                if (
                    first.target_uid == uid
                    and first.relation == "fixed"
                    and first.fixed_coordinate is not None
                    and not first.blockers
                ):
                    removable = {
                        "target_has_no_iteration_axis",
                        "root_coordinate_axis_unproved",
                        f"canonical_relation_unknown:{first.uid}",
                    }
                    remaining = tuple(
                        blocker for blocker in row_blockers if blocker not in removable
                    )
                    if not any(_is_geometry_blocker(blocker) for blocker in remaining):
                        row_blockers = remaining
                        row_proofs = row_proofs + (
                            "canonical_fixed_output_completion_v1",
                        )
                        availability = "COMPLETE"
            paths.append(CanonicalFootprintPath(
                uid=_stable_id("canonical_footprint_path", (
                    uid, row.persistent_uid, row_lo, row_hi, row.access_uids
                )),
                root_uid=uid,
                persistent_source_uid=row.persistent_uid,
                min_offset=row_lo,
                max_offset=row_hi,
                availability=availability,
                access_uids=row.access_uids,
                proof_kinds=row_proofs,
                blockers=row_blockers,
            ))
        availability = _aggregate_availability(x.availability for x in paths)
        if blockers and not paths:
            availability = "UNKNOWN"
        evidence = CanonicalStateFootprintEvidence(
            uid=_stable_id("canonical_state_footprint", (uid, availability)),
            node_uid=uid,
            state_semantic=node.state_semantic,
            persistent_source_uids=tuple(sorted({x.persistent_source_uid for x in paths})),
            paths=tuple(paths),
            availability=availability,
            blockers=blockers,
            note=(
                "exact canonical carried-state leaf" if node.state_semantic == "persistent_state" else
                "policy-state-independent canonical value; backend executability is independent" if node.state_semantic in {"state_free", "auxiliary_recurrence"} else
                "persistent leaves expanded through exact canonical accesses"
            ),
        )
        footprints.append(evidence)
        footprint_by_uid[uid] = evidence

    # Expand each persistent transition body instead of treating the target itself
    # as a leaf.  The resulting exact-UID graph is the carried-state recurrence graph.
    transitions: list[CanonicalTransitionEvidence] = []
    for target_uid in sorted(persistent_uids):
        for access in accesses_by_target.get(target_uid, ()):
            if not access.scheduling_relevant:
                continue
            alo, ahi, proof, relation_blockers, proof_guards = _refined_access_bounds(
                access,
                nodes=nodes,
                functions=functions,
                state_free_uids=state_free_uids,
                callsite=callsite_contexts.get(access.uid),
                inherited_guards=(),
            )
            child_guards = (
                proof_guards
                if (
                    access.source_uid in guard_sensitive_uids
                    and alo is not None and ahi is not None
                    and int(alo) == 0 and int(ahi) == 0
                )
                else ()
            )
            child_paths, child_blockers = resolve_value(access.source_uid, child_guards)
            if not child_paths:
                # No persistent leaf means this helper contributes no state edge.
                continue
            for child in child_paths:
                lo, hi = _compose_bounds(alo, ahi, child.min_offset, child.max_offset)
                blockers = set(child.blockers) | set(child_blockers) | set(relation_blockers)
                if alo is None or ahi is None:
                    blockers.add(f"canonical_relation_unknown:{access.uid}")
                access_uids = (access.uid,) + child.access_uids
                proof_kinds = (proof,) + child.proof_kinds
                if lo is None or hi is None or any(_is_geometry_blocker(x) for x in blockers):
                    plo, phi, pproof, pblockers = _composed_path_bounds(
                        root_uid=target_uid,
                        access_uids=access_uids,
                        nodes=nodes,
                        accesses_by_uid=accesses_by_uid,
                        functions=functions,
                        state_free_uids=state_free_uids,
                    )
                    if plo is not None or phi is not None:
                        lo, hi = plo, phi
                        proof_kinds = proof_kinds + (pproof,)
                        blockers = {x for x in blockers if not _is_geometry_blocker(x)}
                    else:
                        blockers.update(pblockers)
                availability = _canonical_availability_for_bounds(
                    lo, hi, transition_relevant=True
                )
                if blockers and availability == "UNKNOWN":
                    availability = "UNKNOWN"
                transitions.append(CanonicalTransitionEvidence(
                    uid=_stable_id("canonical_transition", (
                        child.persistent_uid, target_uid, lo, hi, access_uids
                    )),
                    source_uid=child.persistent_uid,
                    target_uid=target_uid,
                    min_offset=lo,
                    max_offset=hi,
                    availability=availability,
                    access_uids=access_uids,
                    proof_kinds=proof_kinds,
                    blockers=tuple(sorted(blockers)),
                ))

    # Collapse scheduling-equivalent transition witnesses.  Distinct bounds,
    # availability or blockers are retained because any of them can change the SCC
    # scan/barrier conclusion; only redundant exact witnesses are discarded.
    transition_best: dict[tuple[Any, ...], CanonicalTransitionEvidence] = {}
    for row in transitions:
        key = (
            row.source_uid, row.target_uid, row.min_offset, row.max_offset,
            row.availability, row.blockers,
        )
        prior = transition_best.get(key)
        if prior is None or (len(row.access_uids), row.access_uids) < (len(prior.access_uids), prior.access_uids):
            transition_best[key] = row
    transitions = sorted(
        transition_best.values(),
        key=lambda x: (x.source_uid, x.target_uid, str(x.min_offset), str(x.max_offset), x.availability, x.blockers),
    )


    return tuple(footprints), tuple(transitions)
