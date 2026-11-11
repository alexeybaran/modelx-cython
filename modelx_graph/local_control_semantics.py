from __future__ import annotations

"""Graph-free recognition of small source-local control semantics.

This module contains *semantic* recognizers only.  It deliberately knows nothing
about Stage schedules, storage plans, SCCs, or execution order across Cells.
Recognized statements may be emitted directly inside one recursively callable
native Cell.
"""

import ast
from dataclasses import dataclass
from typing import Iterable, Mapping

from .int64_range_proof import Int64Interval, Int64RangeProof, prove_int64_range


_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class FirstHitRangeSpec:
    """A bounded local ``for range`` returning its iterator on first match."""

    iterator: str
    range_args: tuple[ast.expr, ...]
    predicate: ast.expr
    line: int

    @property
    def range_arity(self) -> int:
        return len(self.range_args)

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "first_hit_range",
            "iterator": self.iterator,
            "range_args": [ast.unparse(x) for x in self.range_args],
            "predicate": ast.unparse(self.predicate),
            "line": int(self.line),
            "execution_scope": "source_local_cell_body",
        }


@dataclass(frozen=True)
class BoundedFilteredAddRangeSpec:
    """A bounded range loop adding one scalar contribution under one filter.

    Historical instances use ``filter_mode='skip_continue'``.  Stage 4A2B1 adds
    ``truth_guard`` instances whose range is independently proven by the signed-
    int64 interval engine and whose loop-local temporaries are explicitly owned
    by the descriptor.
    """

    iterator: str
    range_args: tuple[ast.expr, ...]
    accumulator: str
    line: int
    continue_stmt_id: int | None = None
    predicate: ast.expr | None = None
    owned_temporaries: tuple[str, ...] = ()
    filter_mode: str = "skip_continue"
    range_proof: Int64RangeProof | None = None

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "bounded_filtered_add_range",
            "iterator": self.iterator,
            "range_args": [ast.unparse(x) for x in self.range_args],
            "accumulator": self.accumulator,
            "filter_mode": self.filter_mode,
            "predicate": None if self.predicate is None else ast.unparse(self.predicate),
            "owned_temporaries": list(self.owned_temporaries),
            "range_proof": None if self.range_proof is None else self.range_proof.manifest(),
            "line": int(self.line),
            "execution_scope": "source_local_cell_body",
        }


@dataclass(frozen=True)
class BoundedAddRangeSpec:
    """An exact-bounded range with one scalar additive accumulator update.

    The loop may be nested under ordinary source ``if`` control, but owns no
    branching, iterator mutation, break/continue, or other side effects of its
    own.  Range arguments are exact source integers or preparation-certified
    exact integer globals, so this remains a finite scalar-control semantic
    rather than a general iterable VM.
    """

    iterator: str
    range_args: tuple[ast.expr, ...]
    range_values: tuple[int, ...]
    trip_count: int
    accumulator: str
    range_arg_authorities: tuple[str, ...]
    line: int

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "bounded_add_range",
            "iterator": self.iterator,
            "range_args": [ast.unparse(x) for x in self.range_args],
            "range_values": list(self.range_values),
            "trip_count": int(self.trip_count),
            "accumulator": self.accumulator,
            "range_arg_authorities": list(self.range_arg_authorities),
            "execution_scope": "source_local_cell_body",
        }
@dataclass(frozen=True)
class BoundedFilteredMinRangeSpec:
    """A bounded streaming filtered minimum produced by vector scalarization."""

    iterator: str
    range_args: tuple[ast.expr, ...]
    item_name: str
    found_name: str
    best_name: str
    value_expr: ast.expr
    predicate: ast.expr
    line: int

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "bounded_filtered_min_range",
            "iterator": self.iterator,
            "range_args": [ast.unparse(x) for x in self.range_args],
            "item_name": self.item_name,
            "found_name": self.found_name,
            "best_name": self.best_name,
            "value_expr": ast.unparse(self.value_expr),
            "predicate": ast.unparse(self.predicate),
            "line": int(self.line),
            "execution_scope": "source_local_cell_body",
            "materializes_python_list": False,
        }

@dataclass(frozen=True)
class BoundedMonotoneFirstHitWhileSpec:
    """A proven finite increasing source-local first-hit ``while`` search.

    I4b-1A directly admits positive source-literal steps and exposes an explicit
    positive-int evidence hook. Production does not supply dynamic-step evidence
    until I4b-1B proves it from existing compiler authority.
    """

    search_name: str
    bound: ast.expr
    comparison: str
    step: ast.expr
    step_value: int | None
    step_authority: str
    predicate: ast.expr
    hit_result: ast.expr
    fallback_result: ast.expr
    update_stmt_id: int
    line: int

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "bounded_monotone_first_hit_while",
            "search_name": self.search_name,
            "bound": ast.unparse(self.bound),
            "comparison": self.comparison,
            "step": ast.unparse(self.step),
            "step_value": None if self.step_value is None else int(self.step_value),
            "step_authority": self.step_authority,
            "predicate": ast.unparse(self.predicate),
            "hit_result": ast.unparse(self.hit_result),
            "fallback_result": ast.unparse(self.fallback_result),
            "line": int(self.line),
            "execution_scope": "source_local_cell_body",
            "overflow_policy": "break_to_search_independent_fallback",
        }


@dataclass(frozen=True)
class FixedCountStateUpdateSpec:
    """A statically bounded local scalar state-update ``for range`` loop.

    This is intentionally a sibling of :class:`FirstHitRangeSpec`, not a
    generalized Python-loop abstraction.  I4a-1 admits ranges whose exact integer
    bounds are present in source.  I4a-2 additionally accepts normalized
    ``global_input`` bounds only when the caller supplies authoritative exact-int
    preparation evidence for those input keys.
    """

    iterator: str
    range_args: tuple[ast.expr, ...]
    range_values: tuple[int, ...]
    trip_count: int
    state_names: tuple[str, ...]
    temporary_names: tuple[str, ...]
    range_arg_authorities: tuple[str, ...]
    line: int

    @property
    def range_arity(self) -> int:
        return len(self.range_args)

    def manifest(self) -> dict[str, object]:
        bound_authority = (
            "literal_range"
            if all(x == "source_literal" for x in self.range_arg_authorities)
            else "prepared_exact_global_input"
        )
        return {
            "kind": "fixed_count_scalar_state_update",
            "iterator": self.iterator,
            "range_args": [ast.unparse(x) for x in self.range_args],
            "range_values": list(self.range_values),
            "trip_count": int(self.trip_count),
            "state_names": list(self.state_names),
            "temporary_names": list(self.temporary_names),
            "line": int(self.line),
            "bound_authority": bound_authority,
            "range_arg_authorities": list(self.range_arg_authorities),
            "execution_scope": "source_local_cell_body",
        }


def _literal_int_value(node: ast.expr) -> int | None:
    """Return an exact signed source integer literal, excluding bool."""

    sign = 1
    value_node: ast.expr = node
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        value_node = node.operand
    if not (isinstance(value_node, ast.Constant) and type(value_node.value) is int):
        return None
    value = sign * int(value_node.value)
    if value < _INT64_MIN or value > _INT64_MAX:
        return None
    return value


def _prepared_exact_global_int_value(
    node: ast.expr,
    prepared_exact_int_bounds: Mapping[str, int],
) -> tuple[int, str] | None:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "global_input"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return None
    key = str(node.args[0].value)
    if key not in prepared_exact_int_bounds:
        return None
    value = prepared_exact_int_bounds[key]
    if type(value) is not int or value < _INT64_MIN or value > _INT64_MAX:
        return None
    return int(value), f"prepared_exact_global_input:{key}"


def _exact_range_values(
    call: ast.Call,
    *,
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> tuple[tuple[int, ...], tuple[str, ...], int] | None:
    if not (
        isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and not call.keywords
        and 1 <= len(call.args) <= 3
    ):
        return None
    prepared_exact_int_bounds = prepared_exact_int_bounds or {}
    values: list[int] = []
    authorities: list[str] = []
    for arg in call.args:
        value = _literal_int_value(arg)
        if value is not None:
            values.append(value)
            authorities.append("source_literal")
            continue
        prepared = _prepared_exact_global_int_value(arg, prepared_exact_int_bounds)
        if prepared is None:
            return None
        prepared_value, authority = prepared
        values.append(prepared_value)
        authorities.append(authority)
    if len(values) == 3 and values[2] == 0:
        return None
    try:
        trip_count = len(range(*values))
    except (OverflowError, ValueError):
        return None
    if trip_count < 0 or trip_count > _INT64_MAX:
        return None
    return tuple(values), tuple(authorities), int(trip_count)


def _simple_initialized_names(statements: Iterable[ast.stmt]) -> set[str]:
    """Names definitely initialized by simple source-order assignments."""

    names: set[str] = set()
    for stmt in statements:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            names.add(stmt.targets[0].id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            names.add(stmt.target.id)
    return names


def _assigned_name(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
        return stmt.target.id
    return None


def _loaded_local_names(node: ast.AST, local_names: set[str]) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in local_names
    }


def recognize_fixed_count_state_update_stmt(
    stmt: ast.stmt,
    *,
    initialized_names: Iterable[str] = (),
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> FixedCountStateUpdateSpec | None:
    """Recognize I4a's bounded local scalar update loop.

    Accepted body statements are plain scalar ``Name`` assignments, nested
    ``if/else`` updates, and ``pass``.  New temporaries may be introduced only
    by unconditional top-level assignments in the loop body.  Conditional
    branches may update only already initialized local scalar names.  This
    avoids silently replacing Python's unbound-local behavior with a Cython
    zero-initialized local.

    A prepared bound such as ``range(global_input('g_11'))`` is admitted only
    when ``prepared_exact_int_bounds`` supplies an exact signed-int64 value for
    that key.  Merely being a ``global_input`` is never sufficient evidence.
    """

    if not (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.iter, ast.Call)
        and not stmt.orelse
        and stmt.body
    ):
        return None
    exact_range = _exact_range_values(
        stmt.iter, prepared_exact_int_bounds=prepared_exact_int_bounds
    )
    if exact_range is None:
        return None
    range_values, range_arg_authorities, trip_count = exact_range

    iterator = stmt.target.id
    preinitialized = set(initialized_names)
    if iterator in preinitialized:
        # Rebinding an existing state name as the range iterator is valid
        # Python, but it is outside the narrow state-update contract.
        return None

    assigned_in_loop = {
        name
        for child in ast.walk(stmt)
        if isinstance(child, (ast.Assign, ast.AnnAssign))
        for name in [_assigned_name(child)]
        if name is not None
    }
    if iterator in assigned_in_loop:
        return None

    definite = set(preinitialized)
    definite.add(iterator)
    state_updates: set[str] = set()
    temporaries: set[str] = set()
    saw_if_else = False

    def check_expr(expr: ast.AST, assigned_locals: set[str], definite_now: set[str]) -> bool:
        return _loaded_local_names(expr, assigned_locals) <= definite_now

    def check_block(
        statements: list[ast.stmt],
        definite_in: set[str],
        *,
        conditional: bool,
    ) -> tuple[bool, set[str]]:
        nonlocal saw_if_else
        current = set(definite_in)
        for child in statements:
            name = _assigned_name(child)
            if name is not None:
                value = child.value  # type: ignore[union-attr]
                if value is None or not check_expr(value, assigned_in_loop, current):
                    return False, current
                if name == iterator:
                    return False, current
                if conditional and name not in current:
                    return False, current
                if name in preinitialized:
                    state_updates.add(name)
                elif name not in current:
                    temporaries.add(name)
                current.add(name)
                continue
            if isinstance(child, ast.If):
                if not child.body or not child.orelse:
                    return False, current
                if not check_expr(child.test, assigned_in_loop, current):
                    return False, current
                saw_if_else = True
                ok_body, body_out = check_block(child.body, set(current), conditional=True)
                ok_else, else_out = check_block(child.orelse, set(current), conditional=True)
                if not ok_body or not ok_else:
                    return False, current
                current |= body_out & else_out
                continue
            if isinstance(child, ast.Pass):
                continue
            # No nested loops, returns, exceptions, mutations, comprehensions,
            # iterator/runtime-object machinery, or other statement effects.
            return False, current
        return True, current

    ok, _ = check_block(stmt.body, definite, conditional=False)
    if not ok or not saw_if_else or not state_updates:
        return None
    return FixedCountStateUpdateSpec(
        iterator=iterator,
        range_args=tuple(stmt.iter.args),
        range_values=range_values,
        trip_count=trip_count,
        state_names=tuple(sorted(state_updates)),
        temporary_names=tuple(sorted(temporaries)),
        range_arg_authorities=range_arg_authorities,
        line=int(getattr(stmt, "lineno", 0) or 0),
    )


def fixed_count_state_update_specs(
    fn: ast.FunctionDef,
    *,
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> tuple[FixedCountStateUpdateSpec, ...]:
    """Return supported top-level I4a loops in source order."""

    out: list[FixedCountStateUpdateSpec] = []
    prefix: list[ast.stmt] = []
    for stmt in fn.body:
        initialized = _simple_initialized_names(prefix)
        spec = recognize_fixed_count_state_update_stmt(
            stmt,
            initialized_names=initialized,
            prepared_exact_int_bounds=prepared_exact_int_bounds,
        )
        if spec is not None:
            out.append(spec)
        prefix.append(stmt)
    return tuple(out)


def fixed_count_state_update_specs_by_for_id(
    fn: ast.FunctionDef,
    *,
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> dict[int, FixedCountStateUpdateSpec]:
    """Map admitted top-level ``For`` node identities to their I4a specs."""

    out: dict[int, FixedCountStateUpdateSpec] = {}
    prefix: list[ast.stmt] = []
    for stmt in fn.body:
        spec = recognize_fixed_count_state_update_stmt(
            stmt,
            initialized_names=_simple_initialized_names(prefix),
            prepared_exact_int_bounds=prepared_exact_int_bounds,
        )
        if spec is not None:
            out[id(stmt)] = spec
        prefix.append(stmt)
    return out


def supported_fixed_count_for_ids(
    fn: ast.FunctionDef,
    *,
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> frozenset[int]:
    """Object identities of I4a fixed-count ``For`` nodes."""

    return frozenset(
        fixed_count_state_update_specs_by_for_id(
            fn, prepared_exact_int_bounds=prepared_exact_int_bounds
        )
    )


def range_call_ids_for_supported_fixed_count(
    fn: ast.FunctionDef,
    *,
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> frozenset[int]:
    """Object identities of proven ``range`` calls owned by I4a loops."""

    supported = fixed_count_state_update_specs_by_for_id(
        fn, prepared_exact_int_bounds=prepared_exact_int_bounds
    )
    return frozenset(
        id(stmt.iter)
        for stmt in fn.body
        if isinstance(stmt, ast.For) and id(stmt) in supported and isinstance(stmt.iter, ast.Call)
    )


def recognize_bounded_add_range_stmt(
    stmt: ast.stmt,
    *,
    initialized_names: Iterable[str] = (),
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> BoundedAddRangeSpec | None:
    """Recognize one exact-bounded scalar additive ``for range`` loop.

    The admitted loop body is deliberately one normalized assignment of the
    form ``acc = acc + expr`` or ``acc = expr + acc``.  The accumulator must be
    definitely initialized before the loop.  Expression/provider safety is
    still checked by the ordinary native residual gate; this recognizer owns
    only the local control/update semantics.
    """

    if not (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.iter, ast.Call)
        and not stmt.orelse
        and len(stmt.body) == 1
    ):
        return None
    exact_range = _exact_range_values(
        stmt.iter, prepared_exact_int_bounds=prepared_exact_int_bounds
    )
    if exact_range is None:
        return None
    range_values, range_arg_authorities, trip_count = exact_range

    iterator = stmt.target.id
    initialized = set(initialized_names)
    if iterator in initialized:
        return None
    update = stmt.body[0]
    acc = _assigned_name(update)
    if acc is None or acc not in initialized or acc == iterator:
        return None
    value = update.value  # type: ignore[union-attr]
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
        return None
    left_is_acc = isinstance(value.left, ast.Name) and value.left.id == acc
    right_is_acc = isinstance(value.right, ast.Name) and value.right.id == acc
    if left_is_acc == right_is_acc:
        # Exactly one side must carry the prior accumulator value.  Rejecting
        # ``acc = acc + acc`` keeps the owned update contract unambiguous.
        return None

    return BoundedAddRangeSpec(
        iterator=iterator,
        range_args=tuple(stmt.iter.args),
        range_values=range_values,
        trip_count=trip_count,
        accumulator=acc,
        range_arg_authorities=range_arg_authorities,
        line=int(getattr(stmt, "lineno", 0) or 0),
    )


def bounded_add_range_specs_by_for_id(
    fn: ast.FunctionDef,
    *,
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> dict[int, BoundedAddRangeSpec]:
    """Return exact-bounded additive loops under structured source ``if`` blocks.

    Definite-local state is propagated by source order and intersected across
    ``if`` branches.  No traversal through arbitrary loops/try/with constructs
    is performed, so a nested additive loop is admitted only inside already
    ordinary scalar ``if`` structure.
    """

    out: dict[int, BoundedAddRangeSpec] = {}

    def scan_block(statements: list[ast.stmt], definite_in: set[str]) -> set[str]:
        current = set(definite_in)
        for child in statements:
            if isinstance(child, ast.If):
                body_out = scan_block(child.body, set(current))
                else_out = (
                    scan_block(child.orelse, set(current))
                    if child.orelse
                    else set(current)
                )
                current = body_out & else_out
                continue
            if isinstance(child, ast.For):
                spec = recognize_bounded_add_range_stmt(
                    child,
                    initialized_names=current,
                    prepared_exact_int_bounds=prepared_exact_int_bounds,
                )
                if spec is not None:
                    out[id(child)] = spec
                # The iterator is not definitely assigned after the loop: an
                # exact Python range may be empty, and an enclosing if may not
                # execute.  Do not manufacture definite-local evidence.
                continue
            name = _assigned_name(child)
            if name is not None:
                current.add(name)
        return current

    scan_block(fn.body, set())

    # Cython locals are zero-initialized.  If the iterator is read outside its
    # own loop, an empty/non-executed range could otherwise mask Python's
    # unbound-local behavior.  Reject such ownership rather than approximate it.
    for for_id in tuple(out):
        loop = next(
            (
                node for node in ast.walk(fn)
                if isinstance(node, ast.For) and id(node) == for_id
            ),
            None,
        )
        if loop is None:
            out.pop(for_id, None)
            continue
        owned = {id(node) for node in ast.walk(loop)}
        iterator = out[for_id].iterator
        if any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == iterator
            and id(node) not in owned
            for node in ast.walk(fn)
        ):
            out.pop(for_id, None)
    return out


def bounded_add_range_specs(
    fn: ast.FunctionDef,
    *,
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> tuple[BoundedAddRangeSpec, ...]:
    by_id = bounded_add_range_specs_by_for_id(
        fn, prepared_exact_int_bounds=prepared_exact_int_bounds
    )
    return tuple(sorted(by_id.values(), key=lambda spec: spec.line))


def supported_bounded_add_for_ids(
    fn: ast.FunctionDef,
    *,
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> frozenset[int]:
    return frozenset(
        bounded_add_range_specs_by_for_id(
            fn, prepared_exact_int_bounds=prepared_exact_int_bounds
        )
    )


def range_call_ids_for_supported_bounded_add(
    fn: ast.FunctionDef,
    *,
    prepared_exact_int_bounds: Mapping[str, int] | None = None,
) -> frozenset[int]:
    supported = bounded_add_range_specs_by_for_id(
        fn, prepared_exact_int_bounds=prepared_exact_int_bounds
    )
    return frozenset(
        id(node.iter)
        for node in ast.walk(fn)
        if isinstance(node, ast.For)
        and id(node) in supported
        and isinstance(node.iter, ast.Call)
    )



def _literal_positive_int64(node: ast.expr) -> int | None:
    value = _literal_int_value(node)
    if value is None or value <= 0:
        return None
    return value


def _source_integer_names(
    statements: Iterable[ast.stmt],
    *,
    base_names: Iterable[str] = (),
) -> set[str]:
    """Names with narrow source-only int64 evidence for I4b-1A.

    This intentionally does not infer Cell-call result types.  A local becomes
    integer here only from an exact signed-int64 source literal.  I4b-1B may
    add authoritative runtime/type evidence without weakening this baseline.
    """

    names = set(base_names)
    for stmt in statements:
        name = _assigned_name(stmt)
        if name is None:
            continue
        value = stmt.value  # type: ignore[union-attr]
        if value is not None and _literal_int_value(value) is not None:
            names.add(name)
    return names


def _is_narrow_int_expr(node: ast.expr, integer_names: set[str]) -> bool:
    if _literal_int_value(node) is not None:
        return True
    if isinstance(node, ast.Name):
        return node.id in integer_names
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_narrow_int_expr(node.operand, integer_names)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        return _is_narrow_int_expr(node.left, integer_names) and _is_narrow_int_expr(
            node.right, integer_names
        )
    if isinstance(node, ast.IfExp):
        return _is_narrow_int_expr(node.body, integer_names) and _is_narrow_int_expr(
            node.orelse, integer_names
        )
    return False


def _positive_step_evidence(
    node: ast.expr, positive_int_names: set[str]
) -> tuple[int | None, str] | None:
    literal = _literal_positive_int64(node)
    if literal is not None:
        return literal, "source_literal_positive_int64"
    if isinstance(node, ast.Name) and node.id in positive_int_names:
        return None, f"proven_positive_int64:{node.id}"
    return None


def _increasing_update(
    stmt: ast.stmt,
    search_name: str,
    *,
    positive_int_names: set[str],
) -> tuple[ast.expr, int | None, str] | None:
    """Return a proven positive update step for ``search_name``."""

    if (
        isinstance(stmt, ast.AugAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id == search_name
        and isinstance(stmt.op, ast.Add)
    ):
        evidence = _positive_step_evidence(stmt.value, positive_int_names)
        return None if evidence is None else (stmt.value, evidence[0], evidence[1])
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == search_name
        and isinstance(stmt.value, ast.BinOp)
        and isinstance(stmt.value.op, ast.Add)
    ):
        left, right = stmt.value.left, stmt.value.right
        if isinstance(left, ast.Name) and left.id == search_name:
            evidence = _positive_step_evidence(right, positive_int_names)
            return None if evidence is None else (right, evidence[0], evidence[1])
        if isinstance(right, ast.Name) and right.id == search_name:
            evidence = _positive_step_evidence(left, positive_int_names)
            return None if evidence is None else (left, evidence[0], evidence[1])
    return None


def recognize_bounded_monotone_first_hit_while_stmt(
    stmt: ast.stmt,
    *,
    initialized_names: Iterable[str] = (),
    integer_names: Iterable[str] = (),
    positive_int_names: Iterable[str] = (),
    fallback_return: ast.Return | None = None,
) -> BoundedMonotoneFirstHitWhileSpec | None:
    """Recognize I4b-1A's finite increasing first-hit ``while`` subset.

    Accepted shape::

        <search> = <proven int64 start>
        while <search> <=|< <proven int64 bound>:
            if <scalar predicate>:
                return <scalar hit result>
            <search> += <proven positive int64 step>
        return <fallback not using search>

    The immediate fallback restriction is intentional.  Native emission uses an
    overflow guard that exits to that fallback rather than allowing C int64 wrap.
    Because the fallback cannot read the search variable, this preserves Python
    behavior when the mathematical next value would exceed int64.
    """

    if not (
        isinstance(stmt, ast.While)
        and not stmt.orelse
        and len(stmt.body) == 2
        and isinstance(stmt.body[0], ast.If)
        and fallback_return is not None
        and fallback_return.value is not None
    ):
        return None
    if_stmt = stmt.body[0]
    if not (
        not if_stmt.orelse
        and len(if_stmt.body) == 1
        and isinstance(if_stmt.body[0], ast.Return)
        and if_stmt.body[0].value is not None
    ):
        return None
    if not (
        isinstance(stmt.test, ast.Compare)
        and len(stmt.test.ops) == 1
        and len(stmt.test.comparators) == 1
        and isinstance(stmt.test.left, ast.Name)
        and isinstance(stmt.test.ops[0], (ast.Lt, ast.LtE))
    ):
        return None

    search_name = stmt.test.left.id
    initialized = set(initialized_names)
    ints = set(integer_names)
    if search_name not in initialized or search_name not in ints:
        return None
    bound = stmt.test.comparators[0]
    if search_name in _loaded_local_names(bound, {search_name}):
        return None
    if not _is_narrow_int_expr(bound, ints):
        return None

    update = _increasing_update(
        stmt.body[1], search_name, positive_int_names=set(positive_int_names)
    )
    if update is None:
        return None
    step, step_value, step_authority = update
    if isinstance(step, ast.Name) and step.id == search_name:
        return None

    # The hit predicate/result may read the search local, but the overflow-safe
    # fallback must not.  Other statement effects are excluded by the exact body.
    if search_name in _loaded_local_names(fallback_return.value, {search_name}):
        return None

    comparison = "<=" if isinstance(stmt.test.ops[0], ast.LtE) else "<"
    return BoundedMonotoneFirstHitWhileSpec(
        search_name=search_name,
        bound=bound,
        comparison=comparison,
        step=step,
        step_value=step_value,
        step_authority=step_authority,
        predicate=if_stmt.test,
        hit_result=if_stmt.body[0].value,
        fallback_result=fallback_return.value,
        update_stmt_id=id(stmt.body[1]),
        line=int(getattr(stmt, "lineno", 0) or 0),
    )


def bounded_monotone_first_hit_while_specs_by_while_id(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
    positive_int_names: Iterable[str] = (),
) -> dict[int, BoundedMonotoneFirstHitWhileSpec]:
    """Map admitted top-level I4b-1A ``While`` node identities to specs."""

    out: dict[int, BoundedMonotoneFirstHitWhileSpec] = {}
    prefix: list[ast.stmt] = []
    base_ints = set(integer_names)
    all_local_names = {
        name
        for node in ast.walk(fn)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for name in [_assigned_name(node)]
        if name is not None
    }
    for idx, stmt in enumerate(fn.body):
        fallback = fn.body[idx + 1] if idx + 1 < len(fn.body) else None
        initialized = _simple_initialized_names(prefix)
        spec = recognize_bounded_monotone_first_hit_while_stmt(
            stmt,
            initialized_names=initialized,
            integer_names=_source_integer_names(prefix, base_names=base_ints),
            positive_int_names=positive_int_names,
            fallback_return=fallback if isinstance(fallback, ast.Return) else None,
        )
        if spec is not None:
            # Do not let Cython's zero-initialized locals mask Python unbound-local
            # behavior.  Every local loaded by the search contract must already
            # be definitely initialized before the loop, except the search name
            # itself which is also required above to be initialized.
            loaded = set()
            for expr in (
                spec.bound, spec.step, spec.predicate,
                spec.hit_result, spec.fallback_result,
            ):
                loaded |= _loaded_local_names(expr, all_local_names)
            if loaded <= initialized:
                out[id(stmt)] = spec
        prefix.append(stmt)
    return out


def bounded_monotone_first_hit_while_specs(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
    positive_int_names: Iterable[str] = (),
) -> tuple[BoundedMonotoneFirstHitWhileSpec, ...]:
    supported = bounded_monotone_first_hit_while_specs_by_while_id(
        fn, integer_names=integer_names, positive_int_names=positive_int_names
    )
    return tuple(
        supported[id(stmt)]
        for stmt in fn.body
        if isinstance(stmt, ast.While) and id(stmt) in supported
    )


def supported_bounded_monotone_while_ids(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
    positive_int_names: Iterable[str] = (),
) -> frozenset[int]:
    return frozenset(
        bounded_monotone_first_hit_while_specs_by_while_id(
            fn, integer_names=integer_names, positive_int_names=positive_int_names
        )
    )


def update_stmt_ids_for_supported_bounded_monotone(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
    positive_int_names: Iterable[str] = (),
) -> frozenset[int]:
    return frozenset(
        spec.update_stmt_id
        for spec in bounded_monotone_first_hit_while_specs_by_while_id(
            fn, integer_names=integer_names, positive_int_names=positive_int_names
        ).values()
    )


def bounded_monotone_positive_evidence_requirements_by_while_id(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
) -> dict[int, tuple[str, ...]]:
    """Evidence-ready I4b loops blocked only by named positive progress.

    This is diagnostic/planning evidence, not admission.  It temporarily grants
    the syntactically named update local positive-int authority and asks the same
    semantic recognizer whether the entire contract would then be valid.  The
    returned name must still be proven by production authority before native
    execution may consume the loop.
    """

    out: dict[int, tuple[str, ...]] = {}
    for stmt in fn.body:
        if not (isinstance(stmt, ast.While) and len(stmt.body) == 2):
            continue
        update = stmt.body[1]
        step_name: str | None = None
        if (
            isinstance(update, ast.AugAssign)
            and isinstance(update.target, ast.Name)
            and isinstance(update.op, ast.Add)
            and isinstance(update.value, ast.Name)
        ):
            step_name = update.value.id
        elif (
            isinstance(update, ast.Assign)
            and len(update.targets) == 1
            and isinstance(update.targets[0], ast.Name)
            and isinstance(update.value, ast.BinOp)
            and isinstance(update.value.op, ast.Add)
        ):
            search_name = update.targets[0].id
            left, right = update.value.left, update.value.right
            if isinstance(left, ast.Name) and left.id == search_name and isinstance(right, ast.Name):
                step_name = right.id
            elif isinstance(right, ast.Name) and right.id == search_name and isinstance(left, ast.Name):
                step_name = left.id
        if step_name is None:
            continue
        supported = bounded_monotone_first_hit_while_specs_by_while_id(
            fn,
            integer_names=integer_names,
            positive_int_names={step_name},
        )
        if id(stmt) in supported:
            out[id(stmt)] = (step_name,)
    return out

def _range_args_are_int64(call: ast.Call, integer_names: set[str]) -> bool:
    if not (
        isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and not call.keywords
        and 1 <= len(call.args) <= 3
    ):
        return False
    if not all(_is_narrow_int_expr(arg, integer_names) for arg in call.args):
        return False
    if len(call.args) == 3:
        step = _literal_int_value(call.args[2])
        if step is None or step == 0:
            return False
    return True

def recognize_bounded_filtered_add_range_stmt(
    stmt: ast.stmt,
    *,
    initialized_names: Iterable[str],
    integer_names: Iterable[str] = (),
) -> BoundedFilteredAddRangeSpec | None:
    """Recognize a dynamic bounded filtered scalar-add loop.

    The admitted subset is intentionally close to existing ``ReductionSpec``
    semantics: one top-level skip ``if: continue`` and one final additive update
    to a scalar that was initialized before the loop. Other statements are plain
    scalar assignments.
    """

    if not (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.iter, ast.Call)
        and not stmt.orelse
        and len(stmt.body) >= 2
        and _range_args_are_int64(stmt.iter, set(integer_names))
    ):
        return None
    continue_nodes = [n for n in ast.walk(stmt) if isinstance(n, ast.Continue)]
    if len(continue_nodes) != 1:
        return None
    skip_positions = [
        i for i, child in enumerate(stmt.body)
        if isinstance(child, ast.If)
        and not child.orelse
        and len(child.body) == 1
        and child.body[0] is continue_nodes[0]
    ]
    if len(skip_positions) != 1:
        return None
    skip_pos = skip_positions[0]
    initialized = set(initialized_names)
    iterator = stmt.target.id
    # Prefix and suffix may contain only scalar assignments; no hidden nested
    # control beyond the one skip filter.
    for i, child in enumerate(stmt.body):
        if i == skip_pos:
            continue
        if _assigned_name(child) is None:
            return None
    final = stmt.body[-1]
    acc = _assigned_name(final)
    if acc is None or acc not in initialized or acc == iterator:
        return None
    value = final.value  # type: ignore[union-attr]
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
        return None
    if not (
        isinstance(value.left, ast.Name) and value.left.id == acc
        or isinstance(value.right, ast.Name) and value.right.id == acc
    ):
        return None
    # The accumulator may not be assigned anywhere else in the loop.
    if sum(1 for child in stmt.body if _assigned_name(child) == acc) != 1:
        return None
    return BoundedFilteredAddRangeSpec(
        iterator=iterator,
        range_args=tuple(stmt.iter.args),
        accumulator=acc,
        line=int(getattr(stmt, "lineno", 0) or 0),
        continue_stmt_id=id(continue_nodes[0]),
    )


_SCALAR_TRUTH_DTYPES = frozenset({"bool", "int64", "float64"})
_ADDITIVE_ACCUMULATOR_DTYPES = frozenset({"int64", "float64"})


def recognize_proven_conditional_add_range_stmt(
    stmt: ast.stmt,
    *,
    initialized_names: Iterable[str],
    name_intervals: Mapping[str, Int64Interval] | None = None,
    call_intervals: Mapping[str, Int64Interval] | None = None,
    input_intervals: Mapping[str, Int64Interval] | None = None,
    scalar_dtypes: Mapping[str, str] | None = None,
) -> BoundedFilteredAddRangeSpec | None:
    """Recognize a proof-backed truth-filtered additive ``range`` loop.

    The owned source shape is intentionally narrow::

        for s in range(...):
            tmp0 = scalar_expr
            ...
            tmpN = scalar_expr
            if scalar_name:
                acc = acc + contribution

    The temporary prefix may be empty, but any temporary that exists is local to
    the descriptor, scalar-typed, single-assignment, and used before loop exit.
    The range itself is admitted only through :func:`prove_int64_range`; merely
    looking integer-like is not sufficient.
    """

    if not (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.iter, ast.Call)
        and not stmt.orelse
        and stmt.body
    ):
        return None
    proof = prove_int64_range(
        stmt.iter,
        name_intervals=name_intervals,
        call_intervals=call_intervals,
        input_intervals=input_intervals,
    )
    if proof is None:
        return None

    # Keep execution on the existing Cython-native ``range`` lowering.  A
    # runtime third argument makes Cython treat range as a Python iterable in
    # nogil code, so proof alone is not sufficient to make a dynamic step an
    # executable native construct.  Implicit +1 and signed source literals are
    # both handled by the established emitter and preserve the old semantics.
    if len(stmt.iter.args) == 3 and _literal_int_value(stmt.iter.args[2]) is None:
        return None

    initialized = set(initialized_names)
    iterator = stmt.target.id
    if iterator in initialized:
        return None

    guard = stmt.body[-1]
    if not (
        isinstance(guard, ast.If)
        and not guard.orelse
        and len(guard.body) == 1
        and isinstance(guard.test, ast.Name)
    ):
        return None

    prefix = stmt.body[:-1]
    temp_names: list[str] = []
    temp_statements: list[ast.stmt] = []
    for child in prefix:
        name = _assigned_name(child)
        if name is None:
            return None
        if name in initialized or name == iterator or name in temp_names:
            return None
        temp_names.append(name)
        temp_statements.append(child)

    dtypes = dict(scalar_dtypes or {})
    dtypes.setdefault(iterator, "int64")
    if any(dtypes.get(name) not in _SCALAR_TRUTH_DTYPES for name in temp_names):
        return None

    # A loop-local temporary may depend on the iterator, pre-loop initialized
    # names, or an earlier temporary, never on a later/unbound loop temporary.
    all_temps = set(temp_names)
    available = initialized | {iterator}
    for name, child in zip(temp_names, temp_statements):
        value = child.value  # type: ignore[union-attr]
        forward = {
            node.id
            for node in ast.walk(value)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in all_temps
            and node.id not in available
        }
        if forward:
            return None
        available.add(name)

    predicate_name = guard.test.id
    if predicate_name not in available:
        return None
    if dtypes.get(predicate_name) not in _SCALAR_TRUTH_DTYPES:
        return None

    update = guard.body[0]
    acc = _assigned_name(update)
    if (
        acc is None
        or acc not in initialized
        or acc == iterator
        or acc in all_temps
    ):
        return None
    if dtypes.get(acc) not in _ADDITIVE_ACCUMULATOR_DTYPES:
        return None
    value = update.value  # type: ignore[union-attr]
    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
        return None
    left_is_acc = isinstance(value.left, ast.Name) and value.left.id == acc
    right_is_acc = isinstance(value.right, ast.Name) and value.right.id == acc
    if left_is_acc == right_is_acc:
        return None
    contribution = value.right if left_is_acc else value.left
    # This is an additive reduction, not a general loop-carried state update.
    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == acc
        for node in ast.walk(contribution)
    ):
        return None

    # Every declared loop-local temporary must feed a later temporary, the
    # predicate, or the contribution. Dead writes are outside this semantic.
    for index, name in enumerate(temp_names):
        later_nodes: list[ast.AST] = [guard]
        later_nodes.extend(
            child.value  # type: ignore[union-attr]
            for child in temp_statements[index + 1 :]
        )
        if not any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == name
            for root in later_nodes
            for node in ast.walk(root)
        ):
            return None

    return BoundedFilteredAddRangeSpec(
        iterator=iterator,
        range_args=tuple(stmt.iter.args),
        accumulator=acc,
        line=int(getattr(stmt, "lineno", 0) or 0),
        predicate=guard.test,
        owned_temporaries=tuple(temp_names),
        filter_mode="truth_guard",
        range_proof=proof,
    )


def bounded_filtered_add_range_specs_by_for_id(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
    name_intervals: Mapping[str, Int64Interval] | None = None,
    call_intervals: Mapping[str, Int64Interval] | None = None,
    input_intervals: Mapping[str, Int64Interval] | None = None,
    scalar_dtypes: Mapping[str, str] | None = None,
) -> dict[int, BoundedFilteredAddRangeSpec]:
    out: dict[int, BoundedFilteredAddRangeSpec] = {}
    prefix: list[ast.stmt] = []
    parameter_names = {
        arg.arg
        for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)
    }
    if fn.args.vararg is not None:
        parameter_names.add(fn.args.vararg.arg)
    if fn.args.kwarg is not None:
        parameter_names.add(fn.args.kwarg.arg)
    for stmt in fn.body:
        initialized = _simple_initialized_names(prefix)
        spec = recognize_proven_conditional_add_range_stmt(
            stmt,
            initialized_names=initialized | parameter_names,
            name_intervals=name_intervals,
            call_intervals=call_intervals,
            input_intervals=input_intervals,
            scalar_dtypes=scalar_dtypes,
        )
        if spec is None:
            spec = recognize_bounded_filtered_add_range_stmt(
                stmt,
                initialized_names=initialized,
                integer_names=integer_names,
            )
        if spec is not None:
            out[id(stmt)] = spec
        prefix.append(stmt)

    # Proof-backed loop temporaries are intentionally owned by the descriptor.
    # A Cython local is zero-initialized, so allowing an iterator/temporary to
    # escape an empty loop would silently differ from Python's unbound-local
    # behavior. Reject the whole semantic instead.
    for for_id, spec in tuple(out.items()):
        if spec.filter_mode != "truth_guard":
            continue
        loop = next(
            (
                node for node in ast.walk(fn)
                if isinstance(node, ast.For) and id(node) == for_id
            ),
            None,
        )
        if loop is None:
            out.pop(for_id, None)
            continue
        owned_node_ids = {id(node) for node in ast.walk(loop)}
        owned_names = {spec.iterator, *spec.owned_temporaries}
        if any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in owned_names
            and id(node) not in owned_node_ids
            for node in ast.walk(fn)
        ):
            out.pop(for_id, None)
    return out

def supported_bounded_filtered_add_for_ids(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
    name_intervals: Mapping[str, Int64Interval] | None = None,
    call_intervals: Mapping[str, Int64Interval] | None = None,
    input_intervals: Mapping[str, Int64Interval] | None = None,
    scalar_dtypes: Mapping[str, str] | None = None,
) -> frozenset[int]:
    return frozenset(bounded_filtered_add_range_specs_by_for_id(
        fn,
        integer_names=integer_names,
        name_intervals=name_intervals,
        call_intervals=call_intervals,
        input_intervals=input_intervals,
        scalar_dtypes=scalar_dtypes,
    ))

def range_call_ids_for_supported_bounded_filtered_add(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
    name_intervals: Mapping[str, Int64Interval] | None = None,
    call_intervals: Mapping[str, Int64Interval] | None = None,
    input_intervals: Mapping[str, Int64Interval] | None = None,
    scalar_dtypes: Mapping[str, str] | None = None,
) -> frozenset[int]:
    supported = bounded_filtered_add_range_specs_by_for_id(
        fn,
        integer_names=integer_names,
        name_intervals=name_intervals,
        call_intervals=call_intervals,
        input_intervals=input_intervals,
        scalar_dtypes=scalar_dtypes,
    )
    return frozenset(
        id(stmt.iter) for stmt in fn.body
        if isinstance(stmt, ast.For) and id(stmt) in supported and isinstance(stmt.iter, ast.Call)
    )

def continue_stmt_ids_for_supported_bounded_filtered_add(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
    name_intervals: Mapping[str, Int64Interval] | None = None,
    call_intervals: Mapping[str, Int64Interval] | None = None,
    input_intervals: Mapping[str, Int64Interval] | None = None,
    scalar_dtypes: Mapping[str, str] | None = None,
) -> frozenset[int]:
    return frozenset(
        spec.continue_stmt_id
        for spec in bounded_filtered_add_range_specs_by_for_id(
            fn,
            integer_names=integer_names,
            name_intervals=name_intervals,
            call_intervals=call_intervals,
            input_intervals=input_intervals,
            scalar_dtypes=scalar_dtypes,
        ).values()
        if spec.continue_stmt_id is not None
    )

def recognize_bounded_filtered_min_range_stmt(
    stmt: ast.stmt,
    *,
    initialized_values: Mapping[str, ast.expr],
    integer_names: Iterable[str] = (),
) -> BoundedFilteredMinRangeSpec | None:
    """Recognize the exact streaming-min loop emitted by the vector bridge."""

    if not (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.iter, ast.Call)
        and not stmt.orelse
        and len(stmt.body) == 2
        and _range_args_are_int64(stmt.iter, set(integer_names))
    ):
        return None
    assign_item, outer = stmt.body
    if not (
        isinstance(assign_item, ast.Assign)
        and len(assign_item.targets) == 1
        and isinstance(assign_item.targets[0], ast.Name)
        and isinstance(outer, ast.If)
        and not outer.orelse
        and len(outer.body) == 1
        and isinstance(outer.body[0], ast.If)
    ):
        return None
    item_name = assign_item.targets[0].id
    inner = outer.body[0]
    if not (
        not inner.orelse
        and isinstance(inner.test, ast.BoolOp)
        and isinstance(inner.test.op, ast.Or)
        and len(inner.test.values) == 2
        and isinstance(inner.test.values[0], ast.UnaryOp)
        and isinstance(inner.test.values[0].op, ast.Not)
        and isinstance(inner.test.values[0].operand, ast.Name)
        and isinstance(inner.test.values[1], ast.Compare)
        and len(inner.test.values[1].ops) == 1
        and isinstance(inner.test.values[1].ops[0], ast.Lt)
        and isinstance(inner.test.values[1].left, ast.Name)
        and inner.test.values[1].left.id == item_name
        and len(inner.test.values[1].comparators) == 1
        and isinstance(inner.test.values[1].comparators[0], ast.Name)
        and len(inner.body) == 2
    ):
        return None
    found_name = inner.test.values[0].operand.id
    best_name = inner.test.values[1].comparators[0].id
    best_set, found_set = inner.body
    if not (
        isinstance(best_set, ast.Assign)
        and len(best_set.targets) == 1
        and isinstance(best_set.targets[0], ast.Name)
        and best_set.targets[0].id == best_name
        and isinstance(best_set.value, ast.Name)
        and best_set.value.id == item_name
        and isinstance(found_set, ast.Assign)
        and len(found_set.targets) == 1
        and isinstance(found_set.targets[0], ast.Name)
        and found_set.targets[0].id == found_name
        and isinstance(found_set.value, ast.Constant)
        and found_set.value.value is True
    ):
        return None
    found_init = initialized_values.get(found_name)
    best_init = initialized_values.get(best_name)
    if not (
        isinstance(found_init, ast.Constant) and found_init.value is False
        and isinstance(best_init, ast.Constant)
        and type(best_init.value) in {int, float}
    ):
        return None
    return BoundedFilteredMinRangeSpec(
        iterator=stmt.target.id,
        range_args=tuple(stmt.iter.args),
        item_name=item_name,
        found_name=found_name,
        best_name=best_name,
        value_expr=assign_item.value,
        predicate=outer.test,
        line=int(getattr(stmt, "lineno", 0) or 0),
    )

def bounded_filtered_min_range_specs_by_for_id(
    fn: ast.FunctionDef,
    *,
    integer_names: Iterable[str] = (),
) -> dict[int, BoundedFilteredMinRangeSpec]:
    out: dict[int, BoundedFilteredMinRangeSpec] = {}
    initialized_values: dict[str, ast.expr] = {}
    ints = set(integer_names)
    for stmt in fn.body:
        spec = recognize_bounded_filtered_min_range_stmt(
            stmt, initialized_values=initialized_values, integer_names=ints
        )
        if spec is not None:
            out[id(stmt)] = spec
        name = _assigned_name(stmt)
        if name is not None:
            initialized_values[name] = stmt.value  # type: ignore[union-attr]
    return out

def supported_bounded_filtered_min_for_ids(
    fn: ast.FunctionDef, *, integer_names: Iterable[str] = ()
) -> frozenset[int]:
    return frozenset(bounded_filtered_min_range_specs_by_for_id(fn, integer_names=integer_names))

def range_call_ids_for_supported_bounded_filtered_min(
    fn: ast.FunctionDef, *, integer_names: Iterable[str] = ()
) -> frozenset[int]:
    supported = bounded_filtered_min_range_specs_by_for_id(fn, integer_names=integer_names)
    return frozenset(
        id(stmt.iter) for stmt in fn.body
        if isinstance(stmt, ast.For) and id(stmt) in supported and isinstance(stmt.iter, ast.Call)
    )

def recognize_first_hit_range_stmt(stmt: ast.stmt) -> FirstHitRangeSpec | None:
    """Recognize the deliberately small first-hit loop subset.

    Accepted shape::

        for q in range(...):
            if <native scalar predicate>:
                return q

    ``range`` may have one or two arguments.  A three-argument range is accepted
    only when the step is a compile-time non-zero integer.  This keeps Python's
    zero-step exception out of the nogil native path while preserving ordinary
    positive/negative literal-step semantics.
    """

    if not (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.iter, ast.Call)
        and isinstance(stmt.iter.func, ast.Name)
        and stmt.iter.func.id == "range"
        and not stmt.iter.keywords
        and 1 <= len(stmt.iter.args) <= 3
        and not stmt.orelse
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], ast.If)
    ):
        return None
    if_stmt = stmt.body[0]
    if if_stmt.orelse or len(if_stmt.body) != 1 or not isinstance(if_stmt.body[0], ast.Return):
        return None
    rv = if_stmt.body[0].value
    if not isinstance(rv, ast.Name) or rv.id != stmt.target.id:
        return None
    if len(stmt.iter.args) == 3:
        step = stmt.iter.args[2]
        if not (isinstance(step, ast.Constant) and isinstance(step.value, int) and step.value != 0):
            return None
    return FirstHitRangeSpec(
        iterator=stmt.target.id,
        range_args=tuple(stmt.iter.args),
        predicate=if_stmt.test,
        line=int(getattr(stmt, "lineno", 0) or 0),
    )


def first_hit_range_specs(fn: ast.FunctionDef) -> tuple[FirstHitRangeSpec, ...]:
    """Return supported top-level first-hit loops in source order."""

    return tuple(
        spec
        for stmt in fn.body
        if (spec := recognize_first_hit_range_stmt(stmt)) is not None
    )


def supported_first_hit_for_ids(fn: ast.FunctionDef) -> frozenset[int]:
    """Object identities of ``For`` nodes admitted by the recognizer."""

    return frozenset(
        id(stmt)
        for stmt in fn.body
        if recognize_first_hit_range_stmt(stmt) is not None
    )


def range_call_ids_for_supported_first_hit(fn: ast.FunctionDef) -> frozenset[int]:
    """Object identities of range Calls owned by admitted first-hit loops."""

    return frozenset(
        id(stmt.iter)
        for stmt in fn.body
        if isinstance(stmt, ast.For) and recognize_first_hit_range_stmt(stmt) is not None
    )


def iter_first_hit_for_statements(fn: ast.FunctionDef) -> Iterable[ast.For]:
    for stmt in fn.body:
        if isinstance(stmt, ast.For) and recognize_first_hit_range_stmt(stmt) is not None:
            yield stmt


__all__ = [
    "FirstHitRangeSpec",
    "BoundedMonotoneFirstHitWhileSpec",
    "BoundedFilteredMinRangeSpec",
    "BoundedFilteredAddRangeSpec",
    "BoundedAddRangeSpec",
    "FixedCountStateUpdateSpec",
    "recognize_first_hit_range_stmt",
    "recognize_bounded_monotone_first_hit_while_stmt",
    "recognize_bounded_filtered_min_range_stmt",
    "recognize_bounded_filtered_add_range_stmt",
    "recognize_proven_conditional_add_range_stmt",
    "recognize_bounded_add_range_stmt",
    "recognize_fixed_count_state_update_stmt",
    "first_hit_range_specs",
    "bounded_monotone_first_hit_while_specs",
    "bounded_monotone_first_hit_while_specs_by_while_id",
    "bounded_filtered_min_range_specs_by_for_id",
    "bounded_filtered_add_range_specs_by_for_id",
    "bounded_add_range_specs",
    "bounded_add_range_specs_by_for_id",
    "fixed_count_state_update_specs",
    "fixed_count_state_update_specs_by_for_id",
    "supported_first_hit_for_ids",
    "supported_bounded_monotone_while_ids",
    "supported_bounded_filtered_min_for_ids",
    "supported_bounded_filtered_add_for_ids",
    "supported_bounded_add_for_ids",
    "supported_fixed_count_for_ids",
    "range_call_ids_for_supported_first_hit",
    "range_call_ids_for_supported_bounded_filtered_min",
    "range_call_ids_for_supported_bounded_filtered_add",
    "range_call_ids_for_supported_bounded_add",
    "continue_stmt_ids_for_supported_bounded_filtered_add",
    "update_stmt_ids_for_supported_bounded_monotone",
    "bounded_monotone_positive_evidence_requirements_by_while_id",
    "range_call_ids_for_supported_fixed_count",
    "iter_first_hit_for_statements",
]
