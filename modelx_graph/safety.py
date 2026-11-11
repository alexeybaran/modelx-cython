from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

from .program_semantics import guard_code_from_stmt, ProgramSemanticError


class SemanticSafetyError(RuntimeError):
    """Raised when a formula cannot be proven safe for native lowering."""


@dataclass(frozen=True)
class SafetyResult:
    local_names: tuple[str, ...]
    total_return: bool


def _loaded_names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _assigned_names(fn: ast.FunctionDef) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
                else:
                    raise SemanticSafetyError(
                        f"{fn.name}: only simple local-name assignment is supported by the native fast path"
                    )
        elif isinstance(n, ast.AnnAssign):
            if isinstance(n.target, ast.Name):
                out.add(n.target.id)
            else:
                raise SemanticSafetyError(
                    f"{fn.name}: only simple annotated local-name assignment is supported by the native fast path"
                )
        elif isinstance(n, (ast.AugAssign, ast.NamedExpr)):
            # Reductions are normalized separately.  Ordinary formulas deliberately
            # reject these rather than guessing mutation semantics.
            raise SemanticSafetyError(
                f"{fn.name}: {type(n).__name__} requires fallback outside an explicitly normalized reduction"
            )
    return out


def validate_formula_control_flow(fn: ast.FunctionDef) -> SafetyResult:
    """Conservatively validate Python control-flow and definite assignment.

    The contract is fail-closed: if every path cannot be represented with the
    generated native helper semantics, translation is rejected.  In particular,
    no unassigned local may be read and no path may fall off the end and silently
    become a C zero/previous value.
    """
    locals_ = _assigned_names(fn)

    def check_expr(expr: ast.AST | None, assigned: frozenset[str]) -> None:
        if expr is None:
            return
        missing = sorted((_loaded_names(expr) & locals_) - set(assigned))
        if missing:
            raise SemanticSafetyError(
                f"{fn.name}: local(s) {missing} can be read before definite assignment"
            )

    def block(stmts: Iterable[ast.stmt], in_states: set[frozenset[str]]) -> set[frozenset[str]]:
        states = set(in_states)
        for st in stmts:
            if not states:
                break
            next_states: set[frozenset[str]] = set()
            for assigned in states:
                if guard_code_from_stmt(st) is not None:
                    # Backend-neutral terminal guard: this path raises/fails closed.
                    # There is intentionally no continuation state.
                    pass
                elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                    next_states.add(assigned)
                elif isinstance(st, ast.Pass):
                    next_states.add(assigned)
                elif isinstance(st, ast.Assign):
                    if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                        raise SemanticSafetyError(f"{fn.name}: complex assignment requires fallback")
                    check_expr(st.value, assigned)
                    next_states.add(frozenset(set(assigned) | {st.targets[0].id}))
                elif isinstance(st, ast.AnnAssign):
                    if not isinstance(st.target, ast.Name) or st.value is None:
                        raise SemanticSafetyError(f"{fn.name}: unsupported annotated assignment")
                    check_expr(st.value, assigned)
                    next_states.add(frozenset(set(assigned) | {st.target.id}))
                elif isinstance(st, ast.Return):
                    if st.value is None:
                        raise SemanticSafetyError(f"{fn.name}: bare return is outside the numeric native subset")
                    check_expr(st.value, assigned)
                    # No continuation state: this path returned.
                elif isinstance(st, ast.If):
                    check_expr(st.test, assigned)
                    body_states = block(st.body, {assigned})
                    else_states = block(st.orelse, {assigned}) if st.orelse else {assigned}
                    next_states.update(body_states)
                    next_states.update(else_states)
                elif isinstance(st, ast.Raise):
                    raise SemanticSafetyError(f"{fn.name}: unnormalized raise requires fallback")
                else:
                    raise SemanticSafetyError(
                        f"{fn.name}: statement {type(st).__name__} is outside the verified native subset"
                    )
            states = next_states
        return states

    continuing = block(fn.body, {frozenset()})
    if continuing:
        raise SemanticSafetyError(
            f"{fn.name}: at least one control-flow path falls off the end without returning a numeric value"
        )
    return SafetyResult(tuple(sorted(locals_)), True)
