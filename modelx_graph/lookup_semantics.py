from __future__ import annotations

"""Backend-neutral recognition of ordered lookup semantics.

The recognizer deliberately understands a very small Python/pandas surface and
returns a semantic operation rather than materializing tables or choosing a runtime
kernel.  Provider binding and Cython lowering remain downstream concerns.
"""

import ast
import copy
from typing import Iterable

from .semantic_ir import AsofLookupIR, ast_fingerprint


def _unwrap_float(expr: ast.AST) -> ast.AST:
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "float"
        and len(expr.args) == 1
        and not expr.keywords
    ):
        return expr.args[0]
    return expr


def _is_max_name(expr: ast.AST, name: str) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "max"
        and len(expr.args) == 1
        and not expr.keywords
        and isinstance(expr.args[0], ast.Name)
        and expr.args[0].id == name
    )


def recognize_asof_lookups(fn: ast.FunctionDef) -> tuple[AsofLookupIR, ...]:
    """Recognize exact ``last key <= query`` scalar lookup idioms.

    Accepted whole-function shape (a leading docstring is ignored)::

        sub = table_expr.loc[prefix]
        keys = [k for k in sub.index if k <= query]
        return float(sub.loc[max(keys), value_selector])

    The source semantics of bare ``max(keys)`` are preserved as ``missing='raise'``.
    No provider/table inspection occurs here.  A later physical pass may lower the
    operation only when the ordered domain and finite selectors are proved.
    """

    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 3:
        return ()

    first, second, third = body
    if not (
        isinstance(first, ast.Assign)
        and len(first.targets) == 1
        and isinstance(first.targets[0], ast.Name)
        and isinstance(first.value, ast.Subscript)
        and isinstance(first.value.value, ast.Attribute)
        and first.value.value.attr == "loc"
    ):
        return ()
    sub_name = first.targets[0].id
    table_expr = first.value.value.value
    prefix_expr = first.value.slice

    if not (
        isinstance(second, ast.Assign)
        and len(second.targets) == 1
        and isinstance(second.targets[0], ast.Name)
        and isinstance(second.value, ast.ListComp)
        and len(second.value.generators) == 1
    ):
        return ()
    keys_name = second.targets[0].id
    comp = second.value.generators[0]
    if not (
        isinstance(comp.target, ast.Name)
        and not comp.is_async
        and len(comp.ifs) == 1
        and isinstance(comp.iter, ast.Attribute)
        and comp.iter.attr == "index"
        and isinstance(comp.iter.value, ast.Name)
        and comp.iter.value.id == sub_name
        and isinstance(second.value.elt, ast.Name)
        and second.value.elt.id == comp.target.id
    ):
        return ()
    cond = comp.ifs[0]
    if not (
        isinstance(cond, ast.Compare)
        and len(cond.ops) == 1
        and isinstance(cond.ops[0], ast.LtE)
        and len(cond.comparators) == 1
        and isinstance(cond.left, ast.Name)
        and cond.left.id == comp.target.id
    ):
        return ()
    query_expr = cond.comparators[0]

    if not isinstance(third, ast.Return) or third.value is None:
        return ()
    value = _unwrap_float(third.value)
    if not (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Attribute)
        and value.value.attr == "loc"
        and isinstance(value.value.value, ast.Name)
        and value.value.value.id == sub_name
        and isinstance(value.slice, ast.Tuple)
        and len(value.slice.elts) == 2
    ):
        return ()
    ordered_key, value_selector = value.slice.elts
    if not _is_max_name(ordered_key, keys_name):
        return ()

    return (
        AsofLookupIR(
            table_expr=copy.deepcopy(table_expr),
            equality_prefix=copy.deepcopy(prefix_expr),
            ordered_query=copy.deepcopy(query_expr),
            value_selector=copy.deepcopy(value_selector),
            direction="backward",
            allow_exact=True,
            missing="raise",
            source_ast_sha256=ast_fingerprint(fn),
        ),
    )
