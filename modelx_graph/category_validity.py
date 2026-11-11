from __future__ import annotations

"""Validity-aware exact consumers for finite category codes.

The ordinary exact-axis intrinsic remains the zero-overhead path for non-nullable
categories.  Nullable prepared point categories receive an explicit sibling validity
input (Block 2A); exact consumers must use that state before physical indexing.
"""

import ast
from typing import Any


class CategoryValidityError(RuntimeError):
    pass


def point_category_exact_axis_call(
    registry: Any,
    input_key: str,
    *,
    size: int,
    require_validity: bool,
    start: int = 0,
    step: int = 1,
) -> ast.Call:
    """Build an exact finite-category axis code with optional validity check.

    ``require_validity=True`` is fail-closed: the prepared category code must expose
    the explicit Block-2A validity sibling.  The returned intrinsic raises natively
    on invalid/missing state before a table/array index can be formed.
    """

    spec = registry.specs.get(input_key)
    if spec is None or spec.enum_labels is None:
        raise CategoryValidityError(
            f"exact category selector {input_key!r} is not a frozen enum input"
        )
    if int(size) <= 0:
        raise CategoryValidityError("exact category axis size must be positive")
    code = ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(input_key)], [])
    if not require_validity:
        return ast.Call(
            ast.Name("__exact_axis_code__", ast.Load()),
            [code, ast.Constant(int(start)), ast.Constant(int(step)), ast.Constant(int(size))],
            [],
        )
    validity_key = spec.category_validity_key
    if not validity_key:
        raise CategoryValidityError(
            f"nullable exact category selector {input_key!r} has no explicit validity input"
        )
    validity_spec = registry.specs.get(validity_key)
    if validity_spec is None or validity_spec.dtype != "bool" or validity_spec.ndim != 1:
        raise CategoryValidityError(
            f"nullable exact category selector {input_key!r} has invalid validity input {validity_key!r}"
        )
    valid = ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(validity_key)], [])
    return ast.Call(
        ast.Name("__exact_axis_code_valid__", ast.Load()),
        [
            code,
            valid,
            ast.Constant(int(start)),
            ast.Constant(int(step)),
            ast.Constant(int(size)),
        ],
        [],
    )


__all__ = ["CategoryValidityError", "point_category_exact_axis_call"]
