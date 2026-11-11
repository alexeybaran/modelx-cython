from __future__ import annotations

"""Graph-free reusable source semantics harvested from the legacy compiler.

Nothing in this module constructs a graph, Stage, storage plan, or executable
kernel.  The transforms operate only on formula ASTs and are therefore safe to
share between diagnostic tooling and the fast recursive frontend.
"""

import ast
import copy
from dataclasses import dataclass
from typing import Any

from .local_control_semantics import (
    bounded_add_range_specs,
    bounded_monotone_first_hit_while_specs,
    first_hit_range_specs,
    fixed_count_state_update_specs,
)


class _SubstituteLiteralNames(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, Any]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(ast.Constant(self.mapping[node.id]), node)
        return node


class UnrollLiteralReductions(ast.NodeTransformer):
    """Unroll ``sum`` over a finite literal one-dimensional domain.

    This is the execution-independent semantic core of the historical
    ``_fast_reference.frontend._UnrollLiteralReductions`` implementation.
    """

    @staticmethod
    def _generator(node: ast.AST) -> ast.GeneratorExp | None:
        if isinstance(node, ast.GeneratorExp):
            return node
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"list", "tuple"}
            and len(node.args) == 1
            and isinstance(node.args[0], ast.GeneratorExp)
        ):
            return node.args[0]
        return None

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "sum"
            and 1 <= len(node.args) <= 2
        ):
            return node
        gen = self._generator(node.args[0])
        if gen is None or len(gen.generators) != 1:
            return node
        comp = gen.generators[0]
        if comp.ifs or comp.is_async or not isinstance(comp.target, ast.Name):
            return node
        if not isinstance(comp.iter, (ast.Tuple, ast.List, ast.Set)):
            return node
        values: list[Any] = []
        for element in comp.iter.elts:
            try:
                values.append(ast.literal_eval(element))
            except Exception:
                return node
        terms = [
            _SubstituteLiteralNames({comp.target.id: value}).visit(copy.deepcopy(gen.elt))
            for value in values
        ]
        result: ast.AST = copy.deepcopy(node.args[1]) if len(node.args) == 2 else ast.Constant(0)
        for term in terms:
            result = ast.BinOp(left=result, op=ast.Add(), right=term)
        return ast.copy_location(ast.fix_missing_locations(result), node)


def unroll_literal_reductions(fn: ast.FunctionDef) -> ast.FunctionDef:
    out = UnrollLiteralReductions().visit(copy.deepcopy(fn))
    if not isinstance(out, ast.FunctionDef):
        raise TypeError("literal-reduction transform did not preserve FunctionDef")
    return ast.fix_missing_locations(out)


@dataclass(frozen=True)
class SemanticOracleReport:
    source: str
    recognized: tuple[dict[str, object], ...]
    transformed_source: str
    transformations: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "modelx_graph.semantic_oracle.v1",
            "source": self.source,
            "recognized": list(self.recognized),
            "transformed_source": self.transformed_source,
            "transformations": list(self.transformations),
            "execution_framework": "none",
            "stage_constructed": False,
        }


def diagnose_function_semantics(fn: ast.FunctionDef) -> SemanticOracleReport:
    """Run source-only legacy/shared recognition without building execution IR."""

    original = ast.fix_missing_locations(copy.deepcopy(fn))
    recognized: list[dict[str, object]] = [spec.manifest() for spec in first_hit_range_specs(original)]
    recognized.extend(spec.manifest() for spec in fixed_count_state_update_specs(original))
    recognized.extend(spec.manifest() for spec in bounded_add_range_specs(original))
    recognized.extend(spec.manifest() for spec in bounded_monotone_first_hit_while_specs(original))
    transformed = unroll_literal_reductions(original)
    transformations: list[str] = []
    if ast.dump(original, include_attributes=False) != ast.dump(transformed, include_attributes=False):
        transformations.append("literal_reduction_unroll")
        recognized.append({
            "kind": "finite_literal_reduction",
            "owner": "semantic_reuse.UnrollLiteralReductions",
        })
    return SemanticOracleReport(
        source=ast.unparse(original),
        recognized=tuple(recognized),
        transformed_source=ast.unparse(transformed),
        transformations=tuple(transformations),
    )


__all__ = [
    "UnrollLiteralReductions",
    "unroll_literal_reductions",
    "SemanticOracleReport",
    "diagnose_function_semantics",
]
