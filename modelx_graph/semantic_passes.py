from __future__ import annotations

"""Small ordered pass framework for the semantic formula boundary.

This module is intentionally independent of modelx, pandas and Cython.  Compiler
frontends supply callables that map one ``ast.FunctionDef`` to another; the
pipeline owns ordering, structural validation and deterministic pass evidence.
"""

from dataclasses import dataclass
import ast
import copy
from typing import Callable, Iterable

from .semantic_ir import SemanticFormulaIR


class SemanticPassError(RuntimeError):
    pass


@dataclass(frozen=True)
class AstSemanticPass:
    name: str
    transform: Callable[[ast.FunctionDef], ast.FunctionDef]
    notes: tuple[str, ...] = ()

    def run(self, ir: SemanticFormulaIR) -> SemanticFormulaIR:
        # Preserve the compiler's existing exception taxonomy and messages.
        # Phase 1 is structural: pass ownership must not change admission or
        # fallback diagnostics produced by the mature normalizers.
        result = self.transform(copy.deepcopy(ir.function))
        if not isinstance(result, ast.FunctionDef):
            raise SemanticPassError(
                f"semantic pass {self.name!r} returned {type(result).__name__} "
                f"for {ir.uid}; expected ast.FunctionDef"
            )
        return ir.with_function(result, pass_name=self.name, notes=self.notes)


class SemanticPassPipeline:
    """Run named semantic passes exactly once, in declared order."""

    def __init__(self, passes: Iterable[AstSemanticPass], *, name: str = "semantic") -> None:
        rows = tuple(passes)
        names = tuple(row.name for row in rows)
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise SemanticPassError(
                f"duplicate semantic pass names in {name!r}: {duplicates}"
            )
        self.name = name
        self.passes = rows

    def run(self, ir: SemanticFormulaIR) -> SemanticFormulaIR:
        if not isinstance(ir, SemanticFormulaIR):
            raise TypeError("SemanticPassPipeline requires SemanticFormulaIR")
        current = ir
        for semantic_pass in self.passes:
            current = semantic_pass.run(current)
        return current

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "modelx_graph.semantic_pass_pipeline.v1",
            "name": self.name,
            "passes": [row.name for row in self.passes],
        }
