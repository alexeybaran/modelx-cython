from __future__ import annotations

"""Backend-neutral semantic formula IR boundary for fast compilation.

Phase 1 deliberately keeps the payload as normalized Python AST.  Later phases may
add first-class table/lookup nodes without changing the contract between semantic
normalization and physical Cython lowering.
"""

from dataclasses import dataclass, replace
import ast
import copy
import hashlib
from typing import Any


def ast_fingerprint(node: ast.AST) -> str:
    """Return a stable structural fingerprint that ignores source locations."""
    payload = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticPassRecord:
    """Deterministic evidence for one semantic normalization pass."""

    name: str
    changed: bool
    before_sha256: str
    after_sha256: str
    notes: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "changed": bool(self.changed),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AsofLookupIR:
    """Semantic ordered lookup independent of pandas and physical storage.

    ``equality_prefix`` fixes every leading table-index level; ``ordered_query``
    selects the greatest residual ordered key not exceeding the query for the
    current ``backward`` direction.  AST expressions are semantic operands, not
    executable pandas objects.
    """

    table_expr: ast.AST
    equality_prefix: ast.AST
    ordered_query: ast.AST
    value_selector: ast.AST
    direction: str = "backward"
    allow_exact: bool = True
    missing: str = "raise"
    source_ast_sha256: str = ""
    schema: str = "modelx_graph.asof_lookup_ir.v1"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "table_expr": ast.unparse(self.table_expr),
            "equality_prefix": ast.unparse(self.equality_prefix),
            "ordered_query": ast.unparse(self.ordered_query),
            "value_selector": ast.unparse(self.value_selector),
            "direction": self.direction,
            "allow_exact": bool(self.allow_exact),
            "missing": self.missing,
            "source_ast_sha256": self.source_ast_sha256,
        }


@dataclass(frozen=True)
class SemanticFormulaIR:
    """One formula after semantic normalization and before physical lowering.

    ``function`` is intentionally still an AST in Phase 1.  It is the sole
    semantic input to later lowering for this formula; source syntax remains
    provenance only.  ``operator_uids`` records semantic operators registered by
    normalization (for example table lookups) so future passes can replace AST
    idioms with first-class IR nodes without changing downstream ownership.
    """

    uid: str
    source_name: str
    function: ast.FunctionDef
    pass_records: tuple[SemanticPassRecord, ...] = ()
    operator_uids: tuple[str, ...] = ()
    semantic_ops: tuple[Any, ...] = ()
    schema: str = "modelx_graph.semantic_formula_ir.v2"

    @classmethod
    def from_function(
        cls, uid: str, source_name: str, function: ast.FunctionDef
    ) -> "SemanticFormulaIR":
        copied = copy.deepcopy(function)
        ast.fix_missing_locations(copied)
        return cls(uid=uid, source_name=source_name, function=copied)

    def with_function(
        self,
        function: ast.FunctionDef,
        *,
        pass_name: str,
        notes: tuple[str, ...] = (),
    ) -> "SemanticFormulaIR":
        if not isinstance(function, ast.FunctionDef):
            raise TypeError(
                f"semantic pass {pass_name!r} returned {type(function).__name__}; "
                "expected ast.FunctionDef"
            )
        updated = copy.deepcopy(function)
        ast.fix_missing_locations(updated)
        before = ast_fingerprint(self.function)
        after = ast_fingerprint(updated)
        record = SemanticPassRecord(
            name=pass_name,
            changed=before != after,
            before_sha256=before,
            after_sha256=after,
            notes=tuple(notes),
        )
        return replace(
            self,
            function=updated,
            pass_records=self.pass_records + (record,),
        )

    def with_operator_uids(self, operator_uids: tuple[str, ...]) -> "SemanticFormulaIR":
        return replace(self, operator_uids=tuple(sorted(set(operator_uids))))

    def with_semantic_ops(self, semantic_ops: tuple[Any, ...]) -> "SemanticFormulaIR":
        return replace(self, semantic_ops=tuple(semantic_ops))

    @property
    def fingerprint(self) -> str:
        return ast_fingerprint(self.function)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "uid": self.uid,
            "source_name": self.source_name,
            "semantic_ast_sha256": self.fingerprint,
            "pass_count": len(self.pass_records),
            "passes": [row.manifest() for row in self.pass_records],
            "operator_uids": list(self.operator_uids),
            "semantic_ops": [
                op.manifest() if hasattr(op, "manifest") else {"type": type(op).__name__}
                for op in self.semantic_ops
            ],
        }


@dataclass(frozen=True)
class FormulaNormalizationResult:
    """Semantic IR plus the existing physical-lowering compatibility result.

    ``guards`` contains terminal guards introduced *after* the original source
    guard-normalization pass.  This is needed for semantic IR operations whose
    Python error contract must survive physical lowering (for example an as-of
    lookup whose source uses ``max([])`` below the first anchor).
    """

    semantic: SemanticFormulaIR | None
    function: ast.FunctionDef | None
    reduction: Any | None
    blockers: tuple[tuple[str, str], ...] = ()
    guards: tuple[Any, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "semantic": None if self.semantic is None else self.semantic.manifest(),
            "has_lowered_function": self.function is not None,
            "has_reduction": self.reduction is not None,
            "blockers": [
                {"kind": kind, "detail": detail} for kind, detail in self.blockers
            ],
            "guards": [
                guard.manifest() if hasattr(guard, "manifest") else repr(guard)
                for guard in self.guards
            ],
        }
