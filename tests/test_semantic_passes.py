from __future__ import annotations

import ast
import pytest

from modelx_graph.semantic_ir import SemanticFormulaIR
from modelx_graph.semantic_passes import AstSemanticPass, SemanticPassError, SemanticPassPipeline


def parse_function(source: str) -> ast.FunctionDef:
    fn = ast.parse(source).body[0]
    assert isinstance(fn, ast.FunctionDef)
    return fn


def test_pipeline_is_ordered_and_records_structural_changes():
    ir = SemanticFormulaIR.from_function("u1", "f", parse_function("def f():\n    return x + 0\n"))

    class ZeroToOne(ast.NodeTransformer):
        def visit_Constant(self, node):  # noqa: N802
            if node.value == 0:
                return ast.copy_location(ast.Constant(1), node)
            return node

    def rename_x(fn: ast.FunctionDef) -> ast.FunctionDef:
        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):  # noqa: N802
                if node.id == "x":
                    return ast.copy_location(ast.Name("y", node.ctx), node)
                return node
        out = Rename().visit(fn)
        assert isinstance(out, ast.FunctionDef)
        return ast.fix_missing_locations(out)

    pipeline = SemanticPassPipeline((
        AstSemanticPass("zero_to_one", lambda fn: ZeroToOne().visit(fn)),
        AstSemanticPass("rename_x", rename_x),
    ))
    result = pipeline.run(ir)

    assert ast.unparse(result.function).endswith("return y + 1")
    assert [row.name for row in result.pass_records] == ["zero_to_one", "rename_x"]
    assert all(row.changed for row in result.pass_records)
    # The original IR is immutable at the semantic boundary even when a
    # NodeTransformer mutates its input tree.
    assert ast.unparse(ir.function).endswith("return x + 0")


def test_duplicate_pass_names_fail_closed():
    identity = lambda fn: fn
    with pytest.raises(SemanticPassError, match="duplicate semantic pass names"):
        SemanticPassPipeline((
            AstSemanticPass("same", identity),
            AstSemanticPass("same", identity),
        ))


def test_non_function_result_fails_closed():
    ir = SemanticFormulaIR.from_function("u1", "f", parse_function("def f():\n    return 1\n"))
    pipeline = SemanticPassPipeline((AstSemanticPass("bad", lambda fn: ast.Constant(1)),))
    with pytest.raises(SemanticPassError, match="expected ast.FunctionDef"):
        pipeline.run(ir)


def test_manifest_is_deterministic_and_operator_ids_are_sorted():
    ir = SemanticFormulaIR.from_function("u1", "f", parse_function("def f():\n    return 1\n"))
    ir = SemanticPassPipeline((AstSemanticPass("identity", lambda fn: fn),)).run(ir)
    ir = ir.with_operator_uids(("z", "a", "z"))
    manifest = ir.manifest()
    assert manifest["operator_uids"] == ["a", "z"]
    assert manifest["passes"][0]["changed"] is False
    assert manifest["passes"][0]["before_sha256"] == manifest["passes"][0]["after_sha256"]
