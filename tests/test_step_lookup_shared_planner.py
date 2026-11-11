from __future__ import annotations

import ast
import inspect

from modelx_graph.frontend import GraphModelCompiler


def _nested_function_source(name: str) -> str:
    source = inspect.getsource(GraphModelCompiler._normalize_external_tables)
    tree = ast.parse(inspect.cleandoc(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"nested function {name!r} not found")


def test_step_lookup_has_one_physical_leaf_builder():
    source = inspect.getsource(GraphModelCompiler._normalize_external_tables)
    # The preparation ABI should be allocated in exactly one shared helper, not in
    # each source recognizer independently.
    assert source.count("registry.external_dataframe_step_lookup(") == 1
    physical = _nested_function_source("_step_lookup_leaf_call")
    assert "registry.external_dataframe_step_lookup" in physical
    assert "__step_lookup_1d__" in physical


def test_legacy_and_semantic_asof_share_step_lookup_physicalizer():
    semantic = _nested_function_source("_rewrite_semantic_asof_lookup_function")
    legacy = _nested_function_source("_rewrite_step_lookup_function")

    assert "external_dataframe_step_lookup" not in semantic
    assert "external_dataframe_step_lookup" not in legacy
    assert "_emit_step_lookup_dispatch" in semantic
    assert "_rewrite_step_lookup_result" in legacy
    assert "_plan_step_lookup_prefix" in semantic
    assert "_plan_step_lookup_prefix" in legacy
    assert "_plan_step_lookup_columns" in semantic
    assert "_plan_step_lookup_columns" in legacy
