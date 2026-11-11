from __future__ import annotations

import ast

from modelx_graph.lookup_semantics import recognize_asof_lookups
from modelx_graph.semantic_ir import AsofLookupIR, SemanticFormulaIR
from modelx_graph.modelx_cython_fast_backend import (
    _FormulaSource, _selected_guarded_parameter_lower_bounds,
)


def fn(source: str) -> ast.FunctionDef:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def formula(uid: str, source: str, parameters=()):
    f = fn(source)
    return _FormulaSource(
        uid=uid, name=uid, fullname=uid, function=f,
        parameters=tuple(parameters), defaults={}, guards=(), lexical_source=source,
    )


def test_recognizes_generic_grouped_backward_asof_lookup():
    f = fn('''
def rate(t, column):
    sub = tables.rates().loc[group()]
    keys = [k for k in sub.index if k <= t]
    return float(sub.loc[max(keys), column])
''')
    ops = recognize_asof_lookups(f)
    assert len(ops) == 1
    op = ops[0]
    assert isinstance(op, AsofLookupIR)
    assert ast.unparse(op.table_expr) == "tables.rates()"
    assert ast.unparse(op.equality_prefix) == "group()"
    assert ast.unparse(op.ordered_query) == "t"
    assert ast.unparse(op.value_selector) == "column"
    assert op.direction == "backward"
    assert op.allow_exact is True
    assert op.missing == "raise"

    ir = SemanticFormulaIR.from_function("u", "rate", f).with_semantic_ops(ops)
    manifest = ir.manifest()
    assert manifest["semantic_ops"][0]["schema"] == "modelx_graph.asof_lookup_ir.v1"


def test_rejects_forward_or_non_exact_source_shapes():
    forward = fn('''
def rate(t, column):
    sub = tables.rates().loc[group()]
    keys = [k for k in sub.index if k >= t]
    return float(sub.loc[min(keys), column])
''')
    assert recognize_asof_lookups(forward) == ()


def test_range_comprehension_lower_bound_propagates_to_asof_parameter():
    # Root output builds an aligned result vector over ``range`` through a local
    # sequence alias, then helper->lookup forwards the coordinate.  This is the
    # generic source shape that previously left the lookup query lower bound unknown.
    formulas = {
        "root": formula("root", '''
def root():
    ts = list(range(limit()))
    return sum(helper(t) for t in ts)
'''),
        "helper": formula("helper", '''
def helper(t):
    return lookup(t)
''', ("t",)),
        "lookup": formula("lookup", '''
def lookup(t):
    sub = tables.rates().loc[group()]
    keys = [k for k in sub.index if k <= t]
    return sub.loc[max(keys), "x"]
''', ("t",)),
        "limit": formula("limit", '''
def limit():
    return 10
'''),
        "group": formula("group", '''
def group():
    return 0
'''),
    }
    proven, sources = _selected_guarded_parameter_lower_bounds(formulas, "root")
    assert proven[("lookup", "t")] == 0.0
    assert "selected_closure_source_calls" in sources[("lookup", "t")]
