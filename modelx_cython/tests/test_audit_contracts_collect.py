"""Regression cases found by the architecture/genericity audit: collect admission."""
import ast
from types import SimpleNamespace

import libcst as cst


def test_collection_admission_reuses_the_existing_syntax_without_ast(monkeypatch):
    from modelx_cython.serial_collect import analyze
    monkeypatch.delattr(ast, 'unparse')
    module = SimpleNamespace(classes={}, spec=SimpleNamespace(automatic_optimizations=True),
        visitor=SimpleNamespace(wrapper=SimpleNamespace(module=cst.parse_module('class Values: pass\n'))))
    assert analyze(module) == {}
