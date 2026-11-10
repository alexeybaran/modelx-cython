"""Explicit typed symbol binding uses the shared syntax, without an AST pass."""
import ast
from types import SimpleNamespace

import libcst as cst

from modelx_cython.helpers import HelperCalls


def test_explicit_typed_symbol_binding_uses_no_ast_parser(monkeypatch):
    node = cst.parse_module('def amount(self, age):\n    return self.lookup_float(self.table, ("M", age), "rate")\n')
    refs = {'lookup_float': SimpleNamespace(helper=('modelx_cython.runtime._native_inputs', 'lookup_float'))}
    def forbidden(*args, **kwargs):
        raise AssertionError('explicit helper binding parsed an AST')
    monkeypatch.setattr(ast, 'parse', forbidden)
    providers = set()
    binding = HelperCalls(refs, providers)
    result = node.visit(binding).code
    assert '.lookup_float(self.lookup_float,' in result
    assert providers == {'modelx_cython.runtime._native_inputs'}
    assert binding.operations == {'modelx_cython.runtime._native_inputs.lookup_float'}
