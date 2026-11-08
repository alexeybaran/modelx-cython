"""The retired contextual experiment must not parse or recognize formulas."""
import ast
from types import SimpleNamespace

import libcst as cst

from modelx_cython.helpers import HelperCalls
from modelx_cython.numeric_helpers import ContextualFloat


def test_retired_contextual_bridge_never_parses_or_admits(monkeypatch):
    node = cst.parse_module('def amount(self, age):\n    return float(self.lookup(self.table, age, "rate"))\n')
    def forbidden(*args, **kwargs):
        raise AssertionError('retired helper recognizer parsed an AST')
    monkeypatch.setattr(ast, 'parse', forbidden)
    providers = set()
    bridge = ContextualFloat(None, None, providers)
    assert node.visit(bridge).code == node.code
    assert bridge.classify_call(None, {}) is None
    assert bridge.lower_call(None) is None
    assert providers == set() and bridge.operations == set()


def test_explicit_typed_symbol_binding_uses_no_ast_parser(monkeypatch):
    node = cst.parse_module('def amount(self, age):\n    return self.lookup_float(self.table, ("M", age), "rate")\n')
    refs = {'lookup_float': SimpleNamespace(helper=('lifelib._native_inputs', 'lookup_float'))}
    def forbidden(*args, **kwargs):
        raise AssertionError('explicit helper binding parsed an AST')
    monkeypatch.setattr(ast, 'parse', forbidden)
    providers = set()
    binding = HelperCalls(refs, providers)
    result = node.visit(binding).code
    assert '.lookup_float(self.lookup_float,' in result
    assert providers == {'lifelib._native_inputs'}
    assert binding.operations == {'lifelib._native_inputs.lookup_float'}
