"""Explicit typed symbol binding uses the shared syntax, without an AST pass."""
import ast
from types import SimpleNamespace

import libcst as cst

from modelx_cython.helpers import HelperCalls
from modelx_cython.numeric_helpers import NumericReturns


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


def test_declared_table_method_reuses_existing_bound_lookup_without_ast_parse(monkeypatch):
    node = cst.parse_module('def amount(self, month):\n    return self.schedule().lookup_float(month, "rate", 0.0)\n')
    table = SimpleNamespace(name='schedule', native_table_result=True, has_formula_def=True,
                            has_args=lambda: False, is_locked=False, body_has_closure=False)
    amount = SimpleNamespace(name='amount', cls='Projection', params=('month',),
                             is_locked=False, has_formula_def=True, formula_allows_exact_receiver=True,
                             get_rettype_expr=lambda **_: 'double', get_argtype_expr=lambda *_, **__: 'long long')
    owner = SimpleNamespace(name='_c_Projection', cells={'schedule':table}, refs={},
                            visitor=SimpleNamespace(dynamic_ref_classes=set()))
    def forbidden(*args, **kwargs):
        raise AssertionError('table method lowering parsed an AST')
    providers = set()
    lowering = NumericReturns(owner, amount, providers)
    with monkeypatch.context() as probe:
        probe.setattr(ast, 'parse', forbidden)
        result = node.visit(lowering).code
    assert 'type(self) is _c_Projection and (self._has_schedule' in result
    assert '.prepare_table_method(' in result and '.bound_lookup_f64(' in result
    assert providers == {'modelx_cython.runtime._native_inputs'}
    amount.get_argtype_expr = lambda *_, **__: 'bint'
    rejected = NumericReturns(owner, amount, set())
    assert node.visit(rejected).code == node.code
    assert rejected.contexts == {}


def test_clamped_table_lookup_keeps_a_reassigned_receiver():
    provider = 'modelx_cython.runtime._native_inputs'
    last_key = SimpleNamespace(helper=(provider, 'last_key'))
    table = SimpleNamespace(native_table_result=True)
    owner = SimpleNamespace(
        cells={'schedule': table},
        refs={'tables': SimpleNamespace(helper_members={'last_key': last_key})},
    )
    cell = SimpleNamespace(params=('age',), get_argtype_expr=lambda *_, **__: 'long long')
    lowering = NumericReturns(owner, cell, set(), builtin_keys={'int', 'min'})
    body = cst.parse_module(
        'table = self.schedule()\n'
        'table = min(age, int(self.tables.last_key(table)))\n'
        'return table.lookup_float(table, "rate")\n'
    ).body
    assert lowering.clamped_table_method(body) is None
    assert not lowering.providers
