"""Explicit typed symbol binding uses the shared syntax, without an AST pass."""
import ast
from types import SimpleNamespace

import libcst as cst

from modelx_cython.helpers import HelperCalls
from modelx_cython.numeric_helpers import NumericReturns
from modelx_cython.provider_manifest import (
    NumericBounds, NumericMethod, NumericRecord, NumericTable,
    ProviderBinding, ProviderMember, read_provider_manifest,
)


def builtin_table_binding():
    manifest = read_provider_manifest('modelx_cython.runtime._native_inputs', 'owned-table-1')
    type_id = 'modelx_helpers.immutable_inputs.Table'
    return ProviderBinding(manifest, manifest.types[type_id], type_id, True)


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
    table = SimpleNamespace(name='schedule', native_provider_binding=builtin_table_binding(),
                            params=(), has_formula_def=True,
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
    table = SimpleNamespace(native_provider_binding=builtin_table_binding(), params=())
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


def test_record_cast_uses_the_declared_provider_operations():
    provider = 'rates_extension.kernel'
    numeric = NumericRecord('lookup', 'read_value', 'cast_value', 'field_current', 'cast_current', 'read_bound')
    binding = SimpleNamespace(manifest=SimpleNamespace(provider=provider), declaration=SimpleNamespace(
        numeric_record=numeric, members={'__getitem__': ProviderMember('string_label_to_object', 'select')}))
    record = SimpleNamespace(name='record', observed_provider_type=binding,
                             has_formula_def=True, is_locked=False, get_rettype_expr=lambda **_: 'object')
    owner = SimpleNamespace(name='_c_Projection', cells={'record': record}, refs={})
    cell = SimpleNamespace(name='amount', params=('t',), has_args=lambda: True, is_locked=False,
                           has_formula_def=True, formula_allows_exact_receiver=True,
                           get_rettype_expr=lambda **_: 'double')
    node = cst.parse_module('def amount(self, t):\n    return float(self.record()["rate"])\n')
    providers = set()

    result = node.visit(NumericReturns(owner, cell, providers, builtin_keys={'float'})).code

    assert '.cast_current(' in result and '.read_bound(' in result and '.cast_value(float,' in result
    assert providers == {provider}


def test_clamped_table_lookup_uses_the_declared_bounds_operations():
    provider = 'rates_extension.kernel'
    methods = {'price': NumericMethod('lookup_f64', 'bound_price', 'prime_price')}
    bounds = NumericBounds('first', 'last', 'keys', 'maximum', 'read_bounded', 'read_clamped')
    binding = SimpleNamespace(manifest=SimpleNamespace(provider=provider), declaration=SimpleNamespace(
        numeric_table=NumericTable('owner', 'ready', methods, bounds)))
    owner = SimpleNamespace(cells={'schedule': SimpleNamespace(native_provider_binding=binding, params=())},
                            refs={})
    cell = SimpleNamespace(name='amount', params=('age',), get_argtype_expr=lambda *_, **__: 'long long')
    lowering = NumericReturns(owner, cell, set(), builtin_keys={'min'})
    source = cst.parse_module(
        'table = self.schedule()\n'
        'key = min(age, table.keys.maximum())\n'
        'return table.price(key, "rate")\n'
    )

    result = source.with_changes(body=lowering.clamped_table_method(source.body)).code

    assert '.read_clamped(' in result and 'table.price(key, "rate")' in result
    assert lowering.providers == {provider}
