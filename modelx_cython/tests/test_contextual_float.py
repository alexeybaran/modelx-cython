"""Explicit typed symbol binding uses the shared syntax, without an AST pass."""
import ast
from types import SimpleNamespace

import libcst as cst
import pytest

from modelx_cython.helpers import HelperCalls, module_name
from modelx_cython.numeric_helpers import NumericReturns
from modelx_cython.provider_manifest import GroupedTable, ProviderBinding, read_provider_manifest
from modelx_cython.usage import RESOLVED


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


@pytest.mark.parametrize('source', ['locked', 'unlocked', 'undeclared'])
def test_grouped_methods_follow_owner_capability(source):
    node = cst.parse_module('''\
def amount(self, month):
    group = self.category(month)
    if group not in self.data.groups():
        return 0.0
    key = self.key(month)
    table = self.data.groups().choose(group)
    if key not in table.keys:
        return 0.0
    return min(table.price(key, "rate"), self.cap(month))
''')

    def cell(rettype):
        return SimpleNamespace(is_special=lambda: False, called_with_kwargs=False, body_has_closure=False,
                               native_provider_binding=None, params=('month',),
                               get_rettype_expr=lambda **_: rettype)

    amount = SimpleNamespace(name='amount', cls='Projection', params=('month',), is_locked=False, has_formula_def=True,
                             formula_allows_exact_receiver=True, get_rettype_expr=lambda **_: 'double')
    owner = SimpleNamespace(name='_c_Projection', fqname='example.Projection',
                            cells={'category': cell('long long'), 'key': cell('long long'), 'cap': cell('double')},
                            refs={'data': SimpleNamespace(decl_type_expr='_c_Data')},
                            visitor=SimpleNamespace(dynamic_ref_classes=set()))
    provider = 'example.tariffs'
    binding = SimpleNamespace(
        declaration=SimpleNamespace(grouped_table=GroupedTable(
            'choose', 'keys', 'price', 'probe_tariff', 'read_tariff', 'probe_owned', 'read_owned')),
        manifest=SimpleNamespace(provider=provider),
    )
    name = 'example.Data.groups'
    resolver = SimpleNamespace(
        resolve_chain=lambda *_: (RESOLVED, name),
        observed_provider_results={} if source == 'undeclared' else {name: binding},
        locked_scalar_results={} if source == 'unlocked' else {name: 'object'},
        unlocked_scalar_results={name: 'object'} if source == 'unlocked' else {},
    )
    providers = set()
    lowering = NumericReturns(owner, amount, providers, builtin_keys={'min'}, resolver=resolver)
    result = node.visit(lowering).code
    if source == 'locked':
        alias = module_name(provider)
        for name in ('probe_tariff', 'read_tariff', 'probe_owned', 'read_owned'):
            assert f'{alias}.{name}(' in result
        assert 'type(self.data) is _c_Data' in result
        assert 'table = self.data.groups().choose(group)' in result
        assert 'return min(table.price(key, "rate"), self.cap(month))' in result
        assert providers == {provider}
    else:
        assert result == node.code
        assert not providers
        assert lowering.unlocked_table_reads == ({(name, 'grouped')} if source == 'unlocked' else set())
