"""Namespace metadata shares direct-Reference validation and preserves binding."""
from types import ModuleType, SimpleNamespace

import libcst as cst
import pytest

from modelx_cython.helper_members import helper_reference, module_helpers
from modelx_cython.helpers import helper_capability
from modelx_cython.tracer import RuntimeValueInfo
from modelx_cython.builder import CombinedRefInfo


def operation(values):
    return all(values)


operation.__modelx_cython_helper__ = ('sample.provider', 'all')
operation.__modelx_cython_reduction__ = 'bool_all'


def test_aliases_share_capabilities_without_freezing_member_callables():
    module = ModuleType('sample.operations')
    module.every = operation
    module.alias = operation
    metadata = module_helpers(module)
    assert metadata['every'].helper == ('sample.provider', 'all')
    assert metadata['every'].reduction == 'bool_all'
    assert metadata['alias'] == metadata['every']
    ref = SimpleNamespace(helper_members=metadata)
    call = cst.parse_expression('self.operations.every')
    assert helper_reference(call, {'operations': ref}) is metadata['every']
    replacement = lambda values: 17
    module.every = replacement
    assert module.every is replacement
    # Emission gets the CURRENT member expression, not a captured callable.
    assert not hasattr(metadata['every'], 'function')
    assert module_helpers(module).keys() == {'alias'}


def test_capture_does_not_execute_module_or_object_getattr():
    seen = []
    module = ModuleType('sample.dynamic')
    module.__getattr__ = lambda name: seen.append(name)
    assert module_helpers(module) == {}

    class Namespace:
        def __getattr__(self, name):
            seen.append(name)
            return operation

    assert module_helpers(Namespace()) == {}
    assert seen == []


def test_invalid_member_capability_is_not_silently_accepted():
    def invalid(values):
        return values
    invalid.__modelx_cython_helper__ = ('sample.provider', 'invalid')
    invalid.__modelx_cython_reduction__ = 'invented'
    module = ModuleType('sample.invalid')
    module.invalid = invalid
    with pytest.raises(ValueError, match='invalid helper capability'):
        module_helpers(module)


def test_direct_references_and_foreign_receiver_boundaries():
    direct = SimpleNamespace(helper=('sample.provider', 'all'))
    refs = {'every': direct, 'operations': SimpleNamespace(helper_members={'every': direct})}
    assert helper_reference(cst.parse_expression('self.every'), refs) is direct
    assert helper_reference(cst.parse_expression('other.operations.every'), refs) is None
    assert helper_reference(cst.parse_expression('self.operations.nested.every'), refs) is None


def test_shared_iteration_and_contextual_float_capabilities_reach_namespaces():
    def lookup(*args):
        return args
    lookup.__modelx_cython_helper__ = ('sample.provider', 'lookup')
    lookup.__modelx_cython_iteration__ = ('keys_span', 'KeySpan', 'key_span_count', 'key_span_at')
    lookup.__modelx_cython_float_numeric__ = 'lookup_float_f64'
    lookup.__modelx_cython_keyed_float_numeric__ = tuple(f'lookup_key{i}_float_f64' for i in range(5))
    module = ModuleType('sample.owned')
    module.lookup = lookup
    traced = RuntimeValueInfo(module)
    merged = CombinedRefInfo('exported._mx_classes', '_c_Test', 'operations', traced)
    record = merged.helper_members['lookup']
    assert record.iteration_helper == ('keys_span', 'KeySpan', 'key_span_count', 'key_span_at')
    assert record.float_numeric_helper == 'lookup_float_f64'
    assert record.keyed_float_numeric_helper == tuple(f'lookup_key{i}_float_f64' for i in range(5))


@pytest.mark.parametrize('capability,value', [
    ('iteration', ('keys_span', 'KeySpan', 'key_span_count')),
    ('iteration', ('keys_span', 'bad-name', 'key_span_count', 'key_span_at')),
    ('float_numeric', ('lookup_float_f64',)),
    ('float_numeric', 'bad-name'),
    ('keyed_float_numeric', ('lookup_key0',)),
    ('keyed_float_numeric', ('key0', 'key1', 'bad-name', 'key3', 'key4')),
    ('keyed_float_numeric', ('key0', 'key1', 'for', 'key3', 'key4')),
])
def test_new_capability_shapes_use_the_single_shared_validator(capability, value):
    def invalid():
        pass
    setattr(invalid, '__modelx_cython_' + capability + '__', value)
    with pytest.raises(ValueError, match='invalid helper capability'):
        helper_capability(invalid, capability)


@pytest.mark.parametrize('result', ['float64', 'int64', 'bool', 'unicode', 'object'])
def test_scalar_result_is_physical_abi_metadata_for_direct_and_module_refs(result):
    def lookup(*args):
        raise AssertionError('metadata capture must not invoke a helper')
    lookup.__modelx_cython_helper__ = ('sample.provider', 'lookup')
    lookup.__modelx_cython_scalar_result__ = result
    direct = CombinedRefInfo('exported._mx_classes', '_c_Test', 'lookup', RuntimeValueInfo(lookup))
    module = ModuleType('sample.typed')
    module.lookup = lookup
    merged = CombinedRefInfo('exported._mx_classes', '_c_Test', 'tables', RuntimeValueInfo(module))
    assert direct.scalar_result == result
    assert merged.helper_members['lookup'].scalar_result == result
    assert not hasattr(merged.helper_members['lookup'], 'function')


@pytest.mark.parametrize('result', ['float', 'double', 'int', 'number', 'Float64', ('float64',)])
def test_scalar_result_rejects_undeclared_physical_types(result):
    def lookup():
        pass
    lookup.__modelx_cython_scalar_result__ = result
    with pytest.raises(ValueError, match='invalid helper capability'):
        helper_capability(lookup, 'scalar_result')


@pytest.mark.parametrize('suffix,prefix', [('iteration_f64', 'float'), ('iteration_str', 'str')])
def test_typed_key_iteration_uses_same_module_and_reference_registry(suffix, prefix):
    def keys(table):
        raise AssertionError('metadata capture must not invoke a producer')
    keys.__modelx_cython_helper__ = ('sample.provider', 'keys')
    capability = (f'keys_{prefix}_span', f'{prefix.title()}KeySpan',
                  f'{prefix}_key_span_count', f'{prefix}_key_span_at')
    setattr(keys, '__modelx_cython_' + suffix + '__', capability)
    module = ModuleType('sample.keys')
    module.keys = keys
    field = suffix + '_helper'
    assert getattr(RuntimeValueInfo(keys), field) == capability
    assert getattr(RuntimeValueInfo(module).helper_members['keys'], field) == capability
