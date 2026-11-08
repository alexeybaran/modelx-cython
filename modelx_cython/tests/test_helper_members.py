"""Namespace metadata shares direct-Reference validation and preserves binding."""
from types import ModuleType, SimpleNamespace

import libcst as cst
import pytest

from modelx_cython.helper_members import helper_reference, module_helpers


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
