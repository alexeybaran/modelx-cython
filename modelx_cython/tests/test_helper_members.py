"""Module member metadata and receiver binding."""
from types import ModuleType, SimpleNamespace

import libcst as cst

from modelx_cython.helper_members import helper_reference, module_helpers
from modelx_cython.helpers import HelperCalls


def operation(values):
    return all(values)


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


def test_foreign_receiver_boundaries():
    direct = SimpleNamespace(helper=('sample.provider', 'all'))
    refs = {'every': direct, 'operations': SimpleNamespace(helper_members={'every': direct})}
    assert helper_reference(cst.parse_expression('other.operations.every'), refs) is None
    assert helper_reference(cst.parse_expression('self.operations.nested.every'), refs) is None


def test_rebound_receiver_and_starred_calls_retain_ordinary_binding():
    ref = SimpleNamespace(helper=('sample.provider', 'lookup'), call_layout='keywords')
    for source in ['def formula(self):\n    self = other\n    return self.lookup(1)\n',
                   'def formula(self):\n    return self.lookup(*items)\n']:
        binder = HelperCalls({'lookup': ref}, set())
        assert cst.Module(body=(binder.rewrite(cst.parse_module(source).body[0]),)).code == source
        assert binder.operations == set()


def test_first_class_cells_use_bound_callables_without_changing_direct_calls():
    source = '''def formula(self):
    return (self.term, self.term(1), lambda self: self.term)
'''
    binder = HelperCalls({}, set(), cells={'term'})
    result = cst.Module(body=(binder.rewrite(cst.parse_module(source).body[0]),)).code
    assert '_mx_cy.cast(object, self).term' in result
    assert 'self.term(1)' in result
    assert 'lambda self: self.term' in result
