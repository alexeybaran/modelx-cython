"""Original public values synchronize without prepared-context decoding."""
from enum import IntEnum
from types import SimpleNamespace

import pytest

from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.python_calculator import PythonScope


def test_original_binding_sync_keeps_identity_and_completed_results():
    class Category(IntEnum):
        A = 1

    scope = object.__new__(PythonScope)
    scope.manifest = {'original_space_shell': True,
                      'inputs': {'category': {}, 'resource': {}, 'scalar': {}}}
    scope.namespace = {'library': object(), 'child': object()}
    original = dict(scope.namespace)
    scope.kernel = SimpleNamespace(completed={'child': 12.0})
    scope.bindings = ({'unused': object()},)
    resource = {'curve': [1.0]}
    scope.decode = lambda *_: pytest.fail('Original public field was redundantly decoded')
    scope.bind_original_inputs((Category.A, resource, 2.0))
    assert scope.namespace['category'] is Category.A
    assert scope.namespace['resource'] is resource
    assert scope.record['resource'] is resource
    replacement = {'curve': [3.0]}
    scope.bind_original_inputs((Category.A, replacement, -1.0))
    assert scope.namespace['resource'] is replacement
    assert scope.record['resource'] is replacement
    assert scope.namespace['scalar'] == -1.0
    assert all(scope.namespace[name] is value for name, value in original.items())
    assert scope.kernel.completed == {'child': 12.0}
    scope.manifest['original_space_shell'] = False
    with pytest.raises(ContractError, match='original generated fields'):
        scope.bind_original_inputs((Category.A, resource, 1.0))
