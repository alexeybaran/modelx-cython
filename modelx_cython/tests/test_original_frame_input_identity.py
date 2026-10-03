"""Redundant object binding is skipped without freezing live references."""
import json
from modelx_cython.tests.test_original_frame_shell import _run


def test_object_binding_identity_scope_failure_and_serialization(tmp_path):
    import modelx as mx
    model = mx.new_model('IdentityBindings')
    s = model.new_space('S')
    s.resource = {'a': -1.0, 'b': 0.0, 'c': 3.0, 'd': 4.0}
    s.config = {'fail': False}
    s.hook = None
    for name, formula in {
        'read': 'def read(key):\n return float(resource[key])\n',
        'fresh': 'def fresh(key):\n return float(resource[key])\n',
        'failure': 'def failure(key):\n child = read(key)\n if config["fail"]: raise ValueError("retry")\n return child\n',
        'after': 'def after(key):\n hook()\n return fresh(key)\n',
    }.items():
        s.new_cells(name, formula)
    s.fresh.is_cached = False
    model.export(tmp_path / 'model')
    model.close()
    (tmp_path / 'sample.py').write_text("from model import mx_model\ns=mx_model.S\ns.hook=lambda: None\nfor key in ('a', 'b', 'c', 'd'):\n s.read(key); s.fresh(key); s.failure(key); s.after(key)\n")
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path/'model'), '--no-spec',
                    '--sample', str(tmp_path/'sample.py')], 'build')
    frame = tmp_path / 'model_cy/_mx_frame_S'
    manifest = json.loads((frame / 'manifest.json').read_text())
    assert all(row['kind'] == 'object' for row in manifest['inputs'].values())
    source = next(frame.rglob('_kernel.pyx')).read_text()
    assert 'cdef object _mx_bound_input_scope' in source
    assert 'self._mx_bound_input_scope is self._python_scope' in source
    (tmp_path/'verify.py').write_text("""import copy
import pickle
from model_cy import mx_model
from model_cy._mx_frame_support.python_calculator import PythonScope
from model_cy._mx_frame_owner import SpaceFrameOwner
calls=[]; owners=[]
original=PythonScope.bind_original_inputs
original_init=SpaceFrameOwner.__init__
def observed(self, record):
    calls.append(id(self))
    return original(self, record)
def constructed(self, *args):
    original_init(self, *args)
    owners.append(self)
PythonScope.bind_original_inputs=observed
SpaceFrameOwner.__init__=constructed
m=type(mx_model)(); s=m.S
assert s.read('a') == -1.0
count=len(calls)
for _ in range(8):
    assert s.read('a') == -1.0 and s.fresh('b') == 0.0
assert len(calls) == count
s.resource['b']=17.0
assert s.fresh('b') == 17.0 and len(calls) == count
assert s.read('a') == -1.0
class EqualityTrap(dict):
    def __eq__(self, other): raise AssertionError('binding must not compare equality')
s.resource=EqualityTrap(s.resource)
assert s.fresh('b') == 17.0 and len(calls) == count+1
s.config['fail']=True
for _ in range(2):
    try: s.failure('c')
    except ValueError as error: assert str(error) == 'retry'
    else: raise AssertionError('failure cached')
s.resource['c']=99.0
s.config['fail']=False
assert s.failure('c') == 3.0  # completed child survives
replacement=dict(s.resource); replacement['d']=73.0
def callback():
    s.resource=replacement
    assert s.fresh('b') == 17.0  # reentrant changed-reference read
s.hook=callback
assert s.after('d') == 73.0
s.hook=None; s.resource=dict(s.resource)
assert s.fresh('d') == 73.0
q=pickle.loads(pickle.dumps(m)).S
assert q.read('a') == -1.0 and q.fresh('d') == 73.0
# Scope replacement must bind despite identical stored input identities.
owner=owners[0]; owner.prepare_checked()
record=owner.kernel.bound_inputs()
scope=copy.copy(owner._scope); scope.namespace=dict(owner._scope.namespace)
scope.namespace['resource']=None
owner.kernel.set_python_scope(scope)
owner.kernel.bind_inputs(record)
assert scope.namespace['resource'] is s.resource
# A failed namespace bind cannot publish the new scope as synchronized.
other=copy.copy(scope); other.namespace=dict(scope.namespace)
owner.kernel.set_python_scope(other)
def failed(self, record): raise RuntimeError('bind failure')
PythonScope.bind_original_inputs=failed
try: owner.kernel.bind_inputs(record)
except RuntimeError as error: assert str(error) == 'bind failure'
else: raise AssertionError('binding failure lost')
PythonScope.bind_original_inputs=observed
count=len(calls)
owner.kernel.bind_inputs(record)
assert len(calls) == count+1
owner.kernel.bind_inputs(record)
assert len(calls) == count+1
# Same-scope failure after input assignments must also retry the namespace.
changed=list(record)
# Identify the original resource by identity rather than relying on input order.
resource_index=next(i for i, value in enumerate(record) if value is s.resource)
changed[resource_index]=dict(s.resource)
changed[resource_index]['d']=101.0
changed=tuple(changed)
PythonScope.bind_original_inputs=failed
try: owner.kernel.bind_inputs(changed)
except RuntimeError as error: assert str(error) == 'bind failure'
else: raise AssertionError('same-scope binding failure lost')
PythonScope.bind_original_inputs=observed
count=len(calls)
owner.kernel.bind_inputs(changed)
assert len(calls) == count+1
assert other.namespace['resource'] is changed[resource_index]
owner.kernel.bind_inputs(changed)
assert len(calls) == count+1
print('identity binding behavioral checks passed')
""")
    _run(tmp_path, ['verify.py'], 'verify')
