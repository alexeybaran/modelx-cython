"""Completed object Cells need no preparation of their uncached input work."""
import json

from modelx_cython.tests.test_original_frame_shell import _run


def test_completed_object_entry_preserves_live_misses_and_retry(tmp_path):
    import modelx as mx
    model = mx.new_model('CompletedInputs')
    space = model.new_space('S')
    space.inputs = {'n': 1.0}
    space.fail = False
    for name, formula in {
        'value': 'def value():\n return {"n": inputs["n"]}\n',
        'empty': 'def empty():\n return None\n',
        'child': 'def child():\n return value()\n',
        'fresh': 'def fresh():\n return {"n": inputs["n"]}\n',
        'failure': 'def failure():\n completed = child()\n if fail: raise ValueError("retry")\n return completed\n',
    }.items():
        space.new_cells(name, formula)
    space.fresh.is_cached = False
    model.export(tmp_path / 'model')
    model.close()
    (tmp_path / 'sample.py').write_text('''from model import mx_model
s = mx_model.S
s.value(); s.empty(); s.child(); s.fresh(); s.failure()
''')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py')], 'build')
    folder = tmp_path / 'model_cy/_mx_frame_S'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['cells']['value']['storage'] == 'object_singleton'
    source = next(folder.rglob('_kernel.pyx')).read_text()
    start = source.index('cdef object _mx_shell_value(')
    entry = source[start:source.index('\ncdef ', start + 1)]
    assert entry.index('return owner._object_cell_value()') < entry.index('owner.bind_inputs(')
    assert 'not owner._object_key_guard and not owner._clearing_objects' in entry
    (tmp_path / 'verify.py').write_text('''import pickle
from model_cy import mx_model
from model_cy._mx_frame_support.python_calculator import PythonScope
calls = []
original = PythonScope.bind_original_inputs
def observed(self, record):
    calls.append(id(self))
    return original(self, record)
PythonScope.bind_original_inputs = observed
m = type(mx_model)(); s = m.S
value = s.value()
assert value == {"n": 1.0}
assert s.empty() is None and s.child() is value
count = len(calls)
s.inputs = {"n": -1.0}
for _ in range(10):
    assert s.value() is value and s.child() is value and s.empty() is None
assert len(calls) == count
assert s.fresh() == {"n": -1.0} and len(calls) > count
s.inputs["n"] = 0.0
assert s.fresh() == {"n": 0.0}
s.fail = True
try:
    s.failure()
except ValueError as error:
    assert str(error) == "retry"
else:
    raise AssertionError("Failure must propagate and leave its cache uncommitted")
assert s.child() is value
s.fail = False
assert s.failure() is value
restored = pickle.loads(pickle.dumps(m))
assert restored.S.value() == value and restored.S.empty() is None
assert restored.S.fresh() == {"n": 0.0}
''')
    _run(tmp_path, ['verify.py'], 'verify')
