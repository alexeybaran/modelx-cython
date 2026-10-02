"""Original public classes execute numeric/object failure and cleanup boundaries."""
import importlib
import json

from test_original_frame_shell import _run


def test_original_classes_preserve_checked_object_cleanup_and_retry(tmp_path, monkeypatch):
    import modelx as mx

    (tmp_path / 'shell_boundary_tracker.py').write_text('''
created = destroyed = child_evals = 0
fail = False
callback = None
class Token:
    def __init__(self):
        global created
        created += 1
    def __del__(self):
        global destroyed
        destroyed += 1
def child():
    global child_evals
    child_evals += 1
def maybe_fail():
    if fail: raise KeyError('object failure')
''', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    tracker = importlib.import_module('shell_boundary_tracker')
    model = mx.new_model('OriginalBoundary')
    space = model.new_space('S')
    space.tracker = tracker
    space.level = 0.0
    for name, source in {
        'leaf': 'def leaf(n):\n if n == 2: raise ValueError("numeric failure")\n return float(n-2)\n',
        'parent': 'def parent(n):\n return leaf(n) + 1.0\n',
        'caught': ('def caught(n):\n token = tracker.Token()\n try:\n'
                   '  return {"value": parent(n)}\n except ValueError as error:\n'
                   '  return {"caught": str(error)}\n'),
        'child': 'def child(n):\n tracker.child()\n return {"value": float(n)}\n',
        'failing': ('def failing(n):\n token = tracker.Token()\n completed = child(n)\n'
                    ' tracker.maybe_fail()\n return completed\n'),
        'numeric_parent': 'def numeric_parent(n):\n return float(failing(n)["value"])\n',
        'walk': ('def walk(n):\n token = tracker.Token()\n value = 0.0\n'
                 ' for i in range(200): value += numeric_parent(n)\n'
                 ' return {"value": value}\n'),
        'level_value': 'def level_value(n):\n return level\n',
        'change': 'def change(n):\n tracker.callback()\n return {"value": level_value(n)}\n',
        'nested': ('def nested(n):\n token = tracker.Token()\n tracker.callback()\n'
                   ' return {"value": parent(n)}\n'),
    }.items():
        space.new_cells(name=name, formula=source)
    model.export(tmp_path / 'model')
    model.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\ns=mx_model.S\ns.leaf(1);s.parent(1)\n'
        's.caught(1);s.child(1);s.numeric_parent(1);s.walk(1)\n'
        'import shell_boundary_tracker as tracker\n'
        'tracker.callback=lambda: None\ns.change(1);s.level_value(0)\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(
        repr({'spaces': {'S': {'cells_param_size': {'n': 4}}}}), encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'),
                    '--spec', str(tmp_path / 'spec.py'),
                    '--sample', str(tmp_path / 'sample.py')], 'build')
    manifest = json.loads((tmp_path / 'model_cy/_mx_frame_S/manifest.json').read_text())
    assert manifest['cells']['parent']['coverage']['body_backend'] == 'cython_nogil'
    assert manifest['cells']['caught']['coverage']['body_backend'] == 'cython_gil'
    assert manifest['cells']['numeric_parent']['coverage']['body_backend'] == 'cython_gil'
    assert 'numeric_parent' in manifest['original_checked_numeric_bridges']
    generated = next((tmp_path / 'model_cy/_mx_frame_S').rglob('_kernel.pyx')).read_text()
    assert 'def _gil_entry_numeric_parent(' not in generated
    assert all(not row['coverage']['detached_python'] for row in manifest['cells'].values())
    (tmp_path / 'verify.py').write_text('''
import gc
import shell_boundary_tracker as tracker
from model_cy import mx_model
from model_cy._mx_frame_support.python_calculator import PythonScope
original_calculate = PythonScope.calculate
def reject_redundant_numeric_adapter(self, name, arguments):
    if name in self.manifest.get('original_checked_numeric_bridges', ()):
        raise AssertionError('Compiled numeric entry crossed the Python calculator adapter')
    return original_calculate(self, name, arguments)
PythonScope.calculate = reject_redundant_numeric_adapter
model = type(mx_model)(); item = model.S
assert item.caught(2) == {'caught': 'numeric failure'}
assert item.leaf(1) == -1.0
tracker.fail = True
for _ in range(2):
    try: item.numeric_parent(1)
    except KeyError as error: assert error.args == ('object failure',)
    else: raise AssertionError('failed parent cache committed')
assert tracker.child_evals == 1
gc.collect()
assert tracker.created == tracker.destroyed == 3
tracker.fail = False
assert item.numeric_parent(1) == 1.0
assert tracker.child_evals == 1
def callback():
    try: item.parent(2)
    except ValueError as error: assert str(error) == 'numeric failure'
    else: raise AssertionError('nested numeric error lost')
tracker.callback = callback
assert item.nested(1) == {'value': 0.0}
assert item.parent(1) == 0.0
try: item.parent(2)
except ValueError as error: assert str(error) == 'numeric failure'
else: raise AssertionError('native guard not restored')
gc.collect()
assert tracker.created == tracker.destroyed == 5
original_bind = PythonScope.bind_original_inputs
bindings = []
def record_bind(self, values):
    bindings.append(self)
    return original_bind(self, values)
PythonScope.bind_original_inputs = record_bind
before = len(bindings)
assert item.walk(1) == {'value': 200.0}
assert len(bindings) - before <= 2, 'Unchanged internal inputs were repeatedly rebuilt'
import math
assert math.copysign(1.0, item.level_value(0)) == 1.0
def negative_zero(): item.level = -0.0
tracker.callback = negative_zero
assert math.copysign(1.0, item.change(1)['value']) == -1.0
def positive_zero(): item.level = 0.0
tracker.callback = positive_zero
assert math.copysign(1.0, item.change(2)['value']) == 1.0
def sentinel(): item.level = -1.0
tracker.callback = sentinel
assert item.change(3)['value'] == -1.0
assert math.copysign(1.0, item.level_value(0)) == 1.0
assert math.copysign(1.0, item.level_value(1)) == -1.0
gc.collect()
assert tracker.created == tracker.destroyed == 6
print('original classes: object cleanup, checked bridges, nested guard and retry PASS')
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')
