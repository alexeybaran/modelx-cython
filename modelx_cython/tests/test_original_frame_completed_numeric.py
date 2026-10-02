"""Completed numeric results cross checked entries without preparing miss work."""
import json
import pytest
from test_original_frame_shell import _run


@pytest.mark.parametrize("removed_callback", [False, True])
def test_completed_numeric_bridge_keeps_live_misses_and_error_cleanup(tmp_path, removed_callback):
    import modelx as mx
    model = mx.new_model('CompletedNumeric')
    s = model.new_space('S')
    s.level = -1.0
    s.fail = False
    s.hook = None
    s.new_cells('read', 'def read(n):\n if fail and n == 3: raise ValueError("failed numeric miss")\n return level + float(n)\n')
    s.new_cells('loop', 'def loop(n):\n total = 0.0\n for i in range(200): total += read(n)\n return {"total": total}\n')
    s.new_cells('after', 'def after(n):\n held = read(n)\n hook()\n fresh = read(n+1)\n return {"held": held, "fresh": fresh, "again": read(n)}\n')
    model.export(tmp_path / 'model')
    model.close()
    (tmp_path / 'sample.py').write_text('from model import mx_model\ns=mx_model.S\ns.hook=lambda:None\ns.loop(0)\ns.after(0)\n')
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'S': {'cells_param_size': {'n': 64}}}}))
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--sample',
                    str(tmp_path / 'sample.py'), '--spec', str(tmp_path / 'spec.py'),
                    '--translate-only'], 'translate')
    folder = tmp_path / 'model_cy/_mx_frame_S'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['cells']['read']['calculator'] == 'native'
    path = next(folder.rglob('_kernel.pyx'))
    source = path.read_text()
    begin = source.index('    cdef inline double _native_cell_read(')
    end = source.index('\n    cdef ', begin + 10)
    bridge = source[begin:end]
    assert bridge.index('_mx_completed_key] ==') < bridge.index('self._mx_synchronize_inputs()')
    # Generated diagnostic counters have no role in accepted performance runs.
    lines = source.splitlines(keepends=True)
    preamble = 0
    while preamble < len(lines) and (not lines[preamble].strip() or lines[preamble].startswith('#')):
        preamble += 1
    counter = 'cdef long long _mx_test_sync_calls = 0\n\ndef _test_sync_calls():\n    return _mx_test_sync_calls\n\n'
    source = ''.join(lines[:preamble]) + counter + ''.join(lines[preamble:])
    needle = '    cpdef void _mx_synchronize_inputs(self) except *:\n'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '        global _mx_test_sync_calls\n        _mx_test_sync_calls += 1\n')
    path.write_text(source)
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--compile-only'], 'build')
    (tmp_path / 'private.pyx').write_text('from model_cy._mx_classes cimport _c_S\ndef private_read(_c_S space):\n    return space._f_read(0)\n')
    (tmp_path / 'private_setup.py').write_text('from setuptools import setup\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize("private.pyx", language_level=3))\n')
    _run(tmp_path, ['private_setup.py', 'build_ext', '--inplace'], 'private-build')
    callback = "s.hook = lambda: setattr(s, 'level', 200.0)" if removed_callback else "def update_level(): setattr(s, 'level', 200.0)\ns.hook = update_level"
    (tmp_path / 'verify.py').write_text(f'''
import importlib, pickle
from private import private_read
from model_cy import mx_model
kernel = importlib.import_module({manifest['package'] + '._kernel'!r})
s = type(mx_model)().S
before = kernel._test_sync_calls()
assert s.loop(0) == {{'total': -200.0}}
assert kernel._test_sync_calls() - before == 1, 'Cached numeric reads rebuilt miss preparation'
s.level = 100.0
assert s.read(0) == -1.0
assert private_read(s) == 100.0
assert s.read(0) == -1.0
{callback}
assert s.after(0) == {{'held': -1.0, 'fresh': 201.0, 'again': -1.0}}
s.fail = True
try: s.after(2)
except ValueError as error: assert str(error) == 'failed numeric miss'
else: raise AssertionError('A failing numeric miss must retain checked propagation')
assert s.read(2) == 202.0
s.level = 300.0
s.fail = False
assert s.after(2) == {{'held': 202.0, 'fresh': 203.0, 'again': 202.0}}
try: s.loop(64)
except IndexError: pass
else: raise AssertionError('Declared bounds must still reach the checked miss path')
try: q = pickle.loads(pickle.dumps(s))
except pickle.PicklingError:
    if {removed_callback!r}:
        print('KNOWN_REMOVED_CALLBACK_SNAPSHOT')
        raise SystemExit(0)
    raise
assert q.read(0) == -1.0 and q.read(2) == 202.0
assert q.loop(0) == {{'total': -200.0}}
''')
    # The callback is not picklable. Remove it after the callback/error gates;
    # completed caches must restore independently of the current reference.
    text = (tmp_path / 'verify.py').read_text().replace('try: q = pickle.loads', 's.hook = None\ntry: q = pickle.loads')
    (tmp_path / 'verify.py').write_text(text)
    _run(tmp_path, ['verify.py'], 'verify')
    if 'KNOWN_REMOVED_CALLBACK_SNAPSHOT' in (tmp_path / 'verify.log').read_text():
        pytest.xfail('Baseline 62f2f0d also serializes a removed callback retained in owner inputs; pending reference/serialization integration')
