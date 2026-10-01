"""Real original public ABI with an exact-receiver frame behind it."""
import json
import os
from pathlib import Path
import subprocess
import sys


def _run(folder, args, label):
    env = os.environ.copy()
    env['MX_CY_ORIGINAL_FRAME_SHELL'] = '1'
    env['PYTHONPATH'] = str(folder) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, *args], cwd=folder, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (folder / (label + '.log')).write_text(result.stdout, encoding='utf-8')
    assert result.returncode == 0, result.stdout[-18000:]


def test_original_shell_frame_cache_and_subclass_abi(tmp_path):
    import modelx as mx
    m = mx.new_model('ShellABI')
    s = m.new_space('S')
    s.offset = 0.0
    s.fail = False
    s.counts = {}
    for name, body in {
        'leaf': 'def leaf(n):\n return float(n - 1) + offset\n',
        'parent': 'def parent(n):\n return leaf(n) + leaf(n)\n',
        'fib': 'def fib(n):\n return float(n) if n < 2 else 0.5 * (fib(n-1) + fib(n-2))\n',
        'input_value': 'def input_value():\n return offset\n',
        'failure': 'def failure(n):\n child = leaf(n)\n if fail: raise ValueError("retry")\n return child + 1.0\n',
        'observed': 'def observed(n):\n counts[n] = counts.get(n, 0) + 1\n return parent(n)\n',
        'uncached': 'def uncached(n):\n return leaf(n) + offset\n',
    }.items():
        s.new_cells(name=name, formula=body)
    s.uncached.is_cached = False
    m.export(tmp_path / 'model')
    m.close()
    # Ensure the route works without the frame export's richer metadata.
    metadata = tmp_path / 'model' / '_mx_formula_sources.json'
    if metadata.exists():
        metadata.unlink()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\ns=mx_model.S\n'
        'for n in range(8):\n s.observed(n)\n s.fib(n)\n s.failure(n)\n s.uncached(n)\n'
        's.input_value()\n', encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py'), '--translate-only'], 'translate')
    receipt = json.loads((tmp_path / 'model_cy/_mx_classes.frame-shell.json').read_text())
    routes = {row['cell']: row['execution'] for row in receipt['cells'] if row['space'].endswith('._c_S')}
    assert routes['fib'] == routes['parent'] == routes['failure'] == 'frame_exact'
    assert routes['observed'] == routes['uncached'] == 'original_cython'
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--compile-only'], 'build')
    (tmp_path / 'override.pyx').write_text('''
from model_cy._mx_classes cimport _c_S
cdef class Private(_c_S):
    cdef double _f_leaf(self, long long n) except? -1:
        return 40.0
cdef class Public(_c_S):
    cpdef double leaf(self, long long n) except? -1:
        return 50.0
def uncached_parent(_c_S space, long long n):
    return space._f_parent(n)
def uncached_leaf(_c_S space, long long n):
    return space._f_leaf(n)
''', encoding='utf-8')
    (tmp_path / 'override_setup.py').write_text(
        "from setuptools import setup\nfrom Cython.Build import cythonize\n"
        "setup(ext_modules=cythonize('override.pyx'))\n", encoding='utf-8')
    _run(tmp_path, ['override_setup.py', 'build_ext', '--inplace'], 'override')
    (tmp_path / 'verify.py').write_text('''
import pickle
from model_cy import mx_model
from override import Private, Public, uncached_parent, uncached_leaf
factory = type(mx_model)
m = factory(); s = m.S
assert s.parent(0) == -2.0
assert s.observed(0) == -2.0 and s.counts == {0: 1}
assert s.input_value() == 0.0
s.offset = 10.0
# Public cached and private uncached access share the public child cache.
assert s.leaf(0) == -1.0 and s.input_value() == 0.0
assert uncached_leaf(s, 0) == 9.0 and s.leaf(0) == -1.0
assert uncached_parent(s, 0) == -2.0
assert s.parent(1) == 20.0 and s.uncached(0) == 9.0
s.offset = 20.0
assert s.uncached(0) == 19.0
s.fail = True
for _ in range(2):
    try: s.failure(2)
    except ValueError as e: assert str(e) == 'retry'
    else: raise AssertionError('failed parent was cached')
s.offset = 30.0; s.fail = False
assert s.failure(2) == 22.0  # completed child survives both failures
for n in (-1, 8):
    try: s.leaf(n)
    except IndexError: pass
    else: raise AssertionError('original capacity not preserved')
expected = [0.0, 1.0]
for n in range(2, 8): expected.append(0.5*(expected[n-1]+expected[n-2]))
for n in range(8): assert s.fib(n) == expected[n]
q = pickle.loads(pickle.dumps(s))
q.offset = 100.0
assert q.parent(0) == -2.0 and q.leaf(2) == 21.0
for n in range(8): assert q.fib(n) == expected[n]
for cls, expected in ((Private, 80.0), (Public, 100.0)):
    sub = cls(m); sub.offset = 0.0; sub.fail = False; sub.counts = {}
    assert sub.parent(0) == expected and sub.observed(0) == expected
    assert sub.leaf(0) == expected / 2
class PythonOverride(type(s)):
    def leaf(self, n): return 12.0
sub = PythonOverride(m); sub.offset = 0.0; sub.fail = False; sub.counts = {}
assert sub.parent(0) == 24.0
sub.leaf = lambda n: 20.0
assert sub.parent(1) == 40.0
print('original shell + native exact graph: all behavioral checks passed')
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')


def test_original_shell_heap_frame_allocation_failure_and_restore(tmp_path):
    import modelx as mx
    m = mx.new_model('HeapShell')
    s = m.new_space('S')
    s.new_cells('chain', 'def chain(n):\n return 0.0 if n == 0 else chain(n-1) + 1.0\n')
    m.export(tmp_path / 'model')
    m.close()
    (tmp_path / 'sample.py').write_text('from model import mx_model\nmx_model.S.chain(4)\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(
        repr({'spaces': {'S': {'cells_param_size': {'n': 2001}}}}), encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path/'model'), '--sample',
                    str(tmp_path/'sample.py'), '--spec', str(tmp_path/'spec.py'), '--translate-only'], 'translate')
    generated = next((tmp_path/'model_cy').rglob('_kernel.pyx'))
    text = generated.read_text()
    assert 'overflowcheck=False' in text and 'cdef _Frame* _frame' in text
    text = text.replace('from libc.stdlib cimport malloc, free', '''from libc.stdlib cimport malloc as _mx_real_malloc, free
from libc.stddef cimport size_t
cdef int _mx_test_fail = 0
def _test_fail_next():
    global _mx_test_fail
    _mx_test_fail = 1
cdef void* malloc(size_t n) noexcept:
    global _mx_test_fail
    if _mx_test_fail:
        _mx_test_fail = 0
        return NULL
    return _mx_real_malloc(n)
''')
    generated.write_text(text, encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path/'model'), '--compile-only'], 'build')
    manifest = json.loads(next((tmp_path/'model_cy/_mx_frame_S').glob('manifest.json')).read_text())
    (tmp_path/'verify.py').write_text(f'''
import gc, importlib, pickle
from model_cy import mx_model
module = importlib.import_module({manifest['package'] + '._kernel'!r})
factory = type(mx_model)
m = factory(); s = m.S
module._test_fail_next()
try: s.chain(1200)
except MemoryError: pass
else: raise AssertionError('allocation failure swallowed')
assert s.chain(1200) == 1200.0
for n in range(1201): assert s.chain(n) == float(n)
q = pickle.loads(pickle.dumps(s))
for n in range(1201): assert q.chain(n) == float(n)
for n in (-1, 2001):
    try: q.chain(n)
    except IndexError: pass
    else: raise AssertionError('bounds changed')
for _ in range(100):
    unused = factory(); used = factory(); used.S.chain(1200)
    del unused, used
gc.collect()
print('heap allocation retry, completed cache restore and destruction passed')
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')


def test_original_dynamic_shell_collect_detachment_and_ownership(tmp_path):
    import modelx as mx
    m = mx.new_model('DynamicShell')
    policies = m.new_space(
        'Policies',
        formula='def Policies(policy_id, shock=0.0):\n return {"refs": {"offset": float(policy_id) + shock}}\n',
        refs={'offset': 0.0, 'policy_ids': (1, 2, 3)},
        collect_over={'policy_id': (1, 2, 3)},
        collect_outputs='lambda: {"sum": {"total": fib(7)}, "individual": {"values": {0: fib(0), 7: fib(7)}}}',
    )
    policies.new_cells('seed', 'def seed():\n return offset\n')
    policies.new_cells('fib', 'def fib(n):\n return seed() if n < 2 else 0.5 * (fib(n-1) + fib(n-2))\n')
    outer = m.new_space('Outer')
    outer.Policies = policies
    outer.new_cells('total', 'def total():\n value = 0.0\n for i in Policies.policy_ids:\n  value += Policies[i].fib(7)\n return value\n')
    m.export(tmp_path / 'model')
    m.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\nmx_model.Outer.total()\nmx_model.Policies.collect()\n', encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py')], 'build')
    receipt = json.loads((tmp_path / 'model_cy/_mx_classes.frame-shell.json').read_text())
    assert any(r['cell'] == 'fib' and r['execution'] == 'frame_exact' for r in receipt['cells'])
    (tmp_path / 'verify.py').write_text('''
import pickle
from model_cy import mx_model
from model_cy._mx_frame_owner import SpaceFrameOwner
created = []
init = SpaceFrameOwner.__init__
def record(self, *args):
    init(self, *args); created.append(self)
SpaceFrameOwner.__init__ = record
factory = type(mx_model)
m = factory(); s = m.Policies
# Direct caches retain independent values, including early policies.
assert m.Outer.total() == 6.0 and len(created) == 3
for i in (3, 1, 2):
    for n in range(8): assert s[i].fib(n) == float(i)
assert len(created) == 3
q = pickle.loads(pickle.dumps(s[1]))
assert q.fib(0) == q.fib(7) == 1.0
created.clear()
r = s.collect()
assert r.sum.total == 6.0 and len(created) == 1
for i in (1, 2, 3):
    assert r.individual[i].values == {0: float(i), 7: float(i)}
assert s.collect() is r and len(created) == 1
changed = s.collect(5.0)
assert changed.sum.total == 21.0
assert changed.individual[1].values == {0: 6.0, 7: 6.0}
assert r.individual[1].values == {0: 1.0, 7: 1.0}
# Collect scratch has not replaced direct policy owners.
for i in (1, 2, 3): assert s[i].fib(7) == float(i)
# The modelx export container must also serialize cached collect results.
q = pickle.loads(pickle.dumps(s[1]))
assert q.fib(0) == q.fib(7) == 1.0
assert q._parent.collect().sum.total == 6.0
restored = pickle.loads(pickle.dumps(r))
assert restored.sum.total == 6.0 and restored.individual[3].values == {0: 3.0, 7: 3.0}
print('dynamic construction, compiled portfolio, collect cache and detachment passed')
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')
