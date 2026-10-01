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
        'index_value': 'def index_value(n):\n return n - 1\n',
        'even': 'def even(n):\n return n % 2 == 0\n',
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
        'for n in range(8):\n s.observed(n)\n s.fib(n)\n s.failure(n)\n s.uncached(n)\n s.index_value(n)\n s.even(n)\n'
        's.input_value()\n', encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py'), '--translate-only'], 'translate')
    receipt = json.loads((tmp_path / 'model_cy/_mx_classes.frame-shell.json').read_text())
    assert all(row['formula_compiler'] == 'Cython' for row in receipt['cells'])
    assert all(row['receiver_representation'] ==
               ('native_frame' if row['execution'] == 'frame_exact' else 'original_space')
               for row in receipt['cells'])
    routes = {row['cell']: row['execution'] for row in receipt['cells'] if row['space'].endswith('._c_S')}
    assert routes['fib'] == routes['parent'] == routes['failure'] == 'frame_exact'
    assert routes['observed'] == routes['uncached'] == 'frame_exact'
    shell_source = (tmp_path / 'model_cy/_mx_classes.py').read_text()
    assert '.call(' not in shell_source
    assert '_mx_frame__c_S_fib(self._mx_frame_owner.kernel, self, n,' in shell_source
    private_pxd = next((tmp_path / 'model_cy/_mx_frame_S').rglob('_kernel.pxd')).read_text()
    assert 'cdef double _mx_shell_fib(object kernel, object receiver, long long n,' in private_pxd
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
assert s.index_value(0) == -1 and s.index_value(1) == 0
assert s.even(0) is True and s.even(1) is False
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


def test_original_shell_connected_space_checked_edges(tmp_path):
    """Resolved original Space references enter Cython without a second facade."""
    import modelx as mx
    m = mx.new_model('ConnectedShell')
    data = m.new_space('Data')
    data.values = [10.0, 20.0, 30.0, 40.0]
    data.new_cells('value', 'def value(n):\n return values[n]\n')
    import importlib
    import pandas as pd
    (tmp_path/'fixture_lookup.py').write_text(
        'class Lookup:\n'
        ' def get(self, table, key, column, default=0.0):\n'
        '  return table.loc[key, column] if key in table.index else default\n'
        'lookup = Lookup()\n', encoding='utf-8')
    sys.path.insert(0, str(tmp_path))
    try:
        lookup = importlib.import_module('fixture_lookup').lookup
    finally:
        sys.path.pop(0)
    data.table_data = pd.DataFrame({'value': data.values})
    data.new_cells('table', 'def table():\n return table_data\n')
    s = m.new_space('S')
    s.data = data
    s.other = data
    s.lookup = lookup
    s.hook = None
    s.new_cells('checked_lookup', 'def checked_lookup(n):\n return float(lookup.get(data.table(), n, \'value\', 0.0))\n')
    s.new_cells('read', 'def read(n):\n return data.value(n)\n')
    s.new_cells('alias', 'def alias(n):\n return other.value(n)\n')
    s.new_cells('after_callback',
                'def after_callback(n):\n hook()\n return data.value(n) + read(n)\n')
    m.export(tmp_path / 'model')
    m.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\n'
        's=mx_model.S\ns.hook=lambda: None\n'
        'for n in range(4):\n s.read(n)\n s.alias(n)\n s.after_callback(n)\n s.checked_lookup(n)\n', encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path/'model'), '--no-spec',
                    '--sample', str(tmp_path/'sample.py'), '--translate-only'], 'translate')
    receipt = json.loads((tmp_path/'model_cy/_mx_classes.frame-shell.json').read_text())
    routes = {(row['space'].split('.')[-1], row['cell']): row['execution']
              for row in receipt['cells']}
    for key in (('_c_Data', 'value'), ('_c_Data', 'table'), ('_c_S', 'read'), ('_c_S', 'alias'), ('_c_S', 'after_callback'), ('_c_S','checked_lookup')):
        assert routes[key] == 'frame_exact', (key, routes)
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path/'model'), '--compile-only'], 'build')
    (tmp_path/'verify.py').write_text('''
import pickle
from model_cy import mx_model
from model_cy._mx_frame_owner import SpaceFrameOwner
created = []
original_init = SpaceFrameOwner.__init__
def record_owner(self, module, inputs):
    original_init(self, module, inputs)
    created.append(module)
SpaceFrameOwner.__init__ = record_owner
m = type(mx_model)(); s = m.S
assert s.data is s.other is m.Data
assert s.read(0) == s.alias(0) == s.checked_lookup(0) == 10.0
assert any('_mx_frame_S.' in name for name in created)
assert any('_mx_frame_Data.' in name for name in created)
other_model = type(mx_model)()
other_model.Data.values = [100.0, 200.0, 300.0, 400.0]
other_model.Data.table_data['value'] = other_model.Data.values
s.data = other_model.Data
assert s.read(0) == 10.0  # rebinding does not invalidate a completed result
assert s.read(1) == s.checked_lookup(1) == 200.0 and s.alias(1) == 20.0
def rebind():
    s.data = m.Data
s.hook = rebind
assert s.after_callback(2) == 60.0
assert s.data is m.Data
assert s.read(2) == 30.0
m.Data.values[3] = 44.0
assert s.read(3) == 44.0  # mutable input observed before the first evaluation
q = pickle.loads(pickle.dumps(m))
assert q.S.data is q.S.other is q.Data
q.Data.values[0] = -100.0
assert q.S.read(0) == 10.0 and q.S.read(3) == 44.0
print('original connected Spaces, aliases, live rebinding and retained caches passed')
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
    outer.new_cells('collected_total', 'def collected_total():\n return Policies.collect().sum.total\n')
    m.export(tmp_path / 'model')
    m.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\nmx_model.Outer.total()\nmx_model.Outer.collected_total()\n', encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py')], 'build')
    receipt = json.loads((tmp_path / 'model_cy/_mx_classes.frame-shell.json').read_text())
    assert any(r['cell'] == 'fib' and r['execution'] == 'frame_exact' for r in receipt['cells'])
    shared_plan = json.loads((tmp_path / 'model_cy/_mx_compilation_plan.json').read_text())
    declaration = shared_plan['spaces']['model_cy.Policies']['collection']
    assert declaration['over'] == 'policy_id' and declaration['parameters'] == ['shock']
    assert declaration['domain_source'] == 'tuple(self.policy_ids)'
    assert shared_plan['collection_consumers'] == [{
        'cell': 'model_cy.Outer.collected_total', 'expression': 'self.Policies.collect()',
        'target': 'model_cy.Policies', 'declared': True}]
    (tmp_path / 'verify.py').write_text('''
import pickle
from model_cy import mx_model
from model_cy._mx_frame_owner import SpaceFrameOwner
created = []
init = SpaceFrameOwner.__init__
def record(self, *args):
    init(self, *args)
    if '_mx_frame_Policies.' in self.module: created.append(self)
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


def test_original_shell_object_boundaries_and_relocated_runtime(tmp_path):
    import modelx as mx
    m = mx.new_model('ObjectShell')
    m.allow_none = True
    s = m.new_space('S')
    s.flags = {'fail': False}
    s.factory = dict
    s.offset = -1.0
    for name, source in {
        'child': 'def child(n):\n return offset + float(n)\n',
        'make': 'def make():\n return factory()\n',
        'report': ('def report(n):\n item = make()\n value = child(n)\n'
                   ' if flags["fail"]: raise ValueError("object retry")\n'
                   ' return {"item": item, "value": value}\n'),
        'parent': 'def parent(n):\n return report(n)["value"] * 2.0\n',
        'caught': ('def caught(n):\n try:\n  return parent(n)\n'
                   ' except ValueError:\n  return child(n)\n'),
        'nothing': 'def nothing():\n return None\n',
    }.items():
        s.new_cells(name, source)
    s.make.is_cached = False
    m.export(tmp_path / 'model')
    m.close()
    (tmp_path/'sample.py').write_text(
        'from model import mx_model\ns=mx_model.S\n'
        'for n in range(3):\n s.caught(n)\n s.report(n)\n'
        's.nothing()\n', encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path/'model'), '--no-spec',
                    '--sample', str(tmp_path/'sample.py')], 'build')
    receipt = json.loads((tmp_path/'model_cy/_mx_classes.frame-shell.json').read_text())
    assert all(row['execution'] == 'frame_exact' for row in receipt['cells'])
    manifest = json.loads((tmp_path/'model_cy/_mx_frame_S/manifest.json').read_text())
    assert manifest['cells']['make']['cached'] is False
    assert manifest['cells']['report']['coverage']['body_backend'] == 'cython_gil'
    assert not any(row['coverage']['detached_python'] for row in manifest['cells'].values())
    binding_source = next((tmp_path/'model_cy/_mx_frame_S').rglob('_python.py')).read_text()
    assert 'def report(' not in binding_source and 'def make(' not in binding_source
    # Relocation must not need compiler/modelx imports or the build directory.
    import shutil
    relocated = tmp_path/'relocated'
    shutil.copytree(tmp_path/'model_cy', relocated/'model_cy')
    (relocated/'verify.py').write_text('''
import gc, importlib.abc, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
class NoCompiler(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('modelx', 'modelx_cython'):
            raise AssertionError('Compiled package imported development dependency: ' + name)
sys.meta_path.insert(0, NoCompiler())
from model_cy import mx_model
destroyed = []
class Token:
    def __del__(self): destroyed.append(1)
m = type(mx_model)(); s = m.S
s.factory = Token
s.flags = {'fail': True}
for _ in range(2):
    try: s.parent(0)
    except ValueError as e: assert str(e) == 'object retry'
    else: raise AssertionError('failure was swallowed')
assert len(destroyed) == 2
assert s.caught(1) == 0.0
assert len(destroyed) == 3
s.offset = 100.0
s.flags = {'fail': False}
assert s.parent(0) == -2.0  # completed child survives object failure
assert s.parent(1) == 0.0
assert s.child(2) == 102.0  # replacement before uncached work is synchronized
assert s.report(0) is s.report(0)
assert s.nothing() is None
q = pickle.loads(pickle.dumps(s))
q.offset = 999.0
assert q.parent(0) == -2.0 and q.report(0)['value'] == -1.0
assert q.report(0) is q.report(0) and q.nothing() is None
u = type(mx_model)()
def change_input():
    u.S.offset = 200.0
    return Token()
u.S.factory = change_input
assert u.S.parent(2) == 404.0  # callback changes an input before uncached child work
del u
del s, q, m
gc.collect()
assert len(destroyed) >= 5
print('checked object/numeric failures, uncached effects, cache restore and relocation passed')
''', encoding='utf-8')
    _run(relocated, ['-I', str(relocated/'verify.py')], 'verify')


def test_original_shell_checked_cell_boundary_defaults_and_dense_axes(tmp_path):
    """One unmigrated key path must not remove its callers from the frame."""
    import modelx as mx
    m = mx.new_model('BoundaryShell')
    s = m.new_space('S')
    s.values = {('row', i): float(i * 2) for i in range(4)}
    s.counts = {}
    s.flags = {'fail': False}
    s.offset = 0.0
    s.new_cells('matrix', 'def matrix(i, j=1):\n return float(i * 10 + j) + offset\n')
    s.new_cells('keyed', "def keyed(key):\n counts['keyed'] = counts.get('keyed', 0) + 1\n if flags['fail'] and key[1] == 2: raise ValueError('retry')\n return values[key] + matrix(key[1], j=0)\n")
    s.new_cells('root', "def root(i=1):\n child = matrix(i, j=1)\n return child + float(keyed(('row', i)))\n")
    m.export(tmp_path/'model'); m.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\ns=mx_model.S\nfor i in range(4):\n for j in range(3): s.matrix(i, j=j)\n s.root(i=i)\n', encoding='utf-8')
    (tmp_path/'spec.py').write_text(repr({'spaces': {'S': {'cells_param_size': {'i': 4, 'j': 3}}}}), encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path/'model'), '--sample', str(tmp_path/'sample.py'),
                    '--spec', str(tmp_path/'spec.py'), '--translate-only'], 'translate')
    receipt = json.loads((tmp_path/'model_cy/_mx_classes.frame-shell.json').read_text())
    routes = {r['cell']: r for r in receipt['cells']}
    assert routes['matrix']['execution'] == routes['root']['execution'] == 'frame_exact'
    assert routes['keyed']['execution'] == 'original_cython'
    assert routes['root']['checked_cell_dependencies'] == ['keyed']
    assert routes['matrix']['checked_cell_dependencies'] == []
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path/'model'), '--compile-only'], 'build')
    (tmp_path/'verify.py').write_text("""
import pickle
from model_cy import mx_model
s = type(mx_model)().S
assert s.root() == s.root(i=1) == 23.0
assert s.counts == {'keyed': 1}
s.offset = 100.0; s.values[('row',1)] = 1000.0
assert s.root(1) == 23.0 and s.keyed(('row',1)) == 12.0
s.offset = 0.0; s.flags['fail'] = True
for _ in range(2):
 try: s.root(i=2)
 except ValueError as e: assert str(e) == 'retry'
 else: raise AssertionError('failed parent committed')
assert s.counts == {'keyed': 3}
s.offset = 100.0; s.flags['fail'] = False
assert s.root(i=2) == 145.0  # retained matrix(2,1)=21, newly matrix(2,0)=120
assert s.counts == {'keyed': 4}
for i,j in ((-1,0),(4,0),(0,-1),(0,3)):
 try: s.matrix(i,j=j)
 except IndexError: pass
 else: raise AssertionError('original multidimensional bounds lost')
q = pickle.loads(pickle.dumps(s))
q.offset = -100.0; q.values[('row',1)] = -500.0
assert q.root() == 23.0 and q.keyed(('row',1)) == 12.0
assert q.matrix(2,j=1) == 21.0 and q.root(2) == 145.0
assert q.counts == {'keyed': 4}
print('checked original Cell boundary, defaults, keywords, dense axes and serialization passed')
""", encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')
