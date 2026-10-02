"""Subclass dispatch shares native, typed and object frame caches."""
import json
from test_original_frame_shell import _run


def test_subclasses_use_authoritative_frame_caches_and_checked_failures(tmp_path):
    import modelx as mx
    model = mx.new_model('VirtualMemo'); space = model.new_space('S')
    space.offset = 0.0; space.fail = False; space.counts = {}
    formulas = {
        'dense': 'def dense(n):\n return float(n) + offset\n',
        'keyed': 'def keyed(x):\n return x + offset\n',
        'object_key': 'def object_key(key):\n return float(len(key)) + offset\n',
        'axes': 'def axes(n, key):\n return object_key(key) + n\n',
        'scalar': 'def scalar():\n counts["scalar"] = counts.get("scalar", 0) + 1\n return offset\n',
        'typed_scalar': 'def typed_scalar():\n return float("0.5") + offset\n',
        'obj': 'def obj(key):\n return {key: offset}\n',
        'obj_scalar': 'def obj_scalar():\n return {"offset": offset}\n',
        'parent': 'def parent(n):\n return dense(n) + dense(n)\n',
        'failure': 'def failure(n):\n child = dense(n)\n if fail: raise ValueError("retry")\n return child + 1.0\n',
        'recursive': 'def recursive(n):\n return float(n) if n < 2 else 0.5 * (recursive(n-1) + recursive(n-2))\n',
        'uncached': 'def uncached(n):\n return dense(n) + offset\n',
    }
    for name, formula in formulas.items(): space.new_cells(name, formula)
    space.uncached.is_cached = False
    model.export(tmp_path / 'model'); model.close()
    (tmp_path / 'sample.py').write_text('''from model import mx_model
s = mx_model.S
for n in range(4):
 s.parent(n); s.failure(n); s.recursive(n); s.uncached(n)
 s.keyed(float(n)); s.axes(n, 'aa'); s.obj('aa')
 s.axes(n, ('a', 'b')); s.obj(('a', 'b'))
s.scalar(); s.typed_scalar(); s.obj_scalar()
''', encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py')], 'build')
    rows = json.loads((tmp_path / 'model_cy/_mx_classes.frame-shell.json').read_text())['cells']
    assert all(row['derived_cache'] == 'frame_virtual' for row in rows)
    manifest = json.loads((tmp_path / 'model_cy/_mx_frame_S/manifest.json').read_text())
    assert manifest['cells']['keyed']['storage'] == 'growing_numeric'
    assert manifest['cells']['object_key']['storage'] == 'typed_python_dict'
    assert manifest['cells']['axes']['storage'] == 'typed_epoch_slabs'
    assert manifest['cells']['typed_scalar']['cached'] and manifest['cells']['typed_scalar']['slots'] == 1
    assert manifest['cells']['obj']['storage'] == 'python_dict'
    assert manifest['cells']['obj_scalar']['storage'] == 'object_singleton'
    (tmp_path / 'overrides.pyx').write_text('''from model_cy._mx_classes cimport _c_S
cdef class Private(_c_S):
 cdef double _f_dense(self, long long n) except? -1:
  self.counts[n] = self.counts.get(n, 0) + 1
  return float(n - 1) + self.offset
 cdef double _f_keyed(self, double x) except? -1:
  return x - 1.0 + self.offset
 cdef double _f_object_key(self, object key) except? -1:
  return float(len(key)) - 1.0 + self.offset
 cdef object _f_obj(self, object key):
  return {key: self.object_key(key)}
cdef class Public(_c_S):
 cpdef double dense(self, long long n) except? -1:
  return 40.0
def owner(_c_S s): return s._mx_frame_owner
def private_dense(_c_S s, long long n): return s._f_dense(n)
''', encoding='utf-8')
    (tmp_path / 'setup_override.py').write_text(
        "from setuptools import setup\nfrom Cython.Build import cythonize\n"
        "setup(ext_modules=cythonize('overrides.pyx'))\n", encoding='utf-8')
    _run(tmp_path, ['setup_override.py', 'build_ext', '--inplace'], 'subclass-build')
    (tmp_path / 'verify.py').write_text('''import pickle
from model_cy import mx_model
from overrides import Private, Public, owner, private_dense
m = type(mx_model)(); s = Private(m)
s.offset = 0.0; s.fail = False; s.counts = {}
assert s.parent(0) == -2.0 and s.dense(0) == -1.0
k = owner(s).kernel
assert k.call('parent', (0,)) == -2.0 and k.call('dense', (0,)) == -1.0
assert s.counts == {0: 1}
s.offset = 100.0
assert s.parent(0) == -2.0 and private_dense(s, 0) == 99.0
assert s.dense(0) == -1.0 and s.counts[0] == 2
assert s.uncached(0) == 99.0
s.offset = 200.0; assert s.uncached(0) == 199.0
s.fail = True
for _ in range(2):
 try: s.failure(1)
 except ValueError as e: assert str(e) == 'retry'
 else: raise AssertionError('failed parent was committed')
assert s.counts[1] == 1
s.offset = 300.0; s.fail = False
assert s.failure(1) == 201.0 and k.call('failure', (1,)) == 201.0
assert s.keyed(-0.0) == 299.0 and s.keyed(0.0) == 299.0
for i in range(300): assert s.keyed(i + 0.5) == i + 299.5
assert s.object_key('aa') == 301.0
for n in (-100, -1, 0, 1, 2, 10000): assert s.axes(n, 'aa') == n + 301.0
assert s.obj('aa') == {'aa': 301.0} and k.call('obj', ('aa',)) == {'aa': 301.0}
assert s.scalar() == 300.0 and s.typed_scalar() == 300.5 and s.obj_scalar() == {'offset': 300.0}
s.offset = 500.0
assert s.scalar() == 300.0 and s.typed_scalar() == 300.5 and s.obj_scalar() == {'offset': 300.0}
assert s.counts['scalar'] == 1
expected = [0.0, 1.0]
for n in range(2, 4): expected.append(0.5 * (expected[n-1] + expected[n-2]))
for n, value in enumerate(expected):
 assert s.recursive(n) == value and k.call('recursive', (n,)) == value
q = pickle.loads(pickle.dumps(s)); q.offset = 1000.0
assert q.parent(0) == -2.0 and q.dense(1) == 200.0
assert q.obj('aa') == {'aa': 301.0} and q.axes(10000, 'aa') == 10301.0
assert q.scalar() == 300.0 and q.typed_scalar() == 300.5 and q.keyed(299.5) == 598.5
assert owner(q).kernel.call('parent', (0,)) == -2.0
p = Public(m); p.offset = 0.0; p.fail = False; p.counts = {}
assert p.parent(0) == 80.0 and owner(p).kernel.call('parent', (0,)) == 80.0
class PythonOverride(type(m.S)):
 def dense(self, n): return 12.0
p = PythonOverride(m); p.offset = 0.0; p.fail = False; p.counts = {}
assert p.parent(0) == 24.0 and owner(p).kernel.call('parent', (0,)) == 24.0
p.dense = lambda n: 20.0
assert p.parent(1) == 40.0 and p.parent(0) == 24.0
for n in (-1, 4):
 try: s.dense(n)
 except IndexError: pass
 else: raise AssertionError('dense bounds lost')
print('subclass authoritative frame caches and checked rollback passed')
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')
