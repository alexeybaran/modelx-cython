"""Eligibility and behavioral tests for the complete native execution path."""
import ast
import os
from pathlib import Path
import subprocess
import sys

from modelx_cython.integrated_native import local_indices, EnumTests, simple


def test_loop_eligibility():
    methods = {n.name: n for n in ast.parse('''
def bound(self) -> _mx_cy.longlong: pass
def child(self, t: _mx_cy.longlong) -> _mx_cy.double: pass
def object_child(self, t: object) -> object: pass
''').body}
    def infer(body):
        fn = ast.parse('def f(self):\n' + ''.join('    '+s+'\n' for s in body.splitlines())).body[0]
        return local_indices(fn, methods)
    assert infer('v = 0.0\nfor t in range(self.bound()):\n    v += self.child(t)\nreturn v') == ['t']
    for body in (
        'for t in range(self.bound()):\n    v = t * t\nreturn v',
        'for t in range(self.bound()):\n    v = self.child(t)\nreturn t',
        'for t in range(self.bound()):\n    v = self.object_child(t)',
        'range = lambda x: [x]\nfor t in range(self.bound()):\n    v = self.child(t)',
        'for t in range(self.bound()):\n    t = 2\n    v = self.child(t)',
    ):
        assert infer(body) == []


def test_enum_lowering_truth_only():
    fn = ast.parse('''
def f(self, value):
    raw = value == self._mx_enum_member_0
    if value == self._mx_enum_member_0:
        return raw
''')
    pass_ = EnumTests(); changed = pass_.visit(fn)
    assert pass_.count == 1
    assert isinstance(changed.body[0].body[0].value, ast.Compare)


def test_introspection_excluded():
    for call in ('locals()', 'vars()', 'eval("1")', 'dir()'):
        fn = ast.parse(f'def f(self):\n return float(len({call}))\n').body[0]
        assert not simple(fn)


def test_integrated_mutation_and_exceptions(tmp_path):
    """Exercise borrowed Data indexing, owner lifetime, overflow exclusions and pickle."""
    import modelx as mx
    import numpy as np
    m = mx.new_model('Integrated'); d = m.new_space('Data'); s = m.new_space('S')
    d.values = np.array([2., 3., 5., 7.])
    d.new_cells(name='array', formula='def array():\n return values\n')
    s.data = d; s.bound = 4; s.flags = {'fail': False}; s.counts = {}; s.callback = lambda: 0
    formulas = {
        'length': 'def length():\n return bound\n',
        'leaf': "def leaf(t):\n counts[t] = counts.get(t, 0) + 1\n return data.array()[t]\n",
        'total': 'def total():\n v = 0.0\n for t in range(length()):\n  v += leaf(t)\n return v\n',
        'parent': "def parent(t):\n v = leaf(t)\n if flags['fail']: raise ValueError('retry')\n return v * 2\n",
        'callback_value': 'def callback_value(t):\n return data.array()[callback()] + leaf(t)\n',
        'large': 'def large():\n t = 2 ** 100\n return float(t * t)\n',
        'implicit': "def implicit():\n counts['implicit'] = counts.get('implicit',0)+1\n if bound > 0: return 1.0\n",
    }
    # Exportable callback is supplied via a module, not serialized code objects.
    (tmp_path / 'hooks.py').write_text('def index():\n return 0\n')
    sys.path.insert(0, str(tmp_path))
    import hooks
    s.callback = hooks.index
    for n, f in formulas.items(): s.new_cells(name=n, formula=f)
    m.export(tmp_path / 'model'); m.close()
    (tmp_path / 'sample.py').write_text('from model import mx_model\np=mx_model.S\np.total()\nfor t in range(4): p.parent(t);p.callback_value(t)\np.large()\np.implicit()\n')
    (tmp_path / 'spec.py').write_text("{'spaces': {'Data': {'cells': {'array': {'return_type': 'memoryview'}}}}}")
    # An arbitrary callback index makes automatic usage inference conservative.
    # Exercise the supported explicit view contract for this adversarial case.
    env = os.environ.copy(); env['PYTHONPATH'] = str(tmp_path) + os.pathsep + env.get('PYTHONPATH', '')
    env['MX_CY_INTEGRATED_NATIVE'] = '1'
    def run(args, name):
        cp = subprocess.run([sys.executable, *args], cwd=tmp_path, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (tmp_path / (name + '.log')).write_text(cp.stdout)
        assert cp.returncode == 0, cp.stdout[-18000:]
    run(['-m', 'modelx_cython', 'model', '--sample', 'sample.py', '--spec', 'spec.py'], 'build')
    src = (tmp_path / 'model_cy/_mx_classes.py').read_text()
    frame_manifest = tmp_path / 'model_cy/_mx_frame_S/manifest.json'
    if frame_manifest.exists():
        import json
        routes = json.loads((tmp_path / 'model_cy/_mx_classes.frame-shell.json').read_text())['cells']
        assert all(r['execution'] == 'frame_exact' for r in routes if r['cell'] in ('leaf', 'total', 'large'))
        manifest = json.loads(frame_manifest.read_text())
        assert manifest['cells']['large']['calculator'] == 'cython_scalar'
        assert '_mx_n_c_S_total' not in src  # no competing fused formula entry
    else:
        assert '_mx_data_' in src
    (tmp_path / 'verify.py').write_text('''import pickle, numpy as np
from model_cy import mx_model
factory = type(mx_model)
m=factory(); p=m.S
assert p.total()==17. and p.counts=={0:1,1:1,2:1,3:1}
assert p.total()==17. and p.leaf(0)==2.
m=factory();p=m.S
view=m.Data.array(); m.Data.values[1]=13.
assert p.leaf(1)==13.
p.flags['fail']=True
try:p.parent(2)
except ValueError:pass
else:raise AssertionError('missing exception')
assert p.counts[2]==1
p.flags['fail']=False
assert p.parent(2)==10. and p.counts[2]==1
assert p.large()==float((2**100)**2)
for n in (-3,0,1,4):
 m=factory();p=m.S;p.bound=n
 assert p.total()==sum([2.,3.,5.,7.][:max(n,0)])
m=factory();p=m.S;p.bound=0
assert p.implicit()==0. and p.implicit()==0. and p.counts['implicit']==1
# Exact Data receiver is retained across callback evaluation, even if S.data
# changes. The subscript must continue reading the original view.
m=factory();p=m.S; old=m.Data; old.array()
other=factory();other.Data.values=np.array([20.,30.,50.,70.])
def switch():
 p.data=other.Data
 return 0
p.callback=switch
assert p.callback_value(1)==32.
# A Python override of the Data getter must still run on every access.
class Custom(type(old)):
 def array(self): return np.array([100.,200.,300.,400.])
m=factory();p=m.S;p.data=Custom(m)
assert p.leaf(0)==100. and p.total()==1000.
print('integrated contracts PASS')
''')
    run(['verify.py'], 'verify')


def test_native_enum_comparison_contract(tmp_path):
    from modelx_cython.integrated_native import ENUM_COMPARE_SUPPORT
    (tmp_path/'compare.pxi').write_text(ENUM_COMPARE_SUPPORT)
    (tmp_path/'compare.pyx').write_text('include "compare.pxi"\ndef equal(a,b,invert=False):\n return bool(_mx_enum_equal(a,b,invert))\n')
    (tmp_path/'setup.py').write_text("from setuptools import setup\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize('compare.pyx'))\n")
    cp=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=tmp_path,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert cp.returncode==0,cp.stdout
    (tmp_path/'verify.py').write_text('''from compare import equal
from enum import IntEnum
class E(IntEnum): A=1; B=2; HUGE=2**100
for a in (E.A,E.B,E.HUGE,-1,0,1,2,2**100):
 for b in (E.A,E.B,E.HUGE,-1,0,1,2,2**100):
  assert equal(a,b)==bool(a==b)
  assert equal(a,b,True)==bool(a!=b)
class F(IntEnum): A=1
F.__eq__=lambda a,b: False
assert equal(F.A,F.A) is False
class Bad:
 def __eq__(self,other): raise ValueError('bad')
try: equal(Bad(),E.A)
except ValueError: pass
else: raise AssertionError('exception swallowed')
class Truth:
 def __bool__(self): raise RuntimeError('truth')
class Odd:
 def __eq__(self,other): return Truth()
try: equal(Odd(),E.A)
except RuntimeError: pass
else: raise AssertionError('truth conversion skipped')
''')
    cp=subprocess.run([sys.executable,'verify.py'],cwd=tmp_path,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert cp.returncode==0,cp.stdout
