"""Behavioral cache/storage gates; compile real generated Spaces and overrides."""
import os
from pathlib import Path
import subprocess
import sys


def test_cache_storage_behavior(tmp_path):
    import modelx as mx
    m=mx.new_model('CacheStorage');s=m.new_space('S')
    s.counts={};s.flags={'fail':False};s.offset=0.0
    formulas={
      'leaf': "def leaf(x):\n counts['leaf']=counts.get('leaf',0)+1\n return x-1.0+offset\n",
      'pair': "def pair(i,x):\n counts['pair']=counts.get('pair',0)+1\n return i+x\n",
      'parent': "def parent(x):\n value=leaf(x)\n counts['parent']=counts.get('parent',0)+1\n if flags['fail']: raise ValueError('retry')\n return value+leaf(x)\n",
      'recursive': "def recursive(x):\n counts['recursive']=counts.get('recursive',0)+1\n return 0.0 if x<=0 else recursive(x-1.0)+1.0\n",
      'array_leaf': "def array_leaf(t):\n counts['array_leaf']=counts.get('array_leaf',0)+1\n return float(t-1)+offset\n",
      'array_parent': "def array_parent(t):\n value=array_leaf(t)\n counts['array_parent']=counts.get('array_parent',0)+1\n if flags['fail']: raise ValueError('retry')\n return value+array_leaf(t)\n",
      'defaulted': "def defaulted(x=1.0):\n return leaf(x)\n",
      'object_arg': "def object_arg(key):\n return float(len(key))\n",
      'uncached': "def uncached(x):\n counts['uncached']=counts.get('uncached',0)+1\n return leaf(x)\n",
      'closure': "def closure(t):\n return sum(array_leaf(i) for i in range(t))\n",
      'fallback_child': "def fallback_child(x):\n counts['fallback_child']=counts.get('fallback_child',0)+1\n if x==2.0: fallback_child(float('inf'))\n return 0.0 if x==float('inf') else x\n",
    }
    for name,formula in formulas.items():s.new_cells(name=name,formula=formula)
    s.uncached.is_cached=False
    m.export(tmp_path/'model');m.close()
    (tmp_path/'sample.py').write_text("from model import mx_model\np=mx_model.S\nfor t in range(5):\n p.parent(float(t))\n p.recursive(float(t))\n p.array_parent(t)\n p.uncached(float(t))\n p.pair(t,0.25)\np.defaulted()\np.object_arg('ab')\np.closure(4)\np.fallback_child(0.25)\n")
    env=os.environ.copy();env['PYTHONPATH']=str(tmp_path)+os.pathsep+env.get('PYTHONPATH','')
    def run(cmd, label):
        r=subprocess.run(cmd,cwd=tmp_path,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (tmp_path/(label+'.log')).write_text(r.stdout)
        assert r.returncode==0,r.stdout[-18000:]
    run([sys.executable,'-m','modelx_cython',str(tmp_path/'model'),'--sample',str(tmp_path/'sample.py'),'--no-spec'],'build')
    src=(tmp_path/'model_cy/_mx_classes.py').read_text()
    pxd=(tmp_path/'model_cy/_mx_classes.pxd').read_text()
    if '_mx_nc_' in pxd:
        assert '_mx_nc_leaf' in pxd and '_mx_nc_pair' in pxd
        assert '_mx_nc_defaulted' not in pxd and '_mx_nc_object_arg' not in pxd and '_mx_nc_uncached' not in pxd
    if '_mx_values' in pxd:
        assert 'self._mx_values._v_array_leaf[t]' in src
        assert 'self._has_array_leaf[t]' in src
    (tmp_path/'override.pyx').write_text("from model_cy._mx_classes cimport _c_S\ncdef class Private(_c_S):\n cdef double _f_leaf(self, double x):\n  return 40.0\ncdef class Public(_c_S):\n cpdef double leaf(self, double x):\n  return 50.0\n")
    (tmp_path/'override_setup.py').write_text("from setuptools import setup\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize('override.pyx'))\n")
    run([sys.executable,'override_setup.py','build_ext','--inplace'],'override-build')
    (tmp_path/'verify.py').write_text("""import gc, math
from model_cy import mx_model
factory=type(mx_model)
m=factory();p=m.S
assert p.leaf(-0.0)==-1.0 and p.leaf(0.0)==-1.0
assert p.parent(0.0)==-2.0 and p.counts=={'leaf':1,'parent':1}
assert p.parent(1.0)==0.0 and p.leaf(1.0)==0.0
p.flags['fail']=True
for i in range(2):
 try:p.parent(2.0)
 except ValueError as e:assert str(e)=='retry'
 else:raise AssertionError('exception swallowed')
assert p.counts=={'leaf':3,'parent':4}
p.flags['fail']=False
assert p.parent(2.0)==2.0 and p.counts=={'leaf':3,'parent':5}
# Repeated growth and collisions; every old entry remains cached.
for i in range(3000):assert p.pair(i,float(i)+0.25)==2*i+0.25
for i in reversed(range(3000)):assert p.pair(i,float(i)+0.25)==2*i+0.25
assert p.counts['pair']==3000
assert p.recursive(300.0)==300.0 and p.recursive(0.0)==0.0 and p.counts['recursive']==301
n=p.counts['leaf'];assert p.leaf(float('inf'))==float('inf')
assert p.leaf(float('inf'))==float('inf') and p.counts['leaf']==n+1
assert p.leaf(0.0)==-1.0 and p.leaf(1.0)==0.0 and p.counts['leaf']==n+1
assert math.isnan(p.leaf(float('nan')))
assert p.fallback_child(0.25)==0.25 and p.fallback_child(2.0)==2.0
assert p.fallback_child(2.0)==2.0 and p.fallback_child(0.25)==0.25 and p.counts['fallback_child']==3
assert p.array_leaf(0)==-1.0 and p.array_parent(0)==-2.0
p.flags['fail']=True
for i in range(2):
 try:p.array_parent(1)
 except ValueError:pass
 else:raise AssertionError('exception swallowed')
assert p.counts['array_leaf']==2 and p.counts['array_parent']==3
p.flags['fail']=False;assert p.array_parent(1)==0.0 and p.counts['array_leaf']==2
assert p.defaulted()==0.0 and p.object_arg('xyz')==3.0
p.uncached(0.0);p.uncached(0.0);assert p.counts['uncached']==2
assert p.closure(4)==2.0
try:p.array_leaf(100)
except IndexError:pass
else:raise AssertionError('bounds changed')
# Rebinding and public/private overrides retain ordinary dispatch.
m=factory();p=m.S;p.offset=10.;assert p.parent(0.0)==18.
class Override(type(p)):
 def leaf(self,x):return 12.0
sub=Override(m);sub.counts={};sub.flags={'fail':False};sub.offset=0.
assert sub.parent(0.0)==24.0
sub.leaf=lambda x:20.0
assert sub.parent(1.0)==40.0
from override import Private, Public
for cls,expected in [(Private,80.0),(Public,100.0)]:
 sub=cls(m);sub.counts={};sub.flags={'fail':False};sub.offset=0.
 assert sub.parent(0.0)==expected and sub.leaf(0.0)==expected/2
 assert sub.parent(0.0)==expected and sub.counts=={'parent':1}
# Both unused and populated native storage are destroyed with their owners.
for i in range(500):
 q=factory()
 if i%2:q.S.array_leaf(0);q.S.pair(i,0.25)
 del q
 gc.collect()
""")
    run([sys.executable,'verify.py'],'verify')
    (tmp_path/'pickle_verify.py').write_text("import pickle\nfrom model_cy import mx_model\np=mx_model.S\np.leaf(-0.0)\np.array_leaf(0)\nfor i in range(100):p.pair(i,i+0.25)\nq=pickle.loads(pickle.dumps(p))\nassert q.leaf(0.0)==-1.0 and q.array_leaf(0)==-1.0\nassert q.counts['leaf']==1 and q.counts['array_leaf']==1\nfor i in range(100):assert q.pair(i,i+0.25)==2*i+0.25\nassert q.counts['pair']==100\np.leaf(float('inf'))\nq=pickle.loads(pickle.dumps(p))\nassert q.leaf(0.0)==-1.0 and q.leaf(float('inf'))==float('inf')\n")
    run([sys.executable,'pickle_verify.py'],'pickle-verify')
