"""Compiled contracts for the derived object-key lookup index."""
import os
from pathlib import Path
import subprocess
import sys


def test_object_key_index(tmp_path):
    import modelx as mx
    (tmp_path/'keytypes.py').write_text('''from enum import IntEnum
class Kind(IntEnum): A=1; B=2; NEG=-1; ZERO=0
def plain(x): return float(x)
def index(): return 0
''')
    sys.path.insert(0, str(tmp_path))
    import keytypes
    m=mx.new_model('Keys'); s=m.new_space('S')
    s.Kind=keytypes.Kind; s.category_input={'kind':keytypes.Kind.A}; s.convert=keytypes.plain; s.counts={}; s.fail={'enabled':False}
    s.new_cells(name='key',formula="def key():\n return category_input['kind']\n")
    s.callback=keytypes.index
    s.new_cells(name='index',formula='def index():\n return callback()\n')
    s.new_cells(name='value', formula='''def value(k,t):
 counts[t]=counts.get(t,0)+1
 if fail['enabled']: raise ValueError('retry')
 return convert(k)+t
''')
    s.new_cells(name='child', formula='''def child(k,t):
 if t: return child(k,t-1)+value(k,t)
 return value(k,t)
''')
    s.new_cells(name='path',formula='def path(t):\n return value(key(),t)\n')
    s.new_cells(name='callback_path',formula='def callback_path():\n return value(key(),index())\n')
    s.new_cells(name='fraction',formula="def fraction(k,x):\n counts['fraction']=counts.get('fraction',0)+1\n return convert(k)+x\n")
    s.new_cells(name='fraction_path',formula='def fraction_path(x):\n return fraction(key(),x)\n')
    s.new_cells(name='parent',formula="def parent(k,t):\n v=child(k,t)\n if fail['enabled']:raise ValueError('parent')\n return v\n")
    m.export(tmp_path/'model');m.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\ns=mx_model.S\ns.value(s.Kind.A,0)\ns.child(s.Kind.B,3)\ns.value("3",0)\ns.child("3",3)\ns.path(0)\ns.callback_path()\ns.fraction(s.Kind.A,0.0)\ns.fraction("3",0.0)\ns.fraction_path(0.0)\ns.parent(s.Kind.A,2)\ns.parent("3",2)\n')
    (tmp_path/'spec.py').write_text("{'spaces': {'S': {'cells': {'key': {'return_type': 'object'}}, 'cells_params': {'t': {'size': 650}}}}}")
    env=os.environ.copy();env['PYTHONPATH']=str(tmp_path)+os.pathsep+env.get('PYTHONPATH','')
    env['MX_CY_CONNECTED_NATIVE']='1';env['MX_CY_INTEGRATED_NATIVE']='1'
    def run(args,name):
        p=subprocess.run([sys.executable,*args],cwd=tmp_path,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (tmp_path/(name+'.log')).write_text(p.stdout,encoding='utf-8')
        assert p.returncode==0,p.stdout[-20000:]
    run(['-m','modelx_cython','model','--sample','sample.py','--spec','spec.py'],'build')
    assert (tmp_path/'model_cy/_mx_classes.object-index.json').exists()
    import json
    selected={r['cell'] for r in json.loads((tmp_path/'model_cy/_mx_classes.object-index.json').read_text())}
    assert selected=={'value','fraction'}
    frame_receipt = tmp_path/'model_cy/_mx_classes.frame-shell.json'
    if frame_receipt.exists():
        routes = {r['cell']: r for r in json.loads(frame_receipt.read_text())['cells']}
        assert all(routes[n]['execution'] == 'frame_exact' for n in selected)
        manifest = json.loads((tmp_path/'model_cy/_mx_frame_S/manifest.json').read_text())
        assert all(manifest['cells'][n]['storage'] == 'typed_python_dict' for n in selected)
    else:
        assert '_mx_edge_key_' in (tmp_path/'model_cy/_mx_classes.py').read_text()
    (tmp_path/'verify.py').write_text('''import pickle, math
from enum import IntEnum
from model_cy import mx_model
factory=type(mx_model)
def fresh():return factory().S
p=fresh();K=p.Kind
assert p.path(0)==1 and p.path(1)==2 and p.value(K.A,1)==2 and p.key() is K.A
p=fresh();assert p.fraction(K.A,-0.0)==1 and p.fraction(1,0.0)==1 and p.counts['fraction']==1
assert math.isinf(p.fraction(K.A,float('inf')))
assert math.isnan(p.fraction(K.A,float('nan')))
# Native codes must be checked after evaluating later call arguments.
class CallbackKey(IntEnum): A=1
p=fresh();p.category_input['kind']=CallbackKey.A
assert p.value(CallbackKey.A,0)==1
def change_hash():
 CallbackKey.__hash__=lambda self:17
 return 0
p.callback=change_hash
assert p.callback_path()==1 and p.counts[0]==2
# A failing parent retains completed child entries shared by both paths.
p=fresh();assert p.child(K.A,2)==6
p.fail['enabled']=True
try:p.parent(K.A,2)
except ValueError:pass
else:raise AssertionError('parent did not fail')
p.fail['enabled']=False
assert p.parent(K.A,2)==6 and p.counts=={0:1,1:1,2:1}
p=fresh()
assert p.value(K.NEG,0)==-1 and p.value(K.ZERO,0)==0
assert p.value(K.A,0)==1 and p.value(1,0)==1 and p.counts[0]==3
for t in range(600):assert p.value(K.B,t)==2+t
for t in range(600):assert p.value(2,t)==2+t
assert p.counts[599]==1
assert p.child(K.B,12)==sum(2+t for t in range(13))
# Unsupported domains use ordinary Python equality/hash semantics permanently.
p=fresh();assert p.value(K.A,0)==1
assert p.value(1.0,0)==1 and p.counts[0]==1
assert p.value(2**100,1)==float(2**100)+1
# A stored type's hash can change, even when the next lookup is an ordinary int.
class Mutable(IntEnum): A=1
p=fresh();oracle={(Mutable.A,0):1.0};assert p.value(Mutable.A,0)==1
Mutable.__hash__=lambda self:17
assert ((1,0) in oracle) and p.value(1,0)==1 and p.counts[0]==1
assert (Mutable.A,0) not in oracle
assert p.value(Mutable.A,0)==1 and p.counts[0]==2
# Changing equality must disable lookup acceleration before returning a hit.
class Equality(IntEnum): A=1
p=fresh();assert p.value(Equality.A,0)==1
Equality.__eq__=lambda a,b:False
assert p.value(1,0)==1 and p.counts[0]==2
p=fresh();p.fail['enabled']=True
try:p.value(K.A,1)
except ValueError:pass
else:raise AssertionError('missing exception')
p.fail['enabled']=False
assert p.value(K.A,1)==2 and p.value(K.A,1)==2 and p.counts[1]==2
p=fresh();assert p.value(K.A,0)==1
q=pickle.loads(pickle.dumps(p));assert q.value(K.A,0)==1 and q.counts[0]==1
class Override(type(p)):
 def value(self,k,t):return 20.
q=Override(p._parent)
assert q.child(K.A,2)==60.
print('object index contracts PASS')
''')
    run(['verify.py'],'verify')
