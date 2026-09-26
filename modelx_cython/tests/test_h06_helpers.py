"""Fixed helper bindings, complete-table ownership and input semantics."""
import json, os, pathlib, shutil, subprocess, sys
import numpy as np
import pandas as pd
import pytest


def test_preparation_preserves_keys_and_ownership():
    from modelx_cython.experimental_runtime._mx_helpers import prepare
    from enum import IntEnum
    class Sex(IntEnum): M=0;F=1
    frame=pd.DataFrame({'v':[.1,.2,.3]},index=pd.MultiIndex.from_tuples([(40,'M'),(41,'M'),(40,'F')]))
    curve=prepare(frame,'v','curve',0)
    assert curve.at(curve.code(Sex.M),40)==.1
    assert curve.get(curve.code('missing'),40,7)==7
    assert curve.code(0)==-1  # native group codes are not public labels
    with pytest.raises(KeyError):curve.at(curve.code('M'),42)
    frame.iloc[0,0]=.9
    assert curve.at(curve.code(Sex.M),40)==.1
    assert prepare(frame,'v','curve',0).at(curve.code(Sex.M),40)==.9
    keys=prepare(frame,'','labels',0)
    assert keys.key(40)==40
    with pytest.raises(KeyError):keys.key(39)
    with pytest.raises(ValueError):prepare(pd.DataFrame({'v':[1]},index=[2**53+1]),'v','curve',0)
    with pytest.raises(ValueError):prepare(pd.DataFrame({'v':[1,2]},index=[1,1]),'v','curve',0)
    with pytest.raises(ValueError):prepare(pd.DataFrame({'v':[np.inf]},index=[1]),'v','curve',0)


@pytest.mark.parametrize('helper_namespace', ['mxcf_kernel', 'lifelib.helpers'])
def test_fixed_helper_model_three_environments(tmp_path,monkeypatch,helper_namespace):
    import modelx as mx
    src=pathlib.Path(__file__).parent/'samples/fixed_helpers/mxcf_kernel'
    if helper_namespace == 'lifelib.helpers':
        (tmp_path/'lifelib').mkdir()
        (tmp_path/'lifelib/__init__.py').write_text('')
        destination = tmp_path/'lifelib/helpers'
    else:
        destination = tmp_path/'mxcf_kernel'
    shutil.copytree(src,destination)
    (tmp_path/'colours.py').write_text('from enum import IntEnum\nclass Colour(IntEnum):\n RED=0\n BLUE=1\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    for name in tuple(sys.modules):
        if name == helper_namespace.split('.')[0] or name.startswith(helper_namespace.split('.')[0] + '.') or name == 'colours':
            monkeypatch.delitem(sys.modules, name)
    lookup = importlib.import_module(helper_namespace + '.lookup')
    power = importlib.import_module(helper_namespace + '.lookups').power
    from colours import Colour
    m=mx.new_model('HelperModel');d=m.new_space('Inputs');s=m.new_space('S',formula='lambda key: None')
    d.records_data=pd.DataFrame({'v':[10.,20.,30.], 'kind':['red','blue','red']},index=[1,2,3])
    d.curves=pd.DataFrame({'v':[.1,.2,.3,.4]},index=pd.MultiIndex.from_tuples([('base',1),('base',2),('stress',1),('stress',2)]))
    d.new_cells(name='records',formula='def records():\n return records_data\n')
    d.new_cells(name='table',formula='def table():\n return curves\n')
    s.source_data=d;s.hlp=lookup;s.pow_real=power;s.Colour=Colour
    fs={
      'scenario':"def scenario(value):\n return hlp.key(source_data.table(), value)\n",
      'category':"def category():\n return hlp.category(source_data.records(), key, 'kind', Colour)\n",
      'field':"def field():\n return hlp.field(source_data.records(), key, 'v')\n",
      'curve':"def curve(t):\n return hlp.previous_group(source_data.table(), 'base', t, 'v')\n",
      'linear':"def linear(t):\n return hlp.linear(source_data.records(), t, 'v')\n",
      'missing':"def missing(group, t):\n return hlp.get(source_data.table(), (group,t), 'v', 7.0)\n",
      'total':"def total(t):\n return field() + curve(t) + pow_real(float(t),2.0)\n",
    }
    for name,formula in fs.items():s.new_cells(name=name,formula=formula)
    oracle=[float(s[k].total(2)) for k in (1,2)]
    assert oracle==[14.2,24.2]
    m.export(tmp_path/'model');m.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\nfor k in [1,2]:\n p=mx_model.S[k]\n p.total(2)\n p.scenario("base")\n p.category()\n p.linear(1.5)\n p.missing("base",1)\n p.missing(0,1)\n')
    (tmp_path/'spec.py').write_text("{'experimental_helpers': True}")
    env=os.environ.copy();env['PYTHONPATH']=str(tmp_path)+os.pathsep+env.get('PYTHONPATH','')
    cp=subprocess.run([sys.executable,'-m','modelx_cython',str(tmp_path/'model'),'--sample',str(tmp_path/'sample.py'),'--spec',str(tmp_path/'spec.py')],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (tmp_path/'build.log').write_text(cp.stdout)
    assert cp.returncode==0,cp.stdout[-15000:]
    (tmp_path/'verify.py').write_text('''import importlib
from colours import Colour
from types import SimpleNamespace
for name in ['model','model_cy']:
    root=importlib.import_module(name).mx_model
    m=type(root)()
    p=m.S[1]
    assert p.total(2)==14.2 and m.S[2].total(2)==24.2
    assert p.scenario('stress')=='stress'
    try: p.scenario('bad')
    except KeyError: pass
    else: raise AssertionError('missing label')
    assert p.category() is Colour.RED
    assert p.linear(1.5)==15.
    assert p.missing('absent',1)==7. and p.missing(0,1)==7.
    # Input replacement BEFORE first calculation remains supported. Preparation
    # must not warm the data Cell and silently freeze the serialized input.
    m=type(root)()
    df=m.Inputs.records_data.copy();df.loc[1,'v']=100.
    m.Inputs.records_data=df
    assert m.S[1].field()==100.
    # If the table Cell is cached first, normal upstream semantics retain it.
    m=type(root)();old=m.Inputs.records()
    df=old.copy();df.loc[1,'v']=200.;m.Inputs.records_data=df
    assert m.S[1].field()==10.
    # Rebinding a helper object selects the original formula call, not native code.
    m=type(root)();p=m.S[1]
    p.hlp=SimpleNamespace(field=lambda *args: 123.)
    assert p.field()==123.
    m=type(root)();p=m.S[1];p.pow_real=lambda a,b: 50.
    assert p.total(2)==60.2
print('helper contract passed in exported Python and compiled package')
''')
    cp=subprocess.run([sys.executable,str(tmp_path/'verify.py')],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert cp.returncode==0,cp.stdout
    inventory=json.loads((tmp_path/'model_cy/_mx_helper_inventory.json').read_text())
    assert any(x['status']=='native_fixed_helper' for x in inventory['calls'])
