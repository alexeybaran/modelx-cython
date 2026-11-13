"""Full-key differential checks against the canonical packaged provider."""
import importlib
from pathlib import Path
import pickle
import subprocess
import sys
import pytest
from modelx_cython import runtime as native_runtime
from modelx_helpers import typed_inputs as h
from modelx_helpers.immutable_inputs import Table

@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root=tmp_path_factory.mktemp('keyed-lookup');provider=Path(native_runtime.__file__).parent
    (root/'native.pxd').write_bytes((provider/'_native_inputs.pxd').read_bytes())
    (root/'native.pyx').write_bytes((provider/'_native_inputs.pyx').read_bytes())
    (root/'probe.pyx').write_text('''cimport native
from cpython.ref cimport PyObject
cdef native.LookupContext _context = native.LookupContext()
def invoke(conversion, function, table, key, column, *default):
    cdef native.KeyPart[8] parts
    cdef Py_ssize_t i, n
    values = key if type(key) is tuple else (key,)
    n = len(values)
    if n > 8: raise ValueError('test harness key width')
    for i in range(n):
        value = values[i]
        if type(value) in (int, bool) and -(1 << 63) <= value < (1 << 63):
            native.key_int(&parts[i], value, type(value) is bool)
        else:
            native.key_text(&parts[i], value)
    if default:
        return native.lookup_keyed_float_f64(conversion,function,_context,table,&parts[0],n,type(key) is tuple,column,default[0])
    return native.lookup_keyed_float_f64(conversion,function,_context,table,&parts[0],n,type(key) is tuple,column)
''')
    (root/'setup.py').write_text("from setuptools import setup\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize(['native.pyx','probe.pyx'],compiler_directives={'language_level':3,'freethreading_compatible':True}))\n")
    r=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=root,capture_output=True,text=True)
    (root/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
    sys.path.insert(0,str(root))
    try:yield importlib.import_module('probe')
    finally:sys.path.remove(str(root));sys.modules.pop('probe',None);sys.modules.pop('native',None)

@pytest.mark.parametrize('keys,query', [
    (('M','F'),'M'), ((("M","N",42),("F","Y",41)),('M','N',42)),
    ((("α",-3),("😀",2)),('😀',2)), ((-3,9),9), ((True,),1),
    (((True,'x'),),(1,'x')), (((1,),),(1,)), (((),),()),
    ((),('M',42)), ((("M",42),),('M',43)),
    ((("M",42),),('M',)), ((("M",42),),42),
])
def test_native_and_missing_keys_match_existing_helper(native,keys,query):
    native=native.invoke
    table=Table(keys,[(float(i+1),) for i in range(len(keys))],('rate',))
    try:expected=float(h.lookup(table,query,'rate'))
    except Exception as e:
        with pytest.raises(type(e)) as actual:native(float,h.lookup,table,query,'rate')
        assert actual.value.args==e.args
    else:
        assert native(float,h.lookup,table,query,'rate')==expected
        assert native(float,h.lookup,pickle.loads(pickle.dumps(table)),query,'rate')==expected

@pytest.mark.parametrize('value', [2**53+1,'2.5','bad',None,True,float('nan')])
def test_per_value_conversion_and_fallback(native,value):
    native=native.invoke
    table=Table((('M',42),),((value,),),('rate',))
    try:expected=float(h.lookup(table,('M',42),'rate'))
    except Exception as e:
        with pytest.raises(type(e)) as actual:native(float,h.lookup,table,('M',42),'rate')
        assert actual.value.args==e.args
    else:
        actual=native(float,h.lookup,table,('M',42),'rate')
        assert actual==expected or actual!=actual and expected!=expected

def test_current_helper_default_and_zero_python_fast_path(native):
    native=native.invoke
    table=Table((('M',42),),((1.5,),),('rate',));calls=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is h.lookup.__code__:calls.append('lookup')
    sys.setprofile(profile)
    try:assert native(float,h.lookup,table,('M',42),'rate')==1.5
    finally:sys.setprofile(None)
    assert not calls
    assert native(float,h.lookup,table,('M',43),'rate','2.5')==2.5
    def replacement(*args):calls.append(args);return '7.5'
    assert native(float,replacement,table,('M',42),'rate')==7.5
    assert len(calls)==1


def test_complete_key_and_component_lifetimes(native):
    for key in [(),('α',),(True,),('α',2),('α',False,3),('α','😀',3,True)]:
        table=Table((key,),((1.5,),),('rate',))
        assert native.invoke(float,h.lookup,table,key,'rate')==1.5
        seen=[]
        def replaced(table,key,column):seen.append(key);return '2.5'
        assert native.invoke(float,replaced,table,key,'rate')==2.5
        assert len(seen)==1 and seen[0]==key
        assert tuple(map(type,seen[0]))==tuple(map(type,key))


def test_collision_many_keys_and_column_context_switch(native):
    keys=tuple(('α'+str(i%9),i,i%2==0) for i in range(1000))
    table=Table(keys,[(float(i),float(-i)) for i in range(len(keys))],('rate','other'))
    for i in range(0,len(keys),13):
        assert native.invoke(float,h.lookup,table,keys[i],'rate')==float(i)
        assert native.invoke(float,h.lookup,table,keys[i],'other')==float(-i)
    for i in range(100):
        other=Table(keys,[(float(i),)]*len(keys),('rate',))
        assert native.invoke(float,h.lookup,other,keys[i],'rate')==float(i)


def test_unicode_subclass_keeps_original_fallback_object(native):
    class Text(str):pass
    query=(Text('M'),42)
    table=Table((('M',42),),((1.5,),),('rate',))
    assert native.invoke(float,h.lookup,table,query,'rate')==1.5
    seen=[]
    def replacement(table,key,column):seen.append(key);return 2.5
    assert native.invoke(float,replacement,table,query,'rate')==2.5
    assert seen[0][0] is query[0]


def test_shared_context_concurrent_tables(native):
    from concurrent.futures import ThreadPoolExecutor
    def worker(number):
        table=Table((('M',number),),((float(number),),),('rate',))
        for _ in range(100):
            assert native.invoke(float,h.lookup,table,('M',number),'rate')==float(number)
        return number
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(worker,range(30)))==list(range(30))


@pytest.mark.parametrize('value', ['2.5', 'bad', None, 2**100, 2**10000])
def test_admitted_row_read_never_delegates_to_python_lookup(native,value):
    table=Table((('M',42),),((value,),),('rate',));calls=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is h.lookup.__code__:calls.append('lookup')
    sys.setprofile(profile)
    try:
        try:actual=native.invoke(float,h.lookup,table,('M',42),'rate')
        except (TypeError,ValueError,OverflowError):pass
    finally:sys.setprofile(None)
    assert calls==[]


def test_missing_singletons_keep_conversion_errors_and_pickle(native):
    import pandas as pd
    for value in [pd.NA,pd.NaT]:
        table=Table((('M',42),),((value,),),('rate',))
        assert pickle.loads(pickle.dumps(table))['values'][0][0] is value
        with pytest.raises(TypeError) as expected:float(h.lookup(table,('M',42),'rate'))
        with pytest.raises(TypeError) as actual:native.invoke(float,h.lookup,table,('M',42),'rate')
        assert actual.value.args==expected.value.args


@pytest.mark.parametrize('column', ['absent', 99, -99, 2**100])
@pytest.mark.parametrize('query', [('M',42),('M',43)])
def test_native_column_errors_follow_key_lookup_order(native,column,query):
    table=Table((('M',42),),((1.5,),),('rate',))
    for default in [(),('2.5',)]:
        try:expected=float(h.lookup(table,query,column,*default))
        except Exception as error:
            with pytest.raises(type(error)) as actual:native.invoke(float,h.lookup,table,query,column,*default)
            assert actual.value.args==error.args
        else:assert native.invoke(float,h.lookup,table,query,column,*default)==expected
