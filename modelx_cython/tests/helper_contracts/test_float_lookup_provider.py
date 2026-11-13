"""Differential checks against the actual packaged contextual provider."""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
from modelx_cython import runtime as native_runtime
from modelx_helpers import typed_inputs as h
from modelx_helpers.immutable_inputs import Table


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root=tmp_path_factory.mktemp('float-lookup')
    provider=Path(native_runtime.__file__).parent
    (root/'native.pyx').write_bytes((provider/'_native_inputs.pyx').read_bytes())
    (root/'native.pxd').write_bytes((provider/'_native_inputs.pxd').read_bytes())
    (root/'probe.pyx').write_text('''cimport native
cdef native.LookupContext _context = native.LookupContext()
def invoke(conversion, function, table, long long key, column, *default):
    cdef native.KeyPart part
    native.key_int(&part, key, False)
    if default:
        return native.lookup_keyed_float_f64(conversion, function, _context, table,
                                             &part, 1, False, column, default[0])
    return native.lookup_keyed_float_f64(conversion, function, _context, table,
                                         &part, 1, False, column)
''')
    (root/'setup.py').write_text("from setuptools import setup\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize(['native.pyx','probe.pyx'],compiler_directives={'language_level':3,'freethreading_compatible':True}))\n")
    run=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=root,capture_output=True,text=True)
    (root/'build.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
    sys.path.insert(0,str(root))
    try: yield importlib.import_module('probe').invoke
    finally:
        sys.path.remove(str(root));sys.modules.pop('probe',None);sys.modules.pop('native',None)


@pytest.mark.parametrize('value', [1.5,-0.0,2**53+1,'2.5','bad',None])
def test_value_conversion_matches_explicit_python_float(native,value):
    table=Table((0,),((value,),),('rate',))
    try: expected=float(h.lookup(table,0,'rate'))
    except Exception as e:
        with pytest.raises(type(e)) as actual: native(float,h.lookup,table,0,'rate')
        assert actual.value.args==e.args
    else:
        assert native(float,h.lookup,table,0,'rate')==expected


@pytest.mark.parametrize('default', ['2.5',None,9.0])
def test_missing_key_default_conversion(native,default):
    table=Table((0,),((1.5,),),('rate',))
    try: expected=float(h.lookup(table,1,'rate',default))
    except Exception as e:
        with pytest.raises(type(e)) as actual:native(float,h.lookup,table,1,'rate',default)
        assert actual.value.args==e.args
    else:assert native(float,h.lookup,table,1,'rate',default)==expected


def test_current_helper_and_conversion_order(native):
    table=Table((0,),((1.5,),),('rate',));events=[]
    def helper(*args):events.append('helper');return '2.5'
    def convert(value):events.append(('convert',value));return float(value)+1
    assert native(convert,helper,table,0,'rate')==3.5
    assert events==['helper',('convert','2.5')]
    assert native(float,helper,table,0,'rate')==2.5


def test_native_branch_never_calls_python_lookup(native):
    table=Table((0,),((1.5,),),('rate',));calls=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is h.lookup.__code__:calls.append('lookup')
    sys.setprofile(profile)
    try: assert native(float,h.lookup,table,0,'rate')==1.5
    finally:sys.setprofile(None)
    assert calls==[]


@pytest.mark.parametrize('keys,rows,key,column,default', [
    ((),(),0,'rate',()),
    ((),(),0,'bad',('2.5',)),
    ((0,),((1.5,),),0,'bad',()),
    ((0,),((1.5,),),1,'bad',('2.5',)),
    ((0,),((1.5,),),0,99,()),
    ((0,),((1.5,),),0,-1,()),
])
def test_empty_and_column_exception_order(native,keys,rows,key,column,default):
    table=Table(keys,rows,('rate',))
    try: expected=float(h.lookup(table,key,column,*default))
    except Exception as error:
        with pytest.raises(type(error)) as actual:
            native(float,h.lookup,table,key,column,*default)
        assert actual.value.args==error.args
    else:assert native(float,h.lookup,table,key,column,*default)==expected
