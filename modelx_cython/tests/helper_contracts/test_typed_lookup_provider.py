"""Plain Cython cimports: no modelx/formula matcher is used for helper calls."""
import importlib
from pathlib import Path
import pickle
import subprocess
import sys
import pytest
from modelx_cython import runtime as native_runtime
from modelx_helpers import typed_inputs as h
from modelx_helpers.immutable_inputs import Table,Record,freeze

@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root=tmp_path_factory.mktemp('explicit');provider=Path(native_runtime.__file__).parent
    for ext in ['pyx','pxd']:(root/('native.'+ext)).write_bytes((provider/('_native_inputs.'+ext)).read_bytes())
    (root/'probe.pyx').write_text('''cimport native

def integer(function,source,key,*args):
    cdef long long value
    if not args:value=native.lookup_int(function,(),source,key)
    elif len(args)==1:value=native.lookup_int(function,(),source,key,args[0])
    else:value=native.lookup_int(function,(),source,key,args[0],args[1])
    return value

def floating(function,source,key,*args):
    cdef double value
    if not args:value=native.lookup_float(function,(),source,key)
    elif len(args)==1:value=native.lookup_float(function,(),source,key,args[0])
    else:value=native.lookup_float(function,(),source,key,args[0],args[1])
    return value

def table_integer(source, long long key, column, default):
    cdef native.NativeTable table = native.NativeTable(source)
    return table.lookup_float(key, column, default)

def table_boolean(source, bint key, column):
    cdef native.NativeTable table = native.NativeTable(source)
    return table.lookup_float(key, column)

def table_object(source, object key, column):
    cdef native.NativeTable table = native.NativeTable(source)
    return table.lookup_float(key, column)

def table_keys(source, key, first, last, contains):
    cdef native.NativeTable table = native.NativeTable(source)
    return (native.first_key(first, table), native.last_key(last, table),
            native.contains(contains, table, key))

def clamped(source, namespace, long long key, column, bint lower):
    cdef native.NativeTable table = native.NativeTable(source)
    cdef double result
    if native.try_clamped_table_float(table, native.LookupContext(), lower, namespace, key, column, &result):
        return result
    return None

def record_values(function, sources, key):
    cdef native.LookupContext context = native.LookupContext()
    return [native.record_field_float(function, context, source, key) for source in sources]

def repeated_table_values(source, selections):
    cdef native.LookupContext context = native.LookupContext()
    cdef native.NativeTable table = native.NativeTable(source)
    context._mx_enable_private_lookup()
    return [native.read_table_field_float(context, table, key, column)
            for key, column in selections]

def boolean(function,source,key,*args):
    cdef bint value
    if not args:value=native.lookup_bool(function,(),source,key)
    elif len(args)==1:value=native.lookup_bool(function,(),source,key,args[0])
    else:value=native.lookup_bool(function,(),source,key,args[0],args[1])
    return value

def text(function,source,key,*args):
    cdef str value
    if not args:value=native.lookup_str(function,(),source,key)
    elif len(args)==1:value=native.lookup_str(function,(),source,key,args[0])
    else:value=native.lookup_str(function,(),source,key,args[0],args[1])
    return value

def raw(function,source,key,*args):
    if not args:return native.lookup_object(function,(),source,key)
    elif len(args)==1:return native.lookup_object(function,(),source,key,args[0])
    return native.lookup_object(function,(),source,key,args[0],args[1])

def missing(function,predicate,value):
    return native.is_missing(function,(),predicate,value)

def keyword_read(function,source,key,default):
    return native.lookup_float(function,('default',),source,key,
                               column=native.omitted_argument,default=default)

def keyword_integer(function,source,key,default):
    return native.lookup_int(function,('default',),source,key,
                             column=native.omitted_argument,default=default)

def keyword_object(function,source,key,default):
    return native.lookup_object(function,('default',),source,key,
                                column=native.omitted_argument,default=default)

def reordered_read(function,source,key,default):
    return native.lookup_float(function,('default','key','source'),
                               default=default,key=key,source=source,
                               column=native.omitted_argument)

def keyword_missing(function,predicate,value):
    return native.is_missing(function,('value','predicate'),value=value,predicate=predicate)

def shared_sum_state(values):
    cdef native.F64Sum state
    cdef double value
    native.f64_init(&state)
    for value in values:
        native.f64_add(&state,value)
    return native.f64_result(&state)
''')
    (root/'setup.py').write_text("from pathlib import Path\nfrom setuptools import setup\nfrom Cython.Build import cythonize\nimport modelx_cython\nsetup(ext_modules=cythonize(['native.pyx','probe.pyx'],include_path=[str(Path(modelx_cython.__file__).resolve().parent.parent)],compiler_directives={'language_level':3,'freethreading_compatible':True}))\n")
    r=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=root,capture_output=True,text=True)
    (root/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
    sys.path.insert(0,str(root))
    try:yield importlib.import_module('probe')
    finally:sys.path.remove(str(root));sys.modules.pop('probe',None);sys.modules.pop('native',None)


def test_native_table_method_preserves_integer_default_and_object_keys(native):
    table = Table((1, 2), ((0.25,), (0.5,)), ('rate',))
    assert native.table_integer(table, 2, 'rate', '1.25') == 0.5
    assert native.table_integer(table, 9, 'rate', '1.25') == 1.25
    wide = 2**90
    wide_table = Table((wide, ('M', 42)), ((0.75,), (0.125,)), ('rate',))
    assert native.table_object(wide_table, wide, 'rate') == 0.75
    assert native.table_object(wide_table, ('M', 42), 'rate') == 0.125


def test_native_table_method_preserves_column_and_float_default(native):
    source = Table((2,), ((0.25, 0.5, 0.75),), ('left', 'middle', 'right'))
    assert native.table_integer(source, 2, 'left', -1.0) == 0.25
    assert native.table_integer(source, 2, 'middle', -1.0) == 0.5
    assert native.table_integer(source, 2, ''.join(('ri', 'ght')), -1.0) == 0.75
    assert native.table_integer(source, 2, -1, -1.0) == 0.75
    assert native.table_integer(source, 3, 'right', -1.0) == -1.0


def test_native_table_method_keeps_boolean_index_identity(native):
    import pandas as pd
    from modelx_helpers.pandas_inputs import table

    source = table(pd.DataFrame({'rate': [0.25]}, index=[1]), indexing='at')
    with pytest.raises(KeyError):
        native.table_boolean(source, True, 'rate')


def test_native_table_key_operations_use_the_owner(native):
    import pandas as pd
    from modelx_helpers.pandas_inputs import table

    source = table(pd.DataFrame({'rate': [0.5, 0.25]}, index=[2, 1]))
    operations = (h.first_key, h.last_key, h.contains)
    codes = {operation.__code__ for operation in operations}
    calls = []

    def profile(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            calls.append(frame.f_code.co_name)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        assert native.table_keys(source, 1, *operations) == (1, 2, True)
        assert native.table_keys(source, True, *operations) == (1, 2, False)
    finally:
        sys.setprofile(previous)
    assert not calls
    receiver_types = native.table_keys(source, 1, type, type, lambda value, key: type(value))
    assert all(receiver_type.__name__ == 'NativeTable' for receiver_type in receiver_types)


def test_clamped_table_matches_ordered_index_and_defers_custom_helpers(native, monkeypatch):
    import pandas as pd
    from modelx_helpers.pandas_inputs import table

    source = table(pd.DataFrame({'rate': [0.5, 0.25, 0.75],
                                 'note': ['b', 'a', 'c']}, index=[2, 1, 3]))
    assert native.clamped(source, h, 0, 'rate', True) == 0.25
    assert native.clamped(source, h, 0, 'rate', False) is None
    assert native.clamped(source, h, 4, 'rate', False) == 0.75
    assert native.clamped(source, h, 2, 'rate', False) == 0.5
    assert native.clamped(source, h, 2, 'missing', False) is None
    monkeypatch.setattr(h, 'last_key', lambda table: 2)
    assert native.clamped(source, h, 4, 'rate', True) is None


def test_immutable_record_float_binding_observes_source_changes_and_custom_values(native):
    import numpy as np

    first = Record({'x': np.float64(1.25)}, preserve_scalar_types=True)
    second = Record({'x': np.float64(2.5)}, preserve_scalar_types=True)
    assert native.record_values(h.lookup_float, [first, first, second, first], 'x') == [
        1.25, 1.25, 2.5, 1.25]
    assert native.record_values(lambda source, key: 7.0, [first, first], 'x') == [7.0, 7.0]

    class ChangingFloat:
        calls = 0

        def __float__(self):
            self.calls += 1
            return float(self.calls)

    value = ChangingFloat()
    record = Record({'x': value}, preserve_scalar_types=True)
    assert native.record_values(h.lookup_float, [record, record], 'x') == [1.0, 2.0]


def test_single_row_float_binding_keeps_each_column_separate(native):
    import pandas as pd
    from modelx_helpers.pandas_inputs import table

    source = table(pd.DataFrame({'initial': [18.0], 'years': [15], 'note': ['source']},
                                index=['schedule']))
    fields = [('schedule', 'initial'), ('schedule', 'years')] * 3
    assert native.repeated_table_values(source, fields) == [18.0, 15.0] * 3
    with pytest.raises(KeyError):
        native.repeated_table_values(source, [*fields, ('other', 'years')])


@pytest.mark.parametrize('suffix,entry', [('int','integer'),('float','floating'),('bool','boolean'),('str','text')])
@pytest.mark.parametrize('value',[True,0,2**53+1,2.5,'2.5','bad',None,float('nan'),2**100])
def test_python_and_direct_native_contract(native,suffix,entry,value):
    function=getattr(h,'lookup_'+suffix);call=getattr(native,entry)
    for source,key,args in [(Record({'rate':value}),'rate',()),(Table((('M',42),),((value,),),('rate',)),('M',42),('rate',))]:
        try:expected=function(source,key,*args)
        except Exception as error:
            with pytest.raises(type(error)) as actual:call(function,source,key,*args)
            assert actual.value.args==error.args
        else:
            actual=call(function,source,key,*args)
            assert type(actual)is type(expected)
            assert actual==expected or isinstance(actual,float) and actual!=actual and expected!=expected


def test_direct_calls_use_no_python_table_or_helper_functions(native):
    source=Table((('M',42),),((1.5,3,'Y',True),),('rate','age','label','flag'))
    record=Record({'age':3,'rate':1.5,'flag':True,'label':'Y'})
    codes={getattr(h,'lookup_'+s).__code__ for s in ['int','float','bool','str']}
    codes.update([h.lookup.__code__,h._typed_value.__code__,Table.__getitem__.__code__,Table.__init__.__code__,Record.__getitem__.__code__,freeze.__code__])
    calls=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in codes:calls.append(frame.f_code.co_name)
    sys.setprofile(profile)
    try:
        assert native.floating(h.lookup_float,source,('M',42),'rate')==1.5
        assert native.integer(h.lookup_int,source,('M',42),'age')==3
        assert native.text(h.lookup_str,source,('M',42),'label')=='Y'
        assert native.boolean(h.lookup_bool,source,('M',42),'flag') is True
        assert native.integer(h.lookup_int,record,'age')==3
        assert native.floating(h.lookup_float,record,'rate')==1.5
        assert native.boolean(h.lookup_bool,record,'flag') is True
        assert native.text(h.lookup_str,record,'label')=='Y'
    finally:sys.setprofile(None)
    assert calls==[]


def test_defaults_replacement_pickle_and_at_error(native):
    import pandas as pd
    from modelx_helpers.pandas_inputs import table
    record=Record({'age':3})
    assert native.keyword_integer(h.lookup_int,record,'absent',7)==7
    source=Table((),(),('rate',))
    assert native.floating(h.lookup_float,source,0,'rate','2.5')==2.5
    seen=[]
    def replacement(*args):seen.append(args);return 3.5
    assert native.floating(replacement,source,0,'rate')==3.5
    assert seen==[(source,0,'rate')]
    frame=pd.DataFrame({'rate':[1.5,2.5]},index=pd.MultiIndex.from_tuples([('M',42),('F',43)]))
    owned=table(frame,indexing='at')
    assert native.floating(h.lookup_float,pickle.loads(pickle.dumps(owned)),('M',42),'rate')==1.5
    with pytest.raises(KeyError) as error:native.floating(h.lookup_float,owned,('M',43),'rate')
    assert error.value.args == (('M', 43),)


def test_keyword_default_and_replacement_preserve_original_call_layout(native):
    record = Record({'value': 1.5})
    assert native.keyword_read(h.lookup_float, record, 'absent', 2.5) == 2.5
    assert native.reordered_read(h.lookup_float, record, 'absent', 3.5) == 3.5
    seen = []
    def replacement(*args, **kwargs):
        seen.append((args, tuple(kwargs.items())))
        return 4.5
    assert native.keyword_read(replacement, record, 'absent', None) == 4.5
    assert seen.pop() == ((record, 'absent'), (('default', None),))
    assert native.reordered_read(replacement, record, 'absent', 7.5) == 4.5
    assert seen.pop() == ((), (('default', 7.5), ('key', 'absent'), ('source', record)))
    assert native.floating(replacement, record, 'absent', 8.5) == 4.5
    assert seen.pop() == ((record, 'absent', 8.5), ())
    def keyword_only(source, key, *, default):
        return default
    assert native.keyword_read(keyword_only, record, 'absent', 9.5) == 9.5
    error = RuntimeError('same replacement exception')
    def fail(*args, **kwargs):
        raise error
    with pytest.raises(RuntimeError) as actual:
        native.keyword_read(fail, record, 'absent', 0.0)
    assert actual.value is error


def test_keyword_missing_replacement_keeps_insertion_order(native):
    import pandas as pd
    assert native.keyword_missing(h.is_missing, pd.isna, None) is True
    seen = []
    def replacement(**kwargs):
        seen.append(tuple(kwargs.items()))
        return False
    assert native.keyword_missing(replacement, pd.isna, None) is False
    assert seen == [(('value', None), ('predicate', pd.isna))]


@pytest.mark.parametrize('suffix,entry', [('int','integer'), ('float','floating'),
                                         ('bool','boolean'), ('str','text'), ('object','raw')])
def test_record_column_is_not_a_default(native,suffix,entry):
    record = Record({'value': 1.5})
    function = getattr(h, 'lookup_' + suffix)
    message = 'Record lookup does not accept a column; use default='
    for key in ['value', 'absent']:
        with pytest.raises(TypeError) as python_error:
            function(record, key, column=7)
        assert python_error.value.args == (message,)
        with pytest.raises(TypeError) as native_error:
            getattr(native, entry)(function, record, key, 7)
        assert native_error.value.args == python_error.value.args


def test_explicit_public_default_sentinel_retains_omission_semantics(native):
    record = Record({'value': 1.5})
    assert native.keyword_read(h.lookup_float, record, 'value', h._MISSING) == 1.5
    with pytest.raises(KeyError):
        native.keyword_read(h.lookup_float, record, 'absent', h._MISSING)


@pytest.mark.parametrize('values', [[], [-0.0], [1e16, 1.0, -1e16],
                                    [0.1, 0.2, -0.3], [float('inf'), 1.0],
                                    [float('inf'), -float('inf')], [float('nan')]])
def test_shared_float_state_reexport_preserves_float_start_sum(native,values):
    import math
    expected = sum(values, 0.0)
    actual = native.shared_sum_state(values)
    assert actual == expected or math.isnan(actual) and math.isnan(expected)


@pytest.mark.parametrize('keys,query',[
    ((1,2),1.0), ((1.0,2.0),1), ((True,False),1.0),
    ((-0.0,1.0),0), ((0,1),-0.0),
    ((1.5,2.5),1.5), ((1.5,2.5),1),
    ((2**53,2**53+1),float(2**53+1)),
    ((float(2**53),float(2**53+2)),2**53+1),
    ((float(2**63),-1.5),2**63-1),
    ((("M",42.5),("F",43.5)),("M",42.5)),
])
def test_float_keys_preserve_exact_numeric_equality(native,keys,query):
    table=Table(keys,[(float(i+1),) for i in range(len(keys))],('rate',))
    try:expected=h.lookup_float(table,query,'rate')
    except Exception as error:
        with pytest.raises(type(error)) as actual:native.floating(h.lookup_float,table,query,'rate')
        assert actual.value.args==error.args
    else:assert native.floating(h.lookup_float,table,query,'rate')==expected


def test_float_keys_are_native_not_a_python_helper_fallback(native):
    table=Table((35.5,40.5),((1.5,),(2.5,)),('rate',));calls=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in {h.lookup_float.__code__,h.lookup.__code__,Table.lookup.__code__}:
            calls.append(frame.f_code.co_name)
    sys.setprofile(profile)
    try:assert native.floating(h.lookup_float,table,35.5,'rate')==1.5
    finally:sys.setprofile(None)
    assert not calls


def test_object_lookup_keeps_missing_identity(native):
    import pandas as pd
    for value in [pd.NA,pd.NaT,None]:
        record=Record({'value':value})
        assert native.raw(h.lookup_object,record,'value') is value
        assert h.lookup_object(record,'value') is value
    assert native.keyword_object(h.lookup_object,Record({}),'absent',None) is None


@pytest.mark.parametrize('dtype,value', [('int64', 42), ('float64', 0.25), ('bool', True)])
def test_explicit_native_record_keeps_public_numpy_scalar_types(native,dtype,value):
    import pandas as pd
    from modelx_helpers import pandas_inputs
    public = pd.Series([value], index=['value'], dtype=dtype)
    record = pandas_inputs.record(public, preserve_scalar_types=True)
    raw = native.raw(h.lookup_object, record, 'value')
    assert type(raw) is type(public['value']) and raw == public['value']
    assert native.integer(h.lookup_int, record, 'value') == int(public['value'])
    assert native.floating(h.lookup_float, record, 'value') == float(public['value'])


@pytest.mark.parametrize('indexing', ['loc', 'at'])
@pytest.mark.parametrize('kind', ['int64', 'uint64', 'float32', 'float64', 'bool_'])
def test_numpy_query_keys_use_native_admission_and_original_error_scalars(native,indexing,kind):
    import numpy as np
    import pandas as pd
    from modelx_helpers import pandas_inputs
    scalar = getattr(np, kind)
    labels = [True, False] if kind == 'bool_' else [1, 2]
    frame = pd.DataFrame({'rate': [1.5, 2.5]}, index=pd.Index(labels))
    prepared = pandas_inputs.table(frame, indexing=indexing)
    for query in [scalar(labels[0]), scalar(5)]:
        try: expected = float(getattr(frame, indexing)[query, 'rate'])
        except Exception as error:
            for call in [h.lookup_float, lambda *a: native.floating(h.lookup_float, *a)]:
                with pytest.raises(type(error)) as actual: call(prepared, query, 'rate')
                assert actual.value.args == error.args
                assert [type(x) for x in actual.value.args] == [type(x) for x in error.args]
        else:
            calls = []
            def profile(frame,event,arg):
                if event == 'call' and frame.f_code in {h.lookup_float.__code__, h.lookup.__code__,
                                                       h._typed_value.__code__, Table.lookup.__code__}:
                    calls.append(frame.f_code.co_name)
            sys.setprofile(profile)
            try: assert native.floating(h.lookup_float, prepared, query, 'rate') == expected
            finally: sys.setprofile(None)
            assert calls == []


@pytest.mark.parametrize('value', [None, True, False, 0, 2**100, 'missing', b'',
                                 -0.0, float('nan'), float('inf')])
def test_scalar_missing_predicate_matches_pandas_without_python(native,value):
    import pandas as pd
    values = [value, pd.NA, pd.NaT]
    forbidden = {h.is_missing.__code__, pd.isna.__code__}
    calls = []
    def profile(frame,event,arg):
        if event == 'call' and frame.f_code in forbidden:
            calls.append(frame.f_code.co_name)
    expected = [h.is_missing(pd.isna, item) for item in values]
    sys.setprofile(profile)
    try:
        assert [native.missing(h.is_missing, pd.isna, item) for item in values] == expected
    finally:
        sys.setprofile(None)
    assert calls == []


def test_missing_predicate_current_callable_and_numpy_scalars(native):
    import numpy as np
    import pandas as pd
    for value in [np.int64(42), np.uint64(2**64-1), np.bool_(False),
                  np.float32('nan'), np.float64('nan'), np.float64(0.5)]:
        assert native.missing(h.is_missing,pd.isna,value) is bool(pd.isna(value))
    events = []
    sentinel = RuntimeError('predicate')
    def predicate(value):
        events.append(value)
        raise sentinel
    with pytest.raises(RuntimeError) as error:
        native.missing(h.is_missing,predicate,3)
    assert error.value is sentinel and events == [3]
    def replacement(predicate,value):
        events.append(('helper',value))
        return False
    assert native.missing(replacement,pd.isna,4) is False
    assert events[-1] == ('helper',4)


@pytest.mark.parametrize('replace_alias', [False, True])
def test_predicate_replaced_before_provider_import_is_not_admitted(native, replace_alias):
    source = f'''import pandas as pd
def replacement(value):
    return False
pd.isna = replacement
if {replace_alias}:
    pd.isnull = replacement
from modelx_helpers import typed_inputs as h
import probe
assert probe.missing(h.is_missing, pd.isna, None) is False
'''
    run = subprocess.run([sys.executable, '-c', source], cwd=Path(native.__file__).parent,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize('suffix,entry', [('int','integer'), ('float','floating'),
                                       ('bool','boolean'), ('str','text'), ('object','raw')])
def test_record_subclasses_keep_the_python_compatibility_contract(native,suffix,entry):
    calls = []
    class DerivedRecord(Record):
        def __getitem__(self,key):
            calls.append(key)
            return super().__getitem__(key)
    source = DerivedRecord({'value': 2})
    function = getattr(h, 'lookup_' + suffix)
    expected = function(source, 'value')
    calls.clear()
    actual = getattr(native, entry)(function, source, 'value')
    assert type(actual) is type(expected) and actual == expected
    assert calls == ['value']  # compatibility is preserved, not native admission


@pytest.mark.parametrize('indexing', ['loc', 'at'])
@pytest.mark.parametrize('keys,queries', [
    ([True, False], [True, False, 1, 0, 1.0]),
    ([1, 0], [True, False, 1, 0, 1.0]),
    ([('M', True), ('F', False)], [('M', True), ('M', 1), ('M', False)]),
    ([('M', 42), ('F', 43)], [('M', 42), ('M', 43), ('M', True)]),
])
def test_indexer_metadata_preserves_boolean_keys_and_sparse_errors(native,indexing,keys,queries):
    import numpy as np
    import pandas as pd
    from modelx_helpers import pandas_inputs
    index = pd.MultiIndex.from_tuples(keys) if isinstance(keys[0], tuple) else pd.Index(keys)
    frame = pd.DataFrame({'rate': [1.5, 2.5]}, index=index)
    for source in [frame, frame.iloc[:0]]:
        prepared = pandas_inputs.table(source, indexing=indexing)
        for table in [prepared, pickle.loads(pickle.dumps(prepared))]:
            for query in queries:
                try: expected = float(getattr(source, indexing)[query, 'rate'])
                except Exception as error:
                    args = error.args
                    if (indexing == 'at' and isinstance(index, pd.MultiIndex)
                            and isinstance(error, KeyError) and len(args) == 1
                            and isinstance(args[0], (int, np.integer))):
                        args = (query,)
                    for call in [h.lookup_float, lambda *a: native.floating(h.lookup_float, *a)]:
                        with pytest.raises(type(error)) as actual: call(table, query, 'rate')
                        assert actual.value.args == args
                else:
                    assert h.lookup_float(table, query, 'rate') == expected
                    assert native.floating(h.lookup_float, table, query, 'rate') == expected
