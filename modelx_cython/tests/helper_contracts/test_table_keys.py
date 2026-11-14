"""Live contract and primitive Cython producer for ordered owned-table keys."""
import importlib
from pathlib import Path
import pickle
import subprocess
import sys

from modelx_cython import runtime as native_runtime
import pytest
from modelx_helpers import pandas_inputs
from modelx_helpers.immutable_inputs import Record, Table
from modelx_helpers.table_keys import keys


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root=tmp_path_factory.mktemp('native-indices')
    provider=Path(native_runtime.__file__).parent
    for suffix in ('.pyx','.pxd'):
        (root/('indices'+suffix)).write_bytes((provider/('_native_indices'+suffix)).read_bytes())
    (root/'probe.pyx').write_text('''cimport indices

def raw(function, table):
    return indices.keys(function, (), table)

def keyword_raw(function, table):
    return indices.keys(function, ('table',), table=table)

''')
    (root/'setup.py').write_text("from pathlib import Path\nfrom setuptools import setup\nfrom Cython.Build import cythonize\nimport modelx_cython\nsetup(ext_modules=cythonize(['indices.pyx','probe.pyx'],include_path=[str(Path(modelx_cython.__file__).resolve().parent.parent)],compiler_directives={'language_level':3,'freethreading_compatible':True}))\n")
    run=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=root,capture_output=True,text=True)
    (root/'build.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
    sys.path.insert(0,str(root))
    try:yield importlib.import_module('probe')
    finally:
        sys.path.remove(str(root));sys.modules.pop('probe',None);sys.modules.pop('indices',None)


@pytest.mark.parametrize('values', [(9,-3,5), (), (-(1<<63),(1<<63)-1)])
def test_order_empty_width_bounds_and_pickle(native,values):
    table=Table(values,[(1.0,)]*len(values),('rate',))
    assert keys(table) is table['keys']
    assert native.raw(keys,table) is table['keys']
    assert native.raw(keys,pickle.loads(pickle.dumps(table)))==values


@pytest.mark.parametrize('values', [(True,), (1.0,), ('M',), (1<<63,)])
def test_other_domains_keep_original_sequence(native,values):
    table=Table(values,[(1.0,)],('rate',))
    assert keys(table)==values
    assert native.raw(keys,table)==values


@pytest.mark.parametrize('labels', [(3, 1, 3), (-(1<<63), 7, -(1<<63))])
def test_duplicate_index_uses_existing_record(native,labels):
    import pandas as pd
    public = pd.Index(labels, name='policy')
    original = public.copy(deep=True)
    owned = pandas_inputs.index(public)
    assert type(owned) is Record
    pd.testing.assert_index_equal(public, original, exact=True)
    for source in [owned, pickle.loads(pickle.dumps(owned))]:
        assert native.raw(keys, source) is source['keys']


def test_tuple_index_labels_preserve_order_duplicates_and_public_metadata(native):
    import pandas as pd
    public = pd.MultiIndex.from_tuples([('F', 42), ('M', 40), ('F', 42)], names=['sex', 'age'])
    original = public.copy(deep=True)
    owned = pandas_inputs.index(public)
    expected = keys(owned)
    assert expected == (('F', 42), ('M', 40), ('F', 42))
    assert native.raw(keys, owned) is expected
    pd.testing.assert_index_equal(public, original, exact=True)


def test_record_subclass_keeps_keys_override(native):
    class Changed(Record):
        def __getitem__(self,key):
            return ('changed',)
    source = Changed({'keys': (3, 1, 3)})
    object.__setattr__(source, '_metadata', ('index',))
    assert native.raw(keys, source) == keys(source) == ('changed',)


def test_keyword_keys_preserve_current_function_call_shape(native):
    import pandas as pd
    source = pandas_inputs.index(pd.Index([3, 1, 3]))
    assert native.keyword_raw(keys, source) is source['keys']
    calls = []
    def keyword_only(*, table):
        calls.append(table)
        return ('current',)
    assert native.keyword_raw(keyword_only, source) == ('current',)
    assert calls == [source]
    with pytest.raises(TypeError):
        native.raw(keyword_only, source)


def test_record_fields_do_not_confuse_real_keys_field_with_owned_index(native):
    source = Record({'keys': (3, 1, 3), 'premium': 4.0})
    for value in [source, pickle.loads(pickle.dumps(source))]:
        assert keys(value) == ('keys', 'premium')
        assert native.raw(keys, value) == ('keys', 'premium')
