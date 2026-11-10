"""Float producer ABI, tested directly without generator/compiler recognition."""
import importlib
from pathlib import Path
import pickle
import subprocess
import sys

from modelx_cython import runtime as native_runtime
import pandas as pd
import pytest
from modelx_helpers import pandas_inputs
from modelx_helpers.immutable_inputs import Table
from modelx_helpers.table_keys import keys


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp('float-key-span')
    provider = Path(native_runtime.__file__).parent
    for suffix in ['pyx', 'pxd']:
        (root / ('indices.' + suffix)).write_bytes((provider / ('_native_indices.' + suffix)).read_bytes())
    (root / 'span_probe.pyx').write_text('''cimport indices
def read(function, table):
    cdef indices.FloatKeySpan span = indices.keys_float_span(function, table)
    if span is None:
        return None
    return [indices.float_key_span_at(span, i) for i in range(indices.float_key_span_count(span))]
def at(function, table, Py_ssize_t position):
    cdef indices.FloatKeySpan span = indices.keys_float_span(function, table)
    if span is None:
        raise TypeError('not admitted')
    return indices.float_key_span_at(span, position)
def read_str(function, table):
    cdef indices.StrKeySpan span = indices.keys_str_span(function, table)
    if span is None:
        return None
    return [indices.str_key_span_at(span, i) for i in range(indices.str_key_span_count(span))]
def at_str(function, table, Py_ssize_t position):
    cdef indices.StrKeySpan span = indices.keys_str_span(function, table)
    if span is None:
        raise TypeError('not admitted')
    return indices.str_key_span_at(span, position)
''')
    (root / 'setup.py').write_text("from pathlib import Path\nfrom setuptools import setup\nfrom Cython.Build import cythonize\nimport modelx_cython\nsetup(ext_modules=cythonize(['indices.pyx','span_probe.pyx'],include_path=[str(Path(modelx_cython.__file__).resolve().parent.parent)],compiler_directives={'language_level':3,'freethreading_compatible':True}))\n")
    run = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=root,
                         capture_output=True, text=True)
    (root / 'build.log').write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    sys.path.insert(0, str(root))
    try:
        yield importlib.import_module('span_probe')
    finally:
        sys.path.remove(str(root))
        sys.modules.pop('span_probe', None)
        sys.modules.pop('indices', None)


@pytest.mark.parametrize('labels', [(), (35.5, 20.0, -0.0), (float(2**63), 1.5), (35.5, 20.5, 35.5)])
def test_order_empty_pickle_and_bounds(native, labels):
    table = pandas_inputs.index(pd.Index(labels, dtype=float))
    assert keys(table) == labels
    for source in [table, pickle.loads(pickle.dumps(table))]:
        assert native.read(keys, source) == list(labels)
        if labels:
            assert native.at(keys, source, -1) == labels[-1]
        for position in [len(labels), -len(labels)-1]:
            with pytest.raises(IndexError, match='tuple index out of range'):
                native.at(keys, source, position)


@pytest.mark.parametrize('labels', [(1, 2), (True, False), ('a', 'b'), (1.0, float('inf')), (float('nan'),)])
def test_admission_declines_without_changing_key_types(native, labels):
    table = Table(labels, [(1.0,)] * len(labels), ('rate',))
    assert native.read(keys, table) is None


def test_current_function_is_not_invoked_by_admission(native):
    events = []
    def replacement(table):
        events.append(table)
        return (7.5,)
    table = pandas_inputs.index(pd.Index([35.5, 20.5]))
    assert native.read(replacement, table) is None
    assert events == []


def test_public_index_ownership_and_errors():
    public = pd.Index(['equity', 'bond'], name='fund')
    table = pandas_inputs.index(public)
    assert public.name == 'fund' and type(public) is pd.Index
    assert keys(table) == ('equity', 'bond')
    assert table.lookup('bond', 'position') == 1
    assert keys(pandas_inputs.index(pd.Index(['fund', 'fund']))) == ('fund', 'fund')
    with pytest.raises(TypeError, match='pandas Index'):
        pandas_inputs.index(['fund'])


@pytest.mark.parametrize('labels', [(), ('equity', 'bond'), ('死亡', '基金', ''), ('基金', 'equity', '基金')])
def test_text_span_order_identity_empty_pickle_and_bounds(native, labels):
    table = pandas_inputs.index(pd.Index(labels, dtype=object))
    for source in [table, pickle.loads(pickle.dumps(table))]:
        assert native.read_str(keys, source) == list(labels)
        if labels:
            assert native.at_str(keys, source, -1) is keys(source)[-1]
        for position in [len(labels), -len(labels)-1]:
            with pytest.raises(IndexError, match='tuple index out of range'):
                native.at_str(keys, source, position)


def test_spans_have_no_python_helper_calls_on_cold_and_hot_reads(native):
    from modelx_helpers.immutable_inputs import Record, freeze
    text = pandas_inputs.index(pd.Index(['equity', 'bond', 'equity']))
    floating = pandas_inputs.index(pd.Index([35.5, -0.0, 35.5]))
    codes = {keys.__code__, Table.__init__.__code__, Table.__getitem__.__code__,
             Record.__getitem__.__code__, freeze.__code__}
    events = []
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            events.append(frame.f_code.co_name)
    sys.setprofile(profile)
    try:
        for _ in range(2):
            assert native.read(keys, floating) == [35.5, -0.0, 35.5]
            assert native.read_str(keys, text) == ['equity', 'bond', 'equity']
            assert native.read_str(keys, floating) is None
            assert native.read(keys, text) is None
    finally:
        sys.setprofile(None)
    assert events == []
