"""Wide-key semantics, native execution, ownership and consumer contracts.

This uses the same copied-provider fixture pattern as test_typed_lookup_provider.
No compiler formula transformation, regional name or policy population is used.
"""
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib
import math
from pathlib import Path
import pickle
import subprocess
import sys
import sysconfig

import pytest
from modelx_cython import runtime as native_runtime
from modelx_helpers import pandas_inputs as inputs
from modelx_helpers import table_operations as operations
from modelx_helpers import typed_inputs as h
from modelx_helpers.immutable_inputs import Record, Table, freeze


@pytest.fixture(scope='module')
def native_wide(tmp_path_factory):
    root = tmp_path_factory.mktemp('wide-key-provider')
    provider = Path(native_runtime.__file__).parent
    for suffix in ('pyx', 'pxd'):
        (root / ('wide_native.' + suffix)).write_bytes(
            (provider / ('_native_inputs.' + suffix)).read_bytes())
    (root / 'wide_probe.pyx').write_text('''# cython: freethreading_compatible=True
cimport wide_native as native

def read(kind, function, source, key, column, *default):
    if kind == 'float':
        if default: return native.lookup_float(function, (), source, key, column, default[0])
        return native.lookup_float(function, (), source, key, column)
    if kind == 'int':
        if default: return native.lookup_int(function, (), source, key, column, default[0])
        return native.lookup_int(function, (), source, key, column)
    if kind == 'bool':
        if default: return native.lookup_bool(function, (), source, key, column, default[0])
        return native.lookup_bool(function, (), source, key, column)
    if default: return native.lookup_object(function, (), source, key, column, default[0])
    return native.lookup_object(function, (), source, key, column)

def contains(source, key):
    return native.indexed_contains(source, key)

def position(source, key):
    return native.indexed_position(source, key)

def row(function, source, key):
    return native.row(function, (), source, key)

def consume(kind, function, values):
    if kind == 'any': return native.any_true(function, values)
    if kind == 'all': return native.all_true(function, values)
    return native.max_value(function, values)
''', encoding='utf-8')
    (root / 'setup.py').write_text('''from pathlib import Path
from setuptools import setup
from Cython.Build import cythonize
import modelx_cython
setup(ext_modules=cythonize(['wide_native.pyx', 'wide_probe.pyx'],
    include_path=[str(Path(modelx_cython.__file__).resolve().parent.parent)],
    compiler_directives={'language_level': 3, 'freethreading_compatible': True}))
''', encoding='utf-8')
    result = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                            cwd=root, capture_output=True, text=True)
    (root / 'build.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module('wide_probe')
        if sysconfig.get_config_var('Py_GIL_DISABLED'):
            assert not sys._is_gil_enabled()
        yield module
    finally:
        sys.path.remove(str(root))
        sys.modules.pop('wide_probe', None)
        sys.modules.pop('wide_native', None)
        if sysconfig.get_config_var('Py_GIL_DISABLED'):
            assert not sys._is_gil_enabled()


def key_at(width, row):
    """Two distinct mixed primitive rows; every component differs across rows."""
    choices = (lambda i: ('α' if row == 0 else 'β') + str(i),
               lambda i: i + row * 100,
               lambda i: bool(row),
               lambda i: i + 0.25 + row * 100,
               lambda i: ('😀' if row == 0 else 'λ') + str(i))
    return tuple(choices[i % len(choices)](i) for i in range(width))


def framed(width, mode='loc'):
    import pandas as pd
    index = pd.MultiIndex.from_tuples([key_at(width, 0), key_at(width, 1)],
                                     names=['level_' + str(i) for i in range(width)])
    frame = pd.DataFrame({'rate': [1.5, 2.5], 'count': [3, 4],
                          'flag': [False, True], 'name': ['A', 'B']}, index=index)
    return frame, inputs.table(frame, indexing=mode, rows=True,
                               preserve_scalar_types=True)


def same_outcome(reference, invoke):
    try:
        expected = reference()
    except Exception as error:
        with pytest.raises(type(error)) as actual:
            invoke()
        assert actual.value.args == error.args
    else:
        actual = invoke()
        assert type(actual) is type(expected)
        assert actual == expected or isinstance(actual, float) and math.isnan(actual) and math.isnan(expected)


class PythonCalls:
    """Observe only prohibited reference helpers, not Python protocol callbacks."""
    def __init__(self, *extra):
        self.codes = {h.lookup.__code__, h._typed_value.__code__,
                      Table.lookup.__code__, Table.contains.__code__,
                      Table.__getitem__.__code__, Record.__getitem__.__code__,
                      freeze.__code__, operations.row.__code__,
                      *(getattr(h, 'lookup_' + kind).__code__
                        for kind in ('object', 'float', 'int', 'bool')),
                      *extra}
        self.calls = []
    def hook(self, frame, event, arg):
        if event == 'call' and frame.f_code in self.codes:
            self.calls.append(frame.f_code.co_name)
    def __enter__(self):
        self.previous = sys.getprofile()
        assert self.previous is None, 'Do not silently replace an active profiler'
        sys.setprofile(self.hook)
        return self
    def __exit__(self, *args):
        sys.setprofile(self.previous)


@pytest.mark.parametrize('width', [5, 8, 17])
@pytest.mark.parametrize('mode', ['loc', 'at'])
def test_reference_domain_oracles(width, mode):
    """Runnable before implementation: confirms wide-key oracle domains exist."""
    frame, source = framed(width, mode)
    for row, query in enumerate(frame.index):
        assert h.lookup_float(source, query, 'rate') == float(getattr(frame, mode)[query, 'rate'])
        assert h.lookup_int(source, query, 'count') == int(getattr(frame, mode)[query, 'count'])
        assert h.lookup_bool(source, query, 'flag') is bool(getattr(frame, mode)[query, 'flag'])
        assert h.contains(source, query)
        assert operations.to_pandas(operations.row(source, query)).equals(frame.loc[query])
    absent = (*key_at(width, 0)[:-1], key_at(width, 1)[-1])
    same_outcome(lambda: float(getattr(frame, mode)[absent, 'rate']),
                 lambda: h.lookup_float(source, absent, 'rate'))
    same_outcome(lambda: float(getattr(frame, mode)[absent, 'missing']),
                 lambda: h.lookup_float(source, absent, 'missing'))


@pytest.mark.parametrize('width', [5, 8, 17])
@pytest.mark.parametrize('mode', ['loc', 'at'])
def test_wide_success_uses_native_lookup_position_contains_and_rows(native_wide, width, mode):
    import pandas as pd
    frame, source = framed(width, mode)
    for restored in (source, pickle.loads(pickle.dumps(source))):
        for position, query in enumerate(frame.index):
            with PythonCalls() as observed:
                actual = native_wide.read('object', h.lookup_object, restored, query, 'name')
                assert actual == ('A', 'B')[position]
                assert native_wide.read('float', h.lookup_float, restored, query, 'rate') == (1.5, 2.5)[position]
                assert native_wide.read('int', h.lookup_int, restored, query, 'count') == (3, 4)[position]
                assert native_wide.read('bool', h.lookup_bool, restored, query, 'flag') is bool(position)
                assert native_wide.contains(restored, query) is True
                assert native_wide.position(restored, query) == position
                actual_row = native_wide.row(operations.row, restored, query)
            assert observed.calls == []
            pd.testing.assert_series_equal(operations.to_pandas(actual_row), frame.loc[query])


@pytest.mark.parametrize('width', [5, 17])
@pytest.mark.parametrize('mode', ['loc', 'at'])
def test_wide_error_encoding_column_priority_and_defaults(native_wide, width, mode):
    frame, source = framed(width, mode)
    existing = key_at(width, 0)
    absent = (*existing[:-1], key_at(width, 1)[-1])  # Every level exists; full row does not.
    unknown = (*existing[:-1], 'not-a-level-label')
    for key in (absent, unknown):
        same_outcome(lambda: h.lookup_float(source, key, 'rate'),
                     lambda: native_wide.read('float', h.lookup_float, source, key, 'rate'))
        same_outcome(lambda: h.lookup_object(source, key, 'missing'),
                     lambda: native_wide.read('object', h.lookup_object, source, key, 'missing'))
        assert native_wide.read('float', h.lookup_float, source, key, 'rate', '7.5') == 7.5
        assert native_wide.read('object', h.lookup_object, source, key, 'rate', None) is None
        assert native_wide.contains(source, key) is False
    with PythonCalls() as observed:
        same_outcome(lambda: (_ for _ in ()).throw(KeyError('missing')),
                     lambda: native_wide.read('float', h.lookup_float, source, absent, 'missing'))
    assert observed.calls == []


@pytest.mark.parametrize('component', [2**100, float('nan')])
def test_wide_unsupported_components_retain_reference_fallback(native_wide, component):
    key = ('A', 'B', 3, True, component)
    source = Table((key,), ((1.5,),), ('rate',))
    expected = h.lookup_float(source, key, 'rate')
    with PythonCalls() as observed:
        assert native_wide.read('float', h.lookup_float, source, key, 'rate') == expected
    # Unsupported inputs must keep Python semantics; these are not native proofs.
    assert observed.calls


def test_wide_unicode_subclass_and_replaced_reference_preserve_objects(native_wide):
    class Text(str):
        pass
    query = (Text('A'), 'B', 3, True, 2.5)
    source = Table((('A', 'B', 3, True, 2.5),), ((1.5,),), ('rate',))
    with PythonCalls() as observed:
        assert native_wide.read('float', h.lookup_float, source, query, 'rate') == 1.5
    assert observed.calls
    marker = object()
    calls = []
    def replacement(*args):
        calls.append(args)
        return marker
    assert native_wide.read('object', replacement, source, query, 'rate') is marker
    assert calls[0][1] is query and calls[0][1][0] is query[0]


def test_wide_selected_value_conversion_is_deferred(native_wide):
    error = ValueError('selected conversion')
    events = []
    class Selected:
        def __float__(self):
            events.append('convert')
            raise error
    keys = (key_at(8, 0), key_at(8, 1))
    source = Table(keys, ((1.5,), (Selected(),)), ('rate',), preserve_scalar_types=True)
    assert native_wide.read('float', h.lookup_float, source, keys[0], 'rate') == 1.5
    assert events == []
    with pytest.raises(ValueError) as actual:
        native_wide.read('float', h.lookup_float, source, keys[1], 'rate')
    assert actual.value is error and events == ['convert']


def test_wide_temporary_unicode_keys_and_shared_context_threads(native_wide):
    keys = tuple(tuple('part-%d-row-%d-λ' % (part, row) for part in range(8)) for row in range(32))
    source = Table(keys, tuple((float(i), float(-i)) for i in range(len(keys))), ('rate', 'other'))
    def worker(number):
        for repetition in range(40):
            row = (number + repetition) % len(keys)
            # Re-create noninterned strings and release them between calls.
            query = tuple(value.encode('utf-8').decode('utf-8') for value in keys[row])
            assert all(a == b for a, b in zip(query, keys[row]))
            assert native_wide.read('float', h.lookup_float, source, query, 'rate') == float(row)
            assert native_wide.read('float', h.lookup_float, source, query, 'other') == float(-row)
            if repetition % 10 == 0: gc.collect()
        return number
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(worker, range(12))) == list(range(12))
    if sysconfig.get_config_var('Py_GIL_DISABLED'):
        assert not sys._is_gil_enabled()


@pytest.mark.parametrize('kind,function,values', [
    ('any', h.any_true, (False, False, True)),
    ('all', h.all_true, (True, True, False)),
    ('max', h.max_value, (-0.0, 0.0, float('nan'), 2.5)),
])
def test_canonical_consumers_have_no_python_wrapper_frame(native_wide, kind, function, values):
    expected = function(iter(values))
    with PythonCalls(function.__code__) as observed:
        actual = native_wide.consume(kind, function, iter(values))
    assert actual == expected and observed.calls == []


@pytest.mark.parametrize('kind,function', [('any', h.any_true), ('all', h.all_true), ('max', h.max_value)])
def test_empty_consumers_and_replacement_identity(native_wide, kind, function):
    same_outcome(lambda: function(iter(())), lambda: native_wide.consume(kind, function, iter(())))
    values, marker, seen = object(), object(), []
    def replacement(argument):
        seen.append(argument)
        return marker
    assert native_wide.consume(kind, replacement, values) is marker
    assert seen == [values]
    error = RuntimeError('replacement exception identity')
    def fail(argument):
        assert argument is values
        raise error
    with pytest.raises(RuntimeError) as actual:
        native_wide.consume(kind, fail, values)
    assert actual.value is error


@pytest.mark.parametrize('kind,function,first', [('any', h.any_true, True), ('all', h.all_true, False)])
def test_predicate_short_circuit_preserves_iterator_position(native_wide, kind, function, first):
    events = []
    def values():
        events.append('first')
        yield first
        events.append('second')
        raise AssertionError('short circuit consumed an extra element')
    iterator = values()
    assert native_wide.consume(kind, function, iterator) is first
    assert events == ['first']
    iterator.close()


def test_maximum_preserves_nan_and_first_tie_identity(native_wide):
    class Item:
        def __init__(self, value): self.value = value
        def __gt__(self, other): return self.value > other.value
    first, tied, smaller = Item(3), Item(3), Item(2)
    assert native_wide.consume('max', h.max_value, iter((first, tied, smaller))) is first
    nan = float('nan')
    assert native_wide.consume('max', h.max_value, iter((nan, 2.0))) is nan
    assert native_wide.consume('max', h.max_value, iter((2.0, nan))) == 2.0
