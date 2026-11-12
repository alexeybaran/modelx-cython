"""Public table provider admits physical mixed storage without Python helpers."""
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pandas_inputs as inputs, typed_inputs as h
from modelx_helpers.immutable_inputs import Table, Record, freeze
from modelx_helpers.table_operations import column, row, to_pandas
from modelx_cython.tests.helper_contracts.test_table_operations import native


def frames():
    yield pd.DataFrame({'age': [2**53+1, 42], 'rate': [.1, np.nan]})
    yield pd.DataFrame({'age': [42, 43], 'rate': [.1, np.nan],
                        'sex': ['M', 'F'], 'active': [True, False]})
    yield pd.DataFrame({'a': ['M', 'F'], 'b': ['X', None]})
    yield pd.DataFrame({'a': [True, False], 'b': [False, True]})
    yield pd.DataFrame({'a': pd.Series([1, 2], dtype='int32'),
                        'b': pd.Series([3, 4], dtype='int64')})
    yield pd.DataFrame({'a': pd.Series([.1, .2], dtype='float32'),
                        'b': pd.Series([3., 4.], dtype='float64')})
    yield pd.DataFrame({'a': pd.Series([None, pd.NA], dtype=object), 'b': [.1, .2]})


def guard(operation, *, preparation=False, single_column_pass=False):
    forbidden = {inputs.table.__code__, inputs._report_metadata.__code__,
                 inputs._scalar.__code__, freeze.__code__, Table.__init__.__code__,
                 Record.__init__.__code__, row.__code__, h.lookup_object.__code__}
    if single_column_pass:
        forbidden.add(pd.DataFrame.dtypes.fget.__code__)
    calls = []
    def profile(frame, event, arg):
        if event == 'call' and (frame.f_code in forbidden or not preparation
                and frame.f_globals.get('__name__', '').startswith('pandas')):
            calls.append((frame.f_globals.get('__name__'), frame.f_code.co_name))
    sys.setprofile(profile)
    try:
        actual = operation()
    finally:
        sys.setprofile(None)
    assert calls == []
    return actual


@pytest.mark.parametrize('source', list(frames()))
@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('rows,report', [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize('raw', [False, True])
def test_public_provider_mixed_storage_exact_contract(native, source, empty, rows, report, raw):
    source = source.iloc[:0].copy() if empty else source.copy(deep=True)
    source.index = pd.RangeIndex(9, 9+len(source), name='policy')
    source.columns.name = 'field'
    source.flags.allows_duplicate_labels = False
    expected = inputs.table(source, rows=rows, report=report, preserve_scalar_types=raw)
    owned = guard(lambda: native.prepare_options(inputs.table, source, rows, report, raw),
                  preparation=True, single_column_pass=rows and raw)
    assert type(owned) is Table
    for restored in [owned, pickle.loads(pickle.dumps(owned))]:
        assert restored._preserve_scalar_types is raw
        for actual_row, expected_row in zip(restored._values, expected._values):
            for actual, value in zip(actual_row, expected_row):
                assert type(actual) is type(value)
                if value is None or value is pd.NA or value is pd.NaT:
                    assert actual is value
                else:
                    assert actual == value or pd.isna(actual) and pd.isna(value)
        actual = to_pandas(restored)
        pd.testing.assert_frame_equal(actual, source, check_exact=True, check_flags=rows or report)
        if rows or report:
            assert actual.attrs == source.attrs
        if rows:
            for key in tuple(source.index):
                selected = guard(lambda: native.select(row, restored, key))
                public = source.loc[key]
                pd.testing.assert_series_equal(to_pandas(selected), public, check_exact=True)
                for name in source.columns:
                    assert type(selected[name]) is type(public[name])
        with pytest.raises(KeyError) as error:
            native.raw_value(h.lookup_object, restored, 'absent', 'absent-column')
        assert error.value.args == ('absent-column',)
        assert pickle.loads(pickle.dumps(restored))._native_cache == {}
    if len(source):
        source.iloc[0, 0] = source.iloc[1, 0]
        assert type(owned._values[0][0]) is type(expected._values[0][0])
        value, prior = owned._values[0][0], expected._values[0][0]
        assert value is prior or value == prior or pd.isna(value) and pd.isna(prior)


@pytest.mark.parametrize('indexing', ['loc', 'at'])
def test_mixed_sparse_integer_axes_errors_and_raw_object_identity(native, indexing):
    unused = ['original']
    source = pd.DataFrame({'rate': pd.Series([np.float64(.1), object()], dtype=object).array,
                           'other': [unused, object()]}, index=pd.Index([9, 2], name='policy'))
    owned = guard(lambda: native.prepare_options(inputs.table, source, True, True, True, indexing),
                  preparation=True)
    assert guard(lambda: native.raw_value(h.lookup_object, owned, 9, 'other')) is unused
    assert type(guard(lambda: native.raw_value(h.lookup_object, owned, 9, 'rate'))) is np.float64
    unused.append('edited')
    assert native.raw_value(h.lookup_object, owned, 9, 'other') == ['original', 'edited']
    source.iloc[0, 0] = .9
    assert native.float_value(h.lookup_float, owned, 9, 'rate') == .1
    with pytest.raises(TypeError):
        native.float_value(h.lookup_float, owned, 2, 'rate')
    with pytest.raises(TypeError, match='unsupported immutable input'):
        native.prepare_options(inputs.table, source)


@pytest.mark.parametrize('source', [
    pd.DataFrame({'age': pd.Series([42, pd.NA], dtype='Int64'), 'rate': [.1, .2]}),
    pd.DataFrame({'age': [42], 'rate': [.1]}, index=pd.Index(['a'], name='policy')),
    pd.DataFrame({'age': [42], 'rate': [.1]}, index=pd.MultiIndex.from_tuples([('M', 9)])),
    pd.DataFrame({'rate': []}, index=pd.Index([], dtype=bool))])
def test_unadmitted_layouts_keep_reference_results_and_errors(native, source):
    actual = native.prepare_options(inputs.table, source, True, True)
    expected = inputs.table(source, rows=True, report=True)
    pd.testing.assert_frame_equal(to_pandas(actual), to_pandas(expected), check_exact=True)
    for key in tuple(source.index):
        pd.testing.assert_series_equal(to_pandas(row(actual, key)), source.loc[key], check_exact=True)
    if source.empty:
        with pytest.raises(KeyError) as error:
            native.select(row, actual, False)
        assert error.value.args == (False,)


def test_unsupported_column_selection_and_replacements_keep_keyword_layout(native):
    source = pd.DataFrame({'age': [42], 'rate': [.1]})
    seen = []
    def replacement(frame, *, rows, report, preserve_scalar_types, indexing):
        seen.append((frame, rows, report, preserve_scalar_types, indexing))
        return 'replacement'
    assert native.prepare_options(replacement, source, True, False, True, 'at') == 'replacement'
    assert seen == [(source, True, False, True, 'at')]


@pytest.mark.parametrize('opaque', [False, True])
@pytest.mark.parametrize('raw', [False, True])
def test_duplicate_axes_preserve_reference_exception_priority(native, opaque, raw):
    source = pd.DataFrame({'rate': [.1, .2],
                           'other': [object(), object()] if opaque else [42, 43]}, index=[1, 1])
    with pytest.raises(Exception) as expected:
        inputs.table(source, preserve_scalar_types=raw)
    with pytest.raises(type(expected.value)) as actual:
        native.prepare_options(inputs.table, source, raw=raw)
    assert actual.value.args == expected.value.args


def test_native_report_column_restores_owned_object_containers(native):
    frame = pd.DataFrame({'value': [{'rates': [0.1]}, ('fixed', [0.2])]})
    owned = native.prepare_options(inputs.table, frame, report=True)
    expected = pd.Series([{'rates': [0.1]}, ('fixed', [0.2])], name='value')
    for _ in range(2):
        result = to_pandas(native.report_values(column, owned, 'value'))
        pd.testing.assert_series_equal(result, expected, check_exact=True)
        result.iloc[0]['rates'][0] = 0.9
