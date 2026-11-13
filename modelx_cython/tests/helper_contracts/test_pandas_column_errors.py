"""Selected pandas columns validate before rows and never use key defaults."""
import pickle
import sys

import pandas as pd
import pytest

from modelx_helpers import pandas_inputs as inputs, typed_inputs as h
from modelx_helpers.immutable_inputs import Table
from modelx_cython.tests.helper_contracts.test_table_operations import native


@pytest.mark.parametrize('indexing', ['loc', 'at'])
@pytest.mark.parametrize('empty', [False, True])
def test_scalar_column_errors_match_pandas_and_ignore_defaults(native, indexing, empty):
    frame = pd.DataFrame({'rate': [0.1]}, index=pd.Index([1], name='policy'))
    if empty:
        frame = frame.iloc[:0]
    prepared = inputs.table(frame, indexing=indexing, report=True)
    for source in [prepared, pickle.loads(pickle.dumps(prepared)),
                   native.prepare(inputs.table, frame, False)]:
        for key in [1, 9, False]:
            with pytest.raises(KeyError) as expected:
                getattr(frame, indexing)[key, 'absent']
            operations = [lambda: source.lookup(key, 'absent'),
                          lambda: h.lookup_float(source, key, 'absent'),
                          lambda: h.lookup_float(source, key, 'absent', default=7.0),
                          lambda: native.float_value(h.lookup_float, source, key, 'absent'),
                          lambda: native.float_default(h.lookup_float, source, key, 'absent', 7.0),
                          lambda: native.packed_float_default(h.lookup, source, key, 'absent', 7.0)]
            for operation in operations:
                with pytest.raises(KeyError) as actual:
                    operation()
                assert actual.value.args == expected.value.args
        assert h.lookup_float(source, 9, 'rate', default=7.0) == 7.0
        assert native.float_default(h.lookup_float, source, 9, 'rate', 7.0) == 7.0


def test_native_selected_column_exception_never_calls_python_helper(native):
    source = inputs.table(pd.DataFrame({'rate': [0.1]}, index=[1]))
    forbidden = {h.lookup.__code__, h.lookup_float.__code__, Table.lookup.__code__}
    calls = []
    def profile(frame, event, arg):
        if event == 'call' and (frame.f_code in forbidden or
                frame.f_globals.get('__name__', '').startswith('pandas')):
            calls.append(frame.f_code.co_name)
    sys.setprofile(profile)
    try:
        for key in [1, 9]:
            try:
                native.float_default(h.lookup_float, source, key, 'absent', 7.0)
            except KeyError as error:
                assert error.args == ('absent',)
            else:
                raise AssertionError('unknown column incorrectly used default')
    finally:
        sys.setprofile(None)
    assert calls == []


def test_grouped_and_full_multiindex_columns_retain_error_priority(native):
    frame = pd.DataFrame({'rate': [0.1]}, index=pd.MultiIndex.from_tuples(
        [('M', 42)], names=['sex', 'age']))
    for indexing in ['loc', 'at']:
        source = inputs.table(frame, indexing=indexing)
        for key in [('M', 42), ('missing', 99)]:
            with pytest.raises(KeyError) as expected:
                getattr(frame, indexing)[key, 'absent']
            with pytest.raises(KeyError) as actual:
                native.float_default(h.lookup_float, source, key, 'absent', 7.0)
            assert actual.value.args == expected.value.args
    leaf = inputs.table(frame, groups=('sex',), report=True)['M']
    with pytest.raises(KeyError) as error:
        native.float_default(h.lookup_float, leaf, 99, 'absent', 7.0)
    assert error.value.args == ('absent',)


def test_bare_table_and_legacy_pickle_column_contract_is_unchanged(native):
    source = Table([1], [[0.1]], ['rate'])
    for value in [source, pickle.loads(pickle.dumps(source))]:
        with pytest.raises(ValueError):
            h.lookup_float(value, 1, 'absent', default=7.0)
        assert h.lookup_float(value, 9, 'absent', default=7.0) == 7.0
        with pytest.raises(ValueError):
            native.packed_float_default(h.lookup, value, 1, 'absent', 7.0)
        assert native.packed_float_default(h.lookup, value, 9, 'absent', 7.0) == 7.0
