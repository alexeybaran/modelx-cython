"""PreparedTable (runtime/_tables.pyx) answers keyed reads exactly as pandas does (T-2).

Random keys, misses included, on every index kind the preparation accepts: dense and sparse int, text,
IntEnum, float, unsorted, two and three levels, a Series. Each read is compared with ``.loc[key, col]`` in value
and type, a miss with pandas' KeyError; ``in index`` and ``index.asof`` likewise. A typed probe module
exercises the C-level reads compiled code uses (code_ll, row_codes, prefix_range, f64_at, i64_at).
"""
import enum
import zlib
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'

PROBE = textwrap.dedent('''\
    # cython: freethreading_compatible=True
    from _tables cimport PreparedTable

    def flat_f64(PreparedTable table, long long key, Py_ssize_t slot):
        cdef Py_ssize_t row = table.code_ll(0, key) if table.nlevels == 1 else -1
        if row < 0:
            raise KeyError(key)
        return table.f64_at(row, slot)

    def codes_row(PreparedTable table, codes):
        cdef Py_ssize_t[8] buffer
        for i, c in enumerate(codes):
            buffer[i] = c
        return table.row_codes(buffer, len(codes))

    def prefix_count(PreparedTable table, codes):
        cdef Py_ssize_t[8] buffer
        cdef Py_ssize_t start = 0, stop = 0
        for i, c in enumerate(codes):
            buffer[i] = c
        if not table.prefix_range(buffer, len(codes), &start, &stop):
            return 0
        return stop - start

    def i64_value(PreparedTable table, key, Py_ssize_t slot):
        cdef Py_ssize_t row = table.row(key)
        if row < 0:
            raise KeyError(key)
        return table.i64_at(row, slot)
''')


@pytest.fixture(scope='module')
def tables(tmp_path_factory):
    folder = tmp_path_factory.mktemp('prepared')
    shutil.copy(RUNTIME / '_tables.pyx', folder / '_tables.pyx')
    shutil.copy(RUNTIME / '_tables.pxd', folder / '_tables.pxd')
    (folder / 'tables_probe.pyx').write_text(PROBE, encoding='utf-8')
    (folder / 'setup.py').write_text(
        'from setuptools import setup\nfrom Cython.Build import cythonize\n'
        'setup(ext_modules=cythonize(["_tables.pyx", "tables_probe.pyx"], '
        'compiler_directives={"freethreading_compatible": True, "language_level": 3}))\n', encoding='utf-8')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=folder, check=True,
                   stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(folder))
    try:
        import _tables
        import tables_probe
        yield _tables, tables_probe
    finally:
        sys.path.remove(str(folder))


# read signatures of a model, as the build writes them: (levels, columns of a row read by field, label, kind)
SIGNATURES = (
    ('i', (('q', 0), ('n', 2)), 'n', 2),            # field n (kept as an object) of a row of a float64 and another column
    ('i', (('q', 0), ('n', 2)), 'q', 0),
    ('i', None, 'q', 0),                            # a scalar read of q
)


@pytest.fixture(scope='module')
def model_tables(tmp_path_factory):
    """The runtime as a compiled model holds it: in the model's package, beside its read signatures."""
    folder = tmp_path_factory.mktemp('model')
    package = folder / 'sigmodel'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    shutil.copy(RUNTIME / '_tables.pyx', package / '_tables.pyx')
    shutil.copy(RUNTIME / '_tables.pxd', package / '_tables.pxd')
    (package / '_mx_table_signatures.py').write_text(f'SIGNATURES = {SIGNATURES!r}\n', encoding='utf-8')
    (folder / 'setup.py').write_text(
        'from setuptools import setup\nfrom Cython.Build import cythonize\n'
        'setup(ext_modules=cythonize(["sigmodel/_tables.pyx"], '
        'compiler_directives={"freethreading_compatible": True, "language_level": 3}))\n', encoding='utf-8')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=folder, check=True,
                   stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(folder))
    try:
        from sigmodel import _tables
        yield _tables
    finally:
        sys.path.remove(str(folder))


class Plan(enum.IntEnum):
    BASIC = 1
    PLUS = 2
    MAX = 5


def frames():
    rng = np.random.default_rng(7)
    n = 40
    values = dict(q=rng.random(n), k=rng.integers(-5, 5, n).astype(np.int64),
                  s=[f'v{i}' for i in range(n)], b=rng.random(n) > 0.5)
    yield 'dense int', pd.DataFrame(values, index=pd.Index(range(20, 20 + n), name='age'))
    yield 'sparse int', pd.DataFrame(values, index=pd.Index(rng.permutation(np.arange(0, 3 * n, 3)), name='k'))
    yield 'text', pd.DataFrame(values, index=pd.Index([f'key{i}' for i in rng.permutation(n)]))
    yield 'float', pd.DataFrame(values, index=pd.Index(np.arange(n) * 0.5))
    yield 'two levels', pd.DataFrame(values, index=pd.MultiIndex.from_product([['F', 'M'], range(20)]))
    yield 'three levels enum', pd.DataFrame(
        {'q': rng.random(3 * 2 * 4), 'k': np.arange(24, dtype=np.int64)},
        index=pd.MultiIndex.from_product([list(Plan), [1, 2], [30, 31, 35, 40]]))
    yield 'unsorted two levels', pd.DataFrame(values, index=pd.MultiIndex.from_tuples(
        [(('C', 'A', 'B')[i % 3], int(x)) for i, x in enumerate(rng.permutation(n))]))


def keys_of(frame, rng):
    present = list(frame.index)
    if frame.index.nlevels == 1:
        label = present[0]
        if isinstance(label, str):
            absent = ['nokey', 'key999']
        elif isinstance(label, float):
            absent = [-1.0, 1e9, 0.25]
        else:
            absent = [-999, 10 ** 6, int(present[0]) + 10 ** 5]
    else:
        absent = [tuple('Z' if isinstance(p, str) else 999 for p in present[0])]
    return [present[int(i)] for i in rng.integers(0, len(present), 60)] + absent


def outcome(function):
    try:
        return 'ok', function()
    except KeyError as error:
        return 'KeyError', error.args
    except Exception as error:
        return type(error).__name__, str(error)


@pytest.mark.parametrize('name,frame', list(frames()), ids=[n for n, _ in frames()])
def test_value_equals_loc(tables, name, frame):
    _tables, _ = tables
    table = _tables.prepare(frame)
    assert table is not None, _tables.unprepared_reason(frame)
    assert table.source is frame
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    for key in keys_of(frame, rng):
        for column in frame.columns:
            want, got = outcome(lambda: frame.loc[key, column]), outcome(lambda: table.value(key, column))
            assert want[0] == got[0], (name, key, column, want, got)
            if want[0] == 'ok':
                assert type(want[1]) is type(got[1]) and want[1] == got[1], (name, key, column, want, got)
        assert (key in frame.index) == table.contains(key), (name, key)


def test_series(tables):
    _tables, _ = tables
    series = pd.Series(np.arange(5) * 0.1, index=pd.Index([3, 1, 4, 9, 5]), name='q')
    table = _tables.prepare(series)
    for key in [1, 3, 9, 2, 100]:
        want, got = outcome(lambda: series.loc[key]), outcome(lambda: table.value(key))
        assert want[0] == got[0] and (want[0] != 'ok' or (type(want[1]) is type(got[1]) and want[1] == got[1]))


def test_typed_reads(tables):
    _tables, probe = tables
    frame = pd.DataFrame({'q': np.arange(30) * 0.25, 'k': np.arange(30, dtype=np.int64) * 3},
                         index=pd.Index(np.arange(30) * 2 + 7))
    table = _tables.prepare(frame)
    q, k = table.column('q'), table.column('k')
    for key in list(frame.index) + [8, -1, 10 ** 9]:
        want = outcome(lambda: float(frame.loc[key, 'q']))
        got = outcome(lambda: probe.flat_f64(table, key, 0))
        assert want == got, (key, want, got)
        want = outcome(lambda: int(frame.loc[key, 'k']))
        assert want == outcome(lambda: probe.i64_value(table, key, 0)), key
    multi = pd.DataFrame({'r': np.arange(8) * 1.0},
                         index=pd.MultiIndex.from_product([['A', 'B'], [1, 2, 3, 4]]))
    table = _tables.prepare(multi)
    a = table.level_labels[0].index("A")
    assert probe.prefix_count(table, [a]) == len(multi.loc['A'])
    for key in multi.index:
        assert probe.codes_row(table, table.key_codes(key)) >= 0
    assert probe.codes_row(table, [a, 9]) == -1


def test_asof(tables):
    _tables, _ = tables
    for index in (pd.Index([20, 21, 25, 30]), pd.Index([0.5, 1.5, 3.0]), pd.Index([5, 1, 3])):
        frame = pd.DataFrame({'q': np.arange(len(index)) * 1.0}, index=index)
        table = _tables.prepare(frame)
        sorted_index = index.sort_values()
        # NaN is left out: pandas' asof of NaN differs by version (NaN on pandas 3, the last label on 2.2), so the
        # compiled reads take it to pandas (test_table_probe's asof cells check that against live)
        for x in [-10, 0.5, 1, 1.49, 2, 20, 22.5, 30, 31]:
            want = sorted_index.asof(x)
            got = table.asof(x)
            assert (want != want and got != got) or (type(want) is type(got) and want == got), (list(index), x, want, got)


def test_extremes(tables):
    _tables, _ = tables
    for frame in (pd.DataFrame({'q': [1.0, 2.0]}, index=pd.Index([3, 9])),
                  pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=pd.MultiIndex.from_tuples([('A', 1), ('B', 2), ('B', 5)]))):
        table = _tables.prepare(frame)
        assert type(table.max_label) is type(frame.index.max()) and table.max_label == frame.index.max()
        assert type(table.min_label) is type(frame.index.min()) and table.min_label == frame.index.min()


@pytest.mark.parametrize('frame,reason', [
    (pd.DataFrame({'q': [1.0, 2.0]}, index=pd.Index([1, 1])), 'not unique'),
    (pd.DataFrame({'q': [1.0, 2.0]}, index=pd.Index([1.0, np.nan])), 'NaN'),
    (pd.DataFrame([[1.0, 2.0]], columns=['q', 'q']), 'duplicate column'),
    (pd.DataFrame({'q': [1.0, 2.0]}, index=pd.MultiIndex.from_tuples([('A', 1), ('A', 1)])), 'not unique'),
])
def test_left_to_pandas(tables, frame, reason):
    _tables, _ = tables
    assert _tables.prepare(frame) is None
    assert reason in _tables.unprepared_reason(frame)


def own_rule_frames():
    """Indexes whose labels pandas finds by rules of its own (a text for a date, a number inside an interval), or
    whose labels it cannot order: a read of them is pandas' (the table is not prepared), never a wrong answer or an
    error of the preparation."""
    import datetime
    q = {'q': [1.0, 2.0]}
    yield 'datetime', pd.DataFrame(q, index=pd.to_datetime(['2020-01-01', '2021-01-01'])), [
        '2020-01-01', '2021-01-01', '2019-01-01', datetime.datetime(2020, 1, 1), pd.Timestamp('2021-01-01')]
    yield 'interval', pd.DataFrame(q, index=pd.IntervalIndex.from_breaks([0, 10, 20])), [5, 10, 15, 25, pd.Interval(0, 10)]
    yield 'period', pd.DataFrame(q, index=pd.period_range('2020-01', periods=2, freq='M')), [
        '2020-01', '2020-02', '2020-03', pd.Period('2020-02', 'M')]
    yield 'timedelta', pd.DataFrame(q, index=pd.to_timedelta(['1D', '2D'])), ['1D', '2D', '3D', pd.Timedelta('1D')]
    yield 'categorical', pd.DataFrame(q, index=pd.CategoricalIndex(['a', 'b'])), ['a', 'b', 'c']
    yield 'mixed types', pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=pd.Index([1, 'a', 2.5], dtype=object)), [1, 'a', 2.5, 'b']
    yield 'beyond 64 bits', pd.DataFrame(q, index=pd.Index([1, 2 ** 63 + 5], dtype='uint64')), [1, 2 ** 63 + 5, 7]
    yield 'bool', pd.DataFrame(q, index=pd.Index([True, False])), [True, False, 1, 0, 1.0]
    yield 'date level', pd.DataFrame(q, index=pd.MultiIndex.from_arrays(
        [['a', 'b'], pd.to_datetime(['2020-01-01', '2021-01-01'])])), [('a', '2020-01-01'), ('b', '2021-01-01'), ('a', '2021-01-01')]


@pytest.mark.parametrize('name,frame,keys', list(own_rule_frames()), ids=[n for n, _, _ in own_rule_frames()])
def test_index_pandas_looks_up_by_its_own_rules(tables, name, frame, keys):
    _tables, _ = tables
    table = _tables.prepare(frame)          # must not raise: a frame Cell binds its frame when it is computed
    if table is None:
        assert _tables.unprepared_reason(frame)
        return
    for key in keys:
        want, got = outcome(lambda: frame.loc[key, 'q']), outcome(lambda: table.value(key, 'q'))
        assert want[0] == got[0] and (want[0] != 'ok' or want[1] == got[1]), (name, key, want, got)
        assert (key in frame.index) == table.contains(key), (name, key)


def test_bool_and_huge_keys(tables):
    """A bool is no label of a number level although it equals 1 or 0 (pandas raises KeyError; the dict of the level
    would find 1.0), and an integer beyond 64 bits is absent from an int level (KeyError, not OverflowError)."""
    _tables, _ = tables
    floats = pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=pd.Index([0.0, 1.0, 2.5]))
    two = pd.DataFrame({'q': [1.0, 2.0]}, index=pd.MultiIndex.from_tuples([('a', 1.0), ('b', 0.0)]))
    ints = pd.DataFrame({'q': [1.0, 2.0]}, index=pd.Index([1, 2]))
    cases = [(floats, [True, False, np.True_, np.False_, 1, 0, 1.0, 2.5]),
             (two, [('a', True), ('b', False), ('a', 1), ('a', 1.0), ('b', 0)]),
             (ints, [2 ** 70, -2 ** 70, True, np.True_, 1, 2.0])]
    for frame, keys in cases:
        table = _tables.prepare(frame)
        assert table is not None
        for key in keys:
            want, got = outcome(lambda: frame.loc[key, 'q']), outcome(lambda: table.value(key, 'q'))
            assert want[0] == got[0] and (want[0] != 'ok' or want[1] == got[1]), (key, want, got)
            assert (key in frame.index) == table.contains(key), key


@pytest.mark.parametrize('dtype,native', [('int32', 1), ('float32', 1), ('uint8', 1), ('uint64', 1), (object, 3)])
def test_row_of_a_narrow_number_column_is_pandas(model_tables, dtype, native):
    """A row read by field is compiled for the kinds of every column, a column that is not float64 or int64 as an
    object kept as stored. pandas widens a row of number columns to one dtype (int32 beside float64: float64), so
    the stored numpy.int32 is not the value of the row's field: such a row is read on pandas. A row with a real
    object column keeps every value as stored and is read natively."""
    values = ['x', 'y', 'z'] if dtype is object else np.array([1, 2, 250]).astype(dtype)
    frame = pd.DataFrame({'q': [0.5, 1.5, 2.5], 'n': values}, index=pd.Index([10, 11, 12]))
    row = frame.loc[12]
    assert (row.dtype == object) == (dtype is object)
    if dtype is not object:
        assert type(row['n']) is not type(frame['n'].iat[2])        # widened: not the stored value
    # the two row signatures are native only for the object row; the scalar read of q always is
    assert model_tables.native_slots(frame) == native


def test_object_columns_hold_what_pandas_returns(tables):
    """Every value of a column kept as objects is the value ``.loc`` returns, in type too: text of both pandas
    text dtypes (a missing one included), Python objects, and columns of other dtypes."""
    _tables, _ = tables
    frame = pd.DataFrame({
        'text': pd.Series(['a', None, 'c', 'd'], dtype='str'),
        'objects': pd.Series(['a', 3, None, (1, 2)], dtype=object),
        'small': np.array([1, 2, 3, 4], dtype=np.int32),
        'flag': [True, False, True, False],
        'when': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']),
        'q': [0.5, 1.5, 2.5, 3.5]}).set_axis(pd.Index([4, 2, 3, 1]))
    table = _tables.prepare(frame)
    for key in frame.index:
        for column in frame.columns:
            want, got = frame.loc[key, column], table.value(key, column)
            assert type(want) is type(got) and (want == got or (want != want and got != got)), (key, column, want, got)


def test_copy_at_bind(tables):
    _tables, _ = tables
    frame = pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=pd.Index([10, 11, 12]))
    table = _tables.prepare(frame)
    frame.loc[11, 'q'] = 99.0                   # an in-place edit after binding: the prepared copy is unchanged
    assert table.value(11, 'q') == 2.0 and table.source is frame


def test_frame_bound_again_is_prepared_again(tables):
    """A frame Cell prepares its frame every time it binds it: a frame edited in place and then given to a new
    model is read as it is then, not from the copy an earlier model prepared for the same object."""
    _tables, _ = tables
    from modelx_cython.runtime import table_cells
    bind = table_cells._binders[_tables.__name__]       # what a frame Cell calls when it stores its value
    frame = pd.DataFrame({'q': [1.0, 2.0, 3.0], 'k': np.arange(3, dtype=np.int64)}, index=pd.Index([10, 11, 12]))
    first = bind(frame)
    assert first.value(11, 'q') == 2.0
    frame.loc[11, 'q'] = 99.0
    second = bind(frame)                    # the frame Cell of the new model
    assert second is not first and second.value(11, 'q') == 99.0
    assert first.value(11, 'q') == 2.0      # what the first model's items hold is not changed under them
    _tables.check_unchanged(frame)          # a read that looks the frame up finds the table bound last


@pytest.mark.parametrize('name,frame', list(frames()), ids=[n for n, _ in frames()])
def test_hundred_thousand_reads(tables, name, frame):
    """10^5 random keys per index kind (about 10% absent), one float and one int column: 0 differences."""
    _tables, _ = tables
    table = _tables.prepare(frame)
    rng = np.random.default_rng(zlib.crc32(name.encode()) + 1)
    present = list(frame.index)
    absent = keys_of(frame, rng)[-1]
    expected = {key: (float(frame.loc[key, 'q']), int(frame.loc[key, 'k'])) for key in present}
    picks = rng.integers(0, len(present) + max(1, len(present) // 9), 100_000)
    for pick in picks:
        key = present[pick] if pick < len(present) else absent
        want = expected.get(key)
        row = table.row(key)
        if want is None:
            assert row < 0, (name, key)
            continue
        assert (table.value(key, 'q'), table.value(key, 'k')) == want, (name, key)


def test_check_mode_detects_an_edit(tables):
    """T-4 verify_inputs: a frame edited in place after it was bound is detected; an unchanged one passes."""
    _tables, _ = tables
    frame = pd.DataFrame({'q': [1.0, 2.0, 3.0], 'k': np.arange(3, dtype=np.int64)}, index=pd.Index([10, 11, 12]))
    _tables.check_unchanged(frame)          # binds it, then checks: unchanged
    _tables.check_unchanged(frame)
    frame.loc[11, 'q'] = 9.0
    with pytest.raises(RuntimeError, match='edited in place'):
        _tables.check_unchanged(frame)
