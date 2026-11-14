"""W1.3: type-directed lowering of table reads (``table_reads.TableReads``), on generated text.

No C compile. A formula is lowered against a resolver whose schemas come from the live helper
(``modelx_helpers.immutable_inputs.table_schema``). The text tests pin the generated shape of
every vocabulary item. The value tests run the lowered text and the original side by side in
Python: handle calls go to ``PyHandle``, an implementation of the M1 handle interface that
forwards every call to the helper table (the W1.1 forwarding path), so they check that each
lowered read addresses the same table, key parts, row and column as the original, with the
same conversions and errors. The runtime handles themselves are W1.2's (matrix squares).
"""

import enum
import math
from types import SimpleNamespace

import libcst as cst
import pandas as pd
import pytest

from modelx_cython.admission import table_read_sites, TableReadSite
from modelx_cython.helpers import module_name
from modelx_cython.table_reads import PROVIDER, TableReads
from modelx_cython.usage import NOT_CELLS, RESOLVED

ii = pytest.importorskip('modelx_helpers.immutable_inputs')
if not hasattr(ii, 'table_schema'):     # pragma: no cover - round-2 helper
    pytest.skip('needs the W1.H helper contract table_schema', allow_module_level=True)
from modelx_helpers import pdn

M = module_name(PROVIDER)
TABLE = 'modelx_helpers.immutable_inputs.Table'
RECORD = 'modelx_helpers.immutable_inputs.Record'


# ---------------------------------------------------------------------- the forwarding handle
class PyHandle:
    """The M1 TableHandle interface, each call forwarded to the helper table."""

    def __init__(self, table):
        self.table = table
        self.columns = ii.table_storage(table)[-2]

    @staticmethod
    def _key(parts):
        return parts[0] if len(parts) == 1 else tuple(parts)

    def col(self, name):
        if name not in self.columns:
            raise KeyError(name)
        return self.columns.index(name)

    def _row(self, *parts):
        key = self._key(parts)
        self.table.loc[key]         # KeyError(key) when absent
        return key

    row_1 = row_2 = row_3 = row_4 = row_5 = row_6 = _row

    def _has(self, *parts):
        return self._key(parts) in self.table.index

    has_1 = has_2 = has_3 = has_4 = has_5 = has_6 = _has

    def f64(self, row, col):
        return float(self.table.loc[row, self.columns[col]])

    def i64(self, row, col):
        return int(self.table.loc[row, self.columns[col]])

    def obj(self, row, col):
        return self.table.loc[row, self.columns[col]]

    def first_int(self):
        return int(self.table.index.min())

    def last_int(self):
        return int(self.table.index.max())

    def first_key(self):
        return self.table.index.min()

    def last_key(self):
        return self.table.index.max()

    def asof_int(self, k):
        return int(self.table.index.asof(k))

    def asof_row(self, k):
        return self.table.index.asof(k)

    def searchsorted_right(self, k):
        return int(self.table.index.searchsorted(k, side='right'))

    def key_int(self, i):
        return int(self.table.index[i])

    def _sub(self, *parts):
        return PyHandle(self.table.loc[self._key(parts)])

    sub_1 = sub_2 = sub_3 = sub_4 = sub_5 = _sub

    def table_object(self):
        return self.table

    def row_record(self, row):
        return self.table.loc[row]


class PyRecord:
    def __init__(self, record):
        self.record = record

    def field(self, name):
        if name not in self.record:
            raise KeyError(name)
        return name

    def obj(self, slot):
        return self.record[slot]


RUNTIME = {M: SimpleNamespace(table_handle=PyHandle, record_handle=PyRecord,
                              TableHandle=PyHandle, RecordHandle=PyRecord),
           '_mx_cy': SimpleNamespace(cast=lambda type_, value: int(value) if type_ == 'longlong' else value,
                                     longlong='longlong', Py_ssize_t=int)}


# ---------------------------------------------------------------------- a fake model
class Resolver:
    """Resolver facts of one class ``m.C``: every self-rooted call resolves by its last name."""

    def __init__(self, cells):
        self.cells = cells
        self.table_results, self.table_schemas = {}, {}
        for name, cell in cells.items():
            if cell.result is not None:
                self.table_results[f'm.C.{name}'] = cell.result
            if cell.schema is not None:
                self.table_schemas[f'm.C.{name}'] = cell.schema
        self.cached_cells = {f'm.C.{name}' for name in cells}

    def resolve_chain(self, cls, chain):
        return (RESOLVED, f'm.C.{chain[-1]}') if chain[-1] in self.cells else (NOT_CELLS, '')

    def cell_rettype(self, fqname):
        return self.cells[fqname.rsplit('.', 1)[1]].rettype


class Cell:
    def __init__(self, function, rettype='object'):
        self.function, self.rettype = function, rettype
        self.result = self.schema = None


class Formula:
    def __init__(self, params):
        self.params = list(params)
        self.fqname = 'm.C._f_formula'

    def has_args(self):
        return bool(self.params)

    def has_typeinfo(self):
        return True

    def get_argtype_expr(self, name, c_style=False, formula=False):
        return self.params[name] if isinstance(self.params, dict) else 'long long'


class Model:
    """Cells (zero-argument tables and records, typed scalar cells) and enum References."""

    def __init__(self, **cells):
        self.cells, self.refs, self.values = {}, {}, {}
        for name, value in cells.items():
            self.add(name, value)

    def add(self, name, value, rettype=None):
        if isinstance(value, ii.Table):
            cell = Cell(lambda value=value: value)
            cell.result, cell.schema = TABLE, ii.table_schema(value)
        elif isinstance(value, ii.Record):
            cell = Cell(lambda value=value: value)
            cell.result = RECORD
        elif isinstance(value, type) and issubclass(value, enum.Enum):
            self.refs[name] = SimpleNamespace(enum=value, type_=type(value))
            self.values[name] = value
            return self
        elif callable(value):
            cell = Cell(value, rettype or 'object')
        else:
            raise TypeError(value)
        self.cells[name] = cell
        return self

    def receiver(self):
        space = SimpleNamespace(**{name: cell.function for name, cell in self.cells.items()}, **self.values)
        space.data = space
        return space


def lower(model, source, params=('t',), typed=True, recognisers=False):
    """Lower one formula: (generated text, TableReads, sites)."""
    node = cst.parse_module(source).body[0]
    resolver = Resolver(model.cells)
    cls_info = SimpleNamespace(fqname='m.C', refs=model.refs)
    policy = SimpleNamespace(typed_table_reads=typed, table_recognisers=recognisers)
    sites = table_read_sites(node, resolver, cls_info)
    providers = set()
    reads = TableReads(resolver, cls_info, Formula(params), providers, policy, sites)
    new = reads.lower(node)
    assert (new is node) == (not providers), 'the provider is added exactly when the text changes'
    return cst.Module(body=[new]).code, reads, sites


def _run(source, receiver, args, namespace):
    scope = dict(namespace)
    exec(compile(source, '<formula>', 'exec'), scope)
    function = next(value for name, value in scope.items() if name.startswith('_f_'))
    try:
        return 'value', function(receiver, *args)
    except Exception as error:      # compared by type and payload
        return 'raise', (type(error), error.args)


def same_values(model, source, arguments, **options):
    """Lower ``source`` and check that it gives the original's values (and errors) on every
    argument tuple; returns the TableReads."""
    text, reads, _ = lower(model, source, **options)
    receiver = model.receiver()
    for args in arguments:
        args = args if isinstance(args, tuple) else (args,)
        expected = _run(source, receiver, args, {})
        got = _run(text, receiver, args, RUNTIME)
        assert got == expected and type(got[1]) is type(expected[1]), (args, got, expected, text)
    return reads


def body(text):
    """The formula body without its def line, dedented, one statement per line."""
    lines = text.splitlines()[1:]
    return '\n'.join(line[4:] for line in lines if line.strip())


# ---------------------------------------------------------------------- tables
class Sex(enum.IntEnum):
    M = 0
    F = 1


def mort():
    return pdn.table(pd.DataFrame(
        {'mort_rate': [0.01 * i for i in range(1, 11)], 'band': list(range(10)),
         'flag': [i % 2 == 0 for i in range(10)], 'note': list('abcdefghij')},
        index=pd.Index(range(30, 40), name='age')))


def gaps():
    return pdn.table(pd.DataFrame({'v': [0.5, 1.5, 2.5, 3.5], 'n': [1, 2, 3, 4]},
                                  index=pd.Index([1, 2, 5, 9], name='k')))


def multi():
    frame = pd.DataFrame({'sex': [Sex.M, Sex.M, Sex.F, Sex.F] * 2, 'plan': ['A', 'B'] * 4,
                          'year': [1, 1, 1, 1, 2, 2, 2, 2], 'v': [float(i) for i in range(8)]})
    return pdn.table(frame.set_index(['sex', 'plan', 'year']))


def contiguous():
    return pdn.table(pd.DataFrame({'q': [0.25, 0.5, 0.75]}, index=pd.Index([5, 6, 7], name='k')))


def rectangular():
    index = pd.MultiIndex.from_product([[1, 2], [10, 20, 30]], names=['a', 'b'])
    return pdn.table(pd.DataFrame({'q': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=index))


def points():
    return pdn.table(pd.DataFrame({'premium': [100.0, 250.0], 'age': [40, 50], 'tag': ['x', 'y']},
                                  index=pd.Index([1, 2], name='point_id')))


def base_model():
    model = Model(mort_table=mort(), gap_table=gaps(), multi_table=multi(), c_table=contiguous(),
                  r_table=rectangular(), Sex=Sex)
    model.add('age', lambda t: 30 + t % 12, 'long long')
    model.add('year', lambda t: 1 + t % 3, 'long long')
    model.add('sex', lambda: Sex.F, 'long long')
    model.add('plan', lambda: 'B', 'object')
    model.add('rate', lambda t: 0.5 * t, 'double')
    model.add('flag', lambda t: t % 2 == 0, 'bint')
    model.add('schedule', multi().loc[(Sex.F, 'B')])
    model.add('model_point', points().loc[2])
    model.add('mort_by_age', mort())
    return model


# ---------------------------------------------------------------------- generated text
def test_switch_off_leaves_the_formula_and_reports_nothing_typed():
    source = 'def _f_x(self, t):\n    return float(self.mort_table().loc[self.age(t), "mort_rate"])\n'
    node = cst.parse_module(source).body[0]
    model = base_model()
    _, reads, sites = lower(model, source, typed=False)
    assert reads.typed == 0 and not reads.lowered and len(sites) == 1
    resolver, info = Resolver(model.cells), SimpleNamespace(fqname='m.C', refs={})
    off = TableReads(resolver, info, Formula(['t']), set(), SimpleNamespace(), [])
    assert off.lower(node) is node


def test_loc_float_column_drops_float():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    return float(self.mort_table().loc[self.age(t), "mort_rate"])\n')
    assert body(text) == (
        f'_mx_h0: {M}.TableHandle = {M}.table_handle(self.mort_table())\n'
        'return _mx_h0.f64(_mx_h0.row_1(self.age(t)), _mx_h0.col("mort_rate"))')
    assert reads.typed == 1 and len(reads.lowered) == 1 and not reads.declined
    assert reads.keys == ['loc[key, column], 1 part: int (cell age returns long long)']


def test_at_is_the_same_read_and_columns_choose_the_method():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    a = self.mort_table().at[self.age(t), "mort_rate"]\n'
                           '    b = self.mort_table().loc[self.age(t), "band"]\n'
                           '    c = self.mort_table().loc[self.age(t), "flag"]\n'
                           '    return a, b, c, self.mort_table().loc[self.age(t), "note"]\n')
    assert '_mx_h0.f64(_mx_h0.row_1(self.age(t)), _mx_h0.col("mort_rate"))' in text
    assert '_mx_h0.i64(_mx_h0.row_1(self.age(t)), _mx_h0.col("band"))' in text
    assert '_mx_h0.obj(_mx_h0.row_1(self.age(t)), _mx_h0.col("flag"))' in text
    assert '_mx_h0.obj(_mx_h0.row_1(self.age(t)), _mx_h0.col("note"))' in text
    assert text.count('table_handle(') == 1 and reads.typed == 4


def test_int_of_int_bool_columns_reads_i64():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    return int(self.mort_table().loc[self.age(t), "band"]) + '
                           'int(self.mort_table().loc[self.age(t), "flag"]) + '
                           'int(self.mort_table().loc[self.age(t), "mort_rate"])\n')
    assert 'return _mx_h0.i64(_mx_h0.row_1(self.age(t)), _mx_h0.col("band")) + ' \
           '_mx_h0.i64(_mx_h0.row_1(self.age(t)), _mx_h0.col("flag")) + ' \
           'int(_mx_h0.f64(_mx_h0.row_1(self.age(t)), _mx_h0.col("mort_rate")))' in text


def test_tuple_keys_part_by_part_and_casts():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    k = t + 0.5\n'
                           '    return self.multi_table().loc[(self.sex(), self.plan(), self.year(t)), "v"]'
                           ' + self.r_table().loc[(self.rate(t), k), "q"]'
                           ' + self.r_table().loc[(self.flag(t), self.Sex.F), "q"]\n')
    assert '_mx_h0.row_3(self.sex(), self.plan(), self.year(t))' in text
    assert '_mx_h1.row_2(_mx_cy.cast(object, self.rate(t)), _mx_cy.cast(object, k))' in text
    assert '_mx_h1.row_2(_mx_cy.cast(object, self.flag(t)), _mx_cy.cast(_mx_cy.longlong, self.Sex.F))' in text
    assert reads.keys[0] == ('loc[key, column], 3 parts: int (cell sex returns long long); '
                             'object (cell plan returns object); int (cell year returns long long)')
    assert reads.keys[2].endswith('; enum member')


def test_tuple_key_local_is_read_back_by_position():
    text, _, _ = lower(base_model(), 'def _f_x(self, t):\n'
                       '    key = (self.sex(), self.plan(), self.year(t))\n'
                       '    return self.multi_table().loc[key, "v"] if key in self.multi_table().index else 0.0\n')
    assert '_mx_h0.row_3(key[0], key[1], key[2])' in text and '_mx_h0.has_3(key[0], key[1], key[2])' in text


def test_membership_and_partial_key_membership():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    a = self.age(t) in self.mort_table().index\n'
                           '    b = self.sex() not in self.multi_table().index\n'
                           '    return a and b\n')
    assert 'a = _mx_h0.has_1(self.age(t))' in text and 'b = not _mx_h1.has_1(self.sex())' in text
    assert reads.typed == 2


def test_index_min_max_and_the_clamp():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    tbl = self.mort_table()\n'
                           '    a = min(max(self.age(t), int(tbl.index.min())), int(tbl.index.max()))\n'
                           '    return float(tbl.loc[a, "mort_rate"]), self.multi_table().index.max()\n')
    assert body(text).splitlines()[:3] == [
        f'_mx_h_tbl: {M}.TableHandle',
        f'_mx_h_tbl = {M}.table_handle(self.mort_table())',
        'a = min(max(self.age(t), _mx_h_tbl.first_int()), _mx_h_tbl.last_int())']
    assert '_mx_h_tbl.f64(_mx_h_tbl.row_1(_mx_cy.cast(_mx_cy.longlong, a)), _mx_h_tbl.col("mort_rate"))' in text
    assert '_mx_h0.last_key()' in text and 'tbl = ' not in text.replace('_mx_h_tbl = ', '')
    assert reads.keys[0] == 'loc[key, column], 1 part: int (local a is assigned integers)'


def test_prefix_selection_local_is_a_typed_handle():
    text, reads, sites = lower(base_model(), 'def _f_x(self, t):\n'
                               '    sub = self.multi_table().loc[(self.sex(), self.plan())]\n'
                               '    if self.year(t) in sub.index:\n'
                               '        return float(sub.loc[self.year(t), "v"])\n'
                               '    return float(sub.loc[sub.index.max(), "v"])\n')
    assert body(text).splitlines()[:3] == [
        f'_mx_h_sub: {M}.TableHandle',
        f'_mx_h0: {M}.TableHandle = {M}.table_handle(self.multi_table())',
        '_mx_h_sub = _mx_h0.sub_2(self.sex(), self.plan())']
    assert 'if _mx_h_sub.has_1(self.year(t)):' in text
    assert '_mx_h_sub.f64(_mx_h_sub.row_1(_mx_h_sub.last_int()), _mx_h_sub.col("v"))' in text
    assert 'sub = ' not in text.replace('_mx_h_sub = ', '')
    assert len(reads.lowered) == len(sites) == 5 and not reads.declined


def test_inline_prefix_chain_and_column_selection():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    return self.multi_table().loc[(self.sex(), self.plan())].loc[self.year(t), "v"]'
                           ' + self.mort_table()["mort_rate"][self.age(t)]'
                           ' + self.mort_table()["band"][self.age(t)]\n')
    assert '_mx_h0.sub_2(self.sex(), self.plan()).f64(_mx_h0.sub_2(self.sex(), self.plan()).row_1(' in text
    assert '_mx_h1.f64(_mx_h1.row_1(self.age(t)), _mx_h1.col("mort_rate"))' in text
    assert '_mx_h1.i64(_mx_h1.row_1(self.age(t)), _mx_h1.col("band"))' in text
    assert reads.typed == 3 and not reads.declined


def test_row_local_reads_and_reassignment():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    row = self.mort_table().loc[self.age(t)]\n'
                           '    x = row["mort_rate"]\n'
                           '    if t > 3:\n'
                           '        row = self.mort_table().loc[self.age(t - 1)]\n'
                           '    return x + float(row["mort_rate"]) + int(row["band"])\n')
    assert body(text).splitlines()[:3] == [
        '_mx_row_row: _mx_cy.Py_ssize_t',
        f'_mx_h0: {M}.TableHandle = {M}.table_handle(self.mort_table())',
        '_mx_row_row = _mx_h0.row_1(self.age(t))']
    assert '_mx_row_row = _mx_h0.row_1(self.age(t - 1))' in text
    assert 'return x + _mx_h0.f64(_mx_row_row, _mx_h0.col("mort_rate")) + ' \
           '_mx_h0.i64(_mx_row_row, _mx_h0.col("band"))' in text
    assert reads.typed == 5


def test_row_local_from_two_tables_gets_a_row_handle():
    text, _, _ = lower(base_model(), 'def _f_x(self, t):\n'
                       '    if t:\n'
                       '        row = self.mort_table().loc[self.age(t)]\n'
                       '    else:\n'
                       '        row = self.mort_by_age().loc[self.age(t)]\n'
                       '    return row["mort_rate"]\n')
    assert '_mx_rh_row = _mx_h0' in text and '_mx_rh_row = _mx_h1' in text
    assert 'return _mx_rh_row.f64(_mx_row_row, _mx_rh_row.col("mort_rate"))' in text


def test_escaping_row_reads_row_record_and_returned_row():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    row = self.mort_table().loc[self.age(t)]\n'
                           '    return row["mort_rate"], len(row), self.gap_table().loc[t]\n', )
    assert 'len(_mx_h0.row_record(_mx_row_row))' in text
    assert '_mx_h1.row_record(_mx_h1.row_1(t))' in text


def test_asof_any_position_and_as_row_key():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    sub = self.schedule()\n'
                           '    k = sub.index.asof(self.year(t))\n'
                           '    return float(sub.loc[sub.index.asof(t), "v"]) + 2 * sub.index.asof(t + 1) + k\n')
    assert 'k = _mx_h_sub.asof_int(self.year(t))' in text
    assert '_mx_h_sub.f64(_mx_h_sub.asof_row(t), _mx_h_sub.col("v"))' in text
    assert '2 * _mx_h_sub.asof_int(t + 1)' in text and reads.typed == 4    # asof row: two sites


def test_searchsorted_and_nth_key():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    sub = self.schedule()\n'
                           '    nxt = sub.index[sub.index.searchsorted(t, side="right")]\n'
                           '    lo = sub.index.searchsorted(t + 1)\n'
                           '    return nxt, lo, sub.index[0], sub.index[-1], sub.index.searchsorted(t, side="left")\n')
    assert 'nxt = _mx_h_sub.key_int(_mx_h_sub.searchsorted_right(t))' in text
    assert 'lo = _mx_h_sub.searchsorted_right((t + 1) - 1)' in text
    assert '_mx_h_sub.key_int(0), _mx_h_sub.key_int(-1), _mx_h_sub.searchsorted_right(t - 1)' in text
    assert reads.typed == 6


def test_record_fields_any_position_only_without_recognisers():
    source = ('def _f_x(self, t):\n'
              '    mp = self.model_point()\n'
              '    if self.model_point()["premium"] > 0:\n'
              '        return float(mp["premium"]) * t + int(self.model_point()["age"])\n'
              '    return mp["tag"]\n')
    text, reads, _ = lower(base_model(), source)
    assert f'_mx_r_mp = {M}.record_handle(self.model_point())' in text
    assert f'_mx_r0: {M}.RecordHandle = {M}.record_handle(self.model_point())' in text
    assert 'if _mx_r0.obj(_mx_r0.field("premium")) > 0:' in text
    assert 'float(_mx_r_mp.obj(_mx_r_mp.field("premium"))) * t + int(_mx_r0.obj(_mx_r0.field("age")))' in text
    assert reads.typed == 4
    same, _, _ = lower(base_model(), source, recognisers=True)
    assert same == source


def test_reads_in_comprehensions_and_generators_respect_shadowing():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    tbl = self.mort_table()\n'
                           '    a = [float(tbl.loc[k, "mort_rate"]) for k in range(30, 30 + t)]\n'
                           '    b = sum(tbl.loc[k, "band"] for k in range(30, 32) if k in tbl.index)\n'
                           '    c = [tbl for tbl in (1, 2)]\n'
                           '    return a, b, c\n')
    assert 'a = [_mx_h_tbl.f64(_mx_h_tbl.row_1(_mx_cy.cast(object, k)), _mx_h_tbl.col("mort_rate")) for k in' in text
    assert 'if _mx_h_tbl.has_1(_mx_cy.cast(object, k)))' in text
    assert 'c = [tbl for tbl in (1, 2)]' in text and reads.typed == 3


def test_not_lowered_reads_are_declined_with_reasons():
    text, reads, sites = lower(base_model(), 'def _f_x(self, t, name):\n'
                               '    tbl = self.mort_table()\n'
                               '    s = 0.0\n'
                               '    for k in tbl.index:\n'
                               '        s += tbl.loc[k, name]\n'
                               '    for _, r in self.gap_table().iterrows():\n'
                               '        s += r["v"]\n'
                               '    sub = self.multi_table().loc[self.sex()]\n'
                               '    s += len(sub) + float(sub.loc[(self.plan(), 1), "v"])\n'
                               '    return s, tbl.loc[30:32, "band"], self.mort_table().index.asof(self.rate(t))\n',
                               params={'t': 'long long', 'name': 'object'})
    reasons = {site.read: reads.declined.get(id(site.node)) for site in sites if id(site.node) not in reads.lowered}
    assert reasons['tbl.index'].startswith('T.index used as a Python object')
    assert reasons['tbl.loc[k, name]'].startswith('column is not a string literal')
    assert reasons['self.gap_table().iterrows()'].startswith('iterrows()')
    assert reasons['r["v"]'].startswith('row of iterrows()')
    assert reasons['self.multi_table().loc[self.sex()]'].startswith("local 'sub' could not be lowered") \
        or 'kept as written' in reasons['self.multi_table().loc[self.sex()]']
    assert "sub-table local 'sub' is also used as a Python table (len(sub))" in \
        reasons['sub.loc[(self.plan(), 1), "v"]']
    assert reasons['tbl.loc[30:32, "band"]'].startswith('slice or starred key')
    assert reasons['self.mort_table().index.asof(self.rate(t))'].startswith('asof key is not known')
    assert 'sub = self.multi_table().loc[self.sex()]' in text
    assert all(reason.endswith('runs as a Python call') for reason in reasons.values())


def test_returned_table_valued_selection_is_the_child_table_object():
    source = ('def _f_x(self):\n'
              '    return self.multi_table().loc[(self.sex(), self.plan())]\n')
    text, reads, sites = lower(base_model(), source, params=())
    assert text.endswith('return _mx_h0.sub_2(self.sex(), self.plan()).table_object()\n'), text
    assert reads.typed == 1 and len(reads.lowered) == len(sites) == 1 and not reads.declined
    one = same_values(base_model(), 'def _f_x(self):\n    return self.multi_table().loc[self.sex()]\n',
                      [()], params=())
    assert one.typed == 1
    same_values(base_model(), source, [()], params=())


def test_table_valued_selection_used_otherwise_is_reported_not_built():
    text, reads, sites = lower(base_model(), 'def _f_x(self):\n'
                               '    return len(self.multi_table().loc[(self.sex(), self.plan())])\n', params=())
    assert reads.typed == 0 and text.endswith('return len(self.multi_table().loc[(self.sex(), self.plan())])\n')
    assert reads.declined[id(sites[0].node)].startswith('table-valued selection used as a value')


def test_column_local_of_literals_reads_by_its_common_kind():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    column = ("mort_rate" if t > 2 else "band")\n'
                           '    same = "mort_rate" if t else "mort_rate"\n'
                           '    return self.mort_table().loc[30, column] + float(self.mort_table().loc[31, same])\n')
    assert '_mx_h0.obj(_mx_h0.row_1(_mx_cy.cast(_mx_cy.longlong, 30)), _mx_h0.col(column))' in text
    assert '_mx_h0.f64(_mx_h0.row_1(_mx_cy.cast(_mx_cy.longlong, 31)), _mx_h0.col(same))' in text and reads.typed == 2


def test_untraced_and_key_count_mismatch():
    model = base_model()
    model.cells['mort_table'].schema = None
    _, reads, sites = lower(model, 'def _f_x(self, t):\n'
                            '    return self.mort_table().loc[t, "mort_rate"] + self.multi_table().loc[(1, 2, 3, 4), "v"]'
                            ' + self.multi_table().loc[(self.sex(), self.plan()), "v"]\n')
    reasons = [reads.declined[id(site.node)] for site in sites]
    assert reasons[0].startswith('receiver has no traced table schema')
    assert reasons[1].startswith('4 key parts on a table of 3 levels')
    assert reasons[2].startswith('2 key parts on a table of 3 levels')


def test_a_local_without_lowered_reads_is_left_as_written():
    source = ('def _f_x(self, t):\n'
              '    table = self.mort_table()\n'
              '    mp = self.model_point()\n'
              '    key = min(t, table.index.max_int())\n'
              '    return table.lookup_float(key, "mort_rate"), [k for k in table.index], mp\n')
    text, reads, _ = lower(base_model(), source)
    assert text == source and reads.typed == 0


def test_helper_names_never_collide_with_formula_names():
    text, _, _ = lower(base_model(), 'def _f_x(self, t):\n'
                       '    _mx_h0 = 1\n'
                       '    return float(self.mort_table().loc[self.age(t), "mort_rate"]) + _mx_h0\n')
    assert '_mx_h0_: ' in text and 'return _mx_h0_.f64(' in text and text.endswith('+ _mx_h0\n')


def test_nested_functions_and_lambdas_are_not_lowered():
    text, reads, sites = lower(base_model(), 'def _f_x(self, t):\n'
                               '    tbl = self.mort_table()\n'
                               '    f = lambda k: tbl.loc[k, "mort_rate"]\n'
                               '    return f(30) + tbl.loc[31, "mort_rate"]\n')
    assert 'f = lambda k: tbl.loc[k, "mort_rate"]' in text and 'tbl = self.mort_table()' in text
    assert f'_mx_h_tbl = {M}.table_handle(tbl)' in text and reads.typed == 1


def test_docstring_stays_first_and_one_line_bodies():
    text, _, _ = lower(base_model(), 'def _f_x(self, t):\n'
                       '    """Doc."""\n'
                       '    return self.mort_table().loc[self.age(t), "mort_rate"]\n')
    assert body(text).splitlines()[:2] == ['"""Doc."""', f'_mx_h0: {M}.TableHandle = {M}.table_handle(self.mort_table())']
    text, _, _ = lower(base_model(), 'def _f_x(self, t): return self.mort_table().loc[self.age(t), "mort_rate"]\n')
    assert '_mx_h0.f64(' in text and cst.parse_module(text)


# ---------------------------------------------------------------------- values (forwarding)
VALUE_FORMULAS = {
    'loc': ('return float(self.mort_table().loc[self.age(t), "mort_rate"])', range(-2, 14)),
    'at_int': ('return self.gap_table().at[t, "n"] + int(self.gap_table().loc[t, "n"])', range(0, 11)),
    'cols': ('return (self.mort_table().loc[30 + t % 10, "flag"], int(self.mort_table().loc[30 + t % 10, "flag"]),'
             ' self.mort_table().loc[30 + t % 10, "note"], self.mort_table().loc[30 + t % 10, "nope"])', range(3)),
    'tuple': ('return self.multi_table().loc[(self.sex(), self.plan(), self.year(t)), "v"]', range(5)),
    'tuple_local': ('key = (self.Sex(t % 2), "AB"[t % 3 % 2], 1 + t % 3)\n'
                    '    return float(self.multi_table().loc[key, "v"]) if key in self.multi_table().index else -1.0',
                    range(8)),
    'in': ('return (t in self.gap_table().index, (self.sex(), "B") in self.multi_table().index,'
           ' (self.sex(), "C") in self.multi_table().index, self.Sex.M not in self.multi_table().index)', range(10)),
    'clamp': ('tbl = self.gap_table()\n'
              '    lo, hi = int(tbl.index.min()), int(tbl.index.max())\n'
              '    return float(self.mort_table().loc[min(max(self.age(t), int(self.mort_table().index.min())),'
              ' int(self.mort_table().index.max())), "mort_rate"]), lo, hi', range(-5, 20)),
    'prefix': ('sub = self.multi_table().loc[(self.Sex(t % 2), "AB"[t % 2])]\n'
               '    if self.year(t) in sub.index:\n'
               '        return float(sub.loc[self.year(t), "v"]) + sub.index.max() + sub.index.min()\n'
               '    return -1.0', range(8)),
    'prefix_one': ('sub = self.multi_table().loc[self.Sex(t % 2)]\n'
                   '    return float(sub.loc[("AB"[t % 2], 1 + t % 3), "v"])', range(8)),
    'prefix_missing': ('sub = self.multi_table().loc[(self.Sex(t % 2), "ABC"[t % 3])]\n'
                       '    return float(sub.loc[1, "v"])', range(6)),
    'row': ('row = self.gap_table().loc[t]\n'
            '    x = row["v"]\n'
            '    if t > 3:\n'
            '        row = self.gap_table().loc[t - 3]\n'
            '    return x + float(row["v"]) + int(row["n"])', range(12)),
    'row_escape': ('row = self.gap_table().loc[t]\n'
                   '    return len(row) + row["n"], self.gap_table().loc[t]["v"]', range(4)),
    'asof': ('sub = self.gap_table()\n'
             '    return float(sub.loc[sub.index.asof(t), "v"]) + 2 * sub.index.asof(t), sub.index.asof(t + 2)',
             range(-1, 12)),
    'successor': ('sub = self.gap_table()\n'
                  '    return sub.index[sub.index.searchsorted(t, side="right")], sub.index.searchsorted(t),'
                  ' sub.index.searchsorted(t, side="left"), sub.index[t % 4], sub.index[-1 - t % 4]', range(-1, 10)),
    'column_view': ('return self.gap_table()["v"][t] + self.gap_table()["n"][t]', range(0, 10)),
    'column_choice': ('column = ("mort_rate" if t % 2 else "band")\n'
                      '    return float(self.mort_table().loc[30 + t % 10, column]), self.mort_table().loc[31, column]',
                      range(4)),
    'record': ('mp = self.model_point()\n'
               '    return float(mp["premium"]) * t + int(self.model_point()["age"]), mp["tag"], mp["nope"]', range(2)),
    'compr': ('tbl = self.gap_table()\n'
              '    return [float(tbl.loc[k, "v"]) for k in range(t) if k in tbl.index],'
              ' sum(tbl.loc[k, "n"] for k in tbl.index if k <= t), [tbl for tbl in range(2)]', range(10)),
    'contiguous': ('return self.c_table().loc[t, "q"], t in self.c_table().index, int(self.c_table().index.max()),'
                   ' self.c_table().index.asof(t)', range(4, 10)),
    'rectangular': ('sub = self.r_table().loc[1 + t % 3]\n'
                    '    return float(self.r_table().loc[(1 + t % 2, 10 * (1 + t % 3)), "q"]) + sub.loc[10, "q"]'
                    ' + sub.index.max()', range(6)),
    'float_keys': ('return self.gap_table().loc[float(t), "v"] + self.mort_table().loc[(t + 30.0), "mort_rate"]',
                   range(0, 6)),
}


@pytest.mark.parametrize('name', sorted(VALUE_FORMULAS))
def test_lowered_values_equal_the_original(name):
    statements, arguments = VALUE_FORMULAS[name]
    source = f'def _f_{name}(self, t):\n    {statements}\n'
    reads = same_values(base_model(), source, arguments)
    assert reads.typed > 0, name


# ---------------------------------------------------------------------- prototype cases ported
# integration_tests/test_loc_reads.py of the syntax experiment (96734a5): the formula families
# of each model, lowered here and run on the forwarding handle. Grouped tables (groups=) and
# the round-2 helper methods (slice, lookup_float, previous) are outside the target syntax (R1).
def prototype_model():
    model = base_model()
    frame = pd.DataFrame({'v': [0.5, 1.5, 2.5, 3.5], 'txt': ['1.25', 'abc', '2', 'x'],
                          'sc per 1000': [9.0, 8.0, 7.0, 6.0], 'n': [1, 2, 3, 4]}, index=pd.Index([1, 2, 5, 9], name='k'))
    model.add('gap_text_table', pdn.table(frame))
    model.add('named_table', pdn.table(pd.DataFrame({'v': [10.0, 20.0, 30.0], 'note': ['a', 'b', 'c']},
                                                    index=pd.Index(['UL9', 'UL5', 'X'], name='id'))))
    model.add('flag_table', pdn.table(pd.DataFrame({'v': [100.0, 200.0]}, index=pd.Index([True, False], name='f'))))
    model.add('two_table', pdn.table(pd.DataFrame({'a': [1, 1, 2, 2], 'b': [0, 1, 0, 2], 'v': [1.0, 2.0, 3.0, 4.0]})
                                     .set_index(['a', 'b'])))
    model.add('zero_one_table', pdn.table(pd.DataFrame({'a': [0, 1, 1], 'b': [0, 1, 2], 'v': [8.0, 9.0, 10.0]})
                                          .set_index(['a', 'b'])))
    model.add('bools_table', pdn.table(pd.DataFrame({'f': [True, False, True], 'b': [0, 1, 1], 'v': [5.0, 6.0, 7.0]})
                                       .set_index(['f', 'b'])))
    model.add('ka', lambda t: 1 + t % 3, 'long long')
    model.add('kb', lambda t: t % 3, 'long long')
    model.add('pid', lambda: 'UL5', 'object')
    model.add('ienum', lambda: enum.IntEnum('I', {'A': 1, 'B': 2, 'C': 5}))
    return model


def probe(expression):
    return ('try:\n        return ' + expression + '\n    except KeyError:\n        return -1.0\n'
            '    except ValueError:\n        return -2.0\n    except IndexError:\n        return -3.0')


PROTOTYPE = {
    # _model
    'rate': ('return float(self.mort_table().loc[30 + t % 10, "mort_rate"])', 1),
    'capped': ('tbl = self.mort_table()\n    tbl = self.mort_table()\n'
               '    return tbl.loc[min(30 + t, tbl.index.max()), "mort_rate"]', 2),
    'band': ('return float(self.mort_table().loc[30 + t % 10, "band"])', 1),
    'band_int': ('return int(self.mort_table().loc[30 + t % 10, "band"])', 1),
    'column_by_name': ('name = "mort_rate"\n    return float(self.mort_table().loc[30 + t % 10, name])', 1),
    'column_computed': ('return float(self.mort_table().loc[30 + t % 10, "mort_" + "rate"])', 0),
    'reassigned': ('k = 30 + t % 10\n    row = self.mort_table().loc[k]\n    k = k + 1\n'
                   '    return float(row["mort_rate"])', 2),
    # _edge_model
    'gap': (probe('float(self.gap_text_table().loc[t, "v"])'), 1),
    'gap_bounds': (probe('float(self.gap_text_table().loc[min(max(t, int(self.gap_text_table().index.min())),'
                         ' int(self.gap_text_table().index.max())), "n"])'), 3),
    'gap_text': (probe('float(self.gap_text_table().loc[t, "txt"])'), 1),
    'gap_odd_column': (probe('float(self.gap_text_table().loc[t, "sc per 1000"])'), 1),
    'gap_unknown_column': (probe('float(self.gap_text_table().loc[t, "nope"])'), 1),
    'gap_row': (probe('float(self.gap_text_table().loc[t]["v"]) + self.gap_text_table().loc[t]["n"]'), 2),
    'named': (probe('float(self.named_table().loc[("UL9", "X", "nope")[t % 3], "v"])'), 1),
    'flag': (probe('float(self.flag_table().loc[t % 2 == 0, "v"])'), 1),
    'key_int_enum': (probe('float(self.gap_text_table().loc[self.ienum()((1, 2, 5)[t % 3]), "v"])'), 1),
    'in_kinds': ('g = self.gap_text_table()\n    m = self.two_table()\n'
                 '    return ((1.0 in g.index) + 2 * (True in g.index) + 4 * (3 in g.index) + 8 * ((1,) in m.index)'
                 ' + 16 * ((1, 1, 2) in m.index if False else 0) + 32 * ((1, 1) in m.index)'
                 ' + 64 * (self.ienum()(5) in g.index) + 256 * ("UL9" in g.index))', 7),
    'at_read': (probe('float(self.gap_text_table().at[t, "v"])'), 1),
    'at_varcol': (probe('float(self.gap_text_table().at[t, ("v", "sc per 1000")[t % 2]])'), 0),
    'gap_in': ('tbl = self.gap_text_table()\n    return float(tbl.loc[t, "v"]) if t in tbl.index else 0.0', 2),
    # _variants_model
    'int_nested': ('tbl = self.mort_table()\n    return float(tbl.loc[min(max(t, int(tbl.index.min())),'
                   ' int(tbl.index.max())), "mort_rate"])', 3),
    'int_hoisted': ('tbl = self.mort_table()\n    lo = int(tbl.index.min())\n    hi = int(tbl.index.max())\n'
                    '    return float(tbl.loc[min(max(t, lo), hi), "mort_rate"])', 3),
    'int_reordered': ('m = self.mort_table()\n    upper = int(m.index.max())\n    lower = int(m.index.min())\n'
                      '    return float(m.loc[max(lower, min(t, upper)), "mort_rate"])', 3),
    'text_inline': ('return float(self.named_table().loc[self.pid(), "v"])', 1),
    'text_alias': ('tbl = self.named_table()\n    k = self.pid()\n    return float(tbl.loc[k, "v"])', 1),
    'text_row': ('row = self.named_table().loc[self.pid()]\n    return float(row["v"])', 2),
    'asof_lit': ('tbl = self.mort_table()\n    try:\n        return float(tbl.loc[tbl.index.asof(t), "mort_rate"])\n'
                 '    except KeyError:\n        return -1.0', 2),
    'asof_alias': ('tbl = self.mort_table()\n    try:\n        k = tbl.index.asof(t)\n'
                   '        return float(tbl.loc[k, "mort_rate"])\n    except KeyError:\n        return -1.0', 2),
    'int_in': ('tbl = self.mort_table()\n    return 1.0 if t in tbl.index else 0.0', 1),
    # _int_key_model
    'k2_literal': (probe('float(self.two_table().loc[(self.ka(t), self.kb(t)), "v"])'), 1),
    'k2_local': ('key = (self.ka(t), self.kb(t))\n    ' + probe('float(self.two_table().loc[key, "v"])'), 1),
    'k2_in': ('return 1.0 if (self.ka(t), self.kb(t)) in self.two_table().index else 0.0', 1),
    'k2_pyint_part': ('n = int(("1", "2", "3")[t % 3])\n    ' + probe('float(self.two_table().loc[(n, self.kb(t)), "v"])'), 1),
    'b_cmp_int': (probe('float(self.zero_one_table().loc[(t > 3, self.kb(t)), "v"])'), 1),
    'b_local_int': ('z = t > 5\n    ' + probe('float(self.zero_one_table().loc[(z, self.kb(t)), "v"])'), 1),
    'b_cmp_bool': (probe('float(self.bools_table().loc[(t > 3, self.kb(t) % 2), "v"])'), 1),
    # _prefix_model (int levels), _record_field_model
    'p_two': (probe('sub_two(self, t)').replace('sub_two(self, t)', 'float(self.two_table().loc[self.ka(t)].loc[t % 3, "v"])'), 1),
    'p_asof': ('sub = self.two_table().loc[self.ka(t)]\n    ' + probe('float(sub.loc[sub.index.asof(t), "v"])'), 3),
    'p_kept': ('sub = self.two_table().loc[self.ka(t)]\n    ' + probe('float(len(sub)) + float(sub.loc[t % 3, "v"])'), 0),
    'f_ret': ('return float(self.model_point()["premium"])', 1),
    'f_arith': ('return self.model_point()["premium"] * 2.0 + self.model_point()["age"] + t', 2),
    'f_alias': ('rec = self.model_point()\n    return rec["age"] - t', 1),
    'f_missing': ('try:\n        return float(self.model_point()["nope"])\n    except KeyError:\n        return -1.0', 1),
}


@pytest.mark.parametrize('name', sorted(PROTOTYPE))
def test_prototype_cases_ported(name):
    statements, typed = PROTOTYPE[name]
    source = f'def _f_{name}(self, t):\n    {statements}\n'
    reads = same_values(prototype_model(), source, range(-1, 12))
    assert reads.typed == typed, (name, reads.typed, reads.keys, reads.declined)


# ---------------------------------------------------------------------- report contract
def test_report_contract_with_the_transformer_hook():
    """The fields W1.1's hook reads: typed, lowered ids of sites, declined reasons, keys."""
    _, reads, sites = lower(base_model(), 'def _f_x(self, t):\n'
                            '    tbl = self.mort_table()\n'
                            '    return tbl.loc[self.age(t), "mort_rate"] + len(tbl.index)\n')
    assert isinstance(reads.typed, int) and isinstance(reads.keys, list)
    assert reads.lowered <= {id(site.node) for site in sites}
    assert set(reads.declined) <= {id(site.node) for site in sites} - reads.lowered
    entries = [site.entry(True, reads.declined.get(id(site.node))) for site in sites
               if id(site.node) not in reads.lowered]
    assert [entry['read'] for entry in entries] == ['tbl.index']
    assert isinstance(sites[0], TableReadSite)
