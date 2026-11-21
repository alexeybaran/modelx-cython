"""The prepared table's integer lookup preserves Python key conversion."""

import json
import math
import re
import subprocess
import sys

import modelx as mx
import pandas as pd


def test_builtin_int_key_uses_native_lookup_only_for_proven_int64(tmp_path):
    model = mx.new_model('IntegerKeys')
    try:
        data = model.new_space('Data')
        data.rates = pd.DataFrame({'rate': [3.0, 4.0, 1.0, 2.0]},
                                  index=pd.Index([-(2**63), 0, 20, 21], dtype='int64'))
        data.new_cells('table', 'lambda: rates')
        calc = model.new_space('Calc', refs={'data': data, 'convert': math.sin})
        calc.new_cells('offset', 'lambda t: 20 + t')
        calc.new_cells('native_key', '''def native_key(t):
    a, b = offset(t), offset(t)
    table = data.table()
    return table.loc[int(a), 'rate'] if int(b) in table.index else 0.0
''')
        calc.new_cells('float_key', 'lambda x: data.table().loc[int(x), "rate"]')
        calc.new_cells('string_key', 'lambda x: data.table().loc[int(x), "rate"]')
        calc.new_cells('shadowed', '''def shadowed(x):
    int = convert
    return int(x) in data.table().index
''')
        calc.new_cells('shadowed_asof', '''def shadowed_asof(x):
    int = convert
    return int(data.table().index.asof(x))
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text('''from exported import mx_model as model
calc = model.Calc
for t in (0, 1):
    calc.native_key(t)
    calc.float_key(20.5 + t)
    calc.string_key(str(20 + t))
    calc.shadowed(t)
    calc.shadowed_asof(20 + t)
calc.native_key(2)
''', encoding='utf-8')
    built = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                            '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr

    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    for key in ('a', 'b'):
        assert re.search(rf'\.kcode\([^\n]+_mx_cy\.cast\(_mx_cy\.longlong, int\({key}\)\)', source)
    assert '_mx_cy.cast(_mx_cy.longlong, int(x))' not in source

    checked = subprocess.run([sys.executable, '-c', '''
import math
from exported import mx_model as live
from exported_cy import mx_model as compiled

for model in (live, compiled):
    calc = model.Calc
    assert [calc.native_key(t) for t in (0, 1, 2)] == [1.0, 2.0, 0.0]
    assert calc.float_key(20.5) == calc.string_key('20') == 1.0
    assert [calc.shadowed(x) for x in (0, 1)] == [True, False]
    assert calc.shadowed_asof(20) == math.sin(20)
    for cell, value in ((calc.float_key, float('nan')), (calc.float_key, float(2**70)),
                        (calc.string_key, str(2**70))):
        try:
            cell(value)
        except (ValueError, OverflowError, KeyError):
            pass
        else:
            raise AssertionError(f'{cell} accepted {value!r} as a key')
'''], cwd=tmp_path, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_builtin_zip_of_one_prefix_walks_native_rows_or_keeps_pandas(tmp_path):
    frame = pd.DataFrame(
        {'surr_value': [0.5, 4.0, 7.0, 11.0]},
        index=pd.MultiIndex.from_tuples([('A', -5), ('A', 2**40), ('B', 1), ('B', 3)]),
    )
    model = mx.new_model('ZipPrefix')
    try:
        data = model.new_space('Data')
        data.raw = frame
        data.key = ('A',)
        data.new_cells('table', 'lambda: raw')
        calc = model.new_space('Calc', formula='lambda policy_id: None', refs={'data': data})
        calc.new_cells('selector', '''def selector():
    chosen = data.key[0]
    return chosen(data.table()) if callable(chosen) else chosen
''')
        calc.new_cells('anchors', '''def anchors():
    tbl = data.table().loc[selector()]
    return [(int(k), float(v)) for k, v in zip(tbl.index, tbl["surr_value"])]
''')
        calc.new_cells('other_view', '''def other_view():
    tbl = data.table().loc[selector()]
    other = data.table().loc["B"]
    return [(int(k), float(v)) for k, v in zip(tbl.index, other["surr_value"])]
''')
        calc.new_cells('strict_zip', '''def strict_zip():
    tbl = data.table().loc[selector()]
    return [(int(k), float(v)) for k, v in zip(tbl.index, tbl["surr_value"], strict=True)]
''')
        calc.new_cells('shadowed_zip', '''def shadowed_zip(zip):
    tbl = data.table().loc[selector()]
    return [(int(k), float(v)) for k, v in zip(tbl.index, tbl["surr_value"])]
''')
        calc.new_cells('row', 'lambda: data.table().loc[("A", -5)]')
        model.export(tmp_path / 'exported', locked_spaces=['Data'])
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text('''from exported import mx_model as model
item = model.Calc[1]
item.anchors()
item.other_view()
item.strict_zip()
item.shadowed_zip(zip)
item.row()
''', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({
        'strict_lowering': True,
        'spaces': {'Calc': {'cells': {'selector': {'return_type': 'object'}}}},
    }), encoding='utf-8')
    built = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                            '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr

    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    assert source.count('.zip_live_prefix(') == 1
    anchors = re.search(r'def _f_anchors\(.*?(?=\n    @)', source, re.S).group(0)
    assert anchors.count('.loc\n') == 1
    assert anchors.index('.loc\n') < anchors.index('.zip_live_prefix(')
    assert 'zip(tbl.index, tbl["surr_value"])' in anchors
    assert 'zip(tbl.index, other["surr_value"])' in source
    assert 'zip(tbl.index, tbl["surr_value"], strict=True)' in source
    assert 'zip(tbl.index, tbl["surr_value"])' in re.search(
        r'def _f_shadowed_zip\(.*?(?=\n    @)', source, re.S).group(0)
    assert '.loc[' in re.search(r'def _f_row\(.*?(?=\n    @)', source, re.S).group(0)
    reads = json.loads((tmp_path / 'exported_cy' / '_mx_table_reads.json').read_text(encoding='utf-8'))
    entry, = reads['Calc.anchors']
    assert entry['form'] == 'L8' and 'the current frame' in entry['pandas_when']

    checked = subprocess.run([sys.executable, '-c', '''
from contextlib import contextmanager
import pandas as pd
import sys
from pandas.core.indexing import _LocIndexer
from exported import mx_model as live
from exported_cy import mx_model as compiled

base = live.Data.raw
loc_code = _LocIndexer.__getitem__.__code__


@contextmanager
def count_loc():
    calls = []

    def tracked(frame, event, arg):
        if event == 'call' and frame.f_code is loc_code and type(frame.f_locals['self']) is _LocIndexer:
            calls.append(frame.f_locals['key'])

    sys.setprofile(tracked)
    try:
        yield calls
    finally:
        sys.setprofile(None)


def fresh(template, frame, key):
    model = type(template)()
    model.Data.raw = frame
    model.Data.key = (key,)
    return model


def outcome(call):
    try:
        return [(type(k), type(v), k, repr(v)) for k, v in call()]
    except (KeyError, TypeError, ValueError) as error:
        return type(error), error.args


class StringKey(str):
    pass


class ReversedPrefix(pd.DataFrame):
    @property
    def loc(self):
        ordinary = super().loc

        class Reverse:
            def __getitem__(self, key):
                return ordinary[key].iloc[::-1]

        return Reverse()


cases = [
    (base, 'A', True),
    (base.assign(surr_value=[-0.0, float('nan'), 7.0, 11.0]), 'A', True),
    (base.iloc[[2, 3, 0, 1]], 'A', True),  # unsorted prefix groups, contiguous mask
    (base.iloc[[2, 0, 3, 1]], 'A', False),  # disjoint mask
    (base.iloc[[1, 0, 2, 3]], 'A', True),  # reversed within prefix, actual row order
    (base.iloc[[0, 0, 1, 2, 3]], 'A', False),  # duplicate index
    (pd.concat([base, base + 100.0], axis=1), 'A', False),  # duplicate columns
    (base.rename(columns={'surr_value': 'unknown'}), 'A', False),
    (base.assign(surr_value=[1, 2, 3, 4]), 'A', False),
    (base.set_axis(pd.MultiIndex(levels=base.index.levels, codes=[[0, 0, 1, 1], [-1, 3, 1, 2]])), 'A', False),
    (base.set_axis(pd.MultiIndex.from_arrays([
        base.index.get_level_values(0), base.index.get_level_values(1).astype('float64')
    ])), 'A', False),
    (ReversedPrefix(base), 'A', False),
    (base, 'missing', False),
    (base, StringKey('A'), False),
    (base, ('A', -5), False),
    (base, ['A'], False),
]
for frame, key, native in cases:
    expected = outcome(fresh(live, frame, key).Calc[1].anchors)
    model = fresh(compiled, frame, key)
    with count_loc() as calls:
        actual = outcome(model.Calc[1].anchors)
    assert actual == expected, (frame, key, actual, expected)
    assert len(calls) == (0 if native else 1), (frame, key, calls)
    assert model.Data.table() is frame

def edit_value(frame):
    frame.iat[0, 0] = 42.0
    return 'A'


def replace_column(frame):
    frame['surr_value'] = [22.0, 23.0, 24.0, 25.0]
    return 'A'


def replace_index(frame):
    frame.index = pd.MultiIndex.from_tuples([('C', -7), ('C', 1), ('D', 2), ('D', 9)])
    return 'C'


def remove_column(frame):
    frame.rename(columns={'surr_value': 'removed'}, inplace=True)
    return 'A'


def shadow_get_loc(frame):
    frame.index = frame.index.copy()
    frame.index.get_loc = lambda key: slice(2, 4)
    return 'A'


for mutate, native in [(edit_value, True), (replace_column, True), (replace_index, True),
                       (remove_column, False), (shadow_get_loc, False)]:
    expected = outcome(fresh(live, base.copy(), mutate).Calc[1].anchors)
    with count_loc() as calls:
        actual = outcome(fresh(compiled, base.copy(), mutate).Calc[1].anchors)
    assert actual == expected, (mutate, actual, expected)
    assert len(calls) == (0 if native else 1), (mutate, calls)

for template in (live, compiled):
    model = fresh(template, base, 'A')
    selected = []

    def rebind(frame):
        selected.append(frame)
        model.Data.raw = base.assign(surr_value=[10.0, 20.0, 30.0, 40.0])
        return 'A'

    model.Data.key = (rebind,)
    assert model.Calc[1].anchors() == [(-5, 0.5), (2**40, 4.0)]
    assert len(selected) == 1 and selected[0] is base

item = fresh(compiled, base, 'A').Calc[1]
assert item.anchors() == [(-5, 0.5), (2**40, 4.0)]
assert type(item.anchors()) is list and all(type(pair) is tuple for pair in item.anchors())
assert item.other_view() == [(-5, 7.0), (2**40, 11.0)]
assert item.strict_zip() == item.anchors()
assert item.shadowed_zip(lambda x, y: reversed(list(zip(x, y)))) == item.anchors()[::-1]
assert type(item.row()) is pd.Series and item.row() is item.row()
'''], cwd=tmp_path, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
