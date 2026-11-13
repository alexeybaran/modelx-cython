"""Pandas-spelled table reads compile to typed reads where the traced schema allows it.

The model mixes the forms the rewrite must type and those it must leave alone, and checks
values in live modelx, exported Python and compiled code, plus the admission report.
"""
import json
import subprocess
import sys

import modelx as mx
import pandas as pd
from modelx_helpers import pdn


def _model():
    model = mx.new_model('LocReads')
    data = model.new_space('Data', refs={
        'pdn': pdn,
        'mort': pd.DataFrame({'mort_rate': [0.01 * i for i in range(1, 11)], 'band': list(range(10))},
                             index=pd.Index(range(30, 40), name='age')),
        'prem': pd.DataFrame({'plan': ['A'] * 3 + ['B'] * 3, 'year': [1, 2, 3] * 2,
                              'premium_pp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}).set_index(['plan', 'year'])})
    data.new_cells('mort_table', 'lambda: pdn.table(mort)')
    data.new_cells('prem_table', 'lambda: pdn.table(prem, columns=("premium_pp",), groups=("plan",))')
    import math
    proj = model.new_space('Projection', refs={'data': data, 'pdn': pdn, 'plan': 'B', 'math': math, 'pd': pd})
    formulas = {
        'rate': 'def rate(t):\n    return float(data.mort_table().loc[30 + t, "mort_rate"])',
        'capped': ('def capped(t):\n    tbl = data.mort_table()\n    tbl = data.mort_table()\n'
                   '    return tbl.loc[min(30 + t, tbl.index.max()), "mort_rate"]'),
        'band': 'def band(t):\n    return float(data.mort_table().loc[30 + t, "band"])',
        'band_int': 'def band_int(t):\n    return int(data.mort_table().loc[30 + t, "band"])',
        'premium': ('def premium(t):\n    sched = data.prem_table().slice(plan)\n'
                    '    return float(sched.loc[t, "premium_pp"])'),
        'premium_full_key': 'def premium_full_key(t):\n    return float(data.prem_table().loc[(plan, t), "premium_pp"])',
        'column_by_name': 'def column_by_name(t):\n    name = "mort_rate"\n    return float(data.mort_table().loc[30 + t, name])',
        'column_computed': 'def column_computed(t):\n    return float(data.mort_table().loc[30 + t, "mort_" + "rate"])',
        # guards: a key reassigned after the row is taken, a two-argument math call, isna on a Series
        'reassigned': ('def reassigned(t):\n    k = 30 + t\n    row = data.mort_table().loc[k]\n    k = k + 1\n'
                       '    return float(row["mort_rate"])'),
        'log_base': 'def log_base(t):\n    return math.log(8.0 + t, 2.0) + math.log(2.0 + t)',
        'isna_count': 'def isna_count(t):\n    return float(pd.isna(pd.Series([1.0, float("nan"), t])).sum()) + pd.isna(float("nan"))',
        'total': ('def total():\n    return sum(rate(t) + capped(t + 5) + band(t) + band_int(t) + premium(1 + t % 3)'
                  ' + premium_full_key(1 + t % 3) + column_by_name(t) + column_computed(t) + reassigned(t) + log_base(t) + isna_count(t) for t in range(5))'),
    }
    for name, source in formulas.items():
        proj.new_cells(name, source)
    return model


def test_loc_reads_three_modes(tmp_path):
    model = _model()
    try:
        live = model.Projection.total()
        model.export(tmp_path / 'exported', locked_spaces=['Data'])
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\nm.Projection.total()\n',
                                        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    for package in ('exported', 'exported_cy'):
        got = subprocess.run([sys.executable, '-c', f'from {package} import mx_model as m; print(repr(m.Projection.total()))'],
                             cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
        assert float(got) == live, (package, got, live)
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    cell = lambda name: admission[f'exported_cy._mx_classes._c_Projection._f_{name}']
    assert cell('rate')['loc_reads_typed'] == 1
    assert cell('capped')['loc_reads_typed'] == 2          # the read and index.max()
    assert cell('band')['loc_reads_typed'] == 1            # float() of an integer column
    assert cell('premium')['loc_reads_typed'] == 1         # through slice() of a grouped table
    assert cell('band_int')['loc_reads_untyped'] == ["column 'band' has traced kind int"]
    assert cell('premium_full_key')['loc_reads_untyped'] == [
        'full key into a grouped table: declare the table without groups= to read it by full key']
    assert cell('column_by_name')['loc_reads_typed'] == 1      # a local bound once to a literal
    assert cell('column_computed')['loc_reads_untyped'] == ['column is not a string literal']


def _edge_model():
    model = mx.new_model('LocReadEdges')
    import enum
    data = model.new_space('Data', refs={
        'pdn': pdn, 'enum': enum,
        # integer keys with gaps, a text column holding numbers and words, a non-identifier column
        'gaps': pd.DataFrame({'v': [0.5, 1.5, 2.5, 3.5], 'txt': ['1.25', 'abc', '2', 'x'],
                              'sc per 1000': [9.0, 8.0, 7.0, 6.0], 'n': [1, 2, 3, 4]},
                             index=pd.Index([1, 2, 5, 9], name='k')),
        'named': pd.DataFrame({'v': [10.0, 20.0, 30.0], 'note': ['a', 'b', 'c']},
                              index=pd.Index(['UL9', 'UL5', 'X'], name='id')),
        'multi': pd.DataFrame({'plan': ['A', 'A', 'B', 'B'], 'year': [1, 2, 1, 2], 'v': [1.0, 2.0, 3.0, 4.0],
                               'note': ['p', 'q', 'r', 's']}).set_index(['plan', 'year']),
        'flags': pd.DataFrame({'v': [100.0, 200.0], 'note': ['t', 'f']}, index=pd.Index([True, False], name='f')),
        'deep': pd.DataFrame({'a': ['x', 'x', 'y', 'y'], 'b': [1, 1, 2, 2], 'c': ['p', 'q', 'p', 'q'],
                              'year': [1, 2, 1, 2], 'v': [1.5, 2.5, 3.5, 4.5]}).set_index(['a', 'b', 'c', 'year'])})
    data.new_cells('gap_table', 'lambda: pdn.table(gaps)')
    data.new_cells('named_table', 'lambda: pdn.table(named)')
    data.new_cells('multi_table', 'lambda: pdn.table(multi)')
    data.new_cells('flag_table', 'lambda: pdn.table(flags)')
    data.new_cells('deep_table', 'lambda: pdn.table(deep, columns=("v",), groups=("a", "b", "c"))')
    # key types: enum members and other subclass objects as key parts must give the live values in every mode
    data.new_cells('ienum', 'lambda: enum.IntEnum("I", {"A": 1, "B": 2, "C": 5})')
    data.new_cells('senum', 'lambda: enum.StrEnum("S", {"U": "UL9", "W": "nope"})')
    data.new_cells('penum', 'lambda: enum.Enum("P", {"A": 2})')
    data.new_cells('hashed', 'lambda: type("H", (int,), {"__module__": "edges", "__hash__": lambda self: int.__hash__(self)})')
    data.new_cells('eqstr', 'lambda: type("E", (str,), {"__module__": "edges", "__eq__": lambda self, o: str.__eq__(self, o), '
                            '"__hash__": lambda self: str.__hash__(self)})')
    proj = model.new_space('Projection', refs={'data': data, 'pdn': pdn, 'ids': ('UL9', 'X', 'nope')})

    def probe(name, expression, args='t'):
        return (f'def {name}({args}):\n    try:\n        return {expression}\n'
                f'    except KeyError:\n        return -1.0\n    except ValueError:\n        return -2.0\n'
                f'    except IndexError:\n        return -3.0\n')
    formulas = {
        'gap': probe('gap', 'float(data.gap_table().loc[t, "v"])'),
        'gap_bounds': probe('gap_bounds', 'float(data.gap_table().loc[min(max(t, int(data.gap_table().index.min())), int(data.gap_table().index.max())), "n"])'),
        'gap_text': probe('gap_text', 'float(data.gap_table().loc[t, "txt"])'),
        'gap_odd_column': probe('gap_odd_column', 'float(data.gap_table().loc[t, "sc per 1000"])'),
        'gap_unknown_column': probe('gap_unknown_column', 'float(data.gap_table().loc[t, "nope"])'),
        'gap_default': 'def gap_default(t):\n    return data.gap_table().lookup_float(t, "v", -7.0)',
        'gap_text_method': probe('gap_text_method', 'data.gap_table().lookup_float(t, "txt")'),
        'gap_row': probe('gap_row', 'float(data.gap_table().loc[t]["v"]) + data.gap_table().loc[t]["n"]'),
        'named': probe('named', 'float(data.named_table().loc[ids[t % 3], "v"])'),
        'multi': probe('multi', 'float(data.multi_table().loc[("A" if t % 2 else "B", t % 3), "v"])'),
        'flag': probe('flag', 'float(data.flag_table().loc[t % 2 == 0, "v"])'),
        # a full key into three group levels and a leaf: missing parts at every level, bool parts
        'deep': probe('deep', 'float(data.deep_table().loc[(("x", "y", "x", "z")[t % 4], (1, 2, True, 1)[t % 4], '
                              '"pq"[t % 2], 1 + t % 2), "v"])'),
        'key_int_enum': probe('key_int_enum', 'float(data.gap_table().loc[data.ienum()((1, 2, 5)[t % 3]), "v"])'),
        'key_str_enum': probe('key_str_enum', 'float(data.named_table().loc[data.senum()(("UL9", "nope")[t % 2]), "v"])'),
        'key_plain_enum': probe('key_plain_enum', 'float(data.gap_table().loc[data.penum()(2), "v"])'),
        'key_hash_override': probe('key_hash_override', 'float(data.gap_table().loc[data.hashed()(1 + t % 3), "v"])'),
        'key_eq_override': probe('key_eq_override', 'float(data.named_table().loc[data.eqstr()("UL5"), "v"])'),
        'in_kinds': ('def in_kinds(t):\n    g = data.gap_table()\n    m = data.multi_table()\n'
                     '    return ((1.0 in g.index) + 2 * (True in g.index) + 4 * (3 in g.index) + 8 * (("A",) in m.index)'
                     ' + 16 * (("A", 1, 2) in m.index) + 32 * (("A", 1) in m.index) + 64 * (data.ienum()(5) in g.index)'
                     ' + 128 * (data.senum()("UL9") in data.named_table().index) + 256 * ("UL9" in g.index))'),
        'at_read': probe('at_read', 'float(data.gap_table().at[t, "v"])'),
        'at_varcol': probe('at_varcol', 'float(data.gap_table().at[t, ("v", "sc per 1000")[t % 2]])'),
        'at_multi': probe('at_multi', 'float(data.multi_table().at[("A" if t % 2 else "B", t % 3), "v"])'),
        'prev': probe('prev', 'data.gap_table().previous(t, ("v", "sc per 1000", 0, -2)[t % 4])'),
        'gap_in': ('def gap_in(t):\n    tbl = data.gap_table()\n'
                   '    return (1.0 if t in tbl.index else 0.0) + (10.0 if t not in tbl.index else 0.0)'),
        'walk': ('def walk(t):\n    for _, r in data.named_table().iterrows():\n        if r["v"] > 10.0 * t:\n'
                 '            return float(r["v"]) + len(r["note"])\n    return -5.0'),
        'edge_total': ('def edge_total():\n    return sum(gap(t) + gap_bounds(t) + gap_text(t) + gap_odd_column(t)'
                       ' + gap_unknown_column(t) + gap_default(t) + gap_text_method(t) + gap_row(t) + named(t) + multi(t) + flag(t)'
                       ' + deep(t) + walk(t) + gap_in(t) + key_int_enum(t) + key_str_enum(t) + key_plain_enum(t)'
                       ' + key_hash_override(t) + key_eq_override(t) + in_kinds(t) + prev(t) + at_read(t)'
                       ' + at_varcol(t) + at_multi(t)'
                       ' for t in range(11))'),
    }
    for name, source in formulas.items():
        proj.new_cells(name, source)
    return model


def test_loc_read_edges_three_modes(tmp_path):
    """Gaps, unknown keys and columns, non-numeric cells, defaults, rows and non-integer keys."""
    model = _edge_model()
    try:
        live = model.Projection.edge_total()
        parts = {name: [model.Projection.cells[name](t) for t in range(11)]
                 for name in ('gap', 'gap_text', 'gap_text_method', 'gap_default', 'gap_unknown_column',
                              'gap_row', 'named', 'multi', 'flag', 'deep', 'walk', 'gap_in', 'key_int_enum',
                              'key_str_enum', 'key_plain_enum', 'key_hash_override', 'key_eq_override',
                              'in_kinds', 'prev', 'at_read', 'at_varcol', 'at_multi')}
        model.export(tmp_path / 'exported', locked_spaces=['Data'])
    finally:
        model.close()
    # live semantics the compiled reads must reproduce
    assert parts['gap'][:6] == [-1.0, 0.5, 1.5, -1.0, -1.0, 2.5]           # keys in a gap raise KeyError
    assert parts['gap_text'][:3] == [-1.0, 1.25, -2.0]                     # float('abc') raises ValueError
    assert parts['gap_text_method'][:3] == [-1.0, 1.25, -2.0]
    assert parts['gap_default'][:4] == [-7.0, 0.5, 1.5, -7.0]
    assert set(parts['gap_unknown_column']) == {-1.0}                      # pandas: unknown column is a KeyError
    assert parts['named'][:3] == [10.0, 30.0, -1.0]
    assert parts['deep'][:4] == [1.5, 4.5, 1.5, -1.0]       # a bool group part matches the Record label 1
    assert parts['walk'][:4] == [11.0, 21.0, 31.0, -5.0]
    assert parts['gap_in'][:6] == [10.0, 1.0, 1.0, 10.0, 10.0, 1.0]
    assert parts['key_int_enum'][:3] == [0.5, 1.5, 2.5]
    assert parts['key_str_enum'][:2] == [10.0, -1.0]
    assert parts['key_plain_enum'][0] == -1.0
    assert parts['key_hash_override'][:3] == [0.5, 1.5, -1.0]
    assert parts['key_eq_override'][0] == 20.0
    assert parts['in_kinds'][0] == 1 + 32 + 64 + 128          # True is not the label 1 in a pandas int index
    assert parts['prev'][:6] == [-2.0, 9.0, 1.5, 8.0, 1.5, 7.0]     # last key <= t; variable column
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\nm.Projection.edge_total()\n',
                                        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    names = list(parts)
    script = ('from {0} import mx_model as m; P = m.Projection; '
              'print(repr([P.edge_total()] + [[getattr(P, n)(t) for t in range(11)] for n in {1!r}]))')
    for package in ('exported', 'exported_cy'):
        got = subprocess.run([sys.executable, '-c', script.format(package, names)],
                             cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
        assert got == repr([live] + [parts[n] for n in names]), (package, got)
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    typed = {name: admission[f'exported_cy._mx_classes._c_Projection._f_{name}']['loc_reads_typed']
             for name in ('gap', 'gap_bounds', 'gap_odd_column', 'named', 'multi', 'flag', 'deep', 'gap_in')}
    assert typed == {'gap': 1, 'gap_bounds': 3, 'gap_odd_column': 1, 'named': 1, 'multi': 1, 'flag': 1, 'deep': 0,
                     'gap_in': 0}
    # .at[key, column] is the same read as .loc[key, column]
    py_classes = tmp_path / 'exported_cy/_mx_classes.py'
    for loc_name, at_name, call in (('gap', 'at_read', '.lookup_float('), ('multi', 'at_multi', 'lookup_float_2(')):
        assert call in _formula_py(py_classes, loc_name) and call in _formula_py(py_classes, at_name), at_name
    assert parts['at_read'] == parts['gap'] and parts['at_multi'] == parts['multi']
    # no experiment switch: a scalar membership test is left as written for the statement recognisers
    gap_in = _formula_py(tmp_path / 'exported_cy/_mx_classes.py', 'gap_in')
    assert 'native_contains' not in gap_in and ' in tbl.index' in gap_in


# Equivalent spellings of one integer clamp read and one text-key read, compiled with the statement
# recognisers off: each must end in the same typed NativeTable read (C integer key for the clamp).
_VARIANTS = {
    # integer clamp read
    'int_nested': 'def int_nested(t):\n    tbl = data.mort_table()\n'
                  '    return float(tbl.loc[min(max(t, int(tbl.index.min())), int(tbl.index.max())), "mort_rate"])',
    'int_hoisted': 'def int_hoisted(t):\n    tbl = data.mort_table()\n    lo = int(tbl.index.min())\n'
                   '    hi = int(tbl.index.max())\n    return float(tbl.loc[min(max(t, lo), hi), "mort_rate"])',
    'int_reordered': 'def int_reordered(t):\n    m = data.mort_table()\n    upper = int(m.index.max())\n'
                     '    lower = int(m.index.min())\n    k = max(t, lower)\n    k2 = min(k, upper)\n'
                     '    return float(m.loc[k2, "mort_rate"])',
    'int_inline': 'def int_inline(t):\n    return float(data.mort_table().loc[min(max(t, int(data.mort_table().index.min())), '
                  'int(data.mort_table().index.max())), "mort_rate"])',
    'int_row': 'def int_row(t):\n    tbl = data.mort_table()\n    a = min(max(t, int(tbl.index.min())), int(tbl.index.max()))\n'
               '    row = tbl.loc[a]\n    return float(row["mort_rate"])',
    # text-key read
    'text_inline': 'def text_inline(t):\n    return float(data.named_table().loc[pid(), "v"])',
    'text_alias': 'def text_alias(t):\n    tbl = data.named_table()\n    k = pid()\n    return float(tbl.loc[k, "v"])',
    'text_row': 'def text_row(t):\n    row = data.named_table().loc[pid()]\n    return float(row["v"])',
    # membership: with an experiment switch it is the provider's native_contains
    'int_in': 'def int_in(t):\n    tbl = data.mort_table()\n    return 1.0 if t in tbl.index else 0.0',
}


def _variants_model(n_t=11):
    model = mx.new_model('SpellingVariants')
    data = model.new_space('Data', refs={
        'pdn': pdn,
        'mort': pd.DataFrame({'mort_rate': [0.01 * i for i in range(1, 11)], 'note': list('abcdefghij')},
                             index=pd.Index(range(30, 40), name='age')),
        'named': pd.DataFrame({'v': [10.0, 20.0, 30.0], 'note': ['a', 'b', 'c']},
                              index=pd.Index(['UL9', 'UL5', 'X'], name='id'))})
    data.new_cells('mort_table', 'lambda: pdn.table(mort)')
    data.new_cells('named_table', 'lambda: pdn.table(named)')
    proj = model.new_space('Projection', refs={'data': data, 'pdn': pdn, 'n_t': n_t})
    proj.new_cells('pid', 'lambda: "UL5"')
    for name, source in _VARIANTS.items():
        proj.new_cells(name, source)
    proj.new_cells('total', 'def total():\n    return sum(' + ' + '.join(f'{n}(t)' for n in _VARIANTS)
                   + ' for t in range(25, 25 + n_t))')
    return model


def _formula_c(classes_py, classes_c, cell):
    import re
    py = classes_py.read_text()
    dense = re.search(r"def _f_%s\(self[^\n]*\n\s+return (_mx_dense_formula_\d+)\(" % cell, py).group(1)
    c = classes_c.read_text()
    start = re.search(r"\n(?:static )?(?:CYTHON_INLINE )?\w+ __pyx_f_\w+_%s\([^;{]*\{" % dense, c).start()
    return c[start:c.index('\n}\n', start)]


def test_spelling_variants_recognisers_off(tmp_path):
    import os
    model = _variants_model()
    try:
        live = model.Projection.total()
        model.export(tmp_path / 'exported', locked_spaces=['Data'])
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\nm.Projection.total()\n', encoding='utf-8')
    env = dict(os.environ, MX_CY_OBSERVED_TABLES='1', MX_CY_NO_TABLE_MATCHERS='1', MX_CY_PLAIN_READS='1')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, env=env)
    for package in ('exported', 'exported_cy'):
        got = subprocess.run([sys.executable, '-c', f'from {package} import mx_model as m; print(repr(m.Projection.total()))'],
                             cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
        assert float(got) == live, (package, got, live)
    py, c = tmp_path / 'exported_cy/_mx_classes.py', tmp_path / 'exported_cy/_mx_classes.c'
    assert 'native_contains(' in _formula_py(py, 'int_in')
    for name in _VARIANTS:
        if name == 'int_in':
            continue
        body = _formula_c(py, c, name)
        # fuse_0: the C long long key overload; fuse_2: the object (text) key overload
        expected = '__pyx_fuse_0lookup_float' if name.startswith('int') else '__pyx_fuse_2lookup_float'
        assert expected in body, (name, 'typed read not reached')


def _formula_py(classes_py, cell):
    import re
    src = classes_py.read_text()
    dense = re.search(r"def _f_%s\(self[^\n]*\n\s+return (_mx_dense_formula_\d+)\(" % cell, src).group(1)
    start = src.index('def %s(' % dense)
    end = src.find('\n@', start)
    return src[start:end]


def _int_key_model():
    import itertools
    model = mx.new_model('IntKeys')
    two = pd.DataFrame([(a, b, 10.0 * a + b) for a in (1, 2) for b in (0, 1, 2)],
                       columns=['a', 'b', 'v']).set_index(['a', 'b'])
    three = pd.DataFrame([(a, b, c, 100.0 * a + 10 * b + c) for a in (1, 2) for b in (0, 1, 2) for c in (5, 7)],
                         columns=['a', 'b', 'c', 'v']).set_index(['a', 'b', 'c'])
    five = pd.DataFrame([(*k, float(sum(k))) for k in itertools.product((1, 2), (0, 1), (3, 4), (5, 6), (8, 9))],
                        columns=['a', 'b', 'c', 'd', 'e', 'v']).set_index(['a', 'b', 'c', 'd', 'e'])
    mixed = pd.DataFrame({'p': ['x', 'y'], 'b': [1, 2], 'v': [1.0, 2.0]}).set_index(['p', 'b'])
    bools = pd.DataFrame({'f': [True, False, True], 'b': [0, 1, 1], 'v': [5.0, 6.0, 7.0]}).set_index(['f', 'b'])
    zero_one = pd.DataFrame({'a': [0, 1, 1], 'b': [0, 1, 2], 'v': [8.0, 9.0, 10.0]}).set_index(['a', 'b'])
    data = model.new_space('Data', refs={'pdn': pdn, 'two': two, 'three': three, 'five': five, 'mixed': mixed,
                                         'bools': bools, 'zero_one': zero_one})
    for name in ('two', 'three', 'five', 'mixed', 'bools', 'zero_one'):
        data.new_cells(name + '_table', f'lambda: pdn.table({name})')
    proj = model.new_space('Projection', refs={'data': data})
    proj.new_cells('ka', 'def ka(t):\n    return 1 + t % 3')        # 3 is absent at the first level
    proj.new_cells('kb', 'def kb(t):\n    return t % 3')
    proj.new_cells('flag', 'def flag(t):\n    return t % 2 == 0')
    def probe(name, body):
        return (f'def {name}(t):\n    try:\n' + ''.join(f'        {line}\n' for line in body.splitlines())
                + '    except KeyError:\n        return -1.0\n')
    formulas = {
        'k2_literal': probe('k2_literal', 'return float(data.two_table().loc[(ka(t), kb(t)), "v"])'),
        'k2_local': probe('k2_local', 'key = (ka(t), kb(t))\nreturn float(data.two_table().loc[key, "v"])'),
        'k2_hoisted': probe('k2_hoisted', 'key = (ka(t), kb(t))\ntbl = data.two_table()\nscale = 2.0\n'
                                          'return scale * float(tbl.loc[key, "v"])'),
        'k2_renamed': probe('k2_renamed', 'other = 1.0\npair = (ka(t), kb(t))\nreturn other * float(data.two_table().loc[pair, "v"])'),
        'k2_in_and_read': ('def k2_in_and_read(t):\n    key = (ka(t), kb(t))\n    tbl = data.two_table()\n'
                           '    return float(tbl.loc[key, "v"]) if key in tbl.index else -5.0'),
        'k2_in': ('def k2_in(t):\n    return 1.0 if (ka(t), kb(t)) in data.two_table().index else 0.0'),
        'k3_local': probe('k3_local', 'key = (ka(t), kb(t), 5 + 2 * (t % 2))\nreturn float(data.three_table().loc[key, "v"])'),
        'k5_literal': probe('k5_literal', 'return float(data.five_table().loc[(ka(t), 1, 3, 5, 8), "v"])'),
        'k2_text_level': probe('k2_text_level', 'return float(data.mixed_table().loc[(ka(t), kb(t)), "v"])'),
        'k2_str_part': probe('k2_str_part', 'return float(data.mixed_table().loc[(("x", "y")[t % 2], 1 + t % 3), "v"])'),
        'k2_pyint_part': probe('k2_pyint_part', 'n = int(("1", "2", "3")[t % 3])\nreturn float(data.two_table().loc[(n, kb(t)), "v"])'),
        # bool parts: a bool-traced cell, a comparison and a bool local, on an int level and on a bool level
        'b_cell_int': probe('b_cell_int', 'return float(data.zero_one_table().loc[(flag(t), kb(t)), "v"])'),
        'b_cmp_int': probe('b_cmp_int', 'return float(data.zero_one_table().loc[(t > 3, kb(t)), "v"])'),
        'b_local_int': probe('b_local_int', 'z = t > 5\nreturn float(data.zero_one_table().loc[(z, kb(t)), "v"])'),
        'b_in_int': 'def b_in_int(t):\n    return 1.0 if (flag(t), kb(t)) in data.zero_one_table().index else 0.0',
        'b_cell_bool': probe('b_cell_bool', 'return float(data.bools_table().loc[(flag(t), kb(t) % 2), "v"])'),
        'b_cmp_bool': probe('b_cmp_bool', 'return float(data.bools_table().loc[(t > 3, kb(t) % 2), "v"])'),
    }
    names = tuple(formulas)
    formulas['total'] = 'def total():\n    return sum(' + ' + '.join(f'{n}(t)' for n in names) + ' for t in range(12))'
    for name, source in formulas.items():
        proj.new_cells(name, source)
    return model, names


def test_int_tuple_keys_three_modes(tmp_path):
    """Tuple-expression keys: unpacked parts, C integers only when known non-bool integers."""
    import os
    model, names = _int_key_model()
    try:
        live = model.Projection.total()
        parts = {n: [model.Projection.cells[n](t) for t in range(12)] for n in names}
        model.export(tmp_path / 'exported', locked_spaces=['Data'])
    finally:
        model.close()
    assert parts['k2_literal'][:4] == [10.0, 21.0, -1.0, 10.0]           # (3, 2) is absent: KeyError
    assert parts['k2_in'][:3] == [1.0, 1.0, 0.0]
    assert parts['k2_in_and_read'][:3] == [10.0, 21.0, -5.0]
    assert set(parts['k2_text_level']) == {-1.0}                         # int parts, text level: no row
    assert parts['b_cell_int'][:2] == [-1.0, -1.0]                       # pandas int level: True is not 1
    assert set(parts['b_in_int']) == {0.0}
    assert parts['b_cell_bool'][:2] == [5.0, 6.0]
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\nm.Projection.total()\n', encoding='utf-8')
    env = dict(os.environ, MX_CY_OBSERVED_TABLES='1', MX_CY_PLAIN_READS='1', MX_CY_INFER_TYPES='1')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, env=env)
    script = ('from {0} import mx_model as m; P = m.Projection; '
              'print(repr([P.total()] + [[getattr(P, n)(t) for t in range(12)] for n in {1!r}]))')
    for package in ('exported', 'exported_cy'):
        got = subprocess.run([sys.executable, '-c', script.format(package, names)],
                             cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
        assert got == repr([live] + [parts[n] for n in names]), (package, got)
    py = tmp_path / 'exported_cy/_mx_classes.py'
    calls = {'k2_literal': 'lookup_float_2(', 'k2_local': 'lookup_float_2(', 'k2_hoisted': 'lookup_float_2(',
             'k2_renamed': 'lookup_float_2(', 'k2_in_and_read': 'native_contains_2(', 'k2_in': 'native_contains_2(',
             'k3_local': 'lookup_float_3(', 'k5_literal': 'lookup_float_5(', 'k2_text_level': 'lookup_float_2(',
             'k2_str_part': 'lookup_float_2(', 'k2_pyint_part': 'lookup_float_2(', 'b_cell_int': 'lookup_float_2(',
             'b_cmp_int': 'lookup_float_2(', 'b_local_int': 'lookup_float_2(', 'b_in_int': 'native_contains_2(',
             'b_cell_bool': 'lookup_float_2(', 'b_cmp_bool': 'lookup_float_2('}
    for name, call in calls.items():
        body = _formula_py(py, name)
        assert call in body, (name, call)
        boxed = '_mx_cy.cast(object' in body
        assert boxed == name.startswith(('b_', 'k2_str', 'k2_pyint', 'k3_local')), (name, 'part boxing')
    assert 'lookup_float_2(' in _formula_py(py, 'k2_in_and_read')
    c = tmp_path / 'exported_cy/_mx_classes.c'
    # C integer parts reach the all-integer specialisations; a bool part arrives as an object
    assert '__pyx_fuse_0_0__pyx_f_' in _formula_c(py, c, 'k2_literal')
    assert '__pyx_fuse_0_0_0_0_0__pyx_f_' in _formula_c(py, c, 'k5_literal')
    assert '__pyx_fuse_1_0__pyx_f_' in _formula_c(py, c, 'b_cell_int')
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    keys = admission['exported_cy._mx_classes._c_Projection._f_k2_literal']['loc_reads_keys']
    assert keys == ['loc[key, column], 2 parts: int (cell ka returns long long); int (cell kb returns long long)']


def _record_field_model():
    model = mx.new_model('RecordFieldReads')
    points = pd.DataFrame({'a': [1.5, -2.5], 'b': [3.0, 4.0], 'tag': ['x', 'y']}, index=pd.Index([1, 2], name='point_id'))
    data = model.new_space('Data', refs={'pdn': pdn, 'points': points})
    data.new_cells('point_table', 'lambda: pdn.table(points)')
    proj = model.new_space('Projection', refs={'data': data})
    proj.new_cells('mp', 'def mp():\n    return data.point_table().loc[1 + len(str(data))  % 1]')
    forms = {
        'f_ret': 'def f_ret(t):\n    return float(mp()["a"])',
        'f_assign': 'def f_assign(t):\n    x = mp()["a"]\n    return float(x) + t',
        'f_arith': 'def f_arith(t):\n    return mp()["a"] * 2.0 + mp()["b"] + t',
        'f_cond': 'def f_cond(t):\n    if mp()["a"] > 0:\n        return 1.0 + t\n    return 0.0',
        'f_alias': 'def f_alias(t):\n    rec = mp()\n    return rec["b"] - t',
        'f_missing': 'def f_missing(t):\n    try:\n        return float(mp()["nope"])\n    except KeyError:\n        return -1.0',
    }
    for name, source in forms.items():
        proj.new_cells(name, source)
    proj.new_cells('total', 'def total():\n    return sum(' + ' + '.join(f'{n}(t)' for n in forms) + ' for t in range(4))')
    return model, tuple(forms)


def test_record_field_reads_any_position(tmp_path):
    """With the recognisers off, rec["field"] on a traced Record cell is the provider member read in any position."""
    import os
    model, names = _record_field_model()
    try:
        live = model.Projection.total()
        parts = {n: [model.Projection.cells[n](t) for t in range(4)] for n in names}
        model.export(tmp_path / 'exported', locked_spaces=['Data'])
    finally:
        model.close()
    assert parts['f_missing'] == [-1.0] * 4
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\nm.Projection.total()\n', encoding='utf-8')
    env = dict(os.environ, MX_CY_OBSERVED_TABLES='1', MX_CY_PLAIN_READS='1', MX_CY_NO_TABLE_MATCHERS='1')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, env=env)
    script = ('from {0} import mx_model as m; P = m.Projection; '
              'print(repr([P.total()] + [[getattr(P, n)(t) for t in range(4)] for n in {1!r}]))')
    for package in ('exported', 'exported_cy'):
        got = subprocess.run([sys.executable, '-c', script.format(package, names)],
                             cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
        assert got == repr([live] + [parts[n] for n in names]), (package, got)
    py = tmp_path / 'exported_cy/_mx_classes.py'
    for name in names:
        body = _formula_py(py, name)
        assert 'read_record_item(' in body and '()["' not in body and 'rec["' not in body, name
