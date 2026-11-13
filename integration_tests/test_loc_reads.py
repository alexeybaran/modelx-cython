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
    data = model.new_space('Data', refs={
        'pdn': pdn,
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
        'gap_in': ('def gap_in(t):\n    tbl = data.gap_table()\n'
                   '    return (1.0 if t in tbl.index else 0.0) + (10.0 if t not in tbl.index else 0.0)'),
        'walk': ('def walk(t):\n    for _, r in data.named_table().iterrows():\n        if r["v"] > 10.0 * t:\n'
                 '            return float(r["v"]) + len(r["note"])\n    return -5.0'),
        'edge_total': ('def edge_total():\n    return sum(gap(t) + gap_bounds(t) + gap_text(t) + gap_odd_column(t)'
                       ' + gap_unknown_column(t) + gap_default(t) + gap_text_method(t) + gap_row(t) + named(t) + multi(t) + flag(t)'
                       ' + deep(t) + walk(t) + gap_in(t)'
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
                              'gap_row', 'named', 'multi', 'flag', 'deep', 'walk', 'gap_in')}
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
                     'gap_in': 2}


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
    for name in _VARIANTS:
        body = _formula_c(py, c, name)
        # fuse_0: the C long long key overload; fuse_2: the object (text) key overload
        expected = '__pyx_fuse_0lookup_float' if name.startswith('int') else '__pyx_fuse_2lookup_float'
        assert expected in body, (name, 'typed read not reached')
