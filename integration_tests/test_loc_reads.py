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
    proj = model.new_space('Projection', refs={'data': data, 'pdn': pdn, 'plan': 'B'})
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
        'total': ('def total():\n    return sum(rate(t) + capped(t + 5) + band(t) + band_int(t) + premium(1 + t % 3)'
                  ' + premium_full_key(1 + t % 3) + column_by_name(t) + column_computed(t) for t in range(5))'),
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
    assert cell('premium_full_key')['loc_reads_untyped'] == ['full key into a grouped table: no native grouped read']
    assert cell('column_by_name')['loc_reads_typed'] == 1      # a local bound once to a literal
    assert cell('column_computed')['loc_reads_untyped'] == ['column is not a string literal']
