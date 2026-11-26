"""Prepared table reads through the compiled collect hook (version 1): values bit for bit, holds reset per policy.

The collect halves of the direct-route table tests (held tables, held key codes, held rows, index extremes and
membership): one model whose policies read a shared Data table per step, a Data row by a key fixed per item, the
model-point row through a row cell, an extreme of a prefix and a membership. Collect reuses one item per worker
and resets it between policies: each held table, code and row must be resolved again for the next policy (two
consecutive policies read different rows and codes, one policy's code is absent from its table, one policy is
missing from the model-point table and raises). Compiled at one and four workers equals the export and live.
"""
import json
import subprocess
import sys
import textwrap

import pytest
import modelx as mx
import pandas as pd

pytestmark = pytest.mark.skipif(
    not hasattr(mx.core.space.UserSpace, 'set_collect'), reason='modelx without collect')

PROBE = textwrap.dedent('''\
    import json
    from exported import mx_model as export
    from exported_cy import mx_model as compiled

    def flat(result):
        out = {'sum': {k: (dict(v) if isinstance(v, dict) else v) for k, v in result.sum._fields.items()}}
        out['individual'] = {str(c): r._fields for c, r in result.individual.items()}
        return out

    def error(call):
        try:
            call()
        except Exception as exc:
            return [type(exc).__name__, str(exc)]
        return None

    out = {'export': flat(export.Projection.collect())}
    export.Projection.keys_extra = (99,)
    export.Projection._mx_collect_cache.clear()
    out['export_error'] = error(export.Projection.collect)
    for workers in (1, 4):
        compiled._mx_collect_workers = workers
        compiled.Projection._mx_collect_cache.clear()
        compiled.Projection.keys_extra = ()
        out[workers] = {'value': flat(compiled.Projection.collect()), 'total': compiled.Book.total()}
        compiled.Projection.keys_extra = (99,)
        compiled.Projection._mx_collect_cache.clear()
        out[workers]['error'] = error(compiled.Projection.collect)
    print(json.dumps(out))
''')


def _model(tmp_path):
    keys = tuple(range(1, 31))
    points = pd.DataFrame({'sa': [100.0 * k for k in keys], 'age': [20 + k % 7 for k in keys],
                           'cost': ['A' if k % 3 else 'B' for k in keys], 'scen': [1 + k % 2 for k in keys]},
                          index=pd.Index(keys, name='policy_id'))
    model = mx.new_model('TableCollect')
    try:
        data = model.new_space('Data', refs={'POINTS': points, 'pd': pd})
        data.new_cells('points', 'lambda: POINTS')
        data.new_cells('costs', 'lambda: pd.DataFrame({"fee": [1.5, 2.5], "rate": [0.01, 0.02]}, '
                                'index=pd.Index(["A", "B"]))')
        data.new_cells('rates', 'lambda: pd.DataFrame({"r": [0.01 * i for i in range(10)]}, '
                                'index=pd.MultiIndex.from_product([[1, 2], [0, 1, 2, 3, 4]]))')
        data.new_cells('ages', 'lambda: pd.DataFrame({"q": [0.001 * a for a in range(20, 30)]}, '
                               'index=pd.Index(range(20, 30), name="age"))')
        proj = model.new_space('Projection', formula='lambda policy_id: None',
                               refs={'data': data, 'keys_extra': ()})
        proj.new_cells('keys', f'lambda: {keys!r} + keys_extra')
        proj.new_cells('point', 'lambda: data.points().loc[policy_id]')
        proj.new_cells('sa', 'lambda: float(point()["sa"])')
        proj.new_cells('cost_id', 'lambda: point()["cost"] if policy_id != 7 else "Z"')
        proj.new_cells('fee', '''def fee(t):
    if cost_id() not in data.costs().index:
        return 0.0
    row = data.costs().loc[cost_id()]
    return row["fee"] + row["rate"] * t
''')
        proj.new_cells('rate', '''def rate(t):
    tbl = data.rates()
    scen = int(point()["scen"])
    return tbl.loc[(scen, min(t, tbl.loc[scen].index.max())), "r"]
''')
        proj.new_cells('q', 'lambda t: data.ages().loc[min(int(point()["age"]) + t, data.ages().index.max()), "q"]')
        proj.new_cells('cf', 'lambda t: sa() * (q(t) + rate(t)) - fee(t)')
        proj.new_cells('pv', '''def pv():
    total = 0.0
    for t in range(8):
        total = total + cf(t) / (1.03 ** t)
    return total
''')
        proj.set_collect(over={'policy_id': proj.keys}, outputs='''lambda: {
    'individual': {'pv': pv(), 'fee': fee(1)},
    'sum': {'pv': pv(), 'cfs': {t: cf(t) for t in range(3)}}}''')
        book = model.new_space('Book', refs={'proj': proj})
        book.new_cells('total', 'lambda: proj.collect().sum.pv')
        live = proj.collect().sum.pv
        model.export(tmp_path / 'exported', locked_spaces=['Data'])
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'm.Book.total()\n'
        'for k in range(1, 31):\n    m.Projection[k].collect_outputs()\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'strict_lowering': True}), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py'],
                   cwd=tmp_path, check=True, capture_output=True)
    return live


def test_table_reads_through_collect(tmp_path):
    live = _model(tmp_path)
    generated = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    for read in ('.hold_row(', '.hold_code(', '.f64_row(', '.label_ll(', '.max_ll(', '.contains_1(', '.f64_c('):
        assert read in generated, read
    run = subprocess.run([sys.executable, '-c', PROBE], cwd=tmp_path, check=True, capture_output=True, text=True)
    out = json.loads(run.stdout)
    assert out['export']['sum']['pv'] == live
    for workers in ('1', '4'):
        got = out[workers]
        assert got['value'] == out['export'], workers           # bit for bit, domain order
        assert got['total'] == live
        assert got['error'] == out['export_error'] and got['error'][0] == 'KeyError', got['error']
