"""Views and walks of prepared tables through the compiled collect hook (version 1): values bit for bit, holds
reset per policy.

The collect halves of the direct-route view, walk and iterrows tests: one model whose policies read per step a view
of a prefix fixed per item (its codes and rows held per item), walk the view's index in a comprehension, read the
extreme of a second view, and walk a shared table with ``iterrows()`` once per item. Collect reuses one item per
worker and resets it between policies: each held code and view range must be resolved again for the next policy
(consecutive policies read different prefixes, one policy's prefix is absent from its table and that policy takes
another branch, one policy is missing from the model-point table and raises). Compiled at one and four workers
equals the export and live.
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
    points = pd.DataFrame({'sa': [100.0 * k for k in keys], 'sex': ['M' if k % 2 else 'F' for k in keys],
                           'scen': [1 + k % 3 if k != 7 else 9 for k in keys]},
                          index=pd.Index(keys, name='policy_id'))
    model = mx.new_model('ViewCollect')
    try:
        data = model.new_space('Data', refs={'POINTS': points, 'pd': pd})
        data.new_cells('points', 'lambda: POINTS')
        data.new_cells('grid', 'lambda: pd.DataFrame({"q": [0.001 * i for i in range(12)]}, '
                               'index=pd.MultiIndex.from_product([["M", "F"], [30, 31, 33, 35, 36, 40]]))')
        data.new_cells('scen', 'lambda: pd.DataFrame({"r": [0.01 * i for i in range(9)]}, '
                               'index=pd.MultiIndex.from_product([[1, 2, 3], [0, 2, 5]]))')
        data.new_cells('scen_ids', 'lambda: (1, 2, 3)')
        data.new_cells('bands', 'lambda: pd.DataFrame({"lo": [0.0, 1000.0, 2000.0], "hi": [1000.0, 2000.0, 1e9], '
                                '"v": [1.0, 1.1, 1.2]})')
        proj = model.new_space('Projection', formula='lambda policy_id: None',
                               refs={'data': data, 'keys_extra': ()})
        proj.new_cells('keys', f'lambda: {keys!r} + keys_extra')
        proj.new_cells('point', 'lambda: data.points().loc[policy_id]')
        proj.new_cells('sa', 'lambda: float(point()["sa"])')
        proj.new_cells('sex', 'lambda: point()["sex"]')
        proj.new_cells('scen_id', 'lambda: int(point()["scen"])')
        proj.new_cells('q', '''def q(t):
    grid = data.grid().loc[sex()]
    ages = [a for a in grid.index if a <= 30 + t]
    return float(grid.loc[max(ages), "q"])
''')
        proj.new_cells('r', '''def r(t):
    if scen_id() not in data.scen_ids():
        return 0.0
    sub = data.scen().loc[scen_id()]
    return float(sub.loc[max(m for m in sub.index if m <= t), "r"]) + int(sub.index.max())
''')
        proj.new_cells('band', '''def band():
    for _, b in data.bands().iterrows():
        if b["lo"] <= sa() < b["hi"]:
            return float(b["v"])
    return 0.0
''')
        proj.new_cells('cf', 'lambda t: sa() * (q(t) + r(t)) * band()')
        proj.new_cells('pv', '''def pv():
    total = 0.0
    for t in range(8):
        total = total + cf(t) / (1.03 ** t)
    return total
''')
        proj.set_collect(over={'policy_id': proj.keys}, outputs='''lambda: {
    'individual': {'pv': pv(), 'band': band()},
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


def test_views_and_walks_through_collect(tmp_path):
    live = _model(tmp_path)
    generated = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    for read in ('.view_start(', '.view_stop(', '.label_ll(', '.whole_row(', '.f64_row(', '.f64_c(', '.hold_code('):
        assert read in generated, read
    run = subprocess.run([sys.executable, '-c', PROBE], cwd=tmp_path, check=True, capture_output=True, text=True)
    out = json.loads(run.stdout)
    assert out['export']['sum']['pv'] == live
    for workers in ('1', '4'):
        got = out[workers]
        assert got['value'] == out['export'], workers           # bit for bit, domain order
        assert got['total'] == live
        assert got['error'] == out['export_error'] and got['error'][0] == 'KeyError', got['error']
