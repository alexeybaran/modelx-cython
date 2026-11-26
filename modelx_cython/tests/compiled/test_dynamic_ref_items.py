"""An ItemSpace whose Space formula sets dynamic References, exported by a modelx that exports them.

The export applies the formula after the walk (``_mx_root._mx_apply_formula(...)``). The compiled fast item
construction keeps that call, in the same place, on the typed root; the items' values equal the export's.
"""
import subprocess
import sys
import textwrap

import pytest

import modelx as mx
from modelx.export import exporter


@pytest.mark.skipif(not hasattr(exporter, 'dynamic_ref_formula'),
                    reason='this modelx does not export dynamic References')
def test_dynamic_references_keep_the_fast_item_construction(tmp_path):
    model = mx.new_model('DynamicRefItems')
    try:
        history = model.new_space('History', formula='lambda key: None')
        history.new_cells('opening', 'lambda t: 100.0 * key + t')
        valuation = model.new_space('Valuation', formula='''def _formula(month, key):
    if month < 0:
        raise ValueError('month')
    return {'refs': {'act': History[key]}}
''', refs={'History': history, 'act': None})
        valuation.new_cells('start', 'lambda: act.opening(month + 1)')
        valuation.new_cells('value', 'lambda t: start() * (1.0 + 0.01 * t)')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nm.Valuation[0, 1].value(2)\nm.Valuation(1, 2).value(3)\n',
        encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'Valuation': {'cells_param_size': {'t': 8}},
                                                       'History': {'cells_param_size': {'t': 8}}}}), encoding='utf-8')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py'],
                           cwd=tmp_path, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    call = source.split('class _c_Valuation')[1].split('def __call__(self')[1].split('\n    def ')[0]
    fast = call.split('_mx_fast_root: _c_Valuation = ')[1].split('return _mx_fast_root')[0]
    assert '_mx_fast_root._mx_apply_formula(month, key)' in fast, call
    assert fast.index('_mx_assign_params(_mx_fast_root') < fast.index('_mx_apply_formula')
    assert '_mx_root._mx_apply_formula(month, key)' in call          # the generic walk keeps it too
    probe = textwrap.dedent('''\
        import sys
        m = __import__(sys.argv[1]).mx_model
        out = [m.Valuation[month, key].value(t) for month in (0, 1, 5) for key in (1, 2, 3) for t in (0, 4)]
        out.append(m.Valuation(2, 3).act is m.History[3])
        try:
            m.Valuation(-1, 1)
        except ValueError as error:
            out.append(str(error))
        print(repr(out))
        ''')
    values = []
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        values.append(eval(run.stdout))
    assert values[1] == values[0]
    assert values[0][-2:] == [True, 'month']
