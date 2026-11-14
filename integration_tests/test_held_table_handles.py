"""diag-4w (D71): per-step reads of a shared Data table take no reference on it.

The handle of a zero-argument table Cell (here a Data reader on another Space) is resolved once per
instance into a field, reset with the per-policy caches of the collect loop, and every per-step read
uses it borrowed: no Cell call and no reference count on the shared table or handle per step. The
results must not change: compiled at one and four workers equals the exported model; an evaluated
item read twice gives the same values.
"""
import subprocess
import sys
import textwrap

import modelx as mx
from modelx_helpers.immutable_inputs import Table


def _model(tmp_path):
    keys = tuple(range(1, 41))
    model = mx.new_model('Held')
    try:
        portfolio = model.new_space('Portfolio')
        data = portfolio.new_space('Inputs', refs={
            'rates': Table([1, 2, 3], [(0.05, 2.0), (0.06, 3.0), (0.07, 5.0)], ['rate', 'load'])})
        data.new_cells('curve', '''\
def curve() -> "modelx_helpers.immutable_inputs.Table":
    return rates
''')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'market': data, 'keys': keys},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('rate', '''\
def rate(t):
    k = (t + key) % 3 + 1
    return market.curve().loc[k, "rate"] * market.curve().loc[k, "load"]
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(24):
        total += rate(t) * key
    return total
''')
        projection.new_cells('rare', '''\
def rare(t):
    k = t % 3 + 1
    if t > 20:
        return market.curve().loc[k, "missing"]
    return market.curve().loc[k, "load"]
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    spec = {'parallel_workers': 4, 'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'parallel_collect_reuse': True, 'parallel_collect_order': 'domain'}}}}}
    (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'p.collect()\n'
        'for key in (1, 2, 3):\n'
        '    [p[key].rate(t) for t in range(24)]; p[key].value(); [p[key].rare(t) for t in range(3)]\n',
        encoding='utf-8')


def test_shared_table_reads_use_a_held_handle_and_results_stay(tmp_path):
    _model(tmp_path)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    generated = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    assert 'if self._mx_th_0 is None:' in generated, generated
    assert '_mx_cy.cast(_mx_helper_' in generated and '.TableRef, self._mx_th_0)' in generated
    assert generated.count('.table_handle(self.market.curve())') == 2          # once per formula, not per read
    # with the per-policy caches: reset on the reused root, with the held column indexes (D75)
    assert 'root._mx_th_0 = None' in generated and 'root._mx_tc_0 = 0' in generated, generated
    assert '.held_col(_mx_cy.address(self._mx_tc_' in generated and '.col("rate")' not in generated
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        m = import_module(sys.argv[1]).mx_model
        p = m.Portfolio.Projection
        if sys.argv[1] == 'exported_cy':
            from exported_cy import _mx_sys
            serial = type(m)()
            serial._mx_collect_set_workers_v2(1)
            one = serial.Portfolio.Projection.collect().sum.value
            four = p.collect().sum.value
            print(repr(one), repr(four))
        else:
            print(repr(p.collect().sum.value), repr(p.collect().sum.value))
        item = p[7]
        assert [item.rare(t) for t in range(5)] == [p[7].rare(t) for t in range(5)]
        try:
            item.rare(22)
        except KeyError as error:
            print('KeyError', error.args)
        else:
            print('no error')
        values = [item.rate(t) for t in range(24)] + [item.value()]
        assert [item.rate(t) for t in range(24)] + [item.value()] == values
        print(repr(values))
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    one, four = out['exported_cy'][0].split()
    assert one == out['exported'][0].split()[0], (out, 'one worker equals the exported model')
    assert float(four) == float(one), out                                     # exact sums of a few terms
    assert out['exported_cy'][1:] == out['exported'][1:], out           # the missing column raises at the read
