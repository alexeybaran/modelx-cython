"""diag-mt: per-step code of a collected Projection takes no reference on objects every worker shares.

Two forms took a reference per call on one shared object, which four workers contended on
(free-threaded CPython: atomic reference counts on the same cache line):

- a table bound to a local, ``tbl = market.curve()``: the handle was looked up per call (a reference
  on the shared table and on its handle); it is now held per instance as an inline receiver is (D71),
  with its literal columns (D75);
- a lowered reduction over the builtin ``sum`` / ``min``: the builtin was bound to an ``object``
  local per call (a reference on the one builtin function); the builtin name is now read directly.

Results must not change: compiled at one and at four workers equals the exported model.
"""
import subprocess
import sys
import textwrap

import modelx as mx
from modelx_helpers.immutable_inputs import Table


def _model(tmp_path):
    keys = tuple(range(1, 61))
    model = mx.new_model('SharedRefs')
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
    tbl = market.curve()
    k = (t + key) % 3 + 1
    return tbl.loc[k, "rate"] * tbl.loc[min(k, tbl.index.max()), "load"]
''')
        projection.new_cells('claims', '''\
def claims(t, kind):
    return rate(t) * (kind + 1) / (t + key)
''')
        projection.new_cells('claims_all', '''\
def claims_all(t):
    return sum(claims(t, kind) for kind in range(3))
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(36):
        total += claims_all(t) * key
    return total
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
        '    [p[key].claims_all(t) for t in range(36)]; p[key].value()\n',
        encoding='utf-8')


def test_held_table_local_and_builtin_reduction_take_no_shared_reference(tmp_path):
    _model(tmp_path)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    generated = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    # the table local: resolved once per instance, read borrowed, its columns held
    assert '.TableRef, self._mx_th_0)' in generated and 'if self._mx_th_0 is None:' in generated, generated
    assert generated.count('.table_handle(') == 1, generated
    assert 'root._mx_th_0 = None' in generated and 'root._mx_tc_0 = 0' in generated
    assert '.col("rate")' not in generated and '.col("load")' not in generated
    # the builtin reduction: no object local bound to the builtin
    assert '.f64_accepts(sum)' in generated, generated
    assert '_function = sum' not in generated and '_function: object' not in generated
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        m = import_module(sys.argv[1]).mx_model
        p = m.Portfolio.Projection
        if sys.argv[1] == 'exported_cy':
            serial = type(m)()
            serial._mx_collect_set_workers_v2(1)
            one = serial.Portfolio.Projection.collect().sum.value
            four = p.collect().sum.value
            again = p.collect().sum.value
            assert again == four
            print(repr(one), repr(four))
        else:
            print(repr(p.collect().sum.value), repr(p.collect().sum.value))
        item = p[7]
        values = [item.claims_all(t) for t in range(36)] + [item.rate(t) for t in range(36)] + [item.value()]
        print(repr(values))
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    one, four = out['exported_cy'][0].split()
    assert one == out['exported'][0].split()[0], (out, 'one worker equals the exported model')
    # four chunks merge in domain order: equal to the one-worker total within its rounding
    assert abs(float(four) - float(one)) <= 1e-12 * abs(float(one)), out
    assert out['exported_cy'][1:] == out['exported'][1:], out
