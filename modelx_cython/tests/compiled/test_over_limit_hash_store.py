"""D135: a Cell above the dense limit is held in the hash store on the direct route.

`surv(s, t)` has a 2-D integer window. With ``max_dense_entries`` below its size it is not inline dense;
a direct item keeps its own hash store. Values differ by policy, so a stale entry would change the result.
The build warns, naming the cell, its traced range and the grid-key rule (R9); the report states the cache
kind and why it is not dense. Compiled items equal the exported model. Direct-route twin of
``test_over_limit_hash_store_collect`` (the same model without ``collect_over``).
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx

PREFIX = 'exported_cy._m_Portfolio._mx_classes._c_Projection._f_'


def _model(tmp_path, spec):
    keys = tuple(range(1, 31))
    model = mx.new_model('ReuseDense')
    try:
        portfolio = model.new_space('Portfolio')
        projection = portfolio.new_space('Projection', formula='lambda key: None', refs={'keys': keys})
        projection.new_cells('rate', 'lambda t: 0.01 * ((t + key) % 7)')
        projection.new_cells('surv', '''\
def surv(s, t):
    if t <= s:
        return 1.0
    return surv(s, t - 1) * (1.0 - rate(t - 1))
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(12):
        for s in range(0, t + 1, 1 + key % 3):
            total += surv(s, t) * (t + 1)
    return total
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'for key in (1, 2, 3):\n'
        '    p[key].value(); [p[key].surv(s, t) for s in range(12) for t in range(12)]\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    if not done.returncode == 0:
        return done
    return (done.stdout + done.stderr,
            json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8')))


PROBE = textwrap.dedent('''\
    import sys
    from importlib import import_module
    m = import_module(sys.argv[1]).mx_model
    p = m.Portfolio.Projection
    print(repr([p[key].value() for key in range(1, 31)]))
    item = p[7]
    print(repr([item.surv(s, t) for s in range(12) for t in range(12)] + [item.value()]))
    ''')


def _run(tmp_path):
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', PROBE, package], cwd=tmp_path, capture_output=True,
                             text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    return out


def test_over_limit_cell_is_a_hash_store_on_the_direct_route(tmp_path):
    spec = {'spaces': {'Portfolio': {'spaces': {'Projection': {
        'cells': {'surv': {'max_dense_entries': 16}}}}}}}
    built = _model(tmp_path, spec)
    assert isinstance(built, tuple), built.stdout[-3000:] + built.stderr[-3000:]
    log, report = built
    detail = report[PREFIX + 'surv']['cache_detail']
    assert detail['kind'] == 'native-numeric-keys', detail
    assert 'exceeds max_dense_entries 16' in detail['hash_reason'], detail
    assert 'reused_root' not in detail, detail
    assert any(e['kind'] == 'hash-cache' for e in report[PREFIX + 'surv']['slow_paths'])
    warning = [line for line in log.splitlines() if 'Projection.surv: window' in line]
    assert warning and 'exceeds max_dense_entries 16' in warning[0] and 'every route' in warning[0], log[-3000:]
    assert 's in [' in warning[0] and 't in [' in warning[0] and 'R9' in warning[0], warning
    out = _run(tmp_path)
    assert out['exported_cy'] == out['exported'], out
