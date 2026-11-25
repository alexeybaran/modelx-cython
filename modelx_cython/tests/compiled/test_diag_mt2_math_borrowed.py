"""diag-mt2: a module math member call passes the math Reference borrowed (no reference per step).

``self.math.log(x)`` compiles to ``_native_math.m_log(<the Reference>, x)``, the Reference passed as a
borrowed ``void*`` (``_mx_cy.cast(_mx_cy.p_void, self.math)``). Results must not change: compiled items
equal the exported model. Direct-route twin of ``test_diag_mt2_math_borrowed_collect`` (the same model
without ``collect_over``; items are called directly).
"""
import math
import json
import subprocess
import sys
import textwrap

import modelx as mx


def _model(tmp_path):
    keys = tuple(range(1, 61))
    model = mx.new_model('MathBorrowed')
    try:
        portfolio = model.new_space('Portfolio')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'math': math, 'keys': keys})
        projection.new_cells('rate', '''\
def rate(t):
    return 0.001 * (t + key % 7 + 1)
''')
        projection.new_cells('surv', '''\
def surv(t):
    if t == 0:
        return 1.0
    mu = -math.log(1.0 - rate(t - 1))
    return surv(t - 1) * math.exp(-mu / 12.0) * math.sqrt(1.0 - rate(t) / 4.0)
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(48):
        total += surv(t) * math.floor(t / 12.0 + 1.0) * math.pi
    return total
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'for key in (1, 2, 3):\n'
        '    [p[key].surv(t) for t in range(48)]; p[key].value()\n',
        encoding='utf-8')


def test_math_reference_is_borrowed_and_results_stay(tmp_path):
    _model(tmp_path)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    # the build report: the math calls are native provider operations
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    helpers = {h.rsplit('.', 1)[-1] for entry in admission.values() for h in entry['native_helpers']}
    assert {'m_log', 'm_exp', 'm_sqrt', 'm_floor', 'm_constant'} <= helpers, helpers
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        m = import_module(sys.argv[1]).mx_model
        p = m.Portfolio.Projection
        print(repr(sum(p[key].value() for key in range(1, 61))))
        item = p[7]
        print(repr([item.surv(t) for t in range(48)] + [item.value()]))
        ''')
    out = {}
    for name in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, name], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, name + run.stdout + run.stderr
        out[name] = run.stdout.splitlines()
    assert out['exported_cy'] == out['exported'], out
