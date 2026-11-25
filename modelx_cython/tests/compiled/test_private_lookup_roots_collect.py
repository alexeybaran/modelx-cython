"""The direct route and collect's reused roots of an ItemSpace with a ``round(x, n)`` return give
the values live gives (audit-routes P6, D126). The roots once took a per-item round memo over; the
memo is gone (two threads on one item could read each other's result), the values are unchanged."""
import subprocess
import sys
import textwrap

import modelx as mx



def _build(tmp_path):
    model = mx.new_model('PrivateLookups')
    try:
        keys = (1, 2, 3)
        proj = model.new_space(
            'Projection', formula='lambda key: None', refs={'keys': keys},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        proj.new_cells('rounded', '''\
def rounded(t):
    amount = key * 0.37 + t * 0.011
    adjustment = 0.125
    total = amount + adjustment
    return round(total, min(t, 22))
''')
        proj.new_cells('value', 'lambda: sum(rounded(t) for t in range(4))')
        port = model.new_space('Portfolio', refs={'proj': proj})
        port.new_cells('direct', '''\
def direct():
    total = 0.0
    for k in proj.keys:
        item = proj[k]
        total += item.rounded(2) + item.rounded(3)
    return total
''')
        live = {'collect': proj.collect().sum.value, 'direct': port.direct(),
                'items': [proj[k].rounded(t) for k in keys for t in range(4)]}
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'Projection': {'serial_collect': 'reuse'}}}),
                                      encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'm.Projection.collect(); m.Portfolio.direct()\n'
        'for k in (1, 2, 3):\n    [m.Projection[k].rounded(t) for t in range(4)]\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                    '--spec', 'spec.py'], cwd=tmp_path, check=True)
    return live


def test_direct_items_and_collect_roots_equal_live(tmp_path):
    live = _build(tmp_path)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert 'LookupContext' not in source and '_mx_round_digits' in source
    probe = textwrap.dedent(f'''\
        from exported_cy import mx_model as m
        expected = {live!r}
        p = m.Projection
        assert [p[k].rounded(t) for k in (1, 2, 3) for t in range(4)] == expected['items']
        assert p.collect().sum.value == expected['collect']
        assert m.Portfolio.direct() == expected['direct']
        print('ok')
        ''')
    run = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == 'ok'
