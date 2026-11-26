"""Storage by range. A multi-argument integer-keyed Cell is dense when the rectangle of its declared or sampled
range is within the dense ceiling (65,536 entries); above it the Cell goes to the hash store with a build warning
naming the cell, its ranges, the rectangle and the remedies, whatever share of it the sample filled. The build
report (_mx_admission.json) states per multi-argument Cell the store chosen and why. The store never depends on
how many entries the sample's policies happened to fill."""
import json
import subprocess
import sys
import textwrap

import modelx as mx


def _build(tmp_path, spec, n=400):
    model = mx.new_model('DenseByUse')
    try:
        s = model.new_space('Policy', formula='lambda key: None', refs={'n': n})
        s.new_cells('rate', 'lambda t: 0.001 * (t % 13)')
        # well used: a triangle t >= s of a declared 400 x 400 rectangle, every entry filled by each policy
        s.new_cells('used', textwrap.dedent('''\
            def used(t, s):
                if t <= s:
                    return 1.0
                return used(t - 1, s) * (1.0 - rate(t - 1))
            '''))
        # sparse: the same declared rectangle, a single diagonal of 40 entries filled per policy
        s.new_cells('sparse', 'lambda t, s: rate(t) + s')
        s.new_cells('total', textwrap.dedent('''\
            def total():
                a = sum(used(t, s) for t in range(n) for s in range(0, t + 1, 2))
                b = sum(sparse(10 * k, 10 * k) for k in range(n // 10))
                return a + b
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\nfor k in (1, 2):\n    mx_model.Policy[k].total()\n')
    (tmp_path / 'spec.py').write_text(repr(spec))
    run = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported', '--translate-only'], cwd=tmp_path,
                         check=True, capture_output=True, text=True)
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    cells = {k.rsplit('._f_', 1)[1]: v for k, v in admission.items() if '._c_Policy._f_' in k}
    return cells, run.stdout + run.stderr


def test_a_rectangle_above_the_ceiling_goes_to_the_hash_store_with_a_warning(tmp_path):
    cells, log = _build(tmp_path, {'cells_param_size': {'t': 400, 's': 400}})
    lines = [line for line in log.splitlines() if 'hash store: rectangle' in line]
    # 160,000 entries > 65,536: the well used triangle and the sparse diagonal alike
    for name in ('used', 'sparse'):
        cell = cells[name]
        assert cell['cache'] == 'native-numeric-keys', cell['cache_detail']
        detail = cell['cache_detail']
        assert detail['store_reason'] == ('native-numeric-keys: t [0, 400) declared x s [0, 400) declared = '
                                          '160,000 entries exceeds the max_dense_entries 65,536'), detail
        line, = (line for line in lines if f'Policy.{name} (t, s)' in line)
        for part in ('t [0, 400) declared x s [0, 400) declared = 160,000 entries', 'dense ceiling of 65,536',
                     're-parametrise the cell if its parameters are tied',
                     'mark it not cached if each value is read once', 'tighten or correct the declaration'):
            assert part in line, (part, line)


def test_a_rectangle_within_the_ceiling_is_dense(tmp_path):
    cells, log = _build(tmp_path, {'cells_param_size': {'t': 250, 's': 250}}, n=250)
    for name in ('used', 'sparse'):
        detail = cells[name]['cache_detail']
        assert cells[name]['cache'] == 'dense', detail
        assert detail['store_reason'].startswith(
            'dense: t [0, 250) declared x s [0, 250) declared = 62,500 entries, within max_dense_entries 65,536'), detail
    assert 'hash store: rectangle' not in log, log[-3000:]
