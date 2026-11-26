"""The dense cache of a shared Space read by items: its readiness flag is published after the value (release
fence) and read before it (acquire fence), so a reader never takes the flag without the value (arm64 and
compiler reordering). An owned item's dense cache keeps plain flags. Values equal the
export."""
import subprocess
import sys
import textwrap

import modelx as mx


def test_shared_dense_cache_publishes_its_flags(tmp_path):
    model = mx.new_model('SharedDense')
    try:
        data = model.new_space('Data')
        data.new_cells('factor', 'lambda t: 1.0 + 0.01 * t')
        keys = tuple(range(1, 81))
        policy = model.new_space('Policy', formula='lambda key: None',
                                 refs={'Data': data, 'keys': keys, 'months': tuple(range(48))})
        policy.new_cells('cf', 'lambda t: key * 0.5 * Data.factor(t)')
        policy.new_cells('pv', 'lambda: sum(cf(t) for t in months)')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\nfor k in (1, 2, 3): m.Policy[k].pv()\n',
                                        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    method = lambda name: source.split(f'def {name}(self, t')[1].split('\n    def ')[0]
    factor = method('factor')
    assert '_mx_load_dense_flag(_mx_cy.address(self._has_factor[' in factor
    assert '_mx_store_dense_flag(_mx_cy.address(self._has_factor[' in factor
    cf = method('cf')
    assert 'dense_flag' not in cf and 'self._has_cf[' in cf          # an owned item keeps plain flags
    probe = textwrap.dedent("""\
        import sys
        m = __import__(sys.argv[1]).mx_model
        print(repr([m.Policy[key].pv() for key in range(1, 81)]))
        """)
    values = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        values[package] = eval(run.stdout)
    assert values['exported_cy'] == values['exported']
