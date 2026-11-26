"""A value from Python for a C integer, at a Cell's integer parameter or an integer Reference: an int,
a NumPy integer, a bool or a float that is a whole number gives live modelx's value; a fractional float
or a non-number raises TypeError naming the Cell or the Reference, never truncates."""
import json
import subprocess
import sys
import textwrap

import modelx as mx

PROBE = textwrap.dedent('''\
    import json
    import numpy as np
    from exported_cy import mx_model as m

    def run(call):
        try:
            return ['value', float(call())]
        except Exception as error:
            return [type(error).__name__, str(error)]

    def write(value):
        space = type(m)().S
        space.K = value
        return space.c()

    S = m.S
    print(json.dumps({
        'int': run(lambda: S.f(3)), 'whole': run(lambda: S.f(3.0)), 'numpy': run(lambda: S.f(np.int64(3))),
        'bool': run(lambda: S.f(True)), 'cached': run(lambda: S.g(2.0)),
        'fraction': run(lambda: S.f(1.5)), 'numpy_fraction': run(lambda: S.f(np.float64(2.5))),
        'text': run(lambda: S.f('3')), 'none': run(lambda: S.f(None)), 'nan': run(lambda: S.f(float('nan'))),
        'ref_int': run(lambda: write(4)), 'ref_whole': run(lambda: write(4.0)),
        'ref_fraction': run(lambda: write(0.5)), 'ref_text': run(lambda: write('4')),
    }))
''')


def test_python_values_for_c_integers_are_whole_or_raise(tmp_path):
    model = mx.new_model('IntegerBoundary')
    try:
        space = model.new_space('S', refs={'K': 3})
        space.new_cells('f', 'lambda t: t * 2 + K', is_cached=False)
        space.new_cells('g', 'lambda t: t * 10')
        space.new_cells('c', 'lambda: K * 2')
        live = {'int': space.f(3), 'whole': space.f(3.0), 'numpy': space.f(3), 'bool': space.f(True),
                'cached': space.g(2.0)}
        space.K = 4
        live['ref_int'] = live['ref_whole'] = space.c()
        space.K = 3
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nfor t in range(5):\n    m.S.f(t); m.S.g(t)\nm.S.c()\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    out = json.loads(subprocess.run([sys.executable, '-c', PROBE], cwd=tmp_path, check=True,
                                    capture_output=True, text=True).stdout)
    for name, value in live.items():
        assert out[name] == ['value', float(value)], name
    for name in ('fraction', 'numpy_fraction', 'text', 'none', 'nan'):
        assert out[name][0] == 'TypeError' and out[name][1].startswith('S.f requires an integer'), (name, out[name])
    for name in ('ref_fraction', 'ref_text'):
        assert out[name][0] == 'TypeError' and out[name][1].startswith('S.K requires an integer'), (name, out[name])
