"""A value written from Python to an integer Reference: an int, a NumPy integer, a bool or a float that is a
whole number gives live modelx's value; a fractional float or a non-number raises TypeError naming the
Reference, never truncates."""
import json
import subprocess
import sys
import textwrap

import modelx as mx

PROBE = textwrap.dedent('''\
    import json
    import numpy as np
    from exported_cy import mx_model as m

    def write(value):
        space = type(m)().S
        try:
            space.K = value
            return ['value', float(space.c())]
        except Exception as error:
            return [type(error).__name__, str(error)]

    print(json.dumps({
        'int': write(4), 'whole': write(4.0), 'numpy': write(np.int64(4)), 'bool': write(True),
        'fraction': write(0.5), 'numpy_fraction': write(np.float64(2.5)), 'nan': write(float('nan')),
        'text': write('4'), 'none': write(None),
    }))
''')


def test_integer_reference_writes_are_whole_or_raise(tmp_path):
    model = mx.new_model('IntegerReference')
    try:
        space = model.new_space('S', refs={'K': 3})
        space.new_cells('c', 'lambda: K * 2')
        live = {}
        for name, value in (('int', 4), ('whole', 4.0), ('numpy', 4), ('bool', True)):
            space.K = value
            live[name] = space.c()
        space.K = 3
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\nm.S.c()\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    out = json.loads(subprocess.run([sys.executable, '-c', PROBE], cwd=tmp_path, check=True,
                                    capture_output=True, text=True).stdout)
    for name, value in live.items():
        assert out[name] == ['value', float(value)], name
    for name in ('fraction', 'numpy_fraction', 'nan', 'text', 'none'):
        assert out[name][0] == 'TypeError' and out[name][1].startswith('S.K requires an integer'), (name, out[name])
