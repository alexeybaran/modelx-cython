"""Integer locals whose exact values leave the 64-bit range, compiled.

Formerly the slow paths of the D76 guards, then a checked mode that raised OverflowError here. A
product or sum of another value of the model (a Reference) cannot be proved to fit, so it is
computed on Python ints, as live: every formula's float result equals the exported model's, big
intermediate values included; a small value too.
"""

import json
import subprocess
import sys

import modelx as mx

RUN = '''
import json
from {package} import mx_model as m
out = []
for name, base in (('cmp', (1 << 62) + 7), ('div', -(1 << 63) + 1), ('flt', (1 << 62) + 7),
                   ('loc', (1 << 62) + 7), ('loc', 5)):
    v = type(m)().Values
    v.base = base
    try:
        out.append([name, repr(getattr(v, name)(0, 2))])
    except OverflowError as error:
        out.append([name, 'OverflowError: ' + str(error)])
print(json.dumps(out))
'''


def test_slow_paths_equal_the_exported_model(tmp_path):
    exported, compiled = _slow_paths(tmp_path)
    assert not any(value.startswith('OverflowError') for _, value in exported)
    assert compiled == exported


def _slow_paths(tmp_path):
    model = mx.new_model('GuardSlowPaths')
    try:
        space = model.new_space('Values', refs={'base': 3})
        space.new_cells('cmp', 'def cmp(a, b):\n    k = base + a\n    return 1.0 if k * b <= 0.5 else 0.0\n')
        space.new_cells('div', 'def div(a, b):\n    k = base + a\n    return (k - b) / 12.0\n')
        space.new_cells('flt', 'def flt(a, b):\n    k = base + a\n    return float(k * b)\n')
        space.new_cells('loc', 'def loc(a, b):\n    k = base + a\n    total = 0.0\n'
                               '    for j in range(1, 3):\n        n = k * b + j\n        total += n / 2.0\n'
                               '    return total\n')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nv = m.Values\nfor a in range(5):\n'
        '    v.cmp(a, 2); v.div(a, 2); v.flt(a, 2); v.loc(a, 2)\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    results = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', RUN.format(package=package)], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        results[package] = json.loads(run.stdout.strip().splitlines()[-1])
    return results['exported'], results['exported_cy']
