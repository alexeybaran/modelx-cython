"""Slow paths of guarded integer arithmetic, compiled (D76, D96).

An integer Reference far above the guard bounds drives each guarded form onto its exact
Python path: a comparison, a true division and float() give exactly the exported model's
results; an integer local whose exact value leaves the 64-bit range raises OverflowError
that names the cell (the named conversion `_mx_exact_int64`), where the exported model
carries on with a big int.
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


def test_guarded_slow_paths_are_exact_or_name_the_cell(tmp_path):
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
    text = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert "_mx_exact_int64(" in text and "'Values.loc')" in text, text
    results = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', RUN.format(package=package)], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        results[package] = json.loads(run.stdout.strip().splitlines()[-1])
    exported, compiled = results['exported'], results['exported_cy']
    assert compiled[:3] == exported[:3]                 # comparison, division, float(): exact
    assert compiled[4] == exported[4]                   # a small value: the fast path
    assert not exported[3][1].startswith('OverflowError')
    assert compiled[3][1].startswith('OverflowError: Values.loc requires an exact signed 64-bit integer'), compiled
