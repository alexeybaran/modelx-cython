"""Powers inside min/max over an accumulator local, and a power with a power for exponent, build and equal live.

A fractional power inside ``min``/``max`` makes Cython type the operand complex ("complex types are
unordered"), and a local summed in a loop is the base such a power often has. The shapes
``max(total ** 0.5, floor)`` and ``min((1 + rate) ** (1 / 12) - 1, cap)`` over a loop accumulator, and
``x ** (2 ** -t)``, must compile, and every value must equal live modelx.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx

CELLS = {
    'rate': 'lambda t: 0.01 + 0.002 * t',
    'acc': 'def acc(t):\n    total = 0.0\n    for i in range(t + 1):\n        total += rate(i)\n'
           '    return max(total ** 0.5, 0.01) + min((1 + total) ** (1 / 12) - 1, 1.0)\n',
    'halving': 'lambda t: (1.0 + rate(t)) ** (2 ** -t)',
    'mixed': 'def mixed(t):\n    total = 0.0\n    for i in range(t + 1):\n        total += rate(i) ** (2 ** -i)\n'
             '    return max(total, (1 + rate(t)) ** 0.5)\n',
}
TS = range(6)
PROBE = textwrap.dedent('''\
    import json
    from exported_cy import mx_model as m
    s = m.S
    print(json.dumps({{name: [getattr(s, name)(t) for t in {ts}] for name in {names}}}))
    ''')


def test_power_shapes_over_an_accumulator_build_and_equal_live(tmp_path):
    model = mx.new_model('PowerShapes')
    try:
        s = model.new_space('S')
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        names = [n for n in CELLS if n != 'rate']
        live = {name: [getattr(s, name)(t) for t in TS] for name in names}
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nfor t in range(6):\n'
        + ''.join(f'    m.S.{n}(t)\n' for n in names), encoding='utf-8')
    built = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                           cwd=tmp_path, capture_output=True, text=True)
    assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]
    out = subprocess.run([sys.executable, '-c', PROBE.format(ts=list(TS), names=names)], cwd=tmp_path,
                         check=True, capture_output=True, text=True).stdout
    assert json.loads(out) == live
