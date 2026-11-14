"""diag-us2 (D94): range loop targets and once-assigned locals take intervals from proven bounds.

``for u in range(0, min(t, n() - 1) + 1): s = t - u; ... f(s, s + 12 * (u // 12))`` (LTC_KR_S ann_pay):
t is proven from the Cell's argument window, so u lies in [0, max t] and s in [0, max t]; the argument
needs no D76 guard and no checked multiply. With a stop that has no proven upper bound (``n() + 1``),
the guard stays. Values equal the exported model, and on full-range arguments an unproven loop still
gives the exact value or raises, never a wrapped one.
"""
import re
import subprocess
import sys
import textwrap

import modelx as mx

FORMULAS = {
    'n': 'def n():\n    return 40\n',
    'f': 'def f(a, b):\n    return 0.001 * a + 0.0001 * b\n',
    'proven': textwrap.dedent('''\
        def proven(t):
            total = 0.0
            for u in range(0, min(t, n() - 1) + 1):
                s = t - u
                total += f(s, s + 12 * (u // 12))
            return total
        '''),
    'unproven': textwrap.dedent('''\
        def unproven(t):
            total = 0.0
            for u in range(0, n() + 1):
                s = t - u
                total += f(t, s + 12 * (u // 12))
            return total
        '''),
}
WIDE = textwrap.dedent('''\
    def wide(a, b):
        total = 0.0
        for u in range(a, b):
            total += term(u * 2 + 1)
        return total
    ''')


def _body(source, name):
    found = re.search(rf'def _f_{name}\(.*?(?=\n    @|\n    def |\nclass |\ndef |\Z)', source, re.S)
    if found is None:
        found = re.search(rf'    def {name}\(.*?(?=\n    @|\n    def |\nclass |\ndef |\Z)', source, re.S)
    text = found.group(0)
    entry = re.search(r'return (_mx_dense_formula_\d+)\(', text)
    if entry:
        text = re.search(rf'def {entry.group(1)}\(.*?(?=\n@|\nclass |\ndef |\Z)', source, re.S).group(0)
    return text


def test_loop_local_intervals(tmp_path):
    model = mx.new_model('LoopIntervals')
    try:
        space = model.new_space('Projection', formula=lambda point_id: None)
        for formula in FORMULAS.values():
            space.new_cells(formula=formula)
        space.new_cells('term', 'lambda index: float(index)', is_cached=False)
        space.new_cells(formula=WIDE, is_cached=False)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Projection[1]\n'
        'for t in range(0, 60):\n'
        '    for s in range(-45, 60):\n'
        '        p.f(s, s + 12 * 4)\n'
        '        p.f(s, s)\n'
        '    p.proven(t); p.unproven(t)\n'
        'p.wide(1, 5)\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text('{}\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py'],
                   cwd=tmp_path, check=True)
    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    proven = _body(source, 'proven')
    assert '>> 63' not in proven and '_mx_signed_mul' not in proven and 'OverflowError' not in proven, proven
    unproven = _body(source, 'unproven')
    assert '>> 63' in unproven or '_mx_signed_' in unproven, unproven        # the guard stays
    probe = textwrap.dedent('''\
        from exported import mx_model as original
        from exported_cy import mx_model as compiled
        o, c = original.Projection[1], compiled.Projection[1]
        for t in range(0, 60):
            for name in ('proven', 'unproven'):
                a, b = getattr(o, name)(t), getattr(c, name)(t)
                assert a.hex() == b.hex(), (name, t, a, b)
        LO, HI = -(1 << 63), (1 << 63) - 1
        for a, b in ((1, 5), (-7, 3), (HI // 2 - 3, HI // 2 + 1), (LO // 2, LO // 2 + 3)):
            assert c.wide(a, b) == sum(float(u * 2 + 1) for u in range(a, b)), (a, b)
        for a, b in ((HI // 2 + 1, HI // 2 + 3), (LO // 2 - 2, LO // 2)):
            try:
                c.wide(a, b)
            except OverflowError:
                pass
            else:
                raise AssertionError(('wrapped', a, b))
        print('loop intervals ok')
        ''')
    result = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'loop intervals ok' in result.stdout
