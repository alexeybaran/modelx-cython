"""Exercise the generated fast entries and their existing public contracts."""

import subprocess
import sys
import textwrap

import modelx as mx


def test_native_transfers(tmp_path):
    model = mx.new_model('NativeTransfers')
    try:
        space = model.new_space('Policy', formula='lambda key: None')
        space.new_cells('value', 'lambda t: float(t)')
        space.new_cells('derived', 'lambda t: value(t) + 1.0')
        space.new_cells('divisible', 'lambda t, n: t % n == 0')
        space.new_cells('offset', 'lambda t, n: (t - 1) % n != 0')
        space.new_cells('factor', 'lambda i, b, e: b ** e')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        's = mx_model.Policy[1]\n'
        's.derived(0); s.derived(3); s.divisible(-7, -3); s.offset(-7, -3); s.factor(0, 4.0, 0.5)\n',
        encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({
        'dense_bounds_proofs': True,
        'dense_one_axis': True,
        'spaces': {'Policy': {'cells': {
            'divisible': {'native_divisibility': True},
            'offset': {'native_divisibility': True},
            'factor': {'memoize_power': True},
        }}},
    }), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported'],
                   cwd=tmp_path, check=True)
    generated = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert '_inbounds(' in generated
    assert '_mx_divisible(' in generated and '_mx_divisible_after_decrement(' in generated
    assert '_mx_memo_pow(' in generated
    probe = textwrap.dedent('''\
        import math
        from exported_cy import mx_model
        s = mx_model.Policy[1]
        assert [s.derived(i) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]
        for key in (-(1 << 63), -1, 4, (1 << 63) - 1):
            try:
                s.value(key)
            except IndexError:
                pass
            else:
                raise AssertionError(key)
        for t in (-(1 << 63), -7, 0, 7, (1 << 63) - 1):
            for n in (-(1 << 63), -3, -1, 1, 3):
                assert s.divisible(t, n) == (t % n == 0)
                assert s.offset(t, n) == ((t - 1) % n != 0)
        for operation in (s.divisible, s.offset):
            try:
                operation(1, 0)
            except ZeroDivisionError:
                pass
            else:
                raise AssertionError('zero divisor was accepted')
        for i, operand in enumerate((-0.0, -0.0, 0.0, 0.0, -0.0)):
            assert math.copysign(1.0, s.factor(i, operand, 3.0)) == math.copysign(1.0, operand)
        for _ in range(2):
            try:
                s.factor(0, 0.0, -1.0)
            except ZeroDivisionError:
                pass
            else:
                raise AssertionError('failed power was cached')
        assert s.factor(0, 9.0, 0.5) == 3.0
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
