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
        # Arithmetic probes below exercise the full signed range.
        space.new_cells('divisible', 'lambda t, n: t % n == 0', is_cached=False)
        space.new_cells('offset', 'lambda t, n: (t - 1) % n != 0', is_cached=False)
        space.new_cells('factor', 'lambda i, b, e: b ** e')
        space.new_cells('duration', 'lambda t: t', is_cached=False)
        space.new_cells('fraction', 'def fraction(t):\n    m = duration(t) + 1\n    return (m / 12) * 0.05\n',
                        is_cached=False)
        space.new_cells('horizon', 'lambda: 1200')
        space.new_cells('piecewise', textwrap.dedent('''\
            def piecewise(t):
                year = duration(t)
                if year >= horizon() // 12 - 1:
                    return 0.0
                if year == 1:
                    return min(year, 9) * 0.25
                return max(0.1 - (0.04 / 3.0) * (year - 1), 0.06)
            '''), is_cached=False)
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        's = mx_model.Policy[1]\n'
        's.derived(0); s.derived(3); s.divisible(-7, -3); s.offset(-7, -3); s.factor(0, 4.0, 0.5)\n'
        's.fraction(1); s.fraction(3); s.piecewise(1); s.piecewise(2)\n',
        encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({
        'dense_bounds_proofs': True,
        'dense_one_axis': True,
        'dense_miss_binding': True,
        'propagate_exact_receiver': True,
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
    assert '_mx_inbounds_exact' in generated and '_mx_exact_receiver' in generated
    assert '_mx_divisible(' in generated and '_mx_divisible_after_decrement(' in generated
    assert '_mx_memo_pow(' in generated
    assert '_mx_checked_quotient:' in generated
    assert 'year: _mx_cy.longlong' in generated
    assert '_mx_cy.cast(object, year)' in generated
    (tmp_path / 'override.pyx').write_text(textwrap.dedent('''\
        # cython: freethreading_compatible=True
        from exported_cy._mx_classes cimport _c_Policy

        cdef class Alternate(_c_Policy):
            cdef double _f_value(self, long long t) except? -1:
                return 100.0 + t
        '''), encoding='utf-8')
    (tmp_path / 'setup_override.py').write_text(
        'from setuptools import setup\n'
        'from Cython.Build import cythonize\n'
        'setup(ext_modules=cythonize("override.pyx"))\n', encoding='utf-8')
    subprocess.run([sys.executable, 'setup_override.py', 'build_ext', '--inplace'],
                   cwd=tmp_path, check=True)
    probe = textwrap.dedent('''\
        import math
        from exported_cy import mx_model
        from override import Alternate
        s = mx_model.Policy[1]
        assert [s.derived(i) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]
        alternate = Alternate(mx_model.Policy)
        assert alternate.derived(1) == 102.0
        for operation in (s.value, s.derived):
            for key in (-(1 << 63), -1, 4, (1 << 63) - 1):
                try:
                    operation(key)
                except IndexError:
                    pass
                else:
                    raise AssertionError('out-of-domain cache index reached the formula')
        for t in (-(1 << 63), -7, 0, 7, (1 << 63) - 1):
            for n in (-(1 << 63), -3, -1, 1, 3):
                assert s.divisible(t, n) == (t % n == 0)
                assert s.offset(t, n) == ((t - 1) % n != 0)
        for t in (-(1 << 63), -(1 << 25) - 1, -7, -1, 0, 7, 1 << 25, (1 << 63) - 1):
            assert s.fraction(t).hex() == (((t + 1) / 12) * 0.05).hex()
        for t in (-(1 << 63), -(1 << 63) + 1, -7, 0, 1, 2, 99, (1 << 63) - 1):
            expected = 0.0 if t >= 99 else (min(t, 9) * 0.25 if t == 1
                                            else max(0.1 - (0.04 / 3.0) * (t - 1), 0.06))
            assert s.piecewise(t).hex() == expected.hex()
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
