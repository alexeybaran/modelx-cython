"""Check native numeric results against the ordinary Python call."""

import math
import subprocess
import sys
import textwrap

import modelx as mx


def test_native_builtin_math(tmp_path):
    model = mx.new_model('NativeBuiltinMath')
    try:
        space = model.new_space('Values', refs={'math': math})
        for name, formula in (
            ('one', 'lambda x: round(x)'),
            ('digits', 'lambda x, n: round(x, n)'),
            ('down', 'lambda x: math.floor(x)'),
            ('up', 'lambda x: math.ceil(x)'),
        ):
            space.new_cells(name, formula).is_cached = False
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'v.one(2.5); v.digits(2.675, 2); v.down(1.5); v.up(-1.5)\n',
        encoding='utf-8')
    (tmp_path / 'spec.py').write_text("{'native_math': True}\n", encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported'],
                   cwd=tmp_path, check=True)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert '_mx_round_one(' in source and '_mx_round_digits(' in source
    assert source.count('_mx_integral_math_call(') == 2
    probe = textwrap.dedent('''\
        import math
        import random
        import struct
        from exported_cy import mx_model
        from exported_cy import _mx_classes
        assert _mx_classes._mx_canonical_round is round
        assert _mx_classes._mx_canonical_floor is math.floor
        assert _mx_classes._mx_canonical_ceil is math.ceil
        v = mx_model.Values
        rng = random.Random(107)
        edge = [0., -0., 0.5, -0.5, 2.5, -2.5, 2.675, 1.005, 1.015,
                0.15, -0.15, 0.125, 125., -125., 2.**52, 2.**63, -2.**63,
                5e-324, -5e-324, float('inf'), float('-inf'), float('nan')]
        random_values = [struct.unpack('=d', rng.randbytes(8))[0] for _ in range(4000)]
        decimal_edges = [math.nextafter(x, direction) for n in range(-22, 23)
                         for k in (0.5, 1.5, 2.5, 128.5, 1e7+0.5)
                         for x in (k * 10.0**(-n), -k * 10.0**(-n))
                         for direction in (-math.inf, x, math.inf)]
        for x in edge + random_values + decimal_edges:
            for n in range(-24, 25):
                actual, expected = v.digits(x, n), round(x, n)
                assert (math.isnan(actual) and math.isnan(expected)
                        or struct.pack('=d', actual) == struct.pack('=d', expected)), (x, n, actual, expected)
            for actual, expected in ((v.one, round), (v.down, math.floor), (v.up, math.ceil)):
                try:
                    target = expected(x)
                except (ValueError, OverflowError) as error:
                    try:
                        actual(x)
                    except type(error):
                        pass
                    else:
                        raise AssertionError(x)
                else:
                    result = actual(x)
                    assert type(result) is int and result == target, (x, result, target)
        class Hook:
            def __round__(self, *args): return ('round', args)
            def __floor__(self): return 'floor'
            def __ceil__(self): return 'ceil'
        assert v.one(Hook()) == ('round', ())
        assert v.digits(Hook(), 2) == ('round', (2,))
        assert v.down(Hook()) == 'floor'
        assert v.up(Hook()) == 'ceil'
        from types import SimpleNamespace
        v.math = SimpleNamespace(floor=lambda x: x + 10, ceil=lambda x: x + 20)
        assert v.down(1.5) == 11.5 and v.up(1.5) == 21.5
    ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
