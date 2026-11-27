"""Check scalar calls through a borrowed math Reference on values, errors and rebinding."""

import math
import subprocess
import sys
import textwrap

import modelx as mx


def test_borrowed_scalar_math(tmp_path):
    model = mx.new_model('ScalarMath')
    try:
        values = model.new_space('Values', refs={'numbers': math})
        for name in ('erf', 'exp', 'isinf', 'log', 'sqrt'):
            values.new_cells(name, f'lambda x: numbers.{name}(x)')
        values.new_cells('composed', 'def composed(x):\n    root = numbers.sqrt(x)\n    return numbers.exp(numbers.log(root) / x)\n')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'v.erf(1.5); v.exp(1.5); v.isinf(1.5); v.log(1.5); v.sqrt(1.5); v.composed(1.5)\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    for name in ('erf', 'exp', 'isinf', 'log', 'sqrt'):
        assert f"_mx_sys._mx_math_{name}(_mx_cy.cast(_mx_cy.p_void, self.numbers), self, 'numbers'," in source, name
    probe = textwrap.dedent('''\
        import math
        from types import SimpleNamespace
        from exported_cy import mx_model
        v = mx_model.Values
        assert v.composed(2.5) == math.exp(math.log(math.sqrt(2.5)) / 2.5)
        for name in ('erf', 'exp', 'log', 'sqrt'):
            expected = getattr(math, name)
            for arg in (0.5, 1.0, 2.5, 10.0, math.inf, math.nan):
                left = getattr(v, name)(arg)
                right = expected(arg)
                assert type(left) is type(right), name
                assert left == right or (math.isnan(left) and math.isnan(right))
        assert v.exp(-math.inf) == 0.0 and v.erf(-math.inf) == -1.0
        assert v.isinf(math.inf) is True and v.isinf(-math.inf) is True
        assert v.isinf(math.nan) is False and v.isinf(2.5) is False
        for name, arg, error in (('sqrt', -1.0, ValueError), ('log', 0.0, ValueError),
                                 ('sqrt', -math.inf, ValueError), ('log', -math.inf, ValueError),
                                 ('exp', 1000.0, OverflowError)):
            try:
                getattr(v, name)(arg)
            except error:
                pass
            else:
                raise AssertionError(name)
        v.numbers = SimpleNamespace(log=lambda x: x + 1.0)
        assert v.log(30.0) == 31.0
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
