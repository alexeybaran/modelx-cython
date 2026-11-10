"""Primitive provider extrema semantics used by expression reductions."""
import importlib
import itertools
import math
import subprocess
import sys
from pathlib import Path

import modelx_cython


def test_double_extrema_order_nan_zero_and_exceptions(tmp_path):
    provider = Path(modelx_cython.__file__).parent
    for suffix in ('.pyx', '.pxd'):
        (tmp_path / ('operations' + suffix)).write_bytes((provider / ('_native_iterables' + suffix)).read_bytes())
    (tmp_path / 'extrema.pyx').write_text('''\
cimport operations
def minimum(double a, double b, double c):
    return operations.f64_min_pair(operations.f64_min_pair(a, b), c)
def maximum(double a, double b, double c):
    return operations.f64_max_pair(operations.f64_max_pair(a, b), c)
cdef double value(object function, int index) except *:
    return function(index)
def ordered(object function):
    return operations.f64_max_pair(operations.f64_max_pair(value(function, 0), value(function, 1)), value(function, 2))
''')
    (tmp_path / 'setup.py').write_text('from setuptools import setup\n'
        'from Cython.Build import cythonize\n'
        'setup(ext_modules=cythonize(["extrema.pyx", "operations.pyx"], compiler_directives={"freethreading_compatible": True}))\n')
    result = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                            cwd=tmp_path, text=True, capture_output=True)
    (tmp_path / 'build.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    sys.path.insert(0, str(tmp_path))
    try:
        native = importlib.import_module('extrema')
        values = (-0.0, 0.0, -1.0, 1.0, -math.inf, math.inf, math.nan)
        for operation, compiled in ((min, native.minimum), (max, native.maximum)):
            for operands in itertools.product(values, repeat=3):
                expected, actual = operation(*operands), compiled(*operands)
                if math.isnan(expected):
                    assert math.isnan(actual)
                else:
                    assert actual == expected
                    if expected == 0:
                        assert math.copysign(1, actual) == math.copysign(1, expected)
        for index in range(3):
            for error in (ValueError('operand'), StopIteration('operand')):
                seen = []
                def operand(i):
                    seen.append(i)
                    if i == index:
                        raise error
                    return float(i)
                try:
                    native.ordered(operand)
                except type(error) as actual:
                    assert actual is error
                else:
                    raise AssertionError('operand exception lost')
                assert seen == list(range(index + 1))
        seen = []
        assert native.ordered(lambda i: seen.append(i) or float(i)) == 2.0
        assert seen == [0, 1, 2]
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop('extrema', None)
        sys.modules.pop('operations', None)
