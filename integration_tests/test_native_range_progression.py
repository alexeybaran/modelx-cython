"""Exact int64 range progression, including a step of -2**63."""
import importlib
import itertools
from pathlib import Path
import re
import subprocess
import sys

import modelx_cython
import pytest


def test_native_progression_matches_python_without_overflow(tmp_path):
    provider = Path(modelx_cython.__file__).parent
    for suffix in ('.pyx', '.pxd'):
        (tmp_path / ('operations' + suffix)).write_bytes((provider / ('_native_iterables' + suffix)).read_bytes())
    (tmp_path / 'progression.pyx').write_text('''\
cimport operations

def collect(long long start, long long stop, long long step, Py_ssize_t limit=12):
    cdef operations.I64Range state
    cdef long long value
    cdef Py_ssize_t count = 0
    result = []
    operations.i64_range_init(&state, start, stop, step)
    while count < limit and operations.i64_range_next(&state, &value):
        result.append(value)
        count += 1
    return result
''')
    (tmp_path / 'setup.py').write_text('from setuptools import setup\n'
        'from Cython.Build import cythonize\n'
        'setup(ext_modules=cythonize(["operations.pyx", "progression.pyx"], compiler_directives={"freethreading_compatible": True}))\n')
    result = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                            cwd=tmp_path, text=True, capture_output=True)
    (tmp_path / 'build.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    sys.path.insert(0, str(tmp_path))
    try:
        native = importlib.import_module('progression')
        endpoints = (-(1 << 63), -(1 << 63) + 1, -5, -1, 0, 1, 5, (1 << 63) - 2, (1 << 63) - 1)
        steps = (-(1 << 63), -(1 << 63) + 1, -12, -2, -1, 1, 2, 12, (1 << 63) - 1)
        for start, stop, step in itertools.product(endpoints, endpoints, steps):
            expected = list(itertools.islice(range(start, stop, step), 12))
            assert native.collect(start, stop, step) == expected, (start, stop, step)
        for limit in (0, 12):
            with pytest.raises(ValueError, match=re.escape('range() arg 3 must not be zero')):
                native.collect(0, 0, 0, limit)
        source = (tmp_path / 'operations.c').read_text()
        signatures = list(re.finditer(r'^static [^\n]*__pyx_f_[^\n]*i64_range_next\([^\n]*\) \{', source, re.MULTILINE))
        assert signatures
        for signature in signatures:
            body = source[signature.start():source.index('\n}\n', signature.end())]
            assert not any(token in body for token in ('PyObject_GetIter(', 'PyIter_Next(', 'PyNumber_Add('))
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop('progression', None)
        sys.modules.pop('operations', None)
