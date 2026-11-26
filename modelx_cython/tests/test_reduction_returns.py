"""Every C function of the reduction state header that returns a C number is known to the integer guard.

As ``test_provider_returns`` for the other native headers, and also for the functions the header declares from C
(``cdef extern``): a function the guard does not know is an unknown operand, whose arithmetic with a literal would
be plain C.
"""
import pathlib
import re

from modelx_cython.integer_boundaries import _REDUCTION_RETURNS, native_return_kind

HEADER = pathlib.Path(__file__).resolve().parents[1] / '_reductions.pxd'
KINDS = {'double': 'float', 'bint': 'int', 'int': 'int', 'long long': 'int', 'Py_ssize_t': 'int'}
MODULE_LEVEL = re.compile(r'^cdef\s+(?:inline\s+)?([A-Za-z_][\w ]*?)\s+(\w+)\s*\(', re.MULTILINE)
EXTERN = re.compile(r'^    ([A-Za-z_][\w ]*?)\s+(\w+)\s+"[^"]+"\s*\(', re.MULTILINE)


def _functions():
    text = HEADER.read_text(encoding='utf-8')
    for result, name in MODULE_LEVEL.findall(text) + EXTERN.findall(text):
        result = ' '.join(result.split())
        if result.split()[0] in ('class', 'enum', 'struct', 'union', 'extern'):
            continue
        yield result, name


def test_every_number_returning_function_is_registered():
    seen = list(_functions())
    assert len(seen) > 15, seen
    wrong = []
    for result, name in seen:
        if result in ('object', 'void'):
            continue
        assert result in KINDS, f'{name} returns {result!r}, a type this test does not classify'
        if native_return_kind(name) != KINDS[result]:
            wrong.append(f'{name} returns {result} ({KINDS[result]}), registered as {native_return_kind(name)!r}')
    assert not wrong, '\n'.join(wrong)


def test_registry_names_only_header_functions():
    assert set(_REDUCTION_RETURNS) <= {name for _, name in _functions()}
