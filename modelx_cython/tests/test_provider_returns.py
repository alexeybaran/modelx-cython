"""Every C function of a native support header that returns a C number is known to the integer guard.

The guard flags ``+ - *`` of the signed C integers it knows; a header function it does not know is an unknown
operand, and its arithmetic with a literal would be plain C (silent wrap beyond 64 bits). The headers are the
``.pxd`` files copied into compiled model packages (``native_sources``).
"""
import pathlib
import re

from modelx_cython.integer_boundaries import _PROVIDER_RETURNS, _TABLE_RETURNS, _WORD_RETURNS

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
HEADERS = ('_numeric_cache.pxd', '_wide_numeric_cache.pxd', '_tables.pxd')
KINDS = {'double': 'float', 'float': 'float', 'word': 'word', 'size_t': 'word', 'uint64_t': 'word',
         'bint': 'int', 'int': 'int', 'long': 'int', 'long long': 'int', 'Py_ssize_t': 'int', 'Py_hash_t': 'int'}
# a module-level ``cdef`` function (inline or not); methods of cdef classes are indented and not matched
FUNCTION = re.compile(r'^cdef\s+(?:inline\s+)?([A-Za-z_][\w ]*?)\s+(\w+)\s*\(', re.MULTILINE)


def _functions():
    for header in HEADERS:
        for result, name in FUNCTION.findall((PACKAGE / header).read_text(encoding='utf-8')):
            result = ' '.join(result.split())
            if result.split()[0] in ('class', 'enum', 'struct', 'union', 'extern'):
                continue
            yield header, result, name


def test_every_number_returning_header_function_is_registered():
    seen = list(_functions())
    assert len(seen) > 20, seen
    wrong = []
    for header, result, name in seen:
        if result in ('object', 'void', 'PreparedRef'):
            continue
        assert result in KINDS, f'{header}: {name} returns {result!r}, a type this test does not classify'
        registered = ('word' if name in _WORD_RETURNS else
                      _PROVIDER_RETURNS.get(name) or _TABLE_RETURNS.get(name))
        if registered != KINDS[result]:
            wrong.append(f'{header}: {name} returns {result} ({KINDS[result]}), registered as {registered!r}')
    assert not wrong, '\n'.join(wrong)


def test_registry_names_only_header_functions():
    names = {name for _, _, name in _functions()}
    registry = set(_PROVIDER_RETURNS) | set(_TABLE_RETURNS) | _WORD_RETURNS
    assert registry <= names, registry - names
    assert not set(_PROVIDER_RETURNS) & set(_TABLE_RETURNS)
