"""The integer guard's table of numeric cache helper results matches _numeric_cache.pxd."""
import pathlib
import re

from modelx_cython import integer_boundaries


def test_table_matches_the_pxd():
    source = (pathlib.Path(integer_boundaries.__file__).parent / '_numeric_cache.pxd').read_text(encoding='utf-8')
    found = re.findall(r'^cdef\s+(?:inline\s+)?(long long|double)\s+(\w+)\s*\(', source, re.M)
    assert {name: 'float' if ctype == 'double' else 'int' for ctype, name in found} == integer_boundaries._PROVIDER_RETURNS
