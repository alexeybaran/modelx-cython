"""Emission for ordinary Cells backed by the small numerical key store."""
import libcst as cst

from .helpers import module_name

PROVIDER = 'modelx_cython._numeric_cache'
ALIAS = module_name(PROVIDER)
TYPE = ALIAS + '.NumericCache'


def body(cell):
    cache = 'self._v_' + cell.name
    bits, result = '_mx_cache_bits', '_mx_cache_result'
    while bits in cell.params:
        bits += '_'
    while result in cell.params:
        result += '_'
    keys, reflexive = [], []
    for p in cell.params:
        floating = cell.get_argtype_expr(p, c_style=True) == 'double'
        keys.append(f'{ALIAS}.{"double_word" if floating else "int_word"}({p})')
        if floating:
            reflexive.append(f'{p} == {p}')
    keys += ['0'] * (3 - len(keys))
    args = ', '.join(keys + [' and '.join(reflexive) or 'True'])
    floating = cell.get_rettype_expr(c_style=True) == 'double'
    encode = 'value_word' if floating else 'int_word'
    decode = 'word_double' if floating else 'word_int'
    found = f'if {cache}.get({args}, _mx_cy.address({bits})):\n    return {ALIAS}.{decode}({bits})\n'
    compute = (f'{result} = self._f_{cell.name}({", ".join(cell.params)})\n'
               f'{cache}.put({args}, {ALIAS}.{encode}({result}))\nreturn {result}\n')
    declarations = f'{bits}: {ALIAS}.word\n{result}: {cell.get_rettype_expr()}\n'
    if cell.is_locked:
        import textwrap
        source = (declarations
                  + found + 'with self._mx_lock:\n' + textwrap.indent(found+compute, '    '))
    else:
        source = (declarations
                  + found + compute)
    return cst.IndentedBlock(body=cst.parse_module(source).body)
