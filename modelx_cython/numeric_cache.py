"""Emission for ordinary Cells backed by the small numerical key store."""
import libcst as cst

from .helpers import module_name

PROVIDER = 'modelx_cython._numeric_cache'
ALIAS = module_name(PROVIDER)
TYPE = ALIAS + '.NumericCache'


def exclusively_owned(cell):
    # Exported ItemSpaces are partitioned by the caller. Shared Spaces use
    # locked_spaces. Keep the synchronized path for other receivers.
    owner = cell.parent
    return (not cell.is_locked and owner.name in owner.visitor.itemspace_classes
            and owner.name not in owner.visitor.dynamic_ref_classes)


def primary_capacity(cell):
    from . import automatic
    inferred = automatic.primary_capacity(cell) if automatic.enabled(cell) and exclusively_owned(cell) else 0
    size = cell._spec.get('numeric_cache_capacity', inferred)
    if type(size) is not int or (size and (
            not 2 <= size <= automatic.MAX_PRIMARY_ENTRIES or size & (size - 1) or not exclusively_owned(cell))):
        raise ValueError(f'{cell.fqname}: invalid capacity for a private numeric cache')
    return size


def reserved_body(cell, cache, keys, reflexive, bits, result, encode, decode):
    prefix = '_mx_reservation'
    while any(name.startswith(prefix) for name in cell.params):
        prefix += '_'
    words = [f'{prefix}_{i}' for i in range(len(keys))]
    slot = prefix + '_slot'
    args = ', '.join(words)
    source = (
        ''.join(f'{name}: {ALIAS}.word = {key}\n' for name, key in zip(words, keys))
        + f'{slot}: _mx_cy.Py_ssize_t = -1\n{bits}: {ALIAS}.word\n'
        + f'if {cache}.reserve({args}, {reflexive}, _mx_cy.address({slot}), '
          f'_mx_cy.address({bits})):\n    return {ALIAS}.{decode}({bits})\n'
        + f'{result}: {cell.get_rettype_expr()} = self._f_{cell.name}({", ".join(cell.params)})\n'
        + f'{cache}.publish({args}, {slot}, {ALIAS}.{encode}({result}))\nreturn {result}\n'
    )
    return cst.IndentedBlock(body=cst.parse_module(source).body)


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
    if primary_capacity(cell):
        return reserved_body(cell, cache, keys, ' and '.join(reflexive) or 'True',
                             bits, result, encode, decode)
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
