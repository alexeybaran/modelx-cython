"""Emission for ordinary Cells backed by the small numerical key store."""
import libcst as cst

from .helpers import module_name

PROVIDER = 'modelx_cython._numeric_cache'
ALIAS = module_name(PROVIDER)
TYPE = ALIAS + '.NumericCache'
WIDE_PROVIDER = 'modelx_cython._wide_numeric_cache'
WIDE_ALIAS = module_name(WIDE_PROVIDER)
WIDE_TYPE = WIDE_ALIAS + '.WideNumericCache'


def storage(cell):
    return WIDE_TYPE if len(cell.params) > 3 else TYPE


def providers(module):
    needed = set()
    for cls in module.classes.values():
        for cell in cls.cells.values():
            if not cell.is_special() and cell.uses_native_numeric_cache:
                needed.add(PROVIDER)
                if len(cell.params) > 3:
                    needed.add(WIDE_PROVIDER)
    return needed


def constructor(cell):
    if len(cell.params) > 3:
        from . import automatic
        owned = exclusively_owned(cell)
        capacity = max(8, automatic.primary_capacity(cell)) if owned and automatic.enabled(cell) else 8
        return f'{WIDE_TYPE}({len(cell.params)}, {owned}, {capacity})'
    return f'{TYPE}({primary_capacity(cell) or ""})'


def exclusively_owned(cell):
    # Exported ItemSpaces are partitioned by the caller. Shared Spaces use
    # locked_spaces. Keep the synchronized path for other receivers.
    owner = cell.parent
    return (not cell.is_locked and owner.name in owner.visitor.itemspace_classes
            and owner.name not in owner.visitor.dynamic_ref_classes)


def primary_capacity(cell):
    if len(cell.params) > 3:
        if 'numeric_cache_capacity' in cell._spec:
            raise ValueError(f'{cell.fqname}: numeric_cache_capacity applies to compact numeric keys')
        return 0
    from . import automatic
    inferred = automatic.primary_capacity(cell) if automatic.enabled(cell) and exclusively_owned(cell) else 0
    size = cell._spec.get('numeric_cache_capacity', inferred)
    if type(size) is not int or (size and (
            not 2 <= size <= automatic.MAX_PRIMARY_ENTRIES or size & (size - 1) or not exclusively_owned(cell))):
        raise ValueError(f'{cell.fqname}: invalid capacity for a private numeric cache')
    return size


def _fresh(cell, name):
    while name in cell.params:
        name += '_'
    return name


def dense_prefix(cell, cache, keys, encode, decode):
    """D74: a Cell given a dense window on the reused collect root reads and writes the window inline;
    keys outside it, and every receiver without a window, take the hash store that follows."""
    if len(cell.params) > 3 or cell.name not in getattr(getattr(cell, 'parent', None), 'reuse_dense_windows', {}):
        return ''
    index, value = _fresh(cell, '_mx_dense_index'), _fresh(cell, '_mx_dense_result')
    local = _fresh(cell, '_mx_dense_cache')
    # one typed local: the inline accessors then take it borrowed (no reference count per access)
    return (f'{local}: {TYPE} = {cache}\n'
            f'{index}: _mx_cy.Py_ssize_t = {ALIAS}.dense_find({local}, {", ".join(keys[:3])})\n'
            f'if {index} >= 0:\n'
            f'    if {ALIAS}.dense_ready({local}, {index}):\n'
            f'        return {ALIAS}.{decode}({ALIAS}.dense_value({local}, {index}))\n'
            f'    {value}: {cell.get_rettype_expr()} = self._f_{cell.name}({", ".join(cell.params)})\n'
            f'    {ALIAS}.dense_put({local}, {index}, {ALIAS}.{encode}({value}))\n'
            f'    return {value}\n')


def reserved_body(cell, cache, keys, reflexive, bits, result, encode, decode):
    prefix = '_mx_reservation'
    while any(name.startswith(prefix) for name in cell.params):
        prefix += '_'
    words = [f'{prefix}_{i}' for i in range(len(keys))]
    slot = prefix + '_slot'
    args = ', '.join(words)
    source = (
        dense_prefix(cell, cache, keys, encode, decode)
        + ''.join(f'{name}: {ALIAS}.word = {key}\n' for name, key in zip(words, keys))
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
    key_setup = ''
    if len(keys) > 3:
        key_array = '_mx_cache_key'
        while key_array in cell.params:
            key_array += '_'
        key_hash = '_mx_cache_hash'
        while key_hash in cell.params:
            key_hash += '_'
        key_setup = f'{key_array}: {ALIAS}.word[{len(keys)}]\n{key_hash}: {ALIAS}.word = 0\n'
        for index, name in enumerate(cell.params):
            kind = cell.get_argtype_expr(name, c_style=True)
            key = f'{WIDE_ALIAS}.key_word({name})' if kind == 'double' else f'_mx_cy.cast({ALIAS}.word, {name})'
            key_setup += (f'{key_array}[{index}] = {key}\n'
                          f'{key_hash} = {WIDE_ALIAS}.hash_step({key_hash}, {key_array}[{index}], {index})\n')
        args = f'_mx_cy.address({key_array}[0]), {" and ".join(reflexive) or "True"}, {key_hash}'
        if exclusively_owned(cell):
            slot = key_array + '_slot'
            while slot in cell.params:
                slot += '_'
            source = (key_setup + f'{bits}: {ALIAS}.word\n{slot}: _mx_cy.Py_ssize_t\n'
                      + f'if {cache}.reserve({args}, _mx_cy.address({slot}), '
                        f'_mx_cy.address({bits})):\n    return {ALIAS}.{decode}({bits})\n'
                      + f'{result}: {cell.get_rettype_expr()} = self._f_{cell.name}({", ".join(cell.params)})\n'
                      + f'{cache}.publish(_mx_cy.address({key_array}[0]), {key_hash}, '
                        f'{slot}, {ALIAS}.{encode}({result}))\nreturn {result}\n')
            return cst.IndentedBlock(body=cst.parse_module(source).body)
    elif primary_capacity(cell):
        return reserved_body(cell, cache, keys, ' and '.join(reflexive) or 'True',
                             bits, result, encode, decode)
    found = f'if {cache}.get({args}, _mx_cy.address({bits})):\n    return {ALIAS}.{decode}({bits})\n'
    compute = (f'{result} = self._f_{cell.name}({", ".join(cell.params)})\n'
               f'{cache}.put({args}, {ALIAS}.{encode}({result}))\nreturn {result}\n')
    declarations = (dense_prefix(cell, cache, keys, encode, decode) if len(keys) <= 3 else '') + (
        f'{bits}: {ALIAS}.word\n{result}: {cell.get_rettype_expr()}\n' + key_setup)
    if cell.is_locked:
        import textwrap
        source = (declarations
                  + found + 'with self._mx_lock:\n' + textwrap.indent(found+compute, '    '))
    else:
        source = (declarations
                  + found + compute)
    return cst.IndentedBlock(body=cst.parse_module(source).body)
