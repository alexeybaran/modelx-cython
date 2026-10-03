"""Read an existing completed dense numeric entry at checked boundaries.

Only the existing key domain and validity tag are consulted. No allocation,
formula, input refresh, commit, error protocol or second cache lives here.
Unsupported stores and misses continue through their normal checked entry.
"""


def completed_dense_hit(row, manifest, *, owner, arguments, indent):
    if (manifest.get('diagnostics') or not row['cached'] or row['hashed']
            or row.get('storage') is not None or not row['slots']
            or row['result']['kind'] not in ('float', 'int', 'bool')
            or row.get('direct_input') is not None
            or any(p['type']['kind'] not in ('int', 'enum', 'bool') for p in row['params'])):
        return []
    # Both original-shell and prepared public graphs use this mixed ownership
    # ABI. A completed numeric read owns no Python temporary and cannot throw;
    # misses still enter the scoped guard after Cython's object cleanup boundary.
    if not manifest.get('python_cells'):
        return []
    prefix = ' ' * indent
    valid = [owner + '._frame._scope_valid']
    if manifest.get('object_cells'):
        valid += ['not ' + owner + '._object_key_guard',
                  'not ' + owner + '._clearing_objects']
    index = '0'
    for p, argument in zip(row['params'], arguments):
        lower, size = p['lower'], p['size']
        valid += [f'{lower} <= {argument} < {lower + size}']
        index = f'(({index}) * {size} + ({argument} - ({lower})))'
    index = f'{row["offset"]} + {index}'
    field = '_floats' if row['result']['kind'] == 'float' else '_ints'
    return [prefix + 'cdef Py_ssize_t _mx_completed_key',
            prefix + 'if ' + ' and '.join(valid) + ':',
            prefix + '    _mx_completed_key = ' + index,
            prefix + '    if ' + owner + '._frame._seen[_mx_completed_key] == ' + owner + '._frame._epoch:',
            prefix + '        return ' + owner + '._frame.' + field + '[_mx_completed_key]']
