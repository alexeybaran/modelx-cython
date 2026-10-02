"""Public ABI adapters around the existing prepared Cython calculation graph.

Public declarations come from finalized original compiler metadata. Physical
signatures come from the kernel manifest. Adapters do not rewrite formulas,
own caches, select policies or prepare inputs. Numeric guards end before any
Python result conversion; object calls retain ordinary Cython cleanup.
"""


PUBLIC_TYPES = {'double': 'double', 'long long': 'long long', 'bint': 'bint',
                'object': 'object', 'str': 'str', 'unicode': 'str'}


def signature(name, cell):
    args = ['object kernel'] + [PUBLIC_TYPES[t] + ' _mx_public_arg_' + str(i)
                               for i, (_, t) in enumerate(cell.public_arguments)]
    result = PUBLIC_TYPES[cell.public_result]
    exception = '' if result in ('object', 'str') else ' except? -1'
    return f'cdef {result} _mx_public_{name}(' + ', '.join(args) + ')' + exception


def declarations(space, manifest):
    return '\n'.join(signature(n, space.cells[n]) for n in manifest['cells']) + '\n'


def augment(source, manifest, space):
    """Emit one checked public boundary; preserve all internal kernel entries.

    Category arguments are encoded explicitly rather than casting an object
    capsule to a numeric ABI. The current codec's domain checks remain visible
    in the manifest; this adapter alone is not general-object qualification.
    """
    numeric = [n for n in manifest['cells'] if n not in manifest.get('object_cells', ())]
    exports = []
    inventory = {}
    for name, row in manifest['cells'].items():
        cell = space.cells[name]
        if [p['name'] for p in row['params']] != [n for n, _ in cell.public_arguments]:
            raise ValueError('Public/physical parameter order differs: ' + cell.identity)
        head = [signature(name, cell) + ':', '    cdef Kernel owner = kernel']
        encoded = []
        for i, (param, (_, public)) in enumerate(zip(row['params'], cell.public_arguments)):
            info = param['type']
            variable = '_mx_native_arg_' + str(i)
            value = '_mx_public_arg_' + str(i)
            head.append('    cdef ' + info['ctype'] + ' ' + variable)
            encoded.append(variable)
        if name in numeric:
            head += ['    cdef _Command command']
        if row['result']['kind'] in ('enum', 'table_key'):
            head += ['    cdef ' + row['result']['ctype'] + ' _mx_native_result']
        body = ['    if not owner._frame._scope_valid:',
                '        raise RuntimeError("Select an evaluation scope first")']
        uses_codec = row['result']['kind'] in ('enum', 'table_key')
        for i, (param, (_, public)) in enumerate(zip(row['params'], cell.public_arguments)):
            info = param['type']
            variable, value = encoded[i], '_mx_public_arg_' + str(i)
            if public in ('object', 'str', 'unicode') and info['kind'] != 'object':
                uses_codec = True
                body.append(f'    {variable} = owner._python_scope.encode({info!r}, {value})')
            else:
                body.append(f'    {variable} = {value}')
        if uses_codec and not manifest.get('python_cells'):
            raise ValueError('Public category/key adaptation requires an owned checked codec')
        args = ', '.join(encoded)
        if name in numeric:
            body += ['    command.mode = 0', f'    command.identity = {numeric.index(name)}']
            for i, param in enumerate(row['params']):
                field = 'f' if param['type']['kind'] == 'float' else 'i'
                body.append(f'    command.args[{i}].{field} = {encoded[i]}')
            field = 'f' if row['result']['kind'] == 'float' else 'i'
            body += ['    owner._invoke(&command)']
            result = 'command.value.' + field
        else:
            body += ['    if owner._object_key_guard or owner._clearing_objects:',
                     '        raise RuntimeError("Cell calls during cache maintenance are unsupported")',
                     '    owner._frame._entry_depth += 1', '    try:',
                     f'        return owner._object_cell_{name}({args})',
                     '    finally:', '        owner._frame._entry_depth -= 1']
            result = None
        if result is not None:
            if row['result']['kind'] in ('enum', 'table_key'):
                # All native unwinding is complete before Python owns a value.
                body += [f'    _mx_native_result = {result}',
                         f'    return owner._python_scope.decode({row["result"]!r}, _mx_native_result)']
            else:
                body.append('    return ' + result)
        exports += head + body + ['']
        inventory[name] = dict(public_return=cell.public_result,
                               public_arguments=dict(cell.public_arguments),
                               physical_return=row['result'],
                               physical_arguments={p['name']: p['type'] for p in row['params']},
                               codec_at_public_boundary=uses_codec)
    manifest['public_entry_abi'] = inventory
    manifest['public_entry_limits'] = [
        'Category/key arguments use the existing checked physical domain; '
        'general-object, override and live-binding integration still required.']
    return source.rstrip() + '\n\n' + '\n'.join(exports) + '\n'
