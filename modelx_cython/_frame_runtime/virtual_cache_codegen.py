"""Override-aware, checked entries using the exact receiver's cache owner.

The callback reaches the original virtual private formula. It is invoked only
on a miss; no formula body or cache is duplicated. Native exact entries remain
unchanged, and Python-owning callbacks never unwind through the C++ guard.
"""


def signature(name, row):
    args = ['object kernel', 'object compute']
    args += [p['type']['ctype'] + ' _mx_arg_' + str(i) for i, p in enumerate(row['params'])]
    result = row['result']['ctype']
    exception = '' if result == 'object' else ' except? -1'
    return f'cdef {result} _mx_virtual_{name}(' + ', '.join(args) + ')' + exception


def entries(manifest):
    """Generate checked adapters around existing physical cache primitives."""
    methods, exports = [], []
    for name, row in manifest['cells'].items():
        params = [dict(p, name='_mx_arg_' + str(i)) for i, p in enumerate(row['params'])]
        args = ', '.join(p['name'] for p in params)
        sig = ', '.join(p['type']['ctype'] + ' ' + p['name'] for p in params)
        result = row['result']['ctype']
        exception = '' if result == 'object' else ' except? -1'
        call = f'compute({args})'
        head = (f'    cdef {result} _virtual_cell_{name}(self, object compute'
                + (', ' + sig if sig else '') + ')' + exception + ':')
        storage = row.get('storage', 'uncached')
        if not row['cached']:
            body = [head, '        return ' + call]
        elif storage == 'growing_numeric':
            pointer = ('self._frame' if manifest['physical_layout'] ==
                       'retained_heap_numeric_memo_frame_with_python_owners' else '&self._frame')
            keys = pointer + (', ' + args if args else '')
            body = [head, f'        cdef {result} value',
                    '        cdef Py_ssize_t slot = -1',
                    f'        if _nc_lookup_{name}({keys}, &value): return value']
            if row.get('native_prefix') and row['recursive']:
                body += [f'        slot = _nc_prefix_slot_{name}({keys})',
                         '        if slot >= 0: self._frame._seen[slot] = self._frame._epoch - 1']
            body += ['        try:',
                     '            value = ' + call,
                     f'            _nc_commit_{name}({keys}, value)', '            return value',
                     '        except BaseException:',
                     '            if slot >= 0 and self._frame._seen[slot] == self._frame._epoch - 1:',
                     '                self._frame._seen[slot] = 0','            raise']
        elif storage.startswith('typed_'):
            from .typed_memo_codegen import cached_entry
            body = cached_entry(name, row['typed_signature'], row,
                                manifest['steps'], manifest['diagnostics'], True,
                                virtual=True)
        elif storage in {'object_singleton', 'python_dict'}:
            from .object_codegen import _direct_dict_body
            body = _direct_dict_body(name, row, params, manifest['diagnostics'],
                                     virtual=True)
        elif storage == 'object_slots':
            body = [head, '        cdef _ObjectEntry entry', '        cdef Py_ssize_t index']
            index = '0'
            for p in params:
                n, low, size = p['name'], p['lower'], p['size']
                body += [f'        if {n} < {low} or {n} >= {low + size}:',
                         f'            raise IndexError("Outside declared cache bound: {name}.{n}")']
                index = f'(({index}) * {size} + ({n} - ({low})))'
            body += [f'        index = {index}', f'        entry = self._object_{name}[index]',
                     '        if entry is None:', '            entry = _ObjectEntry()',
                     f'            self._object_{name}[index] = entry',
                     '        if entry.state == 2: return entry.value',
                     '        if entry.state == 1:', f'            raise RuntimeError("Cell cycle: {name}")',
                     '        if not entry.touched:', '            self._object_dirty.append(entry)',
                     '            entry.touched = True', '        entry.state = 1',
                     '        try:', '            entry.value = ' + call,
                     '            entry.state = 2', '            return entry.value',
                     '        except BaseException:', '            entry.state = 0',
                     '            entry.value = None', '            raise']
        else:
            if row['hashed']:
                raise ValueError('Original-shell numeric dictionaries must use growing storage')
            field = '_floats' if row['result']['kind'] == 'float' else '_ints'
            body = [head, '        cdef Py_ssize_t index', f'        cdef {result} value']
            index = '0'
            for p in params:
                n, low, size = p['name'], p['lower'], p['size']
                body += [f'        if {n} < {low} or {n} >= {low + size}:',
                         f'            raise IndexError("Outside declared cache bound: {name}.{n}")']
                index = f'(({index}) * {size} + ({n} - ({low})))'
            body += [f'        index = {row["offset"]} + {index}',
                     '        if self._frame._seen[index] == self._frame._epoch:',
                     f'            return self._frame.{field}[index]']
            if row['recursive']:
                body += ['        if self._frame._seen[index] == self._frame._epoch - 1:',
                         f'            raise RuntimeError("Cell cycle: {name}")',
                         '        self._frame._seen[index] = self._frame._epoch - 1']
            body += ['        try:', '            value = ' + call,
                     f'            self._frame.{field}[index] = value',
                     '            self._frame._seen[index] = self._frame._epoch',
                     '            return value', '        except BaseException:']
            if row['recursive']:
                body.append('            self._frame._seen[index] = 0')
            body.append('            raise')
        methods.extend(body + [''])
        export = [signature(name, row) + ':', '    cdef Kernel owner = kernel']
        if manifest.get('object_cells'):
            export += ['    if owner._object_key_guard or owner._clearing_objects:',
                       '        raise RuntimeError("Cell calls during cache maintenance are unsupported")']
        # Numeric-only kernels do not carry mixed-object rollback fields.
        # Track their checked virtual callbacks on the owner, leaving the exact
        # native frame and recursive functions unchanged.
        depth = ('owner._frame._entry_depth' if manifest.get('python_cells')
                 else 'owner._mx_virtual_depth')
        export += [f'    {depth} += 1', '    try:',
                   f'        return owner._virtual_cell_{name}(compute' + (', ' + args if args else '') + ')',
                   '    finally:', f'        {depth} -= 1']
        exports.extend(export + [''])
    return methods, exports
