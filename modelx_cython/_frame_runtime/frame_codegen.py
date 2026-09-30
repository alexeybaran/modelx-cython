"""Lower the emitter-owned kernel layout to a plain C frame.

This is physical code generation over our fixed template, never analysis of
pandas or model expressions. Python owners keep buffers/Curve objects alive.
"""
import re
import io
import tokenize


def to_c_frame(parts, manifest):
    if manifest.get('object_cells'):
        from .object_codegen import augment
        # Numeric emission/guard analysis must never see Python-owned temporaries.
        physical = dict(manifest, cells={k: v for k, v in manifest['cells'].items()
                                        if k not in manifest['object_cells']})
        physical.pop('object_cells')
        source = to_c_frame(parts, physical)
        for key in ('physical_layout', 'native_error_abi', 'mixed_cache_protocol'):
            manifest[key] = physical[key]
        return augment(source, manifest)
    marker = '@cython.final\ncdef class Kernel:\n'
    head = '\n'.join(parts['header']) + '\n'
    fields = parts['field_types']
    native_guard = manifest['memo'] == 'dense' and manifest['lookups'] == 'native'
    objects = {k: v for k, v in fields.items() if v in {
        'object', 'Curve', 'CurveTable', 'IndexedInt1', 'IndexedFloat1',
        'IndexedInt2', 'IndexedFloat2', 'dict', 'set'}}
    buffers = {k: v for k, v in fields.items() if '[::1]' in v}
    numeric = {k: v for k, v in fields.items() if k not in objects and k not in buffers}
    sizes = {name: max(1, manifest['cache_slots']) if manifest['memo'] == 'dense' else 1
             for name in buffers}
    for cell, row in manifest['cells'].items():
        if row['hashed'] and manifest['memo'] == 'dense':
            for param in row['params']:
                sizes['_keys_' + cell + '_' + param['name']] = row['slots']
    # Large retained frames pay for embedded numeric and in-progress arrays at
    # Space construction, even though validity flags gate every value read.
    # Keep the flags inline and own the other arrays in one uninitialised block.
    # Each cache slot has one statically known result type, so float and integer
    # values can occupy the same 8-byte slot rather than two parallel arrays.
    # Small frames stay inline to avoid adding an allocation to cheap Cells.
    heap_numeric = native_guard and manifest['memo'] == 'dense' and manifest['cache_slots'] >= 1024
    frame = ['from cpython.ref cimport PyObject', 'from libc.string cimport memset']
    if heap_numeric:
        frame.append('from libc.stdlib cimport malloc, free')
    if native_guard:
        from .native_guard_codegen import HEADER
        frame.append(HEADER)
    frame.append('cdef struct _Frame:')
    if native_guard: frame.append('    MxGuard _guard')
    for name, dtype in fields.items():
        t = 'PyObject*' if name in objects else dtype.replace('[::1]', '*')
        if heap_numeric and name == '_seen':
            frame.append(f'    unsigned char _seen[{sizes[name]}]')
        elif heap_numeric and name in ('_floats', '_ints'):
            frame.append(f'    {dtype.replace("[::1]", "")}* {name}')
        elif name in buffers:
            frame.append(f'    {dtype.replace("[::1]", "")} {name}[{sizes[name]}]')
        else:
            frame.append(f'    {t} {name}')
    functions = []
    direct_entries = set()
    for method in parts['methods']:
        if any(line.startswith('cdef inline ') and '(_Frame* state' in line
               for line in method):
            # The emitter already chose the physical receiver for this Cell
            # entry or formula. Do not re-parse or rewrite its source.
            functions.append('\n'.join(method))
            direct_entries.add(len(functions) - 1)
        else:
            functions.append(('\n'.join(x[4:] for x in method)
                              .replace('(self, ', '(_Frame* state, ', 1)
                              .replace('(self)', '(_Frame* state)', 1)))
    outer = parts['runtime']
    def lower_calls(text, pointer):
        text = re.sub(r'self\.(_cell_|_calc_)(\w+)\(\)', lambda m: f'{m[1]}{m[2]}({pointer})', text)
        return re.sub(r'self\.(_cell_|_calc_)(\w+)\(', lambda m: f'{m[1]}{m[2]}({pointer}, ', text)
    lowered = []
    for block_index, block in enumerate(functions):
        if block_index in direct_entries:
            lowered.append(block)
            continue
        offsets = [0]
        for ln in block.splitlines(keepends=True): offsets.append(offsets[-1] + len(ln))
        literals = []
        for tok in tokenize.generate_tokens(io.StringIO(block).readline):
            if tok.type in (tokenize.STRING, getattr(tokenize, 'FSTRING_MIDDLE', -999)):
                start = offsets[tok.start[0]-1] + tok.start[1]
                stop = offsets[tok.end[0]-1] + tok.end[1]
                literals.append((start, stop, tok.string))
        for i, (start, stop, literal) in reversed(list(enumerate(literals))):
            block = block[:start] + f'__mx_literal_{i}__' + block[stop:]
        block = lower_calls(block, 'state')
        for name in sorted(fields, key=len, reverse=True):
            target = f'(<{objects[name]}>state.{name})' if name in objects else 'state.' + name
            block = re.sub(r'\bself\.' + re.escape(name) + r'\b', lambda _: target, block)
        if re.search(r'\bself\.', block):
            raise ValueError('Numeric Cell retained a Python owner reference')
        for i, (_, _, literal) in enumerate(literals):
            block = block.replace(f'__mx_literal_{i}__', literal)
        lowered.append(block)
    # Runtime keeps the ownership fields; scalar values have one location in _Frame.
    owner_decls = ['    cdef _Frame* _frame' if heap_numeric else '    cdef _Frame _frame',
                   '    cdef object _records']
    if heap_numeric:
        owner_decls.append('    cdef void* _cache_storage')
    for name, dtype in objects.items():
        owner_decls.append(f'    cdef {dtype} {name}')
    runtime = lower_calls('\n'.join(outer), '&self._frame')
    for name in {**numeric, **buffers}:
        runtime = re.sub(r'\bself\.' + re.escape(name) + r'\b', lambda _: 'self._frame.' + name, runtime)
    # Synchronize the raw pointer immediately after an owning buffer/object binding.
    result = []
    for line in runtime.splitlines():
        if any(line.strip().startswith('self._frame.' + n + ' = np.') for n in buffers):
            if line.strip().startswith('self._frame._seen ='):
                result.append('        memset(self._frame._seen, 0, sizeof(self._frame._seen))')
            continue
        line = line.replace('np.asarray(self._frame._seen).fill(0)',
                            'memset(self._frame._seen, 0, sizeof(self._frame._seen))')
        result.append(line)
        match = re.match(r'(\s*)self\.(\w+) = ', line)
        if match:
            indent, name = match.groups()
            if name in buffers:
                result.append(f'{indent}self._frame.{name} = &self.{name}[0]')
            elif name in objects:
                result.append(f'{indent}self._frame.{name} = <PyObject*>self.{name}')
    # Generated init has a chained assignment only for counters (numeric), not owners.
    manifest['physical_layout'] = ('retained_heap_numeric_memo_frame_with_python_owners'
                                   if heap_numeric else 'embedded_native_memo_frame_with_python_owners')
    if native_guard:
        from .native_guard_codegen import guarded_functions, runtime as guard_runtime
        aliases, lowered = guarded_functions(lowered, manifest)
        text = '\n'.join(result)
        prefix = '\n'.join(owner_decls) + '\n' + text.split('    def evaluate(', 1)[0]
        def expression(expr):
            expr = expr.replace('&self._frame', 'state').replace('self._frame.', 'state.')
            for name, dtype in objects.items():
                expr = re.sub(r'\bself\.' + re.escape(name) + r'\b',
                              lambda _: f'(<{dtype}>state.{name})', expr)
            if 'self.' in expr: raise ValueError('Unlowered output binding')
            return expr
        length = expression(re.search(r'^        n = (.+)$', text, re.M)[1]) if manifest['outputs'] else '0'
        tables = {}
        for table in manifest['outputs']:
            part = re.split(r'^    (?:def|cdef) ', text.split('    cdef object _table_' + table + '(self):',1)[1], maxsplit=1, flags=re.M)[0]
            tables[table] = [expression(m[1]) for m in re.finditer(r'^            out\[t, \d+\] = (.+)$', part, re.M)]
            if len(tables[table]) != len(manifest['outputs'][table]): raise ValueError('Output template mismatch')
        if manifest.get('python_cells'):
            from .mixed_codegen import augment_frame
            frame, lowered = augment_frame(frame, lowered, manifest, heap_numeric=heap_numeric)
        owner_init = []
        owner_dealloc = None
        if heap_numeric:
            slots = max(1, manifest['cache_slots'])
            has_busy = bool(manifest.get('python_cells'))
            busy_slots = manifest.get('busy_cache_slots', slots) if has_busy else 0
            byte_count = f'sizeof(_Frame) + {slots} * sizeof(double)' + (f' + {busy_slots} * sizeof(unsigned int)' if has_busy else '')
            owner_init = ['        if sizeof(double) != sizeof(long long): raise RuntimeError("Unsupported numeric cache layout")',
                          '        if self._cache_storage == NULL:',
                          f'            self._cache_storage = malloc({byte_count})',
                          '            if self._cache_storage == NULL: raise MemoryError()',
                          '        self._frame = <_Frame*>self._cache_storage',
                          '        memset(self._frame, 0, sizeof(_Frame))',
                          '        self._frame._floats = <double*>(<char*>self._cache_storage + sizeof(_Frame))',
                          '        self._frame._ints = <long long*>(<char*>self._cache_storage + sizeof(_Frame))']
            if has_busy:
                owner_init.append(f'        self._frame._busy_owner = <unsigned int*>(<char*>self._cache_storage + sizeof(_Frame) + {slots} * sizeof(double))')
            owner_dealloc = ('        if self._frame != NULL:',
                             '            mx_guard_clear(&self._frame._guard)',
                             '            free(self._cache_storage)')
        manifest['native_error_abi'] = 'checked_cython_miss_with_cpp_guard'
        runtime_source = guard_runtime(manifest, prefix, length, tables,
                                       owner_init=owner_init,
                                       owner_dealloc=owner_dealloc)
        if manifest.get('python_cells'):
            from .mixed_codegen import augment_runtime
            runtime_source = augment_runtime(runtime_source, manifest)
            manifest['native_error_abi'] = 'reentrant_guard_with_checked_python_miss'
        if heap_numeric:
            # The epoch advances by three for ready/busy/tombstone states.
            # Reset before unsigned-byte tags can wrap or alias an older state.
            runtime_source = runtime_source.replace('self._frame._epoch >= 4294967293',
                                                    'self._frame._epoch >= 252')
            runtime_source = runtime_source.replace('self._frame._epoch >= 4294967294',
                                                    'self._frame._epoch >= 252')
            manifest['seen_tag_width'] = 'uint8_with_clear_before_wrap'
        return head + '\n' + '\n'.join(frame + aliases) + '\n\n' + '\n\n'.join(lowered) + '\n\n' + runtime_source + '\n'
    return head + '\n' + '\n'.join(frame) + '\n\n' + '\n\n'.join(lowered) + '\n\n' + marker + '\n'.join(owner_decls + result) + '\n'
