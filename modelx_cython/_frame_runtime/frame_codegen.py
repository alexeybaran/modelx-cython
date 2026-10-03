"""Emit final native-context and owner structures from the shared layout.

This is physical code generation over our fixed template, never analysis of
pandas or model expressions. Python owners keep buffers/Curve objects alive.
"""


def emit_context(parts, manifest):
    if manifest.get('object_cells'):
        from .object_codegen import augment
        # Numeric emission/guard analysis must never see Python-owned temporaries.
        physical = dict(manifest, cells={k: v for k, v in manifest['cells'].items()
                                        if k not in manifest['object_cells']})
        physical.pop('object_cells')
        source = emit_context(parts, physical)
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
    from .growing_numeric_codegen import frame_parts, owner_parts
    growing_fields, growing_helpers = frame_parts(manifest['cells'])
    frame.extend(growing_fields)
    growing_owner, growing_init, growing_reset, growing_free, growing_methods = owner_parts(
        manifest['cells'], frame_pointer='self._frame' if heap_numeric else '&self._frame')
    functions = []
    for method in parts['methods']:
        if any(line.startswith('cdef inline ') and '(_Frame* state' in line
               for line in method):
            functions.append('\n'.join(method))
        else:
            raise ValueError('Cell emission must use the physical frame receiver')
    from .owner_codegen import emit_prefix
    lowered = functions
    # Generated init has a chained assignment only for counters (numeric), not owners.
    manifest['physical_layout'] = ('retained_heap_numeric_memo_frame_with_python_owners'
                                   if heap_numeric else 'embedded_native_memo_frame_with_python_owners')
    if native_guard:
        from .native_guard_codegen import calculation_aliases, runtime as guard_runtime
        aliases = calculation_aliases(manifest['cells'])
        output_plan = parts['frame_outputs']
        length = output_plan['length']
        tables = output_plan['tables']
        if set(tables) != set(manifest['outputs']) or any(
                len(tables[name]) != len(columns)
                for name, columns in manifest['outputs'].items()):
            raise ValueError('Output plan differs from Cell compilation metadata')
        if manifest.get('python_cells'):
            from .mixed_codegen import emit_invocation_state
            frame = emit_invocation_state(frame, manifest, heap_numeric=heap_numeric)
        frame.extend(growing_helpers)
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
                             *('            ' + line for line in growing_free),
                             '            free(self._cache_storage)')
        elif growing_free:
            owner_dealloc = ('        mx_guard_clear(&self._frame._guard)',
                             *('        ' + line for line in growing_free))
        owner_init.extend(growing_init)
        manifest['native_error_abi'] = 'checked_cython_miss_with_cpp_guard'
        prefix = emit_prefix(manifest, objects, heap=heap_numeric,
                             growing_owner=growing_owner, resets=growing_reset,
                             methods=growing_methods, owner_init=owner_init, guarded=True)
        runtime_source = guard_runtime(manifest, prefix, length, tables,
                                       owner_dealloc=owner_dealloc)
        if manifest.get('python_cells'):
            manifest['native_error_abi'] = 'reentrant_guard_with_checked_python_miss'
        if heap_numeric:
            manifest['seen_tag_width'] = 'uint8_with_clear_before_wrap'
        return head + '\n' + '\n'.join(frame + aliases) + '\n\n' + '\n\n'.join(lowered) + '\n\n' + runtime_source + '\n'
    prefix = emit_prefix(manifest, objects, heap=heap_numeric,
                         growing_owner=growing_owner, resets=growing_reset, methods=growing_methods)
    frame.extend(growing_helpers)
    return head + '\n' + '\n'.join(frame) + '\n\n' + '\n\n'.join(lowered) + '\n\n' + marker + prefix
