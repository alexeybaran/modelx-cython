"""Lower the emitter-owned kernel layout to a plain C frame.

This is physical code generation over our fixed template, never analysis of
pandas or model expressions. Python owners keep buffers/Curve objects alive.
"""
import re
import io
import tokenize


def to_c_frame(source, manifest):
    if manifest.get('object_cells'):
        from .object_codegen import augment
        # Numeric emission/guard analysis must never see Python-owned temporaries.
        physical = dict(manifest, cells={k: v for k, v in manifest['cells'].items()
                                        if k not in manifest['object_cells']})
        physical.pop('object_cells')
        source = to_c_frame(source, physical)
        for key in ('physical_layout', 'native_error_abi', 'mixed_cache_protocol'):
            manifest[key] = physical[key]
        return augment(source, manifest)
    marker = '@cython.final\ncdef class Kernel:\n'
    head, body = source.split(marker, 1)
    native_guard = manifest['memo'] == 'dense' and manifest['lookups'] == 'native'
    lines = body.splitlines()
    end = next(i for i, x in enumerate(lines) if x.startswith('    def __init__('))
    declarations = lines[:end]
    fields = {}
    for line in declarations:
        match = re.fullmatch(r'    cdef (.+?) ([A-Za-z_]\w*(?:, *[A-Za-z_]\w*)*)', line)
        if not match:
            raise ValueError('Unexpected emitter declaration: ' + line)
        dtype, names = match.groups()
        for name in names.split(', '):
            if name != '_records':
                fields[name] = dtype
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
    frame = ['from cpython.ref cimport PyObject', 'from libc.string cimport memset']
    if native_guard:
        from .native_guard_codegen import HEADER
        frame.append(HEADER)
    frame.append('cdef struct _Frame:')
    if native_guard: frame.append('    MxGuard _guard')
    for name, dtype in fields.items():
        t = 'PyObject*' if name in objects else dtype.replace('[::1]', '*')
        if name in buffers:
            frame.append(f'    {dtype.replace("[::1]", "")} {name}[{sizes[name]}]')
        else:
            frame.append(f'    {t} {name}')
    # Extract numeric Cell functions (their decorators belong to that function).
    functions, outer = [], []
    i = end
    while i < len(lines):
        start = i
        while i < len(lines) and lines[i].startswith('    @'):
            i += 1
        if i < len(lines) and lines[i].startswith('    cdef inline ') and ('_cell_' in lines[i] or '_calc_' in lines[i]):
            signature = i
            i += 1
            while i < len(lines) and (lines[i].startswith('        ') or not lines[i].strip()):
                i += 1
            block = '\n'.join(x[4:] for x in lines[start:i])
            block = block.replace('(self, ', '(_Frame* state, ', 1).replace('(self)', '(_Frame* state)', 1)
            functions.append(block)
        else:
            outer.extend(lines[start:i+1])
            i += 1
    def lower_calls(text, pointer):
        text = re.sub(r'self\.(_cell_|_calc_)(\w+)\(\)', lambda m: f'{m[1]}{m[2]}({pointer})', text)
        return re.sub(r'self\.(_cell_|_calc_)(\w+)\(', lambda m: f'{m[1]}{m[2]}({pointer}, ', text)
    lowered = []
    for block in functions:
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
    owner_decls = ['    cdef _Frame _frame', '    cdef object _records']
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
    manifest['physical_layout'] = 'embedded_native_memo_frame_with_python_owners'
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
            frame, lowered = augment_frame(frame, lowered, manifest)
        manifest['native_error_abi'] = 'checked_cython_miss_with_cpp_guard'
        runtime_source = guard_runtime(manifest, prefix, length, tables)
        if manifest.get('python_cells'):
            from .mixed_codegen import augment_runtime
            runtime_source = augment_runtime(runtime_source, manifest)
            manifest['native_error_abi'] = 'reentrant_guard_with_checked_python_miss'
        return head + '\n' + '\n'.join(frame + aliases) + '\n\n' + '\n\n'.join(lowered) + '\n\n' + runtime_source + '\n'
    return head + '\n' + '\n'.join(frame) + '\n\n' + '\n\n'.join(lowered) + '\n\n' + marker + '\n'.join(owner_decls + result) + '\n'
