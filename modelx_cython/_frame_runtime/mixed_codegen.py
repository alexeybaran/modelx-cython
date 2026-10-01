"""Mixed scalar boundary extensions over the fixed CF4 physical template.

Pure-native output is untouched. Python calculators finish Cython cleanup before
numeric code captures their error. Unfinished entries carry the owning invocation
depth; a failure scans and clears only entries from that invocation or its children.
"""
import re


def augment_frame(frame, functions, manifest, heap_numeric=False):
    slots = max(1, manifest['cache_slots'])
    busy_spans = []
    if heap_numeric:
        offset = 0
        for function in functions:
            match = re.search(r'cdef inline .+? _cell_(\w+)\(', function)
            if not match or 'state._seen[_mx_key] = state._epoch - 1' not in function:
                continue
            row = manifest['cells'][match[1]]
            busy_spans.append((row['offset'], row['offset'] + row['slots'], offset))
            offset += row['slots']
        manifest['busy_cache_slots'] = max(1, offset)
    busy_field = '    unsigned int* _busy_owner' if heap_numeric else f'    unsigned int _busy_owner[{slots}]'
    frame += [busy_field, '    unsigned int _entry_depth',
              '    bint _scope_valid']
    hash_spans = sorted((row['offset'], row['offset'] + row['slots'])
                        for row in manifest['cells'].values()
                        if row.get('hashed') and 'offset' in row)
    frame += ['\ncdef inline bint _hash_slot(Py_ssize_t slot) noexcept nogil:']
    frame += [f'    if {start} <= slot < {end}: return True'
              for start, end in hash_spans]
    frame += ['    return False',
              '\ncdef inline Py_ssize_t _busy_slot(Py_ssize_t slot) noexcept nogil:']
    if heap_numeric:
        frame += [f'    if {start} <= slot < {end}: return slot - {start} + {compact}'
                  for start, end, compact in busy_spans]
        frame += ['    return -1']
    else:
        frame += ['    return slot']
    frame += [
              '\ncdef void _rollback(_Frame* state, unsigned int mark) noexcept nogil:',
              '    cdef Py_ssize_t slot, owner_slot',
              f'    for slot in range({slots}):',
              '        if state._seen[slot] == state._epoch - 1:',
              '            owner_slot = _busy_slot(slot)',
              '            if owner_slot >= 0 and state._busy_owner[owner_slot] >= mark:',
              '                state._seen[slot] = state._epoch - 2 if _hash_slot(slot) else 0']
    result = []
    for function in functions:
        match = re.search(r'cdef inline .+? _cell_(\w+)\(', function)
        if not match:
            result.append(function)
            continue
        row = manifest['cells'][match[1]]
        if row['hashed']:
            function = function.replace(
                'if state._seen[_mx_key] != state._epoch and state._seen[_mx_key] != state._epoch - 1:',
                'if state._seen[_mx_key] != state._epoch and state._seen[_mx_key] != state._epoch - 1 and state._seen[_mx_key] != state._epoch - 2:')
        lines = []
        for line in function.splitlines():
            indent = line[:len(line) - len(line.lstrip())]
            if line.strip() == 'state._seen[_mx_key] = state._epoch - 1':
                owner_key = '_mx_key'
                if heap_numeric:
                    owner_span = next((span for span in busy_spans
                                       if span[0] == row['offset'] and span[1] == row['offset'] + row['slots']), None)
                    if owner_span is None:
                        raise ValueError('Busy Cell missing compact owner span')
                    owner_key = f'(_mx_key - {owner_span[0]} + {owner_span[2]})'
                lines.append(indent + f'state._busy_owner[{owner_key}] = state._entry_depth')
            lines.append(line)
        result.append('\n'.join(lines))
    manifest['mixed_cache_protocol'] = 'ready/busy/tombstone; rollback scan by invocation depth'
    return frame, result


def augment_runtime(source, manifest):
    frame_pointer = ('self._frame' if manifest.get('physical_layout')
                     == 'retained_heap_numeric_memo_frame_with_python_owners'
                     else '&self._frame')
    # Three distinct epoch states: READY, BUSY, failed hash tombstone. Previous
    # epochs cannot alias tombstones. Failed slots retain keys so probe chains and
    # successfully cached collision neighbours survive a caught exception.
    source = source.replace('if self._frame._epoch >= 4294967294:', 'if self._frame._epoch >= 4294967293:')
    source = source.replace('self._frame._epoch += 2', 'self._frame._epoch += 3')
    for signature in ('    def reset(self):', '    def replace_records(self, records):'):
        source = source.replace(signature, signature + '\n        self._require_idle()\n        self._frame._scope_valid = False')
    start = source.index('    cdef void _invoke(')
    end = source.index('    def evaluate(', start)
    source = source[:start] + f'''    cdef void _require_idle(self) except *:
        if self._frame._entry_depth:
            raise RuntimeError("Cannot reset, rebind or switch scope during a Cell evaluation")

    def set_python_scope(self, scope):
        self._require_idle()
        self._python_scope = scope
        self._frame._python_scope = <PyObject*>scope
        self._frame._scope_valid = False

    cdef void _select_native_policy(self, Py_ssize_t policy) except *:
        self._require_idle()
        if policy < 0 or policy >= len(self._records):
            raise IndexError(policy)
        # A native-only aggregate has no object Cell or Python receiver edge.
        # Leave the object scope invalid so a later public object entry cannot
        # observe bindings from a previously selected policy.
        self._frame._scope_valid = False
        self._select(self._records[policy])

    cdef void _select_policy(self, Py_ssize_t policy) except *:
        self._require_idle()
        if policy < 0 or policy >= len(self._records):
            raise IndexError(policy)
        self._frame._scope_valid = False
        self._select(self._records[policy])
        if self._python_scope is None:
            raise RuntimeError("Detached Python calculator not bound")
        self._python_scope.select(self._records[policy], policy)
        self._frame._scope_valid = True

    def select_scope(self, Py_ssize_t policy):
        self._select_policy(policy)

    def select_native_scope(self, Py_ssize_t policy):
        self._select_native_policy(policy)

    cdef void _invoke(self, _Command* command) except *:
        cdef MxGuard saved = self._frame._guard
        cdef unsigned int mark = self._frame._entry_depth + 1
        mx_guard_init(&self._frame._guard)
        self._frame._entry_depth += 1
        try:
            if mx_guarded(&self._frame._guard, _execute, {frame_pointer}, command) != 0:
                _rollback({frame_pointer}, mark)
                _raise_guard(&self._frame._guard)
        finally:
            self._frame._entry_depth -= 1
            mx_guard_clear(&self._frame._guard)
            self._frame._guard = saved
''' + source[end:]
    # Every public root selects once. call() below is the only nested entry, and
    # deliberately never selects/rebinds/advances an epoch.
    source = re.sub(r'self\._select\(self\._records\[(policy|i)\]\)', r'self._select_policy(\1)', source)
    # Undo the replacement in the implementation of _select_policy itself.
    source = source.replace('        self._frame._scope_valid = False\n        self._select_policy(policy)',
                            '        self._frame._scope_valid = False\n        self._select(self._records[policy])')
    native_roots = manifest.get('native_scalar_outputs', ())
    if native_roots:
        a = source.index('    def run_scalar(')
        b = source.index('    def run_scalar_one(', a)
        block = source[a:b]
        if block.count('            self._select_policy(i)') != 1:
            raise ValueError('Unexpected scalar traversal template')
        native_test = ' or '.join(f'name == {name!r}' for name in native_roots)
        block = block.replace('        cdef _Command command',
                              '        cdef bint native_only = ' + native_test + '\n        cdef _Command command', 1)
        block = block.replace('            self._select_policy(i)',
                              '            if native_only:\n                self._select_native_policy(i)\n'
                              '            else:\n                self._select_policy(i)', 1)
        source = source[:a] + block + source[b:]
    a = source.index('    def evaluate(')
    b = source.index('    def run_scalar(', a)
    call = source[a:b].replace('def evaluate(self, Py_ssize_t policy, str cell, tuple args):',
                              'def call(self, str cell, tuple args):')
    call = call.replace('        if policy < 0 or policy >= len(self._records): raise IndexError(policy)\n', '')
    call = call.replace('        self._select_policy(policy)',
                        '        if not self._frame._scope_valid: raise RuntimeError("Select an evaluation scope first")')
    source = source[:b] + call + source[b:]
    # Internal direct-root diagnostic: one retained Kernel per policy and one
    # checked Cython entry, without the Python ItemSpace/CellCodec layers.
    # This is not exposed as a model Cell and does not replace normal dispatch.
    indices = {name: i for i, name in enumerate(manifest['cells'])}
    direct = []
    for name in native_roots:
        row = manifest['cells'][name]
        value = 'f' if row['result']['kind'] == 'float' else 'i'
        direct += [f'    def _native_direct_{name}(self):',
                   '        cdef _Command command',
                   '        if len(self._records) != 1: raise RuntimeError("Direct root needs one retained policy")',
                   '        if self._frame._epoch == 0: self._select_native_policy(0)',
                   '        command.mode = 0',
                   f'        command.identity = {indices[name]}',
                   '        self._invoke(&command)',
                   f'        return command.value.{value}', '']
    if direct:
        source += '\n' + '\n'.join(direct)
    return source
