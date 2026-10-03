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


def emit_scope_protocol(frame_pointer):
    """Emit the checked selection/nested-call protocol without rewriting source."""
    return f'''    cdef void _require_idle(self) except *:
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
'''


def emit_direct_roots(manifest):
    """Generate diagnostic native roots from the same dispatch metadata."""
    indices = {name: i for i, name in enumerate(manifest['cells'])}
    direct = []
    for name in manifest.get('native_scalar_outputs', ()):
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
    return '\n'.join(direct)
