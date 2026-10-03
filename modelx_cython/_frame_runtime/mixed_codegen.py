"""Mixed scalar boundary extensions over the fixed CF4 physical template.

Pure-native output is untouched. Python calculators finish Cython cleanup before
numeric code captures their error. Unfinished entries carry the owning invocation
depth; a failure scans and clears only entries from that invocation or its children.
"""


def busy_layout(rows):
    """Describe invocation ownership from the same semantic cache descriptors."""
    offset = 0
    spans = {}
    for name, row in rows.items():
        if (row['cached'] and row['recursive'] and row['direct_input'] is None
                and row.get('storage') != 'growing_numeric'
                and row.get('invocation_owner_required', True)):
            spans[name] = (row['offset'], row['offset'] + row['slots'], offset)
            offset += row['slots']
    return spans, max(1, offset)


def emit_invocation_state(frame, manifest, heap_numeric=False):
    """Emit guard/rollback state directly; Cell entries already publish ownership."""
    slots = max(1, manifest['cache_slots'])
    spans, size = busy_layout(manifest['cells'])
    busy_spans = list(spans.values())
    pure_spans = [(row['offset'], row['offset'] + row['slots'])
                  for row in manifest['cells'].values()
                  if row['cached'] and row['recursive'] and row['direct_input'] is None
                  and not row.get('invocation_owner_required', True)
                  and row.get('storage') != 'growing_numeric']
    if heap_numeric:
        manifest['busy_cache_slots'] = size
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
        frame += [f'    if {start} <= slot < {end}: return -1' for start, end in pure_spans]
        frame += ['    return slot']
    frame += [
              '\ncdef void _rollback(_Frame* state, unsigned int mark) noexcept nogil:',
              '    cdef Py_ssize_t slot, owner_slot',
              f'    for slot in range({slots}):',
              '        if state._seen[slot] == state._epoch - 1:',
              '            owner_slot = _busy_slot(slot)',
              '            if owner_slot < 0 or state._busy_owner[owner_slot] >= mark:',
              '                state._seen[slot] = state._epoch - 2 if _hash_slot(slot) else 0']
    manifest['mixed_cache_protocol'] = 'ready/busy/tombstone; rollback scan by invocation depth'
    return frame


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
