"""Mixed scalar boundary extensions over the fixed CF4 physical template.

Pure-native output is untouched. Python calculators finish Cython cleanup before
numeric code captures their error. Nested entries save/restore guard ownership
and roll back only unfinished cache entries created since that entry's mark.
"""
import re


def augment_frame(frame, functions, manifest):
    slots = max(1, manifest['cache_slots'])
    frame += [f'    Py_ssize_t _busy_slots[{slots}]', f'    unsigned char _busy_hash[{slots}]',
              '    Py_ssize_t _busy_depth', '    unsigned int _entry_depth',
              '    bint _scope_valid']
    frame += ['\ncdef void _rollback(_Frame* state, Py_ssize_t mark) noexcept nogil:',
              '    cdef Py_ssize_t slot',
              '    while state._busy_depth > mark:',
              '        state._busy_depth -= 1',
              '        slot = state._busy_slots[state._busy_depth]',
              '        if state._seen[slot] == state._epoch - 1:',
              '            state._seen[slot] = state._epoch - 2 if state._busy_hash[state._busy_depth] else 0']
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
                lines += [indent + f'if state._busy_depth >= {slots}: mx_fail(&state._guard, 4, b"active cache stack")',
                          indent + 'state._busy_slots[state._busy_depth] = _mx_key',
                          indent + f'state._busy_hash[state._busy_depth] = {int(row["hashed"])}',
                          indent + 'state._busy_depth += 1']
            lines.append(line)
            if row['recursive'] and line.strip() == 'state._seen[_mx_key] = state._epoch':
                lines.append(indent + 'state._busy_depth -= 1')
        result.append('\n'.join(lines))
    manifest['mixed_cache_protocol'] = 'ready/busy/tombstone; rollback by nested-entry watermark'
    return frame, result


def augment_runtime(source, manifest):
    # Three distinct epoch states: READY, BUSY, failed hash tombstone. Previous
    # epochs cannot alias tombstones. Failed slots retain keys so probe chains and
    # successfully cached collision neighbours survive a caught exception.
    source = source.replace('if self._frame._epoch >= 4294967294:', 'if self._frame._epoch >= 4294967293:')
    source = source.replace('self._frame._epoch += 2', 'self._frame._epoch += 3')
    for signature in ('    def reset(self):', '    def replace_records(self, records):'):
        source = source.replace(signature, signature + '\n        self._require_idle()\n        self._frame._scope_valid = False')
    start = source.index('    cdef void _invoke(')
    end = source.index('    def evaluate(', start)
    source = source[:start] + '''    cdef void _require_idle(self) except *:
        if self._frame._entry_depth:
            raise RuntimeError("Cannot reset, rebind or switch scope during a Cell evaluation")

    def set_python_scope(self, scope):
        self._require_idle()
        self._python_scope = scope
        self._frame._python_scope = <PyObject*>scope
        self._frame._scope_valid = False

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

    cdef void _invoke(self, _Command* command) except *:
        cdef MxGuard saved = self._frame._guard
        cdef Py_ssize_t mark = self._frame._busy_depth
        mx_guard_init(&self._frame._guard)
        self._frame._entry_depth += 1
        try:
            if mx_guarded(&self._frame._guard, _execute, &self._frame, command) != 0:
                _rollback(&self._frame, mark)
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
    a = source.index('    def evaluate(')
    b = source.index('    def run_scalar(', a)
    call = source[a:b].replace('def evaluate(self, Py_ssize_t policy, str cell, tuple args):',
                              'def call(self, str cell, tuple args):')
    call = call.replace('        if policy < 0 or policy >= len(self._records): raise IndexError(policy)\n', '')
    call = call.replace('        self._select_policy(policy)',
                        '        if not self._frame._scope_valid: raise RuntimeError("Select an evaluation scope first")')
    source = source[:b] + call + source[b:]
    return source
