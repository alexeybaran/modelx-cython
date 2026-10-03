"""Direct Cython owner emission; no temporary memoryview class or source rewriting.

Object fields retain ownership. Native fields live only in the context; guards
and representation-specific initialization are composed by the context emitter.
"""

def emit_prefix(manifest, objects, *, heap, growing_owner=(), resets=(), methods=(),
                owner_init=(), guarded=False):
    """Emit final owner fields and lifecycle methods from resolved cache metadata."""
    declarations = ['    cdef _Frame* _frame' if heap else '    cdef _Frame _frame','    cdef object _records']
    if heap: declarations.append('    cdef void* _cache_storage')
    declarations += [f'    cdef {tp} {name}' for name,tp in objects.items()]
    declarations += growing_owner
    lines = ['    def __init__(self, records):','        self._records = records','        self._frame._epoch = 0',
             '        memset(self._frame._seen, 0, sizeof(self._frame._seen))',
             '        self._frame._calls = self._frame._misses = 0']
    if guarded:
        lines[1:1] = [*owner_init, '        mx_guard_init(&self._frame._guard)']
    mixed = bool(manifest.get('python_cells'))
    limit = 252 if heap else (4294967293 if mixed else 4294967294)
    step = 3 if mixed else 2
    idle = ['        self._require_idle()', '        self._frame._scope_valid = False'] if mixed else []
    if 'python_cells' in manifest:
        lines += ['        self._python_scope = None','        self._frame._python_scope = <PyObject*>self._python_scope']
    lines += ['    cdef void _advance(self) except *:',*resets,
              f'        if self._frame._epoch >= {limit}:',
              '            memset(self._frame._seen, 0, sizeof(self._frame._seen))',
              '            self._frame._epoch = 0',f'        self._frame._epoch += {step}',
              '    def reset(self):',*idle,'        self._advance()',
              '    def stats(self):',f"        return {{'enabled': {manifest['diagnostics']}, 'calls': self._frame._calls, 'misses': self._frame._misses, 'cache_slots': {manifest['cache_slots']}, 'epoch': self._frame._epoch}}",
              '    def replace_records(self, records):',*idle,'        self._records = records','        self._advance()',
              '    cdef void _select(self, tuple record) except *:']
    for i,name in enumerate(manifest['inputs']):
        if name in objects:
            lines += [f'        self.{name} = record[{i}]',f'        self._frame.{name} = <PyObject*>self.{name}']
        else: lines.append(f'        self._frame.{name} = record[{i}]')
    lines += ['        self._advance()']
    runtime = '\n'.join(lines)
    if resets: runtime += '\n'+'\n'.join(methods)+'\n'
    return '\n'.join(declarations)+'\n'+runtime.rstrip('\n')+'\n'

