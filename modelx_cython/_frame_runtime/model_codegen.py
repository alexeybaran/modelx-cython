"""Public scoped entries for model façades; native formula/cache bodies are unchanged."""
import re


def _method(source, name):
    match = re.search(r'^    def ' + re.escape(name) + r'\(', source, re.M)
    if match is None: raise ValueError('Missing generated entry: ' + name)
    following = re.search(r'\n    (?:def |cdef |cpdef )', source[match.end():])
    end = match.end() + following.start() if following else len(source)
    return source[match.start():end].rstrip() + '\n'


def add_model_entries(source, manifest):
    mixed = bool(manifest.get('python_cells'))
    select = '        self._select_policy(policy)\n' if mixed else '        self._select(self._records[policy])\n'
    check = '        if policy < 0 or policy >= len(self._records): raise IndexError(policy)\n'
    additions = []
    if not mixed:
        additions.append('    def select_scope(self, Py_ssize_t policy):\n' + check + select)
        call = _method(source, 'evaluate').replace('def evaluate(self, Py_ssize_t policy, str cell, tuple args):',
                                                   'def call(self, str cell, tuple args):')
        additions.append(call.replace(check, '').replace(select, ''))
    table = _method(source, 'run_one').replace('def run_one(self, Py_ssize_t policy, str table):',
                                              'def call_table(self, str table):')
    additions.append(table.replace(check, '').replace(select, ''))
    scalar = _method(source, 'run_scalar_one').replace('def run_scalar_one(self, Py_ssize_t policy, str name):',
                                                        'def call_scalar(self, str name):')
    additions.append(scalar.replace(check, '').replace(select, ''))
    object_roots = [name for name in manifest.get('cell_outputs', ())
                    if name in manifest.get('object_cells', ())
                    and not manifest['cells'][name]['params']
                    and manifest['cells'][name]['result']['kind'] == 'object']
    if object_roots:
        # The caller supplies its public result-detachment rule. It runs before
        # the next policy reuses the working frame, while the formula itself is
        # still the one generated Cython Cell body and authoritative cache.
        lines = ['    def run_object_outputs(self, str name, object detach, object consume=None):',
                 '        cdef Py_ssize_t i',
                 '        if not callable(detach): raise TypeError("detach must be callable")',
                 '        if consume is not None and not callable(consume):',
                 '            raise TypeError("consume must be callable")',
                 '        result = [] if consume is None else None',
                 '        for i in range(len(self._records)):']
        lines.append('            self._select_policy(i)' if mixed
                     else '            self._select(self._records[i])')
        for index, name in enumerate(object_roots):
            lines.extend((f"            {'if' if index == 0 else 'elif'} name == {name!r}:",
                          f'                value = self._object_cell_{name}()'))
        lines.extend(('            else: raise KeyError(name)',
                      '            detached = detach(value)',
                      '            if consume is None: result.append(detached)',
                      '            else: consume(i, detached)',
                      '        return result'))
        additions.append('\n'.join(lines) + '\n')
    return source.rstrip() + '\n\n' + '\n'.join(additions)
