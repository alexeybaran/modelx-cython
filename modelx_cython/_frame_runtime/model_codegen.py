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
    return source.rstrip() + '\n\n' + '\n'.join(additions)
