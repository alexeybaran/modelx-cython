"""Stream declared numeric collect maps without per-policy Python dictionaries.

The Cell formulas remain Cython's authoritative implementations. This module
recognizes a literal collect output declaration and emits a Cython traversal
that calls those same cached Cells in the declared order. All other output
formulas retain the existing checked object traversal.
"""
import ast

from .compiler import function_ast


def _literal(value):
    return value.value if isinstance(value, ast.Constant) else None


def _plan(spec, manifest):
    collect = (manifest.get('model') or {}).get('collect') or {}
    if collect.get('object_outputs') != 'collect_outputs':
        return None
    row = manifest['cells'].get('collect_outputs')
    if not row or row['params'] or row['result']['kind'] != 'object':
        return None
    node = function_ast(spec.cells['collect_outputs'])
    body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)) else node.body
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    result = body[0].value
    if not isinstance(result, ast.Dict) or len(result.keys) != 1 or _literal(result.keys[0]) != 'sum':
        return None
    values = result.values[0]
    if not isinstance(values, ast.Dict) or not values.keys:
        return None
    entries = []
    names = set()
    for key, value in zip(values.keys, values.values):
        output = _literal(key)
        if type(output) is not str or output in names:
            return None
        names.add(output)
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and not value.args and not value.keywords):
            cell = value.func.id
            cell_row = manifest['cells'].get(cell)
            if not cell_row or cell_row['params'] or cell_row['result']['kind'] != 'float' or cell_row['calculator'] != 'native':
                return None
            entries.append({'kind': 'scalar', 'name': output, 'cell': cell})
            continue
        if not isinstance(value, ast.DictComp) or len(value.generators) != 1:
            return None
        generator = value.generators[0]
        if (generator.is_async or generator.ifs or not isinstance(generator.target, ast.Name)
                or not isinstance(generator.iter, ast.Name)
                or not isinstance(value.key, ast.Name)
                or value.key.id != generator.target.id
                or not isinstance(value.value, ast.Call)
                or not isinstance(value.value.func, ast.Name)
                or len(value.value.args) != 1 or value.value.keywords
                or not isinstance(value.value.args[0], ast.Name)
                or value.value.args[0].id != generator.target.id):
            return None
        cell = value.value.func.id
        source = generator.iter.id
        cell_row = manifest['cells'].get(cell)
        if (not cell_row or cell_row['calculator'] != 'native'
                or cell_row['result']['kind'] != 'float'
                or len(cell_row['params']) != 1
                or cell_row['params'][0]['type']['kind'] != 'int'
                or source not in manifest.get('python_references', ())
                   and source not in manifest['inputs']):
            return None
        entries.append({'kind': 'map', 'name': output, 'cell': cell,
                        'source': source})
    return {'entries': entries, 'sources': sorted({entry['source'] for entry in entries
                                                  if entry['kind'] == 'map'})}


def _streaming_accumulator(plan):
    """The same typed fold for an already selected ordinary policy owner.

    It does not select inputs, create Spaces or evaluate declarations. Only
    literal numeric sum declarations use it; public declaration calls retain
    their authoritative compiled formula and object cache.
    """
    lines = ['cdef class _CollectMapAccumulator:',
             '    cdef Py_ssize_t _count']
    for i, entry in enumerate(plan['entries']):
        if entry['kind'] == 'scalar':
            lines.append(f'    cdef double _total_{i}')
        else:
            lines += [f'    cdef tuple _keys_{i}',
                      f'    cdef vector[long long] _native_keys_{i}',
                      f'    cdef vector[double] _totals_{i}']
    lines += ['\n    def __init__(self, dict keys_by_source):',
              '        cdef object key']
    if not plan['sources']:
        lines.append('        pass')
    for i, entry in enumerate(plan['entries']):
        if entry['kind'] == 'map':
            lines += [f'        self._keys_{i} = keys_by_source[{entry["source"]!r}]',
                      f'        self._native_keys_{i}.reserve(len(self._keys_{i}))',
                      f'        self._totals_{i}.resize(len(self._keys_{i}))',
                      f'        for key in self._keys_{i}:',
                      f'            self._native_keys_{i}.push_back(<long long>key)']
    lines += ['\n    def matches(self, dict keys_by_source):']
    for i, entry in enumerate(plan['entries']):
        if entry['kind'] == 'map':
            lines += [f'        if keys_by_source[{entry["source"]!r}] != self._keys_{i}:',
                      '            return False']
    lines += ['        return True',
              '\n    cdef void _append(self, Kernel owner) except *:',
              '        cdef Py_ssize_t j', '        cdef double value',
              '        owner._require_idle()']
    for i, entry in enumerate(plan['entries']):
        if entry['kind'] == 'scalar':
            lines += [f'        value = owner._native_cell_{entry["cell"]}()',
                      '        if self._count == 0:',
                      f'            self._total_{i} = value',
                      '        else:', f'            self._total_{i} += value']
        else:
            lines += [f'        for j in range(self._native_keys_{i}.size()):',
                      f'            value = owner._native_cell_{entry["cell"]}(self._native_keys_{i}[j])',
                      '            if self._count == 0:',
                      f'                self._totals_{i}[j] = value',
                      '            else:', f'                self._totals_{i}[j] += value']
    lines += ['        self._count += 1', '\n    def finish(self):',
              '        cdef Py_ssize_t j',
              '        if not self._count:',
              '            raise ValueError("Collection needs a nonempty policy domain")']
    for i, entry in enumerate(plan['entries']):
        if entry['kind'] == 'map':
            lines += [f'        result_{i} = {{}}',
                      f'        for j in range(self._native_keys_{i}.size()):',
                      f'            result_{i}[self._keys_{i}[j]] = self._totals_{i}[j]']
    members = ', '.join(repr(e['name']) + ': ' +
        (f'self._total_{i}' if e['kind'] == 'scalar' else f'result_{i}')
        for i, e in enumerate(plan['entries']))
    lines += ["        return {'sum': {" + members + '}}', '']
    return '\n'.join(lines)


def augment(source, manifest, spec):
    """Add the numeric-map traversal only for a typed, source-preserved shape."""
    plan = _plan(spec, manifest)
    if plan is None:
        return source
    lines = [
        '    def run_compiled_collect_map(self, dict keys_by_source):',
        '        cdef Py_ssize_t i, j',
        '        cdef long long key',
        '        cdef double value',
        '        cdef object key_object',
    ]
    for index, entry in enumerate(plan['entries']):
        if entry['kind'] == 'scalar':
            lines.append(f'        cdef double total_{index}')
        else:
            lines += [f'        cdef tuple keys_{index} = keys_by_source[{entry["source"]!r}]',
                      f'        cdef vector[long long] native_keys_{index}',
                      f'        cdef vector[double] totals_{index}',
                      f'        cdef dict result_map_{index}']
    for index, entry in enumerate(plan['entries']):
        if entry['kind'] == 'map':
            lines += [f'        native_keys_{index}.reserve(len(keys_{index}))',
                      f'        totals_{index}.resize(len(keys_{index}))',
                      f'        for key_object in keys_{index}:',
                      f'            native_keys_{index}.push_back(<long long>key_object)']
    lines += ['        for i in range(len(self._records)):',
              '            self._select_policy(i)']
    for index, entry in enumerate(plan['entries']):
        if entry['kind'] == 'scalar':
            lines += [f'            value = self._native_cell_{entry["cell"]}()',
                      '            if i == 0:',
                      f'                total_{index} = value',
                      '            else:',
                      f'                total_{index} += value']
        else:
            lines += [f'            for j in range(native_keys_{index}.size()):',
                      f'                value = self._native_cell_{entry["cell"]}(native_keys_{index}[j])',
                      '                if i == 0:',
                      f'                    totals_{index}[j] = value',
                      '                else:',
                      f'                    totals_{index}[j] += value']
    for index, entry in enumerate(plan['entries']):
        if entry['kind'] == 'map':
            lines += [f'        result_map_{index} = {{}}',
                      f'        for j in range(native_keys_{index}.size()):',
                      f'            result_map_{index}[keys_{index}[j]] = totals_{index}[j]']
    members = ', '.join(repr(entry['name']) + ': ' +
                        (f'total_{index}' if entry['kind'] == 'scalar' else f'result_map_{index}')
                        for index, entry in enumerate(plan['entries']))
    lines += ["        return {'sum': {" + members + '}}', '']
    lines += [
        '    def new_collect_accumulator(self, dict keys_by_source):',
        '        return _CollectMapAccumulator(keys_by_source)', '',
        '    def append_collect_accumulator(self, _CollectMapAccumulator accumulator):',
        '        accumulator._append(self)', '']
    manifest['compiled_collect_map'] = plan
    # The streaming entry is narrower than bulk prepared traversal: object or
    # callback edges keep the normal checked declaration and exported fold.
    pending = [entry['cell'] for entry in plan['entries']]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        row = manifest['cells'][name]
        if row['calculator'] != 'native':
            manifest['streaming_collect_map'] = False
            break
        pending.extend(row.get('dependencies', ()))
    else:
        manifest['streaming_collect_map'] = True
    # Later owner/public emitters extend Kernel by appending indented methods.
    # Keep that class last; the accumulator is a private module-level type.
    source = source.replace('cdef class Kernel:',
        _streaming_accumulator(plan) + '\ncdef class Kernel:', 1)
    return 'from libcpp.vector cimport vector\n' + source + '\n' + '\n'.join(lines) + '\n'
