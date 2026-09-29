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
                                                  if entry['kind'] == 'map'})} if any(
        entry['kind'] == 'map' for entry in entries) else None


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
    manifest['compiled_collect_map'] = plan
    return 'from libcpp.vector cimport vector\n' + source + '\n' + '\n'.join(lines) + '\n'
