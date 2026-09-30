"""Owned-object memo entries behind the shared compiled Cell interface.

Only object-contract Cells reach this emitter. Numeric C frames are unchanged.
Python/Cython checked frames own objects and unwind normally; callbacks into the
numeric region use M1's checked trampoline. No raw object crosses its C++ guard.
"""


_ENTRY = '''cdef class _ObjectEntry:
    cdef object value
    cdef unsigned char state
    cdef bint touched

'''


def augment(source, manifest):
    rows = {k: manifest['cells'][k] for k in manifest['object_cells']}
    fields = ['    cdef list _object_dirty', '    cdef bint _object_key_guard, _clearing_objects',
              '    cdef bint _object_initialized']
    init = ['        self._object_dirty = []']
    clear = []
    methods = []
    bind = []
    dispatch = []
    stats = []
    for name, row in rows.items():
        storage = row['storage']
        attr = '_object_' + name
        fields += [f'    cdef object _object_fn_{name}',
                   f'    cdef {"list" if storage == "object_slots" else "dict"} {attr}',
                   f'    cdef unsigned long long _ocalls_{name}, _omisses_{name}, _ohits_{name}']
        init += [f'        self.{attr} = ' + (f'[None] * {row["slots"]}' if storage == 'object_slots' else '{}')]
        if storage == 'python_dict':
            clear.append(f'            self.{attr}.clear()')
        if manifest.get('cython_objects') and not row['params']:
            fields.append(f'    cdef object _object_singleton_{name}')
            init.append(f'        self._object_singleton_{name} = _DICT_EMPTY')
            clear.append(f'            self._object_singleton_{name} = _DICT_EMPTY')
        bind.append(f'        self._object_fn_{name} = scope.calculators[{name!r}]')
        params = [dict(p, name='_arg_' + str(i)) for i, p in enumerate(row['params'])]
        signature = ', '.join(p['type']['ctype'] + ' ' + p['name'] for p in params)
        signature = ', ' + signature if signature else ''
        values = ', '.join(p['name'] for p in params)
        arguments = '(' + values + (',' if params else '') + ')'
        body = [f'    cdef object _object_cell_{name}(self{signature}):',
                '        cdef _ObjectEntry entry', '        cdef Py_ssize_t index',
                '        cdef object key']
        if manifest['diagnostics']:
            body += [f'        self._ocalls_{name} += 1', '        self._frame._calls += 1']
        if storage == 'object_slots':
            expr = '0'
            for p in params:
                arg, low, size = p['name'], p['lower'], p['size']
                body += [f'        if {arg} < {low} or {arg} >= {low + size}:',
                         f'            raise IndexError("Outside declared cache bound: {name}.{arg}")']
                expr = f'(({expr}) * {size} + ({arg} - ({low})))'
            body += [f'        index = {expr}', f'        entry = self.{attr}[index]',
                     '        if entry is None:', '            entry = _ObjectEntry()',
                     f'            self.{attr}[index] = entry']
        else:
            key = params[0]['name'] if len(params) == 1 else arguments
            body += [f'        key = {key}', '        self._object_key_guard = True',
                     '        try:', f'            entry = self.{attr}.get(key)',
                     '            if entry is None:', '                entry = _ObjectEntry()',
                     f'                self.{attr}[key] = entry',
                     '        finally:', '            self._object_key_guard = False']
        body += ['        if entry.state == 2:']
        if manifest['diagnostics']:
            body += [f'            self._ohits_{name} += 1']
        body += ['            return entry.value',
                 '        if entry.state == 1:', f'            raise RuntimeError("Cell cycle: {name}")',
                 '        if not entry.touched:', '            self._object_dirty.append(entry)',
                 '            entry.touched = True', '        entry.state = 1']
        if manifest['diagnostics']:
            body += [f'        self._omisses_{name} += 1', '        self._frame._misses += 1']
        # Call the bundled Python body directly from this compiled miss path.
        # Native/object identity conversions require no intermediate Python frame.
        decoded = []
        for param in params:
            value = param['name']
            if param['type']['kind'] in ('enum', 'table_key'):
                value = f"self._python_scope.decode({param['type']!r}, {value})"
            decoded.append(value)
        body += ['        try:', '            if self._python_scope.diagnostics:',
                 f'                self._python_scope.calls[{name!r}] += 1',
                 f'            entry.value = self._object_fn_{name}(' + ', '.join(decoded) + ')']
        if row.get('allow_none') is False:
            body += ['            if entry.value is None:',
                     f'                raise ValueError("None is not allowed for Cell {name}")']
        result = row['result']
        if result['kind'] == 'object':
            expected = result.get('python_type', 'object')
            if expected != 'object':
                typ = 'type(None)' if expected == 'NoneType' else expected
                body += [f'            if not isinstance(entry.value, {typ}):',
                         f'                raise TypeError("Python object boundary requires {expected}")']
        else:
            body += [f"            entry.value = self._python_scope.encode({result!r}, entry.value)"]
        body += ['            entry.state = 2', '            return entry.value',
                 '        except BaseException:',
                 '            entry.state = 0', '            entry.value = None', '            raise']
        if manifest.get('cython_objects') and (storage == 'python_dict' or not params):
            body = _direct_dict_body(name, row, params, manifest['diagnostics'])
            if not params:row['storage'] = 'object_singleton'
        methods += body + ['']
        call = ', '.join(f'args[{i}]' for i in range(len(params)))
        dispatch += [f'        if cell == {name!r}:',
                     f'            if len(args) != {len(params)}: raise TypeError("Use canonical positional arguments")',
                     '            self._frame._entry_depth += 1', '            try:',
                     f'                return self._object_cell_{name}({call})',
                     '            finally:', '                self._frame._entry_depth -= 1']
        stats.append(f'{name!r}: {{"storage": {storage!r}, "calls": self._ocalls_{name}, '
                     f'"misses": self._omisses_{name}, "hits": self._ohits_{name}}}')

    marker = 'cdef class Kernel:\n'
    source = source.replace(marker, marker + '\n'.join(fields) + '\n', 1)
    lazy_object_cache = bool(manifest.get('native_scalar_outputs'))
    if lazy_object_cache:
        source = source.replace('    def __init__(self, records):\n',
                                '    def __init__(self, records):\n        self._object_dirty = []\n', 1)
        setup = ('        if not self._object_initialized:\n' +
                 '\n'.join('    ' + line for line in init[1:]) +
                 '\n            self._object_initialized = True')
        source = source.replace('        self._frame._python_scope = <PyObject*>scope',
                                '        self._frame._python_scope = <PyObject*>scope\n' + setup +
                                '\n' + '\n'.join(bind), 1)
    else:
        source = source.replace('    def __init__(self, records):\n',
                                '    def __init__(self, records):\n' + '\n'.join(init) +
                                '\n        self._object_initialized = True\n', 1)
        source = source.replace('        self._frame._python_scope = <PyObject*>scope',
                                '        self._frame._python_scope = <PyObject*>scope\n' + '\n'.join(bind), 1)
    source = source.replace('    cdef void _advance(self) except *:\n',
                            '    cdef void _advance(self) except *:\n        self._clear_object_cache()\n', 1)
    source = source.replace('    cdef void _require_idle(self) except *:\n',
                            '    cdef void _require_idle(self) except *:\n'
                            '        if self._clearing_objects or self._object_key_guard:\n'
                            '            raise RuntimeError("Cannot re-enter during object cache maintenance")\n', 1)
    start = source.index('    def call(self, str cell, tuple args):')
    end = source.index('    def run_scalar(', start)
    call = source[start:end]
    anchor = '        command.mode = 0'
    assert anchor in call
    call = call.replace(anchor,
        '        if self._object_key_guard or self._clearing_objects:\n'
        '            raise RuntimeError("Cell calls from hash/equality or cache finalization are unsupported")\n'
        + '\n'.join(dispatch) + '\n' + anchor, 1)
    source = source[:start] + call + source[end:]
    clear_body = '''    cdef void _clear_object_cache(self) except *:
        cdef _ObjectEntry entry
        if not self._object_initialized:
            return
        self._clearing_objects = True
        try:
            for entry in self._object_dirty:
                entry.state = 0
                entry.touched = False
                entry.value = None
            self._object_dirty.clear()
''' + '\n'.join(clear) + '''
        finally:
            self._clearing_objects = False

'''
    methods += ['    def object_stats(self):', '        return {' + ', '.join(stats) + '}']
    source += '\n' + clear_body + '\n'.join(methods) + '\n'
    source = source.replace('@cython.final\ncdef class Kernel:', _ENTRY + '@cython.final\ncdef class Kernel:', 1)
    if manifest.get('cython_objects'):
        source = 'cdef object _DICT_EMPTY = object()\ncdef object _DICT_BUSY = object()\n' + source
    manifest['object_memo_protocol'] = 'compiled owned slots/dicts; checked object frames; empty/busy/ready entries'
    return source


def _direct_dict_body(name, row, params, diagnostics):
    """Same empty/busy/ready protocol without allocating a box for every dict value."""
    signature=', '.join(p['type']['ctype']+' '+p['name'] for p in params)
    values=', '.join(p['name'] for p in params)
    key=params[0]['name'] if len(params)==1 else '('+values+(',' if params else '')+')'
    lines=[f'    cdef object _object_cell_{name}(self'+(', '+signature if signature else '')+'):',
           '        cdef object key, value',f'        key = {key}']
    if diagnostics:lines += [f'        self._ocalls_{name} += 1','        self._frame._calls += 1']
    lines += ['        self._object_key_guard = True','        try:',
              f'            value = self._object_{name}.get(key, _DICT_EMPTY)',
              '        finally:', '            self._object_key_guard = False',
              '        if value is _DICT_BUSY:',f'            raise RuntimeError("Cell cycle: {name}")',
              '        if value is not _DICT_EMPTY:']
    if diagnostics:lines += [f'            self._ohits_{name} += 1']
    lines += ['            return value','        self._object_key_guard = True','        try:',
              f'            self._object_{name}[key] = _DICT_BUSY', '        finally:',
              '            self._object_key_guard = False']
    if diagnostics:lines += [f'        self._omisses_{name} += 1','        self._frame._misses += 1']
    lines += ['        try:', '            if self._python_scope.diagnostics:',f'                self._python_scope.calls[{name!r}] += 1',
              f'            value = self._object_fn_{name}({values})']
    if row.get('allow_none') is False:lines += ['            if value is None:',f'                raise ValueError("None is not allowed for Cell {name}")']
    result=row['result'];expected=result.get('python_type','object')
    if result['kind']=='object' and expected!='object':
        typ='type(None)' if expected=='NoneType' else expected
        lines += [f'            if not isinstance(value, {typ}):',f'                raise TypeError("Python object boundary requires {expected}")']
    elif result['kind']!='object':lines += [f'            value = self._python_scope.encode({result!r}, value)']
    lines += ['            self._object_key_guard = True','            try:',f'                self._object_{name}[key] = value',
              '            finally:', '                self._object_key_guard = False', '            return value',
              '        except BaseException:', '            self._object_key_guard = True','            try:',
              f'                self._object_{name}.pop(key, None)', '            finally:',
              '                self._object_key_guard = False', '            raise']
    if not params:
        # The zero-argument key has a single owned slot, not a Python dict lookup.
        # Use exactly the same empty/busy/ready and rollback protocol. Reset drops
        # the owned value under the existing finalization/reentry guard.
        text='\n'.join(lines)
        text=text.replace('        cdef object key, value\n        key = ()', '        cdef object value')
        text=text.replace('        self._object_key_guard = True\n        try:\n'
            +f'            value = self._object_{name}.get(key, _DICT_EMPTY)\n'
            +'        finally:\n            self._object_key_guard = False',
            f'        value = self._object_singleton_{name}')
        text=text.replace('        self._object_key_guard = True\n        try:\n'
            +f'            self._object_{name}[key] = _DICT_BUSY\n'
            +'        finally:\n            self._object_key_guard = False',
            f'        self._object_singleton_{name} = _DICT_BUSY')
        text=text.replace('            self._object_key_guard = True\n            try:\n'
            +f'                self._object_{name}[key] = value\n'
            +'            finally:\n                self._object_key_guard = False',
            f'            self._object_singleton_{name} = value')
        text=text.replace('            self._object_key_guard = True\n            try:\n'
            +f'                self._object_{name}.pop(key, None)\n'
            +'            finally:\n                self._object_key_guard = False',
            f'            self._object_singleton_{name} = _DICT_EMPTY')
        lines=text.splitlines()
    return lines
