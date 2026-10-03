"""Shared exact-key primitives under frame ownership and checked Cython boundaries.

Formula bodies are not transformed here. Dictionary helpers finish ordinary
Cython cleanup before the numeric guard captures an error.
"""


def selected(rows):
    return {n: r for n, r in rows.items() if r.get('storage') == 'growing_numeric'}


def tag(name):
    return '_nc_' + name


def support_header(rows):
    from modelx_cython.numeric_cache import support
    rows = selected(rows)
    if not rows:
        return []
    lines = ['from libc.stdint cimport uint64_t',
             'from libc.stdlib cimport calloc, free',
             'cdef union _mx_bits:', '    double d', '    uint64_t u',
             '_NC_MISSING = object()']
    for name, row in rows.items():
        lines.append(support(tag(name), [p['type']['ctype'] for p in row['params']],
                             row['result']['ctype'], native=True))
    return lines


def cell_entry(name, row, signature, args, formula, diagnostics):
    ret = row['result']['ctype']
    names = ', '.join(p['name'] for p in row['params'])
    nonfinite = ' or '.join('not isfinite(' + p['name'] + ')'
                           for p in row['params'] if p['type']['kind'] == 'float') or 'False'
    types = ', '.join(['_Frame*'] + [p['type']['ctype'] for p in row['params']])
    lines = [f'ctypedef int (*_NcLookup_{name})({types}, {ret}*) noexcept nogil',
             f'ctypedef int (*_NcCommit_{name})({types}, {ret}) noexcept nogil']
    lines += [f'cdef inline {ret} _cell_{name}({signature}) except? -1 nogil:',
             f'    cdef {ret} _mx_value', '    cdef int _mx_status',
             '    cdef Py_ssize_t _mx_found']
    if diagnostics:
        lines.append('    state._calls += 1')
    # A native cache hit has no Python-owned locals or failure path. Keep it
    # beside the Cell entry, rather than crossing the checked dictionary bridge.
    # Formula execution still owns no item pointer: children may grow storage or
    # atomically move completed entries into the authoritative dictionary.
    lines += [f'    if state._numeric_fallback_{name} == NULL and not ({nonfinite}):',
              f'        _mx_found = {tag(name)}_find(&state._numeric_{name}, {names})',
              f'        if _mx_found >= 0: return state._numeric_{name}.items[_mx_found].value',
              '    else:',
              f'        _mx_status = (<_NcLookup_{name}>_nc_lookup_{name})({args}, &_mx_value)',
              '        if _mx_status < 0: mx_capture(&state._guard)',
              '        if _mx_status: return _mx_value']
    if diagnostics:
        lines.append('    state._misses += 1')
    lines += [f'    _mx_value = {formula}',
              f'    _mx_status = (<_NcCommit_{name}>_nc_commit_{name})({args}, _mx_value)',
              '    if _mx_status < 0: mx_capture(&state._guard)',
              '    return _mx_value']
    return lines


def frame_parts(rows):
    rows = selected(rows)
    fields = ['    PyObject* _numeric_owner'] if rows else []
    helpers = []
    for name, row in rows.items():
        t = tag(name); ret = row['result']['ctype']
        params = row['params']
        # These helpers own their local names, not the formula's public ABI.
        # Positional private arguments avoid collisions with key/value/found or
        # any other author parameter without modifying the formula itself.
        arguments = ['_mx_key_' + str(i) for i in range(len(params))]
        declaration = ', '.join(p['type']['ctype'] + ' ' + argument
                                for p, argument in zip(params, arguments))
        names = ', '.join(arguments)
        key = arguments[0] if len(params) == 1 else '(' + names + ')'
        nonfinite = ' or '.join('not isfinite(' + argument + ')'
                               for p, argument in zip(params, arguments)
                               if p['type']['kind'] == 'float') or 'False'
        fields += [f'    {t}_cache _numeric_{name}', f'    PyObject* _numeric_fallback_{name}']
        helpers.append(f'''cdef int _nc_lookup_{name}(_Frame* state, {declaration}, {ret}* value) except -1 nogil:
    cdef Py_ssize_t found
    if state._numeric_fallback_{name} == NULL and not ({nonfinite}):
        found = {t}_find(&state._numeric_{name}, {names})
        if found < 0: return 0
        value[0] = state._numeric_{name}.items[found].value
        return 1
    with gil:
        return _nc_dict_lookup_{name}(state, {names}, value)

cdef int _nc_dict_lookup_{name}(_Frame* state, {declaration}, {ret}* value) except -1:
    cdef dict mapping
    cdef object cached, key
    key = {key}
    if state._numeric_fallback_{name} == NULL:
        mapping = {t}_dictionary(&state._numeric_{name})
        # Publish after every completed native entry has transferred.
        (<Kernel>state._numeric_owner)._numeric_dict_{name} = mapping
        state._numeric_fallback_{name} = <PyObject*>mapping
        free(state._numeric_{name}.items)
        state._numeric_{name}.items = NULL
        state._numeric_{name}.capacity = state._numeric_{name}.count = 0
    else:
        mapping = <dict>state._numeric_fallback_{name}
    cached = mapping.get(key, _NC_MISSING)
    if cached is _NC_MISSING: return 0
    value[0] = cached
    return 1

cdef int _nc_commit_{name}(_Frame* state, {declaration}, {ret} value) except -1 nogil:
    # A child may grow storage or change the authoritative representation.
    # No item pointer survives formula execution.
    if state._numeric_fallback_{name} == NULL:
        return {t}_put(&state._numeric_{name}, {names}, value)
    with gil:
        return _nc_dict_commit_{name}(state, {names}, value)

cdef int _nc_dict_commit_{name}(_Frame* state, {declaration}, {ret} value) except -1:
    (<dict>state._numeric_fallback_{name})[{key}] = value
    return 0
''')
    return fields, helpers


def owner_parts(rows):
    rows = selected(rows)
    fields, initialize, reset, deallocate, methods = [], [], [], [], []
    if rows:
        initialize = ['        self._frame._numeric_owner = <PyObject*>self']
    for name, row in rows.items():
        t = tag(name)
        fields += [f'    cdef object _numeric_dict_{name}']
        reset += [f'        if self._frame._numeric_{name}.items != NULL:',
                  f'            memset(self._frame._numeric_{name}.items, 0, self._frame._numeric_{name}.capacity * sizeof({t}_entry))',
                  f'        self._frame._numeric_{name}.count = 0',
                  f'        self._frame._numeric_fallback_{name} = NULL',
                  f'        self._numeric_dict_{name} = None']
        deallocate += [f'free(self._frame._numeric_{name}.items)']
        params = row['params']
        key = 'key' if len(params) == 1 else ', '.join('key['+str(i)+']' for i in range(len(params)))
        nonfinite = ' or '.join('not isfinite(' + ('key' if len(params) == 1 else 'key['+str(i)+']') + ')'
                               for i, p in enumerate(params) if p['type']['kind'] == 'float') or 'False'
        methods += [f'    def _numeric_snapshot_{name}(self):',
                    f'        if self._frame._numeric_fallback_{name} != NULL:',
                    f'            return dict(self._numeric_dict_{name})',
                    f'        return {t}_dictionary(&self._frame._numeric_{name})',
                    f'    def _numeric_restore_{name}(self, dict values):',
                    '        for key in values:', f'            if {nonfinite}:',
                    '                mapping = dict(values)',
                    f'                self._numeric_dict_{name} = mapping',
                    f'                self._frame._numeric_fallback_{name} = <PyObject*>mapping',
                    '                return',
                    '        for key, value in values.items():',
                    f'            {t}_put(&self._frame._numeric_{name}, {key}, value)']
    return fields, initialize, reset, deallocate, methods
