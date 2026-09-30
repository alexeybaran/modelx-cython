"""Guard a pure numeric Cython call tree using native C++ stack unwinding.

Formula bodies keep Cython's checked Python-exception semantics. Generated cache
wrappers use noexcept *Python* signatures. A capture check is emitted only for a
formula whose own admitted syntax can directly create a Python error state. Errors
from called Cells already unwind through their own boundary.

The native subset admits no Python locals/callbacks, try/finally or user GIL
regions. No Python-owned temporary is live across a native Cell call. This
module is physical runtime generation, not interpretation of formula semantics.
"""
import re

HEADER = r'''
cdef extern from *:
    """
    #include <Python.h>
    #include <exception>
    struct MxGuard {
        PyObject* exception;
        int kind;
        const char* where;
    };
    struct MxAbort { MxGuard* owner; };
    typedef void (*MxCallback)(void*, void*);
    static inline void mx_guard_init(MxGuard* g) {
        g->exception = NULL; g->kind = -1; g->where = "";
    }
    static inline void mx_guard_clear(MxGuard* g) {
        Py_XDECREF(g->exception); mx_guard_init(g);
    }
    static inline void mx_fail(MxGuard* g, int kind, const char* where) {
        g->kind = kind; g->where = where; throw MxAbort{g};
    }
    static inline void mx_capture(MxGuard* g) {
        PyGILState_STATE gil = PyGILState_Ensure();
        PyObject* error = PyErr_Occurred() ? PyErr_GetRaisedException() : NULL;
        PyGILState_Release(gil);
        if (error) { g->exception = error; throw MxAbort{g}; }
    }
    static int mx_guarded(MxGuard* g, MxCallback fn, void* state, void* command) {
        // Called with the GIL. Numeric frames may not acquire owned Python state.
        mx_guard_clear(g);
        try { fn(state, command); return 0; }
        catch (const MxAbort& e) {
            if (e.owner != g) { g->kind = 5; g->where = "foreign native guard"; }
            return -1;
        }
        catch (const std::exception&) { g->kind = 5; g->where = "native C++ error"; return -1; }
        catch (...) { g->kind = 5; g->where = "unknown native error"; return -1; }
    }
    """
    ctypedef struct MxGuard:
        PyObject* exception
        int kind
        const char* where
    ctypedef void (*MxCallback)(void*, void*) noexcept nogil
    void mx_guard_init(MxGuard*) noexcept nogil
    void mx_guard_clear(MxGuard*) noexcept
    void mx_fail(MxGuard*, int kind, const char* where) noexcept nogil
    void mx_capture(MxGuard*) noexcept nogil
    int mx_guarded(MxGuard*, MxCallback, void*, void*) noexcept

cdef void _raise_guard(MxGuard* guard) except *:
    cdef object error
    if guard.exception != NULL:
        error = <object>guard.exception
        mx_guard_clear(guard)
        raise error
    kind = guard.kind
    location = guard.where.decode()
    mx_guard_clear(guard)
    if kind == 0: raise IndexError("Outside declared cache bound: " + location)
    if kind == 1: raise RuntimeError("Cell cycle: " + location)
    if kind == 2: raise ValueError("Non-finite hash key: " + location)
    if kind == 3: raise MemoryError("Hash memo capacity exhausted: " + location)
    raise RuntimeError("Native evaluation aborted: " + location)
'''


def guarded_functions(functions, manifest):
    """Lower Cell wrappers to the native guard with sparse checked boundaries.

    Every calc call is made through its declared noexcept ABI so intermediate Cython
    frames do not add exception branches. Only Cells whose *own* syntax may create a
    checked error inspect the sentinel and capture PyErr. Deeper Cell errors use
    mx_fail/mx_capture at the deeper boundary and unwind directly.
    """
    aliases=[]
    for name,row in manifest['cells'].items():
        types=', '.join(['_Frame*']+[p['type']['ctype'] for p in row['params']])
        aliases.append(f"ctypedef {row['result']['ctype']} (*_Calc_{name})({types}) noexcept nogil")
    result=[]
    for block in functions:
        match=re.search(r'cdef inline \w+(?: \w+)? _cell_(\w+)', block)
        if match:
            name=match.group(1)
            block=block.replace('except? -1 nogil:', 'noexcept nogil:')
            block=re.sub(r'(?<!\w)_calc_(\w+)\(state',r'(<_Calc_\1>_calc_\1)(state',block)
            block=block.replace('_cache_error(', 'mx_fail(&state._guard, ')
            if manifest['cells'][name].get('checked_formula'):
                lines=[]
                for line in block.splitlines():
                    lines.append(line)
                    if '_mx_value = (<_Calc_' in line:
                        indent=line[:len(line)-len(line.lstrip())]
                        lines.append(indent+'if _mx_value == -1: mx_capture(&state._guard)')
                block='\n'.join(lines)
        result.append(block)
    return aliases,result


def runtime(manifest, prefix, length_expression, tables, *, owner_init=(), owner_dealloc=None):
    rows=manifest['cells'];indices={k:i for i,k in enumerate(rows)}
    max_args=max([len(r['params']) for r in rows.values()]+[1])
    lines=['cdef union _Scalar:', '    double f', '    long long i',
           'cdef struct _Command:', '    int mode, identity', '    Py_ssize_t n',
           f'    _Scalar args[{max_args}]', '    _Scalar value', '    double* output',
           '@cython.overflowcheck(False)',
           'cdef void _execute(void* raw_state, void* raw_command) noexcept nogil:',
           '    cdef _Frame* state = <_Frame*>raw_state',
           '    cdef _Command* command = <_Command*>raw_command',
           '    cdef Py_ssize_t t', '    if command.mode == 0:']
    for i,(name,row) in enumerate(rows.items()):
        dest='f' if row['result']['kind']=='float' else 'i'
        args=', '.join('command.args[%d].%s'%(j,'f' if p['type']['kind']=='float' else 'i') for j,p in enumerate(row['params']))
        lines += [f'        {"if" if i==0 else "elif"} command.identity == {i}:',f'            command.value.{dest} = _cell_{name}(state'+(', '+args if args else '')+')']
    lines += [('        else: ' if rows else '        ') + 'mx_fail(&state._guard, 4, b"Cell dispatch")',
              '    elif command.mode == 1:',f'        command.value.i = {length_expression}',
              '    elif command.mode == 2:']
    for i,(name,columns) in enumerate(tables.items()):
        lines += [f'        {"if" if i==0 else "elif"} command.identity == {i}:','            for t in range(command.n):']
        for j,expr in enumerate(columns): lines += [f'                command.output[t * {len(columns)} + {j}] = {expr}']
    lines += [('        else: ' if tables else '        ') + 'mx_fail(&state._guard, 4, b"Table dispatch")',
              '    else: mx_fail(&state._guard, 4, b"Command dispatch")','']
    # The existing owner/lifetime prefix contains no formula calls.
    init = ['    def __init__(self, records):', *owner_init,
            '        mx_guard_init(&self._frame._guard)']
    prefix = prefix.replace('    def __init__(self, records):', '\n'.join(init), 1)
    if owner_dealloc is None:
        owner_dealloc = ('        mx_guard_clear(&self._frame._guard)',)
    lines += ['@cython.final','cdef class Kernel:',prefix,
              '    def __dealloc__(self):', *owner_dealloc,
              '    cdef void _invoke(self, _Command* command) except *:',
              '        if mx_guarded(&self._frame._guard, _execute, &self._frame, command) != 0:',
              '            _raise_guard(&self._frame._guard)',
              '    def evaluate(self, Py_ssize_t policy, str cell, tuple args):',
              '        cdef _Command command',
              "        if policy < 0 or policy >= len(self._records): raise IndexError(policy)",
              '        self._select(self._records[policy])','        command.mode = 0']
    for i,(name,row) in enumerate(rows.items()):
        lines += [f'        {"if" if i==0 else "elif"} cell == {name!r}:',f'            command.identity = {i}',
                  f"            if len(args) != {len(row['params'])}: raise TypeError('Use canonical positional arguments')"]
        for j,p in enumerate(row['params']):
            d='f' if p['type']['kind']=='float' else 'i'; lines += [f'            command.args[{j}].{d} = args[{j}]']
        d='f' if row['result']['kind']=='float' else 'i'
        result = f'command.value.{d}'
        if row['result']['kind'] == 'bool': result = 'bool(command.value.i)'
        lines += ['            self._invoke(&command)',f'            return {result}']
    lines += ['        else: raise KeyError(cell)' if rows else '        raise KeyError(cell)',
              '    def run_scalar(self, str name):','        cdef _Command command','        cdef Py_ssize_t i',
              '        cdef double[::1] values', '        command.mode = 0']
    for i,name in enumerate(manifest['scalar_outputs']):
        lines += [f'        {"if" if i==0 else "elif"} name == {name!r}: command.identity = {indices[name]}']
    lines += ["        else: raise KeyError(name)" if manifest['scalar_outputs'] else "        raise KeyError(name)",
              '        result = np.empty(len(self._records), dtype=np.float64)', '        values = result',
              '        for i in range(len(self._records)):', '            self._select(self._records[i])',
              '            self._invoke(&command)']
    for i,name in enumerate(manifest['scalar_outputs']):
        d='f' if rows[name]['result']['kind']=='float' else 'i'
        lines += [f'            {"if" if i==0 else "elif"} command.identity == {indices[name]}: values[i] = command.value.{d}']
    lines += ["        if not np.isfinite(result).all(): raise ArithmeticError('Non-finite scalar output')",'        return result',
              '    def run_scalar_one(self, Py_ssize_t policy, str name):','        cdef _Command command',
              "        if policy < 0 or policy >= len(self._records): raise IndexError(policy)",
              '        command.mode = 0']
    for i,name in enumerate(manifest['scalar_outputs']):
        lines += [f'        {"if" if i==0 else "elif"} name == {name!r}: command.identity = {indices[name]}']
    lines += ["        else: raise KeyError(name)" if manifest['scalar_outputs'] else "        raise KeyError(name)",
              '        self._select(self._records[policy])','        self._invoke(&command)']
    for i,name in enumerate(manifest['scalar_outputs']):
        d='f' if rows[name]['result']['kind']=='float' else 'i'
        lines += [f'        {"if" if i==0 else "elif"} command.identity == {indices[name]}: return command.value.{d}']
    lines += ['    def run(self, str table, bint sums):','        cdef _Command command','        cdef Py_ssize_t i, n, columns',
              '        cdef double[:,::1] view','        cdef int table_id']
    for i,(name,cols) in enumerate(tables.items()):
        lines += [f'        {"if" if i==0 else "elif"} table == {name!r}:',f'            table_id = {i}',f'            columns = {len(cols)}']
    lines += ['        else: raise KeyError(table)' if tables else '        raise KeyError(table)','        result = []','        for i in range(len(self._records)):',
              '            self._select(self._records[i])','            command.mode = 1','            self._invoke(&command)',
              '            n = command.value.i',f"            if n < 0 or n > {manifest['steps']}: raise IndexError('Output length exceeds declared bound')",
              '            out = np.empty((n,columns),dtype=np.float64)', '            view = out',
              '            command.mode = 2','            command.identity = table_id','            command.n = n',
              '            if n:','                command.output = &view[0,0]','                self._invoke(&command)',
              "            if not np.isfinite(out).all(): raise ArithmeticError('Non-finite kernel output')",
              '            result.append(float(out.sum()) if sums else out)',
              '        return np.asarray(result,dtype=np.float64) if sums else result',
              '    def run_one(self, Py_ssize_t policy, str table):','        cdef _Command command','        cdef Py_ssize_t n, columns',
              '        cdef double[:,::1] view','        cdef int table_id',
              "        if policy < 0 or policy >= len(self._records): raise IndexError(policy)"]
    for i,(name,cols) in enumerate(tables.items()):
        lines += [f'        {"if" if i==0 else "elif"} table == {name!r}:',f'            table_id = {i}',f'            columns = {len(cols)}']
    lines += ['        else: raise KeyError(table)' if tables else '        raise KeyError(table)','        self._select(self._records[policy])',
              '        command.mode = 1','        self._invoke(&command)','        n = command.value.i',
              f"        if n < 0 or n > {manifest['steps']}: raise IndexError('Output length exceeds declared bound')",
              '        out = np.empty((n,columns),dtype=np.float64)','        view = out','        command.mode = 2',
              '        command.identity = table_id','        command.n = n','        if n:',
              '            command.output = &view[0,0]','            self._invoke(&command)',
              "        if not np.isfinite(out).all(): raise ArithmeticError('Non-finite kernel output')",
              '        return out','']
    return '\n'.join(lines)
