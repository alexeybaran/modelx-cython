"""Typed, epoch-reset memo entries shared by Cython and real Python calculators.

The first integral argument selects a dense slab within the declared step bound;
all other axes remain dynamic Python keys. No author chooses whether to cache a
Cell, no policy/scenario is preselected, and no trace topology is frozen.
"""
import re
from .contracts import ContractError

CTYPES={'int':'long long','float':'double','bool':'bint','object':'object','str':'str','curve':'Curve'}
HEADER='''
from libc.stdlib cimport calloc, free
from libc.string cimport memset
from libc.stdint cimport uint64_t
from cpython.long cimport PyLong_Check

cdef struct _MemoAxisHead:
    PyObject* slab
    PyObject* object0
    PyObject* object1
    long long integer0
    long long integer1

cdef union _MemoNumber:
    double real
    long long integer

cdef class _MemoSlab:
    cdef _MemoNumber* values
    cdef uint64_t* stamps
    cdef Py_ssize_t size
    def __cinit__(self, Py_ssize_t size):
        self.size = size
        self.values = <_MemoNumber*>calloc(size, sizeof(_MemoNumber))
        self.stamps = <uint64_t*>calloc(size, sizeof(uint64_t))
        if self.values == NULL or self.stamps == NULL: raise MemoryError()
    def __dealloc__(self):
        free(self.values)
        free(self.stamps)

cdef long long _checked_memo_int(object value) except? -1:
    if not PyLong_Check(value):
        # numpy integral scalars implement __index__; floats/strings must not truncate.
        import operator
        value = operator.index(value)
    return value

cdef double _checked_memo_float(object value) except? -1:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("Numeric Cell contract rejects text")
    return value

'''

def checked_signatures(spec, rows):
    out={}
    for name,hint in (spec.cython_signatures or {}).items():
        if name not in rows:raise ContractError('Unknown typed Cell: '+name)
        params=hint.get('parameters',{})
        if set(params)!={p['name'] for p in rows[name]['params']}:
            raise ContractError('Typed parameter signature mismatch: '+name)
        if any(t not in CTYPES for t in [hint.get('return'),*params.values()]):
            raise ContractError('Unsupported Cython scalar signature: '+name)
        if 'curve' in params.values():raise ContractError('Curve is an owned result, not a Cell key')
        out[name]=hint
    return out


def cached_entry(name, hint, row, steps, diagnostics, compiled, *, virtual=False):
    """One checked cache/rollback template for exact and virtual formula entries."""
    params = row['params']; original_names = [p['name'] for p in params]
    names = ['_mx_arg_' + str(i) for i in range(len(params))]
    kinds = [hint['parameters'][n] for n in original_names]; ct = CTYPES[hint['return']]
    sig = ', '.join(CTYPES[t] + ' ' + n for n, t in zip(names, kinds))
    args = ', '.join(names)
    member = 'real' if hint['return'] == 'float' else 'integer'
    first = bool(kinds) and kinds[0] == 'int'; zero = not params
    primitive = ' and '.join(f'({n} is None or type({n}) is str)' for n, t in zip(names[1:], kinds[1:]) if t in ('object', 'str')) or 'True'
    fullkey = names[0] if len(names) == 1 else '(' + args + (',' if names else '') + ')'
    entry = ('_virtual_cell_' if virtual else '_typed_cell_') + name
    body=['    @cython.overflowcheck(False)', f'    cdef {ct} {entry}(self'+(', object compute' if virtual else '')+(', '+sig if sig else '')+') except? -1:',
          '        cdef _MemoSlab slab = None','        cdef Py_ssize_t index = -1',
          '        cdef object key, cached, raw','        cdef uint64_t stamp',f'        cdef {ct} value',
          '        cdef bint complete = False']
    if diagnostics:body += [f'        self._ocalls_{name} += 1','        self._frame._calls += 1']
    if zero:
        body += [f'        stamp = self._typed_stamp_{name}',
                 '        if stamp == self._typed_epoch + 1:',f'            return self._typed_value_{name}',
                 '        if stamp == self._typed_epoch:',f'            raise RuntimeError("Cell cycle: {name}")',
                 f'        self._typed_stamp_{name} = self._typed_epoch']
    else:
        if first:
            t=names[0]
            body += [f'        if -1 <= {t} <= {steps} and ({primitive}):',f'            index = {t} + 1']
            if len(params)==1:body += [f'            slab = self._typed_slab_{name}']
            else:
                body += [f'            slab = <object>self._typed_axis_{name}('+', '.join(names[1:])+')']
            body += ['            stamp = slab.stamps[index]',
                     '            if stamp == self._typed_epoch + 1:',f'                return slab.values[index].{member}',
                     '            if stamp == self._typed_epoch:',f'                raise RuntimeError("Cell cycle: {name}")',
                     '            slab.stamps[index] = self._typed_epoch']
        body += ['        if slab is None:',f'            key = {fullkey}',
                 '            self._object_key_guard = True','            try:',
                 f'                cached = self._typed_sparse_{name}.get(key, _DICT_EMPTY)',
                 '                if cached is _DICT_BUSY:',f'                    raise RuntimeError("Cell cycle: {name}")',
                 '                if cached is not _DICT_EMPTY: return cached',
                 f'                self._typed_sparse_{name}[key] = _DICT_BUSY',
                 '            finally:','                self._object_key_guard = False']
    if diagnostics:body += [f'        self._omisses_{name} += 1','        self._frame._misses += 1']
    body += ['        try:']
    if not compiled:
        body += ['            if self._python_scope.diagnostics:',f'                self._python_scope.calls[{name!r}] += 1']
    calc='_gil_formula_' if compiled else '_object_fn_'
    if compiled:
        body += [f'            value = compute({args})' if virtual else f'            value = self.{calc}{name}({args})']
    else:
        body += [f'            raw = self.{calc}{name}({args})',
                 '            if raw is None: raise ValueError("None violates a numeric Cell contract")']
        if hint['return']=='bool':
            body += ['            if not isinstance(raw, (bool, np.bool_)): raise TypeError("Boolean Cell result required")',
                     '            value = raw']
        else:
            converter='_checked_memo_float' if hint['return']=='float' else '_checked_memo_int'
            body += [f'            value = {converter}(raw)']
    if zero:body += [f'            self._typed_value_{name} = value',f'            self._typed_stamp_{name} = self._typed_epoch + 1']
    else:
        body += ['            if slab is not None:',f'                slab.values[index].{member} = value',
                 '                slab.stamps[index] = self._typed_epoch + 1','            else:',
                 '                self._object_key_guard = True','                try:',
                 f'                    self._typed_sparse_{name}[key] = value',
                 '                finally:','                    self._object_key_guard = False']
    body += ['            complete = True','            return value','        finally:',
             '            if not complete:']
    if zero:body += [f'                self._typed_stamp_{name} = 0']
    else:
        body += ['                if slab is not None: slab.stamps[index] = 0','                else:',
                 '                    self._object_key_guard = True','                    try:',
                 f'                        self._typed_sparse_{name}.pop(key, None)',
                 '                    finally:','                        self._object_key_guard = False']
    return body + ['']


def augment(source, manifest, spec, signatures, compiled):
    numeric={n:h for n,h in signatures.items() if h['return'] in ('int','float','bool')}
    if not numeric:return source
    fields=['    cdef uint64_t _typed_epoch','    cdef list _typed_slabs']
    init=['        self._typed_epoch = 1','        self._typed_slabs = []']
    methods=[]; stats={}
    for name,hint in numeric.items():
        row=manifest['cells'][name]; params=row['params']; original_names=[p['name'] for p in params]
        names=['_mx_arg_'+str(i) for i in range(len(params))]
        kinds=[hint['parameters'][n] for n in original_names];ct=CTYPES[hint['return']]
        sig=', '.join(CTYPES[t]+' '+n for n,t in zip(names,kinds));args=', '.join(names)
        member='real' if hint['return']=='float' else 'integer'
        first=bool(kinds) and kinds[0]=='int'; zero=not params
        primitive = ' and '.join(f'({n} is None or type({n}) is str)' for n,t in zip(names[1:],kinds[1:]) if t in ('object','str')) or 'True'
        # Slab heads and sparse overflow dictionaries contain memo storage, not inputs.
        fields += [f'    cdef dict _typed_sparse_{name}']
        init += [f'        self._typed_sparse_{name} = {{}}']
        if zero:
            fields += [f'    cdef uint64_t _typed_stamp_{name}',f'    cdef {ct} _typed_value_{name}']
        elif first and len(params)==1:
            fields += [f'    cdef _MemoSlab _typed_slab_{name}']
            init += [f'        self._typed_slab_{name} = _MemoSlab({spec.steps+2})',f'        self._typed_slabs.append(self._typed_slab_{name})']
        elif first:
            fields += [f'    cdef dict _typed_axes_{name}']
            init += [f'        self._typed_axes_{name} = {{}}']
        if first and len(params)>1:
            af, ai, am = axis_entries(name, names[1:], kinds[1:], spec)
            fields += af; init += ai; methods += am
        fullkey=names[0] if len(names)==1 else '('+args+(',' if names else '')+')'
        methods += cached_entry(name, hint, row, spec.steps, manifest['diagnostics'], name in compiled)
        # Replace, rather than layer another cache over, the existing object entry.
        anchor=f'    cdef object _object_cell_{name}('
        start=source.index(anchor);match=re.search(r'^    (?:cdef |def )',source[start+len(anchor):],flags=re.M)
        end=start+len(anchor)+match.start() if match else len(source)
        wrapper=f'    cdef object _object_cell_{name}(self'+(' , '+', '.join('object '+n for n in names) if names else '')+'):\n'
        converted=[f'_checked_memo_int({n})' if k in ('int','bool') else f'_checked_memo_float({n})' if k=='float' else n for n,k in zip(names,kinds)]
        wrapper+=f'        return self._typed_cell_{name}('+', '.join(converted)+')\n\n'
        source=source[:start]+wrapper+source[end:]
        row['typed_signature']=hint
        row['calculator']='cython_scalar' if name in compiled else 'python'
        row['storage']='typed_epoch_slabs' if first else 'typed_scalar' if zero else 'typed_python_dict'
        stats[name]=row['storage']
    fields='\n'.join(fields)+'\n';source=source.replace('cdef class Kernel:\n','cdef class Kernel:\n'+fields,1)
    source=source.replace('    def __init__(self, records):\n','    def __init__(self, records):\n'+'\n'.join(init)+'\n',1)
    # The existing idle/finalisation guard remains the sole reset guard.
    reset=['        self._typed_epoch += 2',
           '        if self._typed_epoch == 0xffffffffffffffff:',
           '            for _slab in self._typed_slabs:',
           '                memset((<_MemoSlab>_slab).stamps, 0, (<_MemoSlab>_slab).size * sizeof(uint64_t))',
           '            self._typed_epoch = 1']
    # scalar epochs also need resetting on overflow; normal epoch transitions are O(1).
    # Dictionaries are sparse non-integral/out-of-bound keys and always clear on reset.
    for name,h in numeric.items():
        reset += [f'        self._typed_sparse_{name}.clear()']
        if not manifest['cells'][name]['params']:
            reset += [f'        if self._typed_stamp_{name} > self._typed_epoch + 1: self._typed_stamp_{name} = 0']
    start=source.index('    cdef void _clear_object_cache(self) except *:\n')
    at=source.index('        try:\n',start)+len('        try:\n')
    source=source[:at]+'\n'.join('    '+line for line in reset)+'\n'+source[at:]
    # Place support definitions before the generated Kernel (after Cython directives).
    source=source.replace('@cython.final\ncdef class Kernel:\n',HEADER+'\n@cython.final\ncdef class Kernel:\n',1)
    manifest['typed_memo_cells']=stats
    manifest['cython_signatures']=signatures
    source=source+'\n'+'\n'.join(methods)+'\n'
    return add_hot_entries(source,manifest,spec,numeric,compiled)


def add_hot_entries(source, manifest, spec, numeric, compiled):
    """Keep Python object ownership/exception cleanup off numeric cache-hit paths."""
    fields=[];init=[];methods=[]
    for name,hint in numeric.items():
        row=manifest['cells'][name];original_names=[p['name'] for p in row['params']]
        names=['_mx_arg_'+str(i) for i in range(len(original_names))]
        kinds=[hint['parameters'][n] for n in original_names];ct=CTYPES[hint['return']]
        if names and kinds[0]!='int':continue
        args=', '.join(names);sig=', '.join(CTYPES[t]+' '+n for n,t in zip(names,kinds))
        old=f'    cdef {ct} _typed_cell_{name}('
        source=source.replace(old,f'    cdef {ct} _typed_compute_{name}(',1)
        body=['    @cython.overflowcheck(False)', f'    cdef inline {ct} _typed_cell_{name}(self'+(', '+sig if sig else '')+') except? -1:']
        member='real' if hint['return']=='float' else 'integer'
        if not names:
            body += [f'        if self._typed_stamp_{name} == self._typed_epoch + 1:',f'            return self._typed_value_{name}']
        elif len(names)==1:
            t=names[0]
            body += [f'        if -1 <= {t} <= {spec.steps}:',
                     f'            if self._typed_slab_{name}.stamps[{t}+1] == self._typed_epoch + 1:',
                     f'                return self._typed_slab_{name}.values[{t}+1].{member}',
                     f'            return self._typed_dense_{name}(&self._typed_slab_{name}.stamps[{t}+1], &self._typed_slab_{name}.values[{t}+1], {args})']
        else:
            t=names[0]
            primitive = ' and '.join(f'({n} is None or type({n}) is str)' for n,t in zip(names[1:],kinds[1:]) if t in ('object','str')) or 'True'
            body += ['        cdef PyObject* found',f'        if -1 <= {t} <= {spec.steps} and ({primitive}):',
                     f'            found = self._typed_axis_{name}('+', '.join(names[1:])+')',
                     f'            if (<_MemoSlab>found).stamps[{t}+1] == self._typed_epoch + 1:',
                     f'                return (<_MemoSlab>found).values[{t}+1].{member}',
                     f'            return self._typed_dense_{name}(&(<_MemoSlab>found).stamps[{t}+1], &(<_MemoSlab>found).values[{t}+1], {args})']
        if manifest['diagnostics']:
            changed=[]
            for line in body:
                if line.lstrip().startswith('return ') and '_typed_dense_' not in line:
                    ind=line[:len(line)-len(line.lstrip())]
                    changed += [ind+f'self._ocalls_{name} += 1',ind+f'self._ohits_{name} += 1',ind+'self._frame._calls += 1']
                changed.append(line)
            body=changed
        body += [f'        return self._typed_compute_{name}({args})','']
        methods+=body
        if names:
            methods += dense_compute(name,hint,names,kinds,manifest,name in compiled)
    source='from cpython.dict cimport PyDict_GetItemWithError\n'+source
    return source+'\n'+'\n'.join(methods)+'\n'


def axis_entries(name, names, kinds, spec):
    """A bounded inline *index*, not a value cache or a closed key domain.

    The owning dictionary remains authoritative and stores every encountered
    axis. Sixteen primitive heads eliminate repeated tuple allocation/hashing
    for common axes; all other keys use normal guarded dictionary lookup.
    """
    sig=', '.join(CTYPES[t]+' '+n for n,t in zip(names,kinds))
    args=', '.join(names)
    eligible=len(names)<=2 and all(t in ('int','bool','object','str') for t in kinds)
    fields=[];init=[];methods=[]
    if eligible:
        fields=[f'    cdef _MemoAxisHead _typed_heads_{name}[16]',
                f'    cdef int _typed_head_count_{name}',
                f'    cdef int _typed_head_last_{name}',
                f'    cdef list _typed_head_owners_{name}']
        init=[f'        self._typed_head_owners_{name} = []']
    key=names[0] if len(names)==1 else '('+args+')'
    m=['    @cython.overflowcheck(False)',
       f'    cdef PyObject* _typed_axis_slow_{name}(self, {sig}) except NULL:',
       '        cdef _MemoSlab slab', '        cdef object key', '        cdef int slot',
       f'        key = {key}', '        self._object_key_guard = True','        try:',
       f'            slab = self._typed_axes_{name}.get(key)',
       '            if slab is None:', f'                slab = _MemoSlab({spec.steps+2})',
       f'                self._typed_axes_{name}[key] = slab',
       '                self._typed_slabs.append(slab)',
       '        finally:', '            self._object_key_guard = False']
    primitive=' and '.join(f'({n} is None or type({n}) is str)' for n,t in zip(names,kinds) if t in ('object','str')) or 'True'
    if eligible:
        m += [f'        if self._typed_head_count_{name} < 16 and ({primitive}):',
              f'            slot = self._typed_head_count_{name}',
              f'            self._typed_head_owners_{name}.append({key})',
              f'            self._typed_heads_{name}[slot].slab = <PyObject*>slab']
        for j,(n,t) in enumerate(zip(names,kinds)):
            field=f'object{j}' if t in ('object','str') else f'integer{j}'
            val='<PyObject*>'+n if t in ('object','str') else n
            m += [f'            self._typed_heads_{name}[slot].{field} = {val}']
        m += [f'            self._typed_head_count_{name} += 1',f'            self._typed_head_last_{name} = slot']
    m += ['        return <PyObject*>slab','']
    m += ['    @cython.overflowcheck(False)',
          f'    cdef inline PyObject* _typed_axis_{name}(self, {sig}) except NULL:']
    if eligible:
        comparisons=[]
        for j,(n,t) in enumerate(zip(names,kinds)):
            field=f'object{j}' if t in ('object','str') else f'integer{j}'
            val='<PyObject*>'+n if t in ('object','str') else n
            comparisons += [f'head.{field} == {val}']
        test=' and '.join(comparisons)
        m += ['        cdef _MemoAxisHead* head', '        cdef int slot',
              f'        if self._typed_head_count_{name}:',
              f'            head = &self._typed_heads_{name}[self._typed_head_last_{name}]',
              f'            if {test}: return head.slab',
              f'            for slot in range(self._typed_head_count_{name}):',
              f'                head = &self._typed_heads_{name}[slot]',
              f'                if {test}:',f'                    self._typed_head_last_{name} = slot',
              '                    return head.slab']
    m += [f'        return self._typed_axis_slow_{name}({args})','']
    return fields,init,m


def dense_compute(name,hint,names,kinds,manifest,compiled):
    """The common miss path has no dictionary or slab-ownership bookkeeping.

    Borrowed pointers remain valid because the kernel owns the slab and resets
    are prohibited during evaluation. Sparse/opaque keys retain the slow path.
    """
    ct=CTYPES[hint['return']];member='real' if hint['return']=='float' else 'integer'
    sig=', '.join(CTYPES[t]+' '+n for n,t in zip(names,kinds));args=', '.join(names)
    body=['    @cython.overflowcheck(False)',
          f'    cdef {ct} _typed_dense_{name}(self, uint64_t* stamp, _MemoNumber* slot, {sig}) except? -1:',
          f'        cdef {ct} value', '        cdef bint complete = False']
    if not compiled:body+=['        cdef object raw']
    if manifest['diagnostics']:body += [f'        self._ocalls_{name} += 1','        self._frame._calls += 1']
    body += ['        if stamp[0] == self._typed_epoch:',f'            raise RuntimeError("Cell cycle: {name}")',
             '        stamp[0] = self._typed_epoch']
    if manifest['diagnostics']:body += [f'        self._omisses_{name} += 1','        self._frame._misses += 1']
    body += ['        try:']
    if compiled:body += [f'            value = self._gil_formula_{name}({args})']
    else:
        body += ['            if self._python_scope.diagnostics:',f'                self._python_scope.calls[{name!r}] += 1',
                 f'            raw = self._object_fn_{name}({args})',
                 '            if raw is None: raise ValueError("None violates a numeric Cell contract")']
        if hint['return']=='bool':
            body += ['            if not isinstance(raw, (bool, np.bool_)): raise TypeError("Boolean Cell result required")',
                     '            value = raw']
        else:
            converter='_checked_memo_float' if hint['return']=='float' else '_checked_memo_int'
            body += [f'            value = {converter}(raw)']
    body += [f'            slot[0].{member} = value', '            stamp[0] = self._typed_epoch + 1',
             '            complete = True','            return value', '        finally:',
             '            if not complete: stamp[0] = 0', '']
    return body


def completed_cache_methods(rows):
    """Serialize logical completed entries without evaluating any formula.

    Existing dense/sparse lookup, epoch reset and key conventions are unchanged.
    Restoring into the same physical store keeps one authoritative cache.
    """
    methods = []
    for name, row in rows.items():
        if row.get('storage') not in {'typed_scalar', 'typed_epoch_slabs', 'typed_python_dict'}:
            continue
        params = row['params']; count = len(params)
        hint = row['typed_signature']
        kinds = [hint['parameters'][p['name']] for p in params]
        first = bool(kinds) and kinds[0] == 'int'
        member = 'real' if hint['return'] == 'float' else 'integer'
        methods += [f'    def _typed_snapshot_{name}(self):',
                    '        cdef _MemoSlab slab', '        cdef Py_ssize_t i',
                    '        result = {}',
                    f'        for key, value in self._typed_sparse_{name}.items():',
                    '            if value is not _DICT_BUSY: result[key] = value']
        if not count:
            methods += [f'        if self._typed_stamp_{name} == self._typed_epoch + 1:',
                        f'            result[()] = self._typed_value_{name}']
        elif first:
            if count == 1:
                methods += [f'        slab = self._typed_slab_{name}',
                            '        for i in range(slab.size):',
                            '            if slab.stamps[i] == self._typed_epoch + 1:',
                            f'                result[i-1] = slab.values[i].{member}']
            else:
                axis = 'axis' if count == 2 else 'axis[' + str(0) + ']'
                tails = ['axis'] if count == 2 else ['axis['+str(j)+']' for j in range(count-1)]
                key = '(' + ', '.join(['i-1', *tails]) + ')'
                methods += [f'        for axis, raw_slab in self._typed_axes_{name}.items():',
                            '            slab = raw_slab', '            for i in range(slab.size):',
                            '                if slab.stamps[i] == self._typed_epoch + 1:',
                            f'                    result[{key}] = slab.values[i].{member}']
        methods += ['        return result', f'    def _typed_restore_{name}(self, dict entries):',
                    '        cdef _MemoSlab slab', '        cdef Py_ssize_t i',
                    '        for key, value in entries.items():']
        if not count:
            methods += ['            if key != (): raise ValueError("Invalid scalar memo key")',
                        f'            self._typed_value_{name} = value',
                        f'            self._typed_stamp_{name} = self._typed_epoch + 1']
        elif first:
            primitive = ' and '.join(f'(key[{j}] is None or type(key[{j}]) is str)'
                                    for j in range(1, count) if kinds[j] in ('object', 'str')) or 'True'
            index = 'key' if count == 1 else 'key[0]'
            if count == 1:
                methods += [f'            slab = self._typed_slab_{name}']
            else:
                args = ', '.join('key['+str(j)+']' for j in range(1,count))
                methods += ['            slab = None', f'            if {primitive}:',
                            f'                slab = <object>self._typed_axis_{name}({args})']
            methods += [f'            i = {index} + 1',
                        '            if slab is not None and 0 <= i < slab.size:',
                        f'                slab.values[i].{member} = value',
                        '                slab.stamps[i] = self._typed_epoch + 1', '            else:',
                        f'                self._typed_sparse_{name}[key] = value']
        else:
            methods += [f'            self._typed_sparse_{name}[key] = value']
    return methods
