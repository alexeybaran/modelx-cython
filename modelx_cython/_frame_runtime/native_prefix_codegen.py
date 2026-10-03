"""Existing indexed memo layout with unrestricted growing overflow.

This is cache generation, not formula lowering. A full hash prefix migrates
completed entries atomically; no item pointer survives formula execution.
"""


def slot_function(name, row, arguments):
    params = row['params']
    declaration = ', '.join(p['type']['ctype'] + ' ' + a for p,a in zip(params,arguments))
    lines = ['@cython.overflowcheck(False)', '@cython.boundscheck(False)',
             '@cython.wraparound(False)', '@cython.initializedcheck(False)',
             f'cdef inline Py_ssize_t _nc_prefix_slot_{name}(_Frame* state, {declaration}) noexcept nogil:',
             '    cdef Py_ssize_t slot, bucket, probe',
             '    cdef unsigned long long h = 1469598103934665603',
             '    cdef _DoubleBits bits',
             f'    if state._numeric_fallback_{name} != NULL: return -1']
    if row['hashed']:
        lines.append(f'    if state._prefix_disabled_{name}: return -1')
        for p,a in zip(params,arguments):
            if p['type']['kind'] == 'float':
                lines += [f'    if not isfinite({a}): return -1',
                          f'    bits.value = 0.0 if {a} == 0.0 else {a}',
                          '    h = (h ^ bits.bits) * <unsigned long long>1099511628211']
            else:
                lines.append(f'    h = (h ^ <unsigned long long>{a}) * <unsigned long long>1099511628211')
        lines += ['    h = (h ^ (h >> 30)) * <unsigned long long>13787848793156543929',
                  '    h = (h ^ (h >> 27)) * <unsigned long long>10723151780598845931',
                  '    h ^= h >> 31', f'    bucket = h & {row["slots"]-1}',
                  f'    for probe in range({row["slots"]}):',
                  f'        slot = {row["offset"]} + bucket']
        occupied = 'state._seen[slot] == state._epoch'
        if row['recursive']:
            occupied += ' or state._seen[slot] == state._epoch - 1'
        equal = ' and '.join(f'state._keys_{name}_{p["name"]}[bucket] == {a}' for p,a in zip(params,arguments))
        lines += [f'        if not ({occupied}) or ({equal}): return slot',
                  f'        bucket = (bucket + 1) & {row["slots"]-1}', '    return -2']
    else:
        index = '0'
        for p,a in zip(params,arguments):
            lines.append(f'    if {a} < {p["lower"]} or {a} >= {p["lower"]+p["size"]}: return -1')
            index = f'(({index}) * {p["size"]} + ({a} - ({p["lower"]})))'
        lines.append(f'    return {row["offset"]} + {index}')
    return '\n'.join(lines)+'\n'


def key_values(name, row, index):
    if row['hashed']:
        return [f'state._keys_{name}_{p["name"]}[{index}]' for p in row['params']]
    result = []
    for i,p in enumerate(row['params']):
        divisor = 1
        for later in row['params'][i+1:]: divisor *= later['size']
        result.append(f'({index} // {divisor}) % {p["size"]} + ({p["lower"]})')
    return result


def snapshot_function(name, row):
    store = '_floats' if row['result']['kind']=='float' else '_ints'
    keys = key_values(name,row,'i')
    key = keys[0] if len(keys)==1 else '('+', '.join(keys)+')'
    lines = [f'cdef dict _nc_prefix_snapshot_{name}(_Frame* state):',
             '    cdef Py_ssize_t i', '    cdef dict result = {}']
    if row['hashed']:
        lines.append(f'    if state._prefix_disabled_{name}: return result')
    lines += ['    if state._epoch == 0: return result',
              f'    for i in range({row["slots"]}):',
              f'        if state._seen[{row["offset"]} + i] == state._epoch:',
              f'            result[{key}] = state.{store}[{row["offset"]} + i]',
              '    return result']
    return '\n'.join(lines)+'\n'


def promote_function(name, row):
    if not row['hashed']: return ''
    from .growing_numeric_codegen import tag
    t=tag(name);store='_floats' if row['result']['kind']=='float' else '_ints'
    keys=', '.join(key_values(name,row,'i'))
    types=', '.join([t+'_cache*']+[p['type']['ctype'] for p in row['params']]+[row['result']['ctype']])
    return f'''ctypedef int (*_NcPrefixPut_{name})({types}) noexcept nogil
cdef int _nc_promote_{name}(_Frame* state) except -1 nogil:
    cdef {t}_cache staged
    cdef Py_ssize_t i
    cdef int status
    memset(&staged, 0, sizeof({t}_cache))
    for i in range({row['slots']}):
        if state._seen[{row['offset']} + i] == state._epoch:
            status = (<_NcPrefixPut_{name}>{t}_put)(&staged, {keys}, state.{store}[{row['offset']} + i])
            if status < 0:
                # The checked allocator has already set the original exception.
                free(staged.items)
                return -1
    free(state._numeric_{name}.items)
    state._numeric_{name} = staged
    state._prefix_disabled_{name} = True
    return 0
'''


def commit_lines(name, row, arguments, value, indent='    '):
    store='_floats' if row['result']['kind']=='float' else '_ints'
    lines=[]
    if row['hashed']:
        for p,a in zip(row['params'],arguments):
            lines.append(f'{indent}state._keys_{name}_{p["name"]}[slot - {row["offset"]}] = {a}')
    lines += [f'{indent}state.{store}[slot] = {value}',
              f'{indent}state._seen[slot] = state._epoch']
    return lines


def cell_entry(name, row, signature, args, formula, diagnostics):
    from .growing_numeric_codegen import cell_entry as growing_entry
    ret=row['result']['ctype'];arguments=[p['name'] for p in row['params']]
    canonical='_Frame* state'+''.join(', '+p['type']['ctype']+' '+p['name'] for p in row['params'])
    cold=growing_entry(name,row,canonical,args,formula,diagnostics)
    cold=[line.replace('_cell_'+name+'(', '_overflow_'+name+'(') for line in cold]
    store='_floats' if row['result']['kind']=='float' else '_ints'
    lines=cold+[f'cdef inline {ret} _cell_{name}({signature}) except? -1 nogil:',
        '    cdef Py_ssize_t slot',f'    cdef {ret} _mx_value','    cdef int status',
        f'    slot = _nc_prefix_slot_{name}({args})',
        f'    if slot < 0: return _overflow_{name}({args})']
    if diagnostics:lines.append('    state._calls += 1')
    lines += ['    if state._seen[slot] == state._epoch:',f'        return state.{store}[slot]']
    if row['recursive']:
        lines += ['    if state._seen[slot] == state._epoch - 1:',f'        _cache_error(1, b"{name}")',
                  '    state._seen[slot] = state._epoch - 1']
    if row['hashed']:
        for p,a in zip(row['params'],arguments):
            lines.append(f'    state._keys_{name}_{p["name"]}[slot - {row["offset"]}] = {a}')
    if diagnostics:lines.append('    state._misses += 1')
    lines.append(f'    _mx_value = {formula}')
    switched=f'state._numeric_fallback_{name} != NULL'
    if row['hashed']:switched+=f' or state._prefix_disabled_{name}'
    lines += [f'    if {switched}:',f'        status = (<_NcCommit_{name}>_nc_commit_{name})({args}, _mx_value)',
              '        if status < 0: mx_capture(&state._guard)','    else:',
              f'        state.{store}[slot] = _mx_value','        state._seen[slot] = state._epoch',
              '    return _mx_value']
    return lines
