"""Opt-in OpenMP driver for pure scalar native dense recursive programs.

Formula functions are unchanged. Each OS thread owns its memo and guard state.
Like the serial generated API, input binding and execution must not overlap
with another invocation on the same module. Object/hash/ranked runtimes are
rejected instead of being silently made concurrent.
"""
import re


def enable_dense_openmp(source: str) -> str:
    names = {"_epoch", "_memo_error", "_guard_error", "_memo_f64",
             "_memo_f64_state", "_memo_i64", "_memo_i64_state",
             "_memo_bool", "_memo_bool_state"}
    pattern = re.compile(r"^cdef (unsigned int|long long|double|unsigned char) (_\w+)(\[\d+\])?(?: = (\d+))?$", re.M)
    declarations = [m for m in pattern.finditer(source) if m.group(2) in names]
    if {m.group(2) for m in declarations} != names or len(declarations) != len(names):
        raise ValueError("OpenMP requires the native dense scalar memo layout")
    for line in source.splitlines():
        if line.startswith("cdef ") and "(" not in line and not line.startswith("cdef extern"):
            if line == "cdef Py_ssize_t _n = 0" or "[::1] ext" in line or "[:, ::1] ext" in line:
                continue
            if not any(line == m.group(0) for m in declarations):
                raise ValueError("OpenMP cannot privatize this runtime declaration: " + line)
    if "with gil" in source or "cdef object" in source:
        raise ValueError("OpenMP driver requires a wholly native scalar closure")
    block = ['# distutils: language=c++', 'from cython.parallel cimport parallel, prange, threadid',
             'cdef extern from "omp.h" nogil:',
             '    int omp_get_num_threads()', '    void omp_set_dynamic(int)',
             'cdef extern from *:', '    """']
    for m in declarations:
        typ, name, shape, initial = m.groups()
        block.append(f"    static thread_local {typ} mx_tls{name}{shape or ''} = " + ("{};" if shape else f"{initial or 0};"))
    block.append('    """')
    for m in declarations:
        typ, name, shape, _ = m.groups()
        block.append(f'    {typ} {name} "mx_tls{name}"{shape or ""}')
    for m in reversed(declarations):
        source = source[:m.start()] + source[m.end():]
    anchor = "cdef Py_ssize_t _n = 0"
    source = source.replace(anchor, "\n".join(block) + "\n" + anchor + "\ncdef int _last_team_size = 0", 1)
    reset = ["cdef inline void _reset_openmp_worker() noexcept nogil:",
             "    global _epoch, _memo_error, _guard_error", "    _epoch = 1",
             "    _memo_error = 0", "    _guard_error = 0"]
    for m in declarations:
        if m.group(2).endswith('_state'):
            reset.append(f"    memset(&{m.group(2)}[0], 0, {m.group(3)[1:-1]} * sizeof(unsigned int))")
    begin = source.index('cpdef cnp.ndarray run_all():')
    end = source.index('cpdef double bench(', begin)
    driver = """
cpdef cnp.ndarray run_all(int threads=1):
    global _last_team_size
    if threads < 1 or threads > 256:
        raise ValueError('threads must be between 1 and 256')
    cdef cnp.ndarray arr = np.empty(_n, dtype=np.float64)
    cdef double[::1] outv = arr
    cdef cnp.ndarray status = np.zeros((_n, 2), dtype=np.int64)
    cdef long long[:, ::1] statusv = status
    cdef cnp.ndarray team = np.zeros(threads, dtype=np.int32)
    cdef int[::1] teamv = team
    cdef Py_ssize_t i
    omp_set_dynamic(0)
    with nogil, parallel(num_threads=threads):
        _reset_openmp_worker()
        teamv[threadid()] = omp_get_num_threads()
        for i in prange(_n, schedule='static'):
            outv[i] = _eval_one(i)
            statusv[i, 0] = _memo_error
            statusv[i, 1] = _guard_error
    _last_team_size = int(team.max())
    if np.any(status[:, 0]):
        raise RuntimeError('recursive memo failure in OpenMP worker')
    if np.any(status[:, 1]):
        raise ValueError('native formula guard failed in OpenMP worker')
    return arr

cpdef int openmp_team_size():
    return _last_team_size

"""
    return source[:begin] + "\n".join(reset) + "\n" + driver + source[end:]
