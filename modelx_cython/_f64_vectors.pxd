"""Float vectors: the cached value of a cell whose value is a list of floats (W0.11).

An F64Vec owns a C double buffer and its length. The formula that allocates one
writes it through vec_set before it returns; after that nothing writes it, so
readers on any thread need no lock. Python sees a read-only sequence.
Sums use the compensated state of _float_sum_state, which reproduces CPython's
built-in float sum bit for bit.
"""
from libc.math cimport fabs, isfinite
from modelx_cython._float_sum_state cimport F64Sum, f64_init, f64_add, f64_result


cdef class F64Vec:
    cdef double* data
    cdef Py_ssize_t n


cdef F64Vec vec_new(Py_ssize_t n)
cdef F64Vec vec_full(Py_ssize_t n, double value)
cdef F64Vec vec_of(object value)
cdef int _vec_index_error() except -1


cdef inline Py_ssize_t vec_index(F64Vec v, long long i) except -1:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        _vec_index_error()
    return <Py_ssize_t>i


cdef inline double vec_get(F64Vec v, long long i) except? -1.0:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        _vec_index_error()
    return v.data[i]


cdef inline int vec_set(F64Vec v, long long i, double value) except -1:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        _vec_index_error()
    v.data[i] = value
    return 0


cdef inline double vec_sum_range(F64Vec v, long long a, long long b) noexcept:
    """sum(v[a:b]) after Python slice normalisation of both bounds."""
    cdef F64Sum s
    cdef Py_ssize_t i
    cdef long long n = v.n
    if a < 0:
        a += n
        if a < 0:
            a = 0
    elif a > n:
        a = n
    if b < 0:
        b += n
        if b < 0:
            b = 0
    elif b > n:
        b = n
    f64_init(&s)
    for i in range(<Py_ssize_t>a, <Py_ssize_t>b):
        f64_add(&s, v.data[i])
    return f64_result(&s)


cdef inline double vec_sum(F64Vec v) noexcept:
    cdef F64Sum s
    cdef Py_ssize_t i
    f64_init(&s)
    for i in range(v.n):
        f64_add(&s, v.data[i])
    return f64_result(&s)
