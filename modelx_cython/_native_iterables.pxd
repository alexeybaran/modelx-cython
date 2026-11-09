"""Identity admission for generic builtin boolean reductions."""
cdef bint all_accepts(object function)
cdef bint any_accepts(object function)
cdef bint min_accepts(object function)
cdef bint max_accepts(object function)
cdef double f64_abs(double value) noexcept
cdef double f64_min_pair(double first, double second) noexcept
cdef double f64_max_pair(double first, double second) noexcept

from modelx_cython._range_state cimport I64Range

cdef void i64_range_init(I64Range* state, long long start, long long stop, long long step) except *
cdef bint i64_range_next(I64Range* state, long long* value) noexcept
cdef long long i64_min_pair(long long first, long long second) noexcept
cdef long long i64_max_pair(long long first, long long second) noexcept
