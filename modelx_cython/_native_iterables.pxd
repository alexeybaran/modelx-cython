"""Identity admission for generic builtin boolean reductions."""
cdef bint all_accepts(object function)
cdef bint any_accepts(object function)
cdef bint min_accepts(object function)
cdef bint max_accepts(object function)
cdef double f64_abs(double value) noexcept
cdef double f64_min_pair(double first, double second) noexcept
cdef double f64_max_pair(double first, double second) noexcept

cdef struct I64Range:
    long long current
    long long stop
    long long step
    bint active

cdef void i64_range_init(I64Range* state, long long start, long long stop, long long step) except *
cdef bint i64_range_next(I64Range* state, long long* value) noexcept
