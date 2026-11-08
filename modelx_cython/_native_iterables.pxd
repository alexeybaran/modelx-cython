"""Identity admission for generic builtin boolean reductions."""
cdef bint all_accepts(object function)
cdef bint any_accepts(object function)
cdef bint min_accepts(object function)
cdef bint max_accepts(object function)
cdef double f64_abs(double value) noexcept
