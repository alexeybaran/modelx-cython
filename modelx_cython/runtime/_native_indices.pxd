cdef class KeySpan:
    cdef object owner
    cdef long long[::1] values

cdef object omitted_argument
cdef object keys(object function, tuple keywords, object table)
cdef KeySpan keys_span(object function, object table)
cdef Py_ssize_t key_span_count(KeySpan span) noexcept
cdef long long key_span_at(KeySpan span, Py_ssize_t position) except? -1

cdef class FloatKeySpan:
    cdef object owner
    cdef double[::1] values

cdef FloatKeySpan keys_float_span(object function, object table)
cdef Py_ssize_t float_key_span_count(FloatKeySpan span) noexcept
cdef double float_key_span_at(FloatKeySpan span, Py_ssize_t position) except? -1

cdef class StrKeySpan:
    cdef object owner
    cdef tuple values

cdef StrKeySpan keys_str_span(object function, object table)
cdef Py_ssize_t str_key_span_count(StrKeySpan span) noexcept
cdef str str_key_span_at(StrKeySpan span, Py_ssize_t position)
