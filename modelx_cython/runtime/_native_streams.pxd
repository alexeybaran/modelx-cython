from modelx_cython._float_sum_state cimport F64Sum, f64_init, f64_add, f64_result
from modelx_cython._range_state cimport I64Range

cdef object omitted_argument

cdef class Stream:
    cdef object source, function, predicate, unpack, current, inner, pending
    cdef bint closed, bounded, flattened, identity
    cdef int source_kind
    cdef Py_ssize_t position
    cdef I64Range counter
    cdef bint prepare_next(self) except *
    cdef object take_pending(self)
    cdef object next_value(self)
    cdef object next_flattened(self)
    cdef object next_input(self)

cdef object periods(object current_callable, tuple keywords, object start, object stop=*, object step=*)
cdef object zip_values(object current_callable, tuple keywords, object sources, object strict=*)
cdef object over(object current_callable, tuple keywords, object source, object function=*, object where=*, object unpack=*)
cdef object flatten(object current_callable, tuple keywords, object sources)
cdef object sum_values(object current_callable, tuple keywords, object source)
cdef bint f64_accepts(object function)
cdef double sum_nonzero_products(object function, object values, object weights) except? -1.0
cdef object shift_scale(object function, object values, object weights, object first)

cdef double sum_float_kernel(Stream source) except *
cdef object sum_float(object current_callable, tuple keywords, object source, object function=*, object where=*)

cdef class ProtocolGenerator:
    cdef object generator
    cdef bint started

cdef class ZipValues:
    cdef tuple cursors, result
    cdef bint strict
    cdef object next_row(self)
