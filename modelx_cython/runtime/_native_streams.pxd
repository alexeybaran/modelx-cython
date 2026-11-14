from modelx_cython._float_sum_state cimport F64Sum, f64_init, f64_add, f64_result
# W1.4: the exact built-in sum state used by lowered sum(...) (see reductions.GeneralReductions)
from modelx_cython._float_sum_state cimport (
    PySum, sum_init, sum_init_ll, sum_init_f64, sum_init_object, sum_add_f64, sum_add_ll,
    sum_add_bint, sum_add_object, sum_result, sum_result_f64, sum_clear)
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

cdef object over(object current_callable, tuple keywords, object source, object function=*, object where=*, object unpack=*)
cdef bint f64_accepts(object function)

cdef class ProtocolGenerator:
    cdef object generator
    cdef bint started

