from libc.math cimport fabs
from modelx_cython._float_sum_state cimport F64Sum, f64_init, f64_add, f64_result

cdef object lookup(object function, object table, object key, object column, object default=*)
cdef Py_ssize_t indexed_position(object table, object key) except? -2
cdef object previous(object function, object table, object key, object column, object default=*)
cdef object contains(object function, object table, object key)
cdef object key_count(object function, object table)
cdef object key_at(object function, object table, object index)
cdef object first_key(object function, object table)
cdef object last_key(object function, object table)
cdef object subtable(object function, object table, object key)
cdef object has_group(object function, object table, object key)
cdef object sum_f64(object function, object values)
cdef object sum_helper(object function, object values)
cdef object any_true(object function, object values)
cdef bint any_true_accepts(object function)
cdef object all_true(object function, object values)
cdef bint all_true_accepts(object function)
cdef object max_value(object function, object values)
cdef bint max_value_accepts(object function)

cdef inline double f64_abs(double value) noexcept nogil:
    return fabs(value)

cdef object prepared_interpolation(object function, object table, object column, object years,
                                   object lookup_fn, object last_fn, object math_module)

cdef bint f64_accepts(object function)


cdef class LookupContext:
    cdef bint _private_owner
    cdef object group_data, bounded_data
    cdef Py_ssize_t bounded_column
    cdef object numeric_source, numeric_column_source, numeric_data
    cdef Py_ssize_t numeric_index
    cdef object first, second
    cdef object data_first, data_second
    cdef object column_first, column_second
    cdef Py_ssize_t index_first, index_second
    cdef tuple grouped
    cdef object keyed_cache
    cdef object get(self, object table, object column, Py_ssize_t* index)
cdef double lookup_f64(object function, LookupContext context, object table,
                       long long key, object column, object default=*) except? -1
cdef double previous_f64(object function, LookupContext context, object table,
                         long long key, object column, object default=*) except? -1
cdef double lookup_clamped_f64(object function, object table, long long key,
                               object last_function, object column) except? -1
cdef double lookup_clamped_context_f64(object function, LookupContext context, object table,
                                       long long key, object last_function, object column) except? -1
cdef bint bounded_lookup_f64(object function, object first_fn, object last_fn,
                             LookupContext context, object table, long long key,
                             str column, double below, double* value) except? -1
cdef int optional_lookup_f64(object contains_fn, object lookup_fn, object table,
                             long long key, object column, double* result) except? -1
cdef object group_probe(object has_fn, object record, long long group, str column,
                        Py_ssize_t* slot)
cdef object group_context_probe(LookupContext context, object has_fn, object record,
                                long long group, str column, Py_ssize_t* slot)
cdef int group_value(object sub_fn, object contains_fn, object lookup_fn, object prepared,
                     Py_ssize_t slot, long long key, double* value) except -1

cdef bint numeric_resource_ready(LookupContext context, const void* source, const void* column) noexcept
cdef double bound_lookup_f64(LookupContext context, long long key, long long column,
                             bint named, bint has_default, double default) except? -1
cdef double bound_previous_f64(LookupContext context, long long key, long long column,
                               bint named, bint has_default, double default) except? -1

cdef double prepare_lookup_f64(object function, LookupContext context, object table,
                               long long key, object column, object default=*) except? -1
cdef double prepare_previous_f64(object function, LookupContext context, object table,
                                 long long key, object column, object default=*) except? -1

cdef bint bound_bounded_f64(LookupContext context, long long key,
                            double below, double* value) except? -1

cdef bint bound_group_probe(LookupContext context, long long group, Py_ssize_t* slot) except? -1
cdef int bound_group_value(LookupContext context, Py_ssize_t slot, long long key, double* value) except -1
cdef int group_context_value(LookupContext context, object sub_fn, object contains_fn,
    object lookup_fn, object prepared, Py_ssize_t slot, long long key, double* value) except -1

# Explicit float conversion, distinct from implicit C double conversion.
cdef double lookup_float_f64(object conversion, object function, LookupContext context,
                             object table, long long key, object column,
                             object default=*) except? -1

# Compound lookup keys are packed on the native stack. Object arguments stay
# strongly referenced for the entire fused wrapper call; exact Unicode admits.
from cpython.ref cimport PyObject
cdef struct KeyPart:
    int kind
    long long number
    PyObject* text
cdef void key_int(KeyPart*, long long, bint) noexcept
cdef void key_text(KeyPart*, object) noexcept
cdef double lookup_keyed_float_f64(object, object, LookupContext, object,
    const KeyPart*, Py_ssize_t, bint, object, object default=*) except? -1
ctypedef long long KeyInteger
ctypedef fused KeyArg0:
    KeyInteger
    bint
    object

ctypedef fused KeyArg1:
    KeyInteger
    bint
    object

ctypedef fused KeyArg2:
    KeyInteger
    bint
    object

ctypedef fused KeyArg3:
    KeyInteger
    bint
    object

cdef double lookup_key0_float_f64(object conversion, object function, LookupContext context, object table, bint tuple_key, object column, object default=*) except? -1

cdef double lookup_key1_float_f64(object conversion, object function, LookupContext context, object table, KeyArg0 key0, bint tuple_key, object column, object default=*) except? -1

cdef double lookup_key2_float_f64(object conversion, object function, LookupContext context, object table, KeyArg0 key0, KeyArg1 key1, bint tuple_key, object column, object default=*) except? -1

cdef double lookup_key3_float_f64(object conversion, object function, LookupContext context, object table, KeyArg0 key0, KeyArg1 key1, KeyArg2 key2, bint tuple_key, object column, object default=*) except? -1

cdef double lookup_key4_float_f64(object conversion, object function, LookupContext context, object table, KeyArg0 key0, KeyArg1 key1, KeyArg2 key2, KeyArg3 key3, bint tuple_key, object column, object default=*) except? -1

cdef object indexed_lookup(object,object,object,object default=*)
cdef object indexed_contains(object,object)

cdef object omitted_argument
cdef double lookup_float(object function, tuple keywords, object source, object key, object column=*, object default=*) except? -1
cdef long long lookup_int(object function, tuple keywords, object source, object key, object column=*, object default=*) except? -1
cdef bint lookup_bool(object function, tuple keywords, object source, object key, object column=*, object default=*) except? -1
cdef str lookup_str(object function, tuple keywords, object source, object key, object column=*, object default=*)

cdef object lookup_object(object function, tuple keywords, object source, object key, object column=*, object default=*)
cdef bint is_missing(object function, tuple keywords, object predicate, object value) except? -1


cdef object row(object function, tuple keywords, object table, object key)
cdef object column(object function, tuple keywords, object table, object name, object report=*)
cdef object group_sum(object function, tuple keywords, object table, object labels, object first=*, object last=*)
cdef object table(object function, tuple keywords, object frame, object columns=*, object groups=*, object indexing=*, object rows=*, object report=*, object preserve_scalar_types=*)
