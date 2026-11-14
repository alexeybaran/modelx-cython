cimport cython
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
    cdef object numeric_source, numeric_column_source, numeric_data, numeric_function
    cdef Py_ssize_t numeric_index
    cdef object first, second
    cdef object data_first, data_second
    cdef object column_first, column_second
    cdef Py_ssize_t index_first, index_second
    cdef tuple grouped
    cdef object keyed_cache
    cdef object record_float, record_namespace, input_first, input_second
    cdef unsigned char clamped_bindings_ready
    cdef object clamped_source, clamped_column, clamped_namespace
    cdef Py_ssize_t clamped_index
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
cdef bint numeric_binding_ready(LookupContext context, const void* source, const void* column,
                                object function) noexcept
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
cdef object group_slice_context_probe(LookupContext context, object record, long long group,
                                      str column, Py_ssize_t* slot)
cdef int group_slice_context_value(LookupContext context, object prepared, Py_ssize_t slot,
                                   long long key, double* value) except -1

# Packed keys borrow object components from the key owner during lookup.
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

# ---- M1 typed table handles (interface fixed by W1.1; bodies W1.2) --------------------
# K = TableKey: a key part is a C long long (int, enum member, bool) or an object (str, other).
# n = 1..6 key parts. Errors are the original pandas errors (KeyError(original key) etc.).
ctypedef fused TableKey:
    long long
    object

cdef enum TableScheme:
    TABLE_DENSE = 1     # int range, row = key - base
    TABLE_SORTED = 2    # sorted int keys
    TABLE_HASHED = 3    # packed KeyPart hash index (KeyedView scheme)
    TABLE_PRODUCT = 4   # rectangular: one index per axis

# Private handle storage (W1.2). A key part normalised for comparison: kinds
# 0 never matches, 1 int (bools too, flagged), 2 non-integral float, 3 str, 4 missing (NaN/None
# under a pandas index spec), 5 any other object (compared with ==, counted as a fallback).
cdef struct HandlePart:
    unsigned char kind
    unsigned char boolean
    long long i
    double f
    PyObject* o             # borrowed: the handle keeps the label alive
    Py_hash_t h

# A key index over `count` keys of `nparts` parts each: DENSE (ints first..last in order),
# SORTED (ascending ints, row of each) or HASHED (open addressing over stored parts).
cdef struct HandleKeys:
    int scheme
    Py_ssize_t count, nparts, pstride, poff
    long long first, last
    const long long* skeys
    const Py_ssize_t* srows
    const HandlePart* parts     # key r, part j: parts[r * pstride + poff + j]
    const Py_ssize_t* slots     # row + 1, 0 empty
    size_t mask
    const unsigned char* modes  # pandas boolean label mode per part: 0 any, 1 bool only, 2 no bool

@cython.final
cdef class TableHandle:
    # Immutable after __init__; built once per bound table, children eagerly (D25 Q2).
    cdef readonly object owner      # helper Table (Python fallbacks, pickling)
    cdef int scheme
    cdef Py_ssize_t rows, width, levels, base   # base/rows: a sub-table views its parent's arrays
    # private (W1.2): everything below is set by the constructor and never written afterwards
    cdef object _hold               # buffers, part arrays, labels: kept alive for the C pointers
    cdef tuple _prefix              # prefix from the root table (() for a root)
    cdef tuple _columns
    cdef tuple _children            # depth-1 sub-tables, by axis position (PRODUCT) or child slot
    cdef tuple _labels              # row labels (row forms), for row_record and errors
    cdef object _first_obj, _last_obj   # index.min()/max() as objects; _no_extreme when none
    cdef object _array              # contiguous ndarray when cells need Python indexing
    cdef HandleKeys _keys           # full-key index (DENSE, SORTED, HASHED)
    cdef HandleKeys _ckeys          # HASHED: first part of each child -> child slot
    cdef HandleKeys* _axes          # PRODUCT: one index per remaining level
    cdef const Py_ssize_t* _spans   # PRODUCT: rows per position of each remaining level
    cdef const unsigned char* _ckind    # per root column: 1 float, 2 int, 3 bool, 4 object
    cdef const double* _dbl         # root arrays, row-major: (base + row) * _stride + _colbase + col
    cdef const long long* _int
    cdef PyObject** _obj
    cdef Py_ssize_t _stride, _colbase
    cdef bint _spec_missing, _int_keys, _increasing, _preserve
    cdef const long long* _okeys    # single-level int keys in row order
    cdef object _table_obj          # table_object(): built with the handle, None for column views
    cdef Py_ssize_t col(self, object name) except -2               # identity, then ==; KeyError(name)
    cdef Py_ssize_t row_1(self, TableKey p0) except -2              # KeyError(original key)
    cdef Py_ssize_t row_2(self, TableKey p0, TableKey p1) except -2
    cdef Py_ssize_t row_3(self, TableKey p0, TableKey p1, TableKey p2) except -2
    cdef Py_ssize_t row_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except -2
    cdef Py_ssize_t row_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3,
                          TableKey p4) except -2
    cdef Py_ssize_t row_6(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3,
                          TableKey p4, TableKey p5) except -2
    cdef bint has_1(self, TableKey p0) except -1
    cdef bint has_2(self, TableKey p0, TableKey p1) except -1
    cdef bint has_3(self, TableKey p0, TableKey p1, TableKey p2) except -1
    cdef bint has_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except -1
    cdef bint has_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4) except -1
    cdef bint has_6(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4,
                    TableKey p5) except -1
    cdef double f64(self, Py_ssize_t row, Py_ssize_t col) except? -1     # float column
    cdef long long i64(self, Py_ssize_t row, Py_ssize_t col) except? -1  # int / bool / enum column
    cdef object obj(self, Py_ssize_t row, Py_ssize_t col)                # str and other columns
    cdef long long first_int(self) except? -1                    # index.min()
    cdef long long last_int(self) except? -1                     # index.max()
    cdef object first_key(self)                                  # non-integer keys
    cdef object last_key(self)
    cdef long long asof_int(self, long long k) except? -1        # KeyError(k) below the first key (D25a)
    cdef Py_ssize_t asof_row(self, long long k) except -2
    cdef Py_ssize_t searchsorted_right(self, long long k) noexcept  # D25b
    cdef long long key_int(self, Py_ssize_t i) except? -1        # sub.index[i]; IndexError, negative i
    cdef TableHandle sub_1(self, TableKey p0)                     # n < levels; shared child, KeyError
    cdef TableHandle sub_2(self, TableKey p0, TableKey p1)
    cdef TableHandle sub_3(self, TableKey p0, TableKey p1, TableKey p2)
    cdef TableHandle sub_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3)
    cdef TableHandle sub_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4)
    cdef object row_record(self, Py_ssize_t row)                 # row local that escapes
    cdef TableHandle column_view(self, Py_ssize_t col)           # D13 column selection T["c"]
    cdef object table_object(self)                               # D45: the bound NativeTable of this handle

@cython.final
cdef class RecordHandle:
    # A Record (model_point()), fields resolved once; f64/i64/obj by slot.
    cdef readonly object owner
    cdef tuple _items               # private (W1.2): the Record's own (name, value) pairs
    cdef Py_ssize_t field(self, object name) except -2
    cdef double f64(self, Py_ssize_t slot) except? -1
    cdef long long i64(self, Py_ssize_t slot) except? -1
    cdef object obj(self, Py_ssize_t slot)

cdef TableHandle table_handle(object value)      # NativeTable.handle or a new handle
cdef RecordHandle record_handle(object value)

# W1.9: borrowed sub-tables and index labels (module functions; nothing is written).
# A TableRef is a borrowed pointer to a shared child handle: the parent handle (held by the
# caller for the whole formula) keeps it alive, so reads through it take no reference;
# use it as <TableHandle>ref. sub_ref_n raises KeyError as sub_n.
ctypedef PyObject* TableRef
cdef TableRef sub_ref_1(TableHandle h, TableKey p0) except NULL
cdef TableRef sub_ref_2(TableHandle h, TableKey p0, TableKey p1) except NULL
cdef TableRef sub_ref_3(TableHandle h, TableKey p0, TableKey p1, TableKey p2) except NULL
cdef TableRef sub_ref_4(TableHandle h, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except NULL
cdef TableRef sub_ref_5(TableHandle h, TableKey p0, TableKey p1, TableKey p2, TableKey p3,
                        TableKey p4) except NULL
cdef object key_label(TableHandle h, Py_ssize_t i)          # h.index[i] in row order: label or tuple
cdef double key_f64(TableHandle h, Py_ssize_t i) except? -1  # float(h.index[i])
# ---- end of the M1 handle interface ------------------------------------------------------

cdef class NativeTable:
    cdef readonly TableHandle handle    # D25 Q8: embedded; None until W1.2 builds it
    cdef object _owner, _index_view, _numeric_view, _indexed_view, _single_label, _input_keys
    cdef object _contiguous_view
    cdef const double* _input_values
    cdef const unsigned char* _input_numeric
    cdef const long long* _input_positions
    cdef long long _input_low, _input_high
    cdef Py_ssize_t _input_width
    cdef double lookup_float(self, KeyArg0 key, object column, object default=*) except? -1
    cdef object previous(self, KeyArg0 key, object column)
    cdef double interpolate_float(self, object key, object column) except? -1
    cpdef object row(self, object key)
    cpdef object slice(self, object key)

cdef NativeTable bind_native_table(object value, str cell)

cdef bint try_clamped_table_float(NativeTable, LookupContext, bint lower, object namespace, long long key,
                                  object column, double* result) except -1
cdef bint read_bounded_table_float(NativeTable table, LookupContext context, object first_fn,
                                   long long key, str column, double below, double* value,
                                   bint owned_index=*) except? -1
cdef double prepare_previous_table_method(LookupContext context, NativeTable table,
                                          long long key, object column) except? -1
cdef double read_previous_table_float(NativeTable table, long long key, object column) except? -1
cdef double read_previous_table_int(NativeTable table, long long key, long long column) except? -1
cdef double record_field_float(object function, LookupContext, object record, object key,
                               object namespace=*) except? -1
cdef double table_field_float(LookupContext, NativeTable, object key, object column) except? -1

cdef class _InputFloat:
    cdef object source, key, column, function
    cdef double value

cdef inline double read_record_field_float(object function, LookupContext context,
                                           object source, object key, object namespace=None) except? -1:
    cdef PyObject* binding = <PyObject*>context.record_float
    if (context._private_owner and binding != <PyObject*>None
            and (<_InputFloat>binding).function is function
            and (<_InputFloat>binding).source is source and (<_InputFloat>binding).key is key):
        return (<_InputFloat>binding).value
    return record_field_float(function, context, source, key, namespace)

cdef inline bint record_field_ready(LookupContext context, const void* source,
                                    const void* key, object namespace) noexcept:
    cdef PyObject* binding = <PyObject*>context.record_float
    return (context._private_owner and context.record_namespace is namespace
            and binding != <PyObject*>None and <void*>(<_InputFloat>binding).source == source
            and <void*>(<_InputFloat>binding).key == key)

cdef inline double bound_record_field(LookupContext context) noexcept:
    return (<_InputFloat><PyObject*>context.record_float).value

cdef object read_record_item(object source, object key)
cdef double read_cast_record_field_float(object conversion, LookupContext context,
                                        object source, object key) except? -1
cdef bint record_cast_ready(LookupContext context, const void* source, const void* key,
                            object conversion) noexcept

cdef inline double read_table_field_float(LookupContext context, NativeTable source,
                                          object key, object column) except? -1:
    cdef PyObject* binding
    if context._private_owner:
        binding = <PyObject*>context.input_first
        if (binding != <PyObject*>None and (<_InputFloat>binding).source is source
                and (<_InputFloat>binding).key is key and (<_InputFloat>binding).column is column):
            return (<_InputFloat>binding).value
        binding = <PyObject*>context.input_second
        if (binding != <PyObject*>None and (<_InputFloat>binding).source is source
                and (<_InputFloat>binding).key is key and (<_InputFloat>binding).column is column):
            return (<_InputFloat>binding).value
        return table_field_float(context, source, key, column)
    return source.lookup_float(<object>key, column)

cdef inline bint read_clamped_table_float(NativeTable table, LookupContext context,
                                          bint lower, object namespace, long long key,
                                          object column, double* result) except -1:
    cdef Py_ssize_t position
    cdef unsigned char mask = 2 if lower else 1
    if (table._input_numeric != NULL and table._input_keys is not None and context._private_owner
            and context.clamped_bindings_ready & mask
            and context.clamped_source is table and context.clamped_column is column
            and context.clamped_namespace is namespace):
        if key < table._input_low:
            if not lower:
                return False
            key = table._input_low
        elif key > table._input_high:
            key = table._input_high
        position = (table._input_positions[key - table._input_low] * table._input_width
                    + context.clamped_index)
        if table._input_numeric[position]:
            result[0] = table._input_values[position]
            return True
    return try_clamped_table_float(table, context, lower, namespace, key, column, result)

cdef double prepare_table_method(LookupContext, NativeTable, long long key, object column,
                                 object default=*) except? -1

cdef object indexed_lookup(object,object,object,object default=*)
cdef object indexed_contains(object,object)

cdef object omitted_argument
cdef double lookup_float(object function, tuple keywords, object source, object key, object column=*, object default=*) except? -1
cdef long long lookup_int(object function, tuple keywords, object source, object key, object column=*, object default=*) except? -1
cdef bint lookup_bool(object function, tuple keywords, object source, object key, object column=*, object default=*) except? -1
cdef str lookup_str(object function, tuple keywords, object source, object key, object column=*, object default=*)

cdef object lookup_object(object function, tuple keywords, object source, object key, object column=*, object default=*)
cdef bint is_missing(object function, tuple keywords, object predicate, object value) except? -1
cdef object isna(object function, object value)


cdef object row(object function, tuple keywords, object table, object key)
cdef object column(object function, tuple keywords, object table, object name, object report=*)
cdef object group_sum(object function, tuple keywords, object table, object labels, object first=*, object last=*)
cdef object table(object function, tuple keywords, object frame, object columns=*, object groups=*, object indexing=*, object rows=*, object report=*, object preserve_scalar_types=*)
cdef object group_record(object function, object tables)
