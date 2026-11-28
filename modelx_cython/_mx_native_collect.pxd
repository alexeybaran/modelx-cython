cdef union NumericValue:
    double floating
    long long integer


cdef struct Field:
    Py_ssize_t offset
    Py_ssize_t width
    long long start
    long long* keys
    int kind


cdef class NativeCollect:
    cdef tuple sections, coordinates, schemas
    cdef object native_result, positions
    cdef dict large_sums
    cdef list object_rows
    cdef Field* individual_fields
    cdef Field* sum_fields
    cdef NumericValue* rows
    cdef NumericValue* sums
    cdef unsigned char* wide
    cdef unsigned char* active
    cdef Py_ssize_t row_width, sum_width, object_width, worker_count
    cdef Py_ssize_t individual_count, sum_count
    cdef bint dense_coordinates
    cdef long long first_coordinate
    cpdef object keys(self, bint individual, Py_ssize_t field)
    cdef void write(self, bint individual, Py_ssize_t row, Py_ssize_t worker,
                    Py_ssize_t field, Py_ssize_t index, double value) noexcept nogil
    cdef void write_int(self, bint individual, Py_ssize_t row, Py_ssize_t worker,
                        Py_ssize_t field, Py_ssize_t index, long long value) except *
    cdef void add_int(self, Py_ssize_t offset, long long value) except *
    cdef void write_object(self, Py_ssize_t row, Py_ssize_t field, object value) except *
    cdef void finish_row(self, Py_ssize_t worker) noexcept nogil
    cdef object box(self, bint individual, Py_ssize_t row, int kind, Py_ssize_t offset)
    cdef object field_as_python(self, bint individual, Py_ssize_t row, Py_ssize_t field)
    cdef object record(self, object views, Py_ssize_t row)
    cdef Py_ssize_t coordinate_position(self, object coordinate) except -1


cdef class NativeRow:
    cdef NativeCollect owner
    cdef Py_ssize_t row
    cdef bint individual


cpdef object integer_keys(object value)
cdef object individual_at(object result, object coordinate)
cdef double read_individual(object record, Py_ssize_t field, str name, long long key,
                            bint series, bint index_read) except? -1
cdef object read_individual_value(object record, Py_ssize_t field, str name, long long key,
                                  bint series, bint index_read)
cdef double read_sum(object result, Py_ssize_t field, str name, long long key,
                     bint series, bint index_read) except? -1
cdef long long read_individual_int(object record, Py_ssize_t field, str name, long long key,
                                   bint series, bint index_read) except? -1
cdef long long read_sum_int(object result, Py_ssize_t field, str name, long long key,
                            bint series, bint index_read) except? -1
