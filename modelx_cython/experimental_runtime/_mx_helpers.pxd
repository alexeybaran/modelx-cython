cdef class FieldTable:
    cdef dict _values
    cpdef object at(self, object key)
    cpdef object category(self, object key, object enum_type)

cdef class GroupKeys:
    cdef dict _labels
    cpdef object key(self, object value)
