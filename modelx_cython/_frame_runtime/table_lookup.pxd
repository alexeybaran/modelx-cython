cimport cython
from .lookups cimport CurveTable, Curve

@cython.final
cdef class IndexedTable:
    cdef tuple columns, maxima, names, labels
    cdef object maximum
    cdef dict rows, positions, groups, ordered, native, codes, curves
    cdef int numeric_axis
    cdef Py_ssize_t levels
    cdef object _full(self, object group, object key)
    cpdef double at(self, object key, object column) except? -1
    cpdef double get(self, object key, object column, double default=*) except? -1
    cpdef double previous(self, object key, object column, double below=*) except? -1
    cpdef double previous_group(self, object group, object key, object column, double below=*) except? -1
    cpdef object max_key(self, object group=*)
    cpdef object max_level(self, object level)
    cpdef object keys(self, object group=*)
    cpdef Curve group_curve(self, object group, object column, object axis=*)
