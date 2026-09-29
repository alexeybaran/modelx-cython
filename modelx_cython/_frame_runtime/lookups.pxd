cimport cython

@cython.final
cdef class Curve:
    cdef const double[::1] _x, _y
    cdef Py_ssize_t _n
    cdef bint _dense
    cdef double _start
    cdef Py_ssize_t _right(self, double x) noexcept nogil
    cdef Py_ssize_t _index(self, double x) noexcept nogil
    cpdef double at(self, double x) except? -1 nogil
    cpdef double get(self, double x, double default=*) except? -1 nogil
    cpdef double previous(self, double x, double below=*) except? -1 nogil
    cpdef double linear(self, double x) except? -1 nogil

cpdef double power(double base, double exponent) except? -1 nogil

@cython.final
cdef class CurveTable:
    cdef object _labels, _codes
    cdef const long long[::1] _groups, _offsets
    cdef const double[::1] _x, _y
    cdef const unsigned char[::1] _dense
    cdef Py_ssize_t _ng
    cdef bint _group_dense
    cdef Py_ssize_t _group_index(self, long long group) noexcept nogil
    cdef Py_ssize_t _right(self, Py_ssize_t gi, double x) noexcept nogil
    cpdef double at(self, long long group, double x) except? -1 nogil
    cpdef double get(self, long long group, double x, double default=*) except? -1 nogil
    cpdef double previous(self, long long group, double x, double below=*) except? -1 nogil
    cpdef double linear(self, long long group, double x) except? -1 nogil
