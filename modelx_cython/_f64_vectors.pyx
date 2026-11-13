# cython: language_level=3, boundscheck=False, wraparound=False
"""Float vectors for vector cells (W0.11); see _f64_vectors.pxd."""
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memset


cdef class F64Vec:
    """A read-only sequence of floats backed by a C double buffer."""

    def __dealloc__(self):
        PyMem_Free(self.data)

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        cdef Py_ssize_t i
        if isinstance(key, slice):
            return [self.data[i] for i in range(*key.indices(self.n))]
        return _vec_get_slow(self, key)

    def __iter__(self):
        cdef Py_ssize_t i
        for i in range(self.n):
            yield self.data[i]

    def __reversed__(self):
        cdef Py_ssize_t i
        for i in range(self.n - 1, -1, -1):
            yield self.data[i]

    def __eq__(self, other):
        try:
            return len(other) == self.n and all(a == b for a, b in zip(self, other))
        except TypeError:
            return NotImplemented

    def __repr__(self):
        return 'F64Vec(%r)' % (list(self),)

    def __reduce__(self):
        return (_vec_from_sequence, (list(self),))


cdef F64Vec vec_new(Py_ssize_t n):
    cdef F64Vec v = F64Vec.__new__(F64Vec)
    if n < 0:
        n = 0
    v.data = <double*>PyMem_Malloc((n if n > 0 else 1) * sizeof(double))
    if v.data == NULL:
        raise MemoryError()
    v.n = n
    return v


cdef F64Vec vec_full(Py_ssize_t n, double value):
    cdef F64Vec v = vec_new(n)
    cdef Py_ssize_t i
    for i in range(v.n):
        v.data[i] = value
    return v


cdef F64Vec vec_of(object value):
    """The value itself if it is an F64Vec, else a copy of a sequence of numbers."""
    cdef F64Vec v
    cdef Py_ssize_t i
    if type(value) is F64Vec:
        return <F64Vec>value
    if value is None:
        raise TypeError("a vector cell or local holds None, not a sequence of floats")
    items = list(value)
    v = vec_new(len(items))
    for i in range(v.n):
        v.data[i] = items[i]
    return v


cdef double _vec_get_slow(F64Vec v, long long i) except? -1.0:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        raise IndexError("list index out of range")
    return v.data[i]


cdef int _vec_set_slow(F64Vec v, long long i, double value) except -1:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        raise IndexError("list assignment index out of range")
    v.data[i] = value
    return 0


def _vec_from_sequence(items):
    """Unpickling: an F64Vec holding the given floats."""
    return vec_of(items)
