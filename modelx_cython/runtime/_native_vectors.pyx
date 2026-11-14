# cython: freethreading_compatible=True, boundscheck=False, wraparound=False
"""Typed vectors of vector cells; see _native_vectors.pxd."""
from cpython.buffer cimport PyBUF_WRITABLE, PyBUF_FORMAT, PyBUF_ND, PyBUF_STRIDES
from cpython.float cimport PyFloat_AS_DOUBLE
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free, PyMem_Calloc
from cpython.ref cimport Py_INCREF, Py_XDECREF
from cpython.object cimport Py_EQ, Py_NE, Py_LT, Py_LE, Py_GT, Py_GE
import operator

cdef extern from "Python.h":
    PyObject* PyObject_RichCompare(object a, object b, int op) except NULL
    Py_ssize_t PyList_GET_SIZE(object)
    PyObject* PyList_GET_ITEM(object, Py_ssize_t)
    Py_ssize_t PyTuple_GET_SIZE(object)
    PyObject* PyTuple_GET_ITEM(object, Py_ssize_t)

cdef char* _F64_FORMAT = b'd'
cdef char* _I64_FORMAT = b'q'
cdef double _TWO53 = 9007199254740992.0


cdef inline bint _is_sequence(object other):
    return type(other) is list or type(other) is tuple or type(other) is F64Vec or type(other) is I64Vec


cdef object _richcmp_items(object items, object other, int op):
    """Comparison of a vector as the list of its items (list semantics)."""
    if op == Py_EQ or op == Py_NE:
        if not _is_sequence(other):
            return NotImplemented
        equal = len(items) == len(other) and all(a == b for a, b in zip(items, other))
        return equal if op == Py_EQ else not equal
    if not _is_sequence(other):
        return NotImplemented
    mine, theirs = list(items), list(other)
    if op == Py_LT:
        return mine < theirs
    if op == Py_LE:
        return mine <= theirs
    if op == Py_GT:
        return mine > theirs
    return mine >= theirs


cdef inline Py_ssize_t _index(object key, Py_ssize_t n) except -1:
    cdef Py_ssize_t i = operator.index(key)
    if i < 0:
        i += n
    if i < 0 or i >= n:
        raise IndexError("list index out of range")
    return i


# ---------------------------------------------------------------- F64Vec
cdef class F64Vec:
    """A read-only sequence of floats backed by a C double buffer.

    Equal to a list or tuple holding the same floats; immutable once the formula
    that built it has returned, so threads read it without a lock.
    """

    __hash__ = None

    def __dealloc__(self):
        PyMem_Free(self.data)

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        cdef Py_ssize_t i
        if isinstance(key, slice):
            return [self.data[i] for i in range(*key.indices(self.n))]
        return self.data[_index(key, self.n)]

    def __setitem__(self, key, value):
        """Item assignment from Python code, as on the list the live model caches (the length
        never changes, so readers stay lock-free). Formulas cannot write a vector they did not
        allocate (build error)."""
        cdef Py_ssize_t i
        if isinstance(key, slice):
            indices = range(*key.indices(self.n))
            items = list(value)
            if len(items) != len(indices):
                raise ValueError('a vector cannot change its length')
            for i, item in zip(indices, items):
                self.data[i] = _exact_float(item)
            return
        self.data[_index(key, self.n)] = _exact_float(value)

    def __iter__(self):
        return _F64Iter(self, False)

    def __reversed__(self):
        return _F64Iter(self, True)

    def __contains__(self, item):
        cdef Py_ssize_t i
        for i in range(self.n):
            if self.data[i] == item:
                return True
        return False

    def index(self, item):
        cdef Py_ssize_t i
        for i in range(self.n):
            if self.data[i] == item:
                return i
        raise ValueError(f'{item!r} is not in list')

    def count(self, item):
        cdef Py_ssize_t i, k = 0
        for i in range(self.n):
            if self.data[i] == item:
                k += 1
        return k

    def tolist(self):
        cdef Py_ssize_t i
        return [self.data[i] for i in range(self.n)]

    def __richcmp__(self, other, int op):
        cdef Py_ssize_t i
        cdef F64Vec w
        cdef double x
        if op == Py_EQ or op == Py_NE:
            if type(other) is F64Vec:
                w = <F64Vec>other
                equal = w.n == self.n
                if equal:
                    for i in range(self.n):
                        if not (self.data[i] == w.data[i]):
                            equal = False
                            break
                return equal if op == Py_EQ else not equal
            if type(other) is list or type(other) is tuple:
                if len(other) != self.n:
                    return op == Py_NE
                for i in range(self.n):
                    item = other[i]
                    if type(item) is float:
                        x = PyFloat_AS_DOUBLE(item)
                        if not (self.data[i] == x):
                            return op == Py_NE
                    elif not (self.data[i] == item):
                        return op == Py_NE
                return op == Py_EQ
        return _richcmp_items(self, other, op)

    def __add__(self, other):
        return list(self) + list(other) if _is_sequence(other) else NotImplemented

    def __radd__(self, other):
        return list(other) + list(self) if _is_sequence(other) else NotImplemented

    def __mul__(self, count):
        return list(self) * count

    def __rmul__(self, count):
        return list(self) * count

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return 'F64Vec(%r)' % (self.tolist(),)

    def __reduce__(self):
        return (_f64_from_sequence, (self.tolist(),))

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        if flags & PyBUF_WRITABLE:
            raise BufferError('F64Vec is read-only')
        self._itemsize = sizeof(double)
        buffer.buf = <void*>self.data
        buffer.obj = self
        buffer.len = self.n * sizeof(double)
        buffer.readonly = 1
        buffer.itemsize = sizeof(double)
        buffer.format = _F64_FORMAT if flags & PyBUF_FORMAT else NULL
        buffer.ndim = 1
        buffer.shape = &self.n if flags & PyBUF_ND else NULL
        buffer.strides = &self._itemsize if flags & PyBUF_STRIDES == PyBUF_STRIDES else NULL
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __releasebuffer__(self, Py_buffer* buffer):
        pass


cdef class _F64Iter:
    cdef F64Vec v
    cdef Py_ssize_t i
    cdef bint backwards

    def __cinit__(self, F64Vec v, bint backwards):
        self.v = v
        self.backwards = backwards
        self.i = v.n - 1 if backwards else 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef Py_ssize_t i = self.i
        if i < 0 or i >= self.v.n:
            raise StopIteration
        self.i = i - 1 if self.backwards else i + 1
        return self.v.data[i]


cdef F64Vec vec_new(Py_ssize_t n):
    cdef F64Vec v = F64Vec.__new__(F64Vec)
    if n < 0:
        n = 0
    v.data = <double*>PyMem_Malloc((n if n > 0 else 1) * sizeof(double))
    if v.data == NULL:
        raise MemoryError()
    v.n = n
    v.cap = n if n > 0 else 1
    return v


cdef F64Vec vec_full(Py_ssize_t n, double value):
    cdef F64Vec v = vec_new(n)
    cdef Py_ssize_t i
    for i in range(v.n):
        v.data[i] = value
    return v


cdef int _vec_grow(F64Vec v) except -1:
    cdef Py_ssize_t cap = v.cap * 2 if v.cap > 0 else 8
    cdef double* data = <double*>PyMem_Realloc(v.data, cap * sizeof(double))
    if data == NULL:
        raise MemoryError()
    v.data = data
    v.cap = cap
    return 0


cdef inline double _exact_float(object item) except? -1.0:
    """A float item, or an int that a double holds exactly (a list built from ints
    on a path the trace did not see); anything else is a TypeError."""
    cdef double x
    if type(item) is float:
        return PyFloat_AS_DOUBLE(item)
    if isinstance(item, float):
        return <double>item
    if isinstance(item, int) and not isinstance(item, bool):
        x = <double>item
        if -_TWO53 <= x <= _TWO53 and int(x) == item:
            return x
        raise TypeError(f'a float vector cannot hold the integer {item!r} exactly')
    raise TypeError(f'a float vector holds floats, not {type(item).__name__}')


cdef F64Vec vec_of(object value):
    """The value itself if it is an F64Vec, else a copy of a sequence of floats."""
    cdef F64Vec v
    cdef Py_ssize_t i, n
    if type(value) is F64Vec:
        return <F64Vec>value
    if value is None:
        raise TypeError("a float vector cell or local holds None, not a list of floats")
    if type(value) is I64Vec:
        value = list(value)
    if type(value) is list:
        n = len(value)
        v = vec_new(n)
        for i in range(n):
            v.data[i] = _exact_float(<object>PyList_GET_ITEM(value, i))
        return v
    if type(value) is not tuple:
        value = tuple(value)
    n = len(value)
    v = vec_new(n)
    for i in range(n):
        v.data[i] = _exact_float(<object>PyTuple_GET_ITEM(value, i))
    return v


cdef F64Vec vec_slice(F64Vec v, long long a, long long b):
    """v[a:b] as a new vector (Python slice normalisation, step 1)."""
    cdef F64Vec r
    cdef Py_ssize_t i
    _clip(&a, &b, v.n)
    r = vec_new(b - a if b > a else 0)
    for i in range(r.n):
        r.data[i] = v.data[a + i]
    return r


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


def _f64_from_sequence(items):
    """Unpickling: an F64Vec holding the given floats."""
    return vec_of(items)


# ---------------------------------------------------------------- I64Vec
cdef class I64Vec:
    """A read-only sequence of integers (int64) backed by a C buffer."""

    __hash__ = None

    def __dealloc__(self):
        PyMem_Free(self.data)

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        cdef Py_ssize_t i
        if isinstance(key, slice):
            return [self.data[i] for i in range(*key.indices(self.n))]
        return self.data[_index(key, self.n)]

    def __setitem__(self, key, value):
        """Item assignment from Python code (see F64Vec.__setitem__)."""
        cdef Py_ssize_t i
        if isinstance(key, slice):
            indices = range(*key.indices(self.n))
            items = list(value)
            if len(items) != len(indices):
                raise ValueError('a vector cannot change its length')
            for i, item in zip(indices, items):
                self.data[i] = _exact_int(item)
            return
        self.data[_index(key, self.n)] = _exact_int(value)

    def __iter__(self):
        return _I64Iter(self, False)

    def __reversed__(self):
        return _I64Iter(self, True)

    def __contains__(self, item):
        return any(x == item for x in self)

    def index(self, item):
        for i, x in enumerate(self):
            if x == item:
                return i
        raise ValueError(f'{item!r} is not in list')

    def count(self, item):
        return sum(1 for x in self if x == item)

    def tolist(self):
        cdef Py_ssize_t i
        return [self.data[i] for i in range(self.n)]

    def __richcmp__(self, other, int op):
        cdef Py_ssize_t i
        cdef I64Vec w
        if op == Py_EQ or op == Py_NE:
            if type(other) is I64Vec:
                w = <I64Vec>other
                equal = w.n == self.n
                if equal:
                    for i in range(self.n):
                        if self.data[i] != w.data[i]:
                            equal = False
                            break
                return equal if op == Py_EQ else not equal
            if type(other) is list or type(other) is tuple:
                if len(other) != self.n:
                    return op == Py_NE
                for i in range(self.n):
                    item = other[i]
                    if type(item) is int:
                        if not (item == self.data[i]):
                            return op == Py_NE
                    elif not (self.data[i] == item):
                        return op == Py_NE
                return op == Py_EQ
        return _richcmp_items(self, other, op)

    def __add__(self, other):
        return list(self) + list(other) if _is_sequence(other) else NotImplemented

    def __radd__(self, other):
        return list(other) + list(self) if _is_sequence(other) else NotImplemented

    def __mul__(self, count):
        return list(self) * count

    def __rmul__(self, count):
        return list(self) * count

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return 'I64Vec(%r)' % (self.tolist(),)

    def __reduce__(self):
        return (_i64_from_sequence, (self.tolist(),))

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        if flags & PyBUF_WRITABLE:
            raise BufferError('I64Vec is read-only')
        self._itemsize = sizeof(long long)
        buffer.buf = <void*>self.data
        buffer.obj = self
        buffer.len = self.n * sizeof(long long)
        buffer.readonly = 1
        buffer.itemsize = sizeof(long long)
        buffer.format = _I64_FORMAT if flags & PyBUF_FORMAT else NULL
        buffer.ndim = 1
        buffer.shape = &self.n if flags & PyBUF_ND else NULL
        buffer.strides = &self._itemsize if flags & PyBUF_STRIDES == PyBUF_STRIDES else NULL
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __releasebuffer__(self, Py_buffer* buffer):
        pass


cdef class _I64Iter:
    cdef I64Vec v
    cdef Py_ssize_t i
    cdef bint backwards

    def __cinit__(self, I64Vec v, bint backwards):
        self.v = v
        self.backwards = backwards
        self.i = v.n - 1 if backwards else 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef Py_ssize_t i = self.i
        if i < 0 or i >= self.v.n:
            raise StopIteration
        self.i = i - 1 if self.backwards else i + 1
        return self.v.data[i]


cdef I64Vec ivec_new(Py_ssize_t n):
    cdef I64Vec v = I64Vec.__new__(I64Vec)
    if n < 0:
        n = 0
    v.data = <long long*>PyMem_Malloc((n if n > 0 else 1) * sizeof(long long))
    if v.data == NULL:
        raise MemoryError()
    v.n = n
    v.cap = n if n > 0 else 1
    return v


cdef I64Vec ivec_full(Py_ssize_t n, long long value):
    cdef I64Vec v = ivec_new(n)
    cdef Py_ssize_t i
    for i in range(v.n):
        v.data[i] = value
    return v


cdef int _ivec_grow(I64Vec v) except -1:
    cdef Py_ssize_t cap = v.cap * 2 if v.cap > 0 else 8
    cdef long long* data = <long long*>PyMem_Realloc(v.data, cap * sizeof(long long))
    if data == NULL:
        raise MemoryError()
    v.data = data
    v.cap = cap
    return 0


cdef inline long long _exact_int(object item) except? -1:
    if type(item) is int:
        return item
    if isinstance(item, int) and not isinstance(item, bool):
        return operator.index(item)
    raise TypeError(f'an integer vector holds integers, not {type(item).__name__}')


cdef I64Vec ivec_of(object value):
    """The value itself if it is an I64Vec, else a copy of a sequence of integers."""
    cdef I64Vec v
    cdef Py_ssize_t i, n
    if type(value) is I64Vec:
        return <I64Vec>value
    if value is None:
        raise TypeError("an integer vector cell or local holds None, not a list of integers")
    if type(value) is not list and type(value) is not tuple:
        value = list(value)
    n = len(value)
    v = ivec_new(n)
    for i in range(n):
        v.data[i] = _exact_int(value[i])
    return v


cdef I64Vec ivec_slice(I64Vec v, long long a, long long b):
    cdef I64Vec r
    cdef Py_ssize_t i
    _clip(&a, &b, v.n)
    r = ivec_new(b - a if b > a else 0)
    for i in range(r.n):
        r.data[i] = v.data[a + i]
    return r


cdef long long _ivec_sum_slow(I64Vec v, Py_ssize_t a, Py_ssize_t b) except? -1:
    total = sum(v.data[i] for i in range(a, b))
    return total     # OverflowError when it does not fit int64


cdef long long _ivec_get_slow(I64Vec v, long long i) except? -1:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        raise IndexError("list index out of range")
    return v.data[i]


cdef int _ivec_set_slow(I64Vec v, long long i, long long value) except -1:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        raise IndexError("list assignment index out of range")
    v.data[i] = value
    return 0


def _i64_from_sequence(items):
    """Unpickling: an I64Vec holding the given integers."""
    return ivec_of(items)


# ---------------------------------------------------------------- dense cache of vector cells
cdef class VecCache:

    def __cinit__(self, Py_ssize_t n, long long lo):
        self.slots = <PyObject**>PyMem_Calloc(n if n > 0 else 1, sizeof(PyObject*))
        if self.slots == NULL:
            raise MemoryError()
        self.n = n
        self.lo = lo

    def __dealloc__(self):
        cdef Py_ssize_t i
        if self.slots != NULL:
            for i in range(self.n):
                Py_XDECREF(self.slots[i])
            PyMem_Free(self.slots)

    cdef object put(self, long long key, object value):
        """Publish value at key unless another thread did first; returns the published one."""
        cdef unsigned long long i = <unsigned long long>(key - self.lo)
        cdef PyObject* p
        if i >= <unsigned long long>self.n:
            return value        # outside the sampled domain: not cached
        Py_INCREF(value)
        if __pyx_vec_publish(&self.slots[i], <PyObject*>value):
            return value
        Py_XDECREF(<PyObject*>value)
        p = __pyx_vec_load(&self.slots[i])
        return <object>p

    def __len__(self):
        cdef Py_ssize_t i, k = 0
        for i in range(self.n):
            if __pyx_vec_load(&self.slots[i]) != NULL:
                k += 1
        return k

    def items(self):
        """(key, value) of the filled slots."""
        cdef Py_ssize_t i
        cdef PyObject* p
        out = []
        for i in range(self.n):
            p = __pyx_vec_load(&self.slots[i])
            if p != NULL:
                out.append((self.lo + i, <object>p))
        return out

    def __reduce__(self):
        return (_vec_cache_from_items, (self.n, self.lo, self.items()))


def _vec_cache_from_items(n, lo, items):
    """Unpickling: a VecCache holding the given (key, value) pairs."""
    cdef VecCache cache = VecCache(n, lo)
    for key, value in items:
        cache.put(key, value)
    return cache
