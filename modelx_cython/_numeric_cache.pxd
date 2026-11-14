from libc.stdint cimport uint64_t
from libc.stddef cimport size_t

ctypedef uint64_t word
cdef struct Entry:
    word a
    word b
    word c
    word value
    bint ready

cdef struct PrimaryEntry:
    word a
    word b
    word c
    word value
    unsigned char state

cdef union Bits:
    double real
    word integer


# Inline in the pxd so compiled model modules inline the key and value conversions (D74).
cdef inline word int_word(long long value) noexcept:
    return <word>value


cdef inline word value_word(double value) noexcept:
    cdef Bits bits
    bits.real = value
    return bits.integer


cdef inline word double_word(double value) noexcept:
    return value_word(value if value != 0.0 else 0.0)


cdef inline double word_double(word value) noexcept:
    cdef Bits bits
    bits.integer = value
    return bits.real


cdef inline long long word_int(word value) noexcept:
    return <long long>value

cdef class NumericCache:
    cdef Entry* entries
    cdef size_t capacity, used
    cdef PrimaryEntry* primary
    cdef size_t primary_capacity, primary_used
    # D74: an optional dense window over integer keys (one private receiver, e.g. a reused collect root)
    cdef word* dense_values
    cdef unsigned char* dense_state
    cdef unsigned int* dense_written
    cdef size_t dense_count, dense_written_n
    cdef long long dense_lower[3]
    cdef unsigned long long dense_size[3]
    cpdef int window(self, long long lower0, long long size0, long long lower1=*, long long size1=*,
                     long long lower2=*, long long size2=*) except -1
    cpdef size_t window_bytes(self)
    cdef bint get(self, word a, word b, word c, bint reflexive, word* value) except? -1
    cdef void put(self, word a, word b, word c, bint reflexive, word value) except *
    cdef bint reserve(self, word a, word b, word c, bint reflexive, Py_ssize_t* slot,
                      word* value) noexcept
    cdef void publish(self, word a, word b, word c, Py_ssize_t slot, word value) except *
    cpdef reset(self)


# D74: the dense window, inline in compiled model modules. dense_find returns the flat index of a key
# inside the window or -1 (no window, or outside it); dense_ready/dense_value read it; dense_put stores
# a value and records a first write for the compact reset.
cdef inline Py_ssize_t dense_find(NumericCache cache, word a, word b, word c) noexcept:
    cdef unsigned long long i0, i1, i2
    if cache.dense_values == NULL:
        return -1
    i0 = <unsigned long long>(<long long>a - cache.dense_lower[0])
    i1 = <unsigned long long>(<long long>b - cache.dense_lower[1])
    i2 = <unsigned long long>(<long long>c - cache.dense_lower[2])
    if i0 >= cache.dense_size[0] or i1 >= cache.dense_size[1] or i2 >= cache.dense_size[2]:
        return -1
    return <Py_ssize_t>((i0 * cache.dense_size[1] + i1) * cache.dense_size[2] + i2)


cdef inline bint dense_ready(NumericCache cache, Py_ssize_t index) noexcept:
    return cache.dense_state[index] != 0


cdef inline word dense_value(NumericCache cache, Py_ssize_t index) noexcept:
    return cache.dense_values[index]


cdef inline void dense_put(NumericCache cache, Py_ssize_t index, word value) noexcept:
    cache.dense_values[index] = value
    if not cache.dense_state[index]:
        cache.dense_state[index] = 1
        cache.dense_written[cache.dense_written_n] = <unsigned int>index
        cache.dense_written_n += 1
