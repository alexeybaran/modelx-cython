from libc.stdint cimport uint64_t
from libc.stddef cimport size_t

ctypedef uint64_t word
cdef struct Entry:
    word a
    word b
    word c
    word value
    bint ready

cdef word int_word(long long value) noexcept
cdef word double_word(double value) noexcept
cdef word value_word(double value) noexcept
cdef double word_double(word value) noexcept
cdef long long word_int(word value) noexcept

cdef class NumericCache:
    cdef Entry* entries
    cdef size_t capacity, used
    cdef bint get(self, word a, word b, word c, bint reflexive, word* value) except? -1
    cdef void put(self, word a, word b, word c, bint reflexive, word value) except *
    cpdef reset(self)
