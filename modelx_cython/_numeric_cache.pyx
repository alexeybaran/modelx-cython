# cython: freethreading_compatible=True
"""Exact fixed-signature numeric keys; no formula or policy execution state."""
from libc.stdlib cimport calloc, free
from libc.string cimport memset
cimport cython

cdef union Bits:
    double real
    word integer


cdef word int_word(long long value) noexcept:
    return <word>value


cdef word double_word(double value) noexcept:
    return value_word(value if value != 0.0 else 0.0)


cdef word value_word(double value) noexcept:
    cdef Bits bits
    bits.real = value
    return bits.integer


cdef double word_double(word value) noexcept:
    cdef Bits bits
    bits.integer = value
    return bits.real


cdef long long word_int(word value) noexcept:
    return <long long>value


cdef inline word mix(word value) noexcept:
    value ^= value >> 30
    value *= <word>0xbf58476d1ce4e5b9
    value ^= value >> 27
    value *= <word>0x94d049bb133111eb
    return value ^ (value >> 31)


cdef inline size_t position(Entry* entries, size_t capacity, word a, word b, word c) noexcept:
    cdef size_t index = (mix(a) ^ mix(b + <word>0x9e3779b97f4a7c15) ^ mix(c + <word>0x517cc1b727220a95)) & (capacity - 1)
    while entries[index].ready:
        if entries[index].a == a and entries[index].b == b and entries[index].c == c:
            break
        index = (index + 1) & (capacity - 1)
    return index


cdef class NumericCache:
    def __cinit__(self):
        self.capacity = 8
        self.entries = <Entry*>calloc(self.capacity, sizeof(Entry))
        if self.entries == NULL:
            raise MemoryError()

    def __dealloc__(self):
        free(self.entries)

    cdef bint get(self, word a, word b, word c, bint reflexive, word* value) except? -1:
        cdef size_t index
        cdef bint found
        if not reflexive:
            return False
        with cython.critical_section(self):
            index = position(self.entries, self.capacity, a, b, c)
            found = self.entries[index].ready
            if found:
                value[0] = self.entries[index].value
        return found

    cdef void put(self, word a, word b, word c, bint reflexive, word value) except *:
        cdef Entry* grown
        cdef size_t index, i, capacity
        if not reflexive:
            return
        with cython.critical_section(self):
            if self.used * 2 >= self.capacity:
                capacity = self.capacity * 2
                grown = <Entry*>calloc(capacity, sizeof(Entry))
                if grown == NULL:
                    raise MemoryError()
                for i in range(self.capacity):
                    if self.entries[i].ready:
                        index = position(grown, capacity, self.entries[i].a,
                                         self.entries[i].b, self.entries[i].c)
                        grown[index] = self.entries[i]
                free(self.entries)
                self.entries = grown
                self.capacity = capacity
            index = position(self.entries, self.capacity, a, b, c)
            if not self.entries[index].ready:
                self.used += 1
            self.entries[index].a = a
            self.entries[index].b = b
            self.entries[index].c = c
            self.entries[index].value = value
            self.entries[index].ready = True

    cpdef reset(self):
        with cython.critical_section(self):
            memset(self.entries, 0, self.capacity * sizeof(Entry))
            self.used = 0
