from libc.stdint cimport uint64_t

ctypedef uint64_t word

cdef inline word mix(word value) noexcept:
    value ^= value >> 30
    value *= <word>0xbf58476d1ce4e5b9
    value ^= value >> 27
    value *= <word>0x94d049bb133111eb
    return value ^ (value >> 31)
