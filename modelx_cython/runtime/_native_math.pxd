ctypedef fused MathArgument:
    object
    double

cdef double erf(object function, MathArgument value) except? -1.0
cdef double exp(object function, MathArgument value) except? -1.0
cdef bint isinf(object function, MathArgument value) except -1
cdef double log(object function, MathArgument value) except? -1.0
cdef double sqrt(object function, MathArgument value) except? -1.0
