ctypedef fused MathArgument:
    object
    double

cdef double erf(object function, MathArgument value) except? -1.0
cdef double exp(object function, MathArgument value) except? -1.0
cdef bint isinf(object function, MathArgument value) except -1
cdef double log(object function, MathArgument value) except? -1.0
cdef double sqrt(object function, MathArgument value) except? -1.0

# W1.4: build-time resolved members of a module Reference (namespace identity, no
# per-call attribute read); see _native_math.pyx.
ctypedef fused MathNumber:
    double
    long long

ctypedef fused MathValue:
    double
    long long
    object

cdef object math_module
cdef object numeric_math_module
cdef double m_erf(object namespace, MathValue value) except? -1.0
cdef double m_exp(object namespace, MathValue value) except? -1.0
cdef bint m_isinf(object namespace, MathValue value) except -1
cdef double m_log(object namespace, MathValue value) except? -1.0
cdef double m_sqrt(object namespace, MathValue value) except? -1.0
cdef long long m_floor_ll(object namespace, MathNumber value) except? -1
cdef long long m_ceil_ll(object namespace, MathNumber value) except? -1
cdef object m_floor(object namespace, object value)
cdef object m_ceil(object namespace, object value)
cdef object m_constant(object namespace, Py_ssize_t index)

# W1.4: min/max of typed operands as Python evaluates them: the first operand is kept
# unless a later one compares strictly smaller (min) or larger (max); nan and -0.0
# follow from the same comparison. Nested left to right for n arguments.
cdef inline double min_f64(double a, double b) noexcept:
    return b if b < a else a

cdef inline double max_f64(double a, double b) noexcept:
    return b if b > a else a

cdef inline long long min_i64(long long a, long long b) noexcept:
    return b if b < a else a

cdef inline long long max_i64(long long a, long long b) noexcept:
    return b if b > a else a
