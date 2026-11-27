# W1.4: build-time resolved members of a module Reference (namespace identity, no
# per-call attribute read); see _native_math.pyx.
ctypedef fused MathNumber:
    double
    long long

ctypedef fused MathValue:
    double
    long long
    object

ctypedef fused MathValue2:
    double
    long long
    object

cdef object math_module
# diag-mt2: ``namespace`` is the module Reference as a BORROWED pointer: the generated call
# passes ``cast(p_void, self.<ref>)`` (the Space keeps the Reference alive), so a per-step
# call takes no reference on the module every worker shares (an atomic increment and decrement
# on one object per call at four workers). Identity checks compare pointers; the Python
# fallback for another namespace turns it into an object there.
cdef double m_erf(void* namespace, MathValue value) except? -1.0
cdef double m_exp(void* namespace, MathValue value) except? -1.0
cdef bint m_isinf(void* namespace, MathValue value) except -1
cdef double m_log(void* namespace, MathValue value) except? -1.0
cdef double m_sqrt(void* namespace, MathValue value) except? -1.0
cdef long long m_floor_ll(void* namespace, MathNumber value) except? -1
# W1.8: math.isclose; ``given`` bit 1: rel_tol written, bit 2: abs_tol written
cdef bint m_isclose(void* namespace, MathValue a, MathValue2 b, double rel_tol, double abs_tol,
                    int given) except -1
cdef long long m_ceil_ll(void* namespace, MathNumber value) except? -1
cdef object m_floor(void* namespace, object value)
cdef object m_ceil(void* namespace, object value)
cdef object m_constant(void* namespace, Py_ssize_t index)
