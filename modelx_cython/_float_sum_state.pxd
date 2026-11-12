"""Header-only float-start sum state; no input-provider runtime dependency.

Contributions must already be exact double values. Python 3.12 introduced
compensated float summation; older interpreters use ordered simple addition.
This is a numeric reducer, not an arbitrary Python sum state machine.
"""
from cpython.version cimport PY_VERSION_HEX
from libc.math cimport fabs, isfinite

cdef struct F64Sum:
    double total
    double correction


cdef inline void f64_init(F64Sum* state) noexcept:
    state.total = 0.0
    state.correction = 0.0


cdef inline void f64_add(F64Sum* state, double value) noexcept:
    cdef double following = state.total + value
    if PY_VERSION_HEX >= 0x030C0000:
        if fabs(state.total) >= fabs(value):
            state.correction += (state.total - following) + value
        else:
            state.correction += (value - following) + state.total
    state.total = following


cdef inline double f64_result(F64Sum* state) noexcept:
    if state.correction != 0.0 and isfinite(state.correction):
        return state.total + state.correction
    return state.total
