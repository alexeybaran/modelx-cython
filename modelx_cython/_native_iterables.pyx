# cython: freethreading_compatible=True
"""Generic builtin reduction identities; no model formulas or cached results."""
import builtins
from types import BuiltinFunctionType
from libc.math cimport fabs
from modelx_cython._range_state cimport range_init, range_next


def _original_builtin(function, name):
    # A Python replacement installed before provider initialization must never
    # become a trusted identity merely because we imported it from builtins.
    return (type(function) is BuiltinFunctionType
            and function.__name__ == name and function.__self__ is builtins)


cdef object _all = builtins.all if _original_builtin(builtins.all, 'all') else None
cdef object _any = builtins.any if _original_builtin(builtins.any, 'any') else None
cdef object _min = builtins.min if _original_builtin(builtins.min, 'min') else None
cdef object _max = builtins.max if _original_builtin(builtins.max, 'max') else None


cdef bint all_accepts(object function):
    return _all is not None and function is _all


cdef bint any_accepts(object function):
    return _any is not None and function is _any


cdef bint min_accepts(object function):
    return _min is not None and function is _min


cdef bint max_accepts(object function):
    return _max is not None and function is _max


cdef double f64_abs(double value) noexcept:
    return fabs(value)


cdef double f64_min_pair(double first, double second) noexcept:
    return second if second < first else first


cdef double f64_max_pair(double first, double second) noexcept:
    return second if second > first else first


cdef void i64_range_init(I64Range* state, long long start, long long stop, long long step) except *:
    range_init(state, start, stop, step)


cdef bint i64_range_next(I64Range* state, long long* value) noexcept:
    return range_next(state, value)


cdef long long i64_min_pair(long long first, long long second) noexcept:
    return second if second < first else first


cdef long long i64_max_pair(long long first, long long second) noexcept:
    return second if second > first else first
