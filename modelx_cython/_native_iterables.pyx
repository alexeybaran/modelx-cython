# cython: freethreading_compatible=True
"""Generic builtin reduction identities; no model formulas or cached results."""
import builtins
from types import BuiltinFunctionType
from libc.math cimport fabs


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
    if step == 0:
        raise ValueError('range() arg 3 must not be zero')
    state.current = start
    state.stop = stop
    state.step = step
    state.active = start < stop if step > 0 else start > stop


cdef bint i64_range_next(I64Range* state, long long* value) noexcept:
    cdef unsigned long long distance, magnitude
    if not state.active:
        return False
    value[0] = state.current
    if state.step > 0:
        distance = <unsigned long long>state.stop - <unsigned long long>state.current
        magnitude = <unsigned long long>state.step
    else:
        distance = <unsigned long long>state.current - <unsigned long long>state.stop
        magnitude = <unsigned long long>0 - <unsigned long long>state.step
    if distance <= magnitude:
        state.active = False
    else:
        # The distance proof guarantees the next value lies strictly inside the
        # signed-int64 endpoints; the terminating increment is never computed.
        state.current += state.step
    return True


cdef long long i64_min_pair(long long first, long long second) noexcept:
    return second if second < first else first


cdef long long i64_max_pair(long long first, long long second) noexcept:
    return second if second > first else first
