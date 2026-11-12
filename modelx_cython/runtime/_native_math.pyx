# cython: freethreading_compatible=True
"""Use the standard C math operation for native floating-point arguments."""
from libc.math cimport erf as c_erf, exp as c_exp, isinf as c_isinf, log as c_log, sqrt as c_sqrt, isfinite
from math import erf as builtin_erf, exp as builtin_exp, isinf as builtin_isinf, log as builtin_log, sqrt as builtin_sqrt

try:
    from modelx_helpers.numeric_math import erf as python_erf, exp as python_exp, isinf as python_isinf, log as python_log, sqrt as python_sqrt
except ModuleNotFoundError as error:
    if error.name not in ('modelx_helpers', 'modelx_helpers.numeric_math'):
        raise
    python_erf, python_exp, python_isinf, python_log, python_sqrt = (
        builtin_erf, builtin_exp, builtin_isinf, builtin_log, builtin_sqrt)


cdef double erf(object function, MathArgument value) except? -1.0:
    if MathArgument is double:
        if (function is builtin_erf or function is python_erf) and isfinite(value):
            return c_erf(value)
    return function(value)


cdef double exp(object function, MathArgument value) except? -1.0:
    cdef double result
    if MathArgument is double:
        if function is builtin_exp or function is python_exp:
            result = c_exp(value)
            if isfinite(result):
                return result
    return function(value)


cdef bint isinf(object function, MathArgument value) except -1:
    if MathArgument is double:
        if function is builtin_isinf or function is python_isinf:
            return c_isinf(value) != 0
    return function(value)


cdef double log(object function, MathArgument value) except? -1.0:
    if MathArgument is double:
        if (function is builtin_log or function is python_log) and value > 0.0:
            return c_log(value)
    return function(value)


cdef double sqrt(object function, MathArgument value) except? -1.0:
    if MathArgument is double:
        if (function is builtin_sqrt or function is python_sqrt) and value >= 0.0:
            return c_sqrt(value)
    return function(value)
