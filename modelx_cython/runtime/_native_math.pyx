# cython: freethreading_compatible=True
"""Use the standard C math operation for native floating-point arguments."""
from libc.math cimport erf as c_erf, exp as c_exp, isinf as c_isinf, log as c_log, sqrt as c_sqrt, isfinite
from modelx_helpers.numeric_math import erf as python_erf, exp as python_exp, isinf as python_isinf, log as python_log, sqrt as python_sqrt
import math as _math


cdef double erf(object function, MathArgument value) except? -1.0:
    if MathArgument is double:
        if (function is python_erf or function is _math.erf) and isfinite(value):
            return c_erf(value)
    return function(value)


cdef double exp(object function, MathArgument value) except? -1.0:
    cdef double result
    if MathArgument is double:
        if (function is python_exp or function is _math.exp):
            result = c_exp(value)
            if isfinite(result):
                return result
    return function(value)


cdef bint isinf(object function, MathArgument value) except -1:
    if MathArgument is double:
        if (function is python_isinf or function is _math.isinf):
            return c_isinf(value) != 0
    return function(value)


cdef double log(object function, MathArgument value) except? -1.0:
    if MathArgument is double:
        if (function is python_log or function is _math.log) and value > 0.0:
            return c_log(value)
    return function(value)


cdef double sqrt(object function, MathArgument value) except? -1.0:
    if MathArgument is double:
        if (function is python_sqrt or function is _math.sqrt) and value >= 0.0:
            return c_sqrt(value)
    return function(value)


import sys as _sys
from libc.math cimport isnan as _c_isnan


cdef bint isna(object function, object value) except -1:
    """pandas.isna for a scalar: a float is checked natively, None is missing, else the function."""
    pandas = _sys.modules.get('pandas')
    if pandas is not None and function is pandas.isna:
        if type(value) is float:
            return _c_isnan(<double>value)
        if value is None:
            return True
    return bool(function(value))
