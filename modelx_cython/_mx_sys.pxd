from libc.math cimport (
    pow as _mx_c_pow, floor as _mx_c_floor, isfinite as _mx_c_isfinite,
    ceil as _mx_c_ceil, fabs, copysign, fma)
from libc.limits cimport LLONG_MIN
from libc.string cimport memcmp
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_CheckExact
from cpython.long cimport PyLong_CheckExact, PyLong_AsLongLongAndOverflow
from cpython.object cimport Py_TYPE
cimport cython


cdef inline int _mx_check_enum_schema(object receiver, object owner, tuple bindings) except -1:
    """Declared model-code bindings are fixed until the model is rebuilt."""
    cdef object actual = <object>Py_TYPE(receiver)
    cdef object base, name
    if actual is owner:
        return 0
    for base in actual.__mro__:
        if base is owner:
            break
        for name in bindings:
            if name in base.__dict__:
                raise TypeError(f"enum definition {name!r} is fixed in this compiled model; rebuild the model to change it")
    return 0


cdef inline double _mx_pow(double b, double e) except? -1.0:
    """``**`` between operands mx2cy has typed as C numbers.

    Cython would otherwise evaluate a fractional power on
    ``double complex``, which is slower and does not compile at all
    where the result is compared.  Only powers whose exponent is
    provably floating are rewritten to this, so integer arithmetic is
    unaffected; see :mod:`modelx_cython.powers`.

    Raises rather than returning the ``nan`` or ``inf`` C ``pow``
    gives for the two inputs a model should not have produced: a
    negative base under a fractional exponent, where Python returns a
    complex number the compiled model has no type for, and ``0.0`` to
    a negative power, where Python itself raises.  Both guards are
    limited to finite operands, because CPython disposes of an
    infinite base or exponent before either case applies and C ``pow``
    already agrees with it there.  Other divergences from Python's
    ``float.__pow__`` are left alone, notably overflow, on which
    Python raises and every C double in the generated model saturates.
    """
    if _mx_c_isfinite(b) and _mx_c_isfinite(e):
        if b < 0.0 and e != _mx_c_floor(e):
            raise ValueError(
                "a negative number cannot be raised to a fractional power")
        if b == 0.0 and e < 0.0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
    return _mx_c_pow(b, e)


cdef inline object _mx_integral_math_call(object fn, object value, object canonical, bint up):
    cdef double x
    if canonical is not None and fn is canonical and PyFloat_CheckExact(value):
        x = PyFloat_AS_DOUBLE(value)
        # Both operations stay within signed 64 bits on this interval.
        if -9223372036854775808.0 <= x < 9223372036854775808.0:
            return <long long>(_mx_c_ceil(x) if up else _mx_c_floor(x))
    return fn(value)


cdef inline double _mx_nearest_even(double scaled, double residual) noexcept:
    cdef double lower = _mx_c_floor(scaled)
    cdef double fraction = scaled - lower
    if (fraction > 0.5 or fraction == 0.5
            and (residual > 0.0 or residual == 0.0 and (<long long>lower & 1))):
        return lower + 1.0
    return lower


cdef inline object _mx_round_one(object fn, object value, object canonical):
    cdef double x, magnitude
    if canonical is not None and fn is canonical and PyFloat_CheckExact(value):
        x = PyFloat_AS_DOUBLE(value)
        magnitude = fabs(x)
        if magnitude < 4503599627370496.0:
            return <long long>copysign(_mx_nearest_even(magnitude, 0.0), x)
    return fn(value)


cdef inline object _mx_round_digits(object fn, object value, object digits, object canonical):
    cdef double x, magnitude, scale = 1.0, scaled, residual = 0.0, rounded
    cdef int overflow
    cdef long long n, i
    if canonical is not None and fn is canonical and PyFloat_CheckExact(value) and PyLong_CheckExact(digits):
        n = PyLong_AsLongLongAndOverflow(digits, &overflow)
        x = PyFloat_AS_DOUBLE(value)
        if not overflow and -22 <= n <= 22:
            if not _mx_c_isfinite(x) or x == 0.0:
                return value
            for i in range(n if n >= 0 else -n):
                scale *= 10.0
            magnitude = fabs(x)
            scaled = magnitude * scale if n >= 0 else magnitude / scale
            if scaled < 4503599627370496.0:
                if scaled - _mx_c_floor(scaled) == 0.5:
                    # A rounded intermediate can land on a half even when
                    # the exact scaled input lies just to one side of it.
                    residual = (fma(magnitude, scale, -scaled) if n >= 0
                                else fma(-scaled, scale, magnitude))
                rounded = _mx_nearest_even(scaled, residual)
                return copysign(rounded / scale if n >= 0 else rounded * scale, x)
    return fn(value, digits)


cdef struct PowerMemo:
    bint ready
    double base
    double exponent
    double value


cdef inline double _mx_memo_pow(PowerMemo* memo, double base, double exponent) except? -1.0:
    cdef double value
    if (memo.ready and memcmp(&memo.base, &base, sizeof(double)) == 0
            and memcmp(&memo.exponent, &exponent, sizeof(double)) == 0):
        return memo.value
    value = _mx_pow(base, exponent)
    if _mx_c_isfinite(base) and _mx_c_isfinite(exponent):
        memo.base = base
        memo.exponent = exponent
        memo.value = value
        memo.ready = True
    return value


@cython.cdivision(True)
cdef inline bint _mx_divisible(long long dividend, long long divisor) except? -1:
    if divisor == 0:
        raise ZeroDivisionError("integer division or modulo by zero")
    return divisor == -1 or dividend % divisor == 0


cdef inline bint _mx_divisible_after_decrement(long long dividend, long long divisor) except? -1:
    if dividend == LLONG_MIN:
        return ((<object>dividend) - 1) % divisor == 0
    return _mx_divisible(dividend - 1, divisor)


cdef extern from *:
    """
    /* Acquire/release accessors for the _has_ flags of locked Spaces.
       The writer stores the cached value, then the flag with release
       semantics, under the model lock; a reader that sees the flag with
       acquire semantics is therefore guaranteed to see the value, without
       taking the lock. pyatomic.h is included by Python.h from CPython 3.13,
       the first version with a free-threaded build; a build with the GIL
       needs no ordering, and neither does an older version of Python. */
    #if defined(Py_GIL_DISABLED)
    #define __MX_LOAD_FLAG(p)      _Py_atomic_load_int_acquire((const int *)(p))
    #define __MX_STORE_FLAG(p, v)  _Py_atomic_store_int_release((int *)(p), (v))
    #else
    #define __MX_LOAD_FLAG(p)      (*(p))
    #define __MX_STORE_FLAG(p, v)  ((void)(*(p) = (v)))
    #endif
    """
    bint _mx_load_flag "__MX_LOAD_FLAG" (bint *p) noexcept nogil
    void _mx_store_flag "__MX_STORE_FLAG" (bint *p, bint v) noexcept nogil

cdef class BaseMxObject:
    pass

cdef class BaseParent(BaseMxObject):

    cdef public dict  _mx_spaces
    cdef public BaseParent _parent
    cdef public BaseModel _model
    cdef public str _name
    # The threading.RLock shared by the locked Spaces of the model, or None.
    # Declared on the common base of the model and of the Spaces: a locked
    # Space reads it through its typed _model field and through itself.
    cdef public object _mx_lock

cdef class BaseModel(BaseParent):
    pass

cdef class BaseSpace(BaseParent):

    cdef bint _mx_is_cells_set
    cdef dict _mx_cells

    cdef BaseSpace _space
    cdef dict _mx_itemspaces
    cdef public list _mx_roots
