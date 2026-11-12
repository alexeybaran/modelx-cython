from libc.math cimport (
    pow as _mx_c_pow, floor as _mx_c_floor, isfinite as _mx_c_isfinite)
from libc.limits cimport LLONG_MAX, LLONG_MIN
from libc.string cimport memcmp
from cpython.long cimport PyLong_AsLongLong, PyLong_CheckExact
from cpython.number cimport PyNumber_Index
from cpython.object cimport Py_TYPE, PyObject_GenericGetAttr, PyObject_GenericSetAttr
from cpython.unicode cimport PyUnicode_Check, PyUnicode_Compare
from modelx_cython._rounding cimport _mx_round_one, _mx_round_digits
cimport cython


cdef inline long long _mx_exact_int64(object value, str path) except? -1:
    """Cross a signed 64-bit boundary through the Python index protocol."""
    try:
        if PyLong_CheckExact(value):
            return PyLong_AsLongLong(value)
        return PyLong_AsLongLong(PyNumber_Index(value))
    except TypeError as error:
        raise TypeError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
        ) from error
    except OverflowError as error:
        raise OverflowError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
        ) from error


ctypedef unsigned long _mx_native_ulong
ctypedef unsigned long long _mx_native_ulonglong

ctypedef fused _mx_integer_source:
    object
    long long
    _mx_native_ulong
    _mx_native_ulonglong
    double


cdef inline long long _mx_integer_result(_mx_integer_source value, str path) except? -1:
    """Check Python objects and other C scalars before storing a signed result."""
    if _mx_integer_source is object:
        return _mx_exact_int64(value, path)
    elif _mx_integer_source is double:
        raise TypeError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
        )
    elif _mx_integer_source is _mx_native_ulong or _mx_integer_source is _mx_native_ulonglong:
        if value > LLONG_MAX:
            raise OverflowError(
                f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
            )
        return <long long>value
    else:
        return value


cdef inline long long _mx_signed_overflow(str path) except? -1:
    raise OverflowError(
        f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
    )


cdef extern from *:
    """
    #include <limits.h>
    #if defined(_MSC_VER) && defined(_M_X64)
    #include <intrin.h>
    #endif

    static CYTHON_INLINE int mx_add_overflow(long long left, long long right,
                                             long long *result) {
    #if defined(__GNUC__) || defined(__clang__)
        return __builtin_add_overflow(left, right, result);
    #elif defined(_MSC_VER)
        /* MSVC's signed integers are two's complement. Detect a sign change
           in the unsigned result without executing a signed overflow. */
        unsigned long long a = (unsigned long long)left;
        unsigned long long b = (unsigned long long)right;
        unsigned long long sum = a + b;
        *result = (long long)sum;
        return ((a ^ sum) & (b ^ sum)) > (unsigned long long)LLONG_MAX;
    #else
        if ((right > 0 && left > LLONG_MAX - right) ||
            (right < 0 && left < LLONG_MIN - right))
            return 1;
        *result = left + right;
        return 0;
    #endif
    }

    static CYTHON_INLINE int mx_sub_overflow(long long left, long long right,
                                             long long *result) {
    #if defined(__GNUC__) || defined(__clang__)
        return __builtin_sub_overflow(left, right, result);
    #elif defined(_MSC_VER)
        unsigned long long a = (unsigned long long)left;
        unsigned long long b = (unsigned long long)right;
        unsigned long long difference = a - b;
        *result = (long long)difference;
        return ((a ^ b) & (a ^ difference)) > (unsigned long long)LLONG_MAX;
    #else
        if ((right > 0 && left < LLONG_MIN + right) ||
            (right < 0 && left > LLONG_MAX + right))
            return 1;
        *result = left - right;
        return 0;
    #endif
    }

    static CYTHON_INLINE int mx_mul_overflow(long long left, long long right,
                                             long long *result) {
    #if defined(__GNUC__) || defined(__clang__)
        return __builtin_mul_overflow(left, right, result);
    #elif defined(_MSC_VER) && defined(_M_X64)
        long long high;
        *result = _mul128(left, right, &high);
        return high != (*result < 0 ? -1 : 0);
    #else
        if (left > 0 ? (right > 0 ? left > LLONG_MAX / right : right < LLONG_MIN / left)
                     : (left < 0 && (right > 0 ? left < LLONG_MIN / right
                                              : right < 0 && left < LLONG_MAX / right)))
            return 1;
        *result = left * right;
        return 0;
    #endif
    }
    """
    bint _mx_add_overflow "mx_add_overflow"(long long, long long, long long*) noexcept
    bint _mx_sub_overflow "mx_sub_overflow"(long long, long long, long long*) noexcept
    bint _mx_mul_overflow "mx_mul_overflow"(long long, long long, long long*) noexcept


cdef inline long long _mx_signed_add(long long left, long long right, str path) except? -1:
    cdef long long value
    if _mx_add_overflow(left, right, &value):
        return _mx_signed_overflow(path)
    return value


cdef inline long long _mx_signed_sub(long long left, long long right, str path) except? -1:
    cdef long long value
    if _mx_sub_overflow(left, right, &value):
        return _mx_signed_overflow(path)
    return value


cdef inline long long _mx_signed_mul(long long left, long long right, str path) except? -1:
    cdef long long value
    if _mx_mul_overflow(left, right, &value):
        return _mx_signed_overflow(path)
    return value


cdef inline long long _mx_signed_floor(long long left, long long right, str path) except? -1:
    if right == -1 and left == LLONG_MIN:
        return _mx_signed_overflow(path)
    return left // right


cdef inline bint _mx_ref_name_is(object name, str expected) except? -1:
    return PyUnicode_Check(name) and PyUnicode_Compare(name, expected) == 0


cdef inline int _mx_generic_setattr(object receiver, object name, object value) except -1:
    return PyObject_GenericSetAttr(receiver, name, value)


cdef inline object _mx_generic_getattr(object receiver, object name):
    return PyObject_GenericGetAttr(receiver, name)


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
    # The reentrant lock shared by the locked Spaces of the model, or None.
    # Declared on the common base of the model and of the Spaces: a locked
    # Space reads it through its typed _model field and through itself.
    cdef public object _mx_lock

cdef class BaseModel(BaseParent):
    cdef public object _mx_collect_workers_override
    cdef public object _mx_collect_dispatch_state

cdef class BaseSpace(BaseParent):

    cdef bint _mx_is_cells_set
    cdef dict _mx_cells

    cdef BaseSpace _space
    cdef dict _mx_itemspaces
    cdef public list _mx_roots
