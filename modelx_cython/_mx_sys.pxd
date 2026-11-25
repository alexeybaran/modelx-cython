from libc.math cimport (
    pow as _mx_c_pow, floor as _mx_c_floor, isfinite as _mx_c_isfinite)
from libc.limits cimport LLONG_MAX, LLONG_MIN
from libc.string cimport memcmp
from cpython.long cimport PyLong_AsLongLong, PyLong_AsLongLongAndOverflow, PyLong_Check, PyLong_CheckExact
from cpython.number cimport PyNumber_Index
from cpython.object cimport Py_TYPE, PyObject_GenericGetAttr, PyObject_GenericSetAttr
from cpython.unicode cimport PyUnicode_Check, PyUnicode_Compare
from libc.math cimport fabs, copysign
cimport cython


cdef inline object _mx_named(object error, str path):
    """``error`` marked as naming the Cell ``path``."""
    error._mx_cell = path
    return error


cdef inline object _mx_named_overflow(object error, str path):
    """The OverflowError to raise from Cell ``path``: ``error`` itself when a Cell already named it,
    else a new one naming ``path``, caused by ``error``."""
    if getattr(error, '_mx_cell', None) is not None:
        return error
    named = _mx_named(OverflowError(f"{path}: {error} (integer arithmetic beyond 64 bits)"), path)
    named.__cause__ = error
    return named


cdef inline long long _mx_exact_int64_checked(object value, str path) except? -1:
    try:
        if PyLong_Check(value):
            return PyLong_AsLongLong(value)
        return PyLong_AsLongLong(PyNumber_Index(value))
    except TypeError as error:
        raise _mx_named(TypeError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
        ), path) from error
    except OverflowError as error:
        raise _mx_named(OverflowError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
        ), path) from error


cdef inline long long _mx_exact_int64(object value, str path) except? -1:
    """Cross a signed 64-bit boundary through the Python index protocol.

    An int or any int subclass (IntEnum, bool) is read directly: PyNumber_Index returns such an
    object unchanged, so the value and the OverflowError are the same, without taking and
    releasing a reference on it (diag-scale P3: enum members are shared by every worker). An exact
    int that fits is read without the exception handling, which every other value takes."""
    cdef int overflow = 0
    cdef long long result
    if PyLong_CheckExact(value):
        result = PyLong_AsLongLongAndOverflow(value, &overflow)
        if not overflow:
            return result
    return _mx_exact_int64_checked(value, path)


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
        raise _mx_named(TypeError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
        ), path)
    elif _mx_integer_source is _mx_native_ulong or _mx_integer_source is _mx_native_ulonglong:
        if value > LLONG_MAX:
            raise _mx_named(OverflowError(
                f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
            ), path)
        return <long long>value
    else:
        return value


cdef extern from *:
    """
    /* Python's a // b on C integers with the branches Cython emits on LP64 platforms, on every
       platform: Cython 3.3.1 compiles its LLONG_MIN // -1 check only where long long is as wide as
       long, so the C division traps there (STATUS_INTEGER_OVERFLOW on Windows). The error value
       is LLONG_MAX (a quotient only of LLONG_MAX // 1), so a quotient of -1 needs no error check.
       The path is the Cell's: the OverflowError names it. */
    static CYTHON_INLINE long long __mx_signed_floordiv(long long a, long long b, PyObject *path) {
        long long q, r;
        if (b == 0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "integer division or modulo by zero");
            return LLONG_MAX;
        }
        if (b == -1 && a == LLONG_MIN) {
            /* named here: the formula keeps no handler around its body (integer guards) */
            PyErr_Format(PyExc_OverflowError, "%U: value too large (integer arithmetic beyond 64 bits)", path);
            return LLONG_MAX;
        }
        q = a / b;
        r = a - q * b;
        return q - ((r != 0) & ((r ^ b) < 0));
    }
    """
    long long _mx_signed_floordiv "__mx_signed_floordiv"(long long left, long long right, str path) \
        except? 9223372036854775807


cdef extern from *:
    """
    /* Python's a % b on C integers: Cython's remainder (zero check, sign adaptation), and 0 for
       b == -1, where C's a % b traps (LLONG_MIN % -1) and Cython does not check it. A Python
       remainder is smaller than b in magnitude, so LLONG_MAX is never one: the error value. */
    static CYTHON_INLINE long long __mx_signed_remainder(long long a, long long b, PyObject *path) {
        long long r;
        (void)path;
        if (b == 0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "integer division or modulo by zero");
            return LLONG_MAX;
        }
        if (b == -1)
            return 0;
        r = a % b;
        return r + ((r != 0) & ((r ^ b) < 0)) * b;
    }
    """
    long long _mx_signed_remainder "__mx_signed_remainder"(long long left, long long right, str path) \
        except 9223372036854775807


@cython.overflowcheck(True)
cdef inline long long _mx_signed_left_shift(long long value, long long count, str path) except? -1:
    """Python's ``value << count`` on C integers: a negative count raises ValueError, a count of
    64 or more is exact for 0 (and -1 << 63), and a result beyond 64 bits raises OverflowError
    naming the Cell. Cython's own checked shift raises for negative values (-3 << 2)."""
    if count < 0:
        raise ValueError("negative shift count")
    if count >= 63:
        if value == 0:
            return 0
        if count == 63 and value == -1:
            return LLONG_MIN
        raise _mx_named(OverflowError(f"{path}: value too large (integer arithmetic beyond 64 bits)"), path)
    try:
        return value * ((<long long>1) << count)
    except OverflowError as error:
        raise _mx_named_overflow(error, path)


@cython.overflowcheck(True)
cdef inline long long _mx_signed_pow(long long base, long long exponent, str path) except? -1:
    """``base ** exponent`` of two C integers, exactly or OverflowError (Cython's overflowcheck on
    the products; Cython's own C integer power wraps). A negative exponent has a float value in
    Python, which a C integer result cannot hold: TypeError naming the Cell. By squaring: each
    square taken is at most the result in magnitude (|base| >= 2), so no check fails for a power
    that fits."""
    cdef long long result = 1
    if exponent < 0:
        raise _mx_named(TypeError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."), path)
    try:
        while True:
            if exponent & 1:
                result = result * base
            exponent >>= 1
            if exponent == 0:
                return result
            base = base * base
    except OverflowError as error:
        raise _mx_named_overflow(error, path)


cdef inline long long _mx_signed_right_shift(long long value, long long count, str path) except? -1:
    if count < 0:
        raise ValueError("negative shift count")
    if count >= 63:
        return -1 if value < 0 else 0
    # Only shift nonnegative C values. Python rounds negative shifts down.
    if value < 0:
        return -1 - ((-1 - value) >> count)
    return value >> count


cdef extern from *:
    """
    /* Flagged integer arithmetic (integer guards). Each operation returns the wrapped C result and
       ORs 1 into *flag when the exact result does not fit in 64 bits; the formula tests the flag
       once, before it returns, and raises OverflowError naming the Cell (_mx_flagged_overflow).
       There is no error path at the operation, so a formula stays one straight-line function that
       the C compiler can inline. GCC and Clang have the checked builtins; elsewhere (MSVC) the
       portable forms compute in unsigned arithmetic, which wraps by definition, and derive the
       overflow from the signs (+, -) or from the high half of the 128-bit product (*). */
    #if defined(__GNUC__) || defined(__clang__)
    #define __MX_ADD_OVERFLOWS(a, b, r) __builtin_add_overflow((a), (b), (r))
    #define __MX_SUB_OVERFLOWS(a, b, r) __builtin_sub_overflow((a), (b), (r))
    #define __MX_MUL_OVERFLOWS(a, b, r) __builtin_mul_overflow((a), (b), (r))
    #else
    static CYTHON_INLINE int __MX_ADD_OVERFLOWS(long long a, long long b, long long *r) {
        unsigned long long sum = (unsigned long long)a + (unsigned long long)b;
        *r = (long long)sum;
        return ((a ^ *r) & (b ^ *r)) < 0;
    }
    static CYTHON_INLINE int __MX_SUB_OVERFLOWS(long long a, long long b, long long *r) {
        unsigned long long difference = (unsigned long long)a - (unsigned long long)b;
        *r = (long long)difference;
        return ((a ^ b) & (a ^ *r)) < 0;
    }
    #if defined(_MSC_VER) && defined(_M_X64)
    #include <intrin.h>
    static CYTHON_INLINE int __MX_MUL_OVERFLOWS(long long a, long long b, long long *r) {
        long long high;
        *r = _mul128(a, b, &high);
        return high != (*r >> 63);
    }
    #elif defined(_MSC_VER) && defined(_M_ARM64)
    #include <intrin.h>
    static CYTHON_INLINE int __MX_MUL_OVERFLOWS(long long a, long long b, long long *r) {
        *r = (long long)((unsigned long long)a * (unsigned long long)b);
        return __mulh(a, b) != (*r >> 63);
    }
    #else
    static CYTHON_INLINE int __MX_MUL_OVERFLOWS(long long a, long long b, long long *r) {
        *r = (long long)((unsigned long long)a * (unsigned long long)b);
        if (a == 0 || b == 0)
            return 0;
        if ((a == -1 && b == LLONG_MIN) || (b == -1 && a == LLONG_MIN))
            return 1;
        return *r / b != a;
    }
    #endif
    #endif
    static CYTHON_INLINE long long __mx_flag_add(long long a, long long b, int *flag) {
        long long r;
        *flag |= __MX_ADD_OVERFLOWS(a, b, &r);
        return r;
    }
    static CYTHON_INLINE long long __mx_flag_sub(long long a, long long b, int *flag) {
        long long r;
        *flag |= __MX_SUB_OVERFLOWS(a, b, &r);
        return r;
    }
    static CYTHON_INLINE long long __mx_flag_mul(long long a, long long b, int *flag) {
        long long r;
        *flag |= __MX_MUL_OVERFLOWS(a, b, &r);
        return r;
    }
    static CYTHON_INLINE long long __mx_flag_neg(long long a, int *flag) {
        *flag |= (a == LLONG_MIN);
        return (long long)(0ULL - (unsigned long long)a);
    }
    """
    long long _mx_flag_add "__mx_flag_add"(long long left, long long right, int *flag) noexcept nogil
    long long _mx_flag_sub "__mx_flag_sub"(long long left, long long right, int *flag) noexcept nogil
    long long _mx_flag_mul "__mx_flag_mul"(long long left, long long right, int *flag) noexcept nogil
    long long _mx_flag_neg "__mx_flag_neg"(long long value, int *flag) noexcept nogil


cdef inline object _mx_flagged_overflow(str path):
    """The OverflowError of a formula whose flagged integer arithmetic left 64 bits (cold path)."""
    return _mx_named(OverflowError(f"{path}: value too large (integer arithmetic beyond 64 bits)"), path)


cdef inline long long _mx_flag_checked(long long value, int *flag, str path) except? -1:
    """``value``, computed with flagged operations, once the flag shows it exact: an argument a
    formula passes on is tested before the call receives it."""
    if flag[0]:
        raise _mx_flagged_overflow(path)
    return value


cdef inline bint _mx_ref_name_is(object name, str expected) except? -1:
    return PyUnicode_Check(name) and PyUnicode_Compare(name, expected) == 0


cdef inline int _mx_generic_setattr(object receiver, object name, object value) except -1:
    return PyObject_GenericSetAttr(receiver, name, value)


cdef inline object _mx_generic_getattr(object receiver, object name):
    return PyObject_GenericGetAttr(receiver, name)


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


cdef extern from *:
    """
    /* The short readiness flags of a dense cache of a shared Space. Workers fill the
       cache concurrently: the value is stored, then the flag after a release fence;
       a reader loads the flag, then an acquire fence orders its read of the value. A
       relaxed 16-bit access and the fences are CPython's (pyatomic.h, 3.13 and later);
       a build with the GIL keeps plain accesses. On x86 the fences emit no instruction. */
    static inline short __mx_load_dense_flag(short *p) {
    #if defined(Py_GIL_DISABLED)
        short flag = (short)_Py_atomic_load_int16_relaxed((int16_t *)p);
        _Py_atomic_fence_acquire();
        return flag;
    #else
        return *p;
    #endif
    }
    static inline void __mx_store_dense_flag(short *p, short v) {
    #if defined(Py_GIL_DISABLED)
        _Py_atomic_fence_release();
        _Py_atomic_store_int16_relaxed((int16_t *)p, (int16_t)v);
    #else
        *p = v;
    #endif
    }
    """
    short _mx_load_dense_flag "__mx_load_dense_flag" (short *p) noexcept nogil
    void _mx_store_dense_flag "__mx_store_dense_flag" (short *p, short v) noexcept nogil


cdef extern from *:
    """
    /* S1 (value_storage.py): the value arrays of dense Cell caches are not zero-filled when an
       object is allocated. A value is read only after its flag is set, and a flag is set only after
       its value is stored. A class registers its value arrays from a probe instance; finish()
       requires them to be adjacent members and installs an allocator that zeroes every other byte,
       as PyType_GenericAlloc zeroes all of them. Pickling and copying reduce a copy of the object
       in which a value is copied only behind a flag read from the live object, so no unwritten byte
       leaves the process. Where the mechanism is unavailable (another interpreter, Python before
       3.10, a debug build) nothing is installed; where the layout check fails, nothing is installed
       and a RuntimeWarning names the class. Test-only macros: MX_POISON_UNWRITTEN_VALUES fills the
       values with 0xA5; MX_TEST_VALUES_FALLBACK=1 fails the layout check, =2 makes it unavailable. */
    typedef struct { Py_ssize_t value, bytes, value_size, flag, flag_size; } __mx_value_array;
    typedef struct __mx_values_layout {
        PyTypeObject *type; char *probe; PyObject *reduce;
        Py_ssize_t lo, hi, count, capacity, failed;
        __mx_value_array *arrays; struct __mx_values_layout *next;
    } __mx_values_layout;

    #if PY_VERSION_HEX >= 0x030D0000
    #define __MX_VALUES_LOCK(op)  Py_BEGIN_CRITICAL_SECTION(op)
    #define __MX_VALUES_UNLOCK()  Py_END_CRITICAL_SECTION()
    #else
    #define __MX_VALUES_LOCK(op)  {
    #define __MX_VALUES_UNLOCK()  }
    #endif

    static inline __mx_values_layout **__mx_values_head(void) {
        static __mx_values_layout *head = NULL;
        return &head;
    }

    static inline __mx_values_layout *__mx_values_find(PyTypeObject *t) {
        for (; t != NULL; t = t->tp_base)
            for (__mx_values_layout *l = *__mx_values_head(); l != NULL; l = l->next)
                if (l->type == t) return l;
        return NULL;
    }

    static inline PyObject *__mx_values_alloc(PyTypeObject *t, Py_ssize_t nitems) {
        /* The registered class only: a static subtype inherits tp_alloc and is zero-filled. */
        __mx_values_layout *l = nitems == 0 ? __mx_values_find(t) : NULL;
        if (l == NULL || l->type != t || t->tp_itemsize != 0 || !PyType_IS_GC(t)) return PyType_GenericAlloc(t, nitems);
        PyObject *obj = PyObject_GC_New(PyObject, t);     /* header initialised, body not zeroed */
        if (obj == NULL) return NULL;
        memset((char *)obj + sizeof(PyObject), 0, (size_t)(l->lo - (Py_ssize_t)sizeof(PyObject)));
    #ifdef MX_POISON_UNWRITTEN_VALUES
        memset((char *)obj + l->lo, 0xA5, (size_t)(l->hi - l->lo));
    #endif
        memset((char *)obj + l->hi, 0, (size_t)(t->tp_basicsize - l->hi));
        PyObject_GC_Track(obj);
        return obj;
    }

    /* One flag of the live object, loaded with acquire semantics as the generated readers load it
       (short: shared dense caches; int: bint flags of locked Spaces), and stored into the copy. */
    static inline int __mx_values_flag(const char *live, char *copy, Py_ssize_t size) {
        if (size == (Py_ssize_t)sizeof(short)) {
            short flag = __mx_load_dense_flag((short *)live);
            memcpy(copy, &flag, sizeof flag);
            return flag != 0;
        }
        int flag = __MX_LOAD_FLAG((int *)live);
        memcpy(copy, &flag, sizeof flag);
        return flag != 0;
    }

    static inline PyObject *__mx_values_reduce(PyObject *self, PyObject *unused) {
        PyTypeObject *t = Py_TYPE(self);
        __mx_values_layout *l = __mx_values_find(t);
        (void)unused;
        if (l == NULL) { PyErr_SetString(PyExc_SystemError, "no dense value layout"); return NULL; }
        if (l->type != t || t->tp_alloc != __mx_values_alloc)
            return PyObject_CallOneArg(l->reduce, self);  /* a zero-filled subtype */
        PyObject *copy = PyObject_GC_New(PyObject, t);    /* untracked; its members are borrowed from self */
        if (copy == NULL) return NULL;
        char *dst = (char *)copy, *src = (char *)self;
        PyObject *result;
    #ifndef Py_GIL_DISABLED
        int gc = PyGC_Disable();   /* no collection, so no finalizer or thread switch, while members are borrowed */
    #endif
        /* Free-threaded: self's critical section, which Cython's reduce and the setters of public members
           take, is held from the copy to the end of the reduce, so no member is replaced meanwhile. */
        __MX_VALUES_LOCK(self);
        memcpy(dst + sizeof(PyObject), src + sizeof(PyObject), (size_t)(l->lo - (Py_ssize_t)sizeof(PyObject)));
        memcpy(dst + l->hi, src + l->hi, (size_t)(t->tp_basicsize - l->hi));
        for (Py_ssize_t a = 0; a < l->count; a++) {   /* a value is copied only behind a flag seen set */
            const __mx_value_array *v = &l->arrays[a];
            for (Py_ssize_t i = 0, n = v->bytes / v->value_size; i < n; i++) {
                Py_ssize_t value = v->value + i * v->value_size, flag = v->flag + i * v->flag_size;
                if (__mx_values_flag(src + flag, dst + flag, v->flag_size))
                    memcpy(dst + value, src + value, (size_t)v->value_size);
                else
                    memset(dst + value, 0, (size_t)v->value_size);
            }
        }
        result = PyObject_CallOneArg(l->reduce, copy);
        __MX_VALUES_UNLOCK();
    #ifndef Py_GIL_DISABLED
        if (gc) PyGC_Enable();
    #endif
        if (Py_REFCNT(copy) != 1) {
            /* The reduce kept the copy (Cython's does not). This function's reference is never released,
               so the copy is never deallocated; it is not returned, because its members are borrowed. */
            Py_XDECREF(result);
            PyErr_SetString(PyExc_SystemError, "the reduce of a dense value copy kept the copy");
            return NULL;
        }
    #if PY_VERSION_HEX >= 0x030D0000
        {   /* PyObject_GC_New reported the copy to a reference tracer; PyObject_GC_Del does not */
            void *data = NULL;
            PyRefTracer tracer = PyRefTracer_GetTracer(&data);
            if (tracer != NULL) tracer(copy, PyRefTracer_DESTROY, data);
        }
    #endif
        if (PyType_HasFeature(t, Py_TPFLAGS_HEAPTYPE)) Py_DECREF(t);   /* taken by PyObject_GC_New */
        PyObject_GC_Del(copy);
        return result;
    }

    static inline void *__mx_values_begin(PyObject *type, PyObject *probe) {
        __mx_values_layout *l = (__mx_values_layout *)PyMem_RawCalloc(1, sizeof(__mx_values_layout));
        if (l != NULL) { l->type = (PyTypeObject *)type; l->probe = (char *)probe; }
        return l;
    }

    static inline void __mx_values_add(void *layout, void *values, Py_ssize_t bytes, Py_ssize_t value_size,
                                       void *flags, Py_ssize_t flag_bytes, Py_ssize_t flag_size) {
        __mx_values_layout *l = (__mx_values_layout *)layout;
        if (l == NULL || l->failed) return;
        if (l->count == l->capacity) {
            Py_ssize_t capacity = 2 * l->capacity + 8;
            void *arrays = PyMem_RawRealloc(l->arrays, (size_t)capacity * sizeof(__mx_value_array));
            if (arrays == NULL) { l->failed = 1; return; }
            l->arrays = (__mx_value_array *)arrays; l->capacity = capacity;
        }
        __mx_value_array v = {(char *)values - l->probe, bytes, value_size, (char *)flags - l->probe, flag_size};
        if (value_size <= 0 || bytes % value_size || flag_bytes != bytes / value_size * flag_size
                || (flag_size != (Py_ssize_t)sizeof(short) && flag_size != (Py_ssize_t)sizeof(int)))
            l->failed = 1;
        l->arrays[l->count++] = v;
    }

    /* 1: installed. 0: not installed, every byte is zeroed as before; a failed check warns. -1: the
       warning was raised as an error. */
    static inline int __mx_values_finish(void *layout) {
        __mx_values_layout *l = (__mx_values_layout *)layout;
        if (l == NULL) return 0;
        PyTypeObject *t = l->type;
        int ok = 0, checked = 0;
    #if !CYTHON_COMPILING_IN_CPYTHON || CYTHON_COMPILING_IN_LIMITED_API || PY_VERSION_HEX < 0x030A0000 \
        || defined(Py_REF_DEBUG) || defined(Py_TRACE_REFS) \
        || (defined(MX_TEST_VALUES_FALLBACK) && MX_TEST_VALUES_FALLBACK == 2)
        /* Unavailable. Py_REF_DEBUG and Py_TRACE_REFS: the scrubbed copy cannot be unregistered from the
           reference total or the list of all objects through the public API. */
        (void)t;
    #else
        checked = 1;
        ok = !l->failed && l->count > 0;
        if (ok) {   /* adjacent value arrays only: no other member lies in [lo, hi) */
            l->lo = l->arrays[0].value;
            for (Py_ssize_t a = 0; a < l->count; a++) if (l->arrays[a].value < l->lo) l->lo = l->arrays[a].value;
            l->hi = l->lo;
            for (Py_ssize_t a = 0; a < l->count && ok; a++) {
                Py_ssize_t b = 0;
                while (b < l->count && l->arrays[b].value != l->hi) b++;
                ok = b < l->count;
                if (ok) l->hi += l->arrays[b].bytes;
            }
            ok = ok && l->lo >= (Py_ssize_t)sizeof(PyObject) && l->hi <= t->tp_basicsize && PyType_IS_GC(t)
                 && t->tp_itemsize == 0 && t->tp_alloc == PyType_GenericAlloc && __mx_values_find(t) == NULL;
        #ifdef Py_TPFLAGS_INLINE_VALUES
            ok = ok && !(t->tp_flags & Py_TPFLAGS_INLINE_VALUES);
        #endif
        #if defined(MX_TEST_VALUES_FALLBACK) && MX_TEST_VALUES_FALLBACK == 1
            ok = 0;
        #endif
        }
        /* Named as Cython names its reduce: the same __name__ as before, and a compiled subtype's
           __Pyx_setup_reduce replaces it with the subtype's own. */
        static PyMethodDef reduce_def = {"__reduce_cython__", (PyCFunction)__mx_values_reduce, METH_NOARGS, NULL};
        PyObject *descr = NULL;
        if (ok) {   /* the class's own __reduce__ (Cython's auto-pickle) is reduced from a scrubbed copy */
            l->reduce = PyDict_GetItemString(t->tp_dict, "__reduce__");
            descr = l->reduce != NULL ? PyDescr_NewMethod(t, &reduce_def) : NULL;
            ok = descr != NULL;
            if (ok) Py_INCREF(l->reduce);
        }
        if (ok && PyDict_SetItemString(t->tp_dict, "__reduce__", descr) < 0) { Py_DECREF(l->reduce); ok = 0; }
        Py_XDECREF(descr);
        if (ok) {
            PyType_Modified(t);
            l->probe = NULL;
            l->next = *__mx_values_head();
            *__mx_values_head() = l;
            t->tp_alloc = __mx_values_alloc;
        }
    #endif
        if (!ok) {
            PyMem_RawFree(l->arrays); PyMem_RawFree(l); PyErr_Clear();
            if (checked && PyErr_WarnFormat(PyExc_RuntimeWarning, 1, "dense cache values of %s are zero-filled "
                                            "at allocation: the value layout check failed", t->tp_name) < 0)
                return -1;
        }
        return ok;
    }
    """
    void *_mx_values_begin "__mx_values_begin" (object type, object probe)
    void _mx_values_add "__mx_values_add" (void *layout, void *values, Py_ssize_t value_bytes, Py_ssize_t value_size,
                                           void *flags, Py_ssize_t flag_bytes, Py_ssize_t flag_size) noexcept
    int _mx_values_finish "__mx_values_finish" (void *layout) except -1

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
    pass

cdef class BaseSpace(BaseParent):

    cdef bint _mx_is_cells_set
    cdef dict _mx_cells

    cdef BaseSpace _space
    cdef dict _mx_itemspaces
    cdef public list _mx_roots
