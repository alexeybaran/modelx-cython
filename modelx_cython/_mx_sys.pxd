from libc.math cimport (
    pow as _mx_c_pow, floor as _mx_c_floor, isfinite as _mx_c_isfinite)
from libc.string cimport memcmp
from cpython.long cimport PyLong_CheckExact
from cpython.number cimport PyNumber_Index
from cpython.object cimport Py_TYPE, PyObject_GenericSetAttr
from cpython.unicode cimport PyUnicode_Check, PyUnicode_Compare
from libc.math cimport fabs, copysign
cimport cython


cdef inline object _mx_named(object error, str path):
    """``error`` marked as naming the Cell ``path``."""
    error._mx_cell = path
    return error


cdef inline object _mx_integral(object value, str path):
    """``value`` from Python for a C integer: an int, a float that is a whole number, or anything
    else through the index protocol. A fractional float or a value that is not an integer raises
    TypeError naming ``path``; a value beyond 64 bits is left to Cython's own conversion."""
    if PyLong_CheckExact(value):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return value
    else:
        try:
            return PyNumber_Index(value)
        except TypeError:
            pass
    raise _mx_named(TypeError(f"{path} requires an integer, got {value!r}"), path)


cdef inline bint _mx_ref_name_is(object name, str expected) except? -1:
    return PyUnicode_Check(name) and PyUnicode_Compare(name, expected) == 0


cdef inline int _mx_generic_setattr(object receiver, object name, object value) except -1:
    return PyObject_GenericSetAttr(receiver, name, value)


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
    /* Unwritten values (value_storage.py): the value arrays of dense Cell caches are not zero-filled when an
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

# Native math-module calls (native_math.py). _mx_stdlib_math is the standard math module, set when a
# model module that lowers such a call is imported. A C function runs only while the Reference is that
# module and only where the C library gives CPython's result (CPython's math calls the same functions
# there); otherwise the Reference's own member runs, so its errors, NaN handling and results are live's.
cdef extern from *:
    """
    static void *_mx_stdlib_math = NULL;
    """
    void *_mx_stdlib_math
from libc.math cimport erf as _mx_c_erf, exp as _mx_c_exp, log as _mx_c_log, sqrt as _mx_c_sqrt
from libc.math cimport isinf as _mx_c_isinf
from cpython.float cimport PyFloat_CheckExact, PyFloat_AS_DOUBLE
from cpython.ref cimport Py_INCREF


cdef inline int _mx_math_init() except -1:
    global _mx_stdlib_math
    import math
    Py_INCREF(math)             # held for the process, as sys.modules holds it
    _mx_stdlib_math = <void*>math
    return 0


cdef inline double _mx_math_value(object value, str name, str path) except? -1.0:
    """The float a rebound Reference's member returned; anything else raises naming the Cell."""
    if PyFloat_CheckExact(value):
        return PyFloat_AS_DOUBLE(value)
    raise _mx_named(TypeError(f"{path}: {name}() of the Reference returned {type(value).__name__}, "
                              f"not float; the compiled Cell computes a float"), path)


ctypedef fused _mx_math_arg:
    double
    long long
    object


cdef inline bint _mx_math_real(_mx_math_arg x, double* out) noexcept:
    """The double math computes with: a C number, or an exact float (any other object: math's own call)."""
    if _mx_math_arg is object:
        if PyFloat_CheckExact(x):
            out[0] = PyFloat_AS_DOUBLE(x)
            return True
        return False
    else:
        out[0] = <double>x      # as PyLong_AsDouble rounds an int of this range
        return True


cdef inline double _mx_math_log(object module, _mx_math_arg x, str path) except? -1.0:
    cdef double v
    if <void*>module == _mx_stdlib_math and _mx_math_real(x, &v) and v > 0.0:
        return _mx_c_log(v)     # v <= 0 raises and NaN returns NaN in math's own call
    return _mx_math_value(module.log(x), 'log', path)


cdef inline double _mx_math_exp(object module, _mx_math_arg x, str path) except? -1.0:
    cdef double v, result
    if <void*>module == _mx_stdlib_math and _mx_math_real(x, &v):
        result = _mx_c_exp(v)
        if _mx_c_isfinite(result):  # overflow raises, inf and NaN: math's own call
            return result
    return _mx_math_value(module.exp(x), 'exp', path)


cdef inline double _mx_math_sqrt(object module, _mx_math_arg x, str path) except? -1.0:
    cdef double v
    if <void*>module == _mx_stdlib_math and _mx_math_real(x, &v) and v >= 0.0:
        return _mx_c_sqrt(v)    # v < 0 raises and NaN returns NaN in math's own call
    return _mx_math_value(module.sqrt(x), 'sqrt', path)


cdef inline double _mx_math_erf(object module, _mx_math_arg x, str path) except? -1.0:
    cdef double v
    if <void*>module == _mx_stdlib_math and _mx_math_real(x, &v):
        return _mx_c_erf(v)     # math.erf is the C library's erf
    return _mx_math_value(module.erf(x), 'erf', path)


cdef inline bint _mx_math_isinf(object module, _mx_math_arg x, str path) except -1:
    cdef double v
    if <void*>module == _mx_stdlib_math and _mx_math_real(x, &v):
        return _mx_c_isinf(v) != 0
    value = module.isinf(x)
    if value is True or value is False:
        return value
    raise _mx_named(TypeError(f"{path}: isinf() of the Reference returned {type(value).__name__}, "
                              f"not bool; the compiled Cell computes a bool"), path)


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
