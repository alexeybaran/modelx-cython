"""Prototype (direct-route diagnosis, approved as D-gap A'): allocate a compiled class without zeroing
its dense value arrays.

CPython's generic allocation zeroes a whole object, so every page of an item's in-struct dense caches
is written (and on fresh memory faulted) when the item is created, although a policy writes only part
of them. A dense value slot is read only after its readiness flag is set, and the flag is set only after
the value is written (the cache entries, the exposed hits, the in-bounds entries and the locked loads all
test the flag first). So the value arrays need no zeroing; everything else does.

Each such class gets its own ``tp_alloc``, installed when the module is imported: it allocates with
``PyObject_GC_New`` (no zeroing), zeroes every byte of the object except the value arrays, and tracks
the object, as ``PyType_GenericAlloc`` does. Both routes construct through the same ``tp_new``, so the
collect root and the direct items share it. A subclass (another exact type) keeps the generic
allocation.

The one reader of whole value arrays is Cython's automatic pickling (``__reduce_cython__``), used by
``pickle`` and ``copy``. A reducer registered in ``copyreg`` for the class first writes zero to every
unset slot, then returns the class's own ``__reduce_cython__``: the emitted state equals the current one.

With ``-DMX_MEMCHECK`` the value ranges are marked undefined for valgrind memcheck, so a read of an
unset slot is reported even on recycled memory.

Basis (CPython 3.14.6, the free-threaded build included), for the classes this applies to: static
(not heap) types, ``Py_TPFLAGS_HAVE_GC``, ``tp_itemsize`` 0, no ``Py_TPFLAGS_MANAGED_DICT`` /
``MANAGED_WEAKREF`` (so no pre-header) and no ``Py_TPFLAGS_INLINE_VALUES``; ``classes`` excludes the
classes with a ``__dict__`` and the allocator falls back to ``PyType_GenericAlloc`` for any other type:

- Doc/c-api/gcsupport.rst: a container's memory "must be allocated using PyObject_GC_New", and
  "Once all the fields which may contain references to other containers are initialized, it must call
  PyObject_GC_Track"; when populating ``tp_alloc``, ``PyType_GenericAlloc`` "is preferred over a
  custom function that simply calls this macro" (a custom function is allowed). Memory is freed with
  ``PyObject_GC_Del``, the ``tp_free`` these types inherit.
- Objects/typeobject.c ``PyType_GenericAlloc`` -> ``_PyType_AllocNoTrack``: ``_PyObject_MallocWithType``,
  ``_PyObject_GC_Link``, ``memset`` of everything after the ``PyObject`` header, ``_PyObject_Init``,
  then ``_PyObject_GC_TRACK``. Python/gc_free_threading.c ``_PyObject_GC_New``: ``gc_alloc``
  (``_PyObject_MallocWithType`` and ``record_allocation``, which is all ``_PyObject_GC_Link`` does
  there), then ``_PyObject_Init``. For these types the two differ only in the body ``memset``, which
  this allocator does itself except for the value arrays; ``PyObject_GC_Track`` is
  ``_PyObject_GC_TRACK`` after an already-tracked check.
- The header is initialised by the API, not here: ``_PyObject_Init`` sets ``ob_type`` (the type
  reference is immortal for a static type, ``_Py_INCREF_TYPE``) and ``_Py_NewReference`` sets
  ``ob_flags``, ``ob_mutex``, ``ob_tid``, ``ob_gc_bits``, ``ob_ref_local`` and ``ob_ref_shared``
  (Objects/object.c ``new_reference``); neither path zeroes the header.
- Doc/c-api/typeobj.rst ``tp_alloc``: static subtypes inherit the slot (they take the generic path here,
  being another exact type); heap subtypes get ``PyType_GenericAlloc``.
"""
from .consts import HAS_PREF, VAR_PREF

_ELEMENT = {'double': 'double', 'long long': 'longlong', 'bint': 'bint', 'short': 'short'}


def dense_cells(cls_info):
    return [cells for cells in cls_info.cells.values()
            if not cells.is_special() and cells.has_formula_def and cells.has_args()
            and cells.has_typeinfo() and cells.is_arrayable()]


def classes(module):
    return {name: cells for name, info in module.classes.items()
            if name not in module.visitor.dynamic_ref_classes and (cells := dense_cells(info))}


_HELPER = r'''
#ifndef MX_ALLOC_VALUES_HELPER
#define MX_ALLOC_VALUES_HELPER
#include <string.h>
#ifdef MX_MEMCHECK
#include <valgrind/memcheck.h>
#define MX_UNDEFINED(p, n) VALGRIND_MAKE_MEM_UNDEFINED((p), (n))
#else
#define MX_UNDEFINED(p, n) ((void)0)
#endif
static PyObject *mx_alloc_values(PyTypeObject *t, Py_ssize_t n, PyTypeObject *exact,
                                 const Py_ssize_t *r, Py_ssize_t count) {
    if (t != exact || n != 0 || !PyType_IS_GC(t) || count == 0)
        return PyType_GenericAlloc(t, n);
    PyObject *o = PyObject_GC_New(PyObject, t);
    if (o == NULL)
        return NULL;
    char *p = (char *)o;
    Py_ssize_t pos = (Py_ssize_t)sizeof(PyObject), end = t->tp_basicsize, i;
    for (i = 0; i < count; i++) {
        if (r[2 * i] > pos)
            memset(p + pos, 0, (size_t)(r[2 * i] - pos));
        MX_UNDEFINED(p + r[2 * i], (size_t)r[2 * i + 1]);
        if (r[2 * i] + r[2 * i + 1] > pos)
            pos = r[2 * i] + r[2 * i + 1];
    }
    if (end > pos)
        memset(p + pos, 0, (size_t)(end - pos));
    PyObject_GC_Track(o);
    return o;
}
static int mx_type_kind_ok(PyTypeObject *tp) {
    unsigned long excluded = Py_TPFLAGS_HEAPTYPE;
#ifdef Py_TPFLAGS_MANAGED_DICT
    excluded |= Py_TPFLAGS_MANAGED_DICT;
#endif
#ifdef Py_TPFLAGS_MANAGED_WEAKREF
    excluded |= Py_TPFLAGS_MANAGED_WEAKREF;
#endif
#ifdef Py_TPFLAGS_INLINE_VALUES
    excluded |= Py_TPFLAGS_INLINE_VALUES;
#endif
    return PyType_IS_GC(tp) && tp->tp_itemsize == 0 && !(tp->tp_flags & excluded)
           && tp->tp_alloc == PyType_GenericAlloc && tp->tp_free == PyObject_GC_Del;
}
static void mx_sort_ranges(Py_ssize_t *r, Py_ssize_t count) {
    Py_ssize_t i, j, a, b;
    for (i = 1; i < count; i++) {
        a = r[2 * i]; b = r[2 * i + 1];
        for (j = i; j > 0 && r[2 * (j - 1)] > a; j--) {
            r[2 * j] = r[2 * (j - 1)]; r[2 * j + 1] = r[2 * (j - 1) + 1];
        }
        r[2 * j] = a; r[2 * j + 1] = b;
    }
}
#endif
'''

_PER_CLASS = r'''
#ifndef MX_ALLOC_{cls}
#define MX_ALLOC_{cls}
static Py_ssize_t mx_ranges_{cls}[{n2}];              /* value (offset, size), sorted at install */
static Py_ssize_t mx_cells_{cls}[{n5}];               /* value offset, size, element; flag offset, element */
static Py_ssize_t mx_count_{cls} = 0;
static PyTypeObject *mx_type_{cls} = NULL;
static PyObject *mx_alloc_{cls}(PyTypeObject *t, Py_ssize_t n) {{
    return mx_alloc_values(t, n, mx_type_{cls}, mx_ranges_{cls}, mx_count_{cls});
}}
static void mx_set_range_{cls}(Py_ssize_t value, Py_ssize_t size, Py_ssize_t element,
                               Py_ssize_t flag, Py_ssize_t flag_element) {{
    Py_ssize_t *c = mx_cells_{cls} + 5 * mx_count_{cls};
    mx_ranges_{cls}[2 * mx_count_{cls}] = value;
    mx_ranges_{cls}[2 * mx_count_{cls} + 1] = size;
    c[0] = value; c[1] = size; c[2] = element; c[3] = flag; c[4] = flag_element;
    mx_count_{cls} += 1;
}}
/* Called first in the class's generated __reduce_cython__ (inserted by build.py), which every pickling
   and copying entry point reaches: an unset value slot is written zero, in place, so the state emitted
   equals the generic allocation's. A subclass reaching the same function was zeroed at allocation. */
static void mx_zero_unset_{cls}(PyObject *o) {{
    char *p = (char *)o;
    Py_ssize_t i, j;
    for (i = 0; i < mx_count_{cls}; i++) {{
        const Py_ssize_t *c = mx_cells_{cls} + 5 * i;
        Py_ssize_t n = c[1] / c[2];
        for (j = 0; j < n; j++) {{
            int ready = c[4] == (Py_ssize_t)sizeof(short) ? ((short *)(p + c[3]))[j] != 0
                                                        : ((int *)(p + c[3]))[j] != 0;
            if (!ready)
                memset(p + c[0] + j * c[2], 0, (size_t)c[2]);
        }}
    }}
}}
static int mx_install_{cls}(void *t) {{
    PyTypeObject *tp = (PyTypeObject *)t;
    mx_sort_ranges(mx_ranges_{cls}, mx_count_{cls});
    /* the basis in lazy_values.py holds only for this type kind; any other keeps the generic path */
    if (!mx_type_kind_ok(tp))
        return 0;
    mx_type_{cls} = tp;
    tp->tp_alloc = mx_alloc_{cls};
    return 1;
}}
#endif
'''

SEPARATOR = '_mxv_'


def _tag(module, cls):
    # C names are global to every module that cimports this .pxd: qualify them with the module
    return module.fqname.replace('.', '_') + SEPARATOR + cls


def pxd_declarations(module):
    """The allocator helpers, as a verbatim C block of the module's .pxd."""
    owners = classes(module)
    if not owners:
        return ''
    code = _HELPER + ''.join(_PER_CLASS.format(cls=_tag(module, cls), n2=2 * len(cells), n5=5 * len(cells))
                             for cls, cells in owners.items())
    lines = ['cdef extern from *:', '    """'] + ['    ' + line for line in code.strip('\n').splitlines()] + ['    """']
    for cls in owners:
        lines += [f'    void mx_set_range_{_tag(module, cls)}(Py_ssize_t value, Py_ssize_t size, Py_ssize_t element, '
                  'Py_ssize_t flag, Py_ssize_t flag_element)',
                  f'    int mx_install_{_tag(module, cls)}(void *t)']
    return '\n'.join(lines) + '\n'


def module_tail(module):
    """Install the allocators (module source appended to the .py). ``_mx_lazy_values_installed`` and
    ``_mx_lazy_values_layout`` (per cell: value offset, size, element size, flag offset, flag element
    size) describe what was installed."""
    owners = classes(module)
    if not owners:
        return ''
    out = ['', '', '_mx_lazy_values_installed = {}', '_mx_lazy_values_layout = {}', '']
    for cls, cells in owners.items():
        tag = _tag(module, cls)
        out += ['@_mx_cy.cfunc', f'def _mx_install_values_{cls}() -> _mx_cy.bint:',
                f'    o: {cls} = _mx_cy.cast({cls}, None)',
                '    base: _mx_cy.Py_ssize_t = _mx_cy.cast(_mx_cy.Py_ssize_t, _mx_cy.cast(_mx_cy.p_void, o))',
                '    layout = []',
                '    value: _mx_cy.Py_ssize_t', '    flag: _mx_cy.Py_ssize_t']
        for c in cells:
            v, h = VAR_PREF + c.name, HAS_PREF + c.name
            out += [f'    value = _mx_cy.cast(_mx_cy.Py_ssize_t, _mx_cy.address(o.{v})) - base',
                    f'    flag = _mx_cy.cast(_mx_cy.Py_ssize_t, _mx_cy.address(o.{h})) - base',
                    f'    mx_set_range_{tag}(value, _mx_cy.sizeof(o.{v}), _mx_cy.sizeof(o.{v}[0]) // '
                    f'({_mx_count(c)}), flag, _mx_cy.sizeof(o.{h}[0]) // ({_mx_count(c)}))',
                    f"    layout.append(('{c.name}', value, _mx_cy.sizeof(o.{v}), _mx_cy.sizeof(o.{v}[0]) // "
                    f"({_mx_count(c)}), flag, _mx_cy.sizeof(o.{h}[0]) // ({_mx_count(c)})))"]
        out += [f"    _mx_lazy_values_layout['{cls}'] = layout",
                f'    return mx_install_{tag}(_mx_cy.cast(_mx_cy.p_void, {cls})) != 0', '', '',
                f"_mx_lazy_values_installed['{cls}'] = _mx_install_values_{cls}()", '']
    return '\n'.join(out) + '\n'


def _mx_count(cells):
    """Elements in one row of the cell's array (its axes after the first): sizeof(a[0]) // this is the
    element size."""
    count = 1
    for axis in cells.dense_axes[1:]:
        count *= axis.stop - axis.lower
    return count


def patch_reduce(path):
    """Insert the zeroing call first in the generated ``__reduce_cython__`` of each class whose
    allocator this C file installs; fail when the generated function is not found exactly once."""
    import re
    from pathlib import Path
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    # a class's allocator is installed (called) only by its own module; the cimporting modules carry
    # the definition alone
    tags = sorted({t for t in re.findall(r'mx_install_(\w+)\((?!void \*t\))', text) if SEPARATOR in t})
    changed = False
    for tag in tags:
        cls = tag.rsplit(SEPARATOR, 1)[1]
        call = f'mx_zero_unset_{tag}((PyObject *)__pyx_v_self);'
        if call in text:
            continue
        pattern = re.compile(r'^(static PyObject \*__pyx_pf_\w*?' + str(len(cls)) + re.escape(cls)
                             + r'_\d+__reduce_cython__\(struct __pyx_obj_\w*?_' + re.escape(cls)
                             + r' \*__pyx_v_self\) \{)$', re.M)
        found = pattern.findall(text)
        if len(found) != 1:
            raise RuntimeError(
                f'{path.name}: lazy_values expected exactly one generated __reduce_cython__ definition for '
                f'{cls}, found {len(found)}; the generated C differs from the supported Cython versions '
                f'(3.2.4, 3.3.1), so unset value slots could be pickled uninitialised')
        text = pattern.sub(lambda m: m.group(1) + '\n  ' + call, text, count=1)
        changed = True
    if changed:
        path.write_text(text, encoding='utf-8')
    return tags
