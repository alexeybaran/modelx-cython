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

With ``-DMX_MEMCHECK`` (the ``MXC_MEMCHECK=1`` build) the value ranges are marked undefined for
valgrind memcheck, so a read of an unset slot is reported even on recycled memory.
"""
import os

from .consts import HAS_PREF, VAR_PREF

_ELEMENT = {'double': 'double', 'long long': 'longlong', 'bint': 'bint', 'short': 'short'}


def enabled():
    return os.environ.get('MXC_LAZY_VALUES', '1') == '1'


def dense_cells(cls_info):
    return [cells for cells in cls_info.cells.values()
            if not cells.is_special() and cells.has_formula_def and cells.has_args()
            and cells.has_typeinfo() and cells.is_arrayable()]


def classes(module):
    if not enabled():
        return {}
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
static Py_ssize_t mx_ranges_{cls}[{n2}];
static Py_ssize_t mx_count_{cls} = 0;
static PyTypeObject *mx_type_{cls} = NULL;
static PyObject *mx_alloc_{cls}(PyTypeObject *t, Py_ssize_t n) {{
    return mx_alloc_values(t, n, mx_type_{cls}, mx_ranges_{cls}, mx_count_{cls});
}}
static void mx_set_range_{cls}(Py_ssize_t offset, Py_ssize_t size) {{
    mx_ranges_{cls}[2 * mx_count_{cls}] = offset;
    mx_ranges_{cls}[2 * mx_count_{cls} + 1] = size;
    mx_count_{cls} += 1;
}}
static void mx_install_{cls}(void *t) {{
    mx_sort_ranges(mx_ranges_{cls}, mx_count_{cls});
    mx_type_{cls} = (PyTypeObject *)t;
    ((PyTypeObject *)t)->tp_alloc = mx_alloc_{cls};
}}
#endif
'''


def _tag(module, cls):
    # C names are global to every module that cimports this .pxd: qualify them with the module
    return module.fqname.replace('.', '_') + '__' + cls


def pxd_declarations(module):
    """The allocator helpers, as a verbatim C block of the module's .pxd."""
    owners = classes(module)
    if not owners:
        return ''
    code = _HELPER + ''.join(_PER_CLASS.format(cls=_tag(module, cls), n2=2 * len(cells))
                             for cls, cells in owners.items())
    lines = ['cdef extern from *:', '    """'] + ['    ' + line for line in code.strip('\n').splitlines()] + ['    """']
    for cls in owners:
        lines += [f'    void mx_set_range_{_tag(module, cls)}(Py_ssize_t offset, Py_ssize_t size)',
                  f'    void mx_install_{_tag(module, cls)}(void *t)']
    return '\n'.join(lines) + '\n'


def module_tail(module):
    """Install the allocators and the pickling sanitizers (module source appended to the .py)."""
    owners = classes(module)
    if not owners:
        return ''
    out = ['', '', 'import copyreg as _mx_copyreg', '']
    for cls, cells in owners.items():
        out += ['@_mx_cy.cfunc', f'def _mx_install_values_{cls}() -> None:',
                f'    o: {cls} = _mx_cy.cast({cls}, None)',
                '    base: _mx_cy.Py_ssize_t = _mx_cy.cast(_mx_cy.Py_ssize_t, _mx_cy.cast(_mx_cy.p_void, o))']
        for c in cells:
            v = VAR_PREF + c.name
            out.append(f'    mx_set_range_{_tag(module, cls)}(_mx_cy.cast(_mx_cy.Py_ssize_t, _mx_cy.address(o.{v})) - base, '
                       f'_mx_cy.sizeof(o.{v}))')
        out += [f'    mx_install_{_tag(module, cls)}(_mx_cy.cast(_mx_cy.p_void, {cls}))', '', '']
        out += ['@_mx_cy.cfunc', f'def _mx_zero_unset_{cls}(self: {cls}) -> None:', '    i: _mx_cy.Py_ssize_t']
        for k, c in enumerate(cells):
            value = _ELEMENT[c.get_rettype_expr(c_style=True)]
            flag = _ELEMENT['bint' if c.is_locked else 'short']
            v, h = VAR_PREF + c.name, HAS_PREF + c.name
            out += [f'    v{k}: _mx_cy.pointer(_mx_cy.{value}) = _mx_cy.cast(_mx_cy.pointer(_mx_cy.{value}), '
                    f'_mx_cy.address(self.{v}))',
                    f'    h{k}: _mx_cy.pointer(_mx_cy.{flag}) = _mx_cy.cast(_mx_cy.pointer(_mx_cy.{flag}), '
                    f'_mx_cy.address(self.{h}))',
                    f'    for i in range(_mx_cy.sizeof(self.{v}) // _mx_cy.sizeof(v{k}[0])):',
                    f'        if not h{k}[i]:',
                    f'            v{k}[i] = 0']
        out += ['', '', f'def _mx_reduce_{cls}(self):',
                '    # unset value slots are not initialised: write zero to them first, so the state',
                "    # emitted by the class's own pickling equals the generic allocation's",
                f'    _mx_zero_unset_{cls}(_mx_cy.cast({cls}, self))',
                '    return self.__reduce_cython__()', '', '',
                f'_mx_install_values_{cls}()', f'_mx_copyreg.pickle({cls}, _mx_reduce_{cls})', '']
    return '\n'.join(out) + '\n'
