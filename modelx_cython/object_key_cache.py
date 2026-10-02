"""Native lookup indexes over authoritative object-key Cell dictionaries.

Only ordinary integer hash/equality domains qualify at runtime. The dictionary
remains authoritative, including its historical stored hashes if a key class
changes. Native entries borrow the dictionary's actual result objects; they do
not own a second Cell result. Unsupported accesses permanently disable the index.
"""
import ast
import copy
import re
import textwrap
from .numeric_cache import support, TYPES


GUARDS = r'''
cdef extern from *:
    """
    static CYTHON_INLINE int __mx_standard_integer_type(PyObject *type) {
        PyTypeObject *t = (PyTypeObject *)type;
        return PyType_IsSubtype(t, &PyLong_Type) &&
            t->tp_hash == PyLong_Type.tp_hash &&
            t->tp_richcompare == PyLong_Type.tp_richcompare;
    }
    static CYTHON_INLINE int __mx_integer_key(PyObject *value, long long *out) {
        if (!PyLong_Check(value) || !__mx_standard_integer_type((PyObject *)Py_TYPE(value))) return 0;
        int overflow = 0;
        *out = PyLong_AsLongLongAndOverflow(value, &overflow);
        if (PyErr_Occurred()) return -1;
        return !overflow;
    }
    static CYTHON_INLINE int __mx_integer_standard(PyObject *value) {
        return PyLong_Check(value) && __mx_standard_integer_type((PyObject *)Py_TYPE(value));
    }
    """
    bint _mx_standard_integer_type(object value) noexcept
    bint _mx_integer_key(object value, long long* out) except -1
    bint _mx_integer_standard "__mx_integer_standard" (object value) noexcept
'''.replace('bint _mx_standard_integer_type(object', 'bint _mx_standard_integer_type "__mx_standard_integer_type" (object').replace(
    'bint _mx_integer_key(object', 'bint _mx_integer_key "__mx_integer_key" (object')


def expr(text):
    return ast.parse(text, mode='eval').body


def owner_source(tag, types):
    generated = 'cimport cython\n' + support(tag, types, 'PyObject*')
    generated = generated[:generated.index(f'cdef inline dict {tag}_dictionary')]
    generated += f'''
cdef class {tag}_owner:
    cdef {tag}_cache data
    cdef readonly dict canonical
    cdef list guard_types
    cdef bint valid(self) noexcept
    cdef void remember_type(self, object value) except *
'''
    implementation = f'''
@_mx_cy.cclass
class {tag}_owner:

    def __init__(self, canonical: dict, guard_types=None):
        self.canonical = canonical
        self.guard_types = [] if guard_types is None else list(guard_types)

    def __dealloc__(self):
        _mx_free(self.data.items)

    def valid(self):
        for kind in self.guard_types:
            if not _mx_standard_integer_type(kind):
                return False
        return True

    def remember_type(self, value):
        kind = type(value)
        for known in self.guard_types:
            if known is kind:
                return
        self.guard_types.append(kind)

    def __reduce__(self):
        # Keep the original dictionary, including its key objects. Rebuild only
        # acceleration entries on later accesses. Space pickle memo shares it.
        return ({tag}_owner, (self.canonical, self.guard_types))
'''
    return generated, implementation


def transform(source, declarations, module, path):
    tree = ast.parse(source)
    edits, support_parts, implementations, imports, inventory = [], [], [], [], []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in module.classes:
            continue
        info = module.classes[cls.name]
        if info.is_locked:
            continue
        fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        field_types = {n.target.id:ast.unparse(n.annotation) for n in cls.body
                       if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
        # This is a connected-execution optimization. Public-only object-key
        # paths can have predominantly cold misses, paying for both a dictionary
        # insertion and derived index without receiving a propagated native key.
        # Keep their ordinary implementation; no usage or key domain is banned.
        connected_cells=set()
        for caller in cls.body:
            if not isinstance(caller,ast.FunctionDef) or not caller.name.startswith('_f_'):continue
            for call in ast.walk(caller):
                if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                        and ast.unparse(call.func.value)=='self' and not call.keywords):continue
                for value in call.args:
                    if (isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute)
                            and ast.unparse(value.func.value)=='self' and not value.args and not value.keywords
                            and '_has_'+value.func.attr in fields
                            and field_types.get('_v_'+value.func.attr)=='object'):
                        connected_cells.add(call.func.attr)
        added = []
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name not in info.cells or fn.name not in connected_cells:
                continue
            cell = info.cells[fn.name]
            args = fn.args
            if (not cell.has_formula_def or cell.is_locked or cell.formula_is_generator
                    or args.defaults or args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs
                    or not fn.returns or ast.unparse(fn.returns) not in TYPES
                    or not any(ast.unparse(d) == '_mx_cy.ccall' for d in fn.decorator_list)
                    or '_v_' + fn.name not in fields):
                continue
            types = [ast.unparse(a.annotation) if a.annotation else 'object' for a in args.args[1:]]
            if 'object' not in types or any(t not in {*TYPES, 'object'} for t in types):
                continue
            # Only the existing dictionary wrapper qualifies, never an object
            # array, uncached formula, locked method or arbitrary user method.
            if not any(isinstance(n, ast.Dict) and not n.keys for n in ast.walk(fn)):
                continue
            name = fn.name
            tag = '_mx_oi' + cls.name + '_' + name
            native_types = ['long long' if t == 'object' else TYPES[t] for t in types]
            support_part, implementation = owner_source(tag, native_types)
            support_parts.append(support_part)
            implementations.append(implementation)
            imports += [tag+'_owner', tag+'_find', tag+'_put']
            added += [f'    cdef {tag}_owner {tag}', f'    cdef bint {tag}_disabled']
            a_names = [a.arg for a in args.args[1:]]
            keys = [f'_mx_oi_k{i}' if t == 'object' else a for i,(a,t) in enumerate(zip(a_names,types))]
            checks = [f'_mx_integer_key({a}, _mx_cy.address({k}))' if t == 'object'
                      else f'_mx_oi_isfinite({a})' for a,t,k in zip(a_names,types,keys) if t in ('object','_mx_cy.double')]
            check = ' and '.join(checks)
            cache = 'self._v_' + name
            key = a_names[0] if len(a_names)==1 else '(' + ', '.join(a_names) + ')'
            joined = ', '.join(keys)
            prefix = '\n'.join(f'_mx_oi_k{i}: _mx_cy.longlong' for i,t in enumerate(types) if t=='object')
            prefix += f'''
_mx_oi_pos: _mx_cy.Py_ssize_t
_mx_oi_value: {ast.unparse(fn.returns)}
_mx_oi_owner: {tag}_owner
if {cache} is None:
    {cache} = {{}}
if not self.{tag}_disabled:
    if not ({check}):
        self.{tag}_disabled = True
        self.{tag} = None
    else:
        if self.{tag} is None:
            self.{tag} = {tag}_owner({cache})
        _mx_oi_owner = self.{tag}
'''
            prefix += ''.join(f'        _mx_oi_owner.remember_type({a})\n' for a,t in zip(a_names,types) if t=='object')
            prefix += f'''        if self.{tag}_disabled or self.{tag} is not _mx_oi_owner or _mx_oi_owner.canonical is not {cache} or not _mx_oi_owner.valid():
            self.{tag}_disabled = True
            self.{tag} = None
        else:
            _mx_oi_pos = {tag}_find(_mx_cy.address(_mx_oi_owner.data), {joined})
            if _mx_oi_pos >= 0:
                return _mx_cy.cast(object, _mx_oi_owner.data.items[_mx_oi_pos].value)
'''
            # On an index miss use the unchanged dictionary/formula path. Its
            # result is authoritative; index maintenance may only borrow it.
            cache_result = f'''if not self.{tag}_disabled and self.{tag} is not None:
    _mx_oi_owner = self.{tag}
    _mx_oi_key = {key}
    if not self.{tag}_disabled and self.{tag} is _mx_oi_owner and _mx_oi_owner.canonical is {cache} and _mx_oi_owner.valid() and ({check}):
        _mx_oi_boxed = {cache}.get(_mx_oi_key, _mx_oi_missing)
        if _mx_oi_boxed is not _mx_oi_missing and not self.{tag}_disabled and self.{tag} is _mx_oi_owner and _mx_oi_owner.canonical is {cache} and _mx_oi_owner.valid():
            {tag}_put(_mx_cy.address(_mx_oi_owner.data), {joined}, _mx_cy.cast(_mx_oi_PyObjectPtr, _mx_oi_boxed))
    else:
        self.{tag}_disabled = True
        self.{tag} = None
'''
            class Returns(ast.NodeTransformer):
                def visit_Return(self, node):
                    return [ast.Assign([ast.Name('_mx_oi_value', ast.Store())], node.value),
                            *ast.parse(cache_result).body, ast.Return(expr('_mx_oi_value'))]
            rewritten = Returns().visit(copy.deepcopy(fn))
            rewritten.body = ast.parse(prefix).body + rewritten.body
            ast.fix_missing_locations(rewritten)
            start = min([fn.lineno]+[d.lineno for d in fn.decorator_list])-1
            edits.append((start, fn.end_lineno, textwrap.indent(ast.unparse(rewritten),'    ')+'\n'))
            inventory.append({'class':cls.name,'cell':name,'object_arguments':[a for a,t in zip(a_names,types) if t=='object'],
                              'cache':'authoritative dictionary; borrowed-value native integer-key index; guarded permanent fallback'})
        if added:
            marker = next(s for s in declarations.splitlines() if s.startswith(f'cdef class {cls.name}('))
            declarations = declarations.replace(marker, marker+'\n'+'\n'.join(added),1)
    if not edits:
        return source, declarations
    # Definitions are in an include, giving the caller Cython direct access to
    # the native data fields while owner references remain ordinary GC objects.
    support_text = '''from libc.stdlib cimport calloc, free
from libc.stdint cimport uint64_t
from cpython.ref cimport PyObject
ctypedef PyObject* _mx_oi_PyObjectPtr
cdef union _mx_bits:
    double d
    uint64_t u
'''+GUARDS+'\n'+'\n'.join(support_parts)
    path.with_name('_mx_object_index.pxd').write_text(support_text,encoding='utf-8')
    path.with_name('_mx_object_index.py').write_text(
        'import cython as _mx_cy\nfrom cython.cimports.libc.stdlib import free as _mx_free\n'
        + '\n'.join(implementations), encoding='utf-8')
    package = module.fqname.rsplit('.',1)[0]
    names = ['_mx_integer_key','_mx_integer_standard','_mx_oi_PyObjectPtr',*imports]
    source_lines = source.splitlines(keepends=True)
    for start,end,value in sorted(edits,reverse=True):
        source_lines[start:end] = [value]
    source = ''.join(source_lines)
    source = (f'from cython.cimports.{package}._mx_object_index import '+', '.join(names)+'\n'
              'from cython.cimports.libc.math import isfinite as _mx_oi_isfinite\n'
              '_mx_oi_missing = object()\n'+source)
    declarations = f'from {package}._mx_object_index cimport '+', '.join(imports)+'\n'+declarations
    path.write_text(__import__('json').dumps(inventory,indent=2),encoding='utf-8')
    return source, declarations
