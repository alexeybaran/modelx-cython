"""Development-only coherent native execution, beneath existing Space APIs.

No formulas are interpreted here.  This pass lowers already-typed generated
Cython and preserves one implementation of each formula and one result cache.
"""
import ast
import copy
import json
import re
import textwrap

NUMERIC = {'_mx_cy.longlong': 'long long', '_mx_cy.double': 'double', '_mx_cy.bint': 'bint'}
OPAQUE = (ast.Lambda, ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp,
          ast.Yield, ast.YieldFrom, ast.AsyncFunctionDef, ast.ClassDef)

ENUM_COMPARE_SUPPORT = '''cdef extern from *:
    """
    static CYTHON_INLINE int __mx_native_enum_equal(PyObject *a, PyObject *b, int invert) {
        if (PyLong_Check(a) && PyLong_Check(b) &&
            Py_TYPE(a)->tp_richcompare == PyLong_Type.tp_richcompare &&
            Py_TYPE(b)->tp_richcompare == PyLong_Type.tp_richcompare) {
            int oa = 0, ob = 0;
            long long av = PyLong_AsLongLongAndOverflow(a, &oa);
            if (PyErr_Occurred()) return -1;
            long long bv = PyLong_AsLongLongAndOverflow(b, &ob);
            if (PyErr_Occurred()) return -1;
            if (!oa && !ob) return invert ? av != bv : av == bv;
        }
        /* Do not use RichCompareBool: its identity shortcut changes custom ==. */
        PyObject *r = PyObject_RichCompare(a, b, invert ? Py_NE : Py_EQ);
        if (r == NULL) return -1;
        int result = PyObject_IsTrue(r);
        Py_DECREF(r);
        return result;
    }
    """
    bint _mx_enum_equal "__mx_native_enum_equal" (object a, object b, bint invert) except -1
'''


class EnumTests(ast.NodeTransformer):
    """Native integer comparison only at truth-use sites, with dynamic slot guards."""
    def __init__(self):
        self.count = 0

    def truth(self, n):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Eq, ast.NotEq)):
            if any(isinstance(x, ast.Attribute) and x.attr.startswith('_mx_enum_member_') for x in ast.walk(n)):
                self.count += 1
                return ast.Call(expr('_mx_enum_equal'), [n.left, n.comparators[0], expr(str(isinstance(n.ops[0], ast.NotEq)))], [])
        if isinstance(n, ast.BoolOp):
            n.values = [self.truth(v) for v in n.values]
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            n.operand = self.truth(n.operand)
        return n

    def visit_If(self, n):
        n = self.generic_visit(n); n.test = self.truth(n.test); return n

    def visit_IfExp(self, n):
        n = self.generic_visit(n); n.test = self.truth(n.test); return n

    def visit_While(self, n):
        n = self.generic_visit(n); n.test = self.truth(n.test); return n


def expr(s):
    return ast.parse(s, mode='eval').body


def typed(fn):
    return ast.unparse(fn.returns) if fn.returns else 'object'


def simple(fn, objects=False):
    a = fn.args
    return (not (a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults)
            and all(x.annotation and (ast.unparse(x.annotation) in NUMERIC or
                                     objects and ast.unparse(x.annotation) == 'object') for x in a.args[1:])
            and not any(isinstance(n, OPAQUE) or (isinstance(n, ast.FunctionDef) and n is not fn)
                        or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id in {'locals', 'vars', 'eval', 'exec', 'dir'})
                        for n in ast.walk(fn)))


def local_indices(fn, methods):
    """Only native-bound ranges with index uses already crossing a native ABI.

No index arithmetic, escaping values, shadowed range, or post-loop reads.
Those exclusions avoid changing Python integer overflow/unbound behavior.
"""
    if any(isinstance(n, ast.Name) and n.id == 'range' and isinstance(n.ctx, ast.Store)
           for n in ast.walk(fn)) or any(a.arg == 'range' for a in fn.args.args):
        return []
    parents = {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
    result = []
    for loop in ast.walk(fn):
        if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name):
            continue
        call = loop.iter
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == 'range' and len(call.args) == 1 and not call.keywords):
            continue
        bound = call.args[0]
        if not (isinstance(bound, ast.Call) and not bound.args and not bound.keywords
                and isinstance(bound.func, ast.Attribute) and ast.unparse(bound.func.value) == 'self'
                and bound.func.attr in methods and typed(methods[bound.func.attr]) == '_mx_cy.longlong'):
            continue
        name = loop.target.id
        nodes = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == name]
        body_nodes = {n for stmt in loop.body for n in ast.walk(stmt)}
        ok = True
        for n in nodes:
            if n is loop.target:
                continue
            p = parents.get(n)
            if (not isinstance(n.ctx, ast.Load) or n not in body_nodes or not isinstance(p, ast.Call)
                    or p.keywords or not isinstance(p.func, ast.Attribute)
                    or ast.unparse(p.func.value) != 'self' or p.func.attr not in methods
                    or n not in p.args):
                ok = False; break
            args = methods[p.func.attr].args.args[1:]
            ix = p.args.index(n)
            if ix >= len(args) or not args[ix].annotation or ast.unparse(args[ix].annotation) != '_mx_cy.longlong':
                ok = False; break
        if ok:
            result.append(name)
    return result


class Edges(ast.NodeTransformer):
    def __init__(self, cls, selected, native_keys=None, indexed=(), compact=(),
                 compact_argument='_mx_owner', compact_self=True):
        self.cls, self.selected = cls, selected
        self.native_keys,self.indexed=native_keys,indexed
        self.compact = compact
        self.compact_argument = compact_argument
        self.compact_self = compact_self

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and ast.unparse(node.func.value) == 'self'
                and node.func.attr in self.selected and not node.keywords):
            args=copy.deepcopy(node.args)
            extra=self.native_keys.extras(node.func.attr,args) if self.native_keys and node.func.attr in self.indexed else []
            if node.func.attr in self.compact:
                direct = ast.Call(expr('_mx_ctx_hot' + self.cls + '_' + node.func.attr),
                                  ([expr('self'), *args, expr(self.compact_argument)]
                                   if self.compact_self else
                                   [expr(self.compact_argument), *args]), [])
            else:
                direct = ast.Call(expr('_mx_n' + self.cls + '_' + node.func.attr),
                                  [expr('self'), *args, expr('True'), expr('True'), *extra], [])
            return ast.IfExp(expr('_mx_exact'), direct, node)
        return node


class ContextSelf(ast.NodeTransformer):
    """Recover a Space only at operations left outside connected native edges.

    The diagnostic context owns the exact receiver for the duration of one
    policy.  Native dependency calls use the context itself; a remaining
    object/input/helper operation crosses back to the ordinary Space there.
    """

    def __init__(self, cls):
        self.cls = cls
        self.bridges = 0

    def visit_Name(self, node):
        if node.id == 'self' and isinstance(node.ctx, ast.Load):
            self.bridges += 1
            return ast.copy_location(
                expr(f'_mx_cy.cast({self.cls}, _mx_ctx.space)'), node)
        return node


class CommitReturns(ast.NodeTransformer):
    def __init__(self, commit):
        self.commit = commit

    def visit_Return(self, node):
        return [ast.Assign([ast.Name('_mx_value', ast.Store())], node.value or expr('None')),
                ast.If(expr('_mx_cached'), copy.deepcopy(self.commit), []), ast.Return(expr('_mx_value'))]


def compact_pair(cls_name, name, owner, public, formula, body, cached, locals_types,
                 array_size=None):
    """Emit one formula body and a compact exact-receiver cache entry.

    The ordinary public/private wrappers remain the compatibility boundary.
    Native dependencies in the formula body receive the held owner, while each
    entry rechecks ownership before reading and before committing after a call.
    """
    args = ', '.join(a.arg for a in public.args.args[1:])
    params = ('self, ' + args + ', ') if args else 'self, '
    calc_name = '_mx_ctx_calc' + cls_name + '_' + name
    hot_name = '_mx_ctx_hot' + cls_name + '_' + name

    calc = copy.deepcopy(formula)
    calc.name = calc_name
    calc.args.args[0].annotation = expr(cls_name)
    calc.args.args.extend([ast.arg('_mx_owner', expr(owner)),
                           ast.arg('_mx_exact', expr('_mx_cy.bint'))])
    calc.decorator_list = [expr('_mx_cy.cfunc'),
                           expr('_mx_cy.exceptval(-1, check=True)')]
    if locals_types:
        calc.decorator_list.append(expr('_mx_cy.locals(' + ','.join(
            x + '=' + t for x, t in locals_types.items()) + ')'))
    calc.body = body + [ast.Return(expr('0'))]

    hot = copy.deepcopy(formula)
    hot.name = hot_name
    hot.args.args[0].annotation = expr(cls_name)
    hot.args.args.append(ast.arg('_mx_owner', expr(owner)))
    hot.decorator_list = [expr('_mx_cy.cfunc'), expr('_mx_cy.inline'),
                          expr('_mx_cy.exceptval(-1, check=True)'),
                          expr('_mx_cy.locals(_mx_value=' + typed(public) + ')')]
    if array_size is not None:
        index = public.args.args[1].arg
        hot_source = f'''
if not 0 <= {index} < {array_size}:
    raise IndexError('array index out of range')
if _mx_owner is None or self._mx_native is not _mx_owner:
    _mx_owner = self._mx_native
if _mx_owner is not None and _mx_owner._has_{name}[{index}]:
    return _mx_owner.values._v_{name}[{index}]
if _mx_owner is None:
    _mx_owner = {owner}()
    self._mx_native = _mx_owner
_mx_value = {calc_name}({params}_mx_owner, True)
if self._mx_native is not _mx_owner:
    _mx_owner = self._mx_native
    if _mx_owner is None:
        _mx_owner = {owner}()
        self._mx_native = _mx_owner
_mx_owner.values._v_{name}[{index}] = _mx_value
_mx_owner._has_{name}[{index}] = True
return _mx_value
'''
    else:
        hot_source = f'''
if self._has_{name}:
    return self._v_{name}
_mx_value = {calc_name}({params}_mx_owner, True)
self._v_{name} = _mx_value
self._has_{name} = True
return _mx_value
'''
    hot.body = ast.parse(hot_source).body

    exact = ast.parse(f'''
if self._mx_native is None:
    self._mx_native = {owner}()
if _mx_exact:
    if _mx_cached:
        return {hot_name}({params}self._mx_native)
    return {calc_name}({params}self._mx_native, True)
''').body
    wrapper = [*exact, ast.If(expr('_mx_cached'), cached, []),
               ast.Return(expr(f'{calc_name}({params}self._mx_native, False)'))]
    for fn in (calc, hot):
        ast.fix_missing_locations(fn)
    return calc, hot, wrapper


def raw_context_pair(cls_name, name, owner, arrays, public, formula, body,
                     cached, locals_types, scalar_types=None,
                     context_formula=False):
    """Diagnostic: carry raw cache pointers across connected numeric entries.

    The ordinary Space still owns the cache. This deliberately does not refresh
    pointers after an effectful callback and is not a compatible production path.
    """
    args = ', '.join(a.arg for a in public.args.args[1:])
    params = ('self, ' + args + ', ') if args else 'self, '
    frame_entry = scalar_types is not None
    ctx_type = '_mx_ctx' + cls_name
    ctx_arg = f'_mx_cy.pointer({ctx_type})'
    calc_name = '_mx_ctx_calc' + cls_name + '_' + name
    hot_name = '_mx_ctx_hot' + cls_name + '_' + name

    calc = copy.deepcopy(formula)
    calc.name = calc_name
    if context_formula:
        calc.args.args = [ast.arg('_mx_ctx', expr(ctx_arg)),
                          *calc.args.args[1:],
                          ast.arg('_mx_exact', expr('_mx_cy.bint'))]
        bridge = ContextSelf(cls_name)
        body = [bridge.visit(stmt) for stmt in body]
    else:
        calc.args.args[0].annotation = expr(cls_name)
        calc.args.args.extend([ast.arg('_mx_ctx', expr(ctx_arg)),
                               ast.arg('_mx_exact', expr('_mx_cy.bint'))])
    calc.decorator_list = [expr('_mx_cy.cfunc'),
                           expr('_mx_cy.exceptval(-1, check=True)')]
    if locals_types:
        calc.decorator_list.append(expr('_mx_cy.locals(' + ','.join(
            x + '=' + t for x, t in locals_types.items()) + ')'))
    calc.body = body + [ast.Return(expr('0'))]

    hot = copy.deepcopy(formula)
    hot.name = hot_name
    if frame_entry:
        hot.args.args = [ast.arg('_mx_ctx', expr(ctx_arg)),
                         *hot.args.args[1:]]
    else:
        hot.args.args[0].annotation = expr(cls_name)
        hot.args.args.append(ast.arg('_mx_ctx', expr(ctx_arg)))
    hot.decorator_list = [expr('_mx_cy.cfunc'), expr('_mx_cy.inline'),
                          expr('_mx_cy.exceptval(-1, check=True)'),
                          expr('_mx_cy.locals(_mx_value=' + typed(public) + ')')]
    if name in arrays:
        index = public.args.args[1].arg
        size = arrays[name][1]
        hot_source = f'''
if not 0 <= {index} < {size}:
    raise IndexError('array index out of range')
if _mx_ctx.has_{name}[{index}]:
    return _mx_ctx.values._v_{name}[{index}]
_mx_value = {calc_name}({params}_mx_ctx, True)
_mx_ctx.values._v_{name}[{index}] = _mx_value
_mx_ctx.has_{name}[{index}] = True
return _mx_value
'''
    else:
        if frame_entry:
            hot_source = f'''
if _mx_ctx.has_{name}[0]:
    return _mx_ctx.value_{name}[0]
_mx_value = {calc_name}({params}_mx_ctx, True)
_mx_ctx.value_{name}[0] = _mx_value
_mx_ctx.has_{name}[0] = True
return _mx_value
'''
        else:
            hot_source = f'''
if self._has_{name}:
    return self._v_{name}
_mx_value = {calc_name}({params}_mx_ctx, True)
self._v_{name} = _mx_value
self._has_{name} = True
return _mx_value
'''
    if context_formula:
        replacement = (f'{calc_name}(_mx_ctx, ' + (args + ', ' if args else '')
                       + 'True)')
        hot_source = hot_source.replace(
            f'{calc_name}({params}_mx_ctx, True)', replacement)
    elif frame_entry:
        receiver = f'_mx_cy.cast({cls_name}, _mx_ctx.space)'
        hot_source = hot_source.replace(
            f'{calc_name}({params}_mx_ctx, True)',
            f'{calc_name}({receiver}, ' + (args + ', ' if args else '') + '_mx_ctx, True)')
    hot.body = ast.parse(hot_source).body

    setup = ''
    if context_formula:
        setup += '_mx_ctx.space = _mx_cy.cast(_mx_cy.p_void, self)\n'
    setup += f'''
if _mx_exact:
    if self._mx_native is None:
        self._mx_native = {owner}()
    _mx_owner = self._mx_native
    _mx_ctx.values = _mx_owner.values
'''
    if frame_entry and not context_formula:
        setup += '    _mx_ctx.space = _mx_cy.cast(_mx_cy.p_void, self)\n'
    for array_name in arrays:
        setup += (f'    _mx_ctx.has_{array_name} = '
                  f'_mx_cy.address(_mx_owner._has_{array_name}[0])\n')
    if frame_entry:
        for scalar_name in scalar_types:
            setup += (f'    _mx_ctx.has_{scalar_name} = '
                      f'_mx_cy.address(self._has_{scalar_name})\n')
            setup += (f'    _mx_ctx.value_{scalar_name} = '
                      f'_mx_cy.address(self._v_{scalar_name})\n')
    hot_call = (f'_mx_cy.address(_mx_ctx)' + (', ' + args if args else '')
                if frame_entry else params + '_mx_cy.address(_mx_ctx)')
    exact_calc = (f'{calc_name}(_mx_cy.address(_mx_ctx), '
                  + (args + ', ' if args else '') + 'True)'
                  if context_formula else
                  f'{calc_name}({params}_mx_cy.address(_mx_ctx), True)')
    fallback_calc = (f'{calc_name}(_mx_cy.address(_mx_ctx), '
                     + (args + ', ' if args else '') + 'False)'
                     if context_formula else
                     f'{calc_name}({params}_mx_cy.NULL, False)')
    setup += f'''
    if _mx_cached:
        return {hot_name}({hot_call})
    return {exact_calc}
'''
    wrapper = ast.parse(setup).body + [ast.If(expr('_mx_cached'), cached, []),
                                       ast.Return(expr(fallback_calc))]
    for fn in (calc, hot):
        ast.fix_missing_locations(fn)
    return calc, hot, wrapper, {'_mx_owner': owner, '_mx_ctx': ctx_type}


class Storage(ast.NodeTransformer):
    def __init__(self, arrays, owner):
        self.arrays, self.owner = arrays, owner

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        v = node.value
        if not (isinstance(v, ast.Attribute) and ast.unparse(v.value) == 'self'):
            return node
        if v.attr.startswith('_has_') and v.attr[5:] in self.arrays:
            node.value = expr('self._mx_native.' + v.attr)
            if isinstance(node.ctx, ast.Load):
                return ast.BoolOp(ast.And(), [expr('self._mx_native is not None'), node])
        elif v.attr.startswith('_v_') and v.attr[3:] in self.arrays:
            node.value = expr('self._mx_native.values.' + v.attr)
        return node

    def visit_Assign(self, node):
        allocate = any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute)
                       and ast.unparse(t.value.value) == 'self' and t.value.attr.startswith('_v_')
                       and t.value.attr[3:] in self.arrays for t in node.targets)
        node = self.generic_visit(node)
        if allocate:
            assert isinstance(node.value, ast.Name) and node.value.id == 'val'
            return [ast.parse('if self._mx_native is None:\n self._mx_native = ' + self.owner + '()').body[0], node]
        return node


class DataReads(ast.NodeTransformer):
    """Borrow an already-cached view for indexing, retaining its exact owner.

Only exact generated Data receivers bypass their public getter. Its cached view
is assigned once; indexing still observes mutations in the underlying array.
The local owner is acquired before index evaluation, including callback indices.
"""
    def __init__(self, references, classes):
        self.references, self.classes = references, classes
        self.locals = {}
        self.changes = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if not (not node.args and not node.keywords and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute) and ast.unparse(node.func.value.value) == 'self'):
            return node
        owner = node.func.value; cls = self.references.get(owner.attr)
        if cls not in self.classes:
            return node
        tree, info = self.classes[cls]; name = node.func.attr
        cell = info.cells.get(name)
        method = next((f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == name), None)
        fields = {n.target.id for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        if (info.is_locked or not cell or not method or not cell.has_formula_def or cell.has_args()
                or typed(method) not in ('object', 'str') or '_has_' + name not in fields):
            return node
        local = '_mx_data_' + str(self.changes); self.changes += 1
        self.locals[local] = cls
        original = copy.deepcopy(node); original.func.value = expr(local)
        check = expr(f'type(({local} := {ast.unparse(owner)})) is {cls} and {local}._has_{name}')
        return ast.IfExp(check, expr(local + '._v_' + name), original)

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        call = node.value
        if not (isinstance(call, ast.Call) and not call.args and not call.keywords
                and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Attribute)
                and ast.unparse(call.func.value.value) == 'self'):
            return node
        owner = call.func.value; cls = self.references.get(owner.attr)
        if cls not in self.classes:
            return node
        tree, info = self.classes[cls]
        if info.is_locked:
            return node
        name = call.func.attr
        method = next((f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == name), None)
        cell = info.cells.get(name)
        if not method or not cell or not cell.has_formula_def or cell.has_args() or '[:' not in typed(method):
            return node
        # Only the standard immutable cached-getter shape qualifies; a native
        # view cannot be publicly reassigned through its private cache field.
        fields = {n.target.id for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        if '_has_' + name not in fields or '_v_' + name not in fields:
            return node
        local = '_mx_data_' + str(self.changes); self.changes += 1
        self.locals[local] = cls
        original = copy.deepcopy(node); original.value.func.value = expr(local)
        direct = copy.deepcopy(node); direct.value = expr(local + '._v_' + name)
        check = expr(f'type(({local} := {ast.unparse(owner)})) is {cls} and {local}._has_{name}')
        return ast.IfExp(check, direct, original)


class KeyCodes(ast.NodeTransformer):
    """Keep immutable label codes native without changing public category Cells.

The final prepared CurveTable owns an immutable mapping. A changed table identity
or changed key misses this derived one-entry mapping. Arbitrary objects (including
mutable/custom Enum behavior) always use the original helper conversion.
"""
    def __init__(self, cls, references):
        self.cls, self.references = cls, references
        self.entries = {}

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == 'code'
                and len(node.args) == 1 and not node.keywords
                and isinstance(node.func.value, ast.Attribute) and ast.unparse(node.func.value.value) == 'self'):
            return node
        helper = node.func.value.attr
        tp = self.references.get(helper, '')
        if not tp.endswith('.CurveTable'):
            return node
        key = ast.unparse(node)
        if key not in self.entries:
            self.entries[key] = ('_mx_k' + self.cls + '_' + str(len(self.entries)), tp)
        name, _ = self.entries[key]
        return ast.Call(expr(name), [expr('self'), node.func.value, node.args[0]], [])

    def emit(self):
        fields, functions = [], []
        for name, tp in self.entries.values():
            fields += [f'    cdef object {name}_value', f'    cdef {tp.replace(".", ".")} {name}_table',
                       f'    cdef long long {name}_code']
            functions.append(f'''@_mx_cy.cfunc
@_mx_cy.inline
@_mx_cy.exceptval(-1, check=True)
def {name}(self: {self.cls}, table: {tp}, value: object) -> _mx_cy.longlong:
    code: _mx_cy.longlong
    if type(value) is str or type(value) is int:
        if value is self.{name}_value and table is self.{name}_table:
            return self.{name}_code
        code = table.code(value)
        self.{name}_value = value
        self.{name}_table = table
        self.{name}_code = code
        return code
    return table.code(value)
''')
        return '\n'.join(fields), functions

def owner_source(name, arrays):
    fields = '\n'.join(f'    {NUMERIC[t]} _v_{n}[{size}]' for n, (t, size) in arrays.items())
    pxd = f'cdef struct {name}_values:\n{fields}\n\ncdef class {name}:\n    cdef {name}_values* values\n'
    pxd += ''.join(f'    cdef unsigned char _has_{n}[{size}]\n' for n, (t, size) in arrays.items())
    py = f'''@_mx_cy.cclass
class {name}:
    def __cinit__(self):
        self.values = _mx_cy.cast(_mx_cy.pointer({name}_values), _mx_malloc(_mx_cy.sizeof({name}_values)))
        if self.values == _mx_cy.NULL:
            raise MemoryError()

    def __dealloc__(self):
        _mx_free(self.values)

    def __reduce__(self):
        state = {{}}
'''
    for n, (t, size) in arrays.items():
        py += f'        state[{n!r}] = [(i, self.values._v_{n}[i]) for i in range({size}) if self._has_{n}[i]]\n'
    py += f'        return ({name}_restore, (state,))\n\ndef {name}_restore(state):\n    result: {name} = {name}()\n'
    for n, (t, size) in arrays.items():
        py += f'    for i, value in state[{n!r}]:\n        if not 0 <= i < {size}:\n            raise ValueError("invalid native cache index")\n        result.values._v_{n}[i] = value\n        result._has_{n}[i] = True\n'
    py += '    return result\n\n'
    return py, pxd


def raw_context_declaration(cls_name, owner, arrays, scalar_types=None):
    name = '_mx_ctx' + cls_name
    fields = [f'cdef struct {name}:', f'    {owner}_values* values']
    if scalar_types is not None:
        fields.append('    void* space')
    fields.extend(f'    unsigned char* has_{array_name}' for array_name in arrays)
    if scalar_types is not None:
        for scalar_name, scalar_type in scalar_types.items():
            fields.append(f'    bint* has_{scalar_name}')
            fields.append(f'    {NUMERIC[scalar_type]}* value_{scalar_name}')
    return '\n'.join(fields) + '\n'


def transform(source, declarations, module, path, *, frame_cells=None):
    import os
    connected = os.environ.get('MX_CY_CONNECTED_NATIVE', '1') == '1'
    compact_context = os.environ.get('MX_CY_COLLECT_COMPACT_CONTEXT', '0') == '1'
    raw_context = os.environ.get('MX_CY_COLLECT_RAW_CONTEXT', '0') == '1'
    frame_entry = os.environ.get('MX_CY_COLLECT_FRAME_ENTRY', '0') == '1'
    context_formula = os.environ.get('MX_CY_COLLECT_CONTEXT_FORMULAS', '0') == '1'
    frame_entry = frame_entry or context_formula
    raw_context = raw_context or frame_entry
    compact_context = compact_context or raw_context
    if connected:
        from .connected_categories import Categories, NativeKeys, key_parameters, use_precomputed_keys, SUPPORT as CATEGORY_SUPPORT
    tree = ast.parse(source)
    classes = {n.name: (n, module.classes[n.name]) for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name in module.classes}
    lines = source.splitlines(keepends=True)
    edits, tail, owners, owner_pxd, inventory = [], [], [], [], []
    enum_tests = EnumTests()
    top_range_shadow = any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(a.asname == 'range' or a.name == 'range' for a in n.names)
                           or isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == 'range'
                           or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'range' for t in n.targets)
                           for n in tree.body)
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in module.classes:
            continue
        info = module.classes[cls.name]
        if info.is_locked:
            continue
        methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
        references = {n.target.id: ast.unparse(n.annotation) for n in cls.body
                      if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
        key_codes = KeyCodes(cls.name, references)
        categories = Categories(cls.name, methods, references, info) if connected else None
        native_keys = NativeKeys(cls.name,methods,references) if connected else None
        indexed = {n for n,f in methods.items() if any(isinstance(x, ast.Name) and x.id.startswith('_mx_oi_') for x in ast.walk(f))}
        selected = {n for n, f in methods.items() if n in info.cells and typed(f) in NUMERIC
                    and info.cells[n].has_formula_def and not info.cells[n].called_with_kwargs
                    and simple(f, connected and n in indexed) and '_f_' + n in methods and simple(methods['_f_' + n], connected and n in indexed)
                    and any(ast.unparse(d) == '_mx_cy.ccall' for d in f.decorator_list)}
        if connected and not getattr(getattr(info,'helper_info',None),'calls',None):
            # Without a typed helper contract, indexing an object-valued Cell
            # keeps Python table operations inside the numeric chain. Preserve
            # the accepted implementation for that class, rather than paying
            # the structural path's storage/call overhead around opaque work.
            opaque_index = any(isinstance(x,ast.Subscript) and isinstance(x.value,ast.Call)
                and isinstance(x.value.func,ast.Attribute) and ast.unparse(x.value.func.value)=='self'
                and x.value.func.attr in methods and typed(methods[x.value.func.attr])=='object'
                for n in selected for x in ast.walk(methods['_f_'+n]))
            if opaque_index:
                inventory.append({'class':cls.name,'accepted_path':'object-valued Cell indexing without typed helper contract'})
                continue
        arrays = {}
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id.startswith('_v_'):
                m = re.fullmatch(r'(_mx_cy\.(?:double|longlong|bint))\[(\d+)\]', ast.unparse(node.annotation))
                if m and node.target.id[3:] in selected:
                    arrays[node.target.id[3:]] = (m[1], int(m[2]))
        compact = set()
        if compact_context and arrays:
            compact = {n for n in selected
                       if (n in arrays or (len(methods[n].args.args) == 1
                           and '_has_' + n in references
                           and references.get('_v_' + n) in NUMERIC))
                       and n not in indexed
                       and all(a.annotation and ast.unparse(a.annotation) in NUMERIC
                               for a in methods[n].args.args[1:])
                       and not any(isinstance(x, (ast.Try, ast.TryStar, ast.With))
                                   for x in ast.walk(methods['_f_' + n]))}
        scalar_types = {n: references['_v_' + n] for n in compact if n not in arrays}
        owner = '_mx_store' + cls.name
        if arrays:
            py, pxd = owner_source(owner, arrays)
            if raw_context:
                pxd += '\n' + raw_context_declaration(
                    cls.name, owner, arrays, scalar_types if frame_entry else None)
            owners.append(py); owner_pxd.append(pxd)
            declarations = declarations.replace(f'cdef class {cls.name}(', f'cdef class {cls.name}(', 1)
            marker = next(line for line in declarations.splitlines() if line.startswith(f'cdef class {cls.name}('))
            declarations = declarations.replace(marker, marker + f'\n    cdef {owner} _mx_native', 1)
            for node in cls.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    field = node.target.id
                    if field.removeprefix('_v_').removeprefix('_has_') in arrays:
                        edits.append((node.lineno-1, node.end_lineno, ''))
            # Scope removal to this class, never fields on another class.
            start = declarations.index(marker); end = declarations.find('\ncdef class ', start+1)
            if end < 0: end = len(declarations)
            section = declarations[start:end]
            for n in arrays:
                section = re.sub(r'^    cdef [^\n]+ (_v_|_has_)' + re.escape(n) + r'\n', '', section, flags=re.M)
            declarations = declarations[:start] + section + declarations[end:]
            edits.append((cls.body[0].lineno-1, cls.body[0].lineno-1, f'    _mx_native: {owner}\n'))
        for n in sorted(selected):
            public = copy.deepcopy(methods[n]); formula = copy.deepcopy(methods['_f_' + n])
            if n in (frame_cells or {}).get(cls.name, ()):
                # The shared frame already owns exact-receiver execution.
                # Keep only the original override-aware cache wrapper and its
                # lazy storage; do not emit a second fused formula entry.
                public = Storage(arrays, owner).visit(public)
                ast.fix_missing_locations(public)
                original = methods[n]
                first = min([original.lineno] + [d.lineno for d in original.decorator_list]) - 1
                edits.append((first, original.end_lineno,
                              textwrap.indent(ast.unparse(public), '    ') + '\n'))
                inventory.append({'class': cls.name, 'cell': n,
                                  'frame_entry': True, 'native_storage': n in arrays,
                                  'original_override_cache_wrapper': True,
                                  'duplicate_fused_formula_entry': False})
                continue
            key_positions=key_parameters(public) if connected and n in indexed else []
            if key_positions:use_precomputed_keys(public,key_positions)
            indices = [] if top_range_shadow else local_indices(formula, methods)
            # One formula body, placed below the cache lookup in the native entry.
            # A private override remains authoritative on a non-exact receiver.
            fn = copy.deepcopy(formula); fn.name = '_mx_n' + cls.name + '_' + n
            fn.args.args[0].annotation = expr(cls.name)
            fn.args.args += [ast.arg('_mx_cached', expr('_mx_cy.bint')), ast.arg('_mx_exact', expr('_mx_cy.bint'))]
            for i in key_positions:
                fn.args.args.extend([ast.arg(f'_mx_pass_valid_{i}',expr('_mx_cy.bint')),ast.arg(f'_mx_pass_key_{i}',expr('_mx_cy.longlong'))])
            default_keys=', False, 0'*len(key_positions)
            fn.decorator_list = [expr('_mx_cy.cfunc'), expr('_mx_cy.inline'), expr('_mx_cy.exceptval(-1, check=True)')]
            args = ', '.join(a.arg for a in public.args.args[1:])
            params = ('self, ' + args + ', ') if args else 'self, '
            # Cache body calls the private formula. Exact receivers instead enter
            # the same native function with caching disabled; Cython can inline it.
            class Miss(ast.NodeTransformer):
                def visit_Call(self, node):
                    node = self.generic_visit(node)
                    if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'self._f_' + n:
                        return ast.IfExp(expr('_mx_exact'), expr(fn.name + '(' + params + 'False, True'+default_keys+')'), node)
                    return node
            cached = [Miss().visit(s) for s in public.body]
            cached = [Storage(arrays, owner).visit(s) for s in cached]
            # NodeTransformer may return statement lists inside compound suites.
            if native_keys:native_keys.begin_formula()
            # A frame-backed dependency must reach its public entry. Sending it
            # to the legacy native entry would publish a competing exact cache.
            edge_targets = selected - set((frame_cells or {}).get(cls.name, ()))
            formula_body = [Edges(cls.name, edge_targets, native_keys, indexed,
                                  compact if n in compact else (),
                                  '_mx_ctx' if raw_context else '_mx_owner',
                                  not frame_entry).visit(s)
                            for s in formula.body]
            data_reads = DataReads(references, classes)
            formula_body = [data_reads.visit(s) for s in formula_body]
            formula_body = [key_codes.visit(s) for s in formula_body]
            formula_body = [enum_tests.visit(s) for s in formula_body]
            if categories:
                formula_body = [categories.visit(s) for s in formula_body]
            locals_types = {x: '_mx_cy.longlong' for x in indices}
            locals_types.update(data_reads.locals)
            if native_keys:locals_types.update(native_keys.locals)
            fused = (n in arrays or (len(public.args.args) == 1 and '_has_' + n in references
                                     and references.get('_v_' + n) in NUMERIC))
            fused = fused and not any(isinstance(x, (ast.Try, ast.TryStar, ast.With)) for x in ast.walk(formula))
            if fused and n not in compact:
                if n in arrays:
                    index = public.args.args[1].arg
                    value = f'self._mx_native.values._v_{n}[{index}]'
                    flag = f'self._mx_native._has_{n}[{index}]'
                    hit = f'self._mx_native is not None and {flag}'
                    commit = ast.parse(f'if self._mx_native is None:\n self._mx_native = {owner}()\n{value} = _mx_value\n{flag} = True').body
                    head = ast.parse(f'if not 0 <= {index} < {arrays[n][1]}:\n raise IndexError("array index out of range")').body
                else:
                    value, flag = 'self._v_' + n, 'self._has_' + n
                    hit = flag
                    commit = ast.parse(f'{value} = _mx_value\n{flag} = True').body
                    head = []
                head += ast.parse(f'if {hit}:\n return {value}').body
                overridden = ast.parse(f'_mx_value = self._f_{n}({args})').body + copy.deepcopy(commit) + [ast.Return(expr('_mx_value'))]
                head += [ast.If(expr('not _mx_exact'), overridden, [])]
                # Lift once; do not duplicate formula bodies or recursively call
                # the same function merely to switch from cache to formula mode.
                wrapper = ast.FunctionDef(name='_body', args=copy.deepcopy(formula.args), body=formula_body,
                                          decorator_list=[], type_params=[])
                # Cython scalar cdef functions return numeric zero on implicit
                # fall-through. Cache that result exactly as the old wrapper did.
                wrapper.body.append(ast.Return(expr('0')))
                wrapper = CommitReturns(commit).visit(wrapper)
                formula_body = wrapper.body
                cached = head
                locals_types['_mx_value'] = typed(public)
            if n in compact:
                if raw_context:
                    calc, hot, wrapper, wrapper_locals = raw_context_pair(
                        cls.name, n, owner, arrays, public, formula,
                        formula_body, cached, locals_types,
                        scalar_types if frame_entry else None,
                        context_formula)
                    locals_types.update(wrapper_locals)
                else:
                    calc, hot, wrapper = compact_pair(
                        cls.name, n, owner, public, formula, formula_body, cached,
                        locals_types, arrays[n][1] if n in arrays else None)
                tail.extend([ast.unparse(calc) + '\n', ast.unparse(hot) + '\n'])
                fn.body = wrapper
            else:
                fn.body = [ast.If(expr('_mx_cached'), cached, []), *formula_body]
            if locals_types:
                fn.decorator_list.append(expr('_mx_cy.locals(' + ','.join(x+'='+t for x,t in locals_types.items()) + ')'))
            ast.fix_missing_locations(fn);tail.append(ast.unparse(fn)+'\n')
            for meth, usecache in ((public, True), (formula, False)):
                # Returning through the original method preserves its signature,
                # cpdef override dispatch, defaults and checked exception contract.
                doc = meth.body[:1] if (meth.body and isinstance(meth.body[0], ast.Expr)
                                       and isinstance(meth.body[0].value, ast.Constant)
                                       and isinstance(meth.body[0].value.value, str)) else []
                meth.body = doc + [ast.Return(expr(fn.name + '(' + params + str(usecache) + ', type(self) is ' + cls.name + default_keys + ')'))]
                ast.fix_missing_locations(meth)
                original = methods[meth.name]
                first = min([original.lineno] + [d.lineno for d in original.decorator_list]) - 1
                edits.append((first, original.end_lineno, textwrap.indent(ast.unparse(meth), '    ')+'\n'))
            inventory.append({'class': cls.name, 'cell': n, 'native_indices': indices,
                              'native_storage': n in arrays, 'fused_cache_formula': fused,
                              'compact_context_entry': n in compact,
                              'borrowed_data_reads': data_reads.changes})
        fields, functions = key_codes.emit()
        key_mapping_count=len(functions)
        if categories:
            category_fields, category_functions = categories.emit()
            fields += '\n' + category_fields if category_fields else ''
            functions += category_functions
            inventory.append({'class':cls.name,'canonical_category_reads':categories.reads,
                              'guarded_native_category_sites':len(category_functions)})
        if native_keys:
            native_fields,native_functions=native_keys.emit()
            fields += '\n'+native_fields if native_fields else ''
            functions += native_functions
            inventory.append({'class':cls.name,'native_key_providers':list(native_keys.providers),'native_key_edges':native_keys.edges})
        if fields:
            marker = next(line for line in declarations.splitlines() if line.startswith(f'cdef class {cls.name}('))
            declarations = declarations.replace(marker, marker + '\n' + fields, 1)
            tail.extend(functions)
            inventory.append({'class': cls.name, 'immutable_key_mappings': key_mapping_count})
    for start, end, content in sorted(edits, reverse=True):
        lines[start:end] = [content]
    source = ''.join(lines)
    if owners:
        source = 'from cython.cimports.libc.stdlib import malloc as _mx_malloc, free as _mx_free\n' + source
        source += '\n' + '\n'.join(owners)
        declarations += '\n' + '\n'.join(owner_pxd)
    if enum_tests.count:
        support = path.with_name('_mx_native_support.pxd')
        support.write_text(ENUM_COMPARE_SUPPORT + (CATEGORY_SUPPORT if connected else ''), encoding='utf-8')
        package = module.fqname.rsplit('.', 1)[0]
        native_names = '_mx_enum_equal' + (', _mx_category_standard, _mx_category_code' if connected else '')
        source = f'from cython.cimports.{package}._mx_native_support import {native_names}\n' + source
        inventory.append({'native_enum_truth_sites': enum_tests.count,
                          'semantics': 'Public objects unchanged; dynamic integer comparison-slot guard; custom comparisons use normal rich comparison and truth conversion.'})
    source += '\n' + '\n'.join(tail)
    ast.parse(source)
    path.write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    return source, declarations
