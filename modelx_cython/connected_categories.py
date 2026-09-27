"""Derived native category values attached to the existing canonical cache.

Public results stay Python objects. Exact-receiver cache reads and guarded native
integer representations remove repeated method calls and integer conversion.
No binding survives a change of object identity or comparison slots. Custom
comparison and overflow retain the ordinary rich-comparison path.
"""
import ast
import copy

SUPPORT = r'''
cdef extern from *:
    """
    static CYTHON_INLINE int __mx_category_standard(PyObject *value) {
        return PyLong_Check(value) &&
            Py_TYPE(value)->tp_richcompare == PyLong_Type.tp_richcompare;
    }
    static CYTHON_INLINE int __mx_category_code(PyObject *value, long long *out) {
        if (!__mx_category_standard(value)) return 0;
        int overflow = 0;
        *out = PyLong_AsLongLongAndOverflow(value, &overflow);
        if (PyErr_Occurred()) return -1;
        return !overflow;
    }
    """
    bint _mx_category_standard "__mx_category_standard" (object value) noexcept
    bint _mx_category_code "__mx_category_code" (object value, long long* out) except -1
'''

def expr(s):return ast.parse(s,mode='eval').body


class Categories(ast.NodeTransformer):
    def __init__(self, cls, methods, references, info):
        self.cls=cls;self.methods=methods;self.references=references;self.info=info
        self.entries={};self.reads=0

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='_mx_enum_equal':
            # Per-expression slots prevent unrelated categories evicting each
            # other. They are derived metadata, never another Cell result.
            key=ast.unparse(node)
            if key not in self.entries:self.entries[key]='_mx_cat'+self.cls+'_'+str(len(self.entries))
            return ast.Call(expr(self.entries[key]),[expr('self'),*node.args],[])
        if (isinstance(node.func,ast.Attribute) and ast.unparse(node.func.value)=='self'
                and not node.args and not node.keywords):
            name=node.func.attr
            cell=self.info.cells.get(name);fn=self.methods.get(name)
            if (cell and fn and cell.has_formula_def and not cell.is_locked
                    and not cell.called_with_kwargs and fn.returns
                    and ast.unparse(fn.returns) in ('object','str')
                    and '_has_'+name in self.references and '_v_'+name in self.references
                    and any(ast.unparse(d)=='_mx_cy.ccall' for d in fn.decorator_list)):
                self.reads+=1
                return ast.IfExp(expr('_mx_exact and self._has_'+name),expr('self._v_'+name),node)
        return node

    def emit(self):
        fields=[];functions=[]
        for name in self.entries.values():
            fields.extend([f'    cdef object {name}_a',f'    cdef object {name}_b',
                           f'    cdef long long {name}_av',f'    cdef long long {name}_bv'])
            functions.append(f'''@_mx_cy.cfunc
@_mx_cy.inline
@_mx_cy.exceptval(-1, check=True)
def {name}(self: {self.cls}, a: object, b: object, invert: _mx_cy.bint) -> _mx_cy.bint:
    av: _mx_cy.longlong
    bv: _mx_cy.longlong
    if _mx_category_standard(a) and _mx_category_standard(b):
        if a is self.{name}_a:
            av = self.{name}_av
        elif _mx_category_code(a, _mx_cy.address(av)):
            self.{name}_a = a
            self.{name}_av = av
        else:
            return _mx_enum_equal(a,b,invert)
        if b is self.{name}_b:
            bv = self.{name}_bv
        elif _mx_category_code(b, _mx_cy.address(bv)):
            self.{name}_b = b
            self.{name}_bv = bv
        else:
            return _mx_enum_equal(a,b,invert)
        # Replacing a held object may run a finalizer. Recheck the comparison
        # contract after those decrefs, before consuming the native values.
        if _mx_category_standard(a) and _mx_category_standard(b):
            return av != bv if invert else av == bv
    return _mx_enum_equal(a,b,invert)
''')
        return '\n'.join(fields),functions


class NativeKeys:
    """Pass a canonical category's derived integer through eligible Cell edges.

    The object accompanies its code for formula misses and fallback. Exact
    ordinary cached providers own it for the entire call. All argument
    expressions retain their original evaluation order; domain validity is
    checked after they execute and again by the receiving cache's type guard.
    """
    def __init__(self,cls,methods,references):
        self.cls,self.methods,self.references=cls,methods,references
        self.providers={};self.locals={};self.edges=0

    def begin_formula(self):self.locals={}

    def extras(self,method,args):
        extra=[]
        for position,arg in enumerate(self.methods[method].args.args[1:]):
            if not arg.annotation or ast.unparse(arg.annotation)!='object':continue
            value=args[position]
            provider=None
            if (isinstance(value,ast.Call) and not value.args and not value.keywords
                    and isinstance(value.func,ast.Attribute) and ast.unparse(value.func.value)=='self'):
                p=value.func.attr
                if '_has_'+p in self.references and self.references.get('_v_'+p)=='object':provider=p
            if provider is None:
                extra.extend([expr('False'),expr('0')]);continue
            name=self.providers.setdefault(provider,'_mx_key'+self.cls+'_'+provider)
            suffix=str(self.edges);self.edges+=1
            obj='_mx_edge_object_'+suffix;key='_mx_edge_key_'+suffix
            self.locals.update({obj:'object',key:'_mx_cy.longlong'})
            args[position]=ast.NamedExpr(ast.Name(obj,ast.Store()),value)
            extra.extend([expr(f'{name}(self,{obj},_mx_cy.address({key}))'),expr(key)])
        return extra

    def emit(self):
        fields=[];functions=[]
        for provider,name in self.providers.items():
            fields.extend([f'    cdef object {name}_value',f'    cdef long long {name}_code'])
            functions.append(f'''@_mx_cy.cfunc
@_mx_cy.inline
@_mx_cy.exceptval(-1, check=True)
def {name}(self: {self.cls}, value: object, out: _mx_cy.pointer(_mx_cy.longlong)) -> _mx_cy.bint:
    out[0] = 0
    if not (self._has_{provider} and value is self._v_{provider} and _mx_integer_standard(value)):
        return False
    if value is self.{name}_value:
        out[0] = self.{name}_code
        return True
    if not _mx_integer_key(value,out):
        return False
    self.{name}_value = value
    self.{name}_code = out[0]
    return _mx_integer_standard(value)
''')
        return '\n'.join(fields),functions


def key_parameters(fn):
    return [i for i,a in enumerate(fn.args.args[1:]) if a.annotation and ast.unparse(a.annotation)=='object']


def use_precomputed_keys(public,positions):
    """Only the entry check uses supplied codes; post-formula commits recheck."""
    for node in public.body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            for i in positions:
                if node.target.id==f'_mx_oi_k{i}':node.value=expr(f'_mx_pass_key_{i}')
    names={public.args.args[i+1].arg:i for i in positions}
    class Entry(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_mx_integer_key' and ast.unparse(node.args[0]) in names:
                i=names[ast.unparse(node.args[0])]
                return ast.BoolOp(ast.Or(),[expr(f'_mx_pass_valid_{i}'),node])
            return self.generic_visit(node)
    for node in public.body:
        if isinstance(node,ast.If) and isinstance(node.test,ast.UnaryOp) and isinstance(node.test.operand,ast.Attribute) and node.test.operand.attr.endswith('_disabled'):
            Entry().visit(node);break
