"""Compile the fixed helper library inline from its shared Python source.

This binds only our declared Curve/CurveTable ABI, not pandas/model expressions.
Cython compiles the function bodies. The ordinary Python helper remains the source
of truth, including every lookup-domain check and exception.
"""
import ast
import inspect
import textwrap
from .lookups import Curve, CurveTable


_SIGNATURES = {
    '_right': ('Py_ssize_t', 'double x', 'noexcept'),
    '_index': ('Py_ssize_t', 'double x', 'noexcept'),
    'at': ('double', 'double x', 'except? -1'),
    'get': ('double', 'double x, double default=0.0', 'except? -1'),
    'previous': ('double', 'double x, double below=NAN', 'except? -1'),
    'linear': ('double', 'double x', 'except? -1'),
}


def inline_lookup_source():
    parts = ['from libc.math cimport isnan, NAN']
    for typ in (Curve, CurveTable):
        signatures = dict(_SIGNATURES)
        if typ is CurveTable:
            signatures.pop('_index')
            signatures['_group_index'] = ('Py_ssize_t','long long group','noexcept')
            signatures['_right'] = ('Py_ssize_t','Py_ssize_t gi, double x','noexcept')
            for key in ('at','get','previous','linear'):
                ret,args,error = signatures[key]
                signatures[key] = (ret,'long long group, '+args,error)
        class Calls(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='self' and node.func.attr in signatures:
                    return ast.copy_location(ast.Call(ast.Name(f'_lookup_{typ.__name__}_{node.func.attr}',ast.Load()),[ast.Name('self',ast.Load())]+[self.visit(a) for a in node.args],node.keywords),node)
                return self.generic_visit(node)
        for name,(ret,args,error) in signatures.items():
            node=ast.parse(textwrap.dedent(inspect.getsource(getattr(typ,name)))).body[0]
            parts += ['@'+ast.unparse(d) for d in node.decorator_list]
            parts += [f'cdef inline {ret} _lookup_{typ.__name__}_{name}({typ.__name__} self, {args}) {error} nogil:']
            for stmt in node.body:
                if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Constant) and isinstance(stmt.value.value,str): continue
                text=ast.unparse(ast.fix_missing_locations(Calls().visit(stmt)))
                parts.extend(textwrap.indent(text,'    ').splitlines())
            parts.append('')
    return '\n'.join(parts)
