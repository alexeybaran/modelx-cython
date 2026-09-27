"""Matching checked exception contracts on generated numeric Cells.

No bodies, caches, dispatch, arithmetic or
public signatures change. A returned -1 remains valid when no error is pending.
Object, void, bint, helper and runtime functions retain their existing contracts.
"""
import ast
import re

def transform(source, declarations):
    tree=ast.parse(source); selected=[]
    lines=source.splitlines(keepends=True); insertions=[]
    for cls in tree.body:
        if not isinstance(cls,ast.ClassDef):continue
        for fn in cls.body:
            if not isinstance(fn,ast.FunctionDef) or fn.name.startswith('__'):continue
            if fn.returns is None or ast.unparse(fn.returns) not in ('_mx_cy.double','_mx_cy.longlong'):continue
            if not any(ast.unparse(d) in ('_mx_cy.ccall','_mx_cy.cfunc') for d in fn.decorator_list):continue
            pattern=r'(?m)^(    (?:cpdef|cdef) (?:double|long long) '+re.escape(fn.name)+r'\('+re.escape(cls.name)+r' self[^\n]*\))$'
            declarations,n=re.subn(pattern,r'\1 except? -1',declarations)
            assert n==1,(cls.name,fn.name,n)
            insertions.append(fn.lineno-1);selected.append(cls.name+'.'+fn.name)
    for index in sorted(insertions,reverse=True):lines.insert(index,'    @_mx_cy.exceptval(-1, check=True)\n')
    assert selected
    return ''.join(lines),declarations,selected
