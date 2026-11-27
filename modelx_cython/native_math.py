"""Calls of the math module through a Reference, compiled to the C library behind an identity check.

``self.<ref>.<name>(x)`` with ``<ref>`` a Reference that held the standard ``math`` module in the
sample and ``<name>`` one of NAMES becomes
``_mx_sys._mx_math_<name>(_mx_cy.cast(_mx_cy.p_void, self.<ref>), self, '<ref>', x, path)``: the
Reference as a borrowed pointer, compared only, so no reference count moves on the shared module. While
the Reference is the math module and ``x`` is a C number or an exact float where the C library and
CPython's math module give the same result, the C function runs; otherwise the Reference's own
member is called, so a domain error, an overflow, a NaN, another argument type or a rebound
Reference behave as live (see _mx_sys.pxd). The standard module's members return a float (a bool
for isinf) whatever they are given, so the call's C result type holds.
"""
import libcst as cst

from .consts import CY_MOD, MX_SELF, MX_SYS_MOD
from .scopes import OPAQUE_SCOPES

NAMES = ('erf', 'exp', 'isinf', 'log', 'sqrt')


def math_references(cls_info):
    return {name for name, ref in cls_info.refs.items() if getattr(ref, 'is_math', False)}


def math_member(call, refs):
    """Recognize a supported direct math call through a traced module Reference."""
    if isinstance(call, cst.Call):
        func = call.func
        if (isinstance(func, cst.Attribute) and func.attr.value in NAMES
                and isinstance(func.value, cst.Attribute) and isinstance(func.value.value, cst.Name)
                and func.value.value.value == MX_SELF and func.value.attr.value in refs
                and len(call.args) == 1 and call.args[0].keyword is None and call.args[0].star == ''):
            return func.value.attr.value, func.attr.value
    return None


class _MathCalls(cst.CSTTransformer):
    def __init__(self, refs, path):
        self.refs, self.path, self.count = refs, path, 0

    def on_visit(self, node):
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)

    def leave_Call(self, before, after):
        member = math_member(before, self.refs)
        if member is None:
            return after
        self.count += 1
        ref, name = member
        borrowed = cst.parse_expression(f'{CY_MOD}.cast({CY_MOD}.p_void, {MX_SELF}.{ref})')
        return cst.Call(cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(f'_mx_math_{name}')),
                        args=(cst.Arg(borrowed), cst.Arg(cst.Name(MX_SELF)), cst.Arg(cst.SimpleString(repr(ref))),
                              after.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT),
                              cst.Arg(cst.SimpleString(repr(self.path)))),
                        lpar=after.lpar, rpar=after.rpar)


def lower(node, cls_info, cells):
    """The formula ``node`` with its math-module calls lowered, and the number lowered."""
    refs = math_references(cls_info)
    if not refs:
        return node, 0
    calls = _MathCalls(refs, cells.model_path)
    return node.with_changes(body=node.body.visit(calls)), calls.count
