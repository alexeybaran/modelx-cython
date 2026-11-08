"""Separate checked and proven calls to a single dense cache body."""

import ast

import libcst as cst

from .cell_entries import EntryModule, dense_condition
from .consts import CY_MOD, FORMULA_PREF
from .dense_formulas import _ExactTests
from .scopes import expression_ast


def _unsigned_check(layout):
    if (len(layout.axes) != 1 or layout.exact_spill
            or layout.axes[0].c_type != "long long"):
        return None
    axis, = layout.axes
    span = axis.stop - axis.lower
    if not 0 < span < 1 << 64:
        return None
    return cst.parse_expression(
        f"({CY_MOD}.cast({CY_MOD}.ulonglong, {axis.key}) - "
        f"{CY_MOD}.cast({CY_MOD}.ulonglong, {axis.lower})) < {span}")


def _public_checks(cls, layouts):
    """Normalize only the matching checked public entry; keep virtual misses."""
    targets = {layout.formula[len(FORMULA_PREF):]: layout for layout in layouts.values()
               if layout.owner == cls.name.value and layout.formula.startswith(FORMULA_PREF)}
    methods = []
    for method in cls.body.body:
        layout = targets.get(method.name.value) if isinstance(method, cst.FunctionDef) else None
        check = _unsigned_check(layout) if layout is not None else None
        if check is not None:
            params = method.params
            axis, = layout.axes
            if (not params.posonly_params and not params.kwonly_params and not params.star_kwarg
                    and not isinstance(params.star_arg, cst.Param)
                    and tuple(p.name.value for p in params.params) == ('self', axis.key)
                    and params.params[1].annotation is not None
                    and params.params[1].annotation.annotation.deep_equals(
                        cst.parse_expression(f'{CY_MOD}.longlong'))
                    and isinstance(method.body, cst.IndentedBlock) and len(method.body.body) == 1):
                guard, = method.body.body
                expected = expression_ast(dense_condition(layout.axes))
                if (isinstance(guard, cst.If) and isinstance(guard.orelse, cst.Else)
                        and ast.dump(expression_ast(guard.test))
                        == ast.dump(expected)):
                    method = method.with_changes(body=method.body.with_changes(
                        body=(guard.with_changes(test=check),)))
        methods.append(method)
    return cls.with_changes(body=cls.body.with_changes(body=tuple(methods)))


class _Calls(cst.CSTTransformer):
    def __init__(self, entries, propagate):
        self.entries = entries
        self.propagate = propagate

    def leave_Call(self, original, updated):
        if not isinstance(original.func, cst.Name) or original.func.value not in self.entries:
            return updated
        if (len(original.args) != 3 or not isinstance(original.args[-1].value, cst.Integer)
                or original.args[-1].value.evaluated_value not in (0, 1)):
            raise ValueError("unexpected one-axis proof call")
        target = self.entries[original.func.value] if original.args[-1].value.evaluated_value else updated.func
        args = (*updated.args[:-2], updated.args[-2].with_changes(comma=cst.MaybeSentinel.DEFAULT))
        if self.propagate and original.args[-1].value.evaluated_value:
            args += (cst.Arg(cst.Name("True")),)
        return updated.with_changes(func=target, args=args)


def specialize_entries(state: EntryModule, layouts, propagate=False):
    module = state.syntax
    names = state.names
    exact = names.reserve('_mx_inbounds_exact')
    bodies, rewritten, proved = [], [], {}
    for stmt in module.body:
        if isinstance(stmt, cst.ClassDef) and isinstance(stmt.body, cst.IndentedBlock):
            rewritten.append(_public_checks(stmt, layouts))
            continue
        if not isinstance(stmt, cst.FunctionDef) or stmt.name.value not in layouts:
            rewritten.append(stmt)
            continue
        layout = layouts[stmt.name.value]
        params = stmt.params.params
        check = _unsigned_check(layout)
        if (check is None or not isinstance(stmt.body, cst.IndentedBlock) or len(stmt.body.body) != 1):
            rewritten.append(stmt)
            continue
        axis, = layout.axes
        guard, = stmt.body.body
        proof = params[-1].name.value if len(params) == 3 else None
        expected = expression_ast(dense_condition(layout.axes, proof))
        if (not isinstance(guard, cst.If) or not isinstance(guard.orelse, cst.Else)
                or ast.dump(expression_ast(guard.test)) != ast.dump(expected)):
            rewritten.append(stmt)
            continue
        if proof is None:
            rewritten.append(stmt.with_changes(body=stmt.body.with_changes(
                body=(guard.with_changes(test=check),))))
            continue
        name = names.reserve(stmt.name.value + '_inbounds')
        proved[stmt.name.value] = cst.Name(name)
        args = tuple(param.with_changes(comma=cst.MaybeSentinel.DEFAULT) for param in params[:-1])
        signature = stmt.params.with_changes(params=args)
        body_signature = signature
        body = guard.body
        flag = ""
        if propagate:
            body_signature = signature.with_changes(params=(*args, cst.Param(
                cst.Name(exact), annotation=cst.Annotation(cst.parse_expression(f"{CY_MOD}.bint")))))
            body = body.visit(_ExactTests(layout.owner, f"({exact} or type(self) is {layout.owner})"))
            flag = ", False"
        bodies.append(stmt.with_changes(name=cst.Name(name), params=body_signature, body=body))
        entry = cst.parse_statement(f"return {name}({args[0].name.value}, {axis.key}{flag})\n")
        rewritten.append(stmt.with_changes(params=signature, body=stmt.body.with_changes(
            body=(guard.with_changes(test=check, body=guard.body.with_changes(body=(entry,))),))))
    return state.replace(module.with_changes(body=(*rewritten, *bodies)).visit(_Calls(proved, propagate)))
