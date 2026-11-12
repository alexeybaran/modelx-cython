"""Expose existing native cache hits; keep every miss on its checked entry."""
from dataclasses import dataclass

import libcst as cst

from .cell_entries import EntryModule
from .consts import CY_MOD
from .scopes import CstFacts
from .syntax import dotted_name, literal_value, same_expression, walk
from .typedefs import primitive_kind

@dataclass(frozen=True)
class _Hit:
    receiver: cst.BaseExpression
    parameters: tuple
    test: cst.BaseExpression
    value: cst.BaseExpression


def _bind_hit(function, hit):
    args = function.params.params
    test = hit.ready
    if hit.guard is not None:
        test = cst.BooleanOperation(hit.guard, cst.And(), test)
    return _Hit(args[0].annotation.annotation,
                tuple((arg.name.value, arg.annotation.annotation) for arg in args),
                test, hit.value)


class _Substitute(cst.CSTTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def leave_Name(self, original, updated):
        return self.mapping.get(original.value, updated)

    def leave_Attribute(self, original, updated):
        return updated.with_changes(attr=original.attr)


class _Calls(cst.CSTTransformer):
    def __init__(self, functions, layouts):
        self.functions, self.layouts = functions, layouts
        self.current = None

    def visit_ClassDef(self, node):
        return False

    def visit_FunctionDef(self, node):
        fn = self.functions.get(node.name.value)
        if self.current is not None or fn is None or not fn.name.value.startswith('_mx_dense_formula_'):
            return False
        nodes = [n for stmt in fn.body.body for n in walk(stmt)]
        if (CY_MOD in CstFacts(fn.body.body).bindings(nodes)
                or any(arg.name.value == CY_MOD for arg in fn.params.params)
                or any(isinstance(n, getattr(cst, 'Match', ())) for n in nodes)):
            return False
        self.current = fn
        return True

    def leave_FunctionDef(self, original, updated):
        if self.current is not None and original.name.value == self.current.name.value:
            self.current = None
        return updated

    def visit_Lambda(self, node):
        return False

    visit_GeneratorExp = visit_ListComp = visit_SetComp = visit_DictComp = visit_Lambda

    def leave_Call(self, original, updated):
        if self.current is None or not isinstance(original.func, cst.Name):
            return updated
        layout = self.layouts.get(original.func.value)
        if layout is None:
            return updated
        parameters = self.current.params.params
        if (not parameters or parameters[0].annotation is None
                or not same_expression(parameters[0].annotation.annotation, layout.receiver)
                or len(updated.args) != len(layout.parameters)
                or any(arg.keyword or arg.star for arg in updated.args)
                or not isinstance(updated.args[0].value, cst.Name)
                or updated.args[0].value.value != parameters[0].name.value):
            return updated
        types = {arg.name.value: arg.annotation.annotation for arg in parameters if arg.annotation is not None}
        mapping = {layout.parameters[0][0]: updated.args[0].value}
        for (name, annotation), arg in zip(layout.parameters[1:], updated.args[1:]):
            if isinstance(arg.value, cst.Name) and arg.value.value not in ('True', 'False', 'None'):
                actual_type = types.get(arg.value.value)
                if actual_type is None or not same_expression(actual_type, annotation):
                    return updated
            else:
                try:
                    value = literal_value(arg.value)
                except (ValueError, TypeError):
                    return updated
                if (dotted_name(annotation), primitive_kind(value)) not in (
                        (f'{CY_MOD}.bint', 'bint'), (f'{CY_MOD}.longlong', 'longlong')):
                    return updated
            mapping[name] = arg.value
        substitute = _Substitute(mapping)
        return cst.IfExp(test=layout.test.visit(substitute), body=layout.value.visit(substitute),
                         orelse=updated, lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))


def expose_entries(state: EntryModule, mode):
    functions = state.functions
    layouts = {name: _bind_hit(functions[name], hit) for name, hit in state.cache_hits.items()
               if not hit.masked
               and (mode in (hit.kind, 'both') or mode == 'auto' and hit.guard is None)}
    return state.replace(state.syntax.visit(_Calls(functions, layouts)))
