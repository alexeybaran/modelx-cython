"""Expose existing native cache hits; keep every miss on its checked entry."""
from dataclasses import dataclass

import libcst as cst

from .cell_entries import EntryModule
from .consts import CY_MOD
from .scopes import CstFacts
from .syntax import arguments, dotted_name, literal_value, same_expression, statements, walk
from .typedefs import primitive_kind

_NATIVE = {f'{CY_MOD}.{name}' for name in ('bint', 'double', 'longlong')}


def _pure_index(node, parameters):
    if isinstance(node, cst.Name):
        return node.value in {*parameters, CY_MOD}
    if isinstance(node, cst.Integer):
        return True
    if isinstance(node, cst.Attribute):
        return isinstance(node.value, cst.Name) and node.value.value == CY_MOD and node.attr.value in {'cast', 'ulonglong'}
    if isinstance(node, cst.UnaryOperation):
        return isinstance(node.operator, (cst.Plus, cst.Minus)) and _pure_index(node.expression, parameters)
    if isinstance(node, cst.BinaryOperation):
        return isinstance(node.operator, cst.Subtract) and _pure_index(node.left, parameters) and _pure_index(node.right, parameters)
    if isinstance(node, cst.BooleanOperation):
        return _pure_index(node.left, parameters) and _pure_index(node.right, parameters)
    if isinstance(node, cst.Comparison):
        return (_pure_index(node.left, parameters) and all(
            isinstance(part.operator, (cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual))
            and _pure_index(part.comparator, parameters) for part in node.comparisons))
    args = arguments(node, f'{CY_MOD}.cast', 2)
    return args is not None and dotted_name(args[0]) == f'{CY_MOD}.ulonglong' and _pure_index(args[1], parameters)


def _index(node):
    return (node.slice[0].slice.value if isinstance(node, cst.Subscript)
            and len(node.slice) == 1 and isinstance(node.slice[0].slice, cst.Index) else None)


@dataclass(frozen=True)
class _Hit:
    kind: str
    receiver: cst.BaseExpression
    parameters: tuple
    test: cst.BaseExpression
    value: cst.BaseExpression


def _layout(fn, scalar_view=False):
    if fn.returns is None:
        return None
    args, body = fn.params.params, statements(fn.body)
    result = dotted_name(fn.returns.annotation)
    if result not in _NATIVE and not (len(args) == 1 and (scalar_view or result in {'object', 'str'})):
        return None
    if (len(args) not in (1, 2, 3) or len(body) != 1 or not isinstance(body[0], cst.If)
            or any(arg.annotation is None for arg in args)):
        return None
    if len(args) > 1 and dotted_name(args[1].annotation.annotation) != f'{CY_MOD}.longlong':
        return None
    if len(args) == 3 and dotted_name(args[2].annotation.annotation) != f'{CY_MOD}.bint':
        return None
    branch, guard = body[0], None
    if len(args) > 1 and not isinstance(branch.test, cst.Subscript):
        nested = statements(branch.body)
        if len(nested) != 1 or not isinstance(nested[0], cst.If):
            return None
        guard, branch = branch.test, nested[0]
        if not _pure_index(guard, {arg.name.value for arg in args[1:]}):
            return None
    body = statements(branch.body)
    if len(body) != 1 or not isinstance(body[0], cst.Return):
        return None
    ready, value = branch.test, body[0].value
    if len(args) > 1:
        index, value_index = _index(ready), _index(value)
        if (index is None or value_index is None or not same_expression(index, value_index)
                or not _pure_index(index, {args[1].name.value})):
            return None
        ready_field, value_field = ready.value, value.value
    else:
        ready_field, value_field = ready, value
    if not (isinstance(ready_field, cst.Attribute) and isinstance(value_field, cst.Attribute)
            and isinstance(ready_field.value, cst.Name) and ready_field.value.value == args[0].name.value
            and isinstance(value_field.value, cst.Name) and value_field.value.value == args[0].name.value
            and ready_field.attr.value.startswith('_has_') and value_field.attr.value.startswith('_v_')
            and ready_field.attr.value[5:] == value_field.attr.value[3:]):
        return None
    test = cst.BooleanOperation(guard, cst.And(), ready) if guard is not None else ready
    return _Hit('scalar' if len(args) == 1 else 'dense', args[0].annotation.annotation,
                tuple((arg.name.value, arg.annotation.annotation) for arg in args), test, value)


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


def expose_entries(state: EntryModule, known, mode):
    functions = state.functions
    layouts = {name: layout for name in (*known, *state.inbounds_entries)
               if (fn := functions.get(name)) is not None
               and (layout := _layout(fn, name in known and known[name].scalar_view))
               and (mode in (layout.kind, 'both') or mode == 'auto' and (
                   layout.kind == 'scalar' or isinstance(layout.test, cst.Subscript)))}
    return state.replace(state.syntax.visit(_Calls(functions, layouts)))
