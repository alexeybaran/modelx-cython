"""Check integer Cell results and ItemSpace keys at their native boundaries."""

import libcst as cst

from .consts import CY_MOD, FORMULA_PREF, MX_SYS_MOD
from .powers import KIND_FLOAT, OperandKind, kind_of_type_expr


_INTEGER_ANNOTATION = cst.Attribute(cst.Name(CY_MOD), cst.Name('longlong'))


def _object_parameters(params, checked):
    return tuple(param.with_changes(annotation=cst.Annotation(cst.Name('object')))
                 if param.name.value in checked else param for param in params)


def _guard_space_parameters(node, cell):
    checked = tuple(name for name in cell.params
                    if cell.get_argtype_expr(name, c_style=True) == 'long long')
    if not checked:
        return node
    assignments = tuple(cst.parse_statement(
        f'{name} = {MX_SYS_MOD}._mx_exact_int64({name}, {cell.model_path!r})\n')
        for name in checked)
    params = _object_parameters(node.params.params, checked)
    statements = node.body.body
    doc_end = 1 if node.get_docstring() is not None else 0
    return node.with_changes(params=node.params.with_changes(params=params),
                             body=node.body.with_changes(
                                 body=(*statements[:doc_end], *assignments, *statements[doc_end:])))


class _GuardReturns(cst.CSTTransformer):
    def __init__(self, path, kinds):
        self.path = path
        self.kinds = kinds

    def visit_FunctionDef(self, node):
        return False

    visit_ClassDef = visit_FunctionDef

    def leave_Return(self, before, after):
        if before.value is None:
            return after
        return after.with_changes(value=self._checked(before.value))

    def _checked(self, value, in_conditional=False):
        if isinstance(value, cst.IfExp):
            # Keep the conditional's evaluation order. Leave a statically
            # floating branch for Cython's existing narrow-return diagnostic.
            return value.with_changes(body=self._checked(value.body, in_conditional=True),
                                      orelse=self._checked(value.orelse, in_conditional=True))
        if ((isinstance(value, cst.Integer) and value.evaluated_value < (1 << 63))
                or (in_conditional and self.kinds.kind(value) == KIND_FLOAT)):
            return value
        target = cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name('_mx_integer_result'))
        args = (cst.Arg(value), cst.Arg(cst.SimpleString(repr(self.path))))
        return cst.Call(target, args)


def _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds):
    if node.returns is None or not node.returns.annotation.deep_equals(_INTEGER_ANNOTATION):
        return node
    params = {name: kind_of_type_expr(cell.get_argtype_expr(name, c_style=True, formula=True))
              for name in cell.params}
    kinds = OperandKind(resolver, cells_kinds, ref_kinds, cls.fqname, params)
    return node.with_changes(body=node.body.visit(_GuardReturns(cell.model_path, kinds)))


def guard_integer_boundaries(state, module_info, resolver, cells_kinds, ref_kinds):
    """Check formula results and ItemSpace keys after entry lowering."""
    classes = []
    changed = False
    for node in state.syntax.body:
        cls = module_info.classes.get(node.name.value) if isinstance(node, cst.ClassDef) else None
        if cls is None:
            origin = state.function_origins.get(node.name.value) if isinstance(node, cst.FunctionDef) else None
            if origin:
                owner, name = origin
                cls = module_info.classes[owner]
                cell = cls.cells[name[len(FORMULA_PREF):]]
                node = _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds)
                changed = True
            classes.append(node)
            continue
        methods = []
        for method in node.body.body:
            if not isinstance(method, cst.FunctionDef):
                methods.append(method)
                continue
            name = method.name.value
            if name.startswith(FORMULA_PREF) and (formula := cls.cells.get(name[len(FORMULA_PREF):])):
                methods.append(_guard_result(method, formula, cls, resolver, cells_kinds, ref_kinds))
                changed = True
                continue
            cell = cls.cells.get(name)
            if name == '__call__' and cell is not None and cell.has_typeinfo():
                methods.append(_guard_space_parameters(method, cell))
                changed = True
                continue
            if cell is not None and not cell.has_formula_def:
                method = _guard_result(method, cell, cls, resolver, cells_kinds, ref_kinds)
                changed = True
            methods.append(method)
        classes.append(node.with_changes(body=node.body.with_changes(body=methods)))
    return state.replace(state.syntax.with_changes(body=classes)) if changed else state
