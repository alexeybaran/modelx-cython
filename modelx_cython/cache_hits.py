"""Expose existing native cache hits; keep every miss on its checked entry."""
import ast
from dataclasses import dataclass

import libcst as cst

from modelx_cython.consts import CY_MOD
from modelx_cython.dense_proofs import _binding_names

_NATIVE = {f'{CY_MOD}.{name}' for name in ('bint', 'double', 'longlong')}


def _pure_index(node, parameters):
    allowed = (ast.Name, ast.Load, ast.Constant, ast.Compare, ast.Lt, ast.LtE,
               ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or, ast.BinOp, ast.Sub,
               ast.UnaryOp, ast.USub, ast.UAdd, ast.Call, ast.Attribute)
    for part in ast.walk(node):
        if not isinstance(part, allowed):
            return False
        if isinstance(part, ast.Name) and part.id not in {*parameters, CY_MOD}:
            return False
        if isinstance(part, ast.Constant) and type(part.value) is not int:
            return False
        if isinstance(part, ast.Attribute) and not (
                isinstance(part.value, ast.Name) and part.value.id == CY_MOD
                and part.attr in {'cast', 'ulonglong'}):
            return False
        if isinstance(part, ast.Call) and not (
                ast.unparse(part.func) == f'{CY_MOD}.cast' and len(part.args) == 2
                and not part.keywords
                and ast.unparse(part.args[0]) == f'{CY_MOD}.ulonglong'):
            return False
    return True


@dataclass(frozen=True)
class _Hit:
    kind: str
    receiver: str
    parameters: tuple
    test: str
    value: str


def _layout(fn):
    if fn.returns is None or ast.unparse(fn.returns) not in _NATIVE:
        return None
    args = fn.args.args
    if (len(args) not in (1, 2, 3) or len(fn.body) != 1
            or not isinstance(fn.body[0], ast.If)
            or any(arg.annotation is None for arg in args)):
        return None
    if len(args) > 1 and ast.unparse(args[1].annotation) != f'{CY_MOD}.longlong':
        return None
    if len(args) == 3 and ast.unparse(args[2].annotation) != f'{CY_MOD}.bint':
        return None
    branch = fn.body[0]
    guard = None
    if len(args) > 1 and not isinstance(branch.test, ast.Subscript):
        if len(branch.body) != 1 or not isinstance(branch.body[0], ast.If):
            return None
        guard, branch = branch.test, branch.body[0]
        if not _pure_index(guard, {arg.arg for arg in args[1:]}):
            return None
    if len(branch.body) != 1 or not isinstance(branch.body[0], ast.Return):
        return None
    ready, value = branch.test, branch.body[0].value
    if len(args) > 1:
        if not (isinstance(ready, ast.Subscript) and isinstance(value, ast.Subscript)
                and ast.dump(ready.slice) == ast.dump(value.slice)
                and _pure_index(ready.slice, {args[1].arg})):
            return None
        ready_field, value_field = ready.value, value.value
    else:
        ready_field, value_field = ready, value
    if not (isinstance(ready_field, ast.Attribute) and isinstance(value_field, ast.Attribute)
            and isinstance(ready_field.value, ast.Name) and ready_field.value.id == args[0].arg
            and isinstance(value_field.value, ast.Name) and value_field.value.id == args[0].arg
            and ready_field.attr.startswith('_has_') and value_field.attr.startswith('_v_')
            and ready_field.attr[5:] == value_field.attr[3:]):
        return None
    test = ast.BoolOp(op=ast.And(), values=[guard, ready]) if guard is not None else ready
    return _Hit('scalar' if len(args) == 1 else 'dense', ast.unparse(args[0].annotation),
                tuple((arg.arg, ast.unparse(arg.annotation)) for arg in args),
                ast.unparse(test), ast.unparse(value))


def _targets(functions, entries):
    """Include the shared inbounds body reached by a generated checked entry."""
    names = set(entries)
    for name in entries:
        fn = functions.get(name)
        if fn is None or len(fn.body) != 1 or not isinstance(fn.body[0], ast.If):
            continue
        body = fn.body[0].body
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            continue
        call = body[0].value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
            target = functions.get(call.func.id)
            if (target is not None and target.args.args and fn.args.args
                    and ast.dump(target.args.args[0].annotation) == ast.dump(fn.args.args[0].annotation)):
                names.add(target.name)
    return names


class _Substitute(cst.CSTTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def leave_Name(self, original, updated):
        return self.mapping.get(original.value, updated)

    def leave_Attribute(self, original, updated):
        # Parameter names can equal an attribute such as "cast" or a field.
        return updated.with_changes(attr=original.attr)


class _Calls(cst.CSTTransformer):
    def __init__(self, functions, layouts):
        self.functions, self.layouts = functions, layouts
        self.current = None

    def visit_ClassDef(self, node):
        return False

    def visit_FunctionDef(self, node):
        fn = self.functions.get(node.name.value)
        if self.current is not None or fn is None or not fn.name.startswith('_mx_dense_formula_'):
            return False
        nodes = [n for stmt in fn.body for n in ast.walk(stmt)]
        if (CY_MOD in _binding_names(nodes) or any(a.arg == CY_MOD for a in fn.args.args)
                or any(isinstance(n, getattr(ast, 'Match', ())) for n in nodes)):
            return False
        self.current = fn
        return True

    def leave_FunctionDef(self, original, updated):
        if self.current is not None and original.name.value == self.current.name:
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
        parameters = self.current.args.args
        if (not parameters or parameters[0].annotation is None
                or ast.unparse(parameters[0].annotation) != layout.receiver
                or len(updated.args) != len(layout.parameters)
                or any(arg.keyword or arg.star for arg in updated.args)
                or not isinstance(updated.args[0].value, cst.Name)
                or updated.args[0].value.value != parameters[0].arg):
            return updated
        types = {arg.arg: ast.unparse(arg.annotation) for arg in parameters if arg.annotation is not None}
        mapping = {layout.parameters[0][0]: updated.args[0].value}
        for (name, annotation), arg in zip(layout.parameters[1:], updated.args[1:]):
            expression = ast.parse(cst.Module([]).code_for_node(arg.value), mode='eval').body
            if isinstance(expression, ast.Name):
                if types.get(expression.id) != annotation:
                    return updated
            else:
                try:
                    value = ast.literal_eval(expression)
                except (ValueError, TypeError):
                    return updated
                if not ((annotation == f'{CY_MOD}.bint' and type(value) is bool)
                        or (annotation == f'{CY_MOD}.longlong' and type(value) is int
                            and -(1 << 63) <= value < (1 << 63))):
                    return updated
            mapping[name] = arg.value
        substitute = _Substitute(mapping)
        return cst.IfExp(test=cst.parse_expression(layout.test).visit(substitute),
                         body=cst.parse_expression(layout.value).visit(substitute), orelse=updated,
                         lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))


def expose_cache_hits(sources, entries, mode):
    """Only already-private, typed calls inherit their existing hit contract."""
    if not hasattr(ast, 'unparse'):
        return
    for module, known in entries.items():
        path = sources[module]
        source = path.read_bytes()
        functions = {fn.name: fn for fn in ast.parse(source).body if isinstance(fn, ast.FunctionDef)}
        layouts = {name: layout for name in _targets(functions, known)
                   if (fn := functions.get(name)) is not None and (layout := _layout(fn))
                   and (mode in (layout.kind, 'both') or mode == 'auto' and (
                       layout.kind == 'scalar' or isinstance(ast.parse(layout.test, mode='eval').body, ast.Subscript)))}
        updated = cst.parse_module(source).visit(_Calls(functions, layouts)).bytes
        if updated != source:
            path.write_bytes(updated)
