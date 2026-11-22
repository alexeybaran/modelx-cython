"""Native calls through an exact ItemSpace already held in an owned Cell cache."""

import libcst as cst
from libcst.metadata.scope_provider import BuiltinAssignment, GlobalScope, ImportAssignment

from .consts import CY_MOD, HAS_PREF, MX_SELF, MX_SYS_MOD, SPACE_PREF, VAR_PREF
from .numeric_cache import exclusively_owned
from .powers import KIND_FLOAT
from .scopes import Names
from .typedefs import primitive_kind
from .syntax import arguments


_NATIVE_RESULTS = {'double', 'long long', 'bint'}


def pin_function(name):
    # Cython can defer a bare cast of an object field until after computing
    # the arguments. A C function returning the object forces an owned
    # temporary at receiver evaluation, just as the ordinary call does.
    return cst.parse_statement(f'@{CY_MOD}.cfunc\n@{CY_MOD}.inline\n'
                               f'def {name}(value: object) -> object:\n    return value\n')


def _traced_target(cell):
    """Use the observed exact class to select a guard, never to narrow storage."""
    if (cell is None or cell.has_args() or cell.is_special() or not cell._can_bind_native_result
            or cell.get_rettype_expr(c_style=True) != 'object' or not exclusively_owned(cell)):
        return None
    trace = getattr(cell._rt, 'ret_type', None)
    if trace is None or trace.is_array:
        return None
    value_type = trace.value_type
    name = getattr(value_type, '__name__', '')
    module = cell.parent.module
    if (not isinstance(value_type, type) or not name.startswith(SPACE_PREF)
            or value_type.__module__.split('.')[1:] != module.fqname.split('.')[1:]
            or name not in module.visitor.itemspace_classes or name in module.visitor.dynamic_ref_classes):
        return None
    return module.classes.get(name)


def _binds_parameter(scope, name):
    bindings = tuple(scope[name])
    return len(bindings) == 1 and isinstance(bindings[0].node, cst.Param)


class _HeldCalls(cst.CSTTransformer):
    def __init__(self, owner, names, integer_params, operand_kind, pin, return_type):
        self.owner = owner
        self.names = names
        self.integer_params = integer_params
        self.operand_kind = operand_kind
        self.pin = pin
        self.return_type = return_type
        self.double_returns = set()
        self.used = False

    def visit_FunctionDef(self, node):
        return False

    visit_ClassDef = visit_Lambda = visit_GeneratorExp = visit_FunctionDef
    visit_ListComp = visit_SetComp = visit_DictComp = visit_FunctionDef

    def visit_Return(self, node):
        if self.return_type == 'double' and isinstance(node.value, cst.Call):
            self.double_returns.add(node.value)

    def _native_argument(self, arg, ctype):
        if ctype == 'double':
            return self.operand_kind.kind(arg) == KIND_FLOAT
        if ctype == 'long long':
            if isinstance(arg, cst.Name):
                return arg.value in self.integer_params
            return isinstance(arg, cst.Integer) and primitive_kind(arg.evaluated_value) == 'longlong'
        return False

    def leave_Call(self, original, updated):
        func = original.func
        if not isinstance(func, cst.Attribute) or not isinstance(func.value, cst.Call):
            return updated
        call = func.value
        if (call.args or not isinstance(call.func, cst.Attribute)
                or not isinstance(call.func.value, cst.Name) or call.func.value.value != MX_SELF
                or any(arg.keyword is not None or arg.star for arg in original.args)):
            return updated
        producer = call.func.attr.value
        target = _traced_target(self.owner.cells.get(producer))
        if target is None or target.name in self.names:
            return updated
        method = target.cells.get(func.attr.value)
        if (method is None or method.is_special() or not method._can_bind_native_result
                or not method.has_typeinfo() or method.params_with_defaults or len(original.args) != len(method.params)
                or method.get_rettype_expr(c_style=True) not in _NATIVE_RESULTS):
            return updated

        args = list(updated.args)
        checked = False
        for i, (arg, param) in enumerate(zip(original.args, method.params)):
            ctype = method.get_argtype_expr(param, c_style=True)
            if self._native_argument(arg.value, ctype):
                # The Python gate converts after evaluating all arguments. Only
                # inert integer loads may follow a conversion moved to the call.
                if checked and ctype != 'long long':
                    return updated
                continue
            if checked or ctype != 'long long' or MX_SYS_MOD in self.names:
                return updated
            checked = True
            value = args[i].value
            path = cst.Arg(cst.SimpleString(repr(method.model_path)))
            constructor = arguments(value, 'int', 1)
            rounded = arguments(constructor[0], f'{MX_SYS_MOD}._mx_round_one', 3) if constructor else None
            if rounded:
                conversion = cst.Call(cst.parse_expression(f'{MX_SYS_MOD}._mx_rounded_index'),
                                      (cst.Arg(value.func), *(cst.Arg(v) for v in rounded), path))
            else:
                conversion = cst.Call(cst.parse_expression(f'{MX_SYS_MOD}._mx_integer_result'),
                                      (cst.Arg(value), path))
            args[i] = args[i].with_changes(value=conversion)

        field = f'{MX_SELF}.{VAR_PREF}{producer}'
        ready = cst.parse_expression(
            f'type({MX_SELF}) is {self.owner.name} and {MX_SELF}.{HAS_PREF}{producer} '
            f'and type({field}) is {target.name}')
        cast = cst.parse_expression(f'{CY_MOD}.cast({target.name}, {self.pin}({field}))')
        native = updated.with_changes(func=cst.Attribute(cast, func.attr), args=args, lpar=(), rpar=())
        fallback = updated.with_changes(lpar=(), rpar=())
        if original in self.double_returns and method.get_rettype_expr(c_style=True) == 'double':
            # A C-double formula immediately converts the result at return.
            # Keep this conversion in the dynamic arm, including its exceptions.
            fallback = cst.Call(cst.parse_expression(f'{CY_MOD}.cast'), (
                cst.Arg(cst.parse_expression(f'{CY_MOD}.double')), cst.Arg(fallback)))
        else:
            native = cst.Call(cst.parse_expression(f'{CY_MOD}.cast'), (
                cst.Arg(cst.Name('object')), cst.Arg(native)))
        self.used = True
        return cst.IfExp(test=ready, body=native, orelse=fallback,
                         lpar=(cst.LeftParen(), *updated.lpar), rpar=(*updated.rpar, cst.RightParen()))


def _runtime_import(binding):
    node = binding.node
    return (isinstance(binding, ImportAssignment) and isinstance(binding.scope, GlobalScope)
            and isinstance(node, cst.ImportFrom) and node.module is None and len(node.relative) == 1
            and not isinstance(node.names, cst.ImportStar)
            and any(isinstance(alias.name, cst.Name) and alias.name.value == MX_SYS_MOD
                    and alias.asname is None for alias in node.names))


def lower(node, owner, cell, scope, operand_kind, pin):
    """Leave the cold or unexpected result expression in its original evaluation order."""
    names = Names(node).names
    if (not exclusively_owned(cell) or 'object' in names or 'type' in names or owner.name in names
            or not _binds_parameter(scope, MX_SELF) or tuple(scope[CY_MOD])
            or not all(scope[name] and all(isinstance(binding, BuiltinAssignment) for binding in scope[name])
                       for name in ('type', 'object'))):
        return node, False
    integers = {param for param in cell.params if _binds_parameter(scope, param)
                and cell.get_argtype_expr(param, c_style=True, formula=True) == 'long long'}
    # An earlier transform may have introduced a qualified runtime helper
    # (for example for round). Its presence is safe with the export's exact import.
    bindings = tuple(scope[MX_SYS_MOD])
    if MX_SYS_MOD in names and len(bindings) == 1 and _runtime_import(bindings[0]):
        names = names - {MX_SYS_MOD}
    calls = _HeldCalls(owner, names, integers, operand_kind, pin, cell.get_rettype_expr(c_style=True))
    body = node.body.visit(calls)
    return (node.with_changes(body=body), True) if calls.used else (node, False)
