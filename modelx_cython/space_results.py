"""Native calls through an exact ItemSpace already held in an owned Cell cache."""

import libcst as cst
from libcst.helpers import parse_template_expression
from libcst.metadata.scope_provider import BuiltinAssignment, GlobalScope, ImportAssignment

from .consts import CY_MOD, HAS_PREF, MX_SELF, MX_SYS_MOD, SPACE_PREF, VAR_PREF
from .numeric_cache import exclusively_owned
from .powers import KIND_FLOAT
from .scopes import Names
from .typedefs import primitive_kind
from .syntax import arguments


_NATIVE_RESULTS = {'double', 'long long', 'bint'}


def pin_function(name, owner, producer):
    """The held item, borrowed: read once, at receiver evaluation, as Python orders it.

    Cython can defer a bare cast of an object field until after computing the arguments; a C function
    returning the field forces the read at receiver evaluation. It returns the pointer without a new
    reference, so a read of an item shared by several threads (an item of a shared Space held by each
    worker's own items) writes nothing to that item's reference count. The holder keeps it alive: the
    call site reads it only when ``root`` is exactly the compiled class and its flag is set; ``root`` is
    alive for the whole call (it is the running formula's ``self``); and the cached field of a
    zero-argument Cell of an exclusively owned item is written once, before its flag, and never again
    while the item lives (an exported model has no cache clearing; releasing the item from its Space
    clears its Cells, roots and Space only). Deleting the held item from its own Space or closing the
    model leaves the holder's reference in place. Only an unlocked (exclusively owned) producer is pinned.
    """
    pointer = f'{CY_MOD}.pointer({CY_MOD}.void)'
    return cst.parse_statement(f'@{CY_MOD}.cfunc\n@{CY_MOD}.inline\n@{CY_MOD}.exceptval(check=False)\n'
                               f'def {name}(root: {owner}) -> {pointer}:\n'
                               f'    return {CY_MOD}.cast({pointer}, root.{VAR_PREF}{producer})\n')


def _traced_target(cell, allow_locked=False):
    """Use the observed exact class to select a guard, never to narrow storage."""
    if (cell is None or cell.is_special() or not cell._can_bind_native_result
            or cell.get_rettype_expr(c_style=True) != 'object'
            or not (exclusively_owned(cell) or allow_locked and cell.is_locked)):
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
    def __init__(self, owner, names, integer_params, operand_kind, pin, return_type, allow_locked):
        self.owner = owner
        self.names = names
        self.locals = Names(used=names)
        self.integer_params = integer_params
        self.operand_kind = operand_kind
        self.pin = pin
        self.return_type = return_type
        self.allow_locked = allow_locked
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
        if (not isinstance(call.func, cst.Attribute)
                or not isinstance(call.func.value, cst.Name) or call.func.value.value != MX_SELF
                or any(arg.keyword is not None or arg.star for arg in original.args)):
            return updated
        producer = call.func.attr.value
        producer_cell = self.owner.cells.get(producer)
        target = _traced_target(producer_cell, self.allow_locked)
        if target is None or target.name in self.names:
            return updated
        parameterized = producer_cell.has_args()
        if producer_cell.is_locked and original not in self.double_returns:
            return updated
        if parameterized:
            if (not call.args or original not in self.double_returns
                    or any(arg.keyword is not None or arg.star for arg in call.args)):
                return updated
        elif call.args:
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
            rounded = arguments(constructor[0] if constructor else value, f'{MX_SYS_MOD}._mx_round_one', 3)
            if rounded:
                helper = '_mx_rounded_index' if constructor else '_mx_bare_round_index'
                prefix = (cst.Arg(value.func),) if constructor else ()
                conversion = cst.Call(cst.parse_expression(f'{MX_SYS_MOD}.{helper}'),
                                      (*prefix, *(cst.Arg(v) for v in rounded), path))
            else:
                conversion = cst.Call(cst.parse_expression(f'{MX_SYS_MOD}._mx_integer_result'),
                                      (cst.Arg(value), path))
            args[i] = args[i].with_changes(value=conversion)

        if parameterized or producer_cell.is_locked:
            # Evaluate the real producer once, preserving overrides and its cache.
            # The strong local survives argument evaluation in both branches.
            selected = self.locals.reserve('_mx_selected_item')
            ready = parse_template_expression(f'type(({selected} := {{receiver}})) is {target.name} '
                                              f'and type({MX_SELF}) is {self.owner.name}',
                                              receiver=updated.func.value)
            cast = cst.parse_expression(f'{CY_MOD}.cast({target.name}, {selected})')
            fallback = updated.with_changes(func=cst.Attribute(cst.Name(selected), func.attr),
                                            lpar=(), rpar=())
        else:
            field = f'{MX_SELF}.{VAR_PREF}{producer}'
            ready = cst.parse_expression(
                f'type({MX_SELF}) is {self.owner.name} and {MX_SELF}.{HAS_PREF}{producer} '
                f'and type({field}) is {target.name}')
            pin = self.pin(self.owner.name, producer)
            cast = cst.parse_expression(f'{CY_MOD}.cast({target.name}, {pin}({MX_SELF}))')
            fallback = updated.with_changes(lpar=(), rpar=())
        native = updated.with_changes(func=cst.Attribute(cast, func.attr), args=args, lpar=(), rpar=())
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
    allow_locked = cell.is_locked and owner.module.spec.get_spec(owner.fqname).get('native_locked_result_calls', False)
    if (not (exclusively_owned(cell) or allow_locked) or 'object' in names or 'type' in names or owner.name in names
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
    calls = _HeldCalls(owner, names, integers, operand_kind, pin, cell.get_rettype_expr(c_style=True), allow_locked)
    body = node.body.visit(calls)
    return (node.with_changes(body=body), True) if calls.used else (node, False)
