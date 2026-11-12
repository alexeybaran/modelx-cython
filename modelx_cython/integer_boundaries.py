"""Check integer Cell results and ItemSpace keys at their native boundaries."""

import libcst as cst

from .checked_arithmetic import CheckedArithmetic
from .consts import CY_MOD, FORMULA_PREF, MX_SELF, MX_SYS_MOD
from .powers import KIND_FLOAT, OperandKind, kind_of_type_expr
from .scopes import Names
from .syntax import walk
from .usage import RESOLVED, _extract_self_chain


_INTEGER_ANNOTATION = cst.Attribute(cst.Name(CY_MOD), cst.Name('longlong'))


_CALL_GATE = """\
@{cy}.cfunc
def {name}(method: object, params: tuple, path: str) -> object:
    from functools import wraps

    @wraps(method)
    def checked(*args, **kwargs):
        values = list(args)
        for position, name in params:
            if position < len(values):
                values[position] = {runtime}._mx_exact_int64(values[position], path)
            if name in kwargs:
                kwargs[name] = {runtime}._mx_exact_int64(kwargs[name], path)
        return method(*values, **kwargs)

    return checked
"""


def integer_parameters(cell):
    if cell.is_special() or cell.body_has_closure or not cell.has_typeinfo():
        return ()
    return tuple((index, name) for index, name in enumerate(cell.params)
                 if cell.get_argtype_expr(name, c_style=True) == 'long long')


def _checked_integer(value, path):
    target = cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name('_mx_integer_result'))
    return cst.Call(target, (cst.Arg(value), cst.Arg(cst.SimpleString(repr(path)))))


def _python_call_gate(cls, entries, names, call_gate):
    """Check Python-bound Cells while preserving their Cython vtable types."""
    routes = {cell.name: (entries[cell.fqname], cell.model_path)
              for cell in cls.cells.values() if cell.fqname in entries}
    if not routes:
        return (), ()
    field = names.reserve(f'_mx_integer_arguments_{cls.name}')
    declaration = cst.parse_statement(f'{field} = {routes!r}\n')
    method = cst.parse_statement(f"""\
def __getattribute__(self, name):
    method = {MX_SYS_MOD}._mx_generic_getattr(self, name)
    entry = {field}.get(name)
    if entry is not None and (
        type(self) is {cls.name}
        or getattr(type(self), name) is getattr({CY_MOD}.cast(object, {cls.name}), name)
    ):
        return {call_gate}(method, entry[0], entry[1])
    return method
""")
    return (declaration,), (method,)


class _GuardCalls(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider,)

    def __init__(self, receiver, integer_params, owner, resolver, entries, cells):
        self.receiver = receiver
        self.integer_params = integer_params
        self.owner = owner
        self.resolver = resolver
        self.entries = entries
        self.cells = cells

    def _native_parameter(self, value):
        if not isinstance(value, cst.Name) or value.value not in self.integer_params:
            return False
        scope = self.get_metadata(cst.metadata.ScopeProvider, value, None)
        bindings = tuple(scope[value.value]) if scope is not None else ()
        return len(bindings) == 1 and bindings[0].node is self.integer_params[value.value]

    def leave_Call(self, before, after):
        chain = _extract_self_chain(before.func)
        if not isinstance(chain, list) or any(arg.star for arg in before.args):
            return after
        base = before.func
        while isinstance(base, cst.Attribute):
            base = base.value
        scope = self.get_metadata(cst.metadata.ScopeProvider, base, None)
        assignments = tuple(scope[MX_SELF]) if scope is not None else ()
        if len(assignments) != 1 or assignments[0].node is not self.receiver:
            return after
        status, target = self.resolver.resolve_chain(self.owner, chain)
        if status != RESOLVED or not (params := self.entries.get(target)):
            return after
        cell = self.cells[target]
        if cell.called_with_kwargs:
            # A Python-dispatched method goes through the normal attribute gate.
            return after
        guarded = dict(params)
        arguments = tuple(arg.with_changes(value=_checked_integer(arg.value, cell.model_path))
                          if index in guarded and arg.keyword is None and not self._native_parameter(old.value)
                          else arg for index, (old, arg) in enumerate(zip(before.args, after.args)))
        return after.with_changes(args=arguments)


def _guard_calls(node, cls, resolver, entries, cells):
    params = (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)
    if not params or params[0].name.value != MX_SELF:
        return node
    integer_params = {param.name.value: param for param in params if param.annotation is not None
                      and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    wrapper = cst.MetadataWrapper(cst.Module(body=(node,)), unsafe_skip_copy=True)
    return wrapper.visit(_GuardCalls(params[0], integer_params, cls.fqname, resolver, entries, cells)).body[0]


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
    def __init__(self, path, kinds, native_helpers, arithmetic):
        self.path = path
        self.kinds = kinds
        self.native_helpers = native_helpers
        self.arithmetic = arithmetic
        self.check_overflow = False

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
                or (isinstance(value, cst.Call) and isinstance(value.func, cst.Name)
                    and value.func.value in self.native_helpers)
                or (in_conditional and self.kinds.kind(value) == KIND_FLOAT)):
            return value
        self.check_overflow |= any(
            isinstance(part, cst.BinaryOperation)
            and isinstance(part.operator, (cst.Add, cst.Subtract, cst.Multiply))
            and self.arithmetic.integer_interval(part, signed=True) is None
            for part in walk(value)
        )
        return _checked_integer(value, self.path)


def _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds, native_helpers=()):
    if node.returns is None or not node.returns.annotation.deep_equals(_INTEGER_ANNOTATION):
        return node
    params = {name: kind_of_type_expr(cell.get_argtype_expr(name, c_style=True, formula=True))
              for name in cell.params}
    kinds = OperandKind(resolver, cells_kinds, ref_kinds, cls.fqname, params)
    arithmetic = CheckedArithmetic(cls, cell)
    arithmetic.parameters = {
        param.name.value for param in node.params.params
        if param.annotation is not None and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)
    }
    guard = _GuardReturns(cell.model_path, kinds, native_helpers, arithmetic)
    body = node.body.visit(guard)
    if not guard.check_overflow or not isinstance(body, cst.IndentedBlock):
        return node.with_changes(body=body)
    error = Names(node).reserve('_mx_integer_overflow')
    message = f'{cell.model_path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec.'
    handler = cst.parse_statement(
        f'try:\n    pass\nexcept OverflowError as {error}:\n    raise OverflowError({message!r}) from {error}\n'
    )
    doc_end = 1 if node.get_docstring() is not None else 0
    handler = handler.with_changes(body=cst.IndentedBlock(body=body.body[doc_end:]))
    decorator = cst.Decorator(cst.parse_expression(f'{CY_MOD}.overflowcheck(True)'))
    return node.with_changes(decorators=(*node.decorators, decorator),
                             body=body.with_changes(body=(*body.body[:doc_end], handler)))


def guard_integer_boundaries(state, module_info, resolver, cells_kinds, ref_kinds, cells):
    """Check formula results, arguments and ItemSpace keys after entry lowering."""
    entries = {name: params for name, cell in cells.items() if (params := integer_parameters(cell))}
    classes = []
    declarations = []
    call_gate = state.names.reserve('_mx_checked_cell_callable')
    changed = False
    native_helpers = {}
    for node in state.syntax.body:
        if (isinstance(node, cst.FunctionDef) and node.returns is not None
                and node.returns.annotation.deep_equals(_INTEGER_ANNOTATION)
                and (origin := state.function_origins.get(node.name.value)) is not None):
            native_helpers.setdefault(origin, set()).add(node.name.value)
    for node in state.syntax.body:
        cls = module_info.classes.get(node.name.value) if isinstance(node, cst.ClassDef) else None
        if cls is None:
            origin = state.function_origins.get(node.name.value) if isinstance(node, cst.FunctionDef) else None
            if origin:
                owner, name = origin
                cls = module_info.classes[owner]
                cell = cls.cells[name[len(FORMULA_PREF):]]
                node = _guard_calls(node, cls, resolver, entries, cells)
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
                method = _guard_calls(method, cls, resolver, entries, cells)
                methods.append(_guard_result(method, formula, cls, resolver, cells_kinds, ref_kinds,
                                             native_helpers.get((cls.name, name), ())))
                changed = True
                continue
            cell = cls.cells.get(name)
            if name == '__call__' and cell is not None and cell.has_typeinfo():
                methods.append(_guard_space_parameters(method, cell))
                changed = True
                continue
            if cell is not None and not cell.has_formula_def:
                method = _guard_calls(method, cls, resolver, entries, cells)
                method = _guard_result(method, cell, cls, resolver, cells_kinds, ref_kinds)
                changed = True
            methods.append(method)
        added, gates = _python_call_gate(cls, entries, state.names, call_gate)
        declarations.extend(added)
        changed |= bool(gates)
        classes.append(node.with_changes(body=node.body.with_changes(body=(*methods, *gates))))
    if declarations:
        classes.extend(declarations)
        classes.append(cst.parse_statement(_CALL_GATE.format(
            cy=CY_MOD, name=call_gate, runtime=MX_SYS_MOD)))
    return state.replace(state.syntax.with_changes(body=classes)) if changed else state
