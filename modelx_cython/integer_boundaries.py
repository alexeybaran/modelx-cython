"""Check integer Cell results and ItemSpace keys at their native boundaries."""

import libcst as cst

from .checked_arithmetic import CheckedArithmetic
from .consts import CY_MOD, FORMULA_PREF, MX_SELF, MX_SYS_MOD
from .powers import KIND_FLOAT, OperandKind, kind_of_type_expr
from .scopes import CstFacts, Names, OPAQUE_SCOPES, lexical_nodes
from .syntax import dotted_name, walk
from .usage import RESOLVED, _extract_self_chain


_INTEGER_ANNOTATION = cst.Attribute(cst.Name(CY_MOD), cst.Name('longlong'))
_SIGNED_OPERATIONS = {
    cst.Add: '_mx_signed_add',
    cst.Subtract: '_mx_signed_sub',
    cst.Multiply: '_mx_signed_mul',
    cst.FloorDivide: '_mx_signed_floor',
}
_INTEGER_CONVERSION = f'{MX_SYS_MOD}._mx_integer_result'
_SIGNED_HELPERS = frozenset((*_SIGNED_OPERATIONS.values(), '_mx_integer_result', '_mx_exact_int64'))


def _boxed_arithmetic(value):
    if isinstance(value, cst.Call):
        return (dotted_name(value.func) == f'{CY_MOD}.cast' and len(value.args) == 2
                and isinstance(value.args[0].value, cst.Name) and value.args[0].value.value == 'object')
    return (isinstance(value, cst.BinaryOperation)
            and type(value.operator) in _SIGNED_OPERATIONS
            and (_boxed_arithmetic(value.left) or _boxed_arithmetic(value.right)))


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


def _boundary_call(function, path, *values):
    target = cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(function))
    return cst.Call(target, tuple(cst.Arg(value) for value in values) +
                    (cst.Arg(cst.SimpleString(repr(path))),))


def _checked_integer(value, path):
    return _boundary_call('_mx_integer_result', path, value)


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


class _GuardIntegers(cst.CSTTransformer):
    def __init__(self, path, kinds, native_helpers, arithmetic, signed_fields,
                 signed_functions, signed_methods, signed_names, signed_return):
        self.path = path
        self.kinds = kinds
        self.native_helpers = native_helpers
        self.arithmetic = arithmetic
        self.check_overflow = False
        self.signed_fields = signed_fields
        self.signed_functions = signed_functions
        self.signed_methods = signed_methods
        self.signed_names = signed_names
        self.signed_return = signed_return

    def on_visit(self, node):
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)

    def leave_Return(self, before, after):
        if not self.signed_return or after.value is None:
            return after
        return after.with_changes(value=self._checked(after.value))

    def leave_Call(self, before, after):
        if (dotted_name(after.func) != _INTEGER_CONVERSION or len(after.args) != 2
                or not isinstance(after.args[1].value, cst.SimpleString)):
            return after
        operand, path = after.args
        value, signed = self._lower(operand.value, path.value.evaluated_value)
        if signed:
            return value.with_changes(lpar=(*after.lpar, *value.lpar), rpar=(*value.rpar, *after.rpar))
        return after.with_changes(args=(operand.with_changes(value=value), path))

    def _signed_expression(self, value, path):
        """Lower signed C expressions, leaving object operators on Cython's object path."""
        if isinstance(value, cst.BinaryOperation) and type(value.operator) in _SIGNED_OPERATIONS:
            left, left_signed = self._signed_expression(value.left, path)
            right, right_signed = self._signed_expression(value.right, path)
            if left_signed and right_signed and self.arithmetic.integer_interval(value, signed=True) is None:
                operation = _SIGNED_OPERATIONS[type(value.operator)]
                checked = _boundary_call(operation, path, left, right)
                return checked.with_changes(lpar=value.lpar, rpar=value.rpar), True
            return value.with_changes(left=left, right=right), left_signed and right_signed
        if isinstance(value, cst.IfExp):
            body, body_signed = self._signed_expression(value.body, path)
            orelse, else_signed = self._signed_expression(value.orelse, path)
            return value.with_changes(body=body, orelse=orelse), body_signed and else_signed
        if isinstance(value, cst.Integer):
            return value, value.evaluated_value <= (1 << 63) - 1
        if isinstance(value, cst.Name):
            return value, value.value in self.signed_names
        if isinstance(value, cst.Call):
            target = value.func
            if isinstance(target, cst.Name):
                return value, target.value in self.signed_functions
            if isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name):
                return value, (target.value.value == MX_SELF and target.attr.value in self.signed_methods
                               or target.value.value == MX_SYS_MOD and target.attr.value in _SIGNED_HELPERS)
        if isinstance(value, cst.Subscript) and any(not isinstance(s.slice, cst.Index) for s in value.slice):
            return value, False
        base = value.value if isinstance(value, cst.Subscript) else value
        if isinstance(base, cst.Attribute) and isinstance(base.value, cst.Name) and base.value.value == MX_SELF:
            dimensions = len(value.slice) if isinstance(value, cst.Subscript) else 0
            return value, self.signed_fields.get(base.attr.value) == dimensions
        return value, False

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
        value, signed = self._lower(value, self.path)
        return value if signed else _checked_integer(value, self.path)

    def _lower(self, value, path):
        value, signed = self._signed_expression(value, path)
        self.check_overflow |= any(
            isinstance(part, cst.BinaryOperation)
            and isinstance(part.operator, (cst.Add, cst.Subtract, cst.Multiply))
            and not _boxed_arithmetic(part)
            and self.arithmetic.integer_interval(part, signed=True) is None
            for part in walk(value)
        )
        return value, signed


def _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds, signatures, native_helpers=()):
    signed_return = node.returns is not None and node.returns.annotation.deep_equals(_INTEGER_ANNOTATION)
    if not signed_return and not any(isinstance(part, cst.Call) and dotted_name(part.func) == _INTEGER_CONVERSION
                                     for part in lexical_nodes(node.body.body)):
        return node
    params = {name: kind_of_type_expr(cell.get_argtype_expr(name, c_style=True, formula=True))
              for name in cell.params}
    kinds = OperandKind(resolver, cells_kinds, ref_kinds, cls.fqname, params)
    arithmetic = CheckedArithmetic(cls, cell)
    arithmetic.parameters = {
        param.name.value for param in node.params.params
        if param.annotation is not None and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)
    }
    facts = CstFacts(node.body.body)
    bindings = facts.bindings(lexical_nodes(node.body.body))
    fields, functions, methods = signatures
    signed_names = arithmetic.parameters | {
        statement.target.value for statement in lexical_nodes(node.body.body)
        if isinstance(statement, cst.AnnAssign) and isinstance(statement.target, cst.Name)
        and statement.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)
    }
    receiver_fields = fields.get(cls.name, {}) if MX_SELF not in bindings else {}
    receiver_methods = methods.get(cls.name, set()) if MX_SELF not in bindings else set()
    guard = _GuardIntegers(cell.model_path, kinds, native_helpers, arithmetic, receiver_fields,
                           functions.difference(bindings), receiver_methods, signed_names, signed_return)
    body = node.body.visit(guard)
    if not guard.check_overflow or not isinstance(body, cst.IndentedBlock):
        return node.with_changes(body=body)
    error = Names(node).reserve('_mx_integer_overflow')
    message = f'{cell.model_path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec.'
    handler = cst.parse_statement(
        # Keep Python exceptions and the diagnostics from checked boundaries.
        f'try:\n    pass\nexcept OverflowError as {error}:\n'
        f'    if {error}.args != ("value too large",):\n        raise\n'
        f'    raise OverflowError({message!r}) from {error}\n'
    )
    doc_end = 1 if node.get_docstring() is not None else 0
    handler = handler.with_changes(body=cst.IndentedBlock(body=body.body[doc_end:]))
    decorator = cst.Decorator(cst.parse_expression(f'{CY_MOD}.overflowcheck(True)'))
    return node.with_changes(decorators=(*node.decorators, decorator),
                             body=body.with_changes(body=(*body.body[:doc_end], handler)))


def _signed_signatures(state):
    def signed_result(node):
        return (node.returns is not None and node.returns.annotation.deep_equals(_INTEGER_ANNOTATION)
                and any(dotted_name(item.decorator) in {f'{CY_MOD}.cfunc', f'{CY_MOD}.ccall'}
                        for item in node.decorators))

    functions = {name for name, node in state.functions.items() if signed_result(node)}
    methods = {}
    for (owner, name), node in state.methods.items():
        if signed_result(node):
            methods.setdefault(owner, set()).add(name)
    fields = {}
    for owner in state.syntax.body:
        if not isinstance(owner, cst.ClassDef) or not isinstance(owner.body, cst.IndentedBlock):
            continue
        declared = fields[owner.name.value] = {}
        for line in owner.body.body:
            if not isinstance(line, cst.SimpleStatementLine):
                continue
            for declaration in line.body:
                if not isinstance(declaration, cst.AnnAssign) or not isinstance(declaration.target, cst.Name):
                    continue
                annotation = declaration.annotation.annotation
                dimensions = 0
                if isinstance(annotation, cst.Subscript) and all(isinstance(s.slice, cst.Index)
                                                                  for s in annotation.slice):
                    dimensions = len(annotation.slice)
                    annotation = annotation.value
                if annotation.deep_equals(_INTEGER_ANNOTATION):
                    declared[declaration.target.value] = dimensions
    return fields, functions, methods


def guard_integer_boundaries(state, module_info, resolver, cells_kinds, ref_kinds, cells):
    """Check formula results, arguments and ItemSpace keys after entry lowering."""
    entries = {name: params for name, cell in cells.items() if (params := integer_parameters(cell))}
    classes = []
    declarations = []
    call_gate = state.names.reserve('_mx_checked_cell_callable')
    changed = False
    signatures = _signed_signatures(state)
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
                node = _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds, signatures)
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
                methods.append(_guard_result(method, formula, cls, resolver, cells_kinds, ref_kinds, signatures,
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
                method = _guard_result(method, cell, cls, resolver, cells_kinds, ref_kinds, signatures)
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
