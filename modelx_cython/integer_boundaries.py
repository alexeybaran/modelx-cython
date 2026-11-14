"""Check integer Cell results and ItemSpace keys at their native boundaries."""

import libcst as cst

from .checked_arithmetic import (
    CheckedArithmetic, divisibility_enabled, divisibility_operands, guarded_integer_intervals,
    signed_local_assignments,
)
from .consts import CY_MOD, FORMULA_PREF, MX_SELF, MX_SYS_MOD
from .powers import KIND_FLOAT, OperandKind, kind_of_type_expr
from .scopes import CstFacts, Names, OPAQUE_SCOPES, has_scope_effects, lexical_nodes
from .syntax import dotted_name, local_assignment, statements, walk
from .usage import RESOLVED, _extract_self_chain


_INTEGER_ANNOTATION = cst.Attribute(cst.Name(CY_MOD), cst.Name('longlong'))
_SIGNED_OPERATIONS = {
    cst.Add: '_mx_signed_add',
    cst.Subtract: '_mx_signed_sub',
    cst.Multiply: '_mx_signed_mul',
    cst.FloorDivide: '_mx_signed_floor',
    cst.Modulo: '_mx_signed_remainder',
    cst.LeftShift: '_mx_signed_left_shift',
    cst.RightShift: '_mx_signed_right_shift',
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


def _overflow_handler(body, path, error, leading_lines=()):
    message = f'{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec.'
    handler = cst.parse_statement(
        # Keep Python exceptions and the diagnostics from checked boundaries.
        f'try:\n    pass\nexcept OverflowError as {error}:\n'
        f'    if {error}.args != ("value too large",):\n        raise\n'
        f'    raise OverflowError({message!r}) from {error}\n'
    )
    return handler.with_changes(body=cst.IndentedBlock(body=body), leading_lines=leading_lines)


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

    def __init__(self, receiver, integer_params, counters, owner, resolver, entries, cells, private_entries):
        self.receiver = receiver
        self.integer_params = integer_params
        self.counters = counters
        self.loop_targets = []
        self.owner = owner
        self.resolver = resolver
        self.entries = entries
        self.cells = cells
        self.private_entries = private_entries

    def visit_For_body(self, node):
        self.loop_targets.append(node.target.value if isinstance(node.target, cst.Name) else None)

    def leave_For_body(self, node):
        self.loop_targets.pop()

    def _native_argument(self, value):
        if not isinstance(value, cst.Name):
            return False
        scope = self.get_metadata(cst.metadata.ScopeProvider, value, None)
        # Range analysis already proved the only store and that its reads
        # stay in the loop body. Check the receiver scope after entry lowering.
        if value.value in self.counters and value.value in self.loop_targets:
            receiver_scope = self.get_metadata(cst.metadata.ScopeProvider, self.receiver, None)
            if scope is receiver_scope:
                return True
        if value.value not in self.integer_params:
            return False
        bindings = tuple(scope[value.value]) if scope is not None else ()
        return len(bindings) == 1 and bindings[0].node is self.integer_params[value.value]

    def leave_Call(self, before, after):
        if isinstance(before.func, cst.Name) and (entry := self.private_entries.get(before.func.value)):
            path, params = entry
            return self._arguments(before, after, path, params, start=1)
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
        return self._arguments(before, after, cell.model_path, params)

    def _arguments(self, before, after, path, params, start=0):
        guarded = {index + start for index, _ in params}
        arguments = tuple(arg.with_changes(value=_checked_integer(arg.value, path))
                          if index in guarded and arg.keyword is None and not self._native_argument(old.value)
                          else arg for index, (old, arg) in enumerate(zip(before.args, after.args)))
        return after.with_changes(args=arguments)


def _guard_calls(node, cls, cell, resolver, entries, cells, private_entries):
    params = (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)
    if not params or params[0].name.value != MX_SELF:
        return node
    integer_params = {param.name.value: param for param in params if param.annotation is not None
                      and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    counters = getattr(cell, 'signed_range_counters', ())
    wrapper = cst.MetadataWrapper(cst.Module(body=(node,)), unsafe_skip_copy=True)
    return wrapper.visit(_GuardCalls(params[0], integer_params, counters, cls.fqname, resolver, entries, cells,
                                    private_entries)).body[0]


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
                 signed_functions, signed_methods, signed_names, signed_return, error):
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
        self.error = error
        self.outer_intervals = []
        self.in_statement = self.statement_overflow = self.function_handler = False
        self.arithmetic.signed_atom = self._signed_atom

    def on_visit(self, node):
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)

    def visit_If_body(self, node):
        self.outer_intervals.append(self.arithmetic.parameter_intervals)
        body = statements(node.body)
        if (len(body) != 1 or not isinstance(body[0], cst.Return) or has_scope_effects(body)
                or any(isinstance(part, cst.Call) for part in walk(body[0]))):
            return
        bounded = guarded_integer_intervals(node.test)
        if bounded and bounded.keys() <= self.signed_names:
            self.arithmetic.parameter_intervals = {**self.arithmetic.parameter_intervals, **bounded}

    def leave_If_body(self, node):
        self.arithmetic.parameter_intervals = self.outer_intervals.pop()

    def visit_SimpleStatementLine(self, node):
        self.in_statement = True
        self.statement_overflow = False

    def leave_SimpleStatementLine(self, before, after):
        self.in_statement = False
        if not self.statement_overflow:
            return after
        return _overflow_handler((after.with_changes(leading_lines=()),), self.path, self.error,
                                 after.leading_lines)

    def leave_Return(self, before, after):
        if not self.signed_return or after.value is None:
            return after
        return after.with_changes(value=self._checked(after.value))

    def leave_Assign(self, before, after):
        if not self.signed_return:
            return after
        value, _ = self._lower(after.value, self.path)
        return after.with_changes(value=value)

    def leave_AnnAssign(self, before, after):
        if not self.signed_return or after.value is None:
            return after
        value, _ = self._lower(after.value, self.path)
        return after.with_changes(value=value)

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
        if isinstance(value, cst.UnaryOperation) and isinstance(value.operator, (cst.Plus, cst.Minus)):
            if self.arithmetic.integer_interval(value, signed=True) is not None:
                return value, True
            operand, signed = self._signed_expression(value.expression, path)
            if signed and isinstance(value.operator, cst.Minus):
                checked = _boundary_call('_mx_signed_sub', path, cst.Integer('0'), operand)
                return checked.with_changes(lpar=value.lpar, rpar=value.rpar), True
            return value.with_changes(expression=operand), signed
        if isinstance(value, cst.Integer):
            return value, value.evaluated_value <= (1 << 63) - 1
        return value, self._signed_atom(value)

    def _signed_atom(self, value):
        if isinstance(value, cst.Name):
            return value.value in self.signed_names
        if isinstance(value, cst.Call):
            target = value.func
            if isinstance(target, cst.Name):
                return target.value in self.signed_functions
            if isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name):
                return (target.value.value == MX_SELF and target.attr.value in self.signed_methods
                        or target.value.value == MX_SYS_MOD and target.attr.value in _SIGNED_HELPERS)
        if isinstance(value, cst.Subscript) and any(not isinstance(s.slice, cst.Index) for s in value.slice):
            return False
        base = value.value if isinstance(value, cst.Subscript) else value
        if isinstance(base, cst.Attribute) and isinstance(base.value, cst.Name) and base.value.value == MX_SELF:
            dimensions = len(value.slice) if isinstance(value, cst.Subscript) else 0
            return self.signed_fields.get(base.attr.value) == dimensions
        return False

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
        needs_overflow_check = any(
            isinstance(part, cst.BinaryOperation)
            and isinstance(part.operator, (cst.Add, cst.Subtract, cst.Multiply))
            and not _boxed_arithmetic(part)
            and self.arithmetic.integer_interval(part, signed=True) is None
            for part in walk(value)
        )
        if needs_overflow_check:
            self.check_overflow = True
            if self.in_statement:
                self.statement_overflow = True
            else:
                self.function_handler = True
        return value, signed


def _signed_operand_names(value):
    """Find the name uses that signed expression lowering treats as C operands."""
    if isinstance(value, cst.Name):
        yield value
    elif isinstance(value, cst.BinaryOperation) and type(value.operator) in _SIGNED_OPERATIONS:
        yield from _signed_operand_names(value.left)
        yield from _signed_operand_names(value.right)
    elif isinstance(value, cst.UnaryOperation) and isinstance(value.operator, (cst.Plus, cst.Minus)):
        yield from _signed_operand_names(value.expression)
    elif isinstance(value, cst.IfExp):
        yield from _signed_operand_names(value.body)
        yield from _signed_operand_names(value.orelse)


def _signed_local_assignments(body, name, operands, guard, facts):
    """Prove a native local across simple assignments and conditional joins."""
    def native_value(value):
        expression, signed = guard._signed_expression(value, guard.path)
        return expression if signed else None

    assignments, reads = signed_local_assignments(body, name, facts, native_value)
    if not assignments:
        return {}
    # A Python-context use could depend on the local's object identity or
    # Python operator semantics. Keep the whole local boxed in that case.
    safe_reads = set()
    for value in (*operands, *(assignment.value for assignment in assignments)):
        local_reads = reads.intersection(_signed_operand_names(value))
        if local_reads and guard._signed_expression(value, guard.path)[1]:
            safe_reads.update(local_reads)
    return assignments if reads <= safe_reads else {}


class _SignedLocalValues(cst.CSTTransformer):
    def __init__(self, assignments):
        self.assignments = assignments

    def leave_Assign(self, before, after):
        value = self.assignments.get(before)
        return after if value is None else after.with_changes(value=value)


def _signed_divisibility(body, facts, guard, parameters):
    """Admit locals whose only uses are signed zero-remainder comparisons."""
    if (has_scope_effects(body.body)
            or facts.reads & {'eval', 'exec', 'locals', 'vars', 'globals', '_getframe', 'currentframe'}):
        return body, ()
    probes = {node: operands for node in lexical_nodes(body.body)
              if (operands := divisibility_operands(node)) is not None
              and all(isinstance(value, cst.Name) for value in operands[:2])}
    uses = {value for operands in probes.values() for value in operands[:2]}
    native = guard.signed_names - {name for name in parameters if facts.stores[name]}
    declarations = []
    for name in sorted({value.value for value in uses} - set(parameters) - native):
        if not facts.stores[name]:
            continue
        assignments, reads = signed_local_assignments(
            body, name, facts,
            lambda value: value if guard.arithmetic.integer_interval(value, signed=True) is not None else None,
        )
        if not assignments or not reads <= uses:
            continue
        # Each comparison must be able to preserve the local's signed value.
        if not all(operand.value in native or operand.value == name
                   for operands in probes.values() if any(value in reads for value in operands[:2])
                   for operand in operands[:2]):
            continue
        native.add(name)
        declarations.append(cst.SimpleStatementLine((
            cst.AnnAssign(cst.Name(name), cst.Annotation(_INTEGER_ANNOTATION)),)))

    replacements = {}
    for probe, (dividend, divisor, operation, negate) in probes.items():
        if not {dividend.value, divisor.value} <= native:
            continue
        call = cst.Call(cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(operation)),
                        (cst.Arg(dividend), cst.Arg(divisor)))
        if negate:
            call = cst.UnaryOperation(cst.Not(), call)
        replacements[probe] = call.with_changes(lpar=probe.lpar, rpar=probe.rpar)

    class LowerDivisibility(cst.CSTTransformer):
        def leave_Comparison(self, before, after):
            return replacements.get(before, after)

    return body.visit(LowerDivisibility()) if replacements else body, tuple(declarations)


def _type_signed_locals(body, guard, facts, parameters):
    """Keep provably signed locals native and preserve existing prefix admission."""
    if (not isinstance(body, cst.IndentedBlock) or has_scope_effects(body.body)
            or facts.reads & {'eval', 'exec', 'locals', 'vars', 'globals', '_getframe', 'currentframe'}):
        return body, ()
    operands = []
    identity_names = set()
    for part in lexical_nodes(body.body):
        if guard.signed_return and isinstance(part, cst.Return) and part.value is not None:
            operands.append(part.value)
        if (isinstance(part, cst.Call) and dotted_name(part.func) == _INTEGER_CONVERSION
                and len(part.args) == 2):
            operands.append(part.args[0].value)
    for part in walk(body):
        if isinstance(part, cst.Comparison) and any(
                isinstance(item.operator, (cst.Is, cst.IsNot)) for item in part.comparisons):
            identity_names.update(item.value for item in walk(part) if isinstance(item, cst.Name))
    needed = {part.value for value in operands for part in walk(value) if isinstance(part, cst.Name)}
    needed.difference_update(identity_names)
    if not needed:
        return body, ()
    statements = list(body.body)
    parameters = set(parameters)
    seen = parameters.copy()
    for index, line in enumerate(statements):
        if not isinstance(line, cst.SimpleStatementLine) or len(line.body) != 1:
            break
        assignment = line.body[0]
        if ((target := local_assignment(assignment)) is not None
                and target.value in needed and target.value not in seen
                and facts.stores[target.value] == 1):
            value, signed = guard._signed_expression(assignment.value, guard.path)
            if signed:
                statements[index] = line.with_changes(body=(
                    cst.AnnAssign(target, cst.Annotation(_INTEGER_ANNOTATION), value),))
                guard.signed_names.add(target.value)
        seen.update(part.value for part in walk(line) if isinstance(part, cst.Name))
    declarations, assignments = [], {}
    for name in sorted(needed - parameters - guard.signed_names):
        if not facts.stores[name]:
            continue
        guard.signed_names.add(name)
        values = _signed_local_assignments(body, name, operands, guard, facts)
        if not values:
            guard.signed_names.remove(name)
            guard.python_locals.add(name)       # D72: examined and left to Python
            continue
        declarations.append(cst.SimpleStatementLine((cst.AnnAssign(
            cst.Name(name), cst.Annotation(_INTEGER_ANNOTATION)),)))
        assignments.update(values)
    body = body.with_changes(body=statements)
    if assignments:
        body = body.visit(_SignedLocalValues(assignments))
    return body, tuple(declarations)


def _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds, signatures, native_helpers=(),
                  argument_axes=()):
    signed_return = node.returns is not None and node.returns.annotation.deep_equals(_INTEGER_ANNOTATION)
    divisibility = (node.returns is not None and dotted_name(node.returns.annotation) == f'{CY_MOD}.bint'
                    and not cell.is_locked and not cell.body_has_closure and divisibility_enabled(cell)
                    and any(divisibility_operands(part) is not None for part in lexical_nodes(node.body.body)))
    if not signed_return and not divisibility and not any(
            isinstance(part, cst.Call) and dotted_name(part.func) == _INTEGER_CONVERSION
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
    arithmetic.parameter_intervals = {axis.key: (axis.lower, axis.stop - 1) for axis in argument_axes
                                      if axis.key in arithmetic.parameters and axis.key not in bindings}
    fields, functions, methods = signatures
    signed_names = arithmetic.parameters | {
        statement.target.value for statement in lexical_nodes(node.body.body)
        if isinstance(statement, cst.AnnAssign) and isinstance(statement.target, cst.Name)
        and statement.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)
    }
    receiver_fields = fields.get(cls.name, {}) if MX_SELF not in bindings else {}
    receiver_methods = methods.get(cls.name, set()) if MX_SELF not in bindings else set()
    error = Names(node).reserve('_mx_integer_overflow')
    guard = _GuardIntegers(cell.model_path, kinds, native_helpers, arithmetic, receiver_fields,
                           functions.difference(bindings), receiver_methods, signed_names, signed_return,
                           error)
    parameters = tuple(parameter.name.value for parameter in
                       (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params))
    guard.python_locals = set()
    if divisibility:
        body, declarations = _signed_divisibility(node.body, facts, guard, parameters)
    else:
        body, declarations = _type_signed_locals(node.body, guard, facts, parameters)
    cell.python_int_locals = set(getattr(cell, 'python_int_locals', ())) | guard.python_locals
    body = body.visit(guard)
    if declarations:
        doc_end = 1 if node.get_docstring() is not None else 0
        body = body.with_changes(body=(*body.body[:doc_end], *declarations, *body.body[doc_end:]))
    if not guard.check_overflow or not isinstance(body, cst.IndentedBlock):
        return node.with_changes(body=body)
    decorator = cst.Decorator(cst.parse_expression(f'{CY_MOD}.overflowcheck(True)'))
    if guard.function_handler:
        prefix = len(declarations) + (1 if node.get_docstring() is not None else 0)
        handler = _overflow_handler(body.body[prefix:], cell.model_path, error)
        body = body.with_changes(body=(*body.body[:prefix], handler))
    return node.with_changes(decorators=(*node.decorators, decorator), body=body)


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


def _python_locals(node, cell):
    """D72: declare ``object`` the locals the integer passes leave to Python on purpose.

    With ``infer_types`` (W1.5 0005) Cython infers a C type for every untyped local, so a local
    the compiler deliberately kept as a Python int would wrap, raise OverflowError where Python
    is exact, or lose its identity (``x is saved[0]`` on a re-boxed int). Declared ``object``:
    every local compared with ``is``/``is not``, every ``range`` target the range pass examined
    and did not type, and every local the signed-local pass examined and did not prove. Locals
    with an annotation of their own (the passes' ``long long`` declarations) are left alone.
    """
    body = node.body
    # the overflow handler the guard itself wraps around the body is an except clause; any
    # other binding construct leaves the formula as it is
    effects = (cst.Global, cst.Nonlocal, cst.Import, cst.ImportFrom, cst.Del, cst.NamedExpr,
               cst.FunctionDef, cst.ClassDef, cst.Lambda)
    if not isinstance(body, cst.IndentedBlock) or any(
            isinstance(part, effects) or isinstance(part, getattr(cst, 'Match', ()))
            for part in lexical_nodes(body.body)):
        return node
    facts = CstFacts(body.body)
    parameters = {parameter.name.value for parameter in
                  (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)}
    annotated = {statement.target.value for statement in lexical_nodes(body.body)
                 if isinstance(statement, cst.AnnAssign) and isinstance(statement.target, cst.Name)}
    names = set(getattr(cell, 'python_int_locals', ())) | set(getattr(cell, 'python_range_locals', ()))
    for part in lexical_nodes(body.body):
        if isinstance(part, cst.Comparison) and any(
                isinstance(item.operator, (cst.Is, cst.IsNot)) for item in part.comparisons):
            names.update(item.value for item in walk(part) if isinstance(item, cst.Name))
    names = sorted(name for name in names
                   if facts.stores[name] and name not in parameters and name not in annotated)
    if not names:
        return node
    declarations = tuple(cst.SimpleStatementLine((cst.AnnAssign(
        cst.Name(name), cst.Annotation(cst.Name('object'))),)) for name in names)
    doc_end = 1 if node.get_docstring() is not None else 0
    cell.python_object_locals = tuple(names)
    return node.with_changes(body=body.with_changes(body=(*body.body[:doc_end], *declarations,
                                                          *body.body[doc_end:])))


def guard_integer_boundaries(state, module_info, resolver, cells_kinds, ref_kinds, cells, infer_types=False):
    """Check formula results, arguments and ItemSpace keys after entry lowering."""
    entries = {name: params for name, cell in cells.items() if (params := integer_parameters(cell))}
    private_entries = {
        name: (cell.model_path, params)
        for cls in module_info.classes.values() for cell in cls.cells.values()
        if (name := getattr(cell, 'inline_entry', None)) and (params := entries.get(cell.fqname))
    }
    private_entries.update({name: private_entries[checked] for name, checked in state.inbounds_entries.items()
                            if checked in private_entries})
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
                node = _guard_calls(node, cls, cell, resolver, entries, cells, private_entries)
                # Physical cache misses and compact spill entries both check
                # the logical domain before reaching the private formula.
                axes = (cell.dense_logical_axes if cell.has_args() and cell.has_typeinfo()
                        and cell.is_arrayable() else ())
                node = _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds, signatures,
                                     argument_axes=axes)
                if infer_types:
                    node = _python_locals(node, cell)
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
                method = _guard_calls(method, cls, formula, resolver, entries, cells, private_entries)
                method = _guard_result(method, formula, cls, resolver, cells_kinds, ref_kinds, signatures,
                                       native_helpers.get((cls.name, name), ()))
                if infer_types:
                    method = _python_locals(method, formula)
                methods.append(method)
                changed = True
                continue
            cell = cls.cells.get(name)
            if name == '__call__' and cell is not None and cell.has_typeinfo():
                methods.append(_guard_space_parameters(method, cell))
                changed = True
                continue
            if cell is not None and not cell.has_formula_def:
                method = _guard_calls(method, cls, cell, resolver, entries, cells, private_entries)
                method = _guard_result(method, cell, cls, resolver, cells_kinds, ref_kinds, signatures)
                if infer_types:
                    method = _python_locals(method, cell)
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
