"""Use a small integer vector directly inside an admitted scalar formula."""

import textwrap

import libcst as cst
from libcst.metadata.scope_provider import BuiltinAssignment

from .consts import CY_MOD
from .scopes import CstFacts, Names, has_scope_effects
from .syntax import arguments, dotted_name, literal_value, local_assignment, single_statement, statements
from .table_cells import module_renames, vector_cells
from .vectors import ALIAS, TYPES


_LIMIT = 1 << 20
_OPERATIONS = {
    cst.Add, cst.Subtract, cst.Divide, cst.Plus, cst.Minus, cst.Not, cst.And, cst.Or,
    cst.Equal, cst.NotEqual, cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual,
}
def _builtin(scope, name):
    return scope[name] and all(isinstance(binding, BuiltinAssignment) for binding in scope[name])


def _tail_size(lines):
    """Count semantic nodes and their load/store contexts in the existing size limit."""
    def cost(node):
        if isinstance(node, cst.Attribute):
            return 2 + cost(node.value)
        if isinstance(node, cst.Name):
            return 1 if node.value in ('True', 'False', 'None') else 2
        count = isinstance(node, (cst.BaseExpression, cst.BaseSmallStatement, cst.BaseCompoundStatement))
        return count + (type(node) in _OPERATIONS) + isinstance(node, cst.Subscript) + sum(map(cost, node.children))
    return sum(map(cost, lines))


class _UnsupportedTail(Exception):
    pass


class _ScalarTail:
    def __init__(self, vector, held, cls, cell, scope, names):
        self.vector, self.held = vector, held
        self.cls, self.cell, self.scope, self.names = cls, cell, scope, names
        self.reads = set()
        self.locals = {}

    def _is_vector(self, node):
        return isinstance(node, cst.Name) and node.value == self.vector

    def _integer_index(self, node):
        if isinstance(node, cst.Name):
            return node.value in self.cell.params
        if isinstance(node, (cst.Integer, cst.Attribute)):
            return True
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, (cst.Plus, cst.Minus)):
            return self._integer_index(node.expression)
        return (isinstance(node, cst.BinaryOperation) and isinstance(node.operator, (cst.Add, cst.Subtract))
                and isinstance(node.right, cst.Integer) and self._integer_index(node.left))

    def expression(self, node):
        if isinstance(node, (cst.Integer, cst.Float)) and -_LIMIT <= literal_value(node) <= _LIMIT:
            return node
        if isinstance(node, cst.Name):
            name = node.value
            if name in self.locals:
                return node.with_changes(value=self.locals[name])
            if name in self.cell.params and self.cell.get_argtype_expr(name, c_style=True, formula=True) == 'long long':
                self.reads.add(name)
                return node
        if isinstance(node, cst.Attribute):
            ref = self.cls.refs.get(node.attr.value)
            if dotted_name(node.value) == 'self' and ref and ref.get_type_expr(c_style=True) == 'long long':
                self.reads.add(f'self.{node.attr.value}')
                return node
        if isinstance(node, cst.Subscript) and self._is_vector(node.value) and len(node.slice) == 1:
            part, = node.slice
            if (part.comma is cst.MaybeSentinel.DEFAULT and isinstance(part.slice, cst.Index)
                    and part.slice.star is None and self._integer_index(part.slice.value)):
                index = self.expression(part.slice.value)
                return cst.Call(cst.Attribute(cst.Name(ALIAS), cst.Name('ivec_get')),
                                (cst.Arg(cst.Name(self.held)), cst.Arg(index)), lpar=node.lpar, rpar=node.rpar)
        if isinstance(node, cst.Call) and isinstance(node.func, cst.Name) and _builtin(self.scope, node.func.value):
            args = arguments(node)
            if args is not None and node.func.value == 'len' and len(args) == 1 and self._is_vector(args[0]):
                return cst.Attribute(cst.Name(self.held), cst.Name('n'), lpar=node.lpar, rpar=node.rpar)
            if args is not None and node.func.value in ('min', 'max') and len(args) == 2:
                return node.with_changes(args=tuple(arg.with_changes(value=self.expression(arg.value)) for arg in node.args))
        if isinstance(node, (cst.BinaryOperation, cst.BooleanOperation)) and type(node.operator) in _OPERATIONS:
            return node.with_changes(left=self.expression(node.left), right=self.expression(node.right))
        if isinstance(node, cst.UnaryOperation) and type(node.operator) in _OPERATIONS:
            return node.with_changes(expression=self.expression(node.expression))
        if isinstance(node, cst.Comparison) and all(type(part.operator) in _OPERATIONS for part in node.comparisons):
            return node.with_changes(left=self.expression(node.left), comparisons=tuple(
                part.with_changes(comparator=self.expression(part.comparator)) for part in node.comparisons))
        raise _UnsupportedTail

    def small_statement(self, stmt):
        if isinstance(stmt, cst.Return):
            return stmt.with_changes(value=self.expression(stmt.value))
        target = local_assignment(stmt)
        if target is not None and target.value not in (*self.cell.params, 'self', self.vector, *self.locals):
            value = self.expression(stmt.value)
            renamed = self.names.reserve('_mx_vec_scalar_' + target.value)
            self.locals[target.value] = renamed
            return stmt.with_changes(value=value, targets=(stmt.targets[0].with_changes(
                target=target.with_changes(value=renamed)),))
        raise _UnsupportedTail

    def suite(self, node):
        body = (self.line if isinstance(node, cst.IndentedBlock) else self.small_statement)
        return node.with_changes(body=tuple(body(stmt) for stmt in node.body))

    def line(self, node):
        if isinstance(node, cst.SimpleStatementLine):
            return node.with_changes(body=tuple(self.small_statement(stmt) for stmt in node.body))
        if isinstance(node, cst.If):
            test = self.expression(node.test)
            outer = self.locals.copy()
            body = self.suite(node.body)
            self.locals = outer.copy()
            otherwise = node.orelse
            if isinstance(otherwise, cst.If):
                otherwise = self.line(otherwise)
            elif otherwise is not None:
                otherwise = otherwise.with_changes(body=self.suite(otherwise.body))
            self.locals = outer
            return node.with_changes(test=test, body=body, orelse=otherwise)
        raise _UnsupportedTail


def lower(node, cls, cell, scope):
    if (not cell._spec.get('native_vector_ref_reads', False)
            or not isinstance(node.body, cst.IndentedBlock) or not _builtin(scope, 'type')
            or tuple(scope[CY_MOD]) or tuple(scope[ALIAS])
            or has_scope_effects(node.body.body) or cell.get_rettype_expr(c_style=True) != 'double'):
        return node
    lines = node.body.body
    names = Names(node)
    for i, line in enumerate(lines):
        statement = single_statement(line)
        local = local_assignment(statement)
        if local is None or local.value in cell.params or arguments(statement.value, count=0) is None:
            continue
        path = (dotted_name(statement.value.func) or '').split('.')
        if len(path) != 3 or path[0] != 'self':
            continue
        ref = cls.refs.get(path[1])
        target = cls.module.classes.get(getattr(ref, 'decl_type_expr', None))
        name = path[2]
        if (target is None or not ref.is_relative or target.name in names.names or cls.name in names.names
                or vector_cells(target).get(name) != 'I'
                or name not in module_renames(cls.module).get(target.name, {})):
            continue
        tail = lines[i + 1:]
        body = statements(tail)
        if not body or not isinstance(body[-1], cst.Return) or _tail_size(tail) > 128:
            continue
        # A native tail cannot delay release of an object held by a prefix local.
        existing = CstFacts(lines[:i]).stores
        rebound = CstFacts(tail).stores
        if set(existing) & (set(rebound) | {local.value}) or 'self' in existing:
            continue
        receiver, held = (names.reserve('_mx_vector_' + label) for label in ('owner', 'held'))
        rewrite = _ScalarTail(local.value, held, cls, cell, scope, names)
        try:
            native = tuple(rewrite.line(part) for part in tail)
        except _UnsupportedTail:
            continue
        checks = ''.join(f' and -{_LIMIT} <= {value} <= {_LIMIT}' for value in sorted(rewrite.reads))
        # Keep Python reference lookup for unexpected classes and retain the
        # already evaluated vector on an exact receiver outside the native domain.
        branch = cst.parse_module(f'''\
{receiver}: object = self.{path[1]}
{held}: {TYPES['I']} = None
if type(self) is {cls.name} and type({receiver}) is {target.name}:
    {held} = {CY_MOD}.cast({target.name}, {receiver}).{name}()
    if {ALIAS}.ivec_small_for_reads({held}){checks}:
{textwrap.indent(cst.Module(native).code.rstrip(), '        ')}
    {local.value} = {held}.tolist()
else:
    {local.value} = {receiver}.{name}()
''').body
        return node.with_changes(body=node.body.with_changes(body=(*lines[:i], *branch, *tail)))
    return node
