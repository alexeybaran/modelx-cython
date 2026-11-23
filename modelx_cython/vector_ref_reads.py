"""Use a small integer vector directly inside an admitted scalar formula."""

import ast
import textwrap

import libcst as cst
from libcst.metadata.scope_provider import BuiltinAssignment

from .consts import CY_MOD
from .scopes import Names, has_scope_effects
from .syntax import arguments, code, dotted_name, local_assignment, single_statement
from .table_cells import module_renames, vector_cells
from .vectors import ALIAS, TYPES


_LIMIT = 1 << 20
_OPERATORS = (ast.Add, ast.Sub, ast.Div, ast.UAdd, ast.USub, ast.Not,
              ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _builtin(scope, name):
    return scope[name] and all(isinstance(binding, BuiltinAssignment) for binding in scope[name])


class _ScalarTail(ast.NodeTransformer):
    def __init__(self, vector, held, cls, cell, scope, names):
        self.vector, self.held = vector, held
        self.cls, self.cell, self.scope, self.names = cls, cell, scope, names
        self.reads = set()
        self.locals = {}
        self.interpret = False

    def generic_visit(self, node):
        if not isinstance(node, (ast.Assign, ast.If, ast.Return, ast.Compare, ast.BoolOp,
                                 ast.BinOp, ast.UnaryOp, ast.Load, *_OPERATORS)):
            self.interpret = True
            return node
        return super().generic_visit(node)

    def visit_Constant(self, node):
        if type(node.value) not in (float, int) or not -_LIMIT <= node.value <= _LIMIT:
            self.interpret = True
        return node

    def visit_Name(self, node):
        if node.id in self.locals:
            return ast.copy_location(ast.Name(id=self.locals[node.id], ctx=node.ctx), node)
        if (node.id not in self.cell.params
                or self.cell.get_argtype_expr(node.id, c_style=True, formula=True) != 'long long'):
            self.interpret = True
        self.reads.add(node.id)
        return node

    def visit_Assign(self, node):
        if (len(node.targets) != 1 or not isinstance(target := node.targets[0], ast.Name)
                or target.id in (*self.cell.params, 'self', self.vector) or target.id in self.locals):
            self.interpret = True
            return node
        node.value = self.visit(node.value)
        local = self.names.reserve('_mx_vec_scalar_' + target.id)
        self.locals[target.id] = local
        target.id = local
        return node

    def visit_If(self, node):
        node.test = self.visit(node.test)
        outer = self.locals.copy()
        node.body = [self.visit(part) for part in node.body]
        self.locals = outer.copy()
        node.orelse = [self.visit(part) for part in node.orelse]
        self.locals = outer
        return node

    def visit_Attribute(self, node):
        ref = self.cls.refs.get(node.attr)
        if (not isinstance(node.value, ast.Name) or node.value.id != 'self'
                or ref is None or ref.get_type_expr(c_style=True) != 'long long'
                or not isinstance(node.ctx, ast.Load)):
            self.interpret = True
        self.reads.add(ast.unparse(node))
        return node

    def _is_vector(self, expr):
        return isinstance(expr, ast.Name) and expr.id == self.vector

    def _integer_index(self, expr):
        if isinstance(expr, ast.Constant):
            return type(expr.value) is int
        if isinstance(expr, ast.Name):
            return expr.id in self.cell.params
        if isinstance(expr, ast.Attribute):
            return True  # visit_Attribute checks the receiver and C type.
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            return self._integer_index(expr.operand)
        return (isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub))
                and isinstance(expr.right, ast.Constant) and type(expr.right.value) is int
                and self._integer_index(expr.left))

    def visit_Subscript(self, node):
        index = node.slice
        if (not self._is_vector(node.value) or not isinstance(node.ctx, ast.Load)
                or not self._integer_index(index)):
            self.interpret = True
            return node
        return ast.parse(f'{ALIAS}.ivec_get({self.held}, {ast.unparse(self.visit(index))})', mode='eval').body

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Name) and not node.keywords and _builtin(self.scope, node.func.id)):
            if node.func.id == 'len' and len(node.args) == 1 and self._is_vector(node.args[0]):
                return ast.parse(f'{self.held}.n', mode='eval').body
            if node.func.id in ('min', 'max') and len(node.args) == 2:
                node.args = [self.visit(arg) for arg in node.args]
                return node
        self.interpret = True
        return node


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
        original = ast.parse(code(cst.Module(lines[i + 1:]))).body
        if (not original or not isinstance(original[-1], ast.Return)
                or sum(len(list(ast.walk(part))) for part in original) > 128):
            continue
        # A native tail cannot delay release of an object held by a prefix local.
        prefix = ast.parse(code(cst.Module(lines[:i])))
        existing = {part.id for part in ast.walk(prefix)
                    if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)}
        rebound = {part.id for stmt in original for part in ast.walk(stmt)
                   if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)}
        if existing & (rebound | {local.value}) or 'self' in existing:
            continue
        receiver, held = (names.reserve('_mx_vector_' + label) for label in ('owner', 'held'))
        rewrite = _ScalarTail(local.value, held, cls, cell, scope, names)
        native = [rewrite.visit(part) for part in original]
        if rewrite.interpret:
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
{textwrap.indent(ast.unparse(ast.Module(body=native, type_ignores=[])), '        ')}
    {local.value} = {held}.tolist()
else:
    {local.value} = {receiver}.{name}()
''').body
        return node.with_changes(body=node.body.with_changes(body=(*lines[:i], *branch, *lines[i + 1:])))
    return node
