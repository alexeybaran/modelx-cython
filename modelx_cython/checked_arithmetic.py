"""Keep small, proven integer expressions in the existing Cython formula."""

import ast
import math

import libcst as cst

_BOUND = 1 << 25


class CheckedArithmetic:
    def __init__(self, cls, cell):
        self.cls = cls
        self.cell = cell
        self.code = cst.Module([]).code_for_node

    def native_call(self, expr):
        if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
                and isinstance(expr.func.value, ast.Name) and expr.func.value.id == 'self'
                and not expr.keywords
                and all(isinstance(arg, ast.Name) and arg.id in self.parameters
                        or isinstance(arg, ast.Constant) and type(arg.value) is int
                        and -_BOUND <= arg.value <= _BOUND for arg in expr.args)):
            return False
        cell = self.cls.cells.get(expr.func.attr)
        return (cell is not None and not cell.is_special() and not cell.called_with_kwargs
                and not cell.body_has_closure and cell.get_rettype_expr(c_style=True) == 'long long')

    def capture(self, expr, local_names=()):
        if isinstance(expr, ast.Constant) and type(expr.value) is int and -_BOUND <= expr.value <= _BOUND:
            return str(expr.value), str(expr.value)
        if self.native_call(expr) or isinstance(expr, ast.Name) and expr.id in (*self.parameters, *local_names):
            name = f'{self.prefix}_{self.next_input}'
            self.next_input += 1
            self.inputs.append((name, ast.unparse(expr)))
            return name, f'_mx_cy.cast(object, {name})'
        if not isinstance(expr, ast.BinOp):
            return None
        operators = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*'}
        operator = operators.get(type(expr.op))
        if operator is None:
            return None
        left = self.capture(expr.left, local_names)
        right = self.capture(expr.right, local_names)
        if left is None or right is None:
            return None
        return tuple(f'({a} {operator} {b})' for a, b in zip(left, right))

    def statements(self, expressions, combine):
        bounds = ' and '.join(f'-{_BOUND} <= {name} <= {_BOUND}' for name, _ in self.inputs)
        assert bounds
        native, boxed = zip(*expressions)
        return (
            ''.join(f'{name}: _mx_cy.longlong = {source}\n' for name, source in self.inputs)
            + f'if {bounds}:\n'
            + f'    return {combine(*native)}\n'
            + f'return {combine(*boxed)}\n'
        )

    def double_tail(self, body, tree):
        if len(tree.body) < 3 or not isinstance(tree.body[-1], ast.Return):
            return body
        left, right, result = tree.body[-3:]
        if not all(isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name) for node in (left, right)):
            return body
        expr = result.value
        if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add)
                and isinstance(expr.left, ast.BinOp) and isinstance(expr.left.op, ast.Div)
                and isinstance(expr.left.left, ast.Name) and expr.left.left.id == left.targets[0].id
                and isinstance(expr.right, ast.Name) and expr.right.id == right.targets[0].id
                and isinstance(expr.left.right, ast.Constant)
                and type(expr.left.right.value) is float and math.isfinite(expr.left.right.value)
                and expr.left.right.value > 0):
            return body
        expressions = (left.value, right.value)
        for value in expressions:
            # At most one product and four operations keeps the native result
            # exactly representable in both signed 64 bits and a C double.
            nodes = [node for node in ast.walk(value) if isinstance(node, ast.BinOp)]
            if not nodes or len(nodes) > 4 or sum(isinstance(node.op, ast.Mult) for node in nodes) > 1:
                return body
        captured = [self.capture(expr) for expr in expressions]
        if None in captured or not self.inputs:
            return body
        divisor = ast.unparse(expr.left.right)
        native = self.statements(captured, lambda a, b: f'{a} / {divisor} + {b}')
        return (*body[:-3], *cst.parse_module(native).body)

    def integer_product(self, body, tree):
        assigned = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                assigned[name] = self.native_call(node.value)
        native_locals = {name for name, typed in assigned.items() if typed
                         and sum(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)
                                 for n in ast.walk(tree)) == 1}

        class Product(cst.CSTTransformer):
            def leave_SimpleStatementLine(inner, before, after):
                if len(before.body) != 1 or not isinstance(before.body[0], cst.Return):
                    return after
                result = before.body[0].value
                if result is None:
                    return after
                expr = ast.parse(self.code(result), mode='eval').body
                if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult)
                        and isinstance(expr.right, ast.BinOp) and isinstance(expr.right.op, ast.Sub)
                        and isinstance(expr.right.right, ast.Constant)
                        and type(expr.right.right.value) is int and expr.right.right.value == 1
                        and self.native_call(expr.right.left)
                        and isinstance(expr.left, ast.Name) and expr.left.id in native_locals):
                    return after
                self.inputs = []
                captured = self.capture(expr, native_locals)
                assert captured
                native = self.statements((captured,), lambda value: value)
                return cst.FlattenSentinel(cst.parse_module(native).body)

        visitor = Product()
        result = cst.Module(body).visit(visitor)
        return result.body

    def rewrite(self, function):
        kind = self.cell.get_rettype_expr(c_style=True)
        if (kind not in ('double', 'long long') or self.cell.is_locked or self.cell.body_has_closure
                or self.cell.cls in self.cls.visitor.dynamic_ref_classes
                or not isinstance(function.body, cst.IndentedBlock)):
            return function
        tree, = ast.parse(self.code(function)).body
        assigned = {node.id for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        self.parameters = {name for name in self.cell.params if name not in assigned
                           and self.cell.get_argtype_expr(name, c_style=True) == 'long long'}
        self.prefix = '_mx_checked'
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        while any(name.startswith(self.prefix) for name in names):
            self.prefix += '_'
        self.inputs = []
        self.next_input = 0
        body = function.body.body
        updated = self.double_tail(body, tree) if kind == 'double' else self.integer_product(body, tree)
        return function.with_changes(body=function.body.with_changes(body=updated))
