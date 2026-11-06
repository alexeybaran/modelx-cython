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

    def divisibility(self, body, tree):
        if any(isinstance(node, (ast.Import, ast.ImportFrom, ast.ExceptHandler,
                                 ast.Global, ast.Nonlocal, ast.ClassDef))
               or isinstance(node, getattr(ast, 'Match', ()))
               or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not tree
               for node in ast.walk(tree)):
            return body
        stores = [node.id for node in ast.walk(tree)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)]
        integers = self.parameters | {
            node.targets[0].id for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and stores.count(node.targets[0].id) == 1 and self.native_call(node.value)
        }

        class Divisibility(cst.CSTTransformer):
            def visit_Lambda(inner, node):
                return False

            def visit_FunctionDef(inner, node):
                return False

            visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_Lambda

            def leave_Comparison(inner, before, after):
                if len(before.comparisons) != 1:
                    return after
                expr = ast.parse(self.code(before), mode='eval').body
                if (not isinstance(expr.ops[0], (ast.Eq, ast.NotEq))
                        or not isinstance(expr.comparators[0], ast.Constant)
                        or type(expr.comparators[0].value) is not int
                        or expr.comparators[0].value != 0
                        or not isinstance(expr.left, ast.BinOp)
                        or not isinstance(expr.left.op, ast.Mod)):
                    return after
                dividend, divisor = expr.left.left, expr.left.right
                op = '_mx_divisible'
                if (isinstance(dividend, ast.BinOp) and isinstance(dividend.op, ast.Sub)
                        and isinstance(dividend.right, ast.Constant)
                        and type(dividend.right.value) is int and dividend.right.value == 1):
                    dividend = dividend.left
                    op = '_mx_divisible_after_decrement'
                if not all(isinstance(operand, ast.Name) and operand.id in integers
                           for operand in (dividend, divisor)):
                    return after
                prefix = 'not ' if isinstance(expr.ops[0], ast.NotEq) else ''
                return cst.parse_expression(f'({prefix}_mx_sys.{op}({dividend.id}, {divisor.id}))')

        return cst.Module(body).visit(Divisibility()).body

    def rewrite(self, function):
        kind = self.cell.get_rettype_expr(c_style=True)
        divisible = getattr(self.cell, '_spec', {}).get('native_divisibility', False)
        if type(divisible) is not bool:
            raise ValueError('native_divisibility must be a boolean')
        if ((kind not in ('double', 'long long') and not divisible)
                or self.cell.is_locked or self.cell.body_has_closure
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
        if divisible:
            body = self.divisibility(body, tree)
        if kind == 'double':
            updated = self.double_tail(body, tree)
        elif kind == 'long long':
            updated = self.integer_product(body, tree)
        else:
            updated = body
        return function.with_changes(body=function.body.with_changes(body=updated))
