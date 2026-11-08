"""Keep range-proven integer expressions in the existing Cython formula."""
import math

import libcst as cst

from .scopes import CstFacts, Names, has_scope_effects
from .syntax import arguments, code, statements

_BOUND = 1 << 25


class CheckedArithmetic:
    def __init__(self, cls, cell):
        self.cls, self.cell = cls, cell

    def native_call(self, expr):
        args = arguments(expr)
        if (args is None or not isinstance(expr.func, cst.Attribute)
                or not isinstance(expr.func.value, cst.Name) or expr.func.value.value != 'self'
                or not all(isinstance(arg, cst.Name) and arg.value in self.parameters
                           or isinstance(arg, cst.Integer) and -_BOUND <= arg.evaluated_value <= _BOUND
                           for arg in args)):
            return False
        cell = self.cls.cells.get(expr.func.attr.value)
        return (cell is not None and not cell.is_special() and not cell.called_with_kwargs
                and not cell.body_has_closure and cell.get_rettype_expr(c_style=True) == 'long long')

    def capture(self, expr, local_names=()):
        if isinstance(expr, cst.Integer) and -_BOUND <= expr.evaluated_value <= _BOUND:
            value = str(expr.evaluated_value)
            return value, value
        if self.native_call(expr) or isinstance(expr, cst.Name) and expr.value in (*self.parameters, *local_names):
            name = f'{self.prefix}_{self.next_input}'
            self.next_input += 1
            self.inputs.append((name, code(expr)))
            return name, f'_mx_cy.cast(object, {name})'
        if not isinstance(expr, cst.BinaryOperation):
            return None
        operator = {cst.Add: '+', cst.Subtract: '-', cst.Multiply: '*'}.get(type(expr.operator))
        if operator is None:
            return None
        left, right = self.capture(expr.left, local_names), self.capture(expr.right, local_names)
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

    def integer_interval(self, expr):
        """Bound every intermediate under the emitted native-input guards."""
        if isinstance(expr, cst.Integer):
            value = expr.evaluated_value
            return (value, value) if -_BOUND <= value <= _BOUND else None
        if self.native_call(expr) or isinstance(expr, cst.Name) and expr.value in self.parameters:
            return -_BOUND, _BOUND
        if not isinstance(expr, cst.BinaryOperation):
            return None
        left, right = self.integer_interval(expr.left), self.integer_interval(expr.right)
        if left is None or right is None:
            return None
        if isinstance(expr.operator, cst.Add):
            interval = left[0] + right[0], left[1] + right[1]
        elif isinstance(expr.operator, cst.Subtract):
            interval = left[0] - right[1], left[1] - right[0]
        elif isinstance(expr.operator, cst.Multiply):
            products = [a * b for a in left for b in right]
            interval = min(products), max(products)
        else:
            return None
        return interval if -(1 << 53) <= interval[0] <= interval[1] <= (1 << 53) else None

    def double_tail(self, body):
        if len(body) < 3 or any(not isinstance(line, cst.SimpleStatementLine) or len(line.body) != 1
                                for line in body[-3:]):
            return body
        left, right, result = (line.body[0] for line in body[-3:])
        if (not isinstance(result, cst.Return) or not all(isinstance(node, cst.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0].target, cst.Name)
                for node in (left, right))):
            return body
        first, second = left.targets[0].target.value, right.targets[0].target.value
        if first == second:
            return body
        expr = result.value
        if not (isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.Add)
                and isinstance(expr.left, cst.BinaryOperation) and isinstance(expr.left.operator, cst.Divide)
                and isinstance(expr.left.left, cst.Name) and expr.left.left.value == first
                and isinstance(expr.right, cst.Name) and expr.right.value == second
                and isinstance(expr.left.right, cst.Float)
                and math.isfinite(expr.left.right.evaluated_value) and expr.left.right.evaluated_value > 0):
            return body
        expressions = left.value, right.value
        if any(self.integer_interval(value) is None for value in expressions):
            return body
        captured = [self.capture(value) for value in expressions]
        if None in captured or not self.inputs:
            return body
        divisor = repr(expr.left.right.evaluated_value)
        native = self.statements(captured, lambda a, b: f'{a} / {divisor} + {b}')
        return (*body[:-3], *cst.parse_module(native).body)

    def _integer_locals(self, body):
        stores = CstFacts(body).stores
        return {node.targets[0].target.value for node in statements(body)
                if isinstance(node, cst.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0].target, cst.Name)
                and stores[node.targets[0].target.value] == 1 and self.native_call(node.value)}

    def integer_product(self, body):
        native_locals = self._integer_locals(body)
        owner = self

        class Product(cst.CSTTransformer):
            def visit_FunctionDef(inner, node):
                return False

            visit_ClassDef = visit_Lambda = visit_GeneratorExp = visit_FunctionDef
            visit_ListComp = visit_SetComp = visit_DictComp = visit_FunctionDef

            def leave_SimpleStatementLine(inner, before, after):
                if len(before.body) != 1 or not isinstance(before.body[0], cst.Return):
                    return after
                expr = before.body[0].value
                if not (isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.Multiply)
                        and isinstance(expr.right, cst.BinaryOperation)
                        and isinstance(expr.right.operator, cst.Subtract)
                        and isinstance(expr.right.right, cst.Integer)
                        and expr.right.right.evaluated_value == 1 and owner.native_call(expr.right.left)
                        and isinstance(expr.left, cst.Name) and expr.left.value in native_locals):
                    return after
                owner.inputs = []
                captured = owner.capture(expr, native_locals)
                assert captured
                native = owner.statements((captured,), lambda value: value)
                return cst.FlattenSentinel(cst.parse_module(native).body)

        return cst.Module(body).visit(Product()).body

    def divisibility(self, body):
        integers = self.parameters | self._integer_locals(body)

        class Divisibility(cst.CSTTransformer):
            def visit_Lambda(inner, node):
                return False

            def visit_FunctionDef(inner, node):
                return False

            visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_Lambda

            def leave_Comparison(inner, before, after):
                if len(before.comparisons) != 1:
                    return after
                comparison = before.comparisons[0]
                expr = before.left
                if (not isinstance(comparison.operator, (cst.Equal, cst.NotEqual))
                        or not isinstance(comparison.comparator, cst.Integer)
                        or comparison.comparator.evaluated_value != 0
                        or not isinstance(expr, cst.BinaryOperation)
                        or not isinstance(expr.operator, cst.Modulo)):
                    return after
                dividend, divisor = expr.left, expr.right
                operation = '_mx_divisible'
                if (isinstance(dividend, cst.BinaryOperation) and isinstance(dividend.operator, cst.Subtract)
                        and isinstance(dividend.right, cst.Integer) and dividend.right.evaluated_value == 1):
                    dividend = dividend.left
                    operation = '_mx_divisible_after_decrement'
                if not all(isinstance(value, cst.Name) and value.value in integers for value in (dividend, divisor)):
                    return after
                prefix = 'not ' if isinstance(comparison.operator, cst.NotEqual) else ''
                return cst.parse_expression(f'({prefix}_mx_sys.{operation}({dividend.value}, {divisor.value}))')

        return cst.Module(body).visit(Divisibility()).body

    def rewrite(self, function):
        kind = self.cell.get_rettype_expr(c_style=True)
        from modelx_cython import automatic
        divisible = getattr(self.cell, '_spec', {}).get('native_divisibility', automatic.enabled(self.cell))
        if type(divisible) is not bool:
            raise ValueError('native_divisibility must be a boolean')
        if ((kind not in ('double', 'long long') and not divisible)
                or self.cell.is_locked or self.cell.body_has_closure
                or self.cell.cls in self.cls.visitor.dynamic_ref_classes
                or not isinstance(function.body, cst.IndentedBlock)):
            return function
        body = function.body.body
        if has_scope_effects(body):
            return function
        assigned = CstFacts(body).stores
        self.parameters = {name for name in self.cell.params if name not in assigned
                           and self.cell.get_argtype_expr(name, c_style=True) == 'long long'}
        self.prefix = Names(function).reserve('_mx_checked', family=True)
        self.inputs, self.next_input = [], 0
        if divisible:
            body = self.divisibility(body)
        if kind == 'double':
            body = self.double_tail(body)
        elif kind == 'long long':
            body = self.integer_product(body)
        return function.with_changes(body=function.body.with_changes(body=body))
