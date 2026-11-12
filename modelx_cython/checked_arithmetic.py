"""Keep range-proven integer expressions in the existing Cython formula."""
import math

import libcst as cst
from libcst.metadata import ExpressionContext

from .scopes import CstFacts, Names, has_scope_effects, lexical_nodes
from .syntax import arguments, code, statements, walk

_BOUND = 1 << 25
_SIGNED64 = -(1 << 63), (1 << 63) - 1
_ARITHMETIC = (cst.Add, cst.Subtract, cst.Multiply, cst.Divide)


class CheckedArithmetic:
    def __init__(self, cls, cell, builtin_keys=()):
        self.cls, self.cell = cls, cell
        self.builtin_keys = builtin_keys
        self.parameter_intervals = {}
        self.signed_atom = None

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

    def integer_interval(self, expr, *, signed=False, locals_=()):
        """Bound each intermediate under a checked or signed-native input domain."""
        inputs = _SIGNED64 if signed else (-_BOUND, _BOUND)
        limits = _SIGNED64 if signed else (-(1 << 53), 1 << 53)
        if isinstance(expr, cst.Integer):
            value = expr.evaluated_value
            return (value, value) if inputs[0] <= value <= inputs[1] else None
        if self.native_call(expr):
            return inputs
        if isinstance(expr, cst.Name) and (expr.value in self.parameters or expr.value in locals_):
            return self.parameter_intervals.get(expr.value, inputs) if signed else inputs
        if signed and self.signed_atom is not None and self.signed_atom(expr):
            return inputs
        if signed and isinstance(expr, cst.IfExp):
            yes = self.integer_interval(expr.body, signed=True, locals_=locals_)
            no = self.integer_interval(expr.orelse, signed=True, locals_=locals_)
            if yes is not None and no is not None:
                return min(yes[0], no[0]), max(yes[1], no[1])
            return None
        if signed and isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, (cst.Plus, cst.Minus)):
            if isinstance(expr.expression, cst.Integer):
                value = expr.expression.evaluated_value
                value = -value if isinstance(expr.operator, cst.Minus) else value
                return (value, value) if limits[0] <= value <= limits[1] else None
            operand = self.integer_interval(expr.expression, signed=True, locals_=locals_)
            if operand is None or isinstance(expr.operator, cst.Plus):
                return operand
            return (-operand[1], -operand[0]) if operand[0] != limits[0] else None
        if not isinstance(expr, cst.BinaryOperation):
            return None
        left = self.integer_interval(expr.left, signed=signed, locals_=locals_)
        right = self.integer_interval(expr.right, signed=signed, locals_=locals_)
        if left is None or right is None:
            return None
        if isinstance(expr.operator, cst.Add):
            interval = left[0] + right[0], left[1] + right[1]
        elif isinstance(expr.operator, cst.Subtract):
            interval = left[0] - right[1], left[1] - right[0]
        elif isinstance(expr.operator, cst.Multiply):
            products = [a * b for a in left for b in right]
            interval = min(products), max(products)
        elif signed and isinstance(expr.operator, cst.FloorDivide):
            if right[0] == right[1] > 0:
                interval = left[0] // right[0], left[1] // right[0]
            elif right[0] > 0:
                interval = min(0, left[0]), max(0, left[1])
            else:
                magnitude = max(abs(left[0]), abs(left[1]))
                interval = -magnitude, magnitude
        elif signed and isinstance(expr.operator, cst.Modulo):
            if left[0] == limits[0] and right[0] <= -1 <= right[1]:
                return None
            interval = min(0, right[0] + 1), max(0, right[1] - 1)
        else:
            return None
        return interval if limits[0] <= interval[0] <= interval[1] <= limits[1] else None

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

    def local_division(self, body):
        """Keep a terminal local integer quotient in the native float expression."""
        if len(body) < 2 or any(not isinstance(line, cst.SimpleStatementLine) or len(line.body) != 1
                                for line in body[-2:]):
            return body
        assignment, result = (line.body[0] for line in body[-2:])
        if (not isinstance(assignment, cst.Assign) or len(assignment.targets) != 1
                or not isinstance(assignment.targets[0].target, cst.Name)
                or not isinstance(result, cst.Return) or result.value is None):
            return body
        name = assignment.targets[0].target.value
        if (name in self.cell.params or self.integer_interval(assignment.value) is None
                or CstFacts(body).stores[name] != 1):
            return body
        nodes = tuple(walk(result.value))
        if sum(isinstance(node, cst.Name) and node.value == name for node in nodes) != 1:
            return body
        division = next((node for node in nodes
                         if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Divide)
                         and isinstance(node.left, cst.Name) and node.left.value == name
                         and isinstance(node.right, cst.Integer)
                         and 0 < node.right.evaluated_value <= _BOUND), None)
        if division is None:
            return body
        captured = self.capture(assignment.value)
        if captured is None or not self.inputs:
            return body
        numerator, boxed = captured
        divisor = division.right.evaluated_value
        quotient = f'{self.prefix}_quotient'
        bounds = ' and '.join(f'-{_BOUND} <= {local} <= {_BOUND}' for local, _ in self.inputs)
        prefix = cst.parse_module(
            ''.join(f'{local}: _mx_cy.longlong = {source}\n' for local, source in self.inputs)
            + f'{quotient}: _mx_cy.double\n'
            + f'if {bounds}:\n'
            + f'    {quotient} = {numerator} / {divisor}.0\n'
            + f'else:\n'
            + f'    {quotient} = {boxed} / {divisor}\n'
        ).body

        class Quotient(cst.CSTTransformer):
            def leave_BinaryOperation(inner, before, after):
                return cst.Name(quotient) if before is division else after

        return (*body[:-2], *prefix, body[-1].visit(Quotient()))

    def _integer_locals(self, body):
        stores = CstFacts(body).stores
        return {node.targets[0].target.value for node in statements(body)
                if isinstance(node, cst.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0].target, cst.Name)
                and stores[node.targets[0].target.value] == 1 and self.native_call(node.value)}

    def _native_index_argument(self, argument, call):
        args = arguments(call)
        if (args is None or not isinstance(call.func, cst.Attribute)
                or not isinstance(call.func.value, cst.Name) or call.func.value.value != 'self'):
            return False
        cell = self.cls.cells.get(call.func.attr.value)
        if cell is None or cell.is_special() or not cell.has_typeinfo() or len(args) != len(cell.params):
            return False
        param = cell.params[call.args.index(argument)]
        return cell.get_argtype_expr(param, c_style=True) == 'long long'

    def _checked_range_expression(self, integer, known, render, type_name):
        """Use the ordinary numeric expression with boxed arithmetic outside its proven bound."""
        nodes = tuple(walk(integer))
        if (any(isinstance(node, cst.Call) for node in nodes)
                or self.integer_interval(integer, locals_=known) is None):
            return None
        names = {node.value for node in nodes if isinstance(node, cst.Name)}
        if not names:
            return None
        bound = ' and '.join(f'-{_BOUND} <= {name} <= {_BOUND}' for name in sorted(names))

        class BoxNames(cst.CSTTransformer):
            def leave_Name(inner, before, after):
                return cst.parse_expression(f'_mx_cy.cast(object, {after.value})')

        native, boxed = render(code(integer)), render(code(integer.visit(BoxNames())))
        return cst.parse_expression(
            f'({native} if {bound} else _mx_cy.cast(_mx_cy.{type_name}, {boxed}))')

    def _range_arithmetic(self, loop, facts, prior, counter):
        """Qualify integer offsets and their scalar consumers inside a bounded range."""
        if not isinstance(loop.body, cst.IndentedBlock):
            return {}, set()
        locals_ = {name for name in self._integer_locals(prior) if facts.stores[name] == 1}
        known = locals_ | self.parameters | {counter}
        replacements, covered = {}, set()
        consumers = tuple(lexical_nodes((loop.body,)))
        loop_lines = statements(loop.body)
        for offset, assignment in enumerate(loop_lines):
            if not (isinstance(assignment, cst.Assign) and len(assignment.targets) == 1
                    and isinstance(target := assignment.targets[0].target, cst.Name)
                    and target.value not in known and facts.stores[target.value] == 1
                    and any(isinstance(item, cst.Name) and item.value == counter
                            for item in walk(assignment.value))):
                continue
            reads = {node for node, context in facts.contexts.items()
                     if isinstance(node, cst.Name) and node.value == target.value
                     and context is ExpressionContext.LOAD}
            if not reads or not reads <= set(lexical_nodes(loop_lines[offset + 1:])):
                continue
            value = self._checked_range_expression(assignment.value, known, lambda expr: expr, 'longlong')
            if value is None:
                continue
            uses = {}
            for node in consumers:
                if isinstance(node, cst.Call):
                    for arg in node.args:
                        if arg.value in reads and self._native_index_argument(arg, node):
                            uses[arg.value] = None
                elif (isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Divide)
                      and isinstance(node.right, cst.Float) and 0 < node.right.evaluated_value < math.inf):
                    signed_reads = reads.intersection(walk(node.left))
                    if len(signed_reads) == 1:
                        divisor = code(node.right)
                        quotient = self._checked_range_expression(
                            node.left, known | {target.value}, lambda expr: f'({expr}) / {divisor}', 'double')
                        if quotient is not None:
                            uses[next(iter(signed_reads))] = (node, quotient)
            if set(uses) != reads:
                continue
            replacements[assignment] = cst.AnnAssign(
                target=target, annotation=cst.Annotation(cst.parse_expression('_mx_cy.longlong')),
                value=value, semicolon=assignment.semicolon)
            replacements.update(value for value in uses.values() if value is not None)
            covered.update(walk(assignment.value))

        for node in consumers:
            if (not isinstance(node, cst.Comparison) or len(node.comparisons) != 1
                    or not any(isinstance(item, cst.Name) and item.value == counter for item in walk(node.left))):
                continue
            comparison, = node.comparisons
            operand = comparison.comparator
            if (not isinstance(comparison.operator, (cst.LessThan, cst.LessThanEqual,
                                                      cst.GreaterThan, cst.GreaterThanEqual))
                    or not isinstance(operand, cst.Name) or operand.value not in self.cell.params
                    or facts.stores[operand.value]
                    or self.cell.get_argtype_expr(operand.value, c_style=True, formula=True) != 'double'):
                continue
            operator = code(comparison.operator).strip()
            value = self._checked_range_expression(
                node.left, known, lambda expr: f'({expr}) {operator} {operand.value}', 'bint')
            if value is not None:
                replacements[node] = value
                covered.update(walk(node.left))
        if replacements:
            # These scalar Cells already return signed values. Keep their local
            # bindings native when every use is inside the qualified loop.
            for assignment in statements(prior):
                if not (isinstance(assignment, cst.Assign) and len(assignment.targets) == 1
                        and isinstance(target := assignment.targets[0].target, cst.Name)
                        and target.value in locals_):
                    continue
                reads = {node for node, context in facts.contexts.items()
                         if isinstance(node, cst.Name) and node.value == target.value
                         and context is ExpressionContext.LOAD}
                if reads and reads <= covered:
                    replacements[assignment] = cst.AnnAssign(
                        target=target, annotation=cst.Annotation(cst.parse_expression('_mx_cy.longlong')),
                        value=assignment.value, semicolon=assignment.semicolon)
        return replacements, covered

    def _range_locals(self, body):
        """Use a signed counter for a unit-step range with proven numeric consumers."""
        if 'range' not in self.builtin_keys:
            return body
        loops = [line for line in body if isinstance(line, cst.For) and line.asynchronous is None
                 and isinstance(line.target, cst.Name) and arguments(line.iter, 'range', 2)]
        if not loops:
            return body
        module = cst.Module(body)
        facts = CstFacts(module)
        if {'locals', 'vars', 'eval', 'exec'} & facts.reads:
            return body
        parents = {child: node for node in walk(module) for child in node.children}
        replacements = {}
        for loop in loops:
            counter = loop.target.value
            start, stop = arguments(loop.iter)
            known = self.parameters | self._integer_locals(body[:body.index(loop)])
            if (counter in self.cell.params or facts.stores[counter] != 1
                    or not isinstance(stop, cst.Name) or stop.value not in known
                    or facts.stores[stop.value] > 1):
                continue
            origin = None
            if (isinstance(start, cst.BinaryOperation) and isinstance(start.operator, cst.Add)
                    and isinstance(start.left, cst.Name) and start.left.value in self.parameters
                    and isinstance(start.right, cst.Integer) and 0 < start.right.evaluated_value <= _BOUND):
                origin = start.left.value
            elif not isinstance(start, cst.Integer) or not 0 <= start.evaluated_value <= _SIGNED64[1]:
                continue
            reads = {node for node, context in facts.contexts.items()
                     if isinstance(node, cst.Name) and node.value == counter
                     and context is ExpressionContext.LOAD}
            if not reads or not reads <= set(lexical_nodes((loop.body,))):
                continue
            consumers, covered = ({}, set()) if origin is not None else self._range_arithmetic(
                loop, facts, body[:body.index(loop)], counter)
            for index in reads:
                if index in covered:
                    continue
                use = parents[index]
                if isinstance(use, cst.Arg) and self._native_index_argument(use, parents[use]):
                    continue
                divide = parents.get(use)
                if not (origin is not None and isinstance(use, cst.BinaryOperation) and use.left is index
                        and isinstance(use.operator, cst.Subtract) and isinstance(use.right, cst.Name)
                        and use.right.value == origin and isinstance(divide, cst.BinaryOperation)
                        and divide.left is use and isinstance(divide.operator, cst.Divide)
                        and isinstance(divide.right, cst.Float) and 0 < divide.right.evaluated_value < math.inf):
                    break
                # The range orders index >= origin. Box only if the difference exceeds int64.
                limit = _SIGNED64[1]
                divisor = code(divide.right)
                consumers[divide] = cst.parse_expression(
                    f'(({counter} - {origin}) / {divisor} '
                    f'if {origin} >= 0 or {counter} <= {limit} + {origin} '
                    f'else _mx_cy.cast(_mx_cy.double, (_mx_cy.cast(object, {counter}) - {origin}) / {divisor}))')
            else:
                replacements.update(consumers)
                if origin is not None:
                    # If the start overflows, Python's unit-step range is empty.
                    limit = _SIGNED64[1]
                    offset = start.right.evaluated_value
                    replacements[start] = cst.parse_expression(
                        f'({origin} + {offset} if {origin} <= {limit - offset} else {limit})')
                replacements[loop] = cst.parse_statement(f'{counter}: _mx_cy.longlong\n')
        if not replacements:
            return body

        class NativeRange(cst.CSTTransformer):
            def on_leave(inner, before, after):
                replacement = replacements.get(before)
                if isinstance(before, cst.For) and replacement is not None:
                    return cst.FlattenSentinel((replacement, after))
                return after if replacement is None else replacement

        return module.visit(NativeRange()).body

    @staticmethod
    def constant_kind(expr):
        if isinstance(expr, cst.Float):
            return float
        if isinstance(expr, cst.Integer):
            return int
        if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, _ARITHMETIC):
            left, right = (CheckedArithmetic.constant_kind(arg) for arg in (expr.left, expr.right))
            if left is not None and right is not None:
                return float if float in (left, right) or isinstance(expr.operator, cst.Divide) else int
        return None

    def float_consumer(self, expr, parents):
        parent = parents.get(expr)
        if not isinstance(parent, cst.BinaryOperation) or not isinstance(parent.operator, _ARITHMETIC):
            return False
        other = parent.right if parent.left is expr else parent.left
        return self.constant_kind(other) is float

    def integer_local(self, body):
        """Keep a Cell's integer local native across comparisons and checked float arithmetic."""
        assignments = [
            node for node in statements(body)
            if isinstance(node, cst.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0].target, cst.Name) and self.native_call(node.value)
        ]
        if not assignments:
            return body
        module = cst.Module(body)
        facts = CstFacts(module)
        visible = set(lexical_nodes(body))
        if any(isinstance(node, cst.Call) and isinstance(node.func, cst.Name)
               and node.func.value in ('locals', 'vars', 'eval', 'exec') for node in visible):
            return body
        parents = {child: node for node in walk(module) for child in node.children}
        replacements = {}
        for assignment in assignments:
            target = assignment.targets[0].target
            name = target.value
            if facts.stores[name] != 1 or name in self.cell.params or name in facts.deleted:
                continue
            shifts = {}
            for read, context in facts.contexts.items():
                if not (isinstance(read, cst.Name) and read.value == name and context is ExpressionContext.LOAD):
                    continue
                if read not in visible:
                    break
                parent = parents[read]
                comparison = parents[parent] if isinstance(parent, cst.ComparisonTarget) else parent
                if isinstance(comparison, cst.Comparison) and len(comparison.comparisons) == 1:
                    test, = comparison.comparisons
                    other = test.comparator if comparison.left is read else comparison.left
                    if (isinstance(test.operator, (cst.Equal, cst.NotEqual, cst.LessThan, cst.LessThanEqual,
                                                   cst.GreaterThan, cst.GreaterThanEqual))
                            and self.integer_interval(other, signed=True) is not None):
                        continue
                if isinstance(parent, cst.Arg):
                    call = parents[parent]
                    builtin = call.func.value if isinstance(call, cst.Call) and isinstance(call.func, cst.Name) else None
                    if builtin in ('min', 'max') and builtin in self.builtin_keys and self.float_consumer(call, parents):
                        args = arguments(call, builtin, 2)
                        if args and self.integer_interval(args[1] if args[0] is read else args[0],
                                                          signed=True) is not None:
                            continue
                if (isinstance(parent, cst.BinaryOperation) and parent.left is read
                        and isinstance(parent.operator, (cst.Add, cst.Subtract))
                        and isinstance(parent.right, cst.Integer)
                        and 0 < (amount := parent.right.evaluated_value) <= _SIGNED64[1]
                        and self.float_consumer(parent, parents)):
                    delta = amount if isinstance(parent.operator, cst.Add) else -amount
                    check = (f'{name} <= {_SIGNED64[1] - delta}' if delta > 0
                             else f'{name} >= {_SIGNED64[0] - delta}')
                    shifts[parent] = cst.parse_expression(
                        f'(_mx_cy.cast(_mx_cy.double, {name} + ({delta})) if {check} '
                        f'else _mx_cy.cast(_mx_cy.double, _mx_cy.cast(object, {name}) + ({delta})))')
                    continue
                break
            else:
                if shifts:
                    replacements.update(shifts)
                    replacements[assignment] = cst.AnnAssign(
                        target=target, annotation=cst.Annotation(cst.parse_expression('_mx_cy.longlong')),
                        value=assignment.value, semicolon=assignment.semicolon)
        if not replacements:
            return body

        class NativeLocal(cst.CSTTransformer):
            def on_leave(inner, before, after):
                return replacements.get(before, after)

        return module.visit(NativeLocal()).body

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
                           and self.cell.get_argtype_expr(name, c_style=True, formula=True) == 'long long'}
        self.prefix = Names(function).reserve('_mx_checked', family=True)
        self.inputs, self.next_input = [], 0
        if divisible:
            body = self.divisibility(body)
        if kind == 'double':
            rewritten = self.local_division(body)
            if rewritten is body:
                rewritten = self.double_tail(body)
            body = self.integer_local(body) if rewritten is body else rewritten
            body = self._range_locals(body)
        elif kind == 'long long':
            body = self.integer_product(body)
        return function.with_changes(body=function.body.with_changes(body=body))
