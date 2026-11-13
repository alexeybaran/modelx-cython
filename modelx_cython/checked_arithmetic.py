"""Keep range-proven integer expressions in the existing Cython formula."""
import math
from collections import defaultdict
from typing import NamedTuple, Optional

import libcst as cst
from libcst.metadata import ExpressionContext

from .scopes import CstFacts, Names, has_scope_effects, lexical_nodes
from .syntax import (
    arguments, code, dotted_name, literal_value, local_assignment, single_statement,
    statements, walk,
)

_BOUND = 1 << 25
_SIGNED64 = -(1 << 63), (1 << 63) - 1
_INTEGER_ARITHMETIC = (cst.Add, cst.Subtract, cst.Multiply)
_ARITHMETIC = (*_INTEGER_ARITHMETIC, cst.Divide)


class _RangeInduction(NamedTuple):
    """An entered iteration satisfies counter >= (origin or zero) + offset."""
    counter: str
    origin: Optional[str]
    offset: int

    def nonnegative_difference(self, expr):
        return (self.origin is not None and isinstance(expr, cst.BinaryOperation)
                and isinstance(expr.operator, cst.Subtract)
                and dotted_name(expr.left) == self.counter and dotted_name(expr.right) == self.origin)


def _signed_assignment(assignment, value):
    return cst.AnnAssign(local_assignment(assignment), cst.Annotation(cst.parse_expression('_mx_cy.longlong')),
                         value, semicolon=assignment.semicolon)


def range_bounds(loop):
    if isinstance(loop, cst.For) and loop.asynchronous is None and isinstance(loop.target, cst.Name):
        args = arguments(loop.iter, 'range')
        if args is not None and len(args) in (1, 2):
            return args
    return None


def local_quotient(expr, divisor_type):
    if (not isinstance(expr, cst.BinaryOperation) or not isinstance(expr.operator, cst.Divide)
            or not isinstance(expr.left, cst.Name) or not isinstance(expr.right, divisor_type)):
        return None
    divisor = expr.right.evaluated_value
    valid = 0 < divisor <= _BOUND if divisor_type is cst.Integer else 0 < divisor < math.inf
    return (expr.left.value, divisor) if valid else None


def divisibility_operands(node):
    if not isinstance(node, cst.Comparison) or len(node.comparisons) != 1:
        return None
    comparison, = node.comparisons
    remainder = node.left
    if (not isinstance(comparison.operator, (cst.Equal, cst.NotEqual))
            or not isinstance(comparison.comparator, cst.Integer)
            or comparison.comparator.evaluated_value != 0
            or not isinstance(remainder, cst.BinaryOperation)
            or not isinstance(remainder.operator, cst.Modulo)):
        return None
    dividend = remainder.left
    operation = '_mx_divisible'
    if (isinstance(dividend, cst.BinaryOperation)
            and isinstance(dividend.right, cst.Integer) and dividend.right.evaluated_value == 1):
        if isinstance(dividend.operator, cst.Subtract):
            operation, dividend = '_mx_divisible_after_decrement', dividend.left
        elif isinstance(dividend.operator, cst.Add):
            operation, dividend = '_mx_divisible_after_increment', dividend.left
    return dividend, remainder.right, operation, isinstance(comparison.operator, cst.NotEqual)


def divisibility_enabled(cell):
    from . import automatic
    enabled = getattr(cell, '_spec', {}).get('native_divisibility', automatic.enabled(cell))
    if type(enabled) is not bool:
        raise ValueError('native_divisibility must be a boolean')
    return enabled


def signed_local_assignments(body, name, facts, native_value):
    """Find all signed definitions and the reads they definitely initialize."""
    reads = {node for node, context in facts.contexts.items()
             if isinstance(node, cst.Name) and node.value == name and context is ExpressionContext.LOAD}
    assignments = {}

    def flow(lines, assigned):
        for line in lines:
            evaluated = line.test if isinstance(line, cst.If) else line
            if not assigned and any(part in reads for part in walk(evaluated)):
                return None
            if isinstance(line, cst.If):
                taken = flow(statements(line.body), assigned)
                other = line.orelse
                if isinstance(other, cst.If):
                    skipped = flow((other,), assigned)
                elif other is not None:
                    skipped = flow(statements(other.body), assigned)
                else:
                    skipped = assigned
                if taken is None or skipped is None:
                    return None
                assigned = taken and skipped
            elif (target := local_assignment(line)) is not None and target.value == name:
                value = native_value(line.value)
                if value is None:
                    return None
                assignments[line] = value
                assigned = True
        return assigned

    if flow(statements(body), False) is None or len(assignments) != facts.stores[name]:
        return {}, reads
    return assignments, reads


def guarded_integer_intervals(test):
    """Read inclusive local bounds from a conjunction of signed comparisons."""
    if isinstance(test, cst.BooleanOperation) and isinstance(test.operator, cst.And):
        left = guarded_integer_intervals(test.left)
        right = guarded_integer_intervals(test.right)
        if left and right:
            return {**left, **right}
        return {}
    if (not isinstance(test, cst.Comparison) or len(test.comparisons) != 2
            or not all(isinstance(part.operator, cst.LessThanEqual) for part in test.comparisons)):
        return {}
    middle, upper = (part.comparator for part in test.comparisons)
    if not isinstance(middle, cst.Name):
        return {}
    try:
        low, high = literal_value(test.left), literal_value(upper)
    except (TypeError, ValueError):
        return {}
    if type(low) is int and type(high) is int and _SIGNED64[0] <= low <= high <= _SIGNED64[1]:
        return {middle.value: (low, high)}
    return {}


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

    def capture(self, expr, inputs, local_names):
        if isinstance(expr, cst.Integer) and -_BOUND <= expr.evaluated_value <= _BOUND:
            value = str(expr.evaluated_value)
            return value, value
        if self.native_call(expr) or isinstance(expr, cst.Name) and expr.value in (*self.parameters, *local_names):
            name = f'{self.prefix}_{self.next_input}'
            self.next_input += 1
            inputs.append((name, code(expr)))
            return name, f'_mx_cy.cast(object, {name})'
        if not isinstance(expr, cst.BinaryOperation):
            return None
        operator = {cst.Add: '+', cst.Subtract: '-', cst.Multiply: '*'}.get(type(expr.operator))
        if operator is None:
            return None
        left = self.capture(expr.left, inputs, local_names)
        right = self.capture(expr.right, inputs, local_names)
        if left is None or right is None:
            return None
        return tuple(f'({a} {operator} {b})' for a, b in zip(left, right))

    def guarded_region(self, expressions, render, *, locals_=(), target=None):
        """Capture signed inputs in order and keep exact arithmetic on a miss."""
        if any(self.integer_interval(value, locals_=locals_) is None for value in expressions):
            return ()
        inputs = []
        captured = [self.capture(value, inputs, locals_) for value in expressions]
        if None in captured or not inputs:
            return ()
        bounds = ' and '.join(f'-{_BOUND} <= {name} <= {_BOUND}' for name, _ in inputs)
        native, boxed = zip(*captured)
        hot, cold = render(native, True), render(boxed, False)
        source = ''.join(f'{name}: _mx_cy.longlong = {value}\n' for name, value in inputs)
        if target is None:
            source += f'if {bounds}:\n    return {hot}\nreturn {cold}\n'
        else:
            source += (f'{target}: _mx_cy.double\nif {bounds}:\n    {target} = {hot}\n'
                       f'else:\n    {target} = {cold}\n')
        return cst.parse_module(source).body

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
            return self.parameter_intervals.get(expr.value, inputs) if isinstance(expr, cst.Name) else inputs
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

    def guarded_float_return(self, body):
        """Lower a terminal quotient from its adjacent, ordered definitions."""
        if len(body) < 2 or not isinstance(result := single_statement(body[-1]), cst.Return):
            return body
        assignment = single_statement(body[-2])
        target = local_assignment(assignment)
        if target is None or result.value is None:
            return body
        nodes = tuple(walk(result.value))
        name = target.value
        if (name not in self.cell.params
                and sum(isinstance(node, cst.Name) and node.value == name for node in nodes) == 1):
            for node in nodes:
                term = local_quotient(node, cst.Integer)
                if term is None or term[0] != name or CstFacts(body).stores[name] != 1:
                    continue
                quotient = f'{self.prefix}_quotient'
                divisor = term[1]

                def render(values, native):
                    denominator = f'{divisor}.0' if native else divisor
                    return f'{values[0]} / {denominator}'

                region = self.guarded_region((assignment.value,), render, target=quotient)
                if region:
                    class Quotient(cst.CSTTransformer):
                        def leave_BinaryOperation(inner, before, after):
                            return cst.Name(quotient) if before is node else after

                    return (*body[:-2], *region, body[-1].visit(Quotient()))
                break
        first = single_statement(body[-3]) if len(body) >= 3 else None
        other_target = local_assignment(first)
        result = result.value
        if (other_target is None or other_target.value == name
                or not isinstance(result, cst.BinaryOperation) or not isinstance(result.operator, cst.Add)):
            return body
        names = other_target.value, name
        for division, other in ((result.left, result.right), (result.right, result.left)):
            term = local_quotient(division, cst.Float)
            if (term is None or not isinstance(other, cst.Name)
                    or {term[0], other.value} != set(names)):
                continue

            def render(values, _):
                terms = dict(zip(names, values))
                parts = f'{terms[term[0]]} / {term[1]!r}', terms[other.value]
                return ' + '.join(parts if division is result.left else reversed(parts))

            region = self.guarded_region((first.value, assignment.value), render)
            if region:
                return (*body[:-3], *region)
        return body

    def _integer_locals(self, body, stores=None):
        stores = CstFacts(body).stores if stores is None else stores
        return {target.value: node for node in statements(body)
                if (target := local_assignment(node)) is not None
                and stores[target.value] == 1 and self.native_call(node.value)}

    def _native_local(self, body, name, facts):
        if name in self.cell.params or name in facts.deleted or not facts.stores[name] or facts.stores['self']:
            return {}, set()
        return signed_local_assignments(
            body, name, facts, lambda value: value if self.native_call(value) else None)

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

    def _range_induction(self, counter, start, stop, known, facts):
        if counter in self.cell.params or facts.stores[counter] != 1:
            return None
        stable_stop = isinstance(stop, cst.Name) and stop.value in known
        if start is None:
            if stable_stop or (arguments(stop, count=0) is not None and self.native_call(stop)
                               and not self.cls.cells[stop.func.attr.value].params):
                return _RangeInduction(counter, None, 0)
            return None
        if not stable_stop:
            return None
        if isinstance(start, cst.Integer) and 0 <= start.evaluated_value <= _SIGNED64[1]:
            return _RangeInduction(counter, None, start.evaluated_value)
        if (isinstance(start, cst.BinaryOperation) and isinstance(start.operator, cst.Add)
                and isinstance(start.left, cst.Name) and start.left.value in self.parameters
                and isinstance(start.right, cst.Integer) and 0 < start.right.evaluated_value <= _BOUND):
            return _RangeInduction(counter, start.left.value, start.right.evaluated_value)
        return None

    def _range_uses(self, reads, parents, known, *, induction=None, comparison_facts=None):
        """Qualify every use, retaining the original boxed expression on a miss."""
        replacements, covered = {}, set()
        for read in reads:
            if read in covered:
                continue
            node = parents[read]
            if isinstance(node, cst.Arg) and self._native_index_argument(node, parents[node]):
                continue
            integer = read
            while isinstance(node, cst.BinaryOperation) and isinstance(node.operator, _INTEGER_ARITHMETIC):
                integer, node = node, parents[node]
            parts = set(walk(integer))
            value = None
            if (isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Divide)
                    and node.left is integer and isinstance(node.right, cst.Float)
                    and 0 < node.right.evaluated_value < math.inf):
                divisor = code(node.right)
                if induction is None and len(reads & parts) == 1:
                    value = self._checked_range_expression(
                        integer, known, lambda expr: f'({expr}) / {divisor}', 'double')
                elif induction is not None and induction.nonnegative_difference(integer):
                    counter, origin, _ = induction
                    limit = _SIGNED64[1]
                    value = cst.parse_expression(
                        f'(({counter} - {origin}) / {divisor} '
                        f'if {origin} >= 0 or {counter} <= {limit} + {origin} '
                        f'else _mx_cy.cast(_mx_cy.double, (_mx_cy.cast(object, {counter}) - {origin}) / {divisor}))')
            elif (comparison_facts is not None and isinstance(node, cst.Comparison)
                  and node.left is integer and len(node.comparisons) == 1):
                comparison, = node.comparisons
                operand = comparison.comparator
                if (isinstance(comparison.operator, (cst.LessThan, cst.LessThanEqual,
                                                     cst.GreaterThan, cst.GreaterThanEqual))
                        and isinstance(operand, cst.Name) and operand.value in self.cell.params
                        and not comparison_facts.stores[operand.value]
                        and self.cell.get_argtype_expr(operand.value, c_style=True, formula=True) == 'double'):
                    operator = code(comparison.operator).strip()
                    value = self._checked_range_expression(
                        integer, known, lambda expr: f'({expr}) {operator} {operand.value}', 'bint')
            if value is None:
                return None
            replacements[node] = value
            covered.update(parts)
        return replacements, covered

    def _range_offsets(self, lines, positions, reads, counter_reads, known, facts, parents):
        """Keep a local native when its offset and every subsequent use qualify."""
        replacements, covered = {}, set()
        for position, assignment in enumerate(lines):
            target = local_assignment(assignment)
            if target is None or target.value in known or facts.stores[target.value] != 1:
                continue
            parts = set(walk(assignment.value))
            if not parts & counter_reads:
                continue
            uses = reads[target.value]
            if not uses or any(positions.get(read, -1) <= position for read in uses):
                continue
            value = self._checked_range_expression(assignment.value, known, lambda expr: expr, 'longlong')
            transfers = self._range_uses(uses, parents, known | {target.value}) if value is not None else None
            if transfers is not None:
                replacements[assignment] = _signed_assignment(assignment, value)
                replacements.update(transfers[0])
                covered.update(parts)
        return replacements, covered

    def _range_locals(self, body):
        """Use a signed counter for a unit-step range with proven numeric consumers."""
        if 'range' not in self.builtin_keys:
            return body
        loops = [(position, line, bounds) for position, line in enumerate(body)
                 if (bounds := range_bounds(line)) is not None]
        if not loops:
            return body
        module = cst.Module(body)
        facts = CstFacts(module)
        if facts.stores['self'] or {'locals', 'vars', 'eval', 'exec'} & facts.reads:
            return body
        parents = {child: node for node in walk(module) for child in node.children}
        reads = defaultdict(set)
        for node, context in facts.contexts.items():
            if isinstance(node, cst.Name) and context is ExpressionContext.LOAD:
                reads[node.value].add(node)
        replacements = {}
        counters = set()
        for position, loop, bounds in loops:
            start, stop = bounds if len(bounds) == 2 else (None, bounds[0])
            prior = self._integer_locals(body[:position], facts.stores)
            known = self.parameters | prior.keys()
            induction = self._range_induction(loop.target.value, start, stop, known, facts)
            if induction is None:
                continue
            counter, origin, offset = induction
            lines = statements(loop.body)
            positions = {node: position for position, line in enumerate(lines) for node in lexical_nodes((line,))}
            counter_reads = reads[counter]
            if not counter_reads or not counter_reads <= positions.keys():
                continue
            known.add(counter)
            offsets = start is not None and origin is None and isinstance(loop.body, cst.IndentedBlock)
            consumers, covered = self._range_offsets(
                lines, positions, reads, counter_reads, known, facts, parents) if offsets else ({}, set())
            direct = self._range_uses(counter_reads - covered, parents, known, induction=induction,
                                      comparison_facts=facts if offsets else None)
            if direct is None:
                continue
            consumers.update(direct[0])
            covered.update(direct[1])
            replacements.update(consumers)
            replacements.update((assignment, _signed_assignment(assignment, assignment.value))
                                for name, assignment in prior.items() if reads[name] and reads[name] <= covered)
            if origin is not None:
                # Since stop is signed, the original range is empty if start exceeds its maximum.
                limit = _SIGNED64[1]
                replacements[start] = cst.parse_expression(
                    f'({origin} + {offset} if {origin} <= {limit - offset} else {limit})')
            replacements[loop] = cst.parse_statement(f'{counter}: _mx_cy.longlong\n')
            if start is None:
                counters.add(counter)
        if not replacements:
            return body
        self.cell.signed_range_counters = frozenset(counters)

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
        candidates = dict.fromkeys(
            target.value for node in lexical_nodes(body)
            if (target := local_assignment(node)) is not None and self.native_call(node.value)
        )
        if not candidates:
            return body
        module = cst.Module(body)
        facts = CstFacts(module)
        visible = set(lexical_nodes(body))
        if any(isinstance(node, cst.Call) and isinstance(node.func, cst.Name)
               and node.func.value in ('locals', 'vars', 'eval', 'exec') for node in visible):
            return body
        parents = {child: node for node in walk(module) for child in node.children}
        replacements = {}
        declarations = []
        top_level = set(statements(body))
        for name in candidates:
            assignments, reads = self._native_local(body, name, facts)
            if not assignments or not reads <= visible:
                continue
            shifts = {}
            for read in reads:
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
                    if len(assignments) == 1 and (assignment := next(iter(assignments))) in top_level:
                        replacements[assignment] = _signed_assignment(assignment, assignment.value)
                    else:
                        declarations.append(cst.parse_statement(f'{name}: _mx_cy.longlong\n'))
        if not replacements:
            return body

        class NativeLocal(cst.CSTTransformer):
            def on_leave(inner, before, after):
                return replacements.get(before, after)

        rewritten = module.visit(NativeLocal()).body
        doc_end = int(module.get_docstring() is not None)
        return (*rewritten[:doc_end], *declarations, *rewritten[doc_end:])

    def exact_integer_returns(self, body):
        """Keep compound terminal arithmetic native under a proven bound.

        Stage signed Cell results and their locals once. Outside the bound,
        use exact object arithmetic until the final signed return conversion.
        """
        assignments = {node: name for name, node in self._integer_locals(body).items()}
        if not assignments:
            return body
        available, replacements = set(), {}
        for statement in body:
            for line in lexical_nodes((statement,)):
                if not isinstance(result := single_statement(line), cst.Return):
                    continue
                value = result.value
                if not (isinstance(value, cst.BinaryOperation)
                        and any(isinstance(part, cst.BinaryOperation) for part in (value.left, value.right))):
                    continue
                parts = tuple(walk(value))
                if not (any(isinstance(part, cst.Name) and part.value in available for part in parts)
                        and any(self.native_call(part) for part in parts)
                        and self.integer_interval(value, signed=True, locals_=available) is None):
                    continue
                region = self.guarded_region((value,), lambda values, _: values[0], locals_=available)
                if region:
                    replacements[line] = cst.FlattenSentinel(region)
            available.update(assignments[node] for node in statements((statement,)) if node in assignments)

        class ExactReturns(cst.CSTTransformer):
            def leave_SimpleStatementLine(inner, before, after):
                return replacements.get(before, after)

        return cst.Module(body).visit(ExactReturns()).body if replacements else body

    def rewrite(self, function):
        self.cell.signed_range_counters = frozenset()
        kind = self.cell.get_rettype_expr(c_style=True)
        if (kind not in ('double', 'long long')
                or self.cell.is_locked or self.cell.body_has_closure
                or self.cell.cls in self.cls.visitor.dynamic_ref_classes
                or not isinstance(function.body, cst.IndentedBlock)):
            return function
        body = function.body.body
        if has_scope_effects(body):
            return function
        facts = CstFacts(body)
        self.parameters = {name for name in self.cell.params if name not in facts.stores
                           and self.cell.get_argtype_expr(name, c_style=True, formula=True) == 'long long'}
        self.prefix = Names(function).reserve('_mx_checked', family=True)
        self.next_input = 0
        if kind == 'double':
            rewritten = self.guarded_float_return(body)
            body = self.integer_local(body) if rewritten is body else rewritten
            body = self._range_locals(body)
        elif kind == 'long long':
            body = self.exact_integer_returns(body)
        return function.with_changes(body=function.body.with_changes(body=body))
