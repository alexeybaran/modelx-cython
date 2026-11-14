"""Guarded integer arithmetic on C-integer locals (D76).

With ``infer_types`` (W1.5 0005) Cython types integer locals and ``range`` targets as C
``long long`` and checks overflow, so an operation whose exact result leaves the 64-bit
range raises OverflowError where Python would carry on with a big int. Instead of keeping
such locals Python objects (D72, which cost DIA_US_S twice its time), every maximal integer
expression over integer names and literals is emitted in guarded form::

    (<the expression> if -B <= a <= B and -B <= b <= B else <the same on Python ints>)

``B`` is chosen per expression so that, with every name within ``[-B, B]``, no
intermediate leaves the 64-bit range (additions about 2**61, a product 2**31, ...); the
fast path is the C operation, the slow path the exact Python computation. The slow path
is shaped by where the value goes:

- a ``float(...)``, a true division, a comparison: the whole call/division/comparison is
  guarded, so the result (double or bool) is exact whatever the operands;
- the value of an integer local (an assignment, ``x += ...``) or a ``range`` bound: the
  exact result converted to ``long long`` (an exact value outside the 64-bit range raises
  there, as a native local always has: test_signed_local_flow);
- anything else: the exact Python result (the expression is then a Python object).

Only names that the local-kind analysis proves integers (parameters, cell results,
literals, locals of integer kind, ``range`` targets) take part; expressions with calls,
attributes or subscripts are left as they are. Identity comparisons keep D72's ``object``
declaration (integer_boundaries).
"""

import libcst as cst

from .consts import CY_MOD, MX_SYS_MOD
from .scopes import OPAQUE_SCOPES

LIMIT = (1 << 63) - 1
_CANDIDATES = tuple(1 << bits for bits in (61, 48, 40, 31, 25, 20, 15, 10, 6))
_ARITHMETIC = (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo)
_COMPARISONS = (cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual, cst.Equal,
                cst.NotEqual)
_AUGMENTED = {cst.AddAssign: cst.Add, cst.SubtractAssign: cst.Subtract, cst.MultiplyAssign: cst.Multiply,
              cst.FloorDivideAssign: cst.FloorDivide, cst.ModuloAssign: cst.Modulo}


def _code(node):
    return cst.Module([]).code_for_node(node)


def _magnitude(expr, bound):
    """Largest |value| of an arithmetic tree with every name within [-bound, bound], or None."""
    if isinstance(expr, cst.Integer):
        return expr.evaluated_value
    if isinstance(expr, cst.Name):
        return bound
    if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, (cst.Minus, cst.Plus)):
        return _magnitude(expr.expression, bound)
    if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, _ARITHMETIC):
        left, right = _magnitude(expr.left, bound), _magnitude(expr.right, bound)
        if left is None or right is None:
            return None
        if isinstance(expr.operator, (cst.Add, cst.Subtract)):
            return left + right
        if isinstance(expr.operator, cst.Multiply):
            return left * right
        return max(left, right) + 1          # floor division and modulo
    return None


class _Arithmetic:
    """Recognises integer expressions over proven integer names and literals."""

    def __init__(self, kind, integer_names):
        self.kind, self.names = kind, integer_names

    def leaf(self, node):
        return (isinstance(node, cst.Integer)
                or isinstance(node, cst.Name) and node.value in self.names)

    def is_expression(self, node):
        """An integer operation (not a bare name or literal) with only integer leaves."""
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
            return self.is_operand(node.expression) and not isinstance(node.expression, cst.Integer)
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, _ARITHMETIC):
            return self.is_operand(node.left) and self.is_operand(node.right)
        return False

    def is_operand(self, node):
        return self.leaf(node) or self.is_expression(node)

    def names_in(self, node):
        found = []
        stack = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, cst.Name):
                if item.value not in found:
                    found.append(item.value)
            else:
                stack.extend(item.children)
        return sorted(found)


def _bound(expr):
    if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.Divide):
        # a true division: each integer side must stay exact before the division
        bounds = [_bound(side) for side in (expr.left, expr.right) if not isinstance(side, (cst.Float, cst.Name))]
        return min(bounds) if bounds and None not in bounds else None
    for bound in _CANDIDATES:
        magnitude = _magnitude(expr, bound)
        if magnitude is not None and magnitude <= LIMIT:
            return bound
    return None


def _native_callee(func):
    """``self.<cell>(..)``, a ``_mx*`` provider or runtime function, builtin min/max/abs."""
    if isinstance(func, cst.Attribute):
        base = func.value
        while isinstance(base, cst.Attribute):
            base = base.value
        return isinstance(base, cst.Name) and (base.value == 'self' or base.value.startswith('_mx'))
    return isinstance(func, cst.Name) and func.value in ('min', 'max', 'abs')


def _parenthesized(node):
    return node.with_changes(lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))


def _boxed(expr):
    class Box(cst.CSTTransformer):
        def leave_Name(self, before, after):
            return cst.parse_expression(f'{CY_MOD}.cast(object, {after.value})')
    return expr.visit(Box())


def _condition(names, bound):
    """-bound <= x < bound for every name, in one comparison: ``x ^ (x >> 63)`` is |x| (or
    |x| - 1) for any int, the OR of non-negative values is below a power of two only if each
    is, and the bound is a C constant (a literal above 2**31 would be a Python int, making
    the test a Python comparison)."""
    folded = ' | '.join(f'({name} ^ ({name} >> 63))' for name in names)
    return f'({folded}) < {CY_MOD}.cast({CY_MOD}.longlong, {bound})'


def native_branch(node):
    """The native expression of a guard this pass emitted, or None.

    A guard ``(E if <bound test> else <E on Python ints>)`` always has the exact value of ``E``:
    the slow branch computes it exactly and the fast branch runs only where no intermediate of
    ``E`` can leave the 64-bit range. So a later analysis may read the guard as ``E``, and where
    ``E`` is proven not to overflow for every value its names can take, ``E`` alone is the same
    value (diag-us2).
    """
    if not isinstance(node, cst.IfExp) or not isinstance(node.test, cst.Comparison):
        return None
    names = _guard_names(node.test)
    if not names or not names <= {item.value for item in _walk_names(node.body)}:
        return None
    return node.body.with_changes(lpar=(), rpar=())


def _guard_names(test):
    """The names of ``(x ^ (x >> 63)) | ... < cast(longlong, B)`` (``_condition``), or None."""
    if len(test.comparisons) != 1 or not isinstance(test.comparisons[0].operator, cst.LessThan):
        return None
    bound = test.comparisons[0].comparator
    if not (isinstance(bound, cst.Call) and len(bound.args) == 2 and _code(bound.func) == f'{CY_MOD}.cast'
            and _code(bound.args[0].value) == f'{CY_MOD}.longlong'
            and isinstance(bound.args[1].value, cst.Integer)
            and bound.args[1].value.evaluated_value in _CANDIDATES):
        return None
    names, stack = set(), [test.left]
    while stack:
        part = stack.pop()
        if isinstance(part, cst.BinaryOperation) and isinstance(part.operator, cst.BitOr):
            stack.extend((part.left, part.right))
        elif (isinstance(part, cst.BinaryOperation) and isinstance(part.operator, cst.BitXor)
              and isinstance(part.left, cst.Name) and isinstance(part.right, cst.BinaryOperation)
              and isinstance(part.right.operator, cst.RightShift)
              and isinstance(part.right.left, cst.Name) and part.right.left.value == part.left.value
              and isinstance(part.right.right, cst.Integer) and part.right.right.evaluated_value == 63):
            names.add(part.left.value)
        else:
            return None
    return names


def _walk_names(node):
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, cst.Name):
            yield item
        else:
            stack.extend(item.children)


def long_long_guard(node):
    """True when ``node`` is a guard whose slow branch takes the exact value as long long (an
    integer local, a range bound, an index, a native argument): there a value outside the
    64-bit range raises on both branches."""
    if native_branch(node) is None:
        return False
    slow = node.orelse
    if not (isinstance(slow, cst.Call) and len(slow.args) == 2):
        return False
    # `cast(longlong, exact)`, or the named conversion of D96 `_mx_exact_int64(exact, path)`
    return ((_code(slow.func) == f'{CY_MOD}.cast' and _code(slow.args[0].value) == f'{CY_MOD}.longlong')
            or _code(slow.func) == f'{MX_SYS_MOD}._mx_exact_int64')


def outer_operation(native):
    """The operands of ``native`` when it is one arithmetic operation (or a negation), else None."""
    if isinstance(native, cst.UnaryOperation) and isinstance(native.operator, cst.Minus):
        return (native.expression,)
    if isinstance(native, cst.BinaryOperation) and isinstance(native.operator, _ARITHMETIC):
        return (native.left, native.right)
    return None


def _walk_names(node):
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, cst.Name):
            yield item
        else:
            stack.extend(item.children)


class GuardIntegers(cst.CSTTransformer):
    def __init__(self, arithmetic, integer_locals, path=None):
        self.a = arithmetic
        self.path = path
        self.integer_locals = integer_locals
        self.guarded = 0
        self._opaque = 0
        self._handled = set()
        self._skip = set()
        self._native_args = set()

    # scopes that bind names of their own are left alone
    def on_visit(self, node):
        if isinstance(node, OPAQUE_SCOPES) and self._opaque_root is not node:
            self._opaque += 1
        return super().on_visit(node)

    def on_leave(self, before, after):
        if isinstance(before, OPAQUE_SCOPES) and self._opaque_root is not before:
            self._opaque -= 1
        return super().on_leave(before, after)

    _opaque_root = None

    def guard(self, original, updated, render, slow_type=None):
        """``render(expr)`` -> code of the context around ``expr``; the guarded IfExp."""
        names = [name for name in self.a.names_in(original)]
        bound = _bound(original)
        if bound is None or not names:
            return None
        self.guarded += 1
        native = render(_code(updated))
        boxed = render(_code(_boxed(updated)))
        if slow_type is not None:
            if slow_type == 'longlong' and self.path is not None:
                # D96: the exact value crosses into long long through the named conversion,
                # so a value beyond 64 bits raises OverflowError naming the cell
                boxed = f'{MX_SYS_MOD}._mx_exact_int64({boxed}, {self.path!r})'
            else:
                boxed = f'{CY_MOD}.cast({CY_MOD}.{slow_type}, {boxed})'
        return _parenthesized(cst.parse_expression(f'({native} if {_condition(names, bound)} else {boxed})'))

    def _expression_parts(self, node):
        return [part for part in (getattr(node, 'left', None), getattr(node, 'right', None)) if part is not None]

    # float(expr) and int(expr)
    def leave_Call(self, before, after):
        if self._opaque:
            return after
        if (isinstance(before.func, cst.Name) and before.func.value in ('float', 'range')
                and before.args and not any(arg.keyword or arg.star for arg in before.args)):
            if before.func.value == 'float' and len(before.args) == 1 and self.a.is_expression(before.args[0].value):
                self._handled.add(id(before.args[0].value))
                guarded = self.guard(before.args[0].value, after.args[0].value, lambda code: f'float({code})', 'double')
                return guarded if guarded is not None else after
            if before.func.value == 'range':
                args = list(after.args)
                for index, arg in enumerate(before.args):
                    if self.a.is_expression(arg.value):
                        self._handled.add(id(arg.value))
                        guarded = self.guard(arg.value, after.args[index].value, lambda code: code, 'longlong')
                        if guarded is not None:
                            args[index] = after.args[index].with_changes(value=guarded)
                return after.with_changes(args=args)
        return after

    def leave_BinaryOperation(self, before, after):
        if self._opaque:
            return after
        if isinstance(before.operator, cst.Divide):
            sides = [side for side in (before.left, before.right) if self.a.is_expression(side)]
            if sides and all(self.a.is_operand(side) or self._float_side(side) for side in (before.left, before.right)):
                for side in sides:
                    self._handled.add(id(side))
                guarded = self.guard(before, after, lambda code: code, 'double')
                return guarded if guarded is not None else after
            return after
        return after

    def _float_side(self, node):
        return isinstance(node, cst.Float) or (isinstance(node, cst.Name) and node.value not in self.a.names
                                                 and self.a.kind.kind(node) == 'float')

    def leave_Comparison(self, before, after):
        if self._opaque or len(before.comparisons) != 1:
            return after
        from .checked_arithmetic import divisibility_operands
        if divisibility_operands(before) is not None:
            return after        # `t % n == 0`: the divisibility pass emits it exactly
        comparison, = before.comparisons
        if not isinstance(comparison.operator, _COMPARISONS):
            return after
        sides = (before.left, comparison.comparator)
        expressions = [side for side in sides if self.a.is_expression(side)]
        if not expressions or not all(self.a.is_operand(side) or self._numeric(side) for side in sides):
            return after
        for side in expressions:
            self._handled.add(id(side))
        guarded = self.guard_comparison(before, after, expressions)
        return guarded if guarded is not None else after

    def _numeric(self, node):
        return isinstance(node, (cst.Float, cst.Integer)) or self.a.kind.kind(node) in ('float', 'int')

    def guard_comparison(self, before, after, expressions):
        names = sorted({name for side in expressions for name in self.a.names_in(side)})
        bounds = [_bound(side) for side in expressions]
        if not names or None in bounds:
            return None
        bound = min(bounds)
        self.guarded += 1

        class BoxNames(cst.CSTTransformer):
            def leave_Name(inner, b, a):
                return cst.parse_expression(f'{CY_MOD}.cast(object, {a.value})') if a.value in names else a
        boxed = _code(after.visit(BoxNames()))
        # both branches bint, so the comparison stays a C value (the slow one is an exact bool)
        return _parenthesized(cst.parse_expression(
            f'({_code(after)} if {_condition(names, bound)} else {CY_MOD}.cast({CY_MOD}.bint, {boxed}))'))

    def leave_Assign(self, before, after):
        if self._opaque or len(before.targets) != 1 or not isinstance(before.targets[0].target, cst.Name):
            return after
        slow = 'longlong' if before.targets[0].target.value in self.integer_locals else None
        if isinstance(before.value, cst.IfExp) and isinstance(after.value, cst.IfExp):
            # ``off = (k - 1) * step if adv else k * step``: each branch is the local's value
            value = after.value
            for field in ('body', 'orelse'):
                branch, updated = getattr(before.value, field), getattr(value, field)
                if self.a.is_expression(branch):
                    guarded = self.guard(branch, updated, lambda code: code, slow)
                    if guarded is not None:
                        value = value.with_changes(**{field: guarded})
            return after.with_changes(value=value)
        if not self.a.is_expression(before.value):
            return after
        self._handled.add(id(before.value))
        guarded = self.guard(before.value, after.value, lambda code: code, slow)
        return after.with_changes(value=guarded) if guarded is not None else after

    def _fallback(self, before, after):
        if self._opaque or not self.a.is_expression(before):
            return after
        guarded = self.guard(before, after, lambda code: code)
        return guarded if guarded is not None else after

    def visit_Call(self, node):
        # float()/range() arguments are guarded with their call (leave_Call)
        if isinstance(node.func, cst.Name) and node.func.value in ('float', 'range'):
            self._skip.update(id(arg.value) for arg in node.args)
        elif _native_callee(node.func):
            self._native_args.update(id(arg.value) for arg in node.args)
        return True

    def leave_Arg(self, before, after):
        if id(before.value) in self._skip or self._opaque or not self.a.is_expression(before.value):
            return after
        # a compiled Cell, a provider/runtime function or a native min/max takes a C integer:
        # the exact value as long long (the call itself would refuse a wider one); any other
        # callee gets the Python value
        slow = 'longlong' if id(before.value) in self._native_args else None
        guarded = self.guard(before.value, after.value, lambda code: code, slow)
        return after.with_changes(value=guarded) if guarded is not None else after

    def leave_Return(self, before, after):
        if before.value is None:
            return after
        return after.with_changes(value=self._fallback(before.value, after.value))

    def leave_Element(self, before, after):
        return after.with_changes(value=self._fallback(before.value, after.value))

    def visit_Subscript(self, node):
        # `v.data[i + 1]`: a typed vector's raw buffer, only emitted (W1.6) inside a loop whose
        # index range the vector lowering has proven, so the index cannot leave it
        if isinstance(node.value, cst.Attribute) and node.value.attr.value == 'data':
            self._skip.update(id(element.slice.value) for element in node.slice
                              if isinstance(element.slice, cst.Index))
        return True

    def leave_Index(self, before, after):
        # a subscript index: the exact value as long long (an index beyond it fails either way)
        if self._opaque or id(before.value) in self._skip or not self.a.is_expression(before.value):
            return after
        guarded = self.guard(before.value, after.value, lambda code: code, 'longlong')
        return after.with_changes(value=guarded) if guarded is not None else after

    def leave_AugAssign(self, before, after):
        if (self._opaque or not isinstance(before.target, cst.Name)
                or before.target.value not in self.integer_locals or type(before.operator) not in _AUGMENTED):
            return after
        expression = cst.BinaryOperation(before.target, _AUGMENTED[type(before.operator)](),
                                         before.value.with_changes(lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),)))
        if not self.a.is_expression(expression):
            return after
        updated = cst.BinaryOperation(after.target, _AUGMENTED[type(before.operator)](),
                                      after.value.with_changes(lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),)))
        guarded = self.guard(expression, updated, lambda code: code, 'longlong')
        if guarded is None:
            return after
        return cst.Assign(targets=(cst.AssignTarget(after.target),), value=guarded)


def guard_integers(node, local_kinds, param_kinds, path=None):
    """Rewrite one formula FunctionDef; returns (node, number of guarded expressions)."""
    if not isinstance(getattr(node, 'body', None), cst.IndentedBlock):
        return node, 0
    names = {name for name, kind in {**local_kinds, **param_kinds}.items() if kind == 'int'}
    if not names:
        return node, 0
    integer_locals = {name for name, kind in local_kinds.items() if kind == 'int'}

    class Kind:
        def kind(self, expr):
            if isinstance(expr, cst.Name):
                return {**local_kinds, **param_kinds}.get(expr.value, 'unknown')
            if isinstance(expr, cst.Float):
                return 'float'
            if isinstance(expr, cst.Integer):
                return 'int'
            return 'unknown'

    guard = GuardIntegers(_Arithmetic(Kind(), names), integer_locals, path)
    guard._opaque_root = node
    body = node.body.visit(guard)
    return node.with_changes(body=body), guard.guarded
