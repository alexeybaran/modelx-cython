"""Integer arithmetic that cannot overflow: C where a range proves it, Python ints elsewhere.

A formula's C integers are its typed parameters, the C integer results of other Cells and
References, typed fields and memoryview elements, and the counters of ``range`` loops.  Python
ints have no limit and C integers wrap (or trap), so every integer operation of two C integers is
checked here, once, at build time, against the ranges of its operands:

* a parameter of a dense cache ranges over its declared or sampled domain, which every entry
  checks before the formula runs; any other C integer ranges over its whole type;
* a literal is itself; a ``range`` counter ranges from its start to its stop;
* ``+``, ``-``, ``*``, unary ``-`` and ``abs`` keep C arithmetic when the ranges prove the result
  fits its C type; ``//`` when the operands cannot be the one overflowing pair (the type's
  minimum and -1); ``%`` always, as its result always fits (through the runtime's exact remainder
  where C's would trap on that pair); ``&``, ``|``, ``^``, ``~`` and comparisons always;
* ``/`` keeps C doubles when both operands are exact doubles (magnitude at most 2**53);
* ``**``, ``<<`` and ``>>`` of two C integers, and every operation not proved above, run on
  Python ints: the left operand is cast to ``object``, as upstream computes an untyped local.

So nothing depends on what the sample reached: a C operation is one that cannot overflow for any
value its operands can hold.  An operand whose type the pass cannot tell (an attribute of a
Space it does not see) counts as a C integer of unknown value.  Locals are left to Cython's
default ("safe") inference, which never gives a C integer type to a local used in arithmetic;
the pass declares a ``range`` counter ``long long`` itself when its range is proved and it is
read only inside its loop, and declares ``object`` a loop variable read after its loop, so a
loop that never ran leaves it unbound, as in Python.
"""

import re
import pathlib

import libcst as cst
from libcst.metadata import ExpressionContext

from .consts import CY_MOD, MX_SYS_MOD
from .scopes import CstFacts, has_scope_effects, lexical_nodes
from .syntax import dotted_name, local_assignment, walk


class CInt:
    """A C integer of C type width ``width`` (32 or 64 bits) holding a value in [lo, hi]."""
    __slots__ = ('lo', 'hi', 'width')

    def __init__(self, lo, hi, width=64):
        self.lo, self.hi, self.width = lo, hi, width

    def fits(self, lo, hi, width=None):
        width = max(self.width, width or self.width)
        return -(1 << (width - 1)) <= lo and hi < (1 << (width - 1))

    def __repr__(self):
        return f'CInt({self.lo}, {self.hi}, {self.width})'


FLOAT = 'float'         # a C double
OBJECT = 'object'       # a Python object, or a value that is never a C integer
MAYBE = 'maybe'         # unknown: possibly a C integer of any value
INT64 = CInt(-(1 << 63), (1 << 63) - 1)
BINT = CInt(0, 1, 32)
_EXACT_DOUBLE = 1 << 53


def _full(width):
    return CInt(-(1 << (width - 1)), (1 << (width - 1)) - 1, width)


class View:
    """A memoryview or C array of ``element`` with ``ndim`` axes."""
    __slots__ = ('element', 'ndim')

    def __init__(self, element, ndim):
        self.element, self.ndim = element, ndim


class Instance:
    """An extension type of this module (or ``None``: of a module the pass does not see)."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name


_SCALARS = {
    f'{CY_MOD}.longlong': INT64, f'{CY_MOD}.Py_ssize_t': INT64, f'{CY_MOD}.ssize_t': INT64,
    f'{CY_MOD}.Py_hash_t': INT64, f'{CY_MOD}.int': _full(32), f'{CY_MOD}.bint': BINT,
    f'{CY_MOD}.double': FLOAT, f'{CY_MOD}.float': FLOAT, 'float': FLOAT,
    'object': OBJECT, 'int': OBJECT, 'str': OBJECT, 'list': OBJECT, 'dict': OBJECT, 'tuple': OBJECT,
}


def annotation_type(expr, classes=()):
    """The value type an annotation declares (see the module docstring)."""
    name = dotted_name(expr) or ''
    if isinstance(expr, cst.SimpleString):
        return MAYBE            # a string annotation: whatever Cython makes of it
    if name in _SCALARS:
        return _SCALARS[name]
    if name in classes:
        return Instance(name)
    if isinstance(expr, cst.Subscript):
        head = dotted_name(expr.value)
        indexes = [item.slice for item in expr.slice]
        if head in (f'{CY_MOD}.const', '_mx_private_receiver', 'Optional') and len(indexes) == 1 \
                and isinstance(indexes[0], cst.Index):
            return annotation_type(indexes[0].value, classes)
        element = annotation_type(expr.value, classes)
        if isinstance(element, CInt) or element == FLOAT:
            return View(element, len(indexes))
        return OBJECT
    if name.startswith(f'{CY_MOD}.'):
        return OBJECT if name in (f'{CY_MOD}.p_void',) else MAYBE
    if '.' in name:
        return Instance(None)       # an extension type of another module
    return OBJECT


_PXD_FUNCTION = re.compile(r'^cdef inline ([\w ]+?) (\w+)\(', re.M)
_PXD_EXTERN = re.compile(r'^    (long long|bint|double|int) (\w+) "', re.M)
_C_NAMES = {'long long': INT64, 'bint': BINT, 'double': FLOAT, 'int': _full(32)}


def runtime_results(text=None):
    """{runtime helper: result type} from ``_mx_sys.pxd``."""
    if text is None:
        text = (pathlib.Path(__file__).parent / f'{MX_SYS_MOD}.pxd').read_text(encoding='utf-8')
    results = {}
    for pattern in (_PXD_FUNCTION, _PXD_EXTERN):
        for kind, name in pattern.findall(text):
            results[name] = _C_NAMES.get(kind, OBJECT)
    return results


_BASE_FIELDS = {'_mx_is_cells_set': BINT}


class ModuleTypes:
    """The C types of one generated module: fields, typed methods and functions."""

    def __init__(self, syntax):
        self.classes = {node.name.value for node in syntax.body if isinstance(node, cst.ClassDef)}
        self.fields, self.methods, self.functions = {}, {}, {}
        typed = (f'{CY_MOD}.cfunc', f'{CY_MOD}.ccall')
        for node in syntax.body:
            if isinstance(node, cst.FunctionDef):
                if node.returns is not None and any(dotted_name(d.decorator) in typed for d in node.decorators):
                    self.functions[node.name.value] = annotation_type(node.returns.annotation, self.classes)
                continue
            if not isinstance(node, cst.ClassDef) or not isinstance(node.body, cst.IndentedBlock):
                continue
            fields = self.fields[node.name.value] = dict(_BASE_FIELDS)
            methods = self.methods[node.name.value] = {}
            for line in node.body.body:
                if isinstance(line, cst.SimpleStatementLine):
                    for small in line.body:
                        if isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name):
                            fields[small.target.value] = annotation_type(small.annotation.annotation, self.classes)
                elif (isinstance(line, cst.FunctionDef) and line.returns is not None
                      and any(dotted_name(d.decorator) in typed for d in line.decorators)):
                    methods[line.name.value] = annotation_type(line.returns.annotation, self.classes)
        self.runtime = runtime_results()


def _numeric(kind):
    return isinstance(kind, CInt) or kind in (FLOAT, MAYBE)


def _as_int(kind):
    return INT64 if kind == MAYBE else kind


def _object(expr):
    """``expr`` evaluated as a Python object (an int for a C integer)."""
    return cst.Call(cst.Attribute(cst.Name(CY_MOD), cst.Name('cast')),
                    (cst.Arg(cst.Name('object')), cst.Arg(expr.with_changes(lpar=(), rpar=()))),
                    lpar=expr.lpar, rpar=expr.rpar)


_REMAINDER = '_mx_signed_remainder'


def _division_traps(left, right, width):
    """Whether C's ``//`` or ``%`` of these ranges can meet the type's minimum divided by -1."""
    return left.lo <= -(1 << (width - 1)) and right.lo <= -1 <= right.hi


def _exact_remainder(operator, left, right):
    """A ``%`` of two known 64-bit C integers, computed by the runtime's exact remainder."""
    return (isinstance(operator, cst.Modulo) and isinstance(left, CInt) and isinstance(right, CInt)
            and max(left.width, right.width) == 64)


def _unsafe_negation(kind):
    """Whether ``-x`` or ``abs(x)`` of a value of ``kind`` could overflow in C (x the type's minimum)."""
    if kind == MAYBE:
        return True
    return isinstance(kind, CInt) and kind.lo <= -(1 << (kind.width - 1))


def _double(expr):
    return cst.Call(cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name('_mx_int_as_double')),
                    (cst.Arg(expr.with_changes(lpar=(), rpar=())),))


def _span(kinds):
    """Cython's spanning type of the branches of a conditional, ``min``/``max`` or ``and``/``or``."""
    if any(kind == OBJECT or not _numeric(kind) for kind in kinds):
        return OBJECT
    if any(kind == FLOAT for kind in kinds):
        return FLOAT
    if any(kind == MAYBE for kind in kinds):
        return MAYBE
    return CInt(min(k.lo for k in kinds), max(k.hi for k in kinds), max(k.width for k in kinds))


def _corners(op, a, b):
    values = [op(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
    return min(values), max(values)


def _floordiv_range(a, b):
    if b.lo > 0 or b.hi < 0:
        return _corners(lambda x, y: x // y, a, b)
    bound = max(abs(a.lo), abs(a.hi))
    return -bound, bound


def _remainder_range(b):
    if b.lo > 0:
        return 0, b.hi - 1
    if b.hi < 0:
        return b.lo + 1, 0
    return min(b.lo + 1, 0), max(b.hi - 1, 0)


def _bits_range(op, a, b, width):
    if a.lo >= 0 and b.lo >= 0:
        top = (1 << max(a.hi.bit_length(), b.hi.bit_length())) - 1
        return (0, min(a.hi, b.hi)) if isinstance(op, cst.BitAnd) else (0, top)
    full = _full(width)
    return full.lo, full.hi


class _Lower(cst.CSTTransformer):
    """Classify and lower the integer arithmetic of one formula function (bottom-up)."""

    def __init__(self, types, owner, env, path, returns_integer=False):
        self.returns_integer = returns_integer
        self.types = types
        self.owner = owner
        self.env = env          # name -> type of every declared name of the function
        self.path = path
        self.nested = []        # names bound by enclosing lambdas and comprehensions
        self.lowered_ints = {}  # id -> integer operation lowered to Python ints (kept alive)
        self.lowered = 0
        self.kinds = {}

    # -- nested scopes: their own bindings are never the function's C integers
    def _push(self, names):
        self.nested.append(set(names))

    def visit_Lambda(self, node):
        self._push(p.name.value for p in (*node.params.posonly_params, *node.params.params,
                                           *node.params.kwonly_params))

    def leave_Lambda(self, before, after):
        self.nested.pop()
        return after

    def _visit_comprehension(self, node):
        names = []
        part = node.for_in
        while part is not None:
            names.extend(n.value for n in walk(part.target) if isinstance(n, cst.Name))
            part = part.inner_for_in
        self._push(names)

    def _leave_comprehension(self, before, after):
        self.nested.pop()
        return after

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _visit_comprehension
    leave_ListComp = leave_SetComp = leave_DictComp = leave_GeneratorExp = _leave_comprehension

    def visit_FunctionDef(self, node):
        return False            # a nested def is a Python function of its own

    # -- types
    def kind(self, expr):
        # keyed by identity, holding the node: a node the transform dropped cannot lend its id
        known = self.kinds.get(id(expr))
        if known is None or known[0] is not expr:
            known = self.kinds[id(expr)] = (expr, self._kind(expr))
        return known[1]

    def _kind(self, expr):
        if isinstance(expr, cst.Integer):
            value = expr.evaluated_value
            if value >= 1 << 63:
                return OBJECT
            return CInt(value, value, 32 if value < 1 << 31 else 64)
        if isinstance(expr, cst.Float):
            return FLOAT
        if isinstance(expr, cst.Name):
            if any(expr.value in scope for scope in self.nested):
                return OBJECT
            if expr.value in ('True', 'False'):
                return BINT
            return self.env.get(expr.value, OBJECT)
        if isinstance(expr, cst.Attribute):
            base = self.kind(expr.value)
            if isinstance(base, Instance):
                if base.name is None:
                    return MAYBE
                return self.types.fields.get(base.name, {}).get(expr.attr.value, OBJECT)
            return MAYBE if base == MAYBE else OBJECT
        if isinstance(expr, cst.Subscript):
            base = self.kind(expr.value)
            if isinstance(base, View):
                if len(expr.slice) == base.ndim and all(isinstance(s.slice, cst.Index) for s in expr.slice):
                    return base.element
                return OBJECT
            return MAYBE if base in (MAYBE,) or isinstance(base, Instance) and base.name is None else OBJECT
        if isinstance(expr, cst.Call):
            return self._call_kind(expr)
        # the kinds of operations are those they have once lowered: an operation the ranges do not
        # prove is a Python one
        if isinstance(expr, cst.BinaryOperation):
            left, right = self.kind(expr.left), self.kind(expr.right)
            if self._integer_pair(left, right) and not self._safe(expr.operator, left, right):
                return CInt(*_remainder_range(right), 64) if _exact_remainder(expr.operator, left, right) else OBJECT
            return self._binary_kind(expr.operator, left, right)
        if isinstance(expr, cst.UnaryOperation):
            operand = self.kind(expr.expression)
            if isinstance(expr.operator, cst.Not):
                return BINT
            if isinstance(expr.operator, cst.Minus) and _unsafe_negation(operand):
                return OBJECT
            if isinstance(operand, CInt):
                if isinstance(expr.operator, cst.Minus):
                    return CInt(-operand.hi, -operand.lo, operand.width)
                if isinstance(expr.operator, cst.BitInvert):
                    return CInt(-operand.hi - 1, -operand.lo - 1, operand.width)
            return operand
        if isinstance(expr, cst.IfExp):
            return _span((self.kind(expr.body), self.kind(expr.orelse)))
        if isinstance(expr, cst.BooleanOperation):
            return _span((self.kind(expr.left), self.kind(expr.right)))
        if isinstance(expr, cst.Comparison):
            # a C truth value for C operands, ``in`` and ``is``; a rich comparison of an object is
            # whatever its type returns (an array, a Series)
            operands = (expr.left, *(item.comparator for item in expr.comparisons))
            if all(isinstance(item.operator, (cst.In, cst.NotIn, cst.Is, cst.IsNot)) for item in expr.comparisons) \
                    or all(isinstance(kind, CInt) or kind == FLOAT for kind in map(self.kind, operands)):
                return BINT
            return OBJECT
        return OBJECT

    def _call_kind(self, call):
        func = call.func
        name = dotted_name(func)
        if name == f'{CY_MOD}.cast' and call.args:
            return annotation_type(call.args[0].value, self.types.classes)
        if isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name) and func.value.value == MX_SYS_MOD:
            return self.types.runtime.get(func.attr.value, OBJECT)
        if isinstance(func, cst.Attribute):
            base = self.kind(func.value)
            if isinstance(base, Instance):
                if base.name is None:
                    return MAYBE
                return self.types.methods.get(base.name, {}).get(func.attr.value, OBJECT)
            if base == MAYBE:
                return MAYBE
            if isinstance(func.value, cst.Name) and func.value.value not in self.env and func.value.value != CY_MOD:
                # a function of a cimported module (a native provider) may return a C integer
                return MAYBE if func.value.value.startswith('_mx_') else OBJECT
            return OBJECT
        if isinstance(func, cst.Name) and func.value not in self.env and not any(
                func.value in scope for scope in self.nested):
            if func.value in self.types.functions:
                return self.types.functions[func.value]
            args = [arg.value for arg in call.args]
            if func.value == 'len':
                return CInt(0, INT64.hi)
            if func.value == 'ord':
                return CInt(0, 0x10FFFF, 32)
            if func.value == 'hash':
                return INT64
            if func.value == 'float':
                return FLOAT
            if func.value in ('min', 'max') and len(args) > 1 and not any(arg.star for arg in call.args):
                return _span([self.kind(arg) for arg in args])
            if func.value == 'abs' and len(args) == 1:
                operand = self.kind(args[0])
                if _unsafe_negation(operand):
                    return OBJECT
                if isinstance(operand, CInt):
                    return CInt(0 if operand.lo <= 0 <= operand.hi else min(abs(operand.lo), abs(operand.hi)),
                                max(abs(operand.lo), abs(operand.hi)), operand.width)
                return operand
        return OBJECT

    @staticmethod
    def _binary_kind(operator, left, right):
        if not (_numeric(left) and _numeric(right)):
            return OBJECT
        if isinstance(operator, cst.Divide) or FLOAT in (left, right):
            return FLOAT if not isinstance(operator, cst.MatrixMultiply) else OBJECT
        if MAYBE in (left, right):
            return MAYBE
        width = max(left.width, right.width)
        if isinstance(operator, cst.Add):
            return CInt(left.lo + right.lo, left.hi + right.hi, width)
        if isinstance(operator, cst.Subtract):
            return CInt(left.lo - right.hi, left.hi - right.lo, width)
        if isinstance(operator, cst.Multiply):
            return CInt(*_corners(lambda x, y: x * y, left, right), width)
        if isinstance(operator, cst.FloorDivide):
            return CInt(*_floordiv_range(left, right), width)
        if isinstance(operator, cst.Modulo):
            return CInt(*_remainder_range(right), width)
        if isinstance(operator, (cst.BitAnd, cst.BitOr, cst.BitXor)):
            return CInt(*_bits_range(operator, left, right, width), width)
        return OBJECT

    # -- lowering
    def _safe(self, operator, left, right):
        """Whether C arithmetic of ``left operator right`` (two C integer kinds) cannot overflow."""
        left, right = _as_int(left), _as_int(right)
        width = max(left.width, right.width)
        if isinstance(operator, (cst.Add, cst.Subtract, cst.Multiply)):
            result = self._binary_kind(operator, left, right)
            return result.fits(result.lo, result.hi, width)
        if isinstance(operator, (cst.FloorDivide, cst.Modulo)):
            return not _division_traps(left, right, width)
        if isinstance(operator, cst.Divide):
            return all(-_EXACT_DOUBLE <= k.lo and k.hi <= _EXACT_DOUBLE for k in (left, right))
        return isinstance(operator, (cst.BitAnd, cst.BitOr, cst.BitXor))

    def _integer_pair(self, left, right):
        """Both operands may be C integers (so Cython would compute in C integer arithmetic)."""
        return all(isinstance(kind, CInt) or kind == MAYBE for kind in (left, right))

    def leave_BinaryOperation(self, before, after):
        left, right = self.kind(after.left), self.kind(after.right)
        if FLOAT in (left, right) and not isinstance(after.operator, (cst.Power, cst.MatrixMultiply)):
            # a Python int beside a double: Python converts it (correctly rounded, OverflowError
            # beyond a double) before the float operation, and so does the cast
            for side in ('left', 'right'):
                operand = getattr(after, side)
                if id(operand) in self.lowered_ints:
                    after = after.with_changes(**{side: _double(operand)})
            return after
        if not self._integer_pair(left, right) or self._safe(after.operator, left, right):
            return after
        if _exact_remainder(after.operator, left, right):
            # a remainder always fits: only C's LLONG_MIN % -1 traps, which the helper returns as 0
            self.lowered += 1
            helper = cst.Call(cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(_REMAINDER)),
                              (cst.Arg(after.left.with_changes(lpar=(), rpar=())),
                               cst.Arg(after.right.with_changes(lpar=(), rpar=()))),
                              lpar=after.lpar, rpar=after.rpar)
            self.kinds[id(helper)] = (helper, CInt(*_remainder_range(right), 64))
            return helper
        self.lowered += 1
        lowered = after.with_changes(left=_object(after.left))
        self.kinds[id(lowered)] = (lowered, OBJECT)
        if isinstance(after.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo)):
            self.lowered_ints[id(lowered)] = lowered
        return lowered

    def leave_UnaryOperation(self, before, after):
        if not isinstance(after.operator, cst.Minus) or not _unsafe_negation(self.kind(after.expression)):
            return after
        self.lowered += 1
        lowered = after.with_changes(expression=_object(after.expression))
        self.kinds[id(lowered)] = (lowered, OBJECT)
        return lowered

    def leave_Call(self, before, after):
        if (dotted_name(after.func) == _INTEGER_RESULT and len(after.args) == 2
                and isinstance(self.kind(after.args[0].value), CInt)):
            # a checked conversion of a value already a C integer: the value itself
            value = after.args[0].value
            return value.with_changes(lpar=(*after.lpar, *value.lpar), rpar=(*value.rpar, *after.rpar))
        if not (isinstance(after.func, cst.Name) and after.func.value == 'abs' and len(after.args) == 1
                and after.func.value not in self.env and not any('abs' in scope for scope in self.nested)):
            return after
        if not _unsafe_negation(self.kind(after.args[0].value)):
            return after
        self.lowered += 1
        lowered = after.with_changes(args=(after.args[0].with_changes(value=_object(after.args[0].value)),))
        self.kinds[id(lowered)] = (lowered, OBJECT)
        return lowered

    def leave_AugAssign(self, before, after):
        if not isinstance(after.target, cst.Name):
            return after
        target = self.kind(after.target)
        operator = _AUGMENTED.get(type(after.operator))
        value = self.kind(after.value)
        if (operator is None or not isinstance(target, CInt) or not self._integer_pair(target, value)
                or self._safe(operator(), target, value)):
            return after
        # a C integer local or parameter: the Python result is stored back through the checked
        # conversion, which raises an OverflowError naming the Cell where it does not fit
        self.lowered += 1
        value = cst.BinaryOperation(_object(after.target), operator(), after.value)
        return cst.Assign((cst.AssignTarget(after.target),), _integer_result(value, self.path))

    def leave_Return(self, before, after):
        if not self.returns_integer or after.value is None:
            return after
        return after.with_changes(value=self._checked_result(after.value))

    def _checked_result(self, value, branch=False):
        """A C integer result: a value already one as it is, any other through the checked
        conversion, which raises naming the Cell for a value beyond 64 bits or not an integer. A
        conditional keeps its evaluation order; a float branch of one is left to Cython, which
        refuses to compile a double returned as a C integer."""
        if isinstance(value, cst.IfExp):
            return value.with_changes(body=self._checked_result(value.body, True),
                                      orelse=self._checked_result(value.orelse, True))
        kind = self.kind(value)
        if isinstance(kind, CInt) or branch and kind == FLOAT:
            return value
        self.lowered += 1
        return _integer_result(value, self.path)


_INTEGER_RESULT = f'{MX_SYS_MOD}._mx_integer_result'


def _integer_result(value, path):
    return cst.Call(cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name('_mx_integer_result')),
                    (cst.Arg(value.with_changes(lpar=(), rpar=())), cst.Arg(cst.SimpleString(repr(path)))),
                    lpar=value.lpar, rpar=value.rpar)


_AUGMENTED = {cst.AddAssign: cst.Add, cst.SubtractAssign: cst.Subtract, cst.MultiplyAssign: cst.Multiply,
              cst.FloorDivideAssign: cst.FloorDivide, cst.ModuloAssign: cst.Modulo,
              cst.PowerAssign: cst.Power, cst.LeftShiftAssign: cst.LeftShift,
              cst.RightShiftAssign: cst.RightShift, cst.DivideAssign: cst.Divide,
              cst.BitAndAssign: cst.BitAnd, cst.BitOrAssign: cst.BitOr, cst.BitXorAssign: cst.BitXor}


def _parameters(node):
    return (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)


def _declared(node, types, owner):
    """{name: type} of the parameters and annotated locals of a function."""
    env = {}
    for index, param in enumerate(_parameters(node)):
        if param.annotation is not None:
            env[param.name.value] = annotation_type(param.annotation.annotation, types.classes)
        elif index == 0 and owner is not None:
            env[param.name.value] = Instance(owner)
        else:
            env[param.name.value] = OBJECT
    if isinstance(node.body, cst.IndentedBlock):
        for part in lexical_nodes(node.body.body):
            if isinstance(part, cst.AnnAssign) and isinstance(part.target, cst.Name):
                env[part.target.value] = annotation_type(part.annotation.annotation, types.classes)
    return env


def _range_arguments(iterable):
    if (isinstance(iterable, cst.Call) and dotted_name(iterable.func) == 'range'
            and 1 <= len(iterable.args) <= 3 and not any(arg.star or arg.keyword for arg in iterable.args)):
        return [arg.value for arg in iterable.args]
    return None


def _counter(lower, args):
    """The range of a ``range`` counter, or None unless its step is 1 or -1 (literal) and its bounds
    are integers: C integers, or Python ones, which Cython converts to the counter's type once, as the
    loop starts (raising where one does not fit, where Python's range would run). The counter only
    takes values between its bounds, so the loop's own step never overflows."""
    if not all(_integer_valued(lower, arg) for arg in args):
        return None
    kinds = [kind if isinstance(kind := lower.kind(arg), CInt) else INT64 for arg in args]
    if len(args) == 1:
        return CInt(0, max(kinds[0].hi - 1, 0))
    start, stop = kinds[:2]
    step = args[2] if len(args) == 3 else cst.Integer('1')
    sign = 1
    if isinstance(step, cst.UnaryOperation) and isinstance(step.operator, cst.Minus):
        sign, step = -1, step.expression
    if not isinstance(step, cst.Integer) or step.evaluated_value != 1:
        return None
    if sign > 0:
        return CInt(start.lo, max(stop.hi - 1, start.lo))
    return CInt(min(stop.lo + 1, start.hi), start.hi)


_INTEGER_RESULTS = (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo, cst.LeftShift, cst.RightShift)


def _integer_valued(lower, expr):
    """A C integer, or an integer operation of C integers (a Python int once lowered)."""
    if isinstance(lower.kind(expr), CInt):
        return True
    return (isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, _INTEGER_RESULTS)
            and _integer_valued(lower, expr.left) and _integer_valued(lower, expr.right))


class _Unknown(Exception):
    """A binding construct the definite-assignment analysis does not follow."""


_UNREACHABLE = 'unreachable'


def _join(*states):
    reachable = [state for state in states if state != _UNREACHABLE]
    return all(reachable) if reachable else _UNREACHABLE


def _binds(target, name):
    """Whether an assignment target binds ``name`` (a name, or one inside a tuple or list target)."""
    if isinstance(target, cst.Name):
        return target.value == name
    if isinstance(target, (cst.Tuple, cst.List)):
        return any(_binds(element.value, name) for element in target.elements)
    return False


def maybe_unbound(body, name, facts):
    """Whether a read of the local ``name`` can come before any assignment to it, on some path of the
    function ``body`` (or the body binds it in a way not followed here). Python raises then; a C local
    has no unbound state, so such a local must stay a Python object."""
    reads = {node for node, context in facts.contexts.items()
             if isinstance(node, cst.Name) and node.value == name and context is ExpressionContext.LOAD}
    for part in walk(cst.Module(list(body))):
        if isinstance(part, (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)):
            comprehension = part.for_in
            while comprehension is not None:
                if _binds(comprehension.target, name):
                    reads.difference_update(walk(part))     # the comprehension's own variable
                comprehension = comprehension.inner_for_in

    class Found(Exception):
        pass

    def read(node, state):
        if state is False and any(part in reads for part in walk(node)):
            raise Found
        if any(isinstance(part, cst.NamedExpr) and _binds(part.target, name) for part in walk(node)):
            raise _Unknown

    def block(lines, state):
        for line in lines:
            if state == _UNREACHABLE:
                break
            state = statement(line, state)
        return state

    def suite(node, state):
        if isinstance(node, cst.SimpleStatementSuite):
            for small in node.body:
                if state == _UNREACHABLE:
                    break
                state = small_statement(small, state)
            return state
        return block(node.body, state)

    def statement(line, state):
        if isinstance(line, cst.SimpleStatementLine):
            for small in line.body:
                state = small_statement(small, state)
            return state
        if isinstance(line, cst.If):
            read(line.test, state)
            taken = suite(line.body, state)
            if isinstance(line.orelse, cst.If):
                other = statement(line.orelse, state)
            elif line.orelse is not None:
                other = suite(line.orelse.body, state)
            else:
                other = state
            return _join(taken, other)
        if isinstance(line, (cst.For, cst.While)):
            if isinstance(line, cst.For):
                read(line.iter, state)
                inside = True if _binds(line.target, name) else state
                if not _binds(line.target, name):
                    read(line.target, state)
            else:
                read(line.test, state)
                inside = state
            suite(line.body, inside)
            # the body may run no time; the else clause follows it
            return _join(state, suite(line.orelse.body, state)) if line.orelse is not None else state
        if isinstance(line, cst.Try):
            body_end = suite(line.body, state)
            ends = [suite(line.orelse.body, body_end) if line.orelse is not None else body_end]
            for handler in line.handlers:
                if handler.name is not None and _binds(handler.name.name, name):
                    raise _Unknown
                if handler.type is not None:
                    read(handler.type, state)
                ends.append(suite(handler.body, state))
            end = _join(*ends)
            if line.finalbody is not None:
                suite(line.finalbody.body, state)
                end = suite(line.finalbody.body, end) if end != _UNREACHABLE else end
            return end
        if isinstance(line, cst.With):
            for item in line.items:
                read(item.item, state)
                if item.asname is not None and _binds(item.asname.name, name):
                    state = True if state != _UNREACHABLE else state
            return suite(line.body, state)
        if isinstance(line, (cst.FunctionDef, cst.ClassDef)):
            read(line, state)
            return True if line.name.value == name else state
        raise _Unknown

    def small_statement(small, state):
        if isinstance(small, cst.Assign):
            read(small.value, state)
            for target in small.targets:
                if _binds(target.target, name):
                    state = True
                else:
                    read(target.target, state)
            return state
        if isinstance(small, cst.AnnAssign):
            if small.value is None:
                return state
            read(small.value, state)
            return True if _binds(small.target, name) else (read(small.target, state) or state)
        if isinstance(small, cst.AugAssign):
            read(small.target, state)
            read(small.value, state)
            return state
        if isinstance(small, (cst.Return, cst.Raise)):
            read(small, state)
            return _UNREACHABLE
        if isinstance(small, (cst.Break, cst.Continue)):
            return _UNREACHABLE
        if isinstance(small, (cst.Expr, cst.Assert, cst.Pass)):
            read(small, state)
            return state
        if isinstance(small, cst.Del) and not any(isinstance(part, cst.Name) and part.value == name
                                                  for part in walk(small)):
            return state
        raise _Unknown          # del, global, nonlocal, import: not followed

    try:
        block(body, False)
    except (Found, _Unknown):
        return True
    return False


_DYNAMIC = frozenset(('eval', 'exec', 'locals', 'vars', 'globals', '_getframe', 'currentframe'))


def _integer_locals(body, facts, lower, taken):
    """{local: its range} of the locals that can be C integers: every store a plain assignment of a
    C integer value (as lowered), and none compared by identity (a C integer is boxed anew for ``is``).
    ``taken`` holds the declared names, among them every local that may be read unbound."""
    if has_scope_effects(body) or facts.reads & _DYNAMIC:
        return {}
    identity = set()
    for part in lexical_nodes(body):
        if isinstance(part, cst.Comparison) and any(
                isinstance(item.operator, (cst.Is, cst.IsNot)) for item in part.comparisons):
            identity.update(item.value for item in walk(part) if isinstance(item, cst.Name))
    values = {}
    for part in lexical_nodes(body):
        if (target := local_assignment(part)) is not None:
            values.setdefault(target.value, []).append(part.value)
    pending = {name: tuple(values[name]) for name, count in facts.stores.items()
               if count and len(values.get(name, ())) == count and name not in taken and name not in identity
               and name not in facts.deleted}
    # greatest fixpoint: assume every candidate a C integer of any value, drop those an assignment refutes
    changed = True
    while changed and pending:
        changed = False
        lower.env.update((name, INT64) for name in pending)
        lower.kinds.clear()
        for name, values in sorted(pending.items()):
            if not all(isinstance(lower.kind(value), CInt) for value in values):
                del pending[name]
                del lower.env[name]
                changed = True
    lower.kinds.clear()
    return {name: _span([lower.kind(value) for value in values]) for name, values in pending.items()}


def lower_integer_arithmetic(node, types, owner, path, bounds=None):
    """Lower the integer arithmetic of the formula function ``node`` (see the module docstring).

    ``owner`` is the extension type of ``self`` (None for a module function, whose receiver
    parameter is annotated); ``bounds`` maps a parameter to the (lower, stop) of its checked
    domain. Returns the function with its counters and loop variables declared."""
    if not isinstance(node.body, cst.IndentedBlock):
        return node
    env = _declared(node, types, owner)
    for name, (lower, stop) in (bounds or {}).items():
        kind = env.get(name)
        if isinstance(kind, CInt) and lower < stop:
            env[name] = CInt(max(kind.lo, lower), min(kind.hi, stop - 1), kind.width)
    body = node.body.body
    facts = CstFacts(body)
    declarations = {}
    returns = node.returns and annotation_type(node.returns.annotation, types.classes)
    lower = _Lower(types, owner, env, path, isinstance(returns, CInt) and returns.width == 64)
    for name, count in facts.stores.items():
        if count and name not in env and maybe_unbound(body, name, facts):
            # read before any assignment on some path: Python raises there, and so does an object local
            declarations[name] = 'object'
    for name, kind in _integer_locals(body, facts, lower, set(env) | set(declarations)).items():
        declarations[name] = f'{CY_MOD}.longlong'
        env[name] = kind
    lower.kinds.clear()
    loops = [part for part in lexical_nodes(body) if isinstance(part, cst.For) and isinstance(part.target, cst.Name)]
    for loop in loops:
        name, args = loop.target.value, _range_arguments(loop.iter)
        if name in env or name in declarations or args is None or facts.stores[name] != 1 or name in facts.deleted:
            continue
        counter = _counter(lower, args)
        if counter is not None:
            declarations[name] = f'{CY_MOD}.longlong'
            env[name] = counter
            lower.kinds.clear()
    lower.kinds.clear()
    lowered = node.body.visit(lower)
    if declarations:
        doc_end = 1 if node.get_docstring() is not None else 0
        lines = tuple(cst.SimpleStatementLine((cst.AnnAssign(
            cst.Name(name), cst.Annotation(cst.parse_expression(annotation))),))
            for name, annotation in sorted(declarations.items()))
        lowered = lowered.with_changes(body=(*lowered.body[:doc_end], *lines, *lowered.body[doc_end:]))
    return node.with_changes(body=lowered)
