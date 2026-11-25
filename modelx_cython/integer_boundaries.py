"""Check integer Cell results and ItemSpace keys at their native boundaries."""

import libcst as cst
from libcst.metadata import ExpressionContext
from libcst.metadata.scope_provider import BuiltinAssignment

from .consts import CY_MOD, FORMULA_PREF, MX_SELF, MX_SYS_MOD
from .powers import KIND_FLOAT, OperandKind, kind_of_type_expr
from .scopes import CstFacts, Names, OPAQUE_SCOPES, has_scope_effects, lexical_nodes, maybe_unbound
from .syntax import arguments, dotted_name, local_assignment, statements, walk
from .usage import RESOLVED, _extract_self_chain


_INTEGER_ANNOTATION = cst.Attribute(cst.Name(CY_MOD), cst.Name('longlong'))
_SIGNED_OPERATIONS = {
    # integer guards: the operators that keep a helper (Python semantics, the checked power);
    # + - * are plain C operators, checked by Cython's overflowcheck in the checked mode
    cst.FloorDivide: '_mx_signed_floordiv',
    cst.Modulo: '_mx_signed_remainder',
    cst.LeftShift: '_mx_signed_left_shift',
    cst.RightShift: '_mx_signed_right_shift',
    cst.Power: '_mx_signed_pow',
}
_PLAIN_OPERATIONS = (cst.Add, cst.Subtract, cst.Multiply)
_INTEGER_OPERATIONS = (*_PLAIN_OPERATIONS, *_SIGNED_OPERATIONS)
_PYTHON_SEMANTICS = (cst.LeftShift, cst.RightShift, cst.FloorDivide, cst.Modulo)
_DIVISIONS = {cst.FloorDivide: '_mx_signed_floordiv', cst.Modulo: '_mx_signed_remainder',
              cst.FloorDivideAssign: '_mx_signed_floordiv', cst.ModuloAssign: '_mx_signed_remainder'}
_HELPER_OPERATIONS = (*_PYTHON_SEMANTICS, cst.Power)
_INTEGER_CONVERSION = f'{MX_SYS_MOD}._mx_integer_result'
# the floor division raises Cython's unnamed OverflowError (LLONG_MIN // -1): the handler names it
_FLOORDIV = f'{MX_SYS_MOD}._mx_signed_floordiv'
_SIGNED_HELPERS = frozenset((*_SIGNED_OPERATIONS.values(), '_mx_integer_result', '_mx_exact_int64',
                             '_mx_flag_add', '_mx_flag_sub', '_mx_flag_mul', '_mx_flag_neg'))


_CALL_GATE = """\
@{cy}.cclass
class {gate}:
    # a Cell called from Python with integer arguments: the arguments are checked, then the compiled method runs;
    # built without functools.wraps (its attributes are read from the method on demand)
    method: object
    params: tuple
    path: str

    def __call__(self, *args, **kwargs):
        values = list(args)
        for position, name in self.params:
            if position < len(values):
                values[position] = {runtime}._mx_exact_int64(values[position], self.path)
            if name in kwargs:
                kwargs[name] = {runtime}._mx_exact_int64(kwargs[name], self.path)
        return self.method(*values, **kwargs)

    @property
    def __wrapped__(self):
        return self.method

    def __getattribute__(self, name):
        # introspection reads the method's own attributes (``__doc__`` included), as functools.wraps copied them
        if name == '__doc__':
            return self.method.__doc__
        try:
            return {runtime}._mx_generic_getattr(self, name)
        except AttributeError:
            return getattr(self.method, name)

    def __repr__(self):
        return repr(self.method)


@{cy}.cfunc
def {name}(method: object, params: tuple, path: str) -> object:
    gate: {gate} = {gate}.__new__({gate})
    gate.method, gate.params, gate.path = method, params, path
    return gate
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


# builtins that only convert or compare a value: a wrapped argument does no harm before the test at the return
_PURE_CONVERSIONS = frozenset(('float', 'int', 'bool', 'abs', 'min', 'max', 'round', 'str', 'repr'))
_FLAG_CALLS = frozenset(f'{MX_SYS_MOD}._mx_flag_{name}' for name in ('add', 'sub', 'mul', 'neg'))
_FLAGGED = {cst.Add: '_mx_flag_add', cst.Subtract: '_mx_flag_sub', cst.Multiply: '_mx_flag_mul',
            cst.AddAssign: '_mx_flag_add', cst.SubtractAssign: '_mx_flag_sub', cst.MultiplyAssign: '_mx_flag_mul'}
# + - * and unary minus: what Cython computes as plain C when both operands are C integers
_WRAPPING = (cst.Add, cst.Subtract, cst.Multiply, cst.AddAssign, cst.SubtractAssign, cst.MultiplyAssign)
_KIND_OF_ANNOTATION = {f'{CY_MOD}.longlong': 'int', f'{CY_MOD}.double': 'float', 'float': 'float',
                       'object': 'object', 'str': 'object', 'list': 'object', 'tuple': 'object', 'dict': 'object'}
# calls whose result is never a C integer in Cython (a Python object or a C double)
_NOT_INTEGER_BUILTINS = frozenset(('float', 'str', 'repr', 'list', 'tuple', 'dict', 'set', 'frozenset', 'sorted',
                                   'format', 'range', 'zip', 'enumerate', 'map', 'filter', 'reversed', 'getattr'))


def _provider_returns():
    """{function name: 'int' or 'float'} of the bundled provider pxd files (helper modules are
    cimported under ``_mx_helper_<hash>`` aliases): C integer and C double results."""
    import pathlib
    import re
    kinds = {}
    pattern = re.compile(r'^\s*cdef\s+(?:inline\s+)?(long long|Py_ssize_t|int|long|double)\s+(\w+)\s*\(', re.M)
    root = pathlib.Path(__file__).parent
    for path in (*root.glob('*.pxd'), *(root / 'runtime').glob('*.pxd')):
        if path.name == '_mx_sys.pxd':
            continue
        for ctype, name in pattern.findall(path.read_text(encoding='utf-8')):
            kind = 'float' if ctype == 'double' else 'int'
            kinds[name] = kind if kinds.get(name, kind) == kind else None
    return kinds


_PROVIDER_RETURNS = _provider_returns()


def _provider_call(value, kind):
    """A call of a provider function (``_mx_helper_<hash>.name(...)``) whose C result is ``kind``."""
    return (isinstance(value, cst.Call) and isinstance(value.func, cst.Attribute)
            and isinstance(value.func.value, cst.Name) and value.func.value.value.startswith('_mx_helper_')
            and _PROVIDER_RETURNS.get(value.func.attr.value) == kind)


# handlers that cannot catch the OverflowError of a flagged formula (the generated reductions catch
# StopIteration); any other handler leaves the formula to the fallback
_HARMLESS_HANDLERS = frozenset(('StopIteration', 'KeyError', 'IndexError', 'TypeError', 'ValueError',
                                'AttributeError', 'LookupError', 'ZeroDivisionError'))


def _catches_overflow(node):
    for part in lexical_nodes(node.body.body):
        if isinstance(part, cst.ExceptHandler):
            names = ([dotted_name(e.value) for e in part.type.elements] if isinstance(part.type, cst.Tuple)
                     else [dotted_name(part.type)] if part.type is not None else [None])
            if not all(name in _HARMLESS_HANDLERS for name in names):
                return True
        elif isinstance(part, getattr(cst, 'ExceptStarHandler', ())):
            return True
    return False


def _annotation_kind(annotation):
    """'int', 'float', 'object' or None (another C type, or unknown) for a declared type."""
    if annotation is None:
        return 'object'
    if isinstance(annotation, cst.Subscript):
        return None
    return _KIND_OF_ANNOTATION.get(dotted_name(annotation))


def _declared_kinds(state):
    """The declared kinds of module functions, methods and fields, for the overflow fallback."""
    functions = {name: _annotation_kind(node.returns.annotation if node.returns else None)
                 for name, node in state.functions.items()}
    methods = {}
    for (owner, name), node in state.methods.items():
        methods.setdefault(owner, {})[name] = _annotation_kind(node.returns.annotation if node.returns else None)
    fields = {}
    for owner in state.syntax.body:
        if isinstance(owner, cst.ClassDef) and isinstance(owner.body, cst.IndentedBlock):
            declared = fields[owner.name.value] = {}
            for line in owner.body.body:
                for item in (line.body if isinstance(line, cst.SimpleStatementLine) else ()):
                    if isinstance(item, cst.AnnAssign) and isinstance(item.target, cst.Name):
                        annotation = item.annotation.annotation
                        element = annotation.value if isinstance(annotation, cst.Subscript) else annotation
                        declared[item.target.value] = (_KIND_OF_ANNOTATION.get(dotted_name(element)),
                                                       isinstance(annotation, cst.Subscript))
    return functions, methods, fields


class _NotInteger:
    """Expressions that Cython never computes as a C integer: a C double or a Python object.
    Anything else may be one; a + - * on it that the flags do not cover keeps Cython's
    overflowcheck on the formula (the fallback), so an integer that leaves 64 bits always raises."""

    def __init__(self, node, cls_name, declared, infer_types):
        functions, methods, fields = declared
        self.functions = functions
        self.methods = methods.get(cls_name, {})
        self.fields = fields.get(cls_name, {})
        params = (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)
        self.names = {param.name.value: _annotation_kind(param.annotation.annotation if param.annotation else None)
                      for param in params[1:]}
        stores = {}
        for part in walk(node.body):
            if isinstance(part, cst.AnnAssign) and isinstance(part.target, cst.Name):
                self.names[part.target.value] = _annotation_kind(part.annotation.annotation)
            elif isinstance(part, cst.Assign) and (target := local_assignment(part)) is not None:
                stores.setdefault(target.value, []).append(part.value)
            elif isinstance(part, (cst.Assign, cst.For, cst.AugAssign, cst.WithItem, cst.NamedExpr,
                                   cst.ExceptHandler, cst.CompFor)):
                targets = ([target.target for target in part.targets] if isinstance(part, cst.Assign)
                           else [getattr(part, 'target', None) or getattr(part, 'asname', None)
                                 or getattr(part, 'name', None)])
                for item in (name for target in targets if target is not None for name in walk(target)):
                    if isinstance(item, cst.Name):
                        stores.setdefault(item.value, []).append(None)
        # unannotated locals: a C double or object when every store is (Cython infers the spanning
        # type of the stores); without inference an unannotated local in arithmetic is an object
        candidates = {name: values for name, values in stores.items() if name not in self.names}
        if not infer_types:
            self.names.update({name: 'object' for name in candidates})
            return
        assumed = set(candidates)
        changed = True
        while changed:
            changed = False
            for name in sorted(assumed):
                self.names[name] = 'object'
                if not all(value is not None and self(value) for value in candidates[name]):
                    assumed.discard(name)
                    del self.names[name]
                    changed = True

    def __call__(self, value):
        if isinstance(value, (cst.Float, cst.Imaginary, cst.SimpleString, cst.ConcatenatedString,
                              cst.FormattedString, cst.List, cst.Tuple, cst.Dict, cst.Set, cst.ListComp,
                              cst.SetComp, cst.DictComp, cst.GeneratorExp, cst.Lambda)):
            return True
        if isinstance(value, cst.Name):
            return self.names.get(value.value) in ('float', 'object')
        if isinstance(value, cst.BinaryOperation):
            if isinstance(value.operator, cst.Divide):
                return True
            return self(value.left) or self(value.right)
        if isinstance(value, cst.UnaryOperation):
            return not isinstance(value.operator, cst.Not) and self(value.expression)
        if isinstance(value, cst.IfExp):
            return self(value.body) and self(value.orelse)
        if isinstance(value, cst.Call):
            func = value.func
            if isinstance(func, cst.Name):
                if func.value in self.functions:
                    return self.functions[func.value] in ('float', 'object')
                if func.value in _NOT_INTEGER_BUILTINS:
                    return True
                if func.value in ('min', 'max', 'abs') and value.args and all(arg.star == '' for arg in value.args):
                    # Cython's min/max/abs of C scalars take their spanning type: a C double or an
                    # object when one argument is
                    return any(self(arg.value) for arg in value.args if arg.keyword is None)
                return False
            if _provider_call(value, 'float'):
                return True
            if dotted_name(func) == f'{CY_MOD}.cast' and value.args:
                return _KIND_OF_ANNOTATION.get(dotted_name(value.args[0].value)) in ('float', 'object')
            if (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                    and func.value.value == MX_SELF):
                return self.methods.get(func.attr.value) in ('float', 'object')
            return False
        base = value.value if isinstance(value, cst.Subscript) else value
        if (isinstance(base, cst.Attribute) and isinstance(base.value, cst.Name)
                and base.value.value == MX_SELF and base.attr.value in self.fields):
            kind, array = self.fields[base.attr.value]
            if array == isinstance(value, cst.Subscript):
                return kind in ('float', 'object')
        return False


def _is_literal(value):
    """An integer constant expression: Cython folds it with Python's semantics."""
    return isinstance(value, cst.Integer) or (
        isinstance(value, cst.UnaryOperation) and isinstance(value.operator, (cst.Minus, cst.Plus))
        and _is_literal(value.expression)) or (
        isinstance(value, cst.BinaryOperation) and isinstance(value.operator, _WRAPPING)
        and _is_literal(value.left) and _is_literal(value.right))


def _literal_value(value):
    """The Python value of an integer constant expression (Cython folds it exactly), else None."""
    if isinstance(value, cst.Integer):
        return value.evaluated_value
    if isinstance(value, cst.UnaryOperation) and isinstance(value.operator, (cst.Minus, cst.Plus)):
        operand = _literal_value(value.expression)
        return None if operand is None else (-operand if isinstance(value.operator, cst.Minus) else operand)
    if isinstance(value, cst.BinaryOperation) and isinstance(value.operator, (cst.Add, cst.Subtract, cst.Multiply)):
        left, right = _literal_value(value.left), _literal_value(value.right)
        if left is None or right is None:
            return None
        return {cst.Add: left + right, cst.Subtract: left - right, cst.Multiply: left * right}[type(value.operator)]
    return None


def _flag_call(name, flag, *values):
    return cst.Call(cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(name)),
                    tuple(cst.Arg(value) for value in values) + (cst.Arg(cst.parse_expression(
                        f'{CY_MOD}.address({flag})')),))


class _FlagArithmetic(cst.CSTTransformer):
    """+ - * and unary minus of two C integers (the guard's signed expressions) through the
    flagged helpers; the operations of nested scopes are left to the fallback."""

    def __init__(self, guard, flag):
        self.guard = guard
        self.flag = flag
        self.count = 0

    def on_visit(self, node):
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)

    def _signed(self, value):
        if _is_literal(value):
            # a constant beyond 64 bits is a Python int: the operation stays on Python objects
            constant = _literal_value(value)
            return constant is not None and -(1 << 63) <= constant < (1 << 63)
        return self.guard._signed_expression(value, self.guard.path)[1]

    def leave_Call(self, before, after):
        # an argument computed with flagged operations is tested before the call receives it, so
        # a wrapped value never reaches another Cell (its range guard, its cache): the OverflowError
        # names the Cell that computed it
        target = dotted_name(after.func) or ''
        if (target.startswith((MX_SYS_MOD + '.', CY_MOD + '.')) or target in _PURE_CONVERSIONS
                or not self.flag):
            return after
        arguments = tuple(
            arg.with_changes(value=cst.Call(
                cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name('_mx_flag_checked')),
                (cst.Arg(arg.value), cst.Arg(cst.parse_expression(f'{CY_MOD}.address({self.flag})')),
                 cst.Arg(cst.SimpleString(repr(self.guard.path))))))
            if arg.star == '' and self._signed(arg.value) and any(
                isinstance(part, cst.Call) and dotted_name(part.func) in _FLAG_CALLS for part in walk(arg.value))
            else arg
            for arg in after.args)
        if all(new is old for new, old in zip(arguments, after.args)):
            return after
        self.checked_arguments = getattr(self, 'checked_arguments', 0) + 1
        return after.with_changes(args=arguments)

    def leave_BinaryOperation(self, before, after):
        name = _FLAGGED.get(type(after.operator))
        if (name is None or (_is_literal(after.left) and _is_literal(after.right))
                or not (self._signed(after.left) and self._signed(after.right))):
            return after
        self.count += 1
        return _flag_call(name, self.flag, after.left, after.right).with_changes(lpar=after.lpar, rpar=after.rpar)

    def leave_UnaryOperation(self, before, after):
        if (not isinstance(after.operator, cst.Minus) or _is_literal(after.expression)
                or not self._signed(after.expression)):
            return after
        self.count += 1
        return _flag_call('_mx_flag_neg', self.flag, after.expression).with_changes(
            lpar=after.lpar, rpar=after.rpar)

    def leave_AugAssign(self, before, after):
        name = _FLAGGED.get(type(after.operator))
        if (name is None or not isinstance(after.target, cst.Name)
                or not (self._signed(after.target) and self._signed(after.value))):
            return after
        self.count += 1
        return cst.Assign((cst.AssignTarget(after.target),), _flag_call(name, self.flag, after.target, after.value))


class _CheckFlag(cst.CSTTransformer):
    """Test the flag once before the formula returns: ``r = value; if flag: raise ...; return r``."""

    def __init__(self, flag, result, path):
        self.flag = flag
        self.result = result
        self.check = cst.parse_statement(f'if {flag}:\n    raise {MX_SYS_MOD}._mx_flagged_overflow({path!r})\n')

    def on_visit(self, node):
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)

    def _lines(self, small):
        if not isinstance(small, cst.Return):
            return [cst.SimpleStatementLine((small,))]
        if small.value is None:
            return [self.check, cst.SimpleStatementLine((small,))]
        return [cst.SimpleStatementLine((cst.Assign((cst.AssignTarget(cst.Name(self.result)),), small.value),)),
                self.check, cst.SimpleStatementLine((small.with_changes(value=cst.Name(self.result)),))]

    def leave_SimpleStatementLine(self, before, after):
        if not any(isinstance(small, cst.Return) for small in after.body):
            return after
        lines = [line for small in after.body for line in self._lines(small)]
        lines[0] = lines[0].with_changes(leading_lines=after.leading_lines)
        return cst.FlattenSentinel(lines)

    def leave_SimpleStatementSuite(self, before, after):
        if not any(isinstance(small, cst.Return) for small in after.body):
            return after
        return cst.IndentedBlock(body=[line for small in after.body for line in self._lines(small)])

    def leave_For(self, before, after):
        # a loop whose iterable reads a flagged result tests the flag first: a wrapped bound could
        # otherwise run a loop of up to 2**63 iterations before the test at the return
        if not any(isinstance(part, cst.Call) and isinstance(part.func, cst.Attribute)
                   and part.func.attr.value.startswith('_mx_flag_') for part in walk(after.iter)):
            return after
        return cst.FlattenSentinel((self.check.with_changes(leading_lines=after.leading_lines),
                                    after.with_changes(leading_lines=())))


def _wrapping_operations(node):
    """The + - * and unary minus left in a formula (in nested scopes as well)."""
    for part in walk(node.body):
        if isinstance(part, cst.BinaryOperation) and isinstance(part.operator, _WRAPPING):
            yield part.left, part.right
        elif isinstance(part, cst.AugAssign) and isinstance(part.operator, _WRAPPING):
            yield part.target, part.value
        elif isinstance(part, cst.UnaryOperation) and isinstance(part.operator, cst.Minus):
            yield part.expression, part.expression


def _name_overflows(node, path):
    """The overflowcheck fallback: one handler around the body of the formula, after its declarations,
    so that an OverflowError of Cython's overflowcheck names the Cell; an error a Cell already named
    (a flagged formula, a call to another Cell, a checked conversion) passes unchanged."""
    lines = list(node.body.body)
    prefix = 1 if node.get_docstring() is not None else 0
    while prefix < len(lines) and isinstance(lines[prefix], cst.SimpleStatementLine) and all(
            isinstance(small, cst.AnnAssign) and small.value is None
            or isinstance(small, cst.AnnAssign) and isinstance(small.value, cst.Integer) for small in lines[prefix].body):
        prefix += 1
    if prefix == len(lines):
        return node
    error = Names(node).reserve('_mx_integer_overflow')
    handler = cst.parse_statement(
        f'try:\n    pass\nexcept OverflowError as {error}:\n'
        f'    raise {MX_SYS_MOD}._mx_named_overflow({error}, {path!r})\n')
    handler = handler.with_changes(body=cst.IndentedBlock(body=tuple(lines[prefix:])))
    return node.with_changes(body=node.body.with_changes(body=(*lines[:prefix], handler)))


def _flag_overflows(node, cell, guard, declared, cls_name, infer_types):
    """integer guards: the C integer + - * and unary minus of one formula set a flag that the formula
    tests once before it returns (OverflowError naming the Cell); a formula with an operation the
    guard cannot show to be on C integers or not (a nested scope, an operand of unknown C type)
    keeps Cython's overflowcheck on that formula alone."""
    if not isinstance(node.body, cst.IndentedBlock):
        return node
    if _catches_overflow(node) or any(isinstance(part, cst.Yield) for part in lexical_nodes(node.body.body)):
        flagged = _FlagArithmetic(guard, '')     # a generator, or a handler that could catch the error
        flagged.count = 0
    else:
        names = Names(node)
        flag = names.reserve('_mx_overflow')
        flagged = _FlagArithmetic(guard, flag)
        body = node.body.visit(flagged)
        if flagged.count:
            result = names.reserve('_mx_result')
            returns = node.returns.annotation if node.returns is not None else cst.Name('object')
            body = body.visit(_CheckFlag(flag, result, cell.model_path))
            lines = list(body.body)
            if not lines or not isinstance(lines[-1], cst.SimpleStatementLine) or not any(
                    isinstance(small, (cst.Return, cst.Raise)) for small in lines[-1].body):
                lines.append(_CheckFlag(flag, result, cell.model_path).check)
            doc_end = 1 if node.get_docstring() is not None else 0
            declarations = (cst.parse_statement(f'{flag}: {CY_MOD}.int = 0\n'),
                            cst.SimpleStatementLine((cst.AnnAssign(cst.Name(result), cst.Annotation(returns)),)))
            node = node.with_changes(body=body.with_changes(
                body=(*lines[:doc_end], *declarations, *lines[doc_end:])))
    cell.flagged_integer_operations = getattr(cell, 'flagged_integer_operations', 0) + flagged.count
    not_integer = _NotInteger(node, cls_name, declared, infer_types)
    if any(not (_is_literal(left) and _is_literal(right)) and not (not_integer(left) or not_integer(right))
           for left, right in _wrapping_operations(node)):
        cell.overflowcheck_fallback = True
        decorator = cst.Decorator(cst.parse_expression(f'{CY_MOD}.overflowcheck(True)'))
        node = _name_overflows(node.with_changes(decorators=(*node.decorators, decorator)), cell.model_path)
    return node



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


def _reference_write_gate(cls_info):
    """A Python write of an integer Reference crosses the same exact 64-bit boundary as a generated
    one (``_mx_exact_int64``): Cython's own setter of a ``cdef public long long`` converts a float
    through ``__int__`` and would keep 0 for 0.5."""
    from .transformer import _integer_reference_paths
    paths = _integer_reference_paths(cls_info)
    if not paths:
        return ()
    tests = ''.join(f"    {'if' if index == 0 else 'elif'} {MX_SYS_MOD}._mx_ref_name_is(name, {name!r}):\n"
                    f"        value = {MX_SYS_MOD}._mx_exact_int64(value, {path!r})\n"
                    for index, (name, path) in enumerate(sorted(paths.items())))
    return (cst.parse_statement(f'def __setattr__(self, name, value):\n{tests}'
                                f'    {MX_SYS_MOD}._mx_generic_setattr(self, name, value)\n'),)


def _unbound_locals(node):
    """A local that a read can reach before any assignment, on some path, is a Python object: Python
    raises UnboundLocalError there (a loop that ran no iteration, a branch not taken), and a C local
    has no unbound state (with type inference, or a declaration of the vector lowering, Cython would
    read an unset value)."""
    if not isinstance(node.body, cst.IndentedBlock):
        return node
    body = node.body.body
    facts = CstFacts(body)
    parameters = {param.name.value for param in (*node.params.posonly_params, *node.params.params,
                                                 *node.params.kwonly_params)}
    names = sorted(name for name, count in facts.stores.items()
                   if count and name not in parameters and maybe_unbound(body, name, facts))
    if not names:
        return node
    declared = {}
    for part in lexical_nodes(body):
        if isinstance(part, cst.AnnAssign) and isinstance(part.target, cst.Name) and part.target.value in names:
            declared.setdefault(part.target.value, []).append(part)
    retyped = {id(item) for name in names for item in declared.get(name, ()) if item.value is None}

    class Retype(cst.CSTTransformer):
        def leave_AnnAssign(self, before, after):
            if id(before) in retyped:
                return after.with_changes(annotation=cst.Annotation(cst.Name('object')))
            return after

    body_node = node.body.visit(Retype()) if retyped else node.body
    added = tuple(cst.SimpleStatementLine((cst.AnnAssign(cst.Name(name), cst.Annotation(cst.Name('object'))),))
                  for name in names if name not in declared)
    doc_end = 1 if node.get_docstring() is not None else 0
    lines = body_node.body
    return node.with_changes(body=body_node.with_changes(
        body=(*lines[:doc_end], *added, *lines[doc_end:])))


# The builtins the call guard folds when the module leaves them unbound
_FOLDED_BUILTINS = ('isinstance', 'int', 'bool')


class _GuardCalls(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider,)

    def __init__(self, receiver, integer_params, owner, resolver, entries, cells, private_entries,
                 safe_builtins=()):
        self.receiver = receiver
        self.integer_params = integer_params
        self.owner = owner
        self.resolver = resolver
        self.entries = entries
        self.cells = cells
        self.private_entries = private_entries
        self.safe_builtins = safe_builtins

    def _native_argument(self, value):
        if not isinstance(value, cst.Name):
            return False
        scope = self.get_metadata(cst.metadata.ScopeProvider, value, None)
        if value.value not in self.integer_params:
            return False
        bindings = tuple(scope[value.value]) if scope is not None else ()
        return len(bindings) == 1 and bindings[0].node is self.integer_params[value.value]

    def leave_Call(self, before, after):
        folded = self._native_isinstance(before, after)
        if folded is not None:
            return folded
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

    def _builtin(self, node, name):
        if name not in self.safe_builtins or not isinstance(node, cst.Name) or node.value != name:
            return False
        scope = self.get_metadata(cst.metadata.ScopeProvider, node, None)
        bindings = tuple(scope[name]) if scope is not None else ()
        return bool(bindings) and all(isinstance(binding, BuiltinAssignment) for binding in bindings)

    def _native_isinstance(self, before, after):
        if not self._builtin(before.func, 'isinstance'):
            return None
        operands = arguments(before, count=2)
        if operands is None or not self._native_argument(operands[0]):
            return None
        selector = operands[1]
        if self._builtin(selector, 'int'):
            return cst.Name('True', lpar=after.lpar, rpar=after.rpar)
        if self._builtin(selector, 'bool'):
            return cst.Name('False', lpar=after.lpar, rpar=after.rpar)
        if (isinstance(selector, cst.Tuple) and len(selector.elements) == 2
                and all(isinstance(item, cst.Element) for item in selector.elements)
                and self._builtin(selector.elements[0].value, 'int')):
            # A tuple evaluates its trailing type expression even when the
            # first type matches. Preserve that read and its exceptions.
            tail = after.args[1].value.elements[1].value
            return cst.parse_expression(f'(({cst.Module([]).code_for_node(tail)}) is None or True)')
        return None

    def _arguments(self, before, after, path, params, start=0):
        guarded = {index + start for index, _ in params}
        arguments = tuple(arg.with_changes(value=_checked_integer(arg.value, path))
                          if index in guarded and arg.keyword is None and not self._native_argument(old.value)
                          else arg for index, (old, arg) in enumerate(zip(before.args, after.args)))
        return after.with_changes(args=arguments)


def _guard_calls(node, cls, cell, resolver, entries, cells, private_entries, classes=None, safe_builtins=(),
                 module=None):
    """``module``: a MetadataWrapper whose module holds ``node`` itself, its scopes resolved.

    The pass asks scopes about the formula's parameters (``self`` among them) and about the
    builtins it folds, which the module binds nowhere (``safe_builtins``). For those names the
    formula's scopes in its module answer as in a module holding the formula alone, unless the
    formula is itself named as one of the builtins.
    """
    params = (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)
    if not params or params[0].name.value != MX_SELF:
        return node
    integer_params = {param.name.value: param for param in params if param.annotation is not None
                      and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    visitor = _GuardCalls(params[0], integer_params, cls.fqname, resolver, entries, cells,
                          private_entries, safe_builtins)
    if module is not None and node.name.value not in _FOLDED_BUILTINS:
        with visitor.resolve(module):
            return node.visit(visitor)
    wrapper = cst.MetadataWrapper(cst.Module(body=(node,)), unsafe_skip_copy=True)
    return wrapper.visit(visitor).body[0]


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


class _GuardIntegers(cst.CSTTransformer):
    def __init__(self, path, kinds, native_helpers, signed_fields,
                 signed_functions, signed_methods, signed_names, signed_return, error, checked=True):
        self.path = path
        # integer guards: False in the unchecked mode, where C integer arithmetic stays plain C
        self.checked = checked
        self.kinds = kinds
        self.native_helpers = native_helpers
        self.signed_fields = signed_fields
        self.signed_functions = signed_functions
        self.signed_methods = signed_methods
        self.signed_names = signed_names
        self.signed_return = signed_return
        self.error = error

    def on_visit(self, node):
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)


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
        if isinstance(value, cst.BinaryOperation) and isinstance(value.operator, _INTEGER_OPERATIONS):
            left, left_signed = self._signed_expression(value.left, path)
            right, right_signed = self._signed_expression(value.right, path)
            # Shifts and the remainder keep their helpers in both modes, for Python's semantics:
            # a right shift rounds a negative value down (C leaves it to the implementation), a
            # left shift of a negative value or by 64 or more is exact in Python (Cython's checked
            # shift raises), LLONG_MIN % -1 is 0 and LLONG_MIN // -1 raises (C traps, and Cython
            # checks the division only where long long is as wide as long). An integer power
            # stored as a C integer takes the checked power in both modes: Cython's own C integer
            # power wraps, and with a variable exponent Cython makes it a double, which a C integer
            # result cannot take (at all, not only beyond 64 bits). Every other
            # operator is plain C: Cython's overflowcheck checks it in the checked mode.
            if (isinstance(value.operator, _HELPER_OPERATIONS) and left_signed and right_signed
                    and not (isinstance(value.operator, cst.FloorDivide) and _positive_literal(value.right))):
                operation = _SIGNED_OPERATIONS[type(value.operator)]
                checked = _boundary_call(operation, path, left, right)
                return checked.with_changes(lpar=value.lpar, rpar=value.rpar), True
            return value.with_changes(left=left, right=right), left_signed and right_signed
        if isinstance(value, cst.IfExp):
            body, body_signed = self._signed_expression(value.body, path)
            orelse, else_signed = self._signed_expression(value.orelse, path)
            return value.with_changes(body=body, orelse=orelse), body_signed and else_signed
        if isinstance(value, cst.UnaryOperation) and isinstance(value.operator, (cst.Plus, cst.Minus)):
            operand, signed = self._signed_expression(value.expression, path)
            if signed and isinstance(value.operator, cst.Minus) and self.checked:
                # Cython's overflowcheck does not check a unary minus (-LLONG_MIN): as 0 - x it does
                return cst.BinaryOperation(cst.Integer('0'), cst.Subtract(), operand,
                                           lpar=(cst.LeftParen(), *value.lpar),
                                           rpar=(*value.rpar, cst.RightParen())), True
            return value.with_changes(expression=operand), signed
        if isinstance(value, cst.Integer):
            return value, value.evaluated_value <= (1 << 63) - 1
        return value, self._signed_atom(value)

    def _signed_atom(self, value):
        if isinstance(value, cst.Name):
            return value.value in self.signed_names
        if (isinstance(value, cst.Call) and dotted_name(value.func) == f'{CY_MOD}.cast' and len(value.args) == 2
                and dotted_name(value.args[0].value) == f'{CY_MOD}.longlong'):
            return True         # the compiler's own C cast to long long (the virtual branch of a dispatch)
        if _provider_call(value, 'int'):
            return True
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
        return self._signed_expression(value, path)


def _signed_operand_names(value):
    """Find the name uses that signed expression lowering treats as C operands."""
    if isinstance(value, cst.Name):
        yield value
    elif isinstance(value, cst.BinaryOperation) and isinstance(value.operator, _INTEGER_OPERATIONS):
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
            guard.python_locals.add(name)       # examined and left to Python
            continue
        declarations.append(cst.SimpleStatementLine((cst.AnnAssign(
            cst.Name(name), cst.Annotation(_INTEGER_ANNOTATION)),)))
        assignments.update(values)
    body = body.with_changes(body=statements)
    if assignments:
        body = body.visit(_SignedLocalValues(assignments))
    return body, tuple(declarations)


def _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds, signatures, native_helpers=(),
                  argument_axes=(), checked=True):
    signed_return = node.returns is not None and node.returns.annotation.deep_equals(_INTEGER_ANNOTATION)
    params = {name: kind_of_type_expr(cell.get_argtype_expr(name, c_style=True, formula=True))
              for name in cell.params}
    kinds = OperandKind(resolver, cells_kinds, ref_kinds, cls.fqname, params)
    integer_parameters = {
        param.name.value for param in node.params.params
        if param.annotation is not None and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)
    }
    facts = CstFacts(node.body.body)
    bindings = facts.bindings(lexical_nodes(node.body.body))
    if not signed_return and not any(
            isinstance(part, cst.Call) and dotted_name(part.func) == _INTEGER_CONVERSION
            for part in lexical_nodes(node.body.body)):
        return node
    fields, functions, methods = signatures
    signed_names = integer_parameters | {
        statement.target.value for statement in lexical_nodes(node.body.body)
        if isinstance(statement, cst.AnnAssign) and isinstance(statement.target, cst.Name)
        and statement.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)
    }
    receiver_fields = fields.get(cls.name, {}) if MX_SELF not in bindings else {}
    receiver_methods = methods.get(cls.name, set()) if MX_SELF not in bindings else set()
    error = Names(node).reserve('_mx_integer_overflow')
    guard = _GuardIntegers(cell.model_path, kinds, native_helpers, receiver_fields,
                           functions.difference(bindings), receiver_methods, signed_names, signed_return,
                           error, checked)
    parameters = tuple(parameter.name.value for parameter in
                       (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params))
    guard.python_locals = set()
    body, declarations = _type_signed_locals(node.body, guard, facts, parameters)
    cell.python_int_locals = set(getattr(cell, 'python_int_locals', ())) | guard.python_locals
    body = body.visit(guard)
    if declarations:
        doc_end = 1 if node.get_docstring() is not None else 0
        body = body.with_changes(body=(*body.body[:doc_end], *declarations, *body.body[doc_end:]))
    return node.with_changes(body=body)


def _positive_literal(value):
    """A divisor that cannot be -1: the division keeps Cython's C operator (no trap)."""
    return isinstance(value, cst.Integer) and value.evaluated_value > 0


class _GuardDivisions(cst.CSTTransformer):
    """Every ``//`` and ``%`` of two C integers takes the checked helper, in both modes and
    wherever it is (a float formula's ``12 // m`` as well as a C integer result): C traps on
    LLONG_MIN // -1 and LLONG_MIN % -1 (see _mx_signed_floordiv). A divisor that is a positive
    literal cannot be -1 and stays a C operator."""

    def __init__(self, guard):
        self.guard = guard

    def on_visit(self, node):
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)

    def _signed(self, value):
        return self.guard._signed_expression(value, self.guard.path)[1]

    def leave_BinaryOperation(self, before, after):
        operation = _DIVISIONS.get(type(after.operator))
        if (operation is None or _positive_literal(after.right)
                or not (self._signed(before.left) and self._signed(before.right))):
            return after
        return _boundary_call(operation, self.guard.path, after.left, after.right).with_changes(
            lpar=after.lpar, rpar=after.rpar)

    def leave_AugAssign(self, before, after):
        operation = _DIVISIONS.get(type(after.operator))
        if (operation is None or not isinstance(after.target, cst.Name)
                or _positive_literal(after.value)
                or not (self._signed(before.target) and self._signed(before.value))):
            return after
        return cst.Assign((cst.AssignTarget(after.target),),
                          _boundary_call(operation, self.guard.path, after.target, after.value))


def _integer_guard(node, cell, cls, signatures, native_helpers=(), inferred=True):
    """The signed expressions of one formula function: parameters, declared and (``inferred``:
    with Cython's type inference) assigned locals, fields, casts and calls of C integer type."""
    fields, functions, methods = signatures
    facts = CstFacts(node.body.body)
    bindings = facts.bindings(lexical_nodes(node.body.body))
    parameters = {param.name.value for param in (*node.params.posonly_params, *node.params.params,
                                                 *node.params.kwonly_params)
                  if param.annotation is not None and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    declared = {statement.target.value for statement in lexical_nodes(node.body.body)
                if isinstance(statement, cst.AnnAssign) and isinstance(statement.target, cst.Name)
                and statement.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    guard = _GuardIntegers(cell.model_path, None, native_helpers,
                           fields.get(cls.name, {}) if MX_SELF not in bindings else {},
                           functions.difference(bindings),
                           methods.get(cls.name, set()) if MX_SELF not in bindings else set(),
                           parameters | declared, False, None, False)
    # locals Cython infers as C integers: every store an assignment (plain or augmented) of a
    # C integer expression; any other store (a loop target, a float value) leaves the local out
    if not inferred:
        return guard
    values = {}
    for part in lexical_nodes(node.body.body):
        if isinstance(part, cst.AugAssign) and isinstance(part.target, cst.Name):
            values.setdefault(part.target.value, []).append(part.value)
        elif (target := local_assignment(part)) is not None:
            values.setdefault(target.value, []).append(part.value)
    candidates = {name for name, count in facts.stores.items()
                  if count and name not in guard.signed_names and len(values.get(name, ())) == count}
    changed = True
    while changed:              # greatest fixpoint: assume all, drop the ones an assignment refutes
        changed = False
        guard.signed_names |= candidates
        for name in sorted(candidates):
            if not all(guard._signed_expression(value, guard.path)[1] for value in values[name]):
                candidates.discard(name)
                guard.signed_names.discard(name)
                changed = True
    return guard


def _guard_divisions(node, cell, cls, signatures, native_helpers=()):
    """Route the C integer ``//`` and ``%`` of one formula function through the helpers. The
    operands are recognised as the result lowering recognises them: parameters, declared and
    assigned locals, fields, casts and calls of C integer type."""
    if not any(isinstance(part, (cst.BinaryOperation, cst.AugAssign)) and type(part.operator) in _DIVISIONS
               for part in lexical_nodes(node.body.body)):
        return node
    guard = _integer_guard(node, cell, cls, signatures, native_helpers)
    return node.with_changes(body=node.body.visit(_GuardDivisions(guard)))


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
    """Declare ``object`` the locals compared by identity.

    With ``infer_types`` Cython infers a C type for every untyped local; a local
    compared with ``is``/``is not`` would lose its identity (``x is saved[0]`` on a re-boxed
    int gives a wrong answer), so it is declared ``object``. Integer locals and ``range``
    targets otherwise stay C integers (their arithmetic: plain C, checked by overflowcheck in the checked mode).
    Locals with an annotation of their own are left alone. The admission report lists each
    declared local (``object_locals``).
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
    # Integer locals and range targets stay C integers with guarded arithmetic
    # (integer guards); only identity keeps the Python object
    names = set()
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
    # integer guards: arguments, keys and results are checked in both modes; arithmetic only in
    # the checked mode
    checked = module_info.spec.policy.integer_checks != 'unchecked'
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
    gate_class = state.names.reserve('_mx_checked_cell_call')
    changed = False
    signatures = _signed_signatures(state)
    declared = _declared_kinds(state)
    module_scopes = cst.MetadataWrapper(state.syntax, unsafe_skip_copy=True)
    global_bindings = module_scopes.resolve(cst.metadata.ScopeProvider)[state.syntax]
    safe_builtins = {name for name in _FOLDED_BUILTINS
                     if all(isinstance(binding, BuiltinAssignment) for binding in global_bindings[name])}
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
                # The private entry of an uncached Cell has the Cell's own name as its origin
                cell = cls.cells[name[len(FORMULA_PREF):] if name.startswith(FORMULA_PREF) else name]
                unbound = _unbound_locals(node)
                node = _guard_calls(unbound, cls, cell, resolver, entries, cells, private_entries, module_info.classes,
                                    safe_builtins, module_scopes if unbound is node else None)
                # Physical cache misses and compact spill entries both check
                # the logical domain before reaching the private formula.
                axes = (cell.dense_logical_axes if cell.has_formula_def and cell.has_args() and cell.has_typeinfo()
                        and cell.is_arrayable() else ())
                node = _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds, signatures,
                                     argument_axes=axes, checked=checked)
                node = _guard_divisions(node, cell, cls, signatures)
                if checked:
                    node = _flag_overflows(node, cell, _integer_guard(node, cell, cls, signatures, inferred=infer_types),
                                           declared, cls.name, infer_types)
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
                unbound = _unbound_locals(method)
                method = _guard_calls(unbound, cls, formula, resolver, entries, cells, private_entries,
                                      module_info.classes, safe_builtins, module_scopes if unbound is method else None)
                method = _guard_result(method, formula, cls, resolver, cells_kinds, ref_kinds, signatures,
                                       native_helpers.get((cls.name, name), ()), checked=checked)
                method = _guard_divisions(method, formula, cls, signatures,
                                          native_helpers.get((cls.name, name), ()))
                if checked:
                    method = _flag_overflows(method, formula, _integer_guard(
                        method, formula, cls, signatures, native_helpers.get((cls.name, name), ()),
                        inferred=infer_types), declared, cls.name, infer_types)
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
                unbound = _unbound_locals(method)
                method = _guard_calls(unbound, cls, cell, resolver, entries, cells, private_entries, module_info.classes,
                                      safe_builtins, module_scopes if unbound is method else None)
                method = _guard_result(method, cell, cls, resolver, cells_kinds, ref_kinds, signatures,
                                       checked=checked)
                method = _guard_divisions(method, cell, cls, signatures)
                if checked:
                    method = _flag_overflows(method, cell, _integer_guard(method, cell, cls, signatures,
                                                                          inferred=infer_types),
                                             declared, cls.name, infer_types)
                if infer_types:
                    method = _python_locals(method, cell)
                changed = True
            methods.append(method)
        added, gates = _python_call_gate(cls, entries, state.names, call_gate)
        declarations.extend(added)
        writes = () if any(isinstance(item, cst.FunctionDef) and item.name.value == '__setattr__'
                           for item in methods) else _reference_write_gate(cls)
        gates = (*gates, *writes)
        changed |= bool(gates)
        classes.append(node.with_changes(body=node.body.with_changes(body=(*methods, *gates))))
    if declarations:
        classes.extend(declarations)
        classes.extend(cst.parse_module(_CALL_GATE.format(
            cy=CY_MOD, name=call_gate, gate=gate_class, runtime=MX_SYS_MOD)).body)
    return state.replace(state.syntax.with_changes(body=classes)) if changed else state
