"""Bind module math members to the math provider without freezing exported References.

Providers supply a .pyx/.pxd pair. Their Cython functions receive the current
module or callable and must preserve its behavior when it is replaced.
"""
import hashlib
from importlib import resources

import libcst as cst

from .scopes import FormulaTransformer
from .consts import CY_MOD

def module_name(provider):
    return '_mx_helper_' + hashlib.sha256(provider.encode()).hexdigest()[:12]


def read_resource(package, name):
    return (resources.files(package).joinpath(name).read_bytes()
            if hasattr(resources, 'files') else resources.read_binary(package, name))


def borrowed_namespace(reference):
    """diag-mt2: the module Reference as the provider's borrowed ``void*`` (no reference taken)."""
    return cst.Call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                    (cst.Arg(cst.Attribute(cst.Name('_mx_cy'), cst.Name('p_void'))), cst.Arg(reference)))


class HelperCalls(FormulaTransformer):
    def __init__(self, refs, used, cells=(), current=None):
        super().__init__()
        self.refs = refs
        self.used = used
        self.operations = set()
        self.cells = cells
        self.current = current          # the formula's cells (argument kinds), when known
        self.math_native = 0            # W1.4: module math members bound at build time
        self.math_untyped = []

    # ---------------------------------------------------------------- math branch (W1.4)
    def module_math(self, original, updated):
        """``self.<ref>.<fn>(x)`` on a module Reference: the provider gets the Reference.

        Returns the new call, ``updated`` itself to keep the Python call (listed in the
        admission report), or None when the call is no module math member.
        """
        from .helper_members import MATH_PROVIDER, argument_kind
        func = updated.func
        if isinstance(func, cst.Attribute) and self.local_math(original.func.value):
            members = _stdlib_math_members()
        elif not (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Attribute)
                  and isinstance(func.value.value, cst.Name) and func.value.value.value == 'self'):
            return None
        else:
            members = getattr(self.refs.get(func.value.attr.value), 'helper_members', {})
        if not any(getattr(member, 'module_math', None) or hasattr(member, 'module_constant')
                   for member in members.values()):
            return None
        name = getattr(members.get(func.attr.value), 'module_math', None)
        if not name:
            self.decline(original, f'math.{func.attr.value} has no native form; runs as a Python call')
            return updated
        if name == 'isclose':
            return self.module_isclose(original, updated, func)
        if len(updated.args) != 1 or updated.args[0].keyword or updated.args[0].star:
            self.decline(original, f'math.{name} with {len(updated.args)} arguments or keywords; '
                         'runs as a Python call')
            return updated
        operation = 'm_' + name
        if name in ('floor', 'ceil'):
            kind = argument_kind(original.args[0].value, self.current, self.cells, self.refs, self.formula_name)
            if kind in ('double', 'longlong'):
                operation += '_ll'
        self.used.add(MATH_PROVIDER)
        self.operations.add(MATH_PROVIDER + '.' + operation)
        self.math_native += 1
        return updated.with_changes(
            func=cst.Attribute(cst.Name(module_name(MATH_PROVIDER)), cst.Name(operation)),
            args=(cst.Arg(self.namespace(func.value)), cst.Arg(updated.args[0].value)))

    def module_isclose(self, original, updated, func):
        """W1.8: ``self.<math>.isclose(a, b[, rel_tol=r][, abs_tol=t])`` ->
        ``m_isclose(self.<math>, a, b, r, t, given)`` (defaults 1e-09 and 0.0; ``given``
        records which keywords were written, for a replaced Reference). The tolerances
        must be read without running code (literals, typed References or parameters), so
        evaluating them in the C call's order keeps the original order."""
        from .helper_members import MATH_PROVIDER, argument_kind
        args = updated.args
        positional = [arg for arg in args if arg.keyword is None]
        keywords = {arg.keyword.value: arg.value for arg in args if arg.keyword is not None}
        if (any(arg.star for arg in args) or len(positional) != 2
                or any(arg.keyword is None for arg in args[2:])
                or set(keywords) - {'rel_tol', 'abs_tol'} or len(keywords) != len(args) - 2):
            self.decline(original, 'math.isclose with other than two positional arguments and '
                         'rel_tol/abs_tol keywords; runs as a Python call')
            return updated
        defaults = {'rel_tol': cst.Float('1e-09'), 'abs_tol': cst.Float('0.0')}
        values = []
        for name in ('rel_tol', 'abs_tol'):
            value = keywords.get(name)
            if value is None:
                values.append(defaults[name])
                continue
            plain = (isinstance(value, (cst.Float, cst.Integer))
                     or isinstance(value, cst.UnaryOperation) and isinstance(value.expression, (cst.Float, cst.Integer))
                     or isinstance(value, cst.Name)
                     or isinstance(value, cst.Attribute) and isinstance(value.value, cst.Name)
                     and value.value.value == 'self')
            written = next(arg.value for arg in original.args if arg.keyword is not None and arg.keyword.value == name)
            if not plain or argument_kind(written, self.current, self.cells, self.refs,
                                          self.formula_name) not in ('double', 'longlong'):
                self.decline(original, f'math.isclose {name} is not a float or integer literal, '
                             'typed Reference or parameter; runs as a Python call')
                return updated
            values.append(value)
        given = ('rel_tol' in keywords) + 2 * ('abs_tol' in keywords)
        self.used.add(MATH_PROVIDER)
        self.operations.add(MATH_PROVIDER + '.m_isclose')
        self.math_native += 1
        return updated.with_changes(
            func=cst.Attribute(cst.Name(module_name(MATH_PROVIDER)), cst.Name('m_isclose')),
            args=(cst.Arg(self.namespace(func.value)), cst.Arg(positional[0].value), cst.Arg(positional[1].value),
                  cst.Arg(values[0]), cst.Arg(values[1]), cst.Arg(cst.Integer(str(given)))))

    def module_constant(self, updated, original=None):
        """``self.<ref>.inf`` (``nan``, ``pi``, ``e``, ``tau``) of the math module."""
        from .helper_members import MATH_PROVIDER
        if original is not None and self.local_math(original.value):
            members = _stdlib_math_members()
        elif not (isinstance(updated.value, cst.Attribute) and isinstance(updated.value.value, cst.Name)
                  and updated.value.value.value == 'self'):
            return None
        else:
            members = getattr(self.refs.get(updated.value.attr.value), 'helper_members', {})
        index = getattr(members.get(updated.attr.value), 'module_constant', None)
        if index is None:
            return None
        self.used.add(MATH_PROVIDER)
        self.operations.add(MATH_PROVIDER + '.m_constant')
        self.math_native += 1
        return cst.Call(cst.Attribute(cst.Name(module_name(MATH_PROVIDER)), cst.Name('m_constant')),
                        (cst.Arg(self.namespace(updated.value)), cst.Arg(cst.Integer(str(index)))),
                        lpar=updated.lpar, rpar=updated.rpar)

    def local_math(self, node):
        """diag-scale P1: ``node`` is a name bound by the formula's own ``import math`` (see
        _ScopedHelperCalls); the base binder knows no lexical scopes and finds none."""
        return False

    def formula_name(self, node):
        """``node`` (a Name) is the formula's own parameter; the base binder knows no scopes."""
        return True

    def namespace(self, value):
        """The borrowed namespace argument of a math provider call: the module Reference itself, or
        the provider's own ``math`` module for a formula-local ``import math`` (the module that
        statement binds), so the provider takes its scalar path and keeps the Python fallback."""
        from .helper_members import MATH_PROVIDER
        if isinstance(value, cst.Name) and value.value in getattr(self, 'math_aliases', ()):
            return cst.Call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                            (cst.Arg(cst.Attribute(cst.Name('_mx_cy'), cst.Name('p_void'))),
                             cst.Arg(cst.Attribute(cst.Name(module_name(MATH_PROVIDER)), cst.Name('math_module')))))
        return borrowed_namespace(value)

    def decline(self, node, reason):
        from .reductions import untyped_entry
        self.math_untyped.append(untyped_entry(getattr(self, 'line_of', None), node, 'math', reason))

    def leave_Call(self, original, updated):
        math = self.module_math(original, updated)
        if math is not None:
            return math
        from .helper_members import MATH_PROVIDER
        func = updated.func
        ref = _declared_reference(func, self.refs)
        contract = ref.helper if ref is not None else None
        # A module member that is a math builtin under another name has no ``m_`` form: the
        # provider operation receives the member itself and checks it.
        if not contract or contract[0] != MATH_PROVIDER or any(arg.star or arg.keyword for arg in updated.args):
            return updated
        provider, operation = contract
        self.used.add(provider)
        self.operations.add(provider + '.' + operation)
        args = tuple(arg.with_changes(value=arg.value.with_changes(
                     lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),)))
                     if isinstance(arg.value, cst.GeneratorExp) and not arg.value.lpar else arg
                     for arg in updated.args)
        return updated.with_changes(
            func=cst.Attribute(cst.Name(module_name(provider)), cst.Name(operation)),
            args=(cst.Arg(func), *args))

    def rewrite(self, formula):
        """Bind registered symbols across lexical scopes, without body analysis.

        ScopeProvider identifies the actual receiver parameter. A lambda that
        shadows it, or a formula that rebinds it, retains its ordinary call.
        """
        binder = _ScopedHelperCalls(self.refs, self.used, self.cells, self.current)
        wrapper = cst.MetadataWrapper(cst.Module(body=(formula,)))
        binder.math_imports = _math_imports(wrapper.module.body[0])
        result = wrapper.visit(binder).body[0]
        if binder.math_lowered:
            result = _drop_unused_math_imports(result, binder.math_lowered)
        self.operations.update(binder.operations)
        self.math_native += binder.math_native
        self.math_untyped.extend(binder.math_untyped)
        return result


class _ScopedHelperCalls(HelperCalls):
    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider, cst.metadata.ParentNodeProvider,
                             cst.metadata.ExpressionContextProvider, cst.metadata.PositionProvider)

    def __init__(self, refs, used, cells, current=None):
        super().__init__(refs, used, cells, current)
        self.receiver = None
        self.first_line = None
        self.math_imports = {}          # diag-scale P1: top-level ``import math [as n]`` node -> name
        self.math_aliases = set()
        self.math_lowered = set()       # alias names with at least one lowered member
        self.formula_params = frozenset()   # ids of the formula's parameter nodes

    def local_math(self, node):
        """``node`` is a load of a name whose only binding in the formula is a top-level
        ``import math`` / ``import math as n`` of the formula, read after that statement: the name
        is then the ``math`` module (live runs the import first, or raises before it)."""
        if not isinstance(node, cst.Name) or not self.math_imports:
            return False
        if self.get_metadata(cst.metadata.ExpressionContextProvider, node, None) is not cst.metadata.ExpressionContext.LOAD:
            return False
        scope = self.get_metadata(cst.metadata.ScopeProvider, node, None)
        if scope is None:
            return False
        assignments = tuple(scope[node.value])
        if len(assignments) != 1 or assignments[0].node not in self.math_imports:
            return False
        statement = assignments[0].node
        if self.math_imports[statement] != node.value:
            return False
        use = self.get_metadata(cst.metadata.PositionProvider, node, None)
        bound = self.get_metadata(cst.metadata.PositionProvider, statement, None)
        if use is None or bound is None or use.start.line <= bound.end.line:
            return False
        self.math_aliases.add(node.value)
        return True

    def formula_name(self, node):
        """``node`` is read where its only binding is a parameter of the formula itself: a lambda or
        comprehension name that shadows a typed parameter has another type."""
        scope = self.get_metadata(cst.metadata.ScopeProvider, node, None)
        if scope is None:
            return False
        assignments = tuple(scope[node.value])
        return bool(assignments) and all(id(a.node) in self.formula_params for a in assignments)

    def line_of(self, node):
        position = self.get_metadata(cst.metadata.PositionProvider, node, None)
        if position is None or self.first_line is None:
            return 0
        return position.start.line - self.first_line + 1

    def on_visit(self, node):
        if self.receiver is None and isinstance(node, cst.FunctionDef):
            parameters = (*node.params.posonly_params, *node.params.params)
            if parameters and parameters[0].name.value == 'self':
                self.receiver = parameters[0]
            self.formula_params = frozenset(id(p) for p in (*parameters, *node.params.kwonly_params))
            position = self.get_metadata(cst.metadata.PositionProvider, node.name, None)
            self.first_line = position.start.line if position is not None else None
        return cst.CSTTransformer.on_visit(self, node)

    def on_leave(self, original_node, updated_node):
        return cst.CSTTransformer.on_leave(self, original_node, updated_node)

    def _owns_receiver(self, expression):
        while isinstance(expression, cst.Attribute):
            expression = expression.value
        if not isinstance(expression, cst.Name) or expression.value != 'self':
            return False
        scope = self.get_metadata(cst.metadata.ScopeProvider, expression, None)
        assignments = tuple(scope['self']) if scope is not None else ()
        return (len(assignments) == 1 and self.receiver is not None
                and assignments[0].node is self.receiver)

    def leave_Attribute(self, original, updated):
        if self.local_math(original.value):
            parent = self.get_metadata(cst.metadata.ParentNodeProvider, original)
            if not (isinstance(parent, cst.Call) and parent.func is original):
                constant = self.module_constant(updated, original)
                if constant is not None:
                    self.math_lowered.add(original.value.value)
                    return constant
            return updated
        if (isinstance(original.value, cst.Attribute) and self._owns_receiver(original)
                and self.get_metadata(cst.metadata.ExpressionContextProvider, original, None)
                is cst.metadata.ExpressionContext.LOAD):
            parent = self.get_metadata(cst.metadata.ParentNodeProvider, original)
            if not (isinstance(parent, cst.Call) and parent.func is original):
                constant = self.module_constant(updated)
                if constant is not None:
                    return constant
        if (original.attr.value not in self.cells
                or not isinstance(original.value, cst.Name)
                or not self._owns_receiver(original)
                or self.get_metadata(cst.metadata.ExpressionContextProvider, original, None)
                   is not cst.metadata.ExpressionContext.LOAD):
            return updated
        parent = self.get_metadata(cst.metadata.ParentNodeProvider, original)
        if isinstance(parent, cst.Call) and parent.func is original:
            return updated
        # A Cell used as a value needs its ordinary bound Python callable.
        # Cython must not coerce a native method pointer into an object tuple.
        # Direct calls retain the existing typed/override-aware Cell path.
        receiver = cst.Call(cst.Attribute(cst.Name(CY_MOD), cst.Name('cast')),
                            (cst.Arg(cst.Name('object')), cst.Arg(updated.value)))
        return updated.with_changes(value=receiver)

    def leave_Call(self, original, updated):
        function = original.func
        if isinstance(function, cst.Attribute) and self.local_math(function.value):
            lowered = self.module_math(original, updated)
            if lowered is not None and lowered is not updated:
                self.math_lowered.add(function.value.value)
                return lowered
            return updated
        if not self._owns_receiver(function):
            return updated
        if _declared_reference(function, self.refs) is None:
            self.module_math(original, updated)       # lists math members with no native form
            return updated
        return super().leave_Call(original, updated)


def _stdlib_math_members():
    """Member metadata of the ``math`` module, as for a Reference bound to it."""
    import math
    from .helper_members import module_helpers
    return module_helpers(math)


def _math_imports(formula):
    """diag-scale P1: the formula's top-level ``import math`` / ``import math as n`` statements
    (one alias each), mapped to the bound name. Conditional or nested imports are not included."""
    found = {}
    body = getattr(getattr(formula, 'body', None), 'body', ())
    for statement in body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small in statement.body:
            if (isinstance(small, cst.Import) and len(small.names) == 1
                    and isinstance(small.names[0].name, cst.Name) and small.names[0].name.value == 'math'):
                alias = small.names[0]
                found[small] = alias.asname.name.value if alias.asname else 'math'
    return found


def _drop_unused_math_imports(formula, names):
    """Remove a top-level ``import math`` whose name is no longer read anywhere in the formula
    (every member was lowered); an import whose name is still read stays as written."""
    from .scopes import CstFacts
    from .syntax import walk
    facts = CstFacts(formula.body.body)
    in_imports = {id(node) for statement in walk(formula.body) if isinstance(statement, (cst.Import, cst.ImportFrom))
                  for node in walk(statement)}
    reads = {node.value for node, context in facts.contexts.items()
             if isinstance(node, cst.Name) and context is cst.metadata.ExpressionContext.LOAD and id(node) not in in_imports}
    unused = {name for name in names if name not in reads}
    if not unused:
        return formula
    statements = []
    for statement in formula.body.body:
        if isinstance(statement, cst.SimpleStatementLine):
            kept = [small for small in statement.body
                    if not (isinstance(small, cst.Import) and len(small.names) == 1
                            and isinstance(small.names[0].name, cst.Name) and small.names[0].name.value == 'math'
                            and (small.names[0].asname.name.value if small.names[0].asname else 'math') in unused)]
            if not kept:
                continue
            if len(kept) != len(statement.body):
                kept[-1] = kept[-1].with_changes(semicolon=cst.MaybeSentinel.DEFAULT)
                statement = statement.with_changes(body=kept)
        statements.append(statement)
    if not statements:
        statements = [cst.SimpleStatementLine([cst.Pass()])]
    return formula.with_changes(body=formula.body.with_changes(body=statements))


def _declared_reference(function, refs):
    # Kept lazy to avoid the shared registry/resolver import cycle.
    from .helper_members import helper_reference
    return helper_reference(function, refs)


def install(providers, model_path):
    """Copy each provider's .pyx/.pxd pair into the model package; returns the .pyx paths to compile.
    A provider with only a .pxd is a header: the .pxd is copied and nothing is compiled for it."""
    modules = []
    for provider in sorted(providers):
        package, _, name = provider.rpartition('.')
        if not package:
            raise ValueError('helper provider must belong to a package')
        alias = module_name(provider)
        try:
            sources = {'.pyx': read_resource(package, name + '.pyx')}
        except FileNotFoundError:
            # a header: inline functions only, compiled into the modules that cimport it
            (model_path / (alias + '.pxd')).write_bytes(read_resource(package, name + '.pxd'))
            continue
        sources['.pxd'] = read_resource(package, name + '.pxd')
        for suffix, payload in sources.items():
            (model_path / (alias + suffix)).write_bytes(payload)
        modules.append(model_path.relative_to(model_path.parent) / (alias + '.pyx'))
    return modules
