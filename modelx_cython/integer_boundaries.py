"""Check integer Cell results and ItemSpace keys at their native boundaries."""

import libcst as cst
from libcst.metadata.scope_provider import BuiltinAssignment

from .integer_arithmetic import ModuleTypes, lower_integer_arithmetic
from .consts import CY_MOD, FORMULA_PREF, MX_SELF, MX_SYS_MOD
from .scopes import CstFacts, lexical_nodes
from .syntax import arguments, walk
from .usage import RESOLVED, _extract_self_chain


_INTEGER_ANNOTATION = cst.Attribute(cst.Name(CY_MOD), cst.Name('longlong'))


_CALL_GATE = """\
@{cy}.cclass
class {gate}:
    # D110 P0: a Python-bound Cell with integer arguments, checked as before, then the compiled method;
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


class _GuardCalls(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider,)

    def __init__(self, receiver, integer_params, owner, resolver, entries, cells, private_entries, safe_builtins=()):
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


def _guard_calls(node, cls, resolver, entries, cells, private_entries, safe_builtins=()):
    params = (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)
    if not params or params[0].name.value != MX_SELF:
        return node
    integer_params = {param.name.value: param for param in params if param.annotation is not None
                      and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    wrapper = cst.MetadataWrapper(cst.Module(body=(node,)), unsafe_skip_copy=True)
    return wrapper.visit(_GuardCalls(params[0], integer_params, cls.fqname, resolver, entries, cells,
                                    private_entries, safe_builtins)).body[0]


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


def _python_locals(node, cell):
    """D72/D76: declare ``object`` the locals compared by identity.

    Cython gives a C type to an untyped local it can prove one (a C integer never used in
    arithmetic, a double); a local compared with ``is``/``is not`` would lose its identity
    (``x is saved[0]`` on a re-boxed int gives a wrong answer), so it is declared ``object``.
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


def _guard_formula(node, cls, cell, resolver, entries, cells, private_entries, safe_builtins, types, owner):
    """Integer arguments, arithmetic and results of one formula function (a method of ``owner`` or,
    with ``owner`` None, a module function whose receiver parameter is annotated)."""
    node = _guard_calls(node, cls, resolver, entries, cells, private_entries, safe_builtins)
    # a dense cache's parameters range over its domain: every entry checks it before the formula
    bounds = ({axis.key: (axis.lower, axis.stop) for axis in cell.dense_logical_axes}
              if cell.has_formula_def and cell.has_args() and cell.has_typeinfo() and cell.is_arrayable()
              else {})
    node = lower_integer_arithmetic(node, types, owner, cell.model_path, bounds)
    return _python_locals(node, cell)


def guard_integer_boundaries(state, module_info, resolver, cells):
    """Check formula results, arguments and ItemSpace keys after entry lowering, and keep C integer
    arithmetic only where it cannot overflow (:mod:`.integer_arithmetic`)."""
    entries = {name: params for name, cell in cells.items() if (params := integer_parameters(cell))}
    private_entries = {
        name: (cell.model_path, params)
        for cls in module_info.classes.values() for cell in cls.cells.values()
        if (name := getattr(cell, 'inline_entry', None)) and (params := entries.get(cell.fqname))
    }
    private_entries.update({name: private_entries[checked] for name, checked in state.inbounds_entries.items()
                            if checked in private_entries})
    types = ModuleTypes(state.syntax)
    classes = []
    declarations = []
    call_gate = state.names.reserve('_mx_checked_cell_callable')
    gate_class = state.names.reserve('_mx_checked_cell_call')
    changed = False
    global_bindings = cst.MetadataWrapper(state.syntax, unsafe_skip_copy=True).resolve(
        cst.metadata.ScopeProvider)[state.syntax]
    safe_builtins = {name for name in ('isinstance', 'int', 'bool')
                     if all(isinstance(binding, BuiltinAssignment) for binding in global_bindings[name])}

    def formula(node, cls, cell, owner):
        return _guard_formula(node, cls, cell, resolver, entries, cells, private_entries, safe_builtins, types,
                              owner)

    for node in state.syntax.body:
        cls = module_info.classes.get(node.name.value) if isinstance(node, cst.ClassDef) else None
        if cls is None:
            origin = state.function_origins.get(node.name.value) if isinstance(node, cst.FunctionDef) else None
            if origin:
                owner, name = origin
                cls = module_info.classes[owner]
                # D139: the private entry of an uncached Cell has the Cell's own name as its origin
                node = formula(node, cls, cls.cells[name[len(FORMULA_PREF):] if name.startswith(FORMULA_PREF)
                                                    else name], None)
                changed = True
            classes.append(node)
            continue
        methods = []
        for method in node.body.body:
            if not isinstance(method, cst.FunctionDef):
                methods.append(method)
                continue
            name = method.name.value
            cell = cls.cells.get(name)
            if name.startswith(FORMULA_PREF) and (cell := cls.cells.get(name[len(FORMULA_PREF):])):
                method = formula(method, cls, cell, node.name.value)
                changed = True
            elif name == '__call__' and cell is not None and cell.has_typeinfo():
                method = _guard_space_parameters(method, cell)
                changed = True
            elif cell is not None and not cell.has_formula_def:
                method = formula(method, cls, cell, node.name.value)
                changed = True
            methods.append(method)
        added, gates = _python_call_gate(cls, entries, state.names, call_gate)
        declarations.extend(added)
        changed |= bool(gates)
        classes.append(node.with_changes(body=node.body.with_changes(body=(*methods, *gates))))
    if declarations:
        classes.extend(declarations)
        classes.extend(cst.parse_module(_CALL_GATE.format(
            cy=CY_MOD, name=call_gate, gate=gate_class, runtime=MX_SYS_MOD)).body)
    return state.replace(state.syntax.with_changes(body=classes)) if changed else state
