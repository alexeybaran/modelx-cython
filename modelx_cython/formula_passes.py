"""Formula passes after entry lowering: Python's semantics where Cython's C typing would change them
(an unbound local, a local compared by identity), and the Python entry of Cells with integer
parameters."""

import libcst as cst
from libcst.metadata.scope_provider import BuiltinAssignment

from .consts import CY_MOD, FORMULA_PREF, MX_SELF, MX_SYS_MOD
from .scopes import CstFacts, lexical_nodes, maybe_unbound
from .syntax import arguments, walk


_INTEGER_ANNOTATION = cst.Attribute(cst.Name(CY_MOD), cst.Name('longlong'))
# The builtins the isinstance folding reads when the module leaves them unbound
_FOLDED_BUILTINS = ('isinstance', 'int', 'bool')


def integer_parameters(cell):
    if cell.is_special() or cell.body_has_closure or not cell.has_typeinfo():
        return ()
    return tuple((index, name) for index, name in enumerate(cell.params)
                 if cell.get_argtype_expr(name, c_style=True) == 'long long')



def _python_call_gate(cls, entries, names, methods):
    """The Python entry of a Cell with integer parameters, keeping its Cython vtable type.

    Each such Cell gets a Python entry with the Cell's own parameters and defaults: it calls the
    compiled method at C level, Cython converting each integer argument to its C parameter type. ``__getattribute__`` returns the entry for the
    Cell's name, unless a subclass overrides the method; the entry carries the Cell's ``__name__``,
    ``__qualname__``, ``__doc__`` and the method as ``__wrapped__``."""
    nodes = {item.name.value: item for item in methods if isinstance(item, cst.FunctionDef)}
    routes = {cell.name: entries[cell.fqname]
              for cell in cls.cells.values() if cell.fqname in entries and cell.name in nodes}
    if not routes:
        return (), ()
    field = names.reserve(f'_mx_integer_arguments_{cls.name}')
    gate = {name: names.reserve(f'_mx_python_entry_{name}') for name in routes}
    added, defs = [cst.parse_statement(f'{field} = {CY_MOD}.declare(dict, {gate!r})\n')], []   # a C global
    for name, params in routes.items():
        signature = nodes[name].params
        plain = signature.with_changes(params=tuple(p.with_changes(annotation=None) for p in signature.params))
        checked = dict(params)
        names_ = [p.name.value for p in signature.params[1:]]
        args = ', '.join(names_)
        header = cst.Module(body=()).code_for_node(plain)
        # The C method without Cython's override dispatch (``Class.method(self, ...)``), which would read the
        # attribute back through __getattribute__ (a class with a __dict__, a Python subclass), called from a
        # module function whose parameters no Cell parameter can shadow
        call = names.reserve(f'_mx_python_call_{cls.name}_{name}')
        slots = ', '.join(f'_mx_a{index}: {CY_MOD}.longlong' if index in checked else f'_mx_a{index}'
                          for index in range(len(names_)))
        added.append(cst.parse_statement(
            f'@{CY_MOD}.cfunc\n@{CY_MOD}.inline\ndef {call}(self: {cls.name}, {slots}) -> object:\n'
            f'    return {cls.name}.{name}(self, {", ".join(f"_mx_a{i}" for i in range(len(names_)))})\n'))
        defs.append(cst.parse_statement(f'def {gate[name]}({header}):\n    return {call}(self, {args})\n'))
        owner = f'{CY_MOD}.cast(object, {cls.name})'      # the Python type, not the C methods
        entry = f'{owner}.{gate[name]}'
        added.extend(cst.parse_module(
            f'{entry}.__name__ = {name!r}\n{entry}.__qualname__ = {cls.name + "." + name!r}\n'
            f'{entry}.__doc__ = {owner}.{name}.__doc__\n{entry}.__wrapped__ = {owner}.{name}\n').body)
    defs.append(cst.parse_statement(f"""\
def __getattribute__(self, name):
    entry = {field}.get(name)
    if entry is not None and (
        type(self) is {cls.name}
        or getattr(type(self), name) is getattr({CY_MOD}.cast(object, {cls.name}), name)
    ):
        name = entry
    return {MX_SYS_MOD}._mx_generic_getattr(self, name)
"""))
    return tuple(added), tuple(defs)



def _unbound_locals(node):
    """A local that a read can reach before any assignment, on some path, is a Python object: Python
    raises UnboundLocalError there (a loop that ran no iteration, a branch not taken), and a C local
    has no unbound state (with type inference, or a declaration of the vector lowering, Cython would
    read an unset value)."""
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



class _FoldIsinstance(cst.CSTTransformer):
    """``isinstance(t, int)`` of a C integer parameter ``t`` folded to its value."""
    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider,)

    def __init__(self, integer_params, safe_builtins=()):
        self.integer_params = integer_params
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
        return after if folded is None else folded

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


def _fold_isinstance(node, safe_builtins=(), module=None):
    """``module``: a MetadataWrapper whose module holds ``node`` itself, its scopes resolved.

    The pass asks scopes about the formula's parameters and about the builtins it folds, which the
    module binds nowhere (``safe_builtins``). For those names the formula's scopes in its module
    answer as in a module holding the formula alone, unless the formula is itself named as one of
    the builtins.
    """
    params = (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)
    if not params or params[0].name.value != MX_SELF:
        return node
    integer_params = {param.name.value: param for param in params if param.annotation is not None
                      and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    visitor = _FoldIsinstance(integer_params, safe_builtins)
    if module is not None and node.name.value not in _FOLDED_BUILTINS:
        with visitor.resolve(module):
            return node.visit(visitor)
    wrapper = cst.MetadataWrapper(cst.Module(body=(node,)), unsafe_skip_copy=True)
    return wrapper.visit(visitor).body[0]


def _python_locals(node, cell):
    """Declare ``object`` the locals compared by identity.

    With ``infer_types`` Cython infers a C type for every untyped local; a local
    compared with ``is``/``is not`` would lose its identity (``x is saved[0]`` on a re-boxed
    int gives a wrong answer), so it is declared ``object``. Integer locals and ``range``
    targets otherwise stay C integers.
    Locals with an annotation of their own are left alone. The admission report lists each
    declared local (``object_locals``).
    """
    body = node.body
    # a binding construct leaves the formula as it is
    effects = (cst.Global, cst.Nonlocal, cst.Import, cst.ImportFrom, cst.Del, cst.NamedExpr,
               cst.FunctionDef, cst.ClassDef, cst.Lambda)
    if any(
            isinstance(part, effects) or isinstance(part, getattr(cst, 'Match', ()))
            for part in lexical_nodes(body.body)):
        return node
    facts = CstFacts(body.body)
    parameters = {parameter.name.value for parameter in
                  (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)}
    annotated = {statement.target.value for statement in lexical_nodes(body.body)
                 if isinstance(statement, cst.AnnAssign) and isinstance(statement.target, cst.Name)}
    # Integer locals and range targets stay C integers; only identity keeps the Python object
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



def lower_formula_semantics(state, module_info, cells, infer_types=False):
    """After entry lowering: unbound locals and (``infer_types``) locals compared by identity stay
    Python objects, ``isinstance`` of a C integer parameter is folded, and each Cell with integer
    parameters gets its Python entry."""
    entries = {name: params for name, cell in cells.items() if (params := integer_parameters(cell))}
    classes = []
    declarations = []
    changed = False
    module_scopes = cst.MetadataWrapper(state.syntax, unsafe_skip_copy=True)
    global_bindings = module_scopes.resolve(cst.metadata.ScopeProvider)[state.syntax]
    safe_builtins = {name for name in _FOLDED_BUILTINS
                     if all(isinstance(binding, BuiltinAssignment) for binding in global_bindings[name])}

    def lower(node, cell):
        unbound = _unbound_locals(node)
        node = _fold_isinstance(unbound, safe_builtins, module_scopes if unbound is node else None)
        return _python_locals(node, cell) if infer_types else node

    for node in state.syntax.body:
        cls = module_info.classes.get(node.name.value) if isinstance(node, cst.ClassDef) else None
        if cls is None:
            origin = state.function_origins.get(node.name.value) if isinstance(node, cst.FunctionDef) else None
            if origin:
                owner, name = origin
                cls = module_info.classes[owner]
                # The private entry of an uncached Cell has the Cell's own name as its origin
                node = lower(node, cls.cells[name[len(FORMULA_PREF):] if name.startswith(FORMULA_PREF) else name])
                changed = True
            classes.append(node)
            continue
        methods = []
        for method in node.body.body:
            if isinstance(method, cst.FunctionDef):
                name = method.name.value
                cell = (cls.cells.get(name[len(FORMULA_PREF):]) if name.startswith(FORMULA_PREF)
                        else cls.cells.get(name))
                if cell is not None and (name.startswith(FORMULA_PREF) or not cell.has_formula_def):
                    method = lower(method, cell)
                    changed = True
            methods.append(method)
        added, gates = _python_call_gate(cls, entries, state.names, methods)
        declarations.extend(added)
        changed |= bool(gates)
        classes.append(node.with_changes(body=node.body.with_changes(body=(*methods, *gates))))
    classes.extend(declarations)
    return state.replace(state.syntax.with_changes(body=classes)) if changed else state
