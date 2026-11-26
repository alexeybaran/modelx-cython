"""Python access to compiled vector cells.

A cell whose C result is a typed vector (``vectors.py``), or a tuple of them, has two members in its
Space class:

- its cached C entry, renamed ``_mx_cell_<name>``: compiled code calls it and passes the typed vector
  without a copy. Every call of the cell in the module whose value compiled code consumes natively is
  renamed with it;
- the public method under the cell's own name, which returns a new list (a tuple of new lists) of the
  values, as live returns a list. Python callers, and every use of the cell's value in Python code inside
  a compiled formula, go through it, so changing the list they get never reaches the cache.

A name is renamed only when no other class of the module has a member of that name.

The pass runs last over a generated module (``finish``), after every pass that looks cells up by name.
"""
import re

import libcst as cst
from libcst.metadata import ParentNodeProvider

from modelx_cython.consts import CY_MOD

CELL_PREF = '_mx_cell_'


def vector_cells(cls_info):
    """``{name: shape}`` of the cells whose C result is a typed vector or a tuple of them: compiled
    code passes the vector itself; a Python caller gets a new list (a tuple of new lists), never the cache."""
    result = {}
    for name, cells in cls_info.cells.items():
        if cells.is_special() or cells.formula_is_generator or not cells.has_typeinfo():
            continue        # a keyword call or a closure keeps its entry: its public method forwards
        shape = cells.vector_shape
        if shape in ('V', 'I') or (isinstance(shape, tuple) and shape and shape[0] == 'T'):
            result[name] = shape
    return result


def _members(cls_info):
    return set(cls_info.cells) | set(cls_info.refs) | set(cls_info.spaces)


def renames(module):
    """``{class name: {member: renamed C member}}`` for the module, conflicts left out."""
    owners = {}
    for cls_name, cls_info in module.classes.items():
        for name in _members(cls_info):
            owners.setdefault(name, set()).add(cls_name)
    result = {}
    for cls_name, cls_info in module.classes.items():
        mapping = {}
        for name in vector_cells(cls_info):
            if owners[name] == {cls_name}:
                mapping[name] = CELL_PREF + name
        result[cls_name] = mapping
    return result


def module_renames(module):
    cached = getattr(module, '_mx_vector_renames', None)
    if cached is None:
        cached = module._mx_vector_renames = renames(module)
    return cached


def public_python(cells):
    """Whether the cell's public entry is a Python method (as the generated class keeps it)."""
    return cells.body_has_closure or cells.called_with_kwargs


def pxd_entry(cls_info, cells):
    """The name of a cell's C entry in the .pxd: renamed for a vector cell."""
    return module_renames(cls_info.module).get(cls_info.name, {}).get(cells.name, cells.name)


def pxd_publics(cls_info, parameters):
    """The .pxd lines of the cpdef public methods of the vector cells, in the order ``rewrite_class`` adds
    them after every other method (``parameters(cells)``: the parameter text of a cell)."""
    mapping = module_renames(cls_info.module).get(cls_info.name, {})
    return [f"cpdef object {name}({parameters(cls_info.cells[name])})\n"
            for name in vector_cells(cls_info) if name in mapping and not public_python(cls_info.cells[name])
            and cls_info.cells[name].has_formula_def]


def _statements(text, config):
    return cst.parse_module(text, config=config).body


def rewrite_class(cls_info, node, config):
    """Rename the C entries of the vector cells of a generated Space class and add their public methods."""
    mapping = module_renames(cls_info.module)[cls_info.name]
    vectors = {n: v for n, v in vector_cells(cls_info).items() if n in mapping}
    if not vectors:
        return node
    body = []
    entries = {}
    for statement in node.body.body:
        if isinstance(statement, cst.FunctionDef):
            name = statement.name.value
            if name in vectors:
                entries[name] = statement
                statement = statement.with_changes(name=cst.Name(mapping[name]))
        body.append(statement)

    added = []
    for name, shape in vectors.items():
        entry = entries.get(name)
        if entry is None:
            continue
        cell = cls_info.cells[name]
        params = ', '.join(cell.params)
        value = f'self.{mapping[name]}({params})'
        result = (f'tuple([v.tolist() for v in {value}])' if isinstance(shape, tuple) else f'{value}.tolist()')
        # the entry's own parameters, annotations and defaults included (a keyword call, a cpdef signature)
        declared = cst.Module(body=[]).code_for_node(entry.params)
        decorator = '' if public_python(cell) else f'@{CY_MOD}.ccall\n'
        returns = '' if public_python(cell) else ' -> object'
        text = (f'{decorator}def {name}({declared}){returns}:\n'
                f'    """A new list of the values, as live returns a list; compiled code calls '
                f'``{mapping[name]}``, which passes the typed vector without a copy."""\n'
                f'    return {result}\n')
        for statement in _statements(text, config):
            added.append(statement.with_changes(leading_lines=(cst.EmptyLine(),)))
    return node.with_changes(body=node.body.with_changes(body=(*body, *added)))


def _lines(lines):
    """The simple statement lines of a function body, nested blocks included (not nested functions or classes)."""
    found = []

    class Lines(cst.CSTVisitor):
        def visit_SimpleStatementLine(self, node):
            found.append(node)
            return False

        def visit_SimpleStatementSuite(self, node):
            found.append(node)
            return False

        def visit_FunctionDef(self, node):
            return False

        def visit_ClassDef(self, node):
            return False

    for line in lines:
        line.visit(Lines())
    return found


class RenameMembers(cst.CSTTransformer):
    """Rename the C entries of vector cells in a whole generated module.

    The last pass over a generated module: it also adds the members of ``rewrite_class`` to each
    Space class, after every other pass has run on the cells under their public names.

    ``x.<cell>(...)`` becomes ``x._mx_cell_<cell>(...)`` only where the receiver ``x`` is known to be an
    instance of the class that owns the member: ``self`` in its class, a parameter or local annotated or cast
    to the class (``base_: _c_X``, ``self: _mx_private_receiver[_c_X]``, ``_mx_cy.cast(_c_X, ...)``), or a
    Reference to a Space declared with that class (``self.data`` with ``data: _c_Data``), recursively, and
    only where compiled code consumes the typed vector (``_vector_native``). Any other call keeps the public
    name, which returns a new list.
    """
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, module, config=None):
        self.module = module
        self.config = config
        self.mapping = module_renames(module)
        self.vectors = {cls_name: vector_cells(cls_info) for cls_name, cls_info in module.classes.items()}
        self.vector_publics = 0     # vector cell calls left on the public method (a new list)
        self._classes = []          # enclosing generated class names
        self._locals = []           # per function: {local name: class name}
        self._vector_locals = []    # per function: local names declared as typed vectors
        self._functions = []        # enclosing FunctionDef nodes

    # ---------------------------------------------------------------- receivers
    @staticmethod
    def _class_name(expr):
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return expr.attr.value
        if isinstance(expr, cst.Subscript) and len(expr.slice) == 1:
            element = expr.slice[0].slice
            if isinstance(element, cst.Index):
                return RenameMembers._class_name(element.value)
        if isinstance(expr, cst.SimpleString):
            return expr.value.strip('\'"').rsplit('.', 1)[-1]
        return None

    def _known(self, name):
        return name if name in self.module.classes else None

    def _ref_class(self, cls_name, attr):
        cls_info = self.module.classes.get(cls_name)
        ref = cls_info.refs.get(attr) if cls_info is not None else None
        decl = getattr(ref, 'decl_type_expr', None)
        return self._known(decl.rsplit('.', 1)[-1]) if decl else None

    def _cast_class(self, call):
        func = call.func
        if (isinstance(func, cst.Attribute) and func.attr.value == 'cast'
                and isinstance(func.value, cst.Name) and func.value.value == CY_MOD and call.args):
            return self._known(self._class_name(call.args[0].value))
        return None

    def _receiver(self, expr):
        if isinstance(expr, cst.Name):
            if self._locals and expr.value in self._locals[-1]:
                return self._locals[-1][expr.value]
            if expr.value == 'self' and self._classes:
                return self._classes[-1]
            return None
        if isinstance(expr, cst.Attribute):
            owner = self._receiver(expr.value)
            return self._ref_class(owner, expr.attr.value) if owner else None
        if isinstance(expr, cst.Call):
            return self._cast_class(expr)
        return None

    # ---------------------------------------------------------------- scopes
    def visit_ClassDef(self, node):
        self._classes.append(node.name.value if node.name.value in self.module.classes else None)

    def leave_ClassDef(self, original, updated):
        name = self._classes.pop()
        if name is not None and self.config is not None and not self._classes:
            # the last pass: every earlier pass has seen the cells under their own names
            updated = rewrite_class(self.module.classes[name], updated, self.config)
        return updated

    def visit_FunctionDef(self, node):
        types_ = {}
        for param in (*node.params.params, *node.params.posonly_params, *node.params.kwonly_params):
            if param.annotation is not None:
                cls = self._known(self._class_name(param.annotation.annotation))
                if cls:
                    types_[param.name.value] = cls
        lines = node.body.body if isinstance(node.body, cst.IndentedBlock) else (node.body,)
        for line in lines:
            if not isinstance(line, (cst.SimpleStatementLine, cst.SimpleStatementSuite)):
                continue
            for small in line.body:
                if isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name):
                    cls = self._known(self._class_name(small.annotation.annotation))
                    if cls is None and isinstance(small.value, cst.Call):
                        cls = self._cast_class(small.value)
                    if cls:
                        types_[small.target.value] = cls
                elif (isinstance(small, cst.Assign) and len(small.targets) == 1
                      and isinstance(small.targets[0].target, cst.Name) and isinstance(small.value, cst.Call)):
                    cls = self._cast_class(small.value)
                    if cls:
                        types_[small.targets[0].target.value] = cls
        self._locals.append(types_)
        from modelx_cython.vectors import TYPES
        vector_types = set(TYPES.values())
        # declared at any depth of the function (the integer guard puts a formula's body in a try)
        self._vector_locals.append({small.target.value for line in _lines(lines)
            if isinstance(line, (cst.SimpleStatementLine, cst.SimpleStatementSuite)) for small in line.body
            if isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name)
            and cst.Module(body=[]).code_for_node(small.annotation.annotation) in vector_types})
        self._functions.append(node)

    def leave_FunctionDef(self, original, updated):
        self._locals.pop()
        self._vector_locals.pop()
        self._functions.pop()
        return updated

    # ---------------------------------------------------------------- vector results
    def _parent(self, node):
        return self.get_metadata(ParentNodeProvider, node, None)

    def _vector_return(self, shape):
        """Whether the enclosing function returns the typed vector (or tuple of them) of ``shape``."""
        from modelx_cython.vectors import TYPES
        if not self._functions or self._functions[-1].returns is None:
            return False
        function = self._functions[-1]
        annotation = cst.Module(body=[]).code_for_node(function.returns.annotation)
        if isinstance(shape, tuple):
            own = self.vectors.get(self._classes[-1] if self._classes else None, {})
            name = function.name.value
            for prefix in ('_f_', CELL_PREF):
                if name.startswith(prefix) and own.get(name[len(prefix):]) == shape:
                    return True
            return False
        return annotation == TYPES.get(shape)

    def _vector_targets(self, targets):
        names = self._vector_locals[-1] if self._vector_locals else set()

        def typed(t):
            if isinstance(t, cst.Name):     # a typed vector local, or a temporary of the lowering
                return t.value in names or re.match(r'_mx_v[0-9]+_', t.value) is not None
            if isinstance(t, cst.Tuple):
                return all(typed(e.value) for e in t.elements)
            return False
        return all(typed(t) for t in targets)

    def _vector_native(self, node, shape):
        """Whether compiled code consumes the typed vector that ``node`` (a vector cell call) gives.

        Native uses: a provider operation's argument, ``.n`` / ``.data`` / ``.tolist()``, ``is (not) None``, a
        typed vector local assigned from it, the return of a function typed to return it, and the elements of
        a tuple of vectors so used. Everywhere else the call stays on the public method, which returns a new
        list: the value Python code sees there is the list live builds, whatever the receiver."""
        from modelx_cython.vectors import ALIAS
        parent = self._parent(node)
        if isinstance(parent, cst.Attribute) and parent.value is node:
            return parent.attr.value in ('n', 'data', 'tolist')
        if isinstance(parent, cst.Arg):
            call = self._parent(parent)
            return (isinstance(call, cst.Call) and isinstance(call.func, cst.Attribute)
                    and isinstance(call.func.value, cst.Name) and call.func.value.value == ALIAS)
        if isinstance(parent, cst.Comparison) and parent.left is node:
            return all(isinstance(c.operator, (cst.Is, cst.IsNot)) and isinstance(c.comparator, cst.Name)
                       and c.comparator.value == 'None' for c in parent.comparisons)
        if isinstance(parent, cst.Subscript) and parent.value is node and isinstance(shape, tuple):
            index = parent.slice[0].slice if len(parent.slice) == 1 else None
            value = getattr(index, 'value', None)
            if not isinstance(value, cst.Integer) or not -len(shape[1]) <= int(value.value) < len(shape[1]):
                return False
            return self._vector_native(parent, shape[1][int(value.value)])
        if isinstance(parent, cst.Element):
            outer = self._parent(parent)
            above = self._parent(outer) if isinstance(outer, cst.Tuple) else None
            if isinstance(above, cst.Return):
                own = self.vectors.get(self._classes[-1] if self._classes else None, {})
                return any(isinstance(k, tuple) and self._vector_return(k) for k in own.values())
            if isinstance(above, cst.Assign) and above.value is outer:
                return all(isinstance(t.target, cst.Tuple) for t in above.targets) and self._vector_targets(
                    [t.target for t in above.targets])
            return False
        if isinstance(parent, cst.IfExp) and parent.test is not node:
            return self._vector_native(parent, shape)
        if isinstance(parent, cst.CompFor) and parent.iter is node:
            # _ObjectUses' own conversion of a tuple of vectors: [_mx_vector.tolist() for _mx_vector in <call>]
            return isinstance(parent.target, cst.Name) and parent.target.value == '_mx_vector'
        if isinstance(parent, cst.Assign) and parent.value is node:
            return self._vector_targets([t.target for t in parent.targets])
        if isinstance(parent, cst.AnnAssign) and parent.value is node:
            return self._vector_targets([parent.target])
        if isinstance(parent, cst.Return):
            return self._vector_return(shape)
        return False

    # ---------------------------------------------------------------- renames
    def _new(self, receiver, name, prefix):
        owner = self._receiver(receiver)
        new = self.mapping.get(owner, {}).get(name) if owner else None
        return new if new is not None and new.startswith(prefix) else None

    def leave_Call(self, original, updated):
        func = original.func
        if isinstance(func, cst.Attribute):
            new = self._new(func.value, func.attr.value, CELL_PREF)
            shape = self.vectors.get(self._receiver(func.value), {}).get(func.attr.value) if new else None
            if new and shape is not None and not self._vector_native(original, shape):
                self.vector_publics += 1
                return updated      # Python code: the public method's new list
            if new:
                updated = updated.with_changes(func=updated.func.with_changes(attr=cst.Name(new)))
        return updated


def finish(syntax, module):
    """The last pass over a generated module (after entry lowering and integer guards): rename the C entries
    of the vector cells and add their public methods. A module without vector cells is returned as it is."""
    if not any(module_renames(module).values()):
        return syntax
    return cst.MetadataWrapper(syntax, unsafe_skip_copy=True).visit(
        RenameMembers(module, syntax.config_for_parsing))
