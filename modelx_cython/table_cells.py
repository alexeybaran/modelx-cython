"""Python access to compiled table cells (api-tables, D101).

Generated members, per Space class, by name only (no ``__getattribute__``/``__setattr__``
override is added):

- a zero-argument cell whose value is a table (a pandas frame, or a native table) keeps its cached
  value; its C entry is renamed ``_mx_cell_<name>`` (D101a) and every call of it in the module is
  renamed with it, so native code calls the C entry directly. Cython requires each C member of a
  cclass to be declared in the ``.pxd`` and rejects a Python property of the same name, and an
  extension type is immutable, so the public name is freed by the rename. The property of the
  public name returns a ``runtime.table_cells.TableCell``, whose call gives the value;
- a name is renamed only when no other class of the module has a member of that name.

As upstream modelx-cython, a compiled model has no clear and no table-cell assignment; its References are
plain ``cdef public`` attributes (integer References keep their generated ``__setattr__``).

The runtime side is ``modelx_cython.runtime.table_cells``.
"""

import re
import types

import libcst as cst
from libcst.metadata import ParentNodeProvider

from modelx_cython.consts import CY_MOD, MX_SYS_MOD, SPACE_PREF, MODULE_PREF

RUNTIME = '_mx_table_cells'
STANDARD_PROVIDER = 'modelx_cython.runtime._native_inputs'
IMPORT = f'from modelx_cython.runtime import table_cells as {RUNTIME}\n'


def table_cells(cls_info):
    """Names of the zero-argument cells whose value is bound to a native table."""
    result = {}
    for name, cells in cls_info.cells.items():
        if (cells.is_special() or cells.has_args() or not cells.has_formula_def
                or cells.body_has_closure or cells.called_with_kwargs or cells.formula_is_generator):
            continue
        binder = cells.native_result_binder
        if binder or frame_cell(cls_info, cells):
            result[name] = cells
    return result


class _BindFrame(cst.CSTTransformer):
    """T-4/T-4.2: in the C entry of a frame cell, prepare the computed frame where it is stored (``self._v_<name> =
    val``), under the cell's lock, and publish the prepared table in ``_mx_ptable_<name>`` (then its flag
    ``_mx_ptready_<name>``, released as the cell's own ``_has_`` flag is): a table is prepared when it is bound,
    and an item binds it with one field read (``_mx_prepared_<name>``)."""

    def __init__(self, name, locked):
        super().__init__()
        self.name, self.locked = name, locked
        self.target = f'self._v_{name}'

    def leave_IndentedBlock(self, original, updated):
        body = []
        for statement in updated.body:
            if (isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1
                    and isinstance(statement.body[0], cst.Assign)
                    and cst.Module(body=()).code_for_node(statement.body[0].targets[0].target) == self.target):
                body.extend(_statements(_publish(self.name, 'val', self.locked), cst.Module(body=()).config_for_parsing))
            body.append(statement)
        return updated.with_changes(body=body)


def _publish(name, value, locked):
    """Statements that prepare ``value`` and publish it as the frame cell's prepared table (T-4.2)."""
    flag = (f'{MX_SYS_MOD}._mx_store_flag({CY_MOD}.address(self._mx_ptready_{name}), True)' if locked
            else f'self._mx_ptready_{name} = True')
    return f'self._mx_ptable_{name} = {RUNTIME}.bind_frame(self, {value})\n{flag}\n'


def _prepared_method(name, mapping, locked):
    """``_mx_prepared_<name>()``: the prepared table of a frame cell for the items that read it (T-4.2). The fast
    path reads the published flag (acquire) and the field; else the cell is computed, which publishes it."""
    if locked:
        ready = f'{MX_SYS_MOD}._mx_load_flag({CY_MOD}.address(self._mx_ptready_{name}))'
        return (f'@{CY_MOD}.cfunc\n'
                f'def _mx_prepared_{name}(self) -> object:\n'
                f'    if {ready}:\n'
                f'        table = self._mx_ptable_{name}     # read once: a clear may reset the field meanwhile\n'
                f'        if table is not None:\n'
                f'            return table\n'
                f'    value = self.{mapping[name]}()\n'
                f'    with self._mx_lock:\n'
                f'        if not self._mx_ptready_{name}:      # cleared since it was bound: bind the frame again\n'
                f'            ' + _publish(name, 'value', True).replace('\n', '\n            ').rstrip() + '\n'
                f'        return self._mx_ptable_{name}\n')
    return (f'@{CY_MOD}.cfunc\n'
            f'def _mx_prepared_{name}(self) -> object:\n'
            f'    if not self._mx_ptready_{name}:\n'
            f'        value = self.{mapping[name]}()\n'
            f'        if not self._mx_ptready_{name}:\n'
            f'            ' + _publish(name, 'value', False).replace('\n', '\n            ').rstrip() + '\n'
            f'    return self._mx_ptable_{name}\n')


def data_frame_cells(cls_info):
    """T-4.2: the frame cells of a class without items whose prepared table is held in the class (renamed cells)."""
    if cls_info.name in cls_info.visitor.itemspace_classes:
        return {}
    mapping = module_renames(cls_info.module)[cls_info.name]
    return {n: c for n, c in table_cells(cls_info).items() if n in mapping and frame_cell(cls_info, c)}


def item_frame_cells(cls_info):
    """T-4.2: the frame cells of an item class (a frame each item computes for itself)."""
    if cls_info.name not in cls_info.visitor.itemspace_classes:
        return {}
    mapping = module_renames(cls_info.module)[cls_info.name]
    return {n: c for n, c in table_cells(cls_info).items() if n in mapping and frame_cell(cls_info, c)}


def pxd_declarations(cls_info):
    """The .pxd lines of the T-4.2 members of a class."""
    lines = []
    for name in data_frame_cells(cls_info):
        lines += [f'cdef object _mx_ptable_{name}\n', f'cdef bint _mx_ptready_{name}\n',
                  f'cdef object _mx_prepared_{name}(self)\n']
    return lines


def frame_cell(cls_info, cells):
    """T-3: a zero-argument cell whose traced value is a pandas DataFrame or Series, compiled with prepared
    tables: it is a table cell whose value is the frame itself (assigned as it is, never converted)."""
    policy = getattr(getattr(getattr(cls_info, 'module', None), 'spec', None), 'policy', None)
    return (bool(getattr(policy, 'prepared_tables', False)) and not cells.native_result_binder
            and getattr(getattr(cells, '_rt', None), 'frame_schema', None) is not None)


def record_cells(cls_info):
    """Names of the zero-argument cells whose value is a row of a table admitted from a frame (D119):
    compiled code reads the Record; a Python caller gets the read-only pandas row, as live."""
    result = {}
    for name, cells in cls_info.cells.items():
        if (cells.is_special() or cells.has_args() or not cells.has_formula_def or cells.body_has_closure
                or cells.called_with_kwargs or cells.formula_is_generator or cells.native_result_binder):
            continue
        if getattr(cells, 'frame_row_result', False):
            result[name] = cells
    return result


def enum_cells(cls_info):
    """``{name: (module, qualname)}`` of the cells whose C result is the value of an admitted IntEnum (D103)."""
    from modelx_cython.enums import _traced_enum
    result = {}
    for name, cells in cls_info.cells.items():
        if cells.is_special() or cells.formula_is_generator or not cells.has_typeinfo():
            continue        # a keyword call or a closure keeps its entry: its public method forwards (D154)
        if cells.get_rettype_expr(c_style=True) != 'long long':
            continue
        enum = _traced_enum(cells, None)
        if enum is not None:
            result[name] = enum
    return result


def vector_cells(cls_info):
    """``{name: shape}`` of the cells whose C result is a typed vector or a tuple of them (D154): compiled
    code passes the vector itself; a Python caller gets a new list (a tuple of new lists), never the cache."""
    result = {}
    for name, cells in cls_info.cells.items():
        if cells.is_special() or cells.formula_is_generator or not cells.has_typeinfo():
            continue        # a keyword call or a closure keeps its entry: its public method forwards (D154)
        shape = cells.vector_shape
        if shape in ('V', 'I') or (isinstance(shape, tuple) and shape and shape[0] == 'T'):
            result[name] = shape
    return result


def path_of(cls_info, name):
    owners = [part[len(MODULE_PREF):] for part in cls_info.module.fqname.split('.')[1:-1]]
    owners.append(cls_info.name[len(SPACE_PREF):])
    return '.'.join((*owners, name))


CELL_PREF, REF_PREF = '_mx_cell_', '_mx_ref_'
_TABLE_TYPES = {('pandas', 'DataFrame'), ('pandas', 'Series'), ('modelx_helpers', 'Table')}




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
        for name in (*table_cells(cls_info), *enum_cells(cls_info), *record_cells(cls_info),
                     *vector_cells(cls_info)):
            if owners[name] == {cls_name}:
                mapping[name] = CELL_PREF + name
        result[cls_name] = mapping
    return result


def module_renames(module):
    cached = getattr(module, '_mx_table_renames', None)
    if cached is None:
        cached = module._mx_table_renames = renames(module)
    return cached




def _statements(text, config):
    return cst.parse_module(text, config=config).body


def _renew_on_new_model(cls_info, init, config):
    """In a Space class without items of a model that reads frames: its ``__init__``, when the Space is a top-level
    Space of a new model (its parent is the model), starts a new generation of prepared tables (runtime
    ``table_cells.new_model``)."""
    params = [p.name.value for p in init.params.params]
    if (cls_info.name in cls_info.visitor.itemspace_classes or params[:2] != ['self', 'parent']
            or not any(frame_cell(c, cells) for c in cls_info.module.classes.values()
                       for cells in table_cells(c).values())):
        return init
    hook = _statements(f'if parent is self._model:\n    {RUNTIME}.new_model(self)\n', config)
    return init.with_changes(body=init.body.with_changes(body=(*init.body.body, *hook)))


def rewrite_class(cls_info, node, config):
    """Add the members described in the module docstring to a generated Space class."""
    mapping = module_renames(cls_info.module)[cls_info.name]
    tables = {n: c for n, c in table_cells(cls_info).items() if n in mapping}
    enums = {n: e for n, e in enum_cells(cls_info).items() if n in mapping}
    records = [n for n in record_cells(cls_info) if n in mapping]
    vectors = {n: v for n, v in vector_cells(cls_info).items() if n in mapping}
    body = []
    init_names = set()
    signatures = {}     # vector cells: the parameters of the entry, defaults included (keyword calls)
    for statement in node.body.body:
        if isinstance(statement, cst.FunctionDef):
            name = statement.name.value
            if name in tables or name in enums or name in records or name in vectors:
                if name in vectors:
                    signatures[name] = statement.params
                statement = statement.with_changes(name=cst.Name(mapping[name]))
                if (name in tables and frame_cell(cls_info, tables[name])
                        and cls_info.name not in cls_info.visitor.itemspace_classes):
                    # T-4: prepared when bound (Data-like); T-4.2: published for the items that read it
                    statement = statement.visit(_BindFrame(name, tables[name].is_locked))
            elif name == '__init__':
                for line in statement.body.body:
                    if not isinstance(line, cst.SimpleStatementLine):
                        continue
                    for small in line.body:
                        if isinstance(small, cst.Assign) and isinstance(small.targets[0].target, cst.Attribute):
                            init_names.add(small.targets[0].target.attr.value)
                statement = _renew_on_new_model(cls_info, statement, config)
        body.append(statement)

    lines = []
    for name, cells in tables.items():
        path = path_of(cls_info, name)
        if frame_cell(cls_info, cells) and cls_info.name not in cls_info.visitor.itemspace_classes:
            lines.append(_prepared_method(name, mapping, cells.is_locked))
        lines.append(
            f'@property\n'
            f'def {name}(self):\n'
            f'    return {RUNTIME}.TableCell(self, {name!r}, {path!r})\n')
    for name in records:
        lines.append(
            f'def {name}(self):\n'
            f'    """The pandas row, as live (D130); compiled code calls ``{mapping[name]}``."""\n'
            f'    return {RUNTIME}.row_series(self.{mapping[name]}(), {path_of(cls_info, name)!r})\n')
    for name, (module, qualname) in enums.items():
        params = ', '.join(cls_info.cells[name].params)
        lines.append(
            f'def {name}(self{", " if params else ""}{params}):\n'
            f'    """The live IntEnum member (D103); compiled code calls ``{mapping[name]}``."""\n'
            f'    return {RUNTIME}.enum_member({module!r}, {qualname!r}, self.{mapping[name]}({params}))\n')
    for name, shape in vectors.items():
        params = ', '.join(cls_info.cells[name].params)
        value = f'self.{mapping[name]}({params})'
        result = (f'tuple([v.tolist() for v in {value}])' if isinstance(shape, tuple) else f'{value}.tolist()')
        declared = params
        cell = cls_info.cells[name]
        if name in signatures and (cell.called_with_kwargs or cell.body_has_closure):
            # a keyword call reaches the public method: the entry's own parameters, defaults included
            rest = signatures[name].with_changes(params=signatures[name].params[1:])
            declared = cst.Module(body=[]).code_for_node(rest)
        lines.append(
            f'def {name}(self{", " if declared else ""}{declared}):\n'
            f'    """A new list of the values, as live returns a list (D154); compiled code calls '
            f'``{mapping[name]}``, which passes the typed vector without a copy."""\n'
            f'    return {result}\n')
    if tables:
        get = ['def _mx_table_cell_value(self, name):']
        for name, cells in tables.items():
            get += [f'    if name == {name!r}:', f'        return self.{mapping[name]}()']
        get.append('    raise KeyError(name)')
        lines.append('\n'.join(get) + '\n')
    added = []
    for text in lines:
        for statement in _statements(text, config):
            added.append(statement.with_changes(leading_lines=(cst.EmptyLine(),)))
    return node.with_changes(body=node.body.with_changes(body=(*body, *added)))



class RenameMembers(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)
    """Rename the C members of table cells and table References in a whole generated module.

    The last pass over a generated module: it also adds the members of ``rewrite_class`` to each
    Space class, after every other pass has run on the cells under their public names.

    ``x.<cell>(...)`` becomes ``x._mx_cell_<cell>(...)`` and ``x.<ref>`` becomes
    ``x._mx_ref_<ref>`` only where the receiver ``x`` is known to be an instance of the class
    that owns the member: ``self`` in its class, a parameter or local annotated or cast to the
    class (``base_: _c_X``, ``self: _mx_private_receiver[_c_X]``, ``_mx_cy.cast(_c_X, ...)``), or
    a Reference to a Space declared with that class (``self.data`` with ``data: _c_Data``),
    recursively. Any other receiver keeps the public name (a Python attribute: the property).
    Class-body annotations of the References are renamed with them.
    """

    def __init__(self, module, config=None):
        self.module = module
        self.config = config
        self.mapping = module_renames(module)
        self.enums = {}
        for cls_name, cls_info in module.classes.items():
            for name, enum in enum_cells(cls_info).items():
                if name in self.mapping.get(cls_name, {}):
                    self.enums[name] = enum
        self.wrapped = 0
        self._params = []           # per function: {parameter: enum} of C-integer parameters traced as one IntEnum
        self._owners = []           # per function: (class, Cell) whose formula it is, or None
        self.shown = set()          # (class, Cell, parameter) whose value is used as an object
        self.foreign = []           # (class, Cell, parameter, caller): a member of another class passed to it
        self.constants = {}         # enum constant -> (module, qualname) of its class
        for (owner, ref, _), (symbol, _) in getattr(module, 'code_enum_symbols', {}).items():
            enum = getattr(module.classes[owner].refs.get(ref), 'enum', None)
            if enum:
                self.constants[symbol] = tuple(enum[:2])
        self.vectors = {cls_name: vector_cells(cls_info) for cls_name, cls_info in module.classes.items()}
        self.vector_publics = 0     # D154: vector cell calls left on the public method (a new list)
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
            cls_info = self.module.classes[name]
            updated = rewrite_class(cls_info, updated, self.config)
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
        self._vector_locals.append({small.target.value for line in lines
            if isinstance(line, (cst.SimpleStatementLine, cst.SimpleStatementSuite)) for small in line.body
            if isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name)
            and cst.Module(body=[]).code_for_node(small.annotation.annotation) in vector_types})
        self._functions.append(node)
        self._params.append(self._enum_params(node))

    def leave_FunctionDef(self, original, updated):
        self._locals.pop()
        self._vector_locals.pop()
        self._functions.pop()
        self._params.pop()
        self._owners.pop()
        return updated

    # ---------------------------------------------------------------- enum parameters used as objects
    def _enum_params(self, node):
        """{parameter: enum} of a Cell formula's C-integer parameters traced as one IntEnum class: where one is
        used as an object, it is the member, as live holds it (its ``.name``, ``str``, ``repr``)."""
        from modelx_cython.enums import _traced_enum
        cls_info = self.module.classes.get(self._classes[-1]) if self._classes and self._classes[-1] else None
        name = node.name.value
        cell = cls_info.cells.get(name[3:] if name.startswith('_f_') else name) if cls_info is not None else None
        self._owners.append((cls_info.name, cell.name) if cell is not None else None)
        if cell is None or not cell.has_typeinfo():
            return {}
        found = {}
        for param in node.params.params:
            annotation = param.annotation and cst.Module([]).code_for_node(param.annotation.annotation)
            if annotation == f'{CY_MOD}.longlong' and param.name.value in cell.params:
                enum = _traced_enum(cell, param.name.value)
                if enum is not None:
                    found[param.name.value] = enum
        return found

    def leave_Name(self, original, updated):
        enum = self._params[-1].get(original.value) if self._params else None
        if enum is None:
            return updated
        if not self._shows_class(original):
            return updated
        self.shown.add((*self._owners[-1], original.value))
        return self._member_of(enum, updated)

    _SHOWING_BUILTINS = frozenset(('str', 'repr', 'format', 'ascii', 'type', 'isinstance', 'getattr', 'hasattr', 'id'))

    def _shows_class(self, node):
        """A site where the member's class shows: an attribute of it (``.name``), the first argument of
        ``str``/``repr``/``format``/``type``/``isinstance``/``getattr``/..., an f-string field or a ``%`` operand.
        Elsewhere the parameter stays the C integer, as in a formula that never shows its class."""
        parent = self._parent(node)
        if isinstance(parent, cst.Attribute):
            return parent.value is node
        if isinstance(parent, cst.FormattedStringExpression):
            return True
        if isinstance(parent, cst.BinaryOperation) and isinstance(parent.operator, cst.Modulo):
            return parent.right is node and isinstance(parent.left, (cst.SimpleString, cst.ConcatenatedString))
        if isinstance(parent, cst.Element) and isinstance(self._parent(parent), cst.Tuple):
            outer = self._parent(self._parent(parent))
            return (isinstance(outer, cst.BinaryOperation) and isinstance(outer.operator, cst.Modulo)
                    and isinstance(outer.left, (cst.SimpleString, cst.ConcatenatedString)))
        if isinstance(parent, cst.Arg):
            call = self._parent(parent)
            return (isinstance(call, cst.Call) and isinstance(call.func, cst.Name)
                    and call.func.value in self._SHOWING_BUILTINS and call.args and call.args[0] is parent)
        return False

    def _foreign_members(self, original):
        """Record a member constant of one class passed to a compiled Cell's parameter traced as another."""
        from modelx_cython.enums import _traced_enum
        func = original.func
        owner = self._receiver(func.value) if isinstance(func, cst.Attribute) else None
        cls_info = self.module.classes.get(owner) if owner else None
        cell = cls_info.cells.get(func.attr.value) if cls_info is not None else None
        if cell is None or not cell.has_typeinfo() or any(a.keyword or a.star for a in original.args):
            return
        for arg, param in zip(original.args, cell.params):
            value = arg.value
            given =self.constants.get(value.value) if isinstance(value, cst.Name) else None
            expected = _traced_enum(cell, param) if given else None
            if expected is not None and given != expected:
                owners = [o for o in self._owners if o]
                where = f'{owners[-1][0][len("_c_"):]}.{owners[-1][1]}' if owners else self.module.fqname
                self.foreign.append((owner, cell.name, param, where))

    # ---------------------------------------------------------------- D154: vector results
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
        """Whether compiled code consumes the typed vector that ``node`` (a vector cell call) gives (D154).

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
        self._foreign_members(original)
        if isinstance(func, cst.Attribute):
            new = self._new(func.value, func.attr.value, CELL_PREF)
            shape = self.vectors.get(self._receiver(func.value), {}).get(func.attr.value) if new else None
            if new and shape is not None and not self._vector_native(original, shape):
                self.vector_publics += 1
                return updated      # Python code: the public method's new list (D154)
            if new:
                updated = updated.with_changes(func=updated.func.with_changes(attr=cst.Name(new)))
                if (func.attr.value in self.enums and self._object_use(original)
                        and not isinstance(self._parent(original), cst.IfExp)):
                    return self._member(func.attr.value, updated)
        return updated

    def leave_IfExp(self, original, updated):
        # an inline entry: `(entry(self, t) if exact else cast(T, self.cell(t)))`
        name = self._enum_of(original.orelse)
        if name is not None and self._object_use(original):
            return self._member(name, updated)
        return updated

    # ---------------------------------------------------------------- D103: enum results
    def _parent(self, node):
        return self.get_metadata(ParentNodeProvider, node, None)

    def _enum_of(self, expr):
        """The enum cell whose C value ``expr`` is (a call of it, possibly cast), else None."""
        if isinstance(expr, cst.Call):
            func = expr.func
            if (isinstance(func, cst.Attribute) and func.attr.value == 'cast'
                    and isinstance(func.value, cst.Name) and func.value.value == CY_MOD and len(expr.args) == 2):
                return self._enum_of(expr.args[1].value)
            if isinstance(func, cst.Attribute) and func.attr.value in self.enums and self._receiver(func.value):
                return func.attr.value
        if isinstance(expr, cst.IfExp):
            return self._enum_of(expr.orelse)
        return None

    def _member(self, name, expr):
        return self._member_of(self.enums[name], expr)

    def _member_of(self, enum, expr):
        module, qualname = enum
        self.wrapped += 1
        return cst.parse_expression(f'{RUNTIME}.enum_member({module!r}, {qualname!r}, _x)').with_changes(
            args=(cst.Arg(cst.SimpleString(repr(module))), cst.Arg(cst.SimpleString(repr(qualname))),
                  cst.Arg(expr.with_changes(lpar=(), rpar=()))))

    _TYPED_BUILTINS = {'int', 'float', 'bool', 'abs', 'min', 'max', 'round', 'range', 'len'}

    def _object_use(self, node):
        """Whether the value of ``node`` reaches Python as an object (D103 use sites)."""
        parent = self._parent(node)
        if isinstance(parent, cst.Attribute):
            return parent.value is node
        if isinstance(parent, cst.Element) and isinstance(self._parent(parent), cst.Tuple):
            # a tuple is usually a key (table, cache): its elements follow the tuple's own use
            return self._object_use(self._parent(parent))
        if isinstance(parent, (cst.Element, cst.StarredElement, cst.DictElement, cst.StarredDictElement,
                               cst.Yield)):
            return True
        if isinstance(parent, (cst.ListComp, cst.SetComp, cst.GeneratorExp)):
            return parent.elt is node
        if isinstance(parent, cst.DictComp):
            return parent.key is node or parent.value is node
        if isinstance(parent, cst.IfExp):
            return parent.test is not node and self._object_use(parent)
        if isinstance(parent, cst.Return):
            return self._returns_object(parent)
        if isinstance(parent, cst.Arg):
            call = self._parent(parent)
            return isinstance(call, cst.Call) and not self._typed_call(call)
        return False

    def _typed_call(self, call):
        func = call.func
        if isinstance(func, cst.Name):
            return func.value.startswith('_mx_') or func.value in self._TYPED_BUILTINS
        if isinstance(func, cst.Attribute):
            if func.attr.value.startswith('_mx_'):
                return True
            root = func.value
            while isinstance(root, (cst.Attribute, cst.Call, cst.Subscript)):
                root = root.func if isinstance(root, cst.Call) else root.value
            if isinstance(root, cst.Name) and (root.value.startswith('_mx_') or root.value == CY_MOD):
                return True     # a generated helper, a provider handle or a cast of one
            return self._receiver(func.value) is not None      # a cell or method of a compiled Space
        return False

    def _returns_object(self, node):
        while node is not None and not isinstance(node, (cst.FunctionDef, cst.Lambda)):
            node = self._parent(node)
        if not isinstance(node, cst.FunctionDef) or node.returns is None:
            return True
        annotation = cst.Module(body=[]).code_for_node(node.returns.annotation)
        return annotation == 'object'

    def leave_Attribute(self, original, updated):
        new = self._new(original.value, original.attr.value, REF_PREF)
        if new:
            return updated.with_changes(attr=cst.Name(new))
        return updated

    def leave_AnnAssign(self, original, updated):
        target = updated.target
        if (self._classes and self._classes[-1] and not self._locals and isinstance(target, cst.Name)):
            new = self.mapping.get(self._classes[-1], {}).get(target.value)
            if new and new.startswith(REF_PREF):
                return updated.with_changes(target=cst.Name(new))
        return updated




def finish(syntax, module):
    """The last pass over a generated module (after entry lowering and the formula passes):
    rename the members of table and enum cells and table References, add the Python members."""
    rename = RenameMembers(module, syntax.config_for_parsing)
    result = cst.MetadataWrapper(syntax, unsafe_skip_copy=True).visit(rename)
    for owner, cell, param, where in rename.foreign:
        if (owner, cell, param) in rename.shown:
            raise TypeError(f'{where}: passes a member of one IntEnum class to the parameter {param!r} of '
                            f'{cell}, traced as another IntEnum class and used there as an object, so its value '
                            'would differ from live modelx')
    return _drop_unused_import(result)


def _drop_unused_import(syntax):
    """The module without the runtime import when nothing reads ``_mx_table_cells``: a compiled model
    without table cells imports without modelx_cython installed, as upstream's does."""
    from .syntax import walk
    imports = [statement for statement in syntax.body
               if isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1
               and isinstance(statement.body[0], cst.ImportFrom)
               and any(isinstance(alias.asname, cst.AsName) and alias.asname.name.value == RUNTIME
                       for alias in (statement.body[0].names if isinstance(statement.body[0].names, tuple) else ()))]
    if len(imports) != 1:
        return syntax
    used = any(isinstance(node, cst.Name) and node.value == RUNTIME
               for statement in syntax.body if statement is not imports[0] for node in walk(statement))
    return syntax if used else syntax.with_changes(
        body=tuple(statement for statement in syntax.body if statement is not imports[0]))
