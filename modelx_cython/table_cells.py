"""Python access to compiled table cells and input References (api-tables, D101).

Generated members, per Space class, by name only (no ``__getattribute__``/``__setattr__``
override is added):

- a zero-argument cell bound to a native table (``native_result_binder``) keeps its cached
  value; its C entry is renamed ``_mx_cell_<name>`` (D101a) and every call of it in the
  module is renamed with it, so native code calls the C entry directly. Cython requires each
  C member of a cclass to be declared in the ``.pxd`` and rejects a Python property of the same
  name, and an extension type is immutable, so the public name is freed by the rename. The
  property of the public name returns a ``runtime.table_cells.TableCell``; its setter converts
  the assigned value with the provider's ``table_from_value`` against the build-time schema and
  assigns it, without clearing anything (D130); its miss path rebinds from the frame the cell holds
  when a clear of another Space marked it (``_mx_stale_tables``), so an in-place edit is read;
- a Reference holding a table (a pandas DataFrame or Series, or a helper Table) is stored as
  ``_mx_ref_<name>``, every read of it in the module renamed; the property of its name returns
  the value and its setter stores the new value (nothing is cleared, D130). Other References keep
  their names and declarations (integer References keep their generated ``__setattr__``);
- a name is renamed only when no other class of the module has a member of that name;
- ``clear_all``: live modelx's ``Space.clear_all()`` (D130); live's other clears (``clear_cells``,
  ``clear_items``, ``clear_at``) raise, naming ``Space.clear_all()`` and ``Model.clear_all()``;
  ``_mx_clear_values`` empties the Space's caches and drops its ItemSpaces;
- ``collect`` counts the run so that an assignment during it raises.

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
    """T-4: in the C entry of a frame cell, prepare the computed frame where it is stored (``self._v_<name> =
    val``), under the cell's lock: a table is prepared when it is bound, never on a formula's first read."""

    def __init__(self, name):
        super().__init__()
        self.target = f'self._v_{name}'

    def leave_IndentedBlock(self, original, updated):
        body = []
        for statement in updated.body:
            if (isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1
                    and isinstance(statement.body[0], cst.Assign)
                    and cst.Module(body=()).code_for_node(statement.body[0].targets[0].target) == self.target):
                body.append(cst.parse_statement(f'{RUNTIME}.bind_frame(self, val)'))
            body.append(statement)
        return updated.with_changes(body=body)


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


def _holds_table(ref):
    kind = getattr(ref, 'type_', None)
    if not isinstance(kind, type):
        return False
    return any((k.__module__.split('.')[0], k.__name__) in _TABLE_TYPES for k in kind.__mro__)


def _members(cls_info):
    return set(cls_info.cells) | set(cls_info.refs) | set(cls_info.spaces)


def table_refs(cls_info, integer_refs=()):
    """Names of the References of ``cls_info`` that hold a table (D101a)."""
    readonly = set(getattr(cls_info, 'readonly_refs', ()))
    return [name for name, ref in cls_info.refs.items()
            if name not in cls_info.spaces and name not in readonly and name not in integer_refs
            and not getattr(ref, 'decl_type_expr', None) and _holds_table(ref)]


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
        for name in table_refs(cls_info):
            if owners[name] == {cls_name}:
                mapping[name] = REF_PREF + name
        result[cls_name] = mapping
    return result


def module_renames(module):
    cached = getattr(module, '_mx_table_renames', None)
    if cached is None:
        cached = module._mx_table_renames = renames(module)
    return cached


def _schema_literal(cells):
    schema = cells.table_schema
    if not isinstance(schema, dict) or not schema.get('keys'):
        return 'None'
    return repr({'keys': tuple(schema['keys']), 'columns': dict(schema.get('columns') or {})})


def _statements(text, config):
    return cst.parse_module(text, config=config).body


def _reset_lines(cls_info):
    from modelx_cython.serial_collect import _reset_lines as collect_reset
    for line in collect_reset(cls_info):
        yield line.replace('root.', 'self.', 1).replace('(root.', '(self.')


def rewrite_class(cls_info, node, config, integer_refs):
    """Add the members described in the module docstring to a generated Space class."""
    mapping = module_renames(cls_info.module)[cls_info.name]
    tables = {n: c for n, c in table_cells(cls_info).items() if n in mapping}
    enums = {n: e for n, e in enum_cells(cls_info).items() if n in mapping}
    records = [n for n in record_cells(cls_info) if n in mapping]
    vectors = {n: v for n, v in vector_cells(cls_info).items() if n in mapping}
    refs = [n for n in table_refs(cls_info, integer_refs) if n in mapping]
    body = []
    init_names = set()
    has_collect = False
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
                    statement = statement.visit(_BindFrame(name))     # T-4: prepared when bound (Data-like)
            elif name == '__init__':
                for line in statement.body.body:
                    if not isinstance(line, cst.SimpleStatementLine):
                        continue
                    for small in line.body:
                        if isinstance(small, cst.Assign) and isinstance(small.targets[0].target, cst.Attribute):
                            init_names.add(small.targets[0].target.attr.value)
            elif name == 'collect' and not statement.params.params[1:]:
                has_collect = True
                statement = _counted_collect(statement, config)
        body.append(statement)

    lines = []
    for name, cells in tables.items():
        path = path_of(cls_info, name)
        if frame_cell(cls_info, cells):
            bind = ('' if cls_info.name in cls_info.visitor.itemspace_classes
                    else f'    {RUNTIME}.bind_frame(self, value)\n')
            assign = bind + f'    {RUNTIME}.assign_table_cell(self, {name!r}, {path!r}, value)\n'
            lines.append(
                f'@property\n'
                f'def {name}(self):\n'
                f'    return {RUNTIME}.TableCell(self, {name!r}, {path!r})\n\n'
                f'@{name}.setter\n'
                f'def {name}(self, value):\n' + assign)
            continue
        helper = cells.native_result_binder.rsplit('.', 1)[0]
        if cells.native_provider_binding.manifest.provider == STANDARD_PROVIDER:
            assign = (f'    {RUNTIME}.assign_table_cell(self, {name!r}, {path!r},\n'
                      f'        {helper}.table_from_value(value, {path!r}, {_schema_literal(cells)}))\n')
        else:   # another provider's table: no conversion function to rebuild it from a frame
            assign = (f'    raise NotImplementedError({path!r} + ": this table is bound by the provider "\n'
                      f'        {cells.native_provider_binding.manifest.provider!r} + ", which cannot rebuild it from an assigned value")\n')
        lines.append(
            f'@property\n'
            f'def {name}(self):\n'
            f'    return {RUNTIME}.TableCell(self, {name!r}, {path!r})\n\n'
            f'@{name}.setter\n'
            f'def {name}(self, value):\n' + assign)
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
        # D130: the table cells marked by a clear of another Space rebind from the frame they hold
        stale = ['def _mx_stale_tables(self):']
        for name, cells in tables.items():
            if frame_cell(cls_info, cells):
                continue        # T-3: a frame cell holds the frame itself; nothing to rebuild
            if cells.is_locked:
                stale.append(f'    {MX_SYS_MOD}._mx_store_flag({CY_MOD}.address(self._has_{name}), False)')
            else:
                stale.append(f'    self._has_{name} = False')
        if len(stale) == 1:
            stale.append('    pass')
        lines.append('\n'.join(stale) + '\n')
        get = ['def _mx_table_cell_value(self, name):']
        put = ['def _mx_set_table_cell(self, name, value):']
        for name, cells in tables.items():
            get += [f'    if name == {name!r}:', f'        return self.{mapping[name]}()']
            put += [f'    if name == {name!r}:', f'        self._v_{name} = value']
            if cells.is_locked:
                put.append(f'        {MX_SYS_MOD}._mx_store_flag({CY_MOD}.address(self._has_{name}), True)')
            else:
                put.append(f'        self._has_{name} = True')
            put.append('        return')
        get.append('    raise KeyError(name)')
        put.append('    raise KeyError(name)')
        lines += ['\n'.join(get) + '\n', '\n'.join(put) + '\n']
    for name in refs:
        path = path_of(cls_info, name)
        lines.append(
            f'@property\n'
            f'def {name}(self):\n'
            f'    return self.{mapping[name]}\n\n'
            f'@{name}.setter\n'
            f'def {name}(self, value):\n'
            f'    {RUNTIME}.before_reference(self, {path!r})\n'
            f'    self.{mapping[name]} = value\n')   # D109: published caches stay, as exported
    members = _members(cls_info)
    # D130: of live modelx's clears only Space.clear_all() and Model.clear_all() are provided (no cell-level
    # or partial clear: without dependency tracking they would leave dependents stale)
    if 'clear_all' not in members:
        lines.append(f'def clear_all(self):\n'
                     f'    """Clear the values of this Space and the Spaces below it, their assigned inputs and '
                     f'their ItemSpaces, as live modelx."""\n'
                     f'    {RUNTIME}.clear_all(self)\n')
    for removed in ('clear_cells', 'clear_items', 'clear_at'):
        if removed not in members:
            # D130: live modelx's other Space clears are not provided; calling one names the two clears
            lines.append(f'def {removed}(self, *args, **kwargs):\n'
                         f'    {RUNTIME}.not_provided(self, {removed!r})\n')
    clear = ['def _mx_clear_values(self):',
             '    """Empty every cache of this Space and drop its ItemSpaces (D101, D130)."""']
    clear += ['    ' + line for line in _reset_lines(cls_info)]
    if '_mx_itemspaces' in init_names:
        if cls_info.name in cls_info.visitor.itemspace_classes:
            # audit-routes P2: the dropped items are released as by __delitem__ (D114, D116), so they
            # are freed by their reference count instead of waiting for the cyclic GC
            from modelx_cython.item_roots import release_item
            clear.append('    for _mx_item in self._mx_itemspaces.values():')
            clear.append(f'        {release_item(cls_info.name)}(_mx_item)')
        clear.append('    self._mx_itemspaces.clear()')
    if has_collect or '_mx_collect_cache' in init_names:
        clear.append('    self._mx_collect_cache.clear()')
    lines.append('\n'.join(clear) + '\n')
    added = []
    for text in lines:
        for statement in _statements(text, config):
            added.append(statement.with_changes(leading_lines=(cst.EmptyLine(),)))
    return node.with_changes(body=node.body.with_changes(body=(*body, *added)))


def _counted_collect(function, config):
    """``collect`` marks the model as running for the duration of the call."""
    inner = function.body.body
    wrapper = cst.parse_statement(
        'try:\n    pass\nfinally:\n'
        f'    {RUNTIME}.collect_end(self._model)\n', config=config)
    wrapper = wrapper.with_changes(body=wrapper.body.with_changes(body=tuple(inner)))
    begin = cst.parse_statement(f'{RUNTIME}.collect_begin(self._model)\n', config=config)
    return function.with_changes(body=function.body.with_changes(body=(begin, wrapper)))


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

    def __init__(self, module, config=None, integer_refs=None):
        self.module = module
        self.config = config
        self.integer_refs = integer_refs
        self.mapping = module_renames(module)
        self.enums = {}
        for cls_name, cls_info in module.classes.items():
            for name, enum in enum_cells(cls_info).items():
                if name in self.mapping.get(cls_name, {}):
                    self.enums[name] = enum
        self.wrapped = 0
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
            updated = rewrite_class(cls_info, updated, self.config, self.integer_refs(cls_info))
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

    def leave_FunctionDef(self, original, updated):
        self._locals.pop()
        self._vector_locals.pop()
        self._functions.pop()
        return updated

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
        module, qualname = self.enums[name]
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


MEMSET = 'from cython.cimports.libc.string import memset as _mx_collect_memset'


def finish(syntax, module):
    """The last pass over a generated module (after entry lowering and integer guards):
    rename the members of table and enum cells and table References, add the Python members."""
    from modelx_cython.transformer import _integer_reference_paths
    result = cst.MetadataWrapper(syntax, unsafe_skip_copy=True).visit(
        RenameMembers(module, syntax.config_for_parsing, _integer_reference_paths))
    if MEMSET not in syntax.code and '_mx_collect_memset(' in result.code:
        # _mx_clear_values resets array flags; the module imported memset only for a serial collect
        result = result.with_changes(body=(cst.parse_statement(MEMSET + '\n'), *result.body))
    return result
