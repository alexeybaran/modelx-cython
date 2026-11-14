"""Type-directed lowering of pandas-spelled table reads to typed handle reads (M1, W1.3).

Each read is lowered on its own, from the traced type of its receiver; the shape of the
surrounding statement never matters. A receiver is a *table* when it is a self-rooted
zero-argument call of a Cell whose every sampled result is a helper ``Table``
(``resolver.table_results``) with a traced schema (``resolver.table_schemas``), a local
assigned only from tables (an alias), or a selection ``T.loc[prefix]`` (a sub-table, inline or
held in a local). A *record* is a zero-argument call of a Cell whose sampled results are helper
``Record`` objects (``model_point()``), or a local assigned only from such calls.

Generated shape (``H`` a ``TableHandle``, ``M`` the cimported ``_native_inputs`` provider):

- an inline receiver gets one handle per formula, hoisted before the first top-level statement
  that reads it: ``_mx_h0: M.TableHandle = M.table_handle(self.T())``;
- a table or sub-table local ``sub`` gets a companion handle ``_mx_h_sub`` (declared at the top
  of the formula) assigned where ``sub`` is assigned. A sub-table whose every use is lowered
  is never built (``_mx_h_sub = H.sub_1(p)``); one that escapes (returned, passed, iterated,
  read in a nested function) is kept as written and every read through it stays Python;
  a table alias that escapes keeps its Python assignment next to the companion;
- ``T.loc[k, "c"]``, ``T.at[k, "c"]``, ``T.loc[k]["c"]``, ``T["c"][k]``:
  ``H.f64(H.row_n(k...), H.col("c"))`` for a float column, ``i64`` for an int column, ``obj``
  otherwise; ``int(...)`` of an int, bool or enum column reads ``i64``; ``float(...)`` around a
  float read is dropped;
- ``return T.loc[prefix]`` (a table-valued result): ``H.sub_n(prefix...).table_object()``, the
  bound table of the shared child handle (D45, D53);
- ``row = T.loc[k]`` (complete key): ``_mx_row_row = H.row_n(k...)``; ``row["c"]`` reads that
  row; any other use of ``row`` becomes ``H.row_record(_mx_row_row)``; an inline row that is
  not read by field (``return T.loc[k]``) is ``H.row_record(H.row_n(...))``;
- ``k in T.index`` / ``not in``: ``H.has_n(...)`` (``n`` may be fewer parts than levels);
  ``T.index.min()/max()``: ``H.first_int()/last_int()`` on a single integer level
  (``int(...)`` around them dropped), ``first_key()/last_key()`` otherwise;
- ``sub.index.asof(x)``: ``H.asof_int(x)`` in any position, ``H.asof_row(x)`` as the row key of
  a read of the same table; ``sub.index.searchsorted(x, side="right")``:
  ``H.searchsorted_right(x)``; ``side="left"`` (the default) is ``H.searchsorted_right(x - 1)``,
  the same position on integer keys; ``sub.index[i]``: ``H.key_int(i)``;
- ``rec["f"]`` on a record: ``R.obj(R.field("f"))`` (``R: M.RecordHandle``; record field kinds
  are not traced, so conversions around the read stay). Only with ``"table_recognisers":
  false``: with the round-2 recognisers on they keep handling record reads.

Key parts are passed one by one. A part goes bare (Cython picks the C ``long long``
specialisation for a C integer, the object one for an object) unless its static type could be
a C ``double`` or ``bint``; such parts are cast to object, so a float key or a bool label
arrives as the Python value. Lowering is enabled by the spec key ``"typed_table_reads"``.

Everything else on a traced table (``iterrows``, iteration of the index, slices, a column
chosen by a variable, a table-valued selection used other than as the returned value, a
receiver without a traced schema) is left as written and runs as a Python call on the bound table; ``declined`` carries the reason, which
the admission report shows.
"""

import libcst as cst

from .usage import RESOLVED, _extract_self_chain

TABLE_TYPE = 'modelx_helpers.immutable_inputs.Table'
RECORD_TYPE = 'modelx_helpers.immutable_inputs.Record'
PROVIDER = 'modelx_cython.runtime._native_inputs'
MAX_PARTS = 6
INT_TYPES = frozenset(('long long', 'long', 'int', 'short', 'Py_ssize_t', 'size_t'))
UNSAFE_BARE = frozenset(('double', 'float', 'bint'))
SKIPPED_SCOPES = (cst.FunctionDef, cst.Lambda, cst.ClassDef)
COMPREHENSIONS = (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)
LONG_LONG_METHODS = frozenset(('first_int', 'last_int', 'asof_int', 'key_int', 'i64'))
C_INT_METHODS = frozenset(('first_int', 'last_int', 'asof_int', 'key_int', 'searchsorted_right', 'i64', 'rows'))
INDEX_BUILTINS = frozenset(('len', 'min', 'max', 'list', 'tuple', 'sorted'))

PYTHON = 'runs as a Python call'
ESCAPES = 'sub-table local {name!r} is read in a nested function ({use}): kept as written; ' + PYTHON
KEPT = 'local {name!r} could not be lowered ({why}): kept as written; ' + PYTHON
RESULT = ('table-valued selection used as a value ({use}): the helper sub-table is built; '
          'readers of a table-valued cell use its bound handle; ' + PYTHON)
ITERROWS = 'iterrows(): the handle has no row iteration in table order; ' + PYTHON
ITERROWS_ROW = 'row of iterrows(): ' + PYTHON
INDEX_OBJECT = ('T.index used as a Python object (iteration, len, list, ...), not by `k in T.index`, '
                'min()/max(), asof, searchsorted or [i]; ' + PYTHON)
NO_SCHEMA = 'receiver has no traced table schema; ' + PYTHON
INDEX_NOT_INT = ('iteration of an index of several levels (tuple keys) other than inside min, max, any, '
                 'all, sorted or set, or with a target that does not unpack every level; ' + PYTHON)
UNORDERED = frozenset(('min', 'max', 'any', 'all', 'sorted', 'set'))
COLUMN_VIEW = 'column selection T["c"] used other than by a key read; ' + PYTHON


def _code(node):
    return cst.Module(body=()).code_for_node(node)


def _call(func, *args):
    return cst.Call(func=func, args=[cst.Arg(arg) for arg in args])


def _method(handle, name, *args):
    return _call(cst.Attribute(handle, cst.Name(name)), *args)


def _plain_args(call):
    return not any(arg.keyword or arg.star for arg in call.args)


def _literal_str(node):
    if isinstance(node, cst.SimpleString):
        value = node.evaluated_value
        return value if type(value) is str else None
    return None


def _index_attr(node, attr):
    """``X.attr`` -> X, else None."""
    if isinstance(node, cst.Attribute) and node.attr.value == attr:
        return node.value
    return None


def _subscript_parts(node):
    """The index expressions of a plain subscript (no slices), else None."""
    if not isinstance(node, cst.Subscript):
        return None
    parts = []
    for element in node.slice:
        if not isinstance(element.slice, cst.Index) or element.slice.star is not None:
            return None
        parts.append(element.slice.value)
    return parts


def _bound_names(target):
    names = []

    class Collect(cst.CSTVisitor):
        def visit_Attribute(self, node):
            return False        # x.a = ...: binds nothing

        def visit_Subscript(self, node):
            return False

        def visit_Name(self, node):
            names.append(node.value)
    target.visit(Collect())
    return names


def _is_docstring(statement):
    return (isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1
            and isinstance(statement.body[0], cst.Expr)
            and isinstance(statement.body[0].value, (cst.SimpleString, cst.ConcatenatedString)))


class Schema:
    """Levels and column kinds of a traced table (helper ``table_schema(t)``)."""
    __slots__ = ('levels', 'keys', 'columns')

    def __init__(self, keys, columns):
        self.keys, self.columns = tuple(keys), dict(columns)
        self.levels = len(self.keys)

    @classmethod
    def of(cls, schema):
        if not isinstance(schema, dict) or not schema.get('keys'):
            return None
        return cls(schema['keys'], schema.get('columns') or {})

    def child(self, depth):
        return Schema(self.keys[depth:], self.columns)

    def same(self, other):
        return self.keys == other.keys and self.columns == other.columns

    @property
    def int_key(self):
        """A single integer level (enum-labelled levels are 'int', W1.H D27)."""
        return self.levels == 1 and self.keys[0] == 'int'


class _Formula(cst.CSTVisitor):
    """Bindings and loads of each name in one formula.

    ``values[name]`` lists the values of plain ``name = value`` assignments (one Name target),
    ``other`` the names bound any other way (loops, unpacking, augmented assignment, walrus,
    ``with``, ``except``, ``del``, imports, ``global``/``nonlocal``, nested definitions).
    ``loads[name]`` lists ``(Name node, nested)`` of each read of the name outside the
    comprehensions that rebind it; ``nested`` is True inside a nested function or lambda.
    ``parents`` maps id(node) -> parent node. Comprehension targets are local to their
    comprehension.
    """
    def __init__(self, root):
        self.root = root
        self.values, self.other, self.loads, self.parents = {}, set(), {}, {}
        self.binds = {}             # name -> number of bindings outside comprehensions
        self.names = set()
        self._depth = 0
        self._shadow = []
        self._stores = set()

    def on_visit(self, node):
        for child in node.children:
            self.parents[id(child)] = node
        if isinstance(node, SKIPPED_SCOPES):
            if node is not self.root:
                if isinstance(node, (cst.FunctionDef, cst.ClassDef)):
                    self.other.add(node.name.value)
                    self._bind(node.name.value)
                    self._stores.add(id(node.name))
                params = getattr(node, 'params', None)
                names = set()
                if params is not None:
                    for param in (*params.params, *params.kwonly_params, *params.posonly_params,
                                  *([params.star_arg] if isinstance(params.star_arg, cst.Param) else []),
                                  *([params.star_kwarg] if params.star_kwarg else [])):
                        names.add(param.name.value)
                        self._stores.add(id(param.name))
                self._shadow.append(names)
            else:
                for param in (*node.params.params, *node.params.kwonly_params, *node.params.posonly_params):
                    self._stores.add(id(param.name))
                self._stores.add(id(node.name))
            self._depth += 1
        if isinstance(node, COMPREHENSIONS):
            names = set()
            current = node.for_in
            while current is not None:
                names.update(_bound_names(current.target))
                current = current.inner_for_in
            self._shadow.append(names)
        return super().on_visit(node)

    def on_leave(self, node):
        super().on_leave(node)
        if isinstance(node, SKIPPED_SCOPES):
            self._depth -= 1
            if node is not self.root:
                self._shadow.pop()
        if isinstance(node, COMPREHENSIONS):
            self._shadow.pop()

    def _bind(self, name):
        self.binds[name] = self.binds.get(name, 0) + 1

    def _store(self, target, value=None, plain=False):
        if plain and isinstance(target, cst.Name):
            if not any(target.value in names for names in self._shadow):
                self.values.setdefault(target.value, []).append(value)
                self._bind(target.value)
            self._stores.add(id(target))
            return

        class Collect(cst.CSTVisitor):
            def visit_Attribute(inner, node):
                return False

            def visit_Subscript(inner, node):
                return False

            def visit_Name(inner, node):
                self._stores.add(id(node))
                if not any(node.value in names for names in self._shadow):
                    self.other.add(node.value)
                    self._bind(node.value)
        target.visit(Collect())

    def visit_Assign(self, node):
        for target in node.targets:
            self._store(target.target, node.value, plain=len(node.targets) == 1 and self._depth == 1)

    def visit_AnnAssign(self, node):
        self._store(node.target, node.value, plain=node.value is not None and self._depth == 1)

    def visit_AugAssign(self, node):
        self._store(node.target)

    def visit_For(self, node):
        self._store(node.target)

    def visit_CompFor(self, node):
        class Collect(cst.CSTVisitor):
            def visit_Name(inner, name):
                self._stores.add(id(name))
        node.target.visit(Collect())

    def visit_NamedExpr(self, node):
        self._store(node.target)

    def visit_WithItem(self, node):
        if node.asname is not None:
            self._store(node.asname.name)

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self._store(node.name.name)

    def visit_Del(self, node):
        self._store(node.target)

    def visit_Global(self, node):
        self.other.update(item.name.value for item in node.names)
        for item in node.names:
            self._bind(item.name.value)
            self._bind(item.name.value)

    def visit_Nonlocal(self, node):
        self.other.update(item.name.value for item in node.names)
        for item in node.names:
            self._bind(item.name.value)
            self._bind(item.name.value)

    def visit_ImportAlias(self, node):
        name = node.asname.name if node.asname else node.name
        while isinstance(name, cst.Attribute):
            name = name.value
        if isinstance(name, cst.Name):
            self.other.add(name.value)
            self._bind(name.value)
            self._stores.add(id(name))

    def visit_Attribute(self, node):
        self._stores.add(id(node.attr))         # x.attr: attr is not a name read

    def visit_Arg(self, node):
        if node.keyword is not None:
            self._stores.add(id(node.keyword))

    def visit_Name(self, node):
        self.names.add(node.value)
        if id(node) in self._stores or any(node.value in names for names in self._shadow):
            return
        self.loads.setdefault(node.value, []).append((node, self._depth > 1))


class _Ref:
    """A lowered receiver: kind 'table' (handle expression), 'row', 'column' or 'record';
    ``sites`` are the original nodes consumed when a read through it is lowered, ``notes``
    the key notes of the selections it carries."""
    __slots__ = ('kind', 'schema', 'handle', 'sites', 'extra', 'notes')

    def __init__(self, kind, schema, handle, sites=(), extra=None, notes=()):
        self.kind, self.schema, self.handle = kind, schema, handle
        self.sites, self.extra, self.notes = tuple(sites), extra, tuple(notes)


class _Parts(list):
    """Key part expressions, with the note on their types for ``table_reads_keys``."""
    note = None


class TableReads:
    """The W1.1 contract (M1-INTERFACE section 3): ``lower(node)`` returns the formula with
    the reads it can type rewritten; ``typed``, ``lowered``, ``declined`` and ``keys``
    describe what it did."""

    def __init__(self, resolver, cls_info, cells, providers, policy, sites):
        self.resolver, self.cls_info, self.cells = resolver, cls_info, cells
        self.providers, self.policy, self.sites = providers, policy, sites
        self.typed = 0
        self.lowered = set()
        self.declined = {}
        self.keys = []
        self._site_ids = {id(site.node) for site in sites}
        self._key_locals = set()    # loop variables of lowered `for k in T.index:` (C long long)

    # ------------------------------------------------------------------ driver
    def enabled(self):
        return bool(getattr(self.policy, 'typed_table_reads', False))

    def records_enabled(self):
        return self.enabled() and not getattr(self.policy, 'table_recognisers', True)

    def lower(self, node):
        if not self.enabled() or self.resolver is None or not (self.sites or self.records_enabled()):
            return node
        from .helpers import module_name
        self.module = module_name(PROVIDER)
        self.formula = _Formula(node)
        node.visit(self.formula)
        self._params = set(getattr(self.cells, 'params', ()) or ())
        kept, escaped, first_reasons = {}, set(), {}
        for _ in range(12):     # locals that cannot be lowered or escape: grow to a fixed point
            attempt = _Lowering(self, kept, escaped)
            new = node.visit(attempt)
            for key, reason in attempt.reasons.items():     # the most specific reason comes first
                first_reasons.setdefault(key, reason)
            more_kept, more_escaped = attempt.escapes()
            if set(more_kept) <= set(kept) and more_escaped <= escaped:
                break
            for name, why in more_kept.items():
                kept.setdefault(name, why)
            escaped |= more_escaped
        self.typed, self.lowered, self.keys = attempt.typed, attempt.lowered, attempt.keys
        for site in self.sites:
            if id(site.node) in self.lowered:
                continue
            reason = first_reasons.get(id(site.node)) or attempt.default_reason(site)
            if reason:
                self.declined[id(site.node)] = reason
        if not attempt.changed:
            return node
        self.providers.add(PROVIDER)
        return new

    # ------------------------------------------------------------------ facts
    def resolve(self, func):
        chain = _extract_self_chain(func)
        if not isinstance(chain, list) or not chain:
            return None, None
        status, name = self.resolver.resolve_chain(self.cls_info.fqname, chain)
        return (name, chain) if status == RESOLVED else (None, chain)

    def cell_of(self, expr):
        """Cells fqname of a self-rooted zero-argument call, else None."""
        if not (isinstance(expr, cst.Call) and not expr.args):
            return None
        return self.resolve(expr.func)[0]

    def call_rettype(self, expr):
        """(C return type, cell name) of a self-rooted cell call, else (None, None)."""
        if not isinstance(expr, cst.Call) or not _plain_args(expr):
            return None, None
        name, chain = self.resolve(expr.func)
        if name is None:
            return None, None
        return self.resolver.cell_rettype(name), chain[-1]

    def param_type(self, name):
        if (name not in self._params or name in self.formula.values or name in self.formula.other):
            return None
        cells = self.cells
        if not (cells.has_args() and cells.has_typeinfo()):
            return 'object'
        try:
            return cells.get_argtype_expr(name, c_style=True, formula=True)
        except Exception:      # pragma: no cover - defensive: an untraced parameter
            return None

    def self_ref(self, expr):
        """The Reference read by ``self.<name>``, else None."""
        if (isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Name)
                and expr.value.value == 'self'):
            return getattr(self.cls_info, 'refs', {}).get(expr.attr.value)
        return None

    def enum_member(self, expr):
        """``self.<Enum>.<member>``, ``self.<Enum>(value)`` or ``self.<Enum>[label]`` on an enum
        Reference."""
        if isinstance(expr, cst.Attribute):
            base = expr.value
        elif isinstance(expr, cst.Call):
            base = expr.func
        elif isinstance(expr, cst.Subscript):
            base = expr.value
        else:
            return False
        ref = self.self_ref(base)
        return ref is not None and bool(getattr(ref, 'enum', None))

    def table_cell(self, expr):
        """Schema of a zero-argument Cell call traced as a helper Table, else None."""
        name = self.cell_of(expr)
        if name is None or self.resolver.table_results.get(name) != TABLE_TYPE:
            return None
        return Schema.of(self.resolver.table_schemas.get(name))

    def untraced_table(self, expr):
        """A table receiver that has no traced schema (a Reference, a cell without schema)."""
        name = self.cell_of(expr)
        if name is not None:
            return self.resolver.table_results.get(name) == TABLE_TYPE
        ref = self.self_ref(expr)
        type_ = getattr(ref, 'type_', None)
        return type_ is not None and f'{type_.__module__}.{type_.__qualname__}' == TABLE_TYPE

    def record_cell(self, expr):
        name = self.cell_of(expr)
        return name is not None and self.resolver.table_results.get(name) == RECORD_TYPE

    def is_int(self, expr, depth=0):
        """Why ``expr`` is a non-bool integer value, or None (original nodes)."""
        if depth > 6:
            return None
        if isinstance(expr, cst.Integer):
            return 'integer literal'
        if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, (cst.Minus, cst.Plus)):
            return self.is_int(expr.expression, depth + 1)
        if isinstance(expr, cst.Name):
            kind = self.param_type(expr.value)
            if kind in INT_TYPES:
                return f'parameter {expr.value} is {kind}'
            if expr.value in self._key_locals:
                return f'local {expr.value} is a key of an integer index'
            values = self.formula.values.get(expr.value)
            if (values and expr.value not in self.formula.other and expr.value not in self._params
                    and all(self.is_int(value, depth + 1) for value in values)):
                return f'local {expr.value} is assigned integers'
            return None
        if isinstance(expr, cst.Attribute):
            ref = self.self_ref(expr)
            type_ = getattr(ref, 'type_', None)
            if isinstance(type_, type) and issubclass(type_, int) and not issubclass(type_, bool) \
                    and not getattr(ref, 'enum', None):
                return f'reference {expr.attr.value} is int'
            return None
        if isinstance(expr, cst.BinaryOperation) and isinstance(
                expr.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo)):
            if self.is_int(expr.left, depth + 1) and self.is_int(expr.right, depth + 1):
                return 'integer arithmetic'
            return None
        if isinstance(expr, cst.Call) and _plain_args(expr):
            func = expr.func
            if isinstance(func, cst.Name) and func.value in ('min', 'max', 'abs') and expr.args:
                return ('integer min/max' if all(self.is_int(arg.value, depth + 1) for arg in expr.args)
                        else None)
            if isinstance(func, cst.Name) and func.value == 'len' and len(expr.args) == 1:
                return 'len()'
            if isinstance(func, cst.Name) and func.value == 'int' and len(expr.args) == 1:
                inner = expr.args[0].value
                if self.is_int(inner, depth + 1) or self._int_index_read(inner):
                    return 'int() of an integer'
                return None
            kind, cell = self.call_rettype(expr)
            if kind in INT_TYPES:
                return f'cell {cell} returns {kind}'
        return None

    def _int_index_read(self, expr):
        """``T.index.min()/max()`` or ``T.index.asof(k)`` on a single integer level."""
        if not (isinstance(expr, cst.Call) and isinstance(expr.func, cst.Attribute)
                and expr.func.attr.value in ('min', 'max', 'asof')):
            return False
        receiver = _index_attr(expr.func.value, 'index')
        if receiver is None and isinstance(expr.func.value, cst.Name):     # an index local
            values = self.formula.values.get(expr.func.value.value) or ()
            if len(values) == 1 and expr.func.value.value not in self.formula.other:
                receiver = _index_attr(values[0], 'index')
        if receiver is None:
            return False
        schema = self.table_cell(receiver)
        if schema is None and isinstance(receiver, cst.Name):
            values = self.formula.values.get(receiver.value) or ()
            schemas = [self.table_cell(value) for value in values]
            schema = schemas[0] if schemas and all(s is not None and s.same(schemas[0]) for s in schemas) else None
        return schema is not None and schema.int_key

    def bare_ok(self, expr):
        """Why ``expr`` can be passed bare as a key part (its static type is not a C double or
        bint), or None (original nodes)."""
        why = self.is_int(expr)
        if why:
            return f'int ({why})'
        if isinstance(expr, (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString)):
            return 'object (string)'
        if self.enum_member(expr):
            return 'enum member'
        if isinstance(expr, cst.Name):
            kind = self.param_type(expr.value)
            if kind is not None and kind not in UNSAFE_BARE:
                return f'{"int" if kind in INT_TYPES else "object"} (parameter {expr.value} is {kind})'
            return None
        if isinstance(expr, cst.Attribute):
            ref = self.self_ref(expr)
            if ref is not None and getattr(ref, 'enum', None) is None:
                type_ = getattr(ref, 'type_', None)
                if isinstance(type_, type) and type_ in (str,):
                    return f'object (reference {expr.attr.value} is str)'
            return None
        if isinstance(expr, cst.Call) and isinstance(expr.func, cst.Name) and expr.func.value == 'str':
            return 'object (str())'
        kind, cell = self.call_rettype(expr)
        if kind is not None and kind not in UNSAFE_BARE:
            return f'{"int" if kind in INT_TYPES else "object"} (cell {cell} returns {kind})'
        return None

    def exact_long_long(self, expr):
        """``expr`` is statically a C ``long long`` (a cell result or a parameter)."""
        if isinstance(expr, cst.Name):
            return self.param_type(expr.value) == 'long long' or expr.value in self._key_locals
        return isinstance(expr, cst.Call) and self.call_rettype(expr)[0] == 'long long'

    def scalar(self, expr):
        """``expr`` is certainly a single key part (not a tuple)."""
        if self.bare_ok(expr) or isinstance(expr, cst.Float):
            return True
        if isinstance(expr, cst.Name):
            return self.param_type(expr.value) in UNSAFE_BARE
        if isinstance(expr, cst.Call):
            return self.call_rettype(expr)[0] in UNSAFE_BARE
        return self.record_field(expr)

    def c_local(self, expr):
        """A local assigned only values of a C scalar type (cell results, parameters, literals):
        Cython types it as that C type, so it cannot hold a tuple."""
        values = self.formula.values.get(expr.value) or ()
        for value in values:
            if isinstance(value, (cst.Integer, cst.Float)):
                continue
            if isinstance(value, cst.Name) and self.param_type(value.value) in (INT_TYPES | UNSAFE_BARE):
                continue
            if self.call_rettype(value)[0] in (INT_TYPES | UNSAFE_BARE):
                continue
            return False
        return bool(values)

    def record_field(self, expr):
        """``rec["f"]`` on a traced Record (a model point row): one scalar value."""
        parts = _subscript_parts(expr)
        if parts is None or len(parts) != 1 or _literal_str(parts[0]) is None:
            return False
        value = expr.value
        if self.record_cell(value):
            return True
        if isinstance(value, cst.Name) and value.value not in self.formula.other:
            values = self.formula.values.get(value.value) or ()
            return bool(values) and all(self.record_cell(item) for item in values)
        return False


class _Lowering(cst.CSTTransformer):
    """One lowering attempt over a formula.

    ``kept`` are locals left as written (name -> why), ``escaped`` locals that are lowered but
    whose Python value is also needed (a table alias or record keeps its assignment, a row
    local's other uses read ``row_record``).
    """

    def __init__(self, owner, kept, escaped):
        super().__init__()
        self.o, self.f = owner, owner.formula
        self.kept, self.escaped = kept, escaped
        self.typed = 0
        self.lowered = set()
        self.keys = []
        self.reasons = {}
        self.changed = False
        self.consumed = set()       # id(Name) of local loads replaced by a lowering
        self.failed = {}            # local name -> why its assignment could not be lowered
        self._root = None
        self._top = 0
        self._top_index = {}
        self._shadow = []
        self._memo = {}             # id(original expr) -> _Ref of an inline selection
        self._kinds = {}            # id(emitted node) -> (method, column kind, handle, row, col)
        self._hoisted = {}          # receiver code -> [name, receiver, kind, first top index]
        self._declare = []          # (name, type) declared at the top of the formula
        self._companions = {}
        self._rows = {}
        self._used = set(self.f.names)
        self._keyvars = []          # per comprehension: loop name -> (handle, position name)
        self._compfor = {}          # id(original CompFor) -> (handle, position name, index ref)
        self.locals = self._classify()
        owner._key_locals = self._loop_keys()
        self._escaping_rows = {
            id(node) for name in escaped if self.locals.get(name, ('',))[0] == 'row'
            for node, nested in self.f.loads.get(name, ()) if not nested}
        # a sub-table local whose value is also needed as a Python table stays a typed handle;
        # the Python table is the bound table of that handle, taken at each such use only
        self._escaping_subs = {
            id(node): name for name in escaped if self.locals.get(name, ('',))[0] == 'sub'
            for node, nested in self.f.loads.get(name, ()) if not nested}

    # ------------------------------------------------------------------ results
    def escapes(self):
        """(locals to keep -> why, locals that escape) found by this attempt."""
        kept, escaped = dict(self.failed), set()
        for name, (kind, _) in self.locals.items():
            if name in self.kept:
                continue
            every = self.f.loads.get(name, ())
            if kind in ('table', 'record', 'index') and not any(id(node) in self.consumed for node, _ in every):
                kept.setdefault(name, 'no read through it is lowered')     # left exactly as written
                continue
            loads = [(node, nested) for node, nested in every if id(node) not in self.consumed]
            if not loads:
                continue
            use = self._use(loads[0][0])
            if kind == 'sub' and any(nested for _, nested in loads):
                kept.setdefault(name, ESCAPES.format(name=name, use=use))
            elif kind == 'row' and any(nested for _, nested in loads):
                kept.setdefault(name, f'row local used in a nested function ({use})')
            else:
                escaped.add(name)
        return kept, escaped

    def _use(self, node):
        parent = self.f.parents.get(id(node))
        while isinstance(parent, (cst.Arg, cst.Element, cst.Index, cst.SubscriptElement)):
            parent = self.f.parents.get(id(parent))
        text = _code(parent if parent is not None and not isinstance(
            parent, (cst.SimpleStatementLine, cst.IndentedBlock)) else node)
        text = ' '.join(text.split())
        return text if len(text) <= 60 else text[:57] + '...'

    def default_reason(self, site):
        if site.kind == 'iterrows':
            return ITERROWS
        if site.kind == 'row':
            return ITERROWS_ROW
        if site.kind == 'index':
            return INDEX_OBJECT
        if site.kind == 'column':
            return COLUMN_VIEW
        return None

    # ------------------------------------------------------------------ names
    def reserve(self, base):
        name = base
        while name in self._used:
            name += '_'
        self._used.add(name)
        return name

    def mtype(self, name):
        return f'{self.o.module}.{name}'

    def m(self, name):
        return cst.Attribute(cst.Name(self.o.module), cst.Name(name))

    # ------------------------------------------------------------------ local classification
    def _classify(self):
        """Locals that hold tables (alias or sub-table), rows or records, each of one kind.

        Every binding must be a plain assignment of the same kind (and, for tables and rows,
        the same schema); any other binding disqualifies the name.
        """
        o, kinds = self.o, {}
        for _ in range(4):      # locals assigned from other locals
            for name, values in self.f.values.items():
                if name in self.f.other or name in o._params or name in kinds or name in self.kept:
                    continue
                found = [self._value_kind(value, kinds) for value in values]
                if not found or any(item is None for item in found):
                    continue
                if len({item[0] for item in found}) != 1:
                    continue
                first = found[0]
                if first[1] is not None and not all(item[1].same(first[1]) for item in found):
                    continue
                kinds[name] = first
        return kinds

    def _value_kind(self, value, kinds):
        """(kind, schema) of an assigned value: 'table' (alias), 'sub', 'row' or 'record'."""
        o = self.o
        schema = o.table_cell(value)
        if schema is not None:
            return ('table', schema)
        if isinstance(value, cst.Name) and value.value in kinds:
            kind, schema = kinds[value.value]
            return ('table', schema) if kind in ('table', 'sub') else (kind, schema) if kind == 'record' else None
        if o.records_enabled() and o.record_cell(value):
            return ('record', None)
        receiver = _index_attr(value, 'index')
        if receiver is not None:        # an index local: years = T.index (W1.9)
            base = self._static_schema(receiver, kinds)
            return ('index', base) if base is not None else None
        parts = _subscript_parts(value)
        receiver = _index_attr(value.value, 'loc') if parts is not None else None
        if receiver is None or len(parts) != 1:
            return None
        base = self._static_schema(receiver, kinds)
        if base is None:
            return None
        count = self._count(parts[0], base)
        if count is None or count > base.levels:
            return None
        return ('row', base) if count == base.levels else ('sub', base.child(count))

    def _static_schema(self, expr, kinds):
        schema = self.o.table_cell(expr)
        if schema is not None:
            return schema
        if isinstance(expr, cst.Name) and expr.value in kinds and kinds[expr.value][0] in ('table', 'sub'):
            return kinds[expr.value][1]
        parts = _subscript_parts(expr)
        receiver = _index_attr(expr.value, 'loc') if parts is not None else None
        if receiver is not None and len(parts) == 1:
            base = self._static_schema(receiver, kinds)
            if base is not None:
                count = self._count(parts[0], base)
                if count is not None and count < base.levels:
                    return base.child(count)
        return None

    def _count(self, key, schema):
        """Number of parts of ``key`` on a table of ``schema``, or None if unknown."""
        items = self._items(key)
        if items is not None:
            return len(items[1])
        if schema.levels == 1 or self.o.scalar(key):
            return 1
        return None

    def _items(self, key):
        """(read-back expressions, source expressions) of a tuple-display key, or of a local
        assigned once to one (its items are read back by position), else None."""
        tuple_ = key
        if isinstance(key, cst.Name):
            values = self.f.values.get(key.value)
            if (values and len(values) == 1 and key.value not in self.f.other
                    and isinstance(values[0], cst.Tuple)):
                tuple_ = values[0]
        if not isinstance(tuple_, cst.Tuple) or any(
                not isinstance(item, cst.Element) or isinstance(item, cst.StarredElement)
                for item in tuple_.elements):
            return None
        sources = [item.value for item in tuple_.elements]
        if tuple_ is key:
            return None, sources
        reads = [cst.Subscript(value=cst.Name(key.value),
                               slice=[cst.SubscriptElement(cst.Index(cst.Integer(str(i))))])
                 for i in range(len(sources))]
        return reads, sources

    # ------------------------------------------------------------------ scopes
    def on_visit(self, node):
        if isinstance(node, SKIPPED_SCOPES):
            if self._root is not None:
                return False
            self._root = node
            if isinstance(node.body, cst.IndentedBlock):
                for index, statement in enumerate(node.body.body):
                    self._top_index[id(statement)] = index
        elif id(node) in self._top_index:
            self._top = self._top_index[id(node)]
        if isinstance(node, COMPREHENSIONS):
            names = set()
            current = node.for_in
            while current is not None:
                names.update(_bound_names(current.target))
                current = current.inner_for_in
            mapping = {}
            current = node.for_in
            while current is not None:      # for k in R.index: k is H.key_int(i), i a position
                target = current.target
                unpack = (isinstance(target, cst.Tuple) and all(
                    isinstance(item, cst.Element) and isinstance(item.value, cst.Name) for item in target.elements))
                if ((isinstance(target, cst.Name) or unpack) and not self._names_in(current.iter) & names
                        and self._static_index(current.iter) is not None):
                    iref = self.index_ref(current.iter)
                    found = None
                    if iref is not None and iref.schema.levels == 1 and isinstance(target, cst.Name):
                        found = {target.value: (iref.handle, None, 'int' if iref.schema.int_key else 'label')}
                    elif (iref is not None and unpack and len(target.elements) == iref.schema.levels
                          and self._unordered(node)):
                        found = {item.value.value: (iref.handle, None, j) for j, item in enumerate(target.elements)}
                    if found is not None:
                        position = self.reserve('_mx_i')
                        mapping.update({name: (handle, position, mode) for name, (handle, _, mode) in found.items()})
                        self._compfor[id(current)] = (iref.handle, position, iref)
                    elif iref is not None:
                        self.decline(iref.sites[0], INDEX_NOT_INT)
                current = current.inner_for_in
            self._shadow.append(names)
            self._keyvars.append(mapping)
        return super().on_visit(node)

    def on_leave(self, original, updated):
        if isinstance(original, COMPREHENSIONS):
            self._shadow.pop()
            self._keyvars.pop()
        return super().on_leave(original, updated)

    def _unordered(self, node):
        """The comprehension is the only argument of min, max, any, all, sorted or set, whose
        result does not depend on the order of the keys."""
        arg = self.f.parents.get(id(node))
        call = self.f.parents.get(id(arg)) if isinstance(arg, cst.Arg) else None
        return (isinstance(node, cst.GeneratorExp) and isinstance(call, cst.Call) and len(call.args) == 1
                and isinstance(call.func, cst.Name) and call.func.value in UNORDERED)

    @staticmethod
    def _names_in(expr):
        found = set()

        class Names(cst.CSTVisitor):
            def visit_Name(inner, node):
                found.add(node.value)
        expr.visit(Names())
        return found

    def leave_CompFor(self, original, updated):
        found = self._compfor.get(id(original))
        if found is None:
            return updated
        handle, position, iref = found
        self.mark(*iref.sites, notes=iref.notes)
        return updated.with_changes(target=cst.Name(position), iter=self._positions(handle))

    def leave_For(self, original, updated):
        target = original.target
        if not isinstance(target, cst.Name):
            return updated
        ints = target.value in self.o._key_locals
        if not ints and not (self.f.binds.get(target.value) == 1 and target.value not in self.o._params):
            return updated
        if self._static_index(original.iter) is None:
            return updated
        iref = self.index_ref(original.iter)
        if iref is None or iref.schema.levels != 1 or ints != iref.schema.int_key:
            return updated
        position = self.reserve('_mx_i')
        self._declare.append((position, '_mx_cy.Py_ssize_t'))
        if ints and (target.value, '_mx_cy.longlong') not in self._declare:
            self._declare.append((target.value, '_mx_cy.longlong'))
        self.mark(*iref.sites, notes=iref.notes)
        key = (_method(iref.handle, 'key_int', cst.Name(position)) if ints
               else _call(self.m('key_label'), iref.handle, cst.Name(position)))
        first = cst.SimpleStatementLine([cst.Assign([cst.AssignTarget(cst.Name(target.value))], key)])
        body = updated.body
        if isinstance(body, cst.IndentedBlock):
            body = body.with_changes(body=[first, *body.body])
        else:
            body = cst.IndentedBlock(body=[first, cst.SimpleStatementLine(body=body.body)])
        return updated.with_changes(target=cst.Name(position), iter=self._positions(iref.handle), body=body)

    def local(self, name):
        """(kind, schema) of a lowered local read here, else None."""
        if any(name in names for names in self._shadow) or name in self.kept:
            return None
        return self.locals.get(name)

    # ------------------------------------------------------------------ handles
    def companion(self, name, type_name='TableHandle', prefix='_mx_h_'):
        handle = self._companions.get(name)
        if handle is None:
            handle = self._companions[name] = self.reserve(prefix + name)
            self._declare.append((handle, self.mtype(type_name)))
        return handle

    def row_names(self, name):
        """(row index name, handle expression) of a row local.

        The handle is the single table its assignments read (a hoisted receiver or a table
        local's companion), else a companion ``_mx_rh_<name>`` assigned with the row."""
        found = self._rows.get(name)
        if found is None:
            index = self.reserve('_mx_row_' + name)
            self._declare.append((index, '_mx_cy.Py_ssize_t'))
            receivers = {}
            for value in self.f.values.get(name, ()):
                receiver = _index_attr(value.value, 'loc')
                receivers.setdefault(_code(receiver), receiver)
            handle, shared = None, False
            if len(receivers) == 1:
                (receiver,) = receivers.values()
                if self.o.table_cell(receiver) is not None:
                    handle = self.hoist(receiver, 'table')
                elif isinstance(receiver, cst.Name) and self.local(receiver.value):
                    handle = self.table_ref(receiver).handle
            if handle is None:
                shared = True
                handle = cst.Name(self.reserve('_mx_rh_' + name))
                self._declare.append((handle.value, self.mtype('TableHandle')))
            found = self._rows[name] = (index, handle, shared)
        return found

    def hoist(self, receiver, kind):
        code = _code(receiver)
        found = self._hoisted.get(code)
        if found is None:
            base = '_mx_h' if kind == 'table' else '_mx_r'
            name = self.reserve(f'{base}{sum(1 for item in self._hoisted.values() if item[2] == kind)}')
            found = self._hoisted[code] = [name, receiver, kind, self._top]
        found[3] = min(found[3], self._top)
        return cst.Name(found[0])

    def table_ref(self, original):
        """_Ref of an original table expression (with a handle expression), else None."""
        schema = self.o.table_cell(original)
        if schema is not None:
            return _Ref('table', schema, self.hoist(original, 'table'))
        if isinstance(original, cst.Name):
            found = self.local(original.value)
            if found is not None and found[0] == 'table':
                return _Ref('table', found[1], cst.Name(self.companion(original.value)), (original,))
            if found is not None and found[0] == 'sub':
                return _Ref('table', found[1], self.sub_handle(original.value), (original,))
            return None
        memo = self._memo.get(id(original))
        return memo if memo is not None and memo.kind == 'table' else None

    def sub_handle(self, name):
        """The handle of a sub-table local: its borrowed companion ``TableRef`` read as a
        ``TableHandle`` (no reference is taken on the shared child; the parent handle, held for
        the whole formula, keeps it alive; D47)."""
        ref = self.companion(name, 'TableRef')
        return _call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                     cst.parse_expression(self.mtype('TableHandle')), cst.Name(ref))

    def record_ref(self, original):
        if not self.o.records_enabled():
            return None
        if self.o.record_cell(original):
            return _Ref('record', None, self.hoist(original, 'record'))
        if isinstance(original, cst.Name):
            found = self.local(original.value)
            if found is not None and found[0] == 'record':
                handle = self.companion(original.value, 'RecordHandle', '_mx_r_')
                return _Ref('record', None, cst.Name(handle), (original,))
        return None

    def row_ref(self, original):
        """_Ref('row') of a row local: its handle, and its row index name in ``extra``."""
        if isinstance(original, cst.Name):
            found = self.local(original.value)
            if found is not None and found[0] == 'row':
                index, handle, _ = self.row_names(original.value)
                return _Ref('row', found[1], handle, (original,), cst.Name(index))
        return None

    def index_ref(self, original):
        """_Ref of an index expression: ``R.index`` on a lowered table, or an index local
        (``years = R.index``, assigned once). ``sites`` are the nodes a lowering consumes."""
        receiver = _index_attr(original, 'index')
        if receiver is not None:
            ref = self.table_ref(receiver)
            if ref is None:
                return None
            return _Ref('table', ref.schema, ref.handle, (original, *ref.sites), notes=ref.notes)
        if isinstance(original, cst.Name):
            found = self.local(original.value)
            if found is None or found[0] != 'index':
                return None
            values = self.f.values.get(original.value) or ()
            if len(values) != 1:
                return None
            ref = self.table_ref(_index_attr(values[0], 'index'))
            if ref is None:
                return None
            return _Ref('table', ref.schema, ref.handle, (original, values[0], *ref.sites), notes=ref.notes)
        return None

    def _static_index(self, expr):
        """Schema of an index expression known before lowering (a table cell, a table or
        sub-table local, an index local), else None."""
        receiver = _index_attr(expr, 'index')
        if receiver is not None:
            if self.o.table_cell(receiver) is not None:
                return self.o.table_cell(receiver)
            if isinstance(receiver, cst.Name):
                found = self.locals.get(receiver.value)
                if receiver.value not in self.kept and found and found[0] in ('table', 'sub'):
                    return found[1]
            return None
        if isinstance(expr, cst.Name) and expr.value not in self.kept:
            found = self.locals.get(expr.value)
            if found and found[0] == 'index' and len(self.f.values.get(expr.value) or ()) == 1:
                return found[1]
        return None

    def _loop_keys(self):
        """Names bound only as the target of ``for k in R.index:`` over a single integer level."""
        found, lowering = set(), self

        class Loops(cst.CSTVisitor):
            def visit_FunctionDef(inner, node):
                return node is lowering.f.root

            def visit_Lambda(inner, node):
                return False

            def visit_For(inner, node):
                target = node.target
                if (isinstance(target, cst.Name) and lowering.f.binds.get(target.value) == 1
                        and target.value not in lowering.o._params):
                    schema = lowering._static_index(node.iter)
                    if schema is not None and schema.int_key:
                        found.add(target.value)
        self.f.root.visit(Loops())
        return found

    def _positions(self, handle):
        return _call(cst.Name('range'), cst.Attribute(handle, cst.Name('rows')))

    def _key_list(self, handle, ints=True):
        """``[H.key_int(i) for i in range(H.rows)]`` (``M.key_label(H, i)`` for labels that are
        not integers): the keys in row order, as the index."""
        index = self.reserve('_mx_i')
        elt = (_method(handle, 'key_int', cst.Name(index)) if ints
               else _call(self.m('key_label'), handle, cst.Name(index)))
        return cst.ListComp(elt=elt, for_in=cst.CompFor(
            target=cst.Name(index), iter=self._positions(handle)))

    def _index_builtin(self, name, iref, original, updated):
        """``len/min/max/list/tuple/sorted(R.index)`` on a lowered table."""
        handle, schema = iref.handle, iref.schema
        site = iref.sites[0]
        info = None
        if name == 'len':
            node, info = cst.Attribute(handle, cst.Name('rows')), ('rows', None, handle, None, None)
        elif name in ('min', 'max'):
            if schema.int_key:
                method = 'first_int' if name == 'min' else 'last_int'
                node, info = _method(handle, method), (method, None, handle, None, None)
            else:
                node = _method(handle, 'first_key' if name == 'min' else 'last_key')
        elif schema.levels != 1:
            self.decline(site, INDEX_NOT_INT)
            return updated
        else:
            node = self._key_list(handle, schema.int_key)
            if name != 'list':
                node = _call(cst.Name(name), node)
        self.mark(*iref.sites, notes=iref.notes)
        return self.out(node, updated, info)

    # ------------------------------------------------------------------ bookkeeping
    def mark(self, *nodes, notes=()):
        self.keys.extend(note for note in notes if note)
        for node in nodes:
            if id(node) in self.o._site_ids:
                self.lowered.add(id(node))
            self.consumed.add(id(node))
        self.typed += 1
        self.changed = True

    def decline(self, node, reason):
        if reason and id(node) in self.o._site_ids and id(node) not in self.lowered:
            self.reasons.setdefault(id(node), reason)

    def out(self, node, updated, info=None):
        if updated.lpar or not isinstance(node, cst.IfExp):     # a guarded read keeps its parentheses
            node = node.with_changes(lpar=updated.lpar, rpar=updated.rpar)
        if info is not None:
            self._kinds[id(node)] = info
        return node

    def parts(self, key_original, key_updated, schema, where, allow_prefix):
        """Key part expressions for ``key`` on ``schema``: (parts, None) or (None, reason)."""
        items = self._items(key_original)
        if items is not None:
            reads, sources = items
            if reads is None:       # a tuple display: its updated elements
                reads = [element.value for element in key_updated.elements] \
                    if isinstance(key_updated, cst.Tuple) and len(key_updated.elements) == len(sources) \
                    else sources
        elif schema.levels == 1 or self.o.scalar(key_original):
            reads, sources = [key_updated], [key_original]
        else:
            return None, 'key is not a tuple display and may hold several parts'
        count = len(reads)
        if not count or count > MAX_PARTS:
            return None, f'{count} key parts'
        if count > schema.levels or (count < schema.levels and not allow_prefix):
            return None, f'{count} key part{"s" if count > 1 else ""} on a table of {schema.levels} levels'
        result, notes = _Parts(), []
        for read, source in zip(reads, sources):
            why = self.o.bare_ok(source)
            info = self._kinds.get(id(read))
            if why is None and info is not None and info[0] in C_INT_METHODS:
                why = f'int (typed handle {info[0]})'
            if why is None:
                read = _call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')), cst.Name('object'), read)
                notes.append('object (cast: not known to be a C integer or an object)')
            else:
                exact = self.o.exact_long_long(source) or (info is not None and info[0] in LONG_LONG_METHODS)
                if (why.startswith('int') or why == 'enum member') and not exact:
                    # the fused key takes exactly long long or object: a C long literal, a
                    # Py_ssize_t or a Python int is converted (an enum member is its value)
                    read = _call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                                 cst.Attribute(cst.Name('_mx_cy'), cst.Name('longlong')), read)
                notes.append(why)
            result.append(read)
        result.note = f'{where}, {count} part{"s" if count > 1 else ""}: ' + '; '.join(notes)
        return result, None

    def column_values(self, expr, depth=0):
        """The possible values of a column expression built only from string literals: a
        literal, a conditional between such expressions, or a local assigned only such
        expressions. None for anything else."""
        if depth > 4:
            return None
        name = _literal_str(expr)
        if name is not None:
            return {name}
        if isinstance(expr, cst.IfExp):
            a, b = self.column_values(expr.body, depth + 1), self.column_values(expr.orelse, depth + 1)
            return a | b if a is not None and b is not None else None
        if isinstance(expr, cst.Name) and expr.value not in self.f.other and expr.value not in self.o._params:
            values = self.f.values.get(expr.value)
            found = set()
            for value in values or ():
                more = self.column_values(value, depth + 1)
                if more is None:
                    return None
                found |= more
            return found or None
        return None

    def column_read(self, ref, row, column, column_original):
        """``H.f64/i64/obj(row, H.col("c"))`` for a column whose possible values are string
        literals of one kind: (node, info) or (None, reason)."""
        names = self.column_values(column_original)
        if names is None:
            return None, 'column is not a string literal (rule 5); ' + PYTHON
        kinds = {ref.schema.columns.get(name) for name in names}
        kind = kinds.pop() if len(kinds) == 1 else None
        method = {'float': 'f64', 'int': 'i64'}.get(kind, 'obj')
        col = _method(ref.handle, 'col', column)
        return _method(ref.handle, method, row, col), (method, kind, ref.handle, row, col)

    def row_of(self, ref, key_original, key_updated, where):
        """Row expression for a complete key of ``ref``: (node, notes) or (None, reason)."""
        info = self._kinds.get(id(key_updated))
        if (info is not None and info[0] in ('key_int', 'key_label') and info[3] is not None and ref.schema.levels == 1
                and _code(info[2]) == _code(ref.handle)):
            return info[3], (f'{where}: the row of the iterated key',)
        if (info is not None and info[0] == 'asof_int' and ref.schema.levels == 1
                and _code(info[2]) == _code(ref.handle)):
            return _method(ref.handle, 'asof_row', info[3]), (f'{where}: asof key as the row',)
        parts, reason = self.parts(key_original, key_updated, ref.schema, where, allow_prefix=False)
        if parts is None:
            return None, reason
        return _method(ref.handle, f'row_{len(parts)}', *parts), (parts.note,)

    def consumer(self, original):
        """How the parent uses an inline selection: 'read' (field, column, loc/at/index),
        'assign' (value of a lowered local), or None."""
        parent = self.f.parents.get(id(original))
        if isinstance(parent, cst.Subscript) and parent.value is original:
            return 'read'
        if isinstance(parent, cst.Attribute) and parent.value is original and parent.attr.value in (
                'loc', 'at', 'index'):
            return 'read'
        if isinstance(parent, (cst.Assign, cst.AnnAssign)) and parent.value is original:
            target = parent.targets[0].target if isinstance(parent, cst.Assign) else parent.target
            if (isinstance(target, cst.Name) and self.local(target.value) is not None
                    and (not isinstance(parent, cst.Assign) or len(parent.targets) == 1)):
                return 'assign'
        return None

    # ------------------------------------------------------------------ reads: subscripts
    def leave_Subscript(self, original, updated):
        parts = _subscript_parts(original)
        value = original.value
        if parts is None:
            for attr in ('loc', 'at'):
                if _index_attr(value, attr) is not None:
                    self.decline(value, 'slice or starred key; ' + PYTHON)
            return updated
        for attr in ('loc', 'at'):
            receiver = _index_attr(value, attr)
            if receiver is not None:
                return self._loc(original, updated, receiver, attr, parts)
        if len(parts) == 1 and self.index_ref(value) is not None:
            return self._key_at(original, updated, value, parts[0])
        one = len(parts) == 1
        new_key = updated.slice[0].slice.value
        row = self.row_ref(value)
        if row is not None:
            if not one:
                return updated
            node, info = self.column_read(row, row.extra, new_key, parts[0])
            if node is None:
                self.decline(original, info)
                return updated
            self.mark(original, *row.sites)
            return self.out(node, updated, info)
        record = self.record_ref(value)
        if record is not None and one and _literal_str(parts[0]) is not None:
            field = _method(record.handle, 'field', new_key)
            self.mark(original, *record.sites)
            return self.out(_method(record.handle, 'obj', field), updated,
                            ('obj', None, record.handle, field, None))
        inner = self._memo.get(id(value))
        if inner is not None and one and inner.kind == 'row':
            ref, row_expr = inner.handle, inner.extra
            node, info = self.column_read(ref, row_expr, new_key, parts[0])
            if node is None:
                self.decline(original, info)
                return updated
            self.mark(original, *inner.sites, notes=inner.notes)
            return self.out(node, updated, info)
        if inner is not None and one and inner.kind == 'column':
            return self._column_key(original, updated, inner, parts[0], new_key)
        ref = self.table_ref(value)
        if ref is None:
            if self.o.untraced_table(value):
                self.decline(original, NO_SCHEMA)
            return updated
        if one and _literal_str(parts[0]) is not None:
            self._memo[id(original)] = _Ref('column', ref.schema, ref, (original, *ref.sites), new_key)
            if self.consumer(original) != 'read':
                self.decline(original, COLUMN_VIEW)
            return updated
        self.decline(original, 'column is not a string literal (rule 5); ' + PYTHON)
        return updated

    def _chained(self, site, ref, key_original, key_updated, column_updated, column_original, where):
        """A complete-row read through an inline selection, ``T.loc[a].loc[b, "c"]`` or
        ``T.loc[a]["c"][b]``: one row of the parent table for the prefix and key parts together,
        so no child handle is taken. (node, info) or None."""
        origin = ref.extra if isinstance(ref.extra, tuple) and ref.extra and ref.extra[0] == 'sub' else None
        if origin is None:
            return None
        parent, prefix = origin[1], origin[2]
        key_parts, _ = self.parts(key_original, key_updated, ref.schema, where, allow_prefix=False)
        if key_parts is None or len(prefix) + len(key_parts) > MAX_PARTS:
            return None
        row = _method(parent.handle, f'row_{len(prefix) + len(key_parts)}', *prefix, *key_parts)
        node, info = self.column_read(parent, row, column_updated, column_original)
        if node is None:
            return None
        self.mark(site, *ref.sites, notes=(*ref.notes, f'{where} through a selection: one row of the parent',
                                           key_parts.note))
        return node, info

    def _column_key(self, original, updated, column, key_original, key_updated):
        """``T["c"][k]``: the read of column c at k (D13)."""
        ref = column.handle
        chained = self._chained(original, ref, key_original, key_updated, column.extra,
                                _subscript_parts(self._column_original(column))[0], 'T["c"][key]')
        if chained is not None:
            self.mark(*column.sites)
            return self.out(chained[0], updated, chained[1])
        row, notes = self.row_of(ref, key_original, key_updated, 'T["c"][key]')
        node = info = None
        if row is not None:
            node, info = self.column_read(ref, row, column.extra, _subscript_parts(
                self._column_original(column))[0])
        if node is None:
            self.decline(original, info if row is not None else notes + '; ' + PYTHON)
            return updated
        self.mark(original, *column.sites, notes=(*ref.notes, *notes))
        return self.out(node, updated, info)

    def _column_original(self, column):
        return column.sites[0]

    def _key_at(self, original, updated, index, key):
        """``T.index[i]``: H.key_int(i) on a single integer level."""
        ref = self.index_ref(index)
        site = original.value
        if ref is None:
            return updated
        new_key = updated.slice[0].slice.value
        info = self._kinds.get(id(new_key))
        if not ref.schema.int_key or not (self.o.is_int(key) or (info and info[0] in C_INT_METHODS)):
            self.decline(site, 'index[i] needs a single integer level and an integer position; ' + PYTHON)
            return updated
        self.mark(site, *ref.sites, notes=ref.notes)
        return self.out(_method(ref.handle, 'key_int', new_key), updated, ('key_int', None, ref.handle, None, None))

    def _loc(self, original, updated, receiver, attr, parts):
        site = original.value
        ref = self.table_ref(receiver)
        if ref is None:
            if isinstance(receiver, cst.Name) and receiver.value in self.kept:
                why = self.kept[receiver.value]
                self.decline(site, why if why.endswith(PYTHON) else KEPT.format(name=receiver.value, why=why))
            elif self.o.untraced_table(receiver):
                self.decline(site, NO_SCHEMA)
            return updated
        key_updated = updated.slice[0].slice.value
        if len(parts) == 2:
            chained = self._chained(site, ref, parts[0], key_updated, updated.slice[1].slice.value, parts[1],
                                    f'{attr}[key, column]')
            if chained is not None:
                return self.out(chained[0], updated, chained[1])
            row, notes = self.row_of(ref, parts[0], key_updated, f'{attr}[key, column]')
            node = info = None
            if row is not None:
                node, info = self.column_read(ref, row, updated.slice[1].slice.value, parts[1])
            elif self._tuple_name(parts[0], ref.schema):
                return self._guarded_read(original, updated, ref, attr, parts, site)
            if node is None:
                self.decline(site, info if row is not None else notes + '; ' + PYTHON)
                return updated
            self.mark(site, *ref.sites, notes=(*ref.notes, *notes))
            return self.out(node, updated, info)
        if len(parts) != 1 or attr != 'loc':
            self.decline(site, f'{attr}[...] with {len(parts)} subscript(s); ' + PYTHON)
            return updated
        count = self._count(parts[0], ref.schema)
        if count is None or count > ref.schema.levels:
            self.decline(site, ('key is not a tuple display and may hold several parts' if count is None
                                else f'{count} key parts on a table of {ref.schema.levels} levels') + '; ' + PYTHON)
            return updated
        use = self.consumer(original)
        if count == ref.schema.levels:      # a row
            row, notes = self.row_of(ref, parts[0], key_updated, 'loc[key] row')
            if row is None:
                self.decline(site, notes + '; ' + PYTHON)
                return updated
            if use is None:                 # used as a value: the row record
                self.mark(site, *ref.sites, notes=(*ref.notes, *notes))
                return self.out(_method(ref.handle, 'row_record', row), updated)
            self._memo[id(original)] = _Ref('row', ref.schema, ref, (site, *ref.sites), row,
                                            (*ref.notes, *notes))
            return updated
        key_parts, reason = self.parts(parts[0], key_updated, ref.schema, 'loc[prefix]', allow_prefix=True)
        if key_parts is None:
            self.decline(site, reason + '; ' + PYTHON)
            return updated
        if use is None:
            parent = self.f.parents.get(id(original))
            if isinstance(parent, cst.Return) and parent.value is original:
                # a table-valued result (D53): the bound table of the shared child handle
                self.mark(site, *ref.sites, notes=(*ref.notes, key_parts.note))
                child = _method(ref.handle, f'sub_{len(key_parts)}', *key_parts)
                return self.out(_method(child, 'table_object'), updated)
            target = getattr(parent, 'targets', None) and parent.targets[0].target
            if isinstance(target, cst.Name) and target.value in self.kept:
                self.decline(site, KEPT.format(name=target.value, why=self.kept[target.value]))
            else:
                self.decline(site, RESULT.format(use=self._use(original)))
            return updated
        handle = _method(ref.handle, f'sub_{len(key_parts)}', *key_parts)
        self._memo[id(original)] = _Ref('table', ref.schema.child(len(key_parts)), handle, (site, *ref.sites),
                                        extra=('sub', ref, key_parts), notes=(*ref.notes, key_parts.note))
        return updated

    def _tuple_name(self, key, schema):
        """A local or parameter key of unknown arity on a table of 2..6 levels."""
        return (isinstance(key, cst.Name) and 2 <= schema.levels <= MAX_PARTS
                and key.value not in self.f.other and not self.o.scalar(key) and not self.o.c_local(key)
                and (key.value in self.o._params or len(self.f.values.get(key.value) or ()) == 1))

    def _python_read(self, ref, updated, attr):
        """The read as written, on the handle's bound table (the local may not exist)."""
        return updated.with_changes(value=cst.Attribute(_method(ref.handle, 'table_object'), cst.Name(attr)),
                                    lpar=[], rpar=[])

    def _guarded_read(self, original, updated, ref, attr, parts, site):
        """``T.loc[k, "c"]`` with k a local of unknown arity: a typed read when k is a tuple of as
        many parts as the table has levels, else the read as written (pandas reads a shorter
        tuple as a prefix, a scalar as one level)."""
        key, levels = parts[0].value, ref.schema.levels
        items = [_call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')), cst.Name('object'),
                       cst.Subscript(cst.Name(key), [cst.SubscriptElement(cst.Index(cst.Integer(str(i))))]))
                 for i in range(levels)]
        row = _method(ref.handle, f'row_{levels}', *items)
        node, info = self.column_read(ref, row, updated.slice[1].slice.value, parts[1])
        if node is None:
            self.decline(site, info)
            return updated
        fallback = self._python_read(ref, updated, attr)
        if info[0] == 'f64':
            fallback = _call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                             cst.Attribute(cst.Name('_mx_cy'), cst.Name('double')), fallback)
        else:
            info = None
        test = cst.BooleanOperation(
            left=cst.Comparison(_call(cst.Name('type'), cst.Name(key)),
                                [cst.ComparisonTarget(cst.Is(), cst.Name('tuple'))]),
            operator=cst.And(),
            right=cst.Comparison(_call(cst.Name('len'), cst.Name(key)),
                                 [cst.ComparisonTarget(cst.Equal(), cst.Integer(str(levels)))]))
        guarded = cst.IfExp(test=test, body=node, orelse=fallback, lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        self.mark(site, *ref.sites, notes=(*ref.notes, f'{attr}[key, column], {levels} parts: '
                                           f'local {key} read by position when it is a tuple of {levels}'))
        result = guarded.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        if info is not None:
            self._kinds[id(result)] = info
        return result

    # ------------------------------------------------------------------ reads: index
    def leave_Comparison(self, original, updated):
        if len(original.comparisons) != 1:
            return updated
        target = original.comparisons[0]
        if not isinstance(target.operator, (cst.In, cst.NotIn)):
            return updated
        ref = self.index_ref(target.comparator)
        if ref is None:
            return updated
        parts, reason = self.parts(original.left, updated.left, ref.schema, 'key in index', allow_prefix=True)
        if parts is None:
            self.decline(target.comparator, reason + '; ' + PYTHON)
            return updated
        call = _method(ref.handle, f'has_{len(parts)}', *parts)
        self.mark(target.comparator, *ref.sites, notes=(*ref.notes, parts.note))
        if isinstance(target.operator, cst.NotIn):
            return cst.UnaryOperation(operator=cst.Not(), expression=call, lpar=updated.lpar, rpar=updated.rpar)
        return self.out(call, updated)

    def leave_Call(self, original, updated):
        func = original.func
        if (isinstance(func, cst.Name) and func.value in ('float', 'int') and len(updated.args) == 1
                and _plain_args(updated)):
            return self._conversion(func.value, updated)
        if (isinstance(func, cst.Name) and func.value in INDEX_BUILTINS and len(original.args) == 1
                and _plain_args(original)):
            iref = self.index_ref(original.args[0].value)
            if iref is not None:
                return self._index_builtin(func.value, iref, original, updated)
            return updated
        if not isinstance(func, cst.Attribute):
            return updated
        ref = self.index_ref(func.value)
        if ref is None:
            if func.attr.value == 'iterrows':
                self.decline(func, ITERROWS)
            return updated
        name = func.attr.value
        site = ref.sites[0]
        if name in ('min', 'max') and not original.args:
            if ref.schema.int_key:
                node = _method(ref.handle, 'first_int' if name == 'min' else 'last_int')
                info = ('first_int', None, ref.handle, None, None)
            else:
                node = _method(ref.handle, 'first_key' if name == 'min' else 'last_key')
                info = None
            self.mark(site, *ref.sites, notes=ref.notes)
            return self.out(node, updated, info)
        if name == 'asof' and len(original.args) == 1 and _plain_args(original):
            value = updated.args[0].value
            known = self._kinds.get(id(value))
            if not ref.schema.int_key:
                self.decline(site, 'asof on a table without a single integer level; ' + PYTHON)
                return updated
            if not (self.o.is_int(original.args[0].value) or (known and known[0] in C_INT_METHODS)):
                key = original.args[0].value
                if isinstance(key, cst.Name) and (key.value in self.o._params or (
                        key.value not in self.f.other and len(self.f.values.get(key.value) or ()) == 1)):
                    # a typed asof when the key is an int at run time, else the helper's asof
                    self.mark(site, *ref.sites, notes=(*ref.notes, f'asof: local {key.value} typed when an int'))
                    typed = _method(ref.handle, 'asof_int', _call(
                        cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                        cst.Attribute(cst.Name('_mx_cy'), cst.Name('longlong')), cst.Name(key.value)))
                    python = _call(cst.Attribute(cst.Attribute(_method(ref.handle, 'table_object'),
                                                               cst.Name('index')), cst.Name('asof')),
                                   cst.Name(key.value))
                    test = cst.Comparison(_call(cst.Name('type'), cst.Name(key.value)),
                                          [cst.ComparisonTarget(cst.Is(), cst.Name('int'))])
                    return cst.IfExp(test=test, body=typed, orelse=python,
                                     lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
                self.decline(site, 'asof key is not known to be an integer; ' + PYTHON)
                return updated
            self.mark(site, *ref.sites, notes=ref.notes)
            return self.out(_method(ref.handle, 'asof_int', value), updated,
                            ('asof_int', None, ref.handle, value, None))
        if name == 'searchsorted':
            return self._searchsorted(original, updated, ref, site)
        self.decline(site, f'index.{name}(...) is not in the typed vocabulary; ' + PYTHON)
        return updated

    def _searchsorted(self, original, updated, ref, site):
        positional = [i for i, arg in enumerate(original.args) if arg.keyword is None and not arg.star]
        keywords = {arg.keyword.value: arg for arg in original.args if arg.keyword is not None}
        side = _literal_str(keywords['side'].value) if 'side' in keywords else 'left'
        if len(positional) != 1 or set(keywords) - {'side'} or any(arg.star for arg in original.args) \
                or side not in ('left', 'right'):
            self.decline(site, 'searchsorted arguments other than (x, side="left"|"right"); ' + PYTHON)
            return updated
        value = updated.args[positional[0]].value
        known = self._kinds.get(id(value))
        if not ref.schema.int_key or not (self.o.is_int(original.args[positional[0]].value)
                                          or (known and known[0] in C_INT_METHODS)):
            self.decline(site, 'searchsorted needs a single integer level and an integer key; ' + PYTHON)
            return updated
        if side == 'left':      # first position with key >= x == first with key > x - 1 (integers)
            if isinstance(value, (cst.BinaryOperation, cst.BooleanOperation, cst.IfExp, cst.Comparison)):
                value = value.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
            value = cst.BinaryOperation(left=value, operator=cst.Subtract(), right=cst.Integer('1'))
        self.mark(site, *ref.sites, notes=ref.notes)
        return self.out(_method(ref.handle, 'searchsorted_right', value), updated,
                        ('searchsorted_right', None, ref.handle, None, None))

    def _conversion(self, name, updated):
        inner = updated.args[0].value
        info = self._kinds.get(id(inner))
        if info is None:
            return updated
        method, kind = info[0], info[1]
        if name == 'float' and method == 'key_label':
            return self.out(_call(self.m('key_f64'), info[2], info[3]), updated)
        if name == 'float' and method == 'f64':
            return self.out(inner, updated, info)
        if name == 'int' and method in C_INT_METHODS:
            return self.out(inner, updated, info)
        if (name == 'int' and method == 'obj' and info[4] is not None
                and (kind in ('int', 'bool') or (isinstance(kind, str) and kind.startswith('enum:')))):
            node = _method(info[2], 'i64', info[3], info[4])
            return self.out(node, updated, ('i64', kind, info[2], info[3], info[4]))
        return updated

    # ------------------------------------------------------------------ names and statements
    def leave_Name(self, original, updated):
        if self._keyvars and id(original) not in self.f._stores:
            for shadow, mapping in zip(reversed(self._shadow), reversed(self._keyvars)):
                if original.value in mapping:
                    handle, position, mode = mapping[original.value]
                    self.changed = True
                    if mode == 'int':
                        return self.out(_method(handle, 'key_int', cst.Name(position)), updated,
                                        ('key_int', None, handle, cst.Name(position), None))
                    label = _call(self.m('key_label'), handle, cst.Name(position))
                    if mode == 'label':
                        return self.out(label, updated, ('key_label', None, handle, cst.Name(position), None))
                    return self.out(cst.Subscript(label, [cst.SubscriptElement(cst.Index(cst.Integer(str(mode))))]),
                                    updated)
                if original.value in shadow:
                    break
        name = self._escaping_subs.get(id(original))
        if name is not None:
            self.changed = True
            return _method(self.sub_handle(name), 'table_object')
        if id(original) in self._escaping_rows:
            index, handle, _ = self.row_names(original.value)
            self.changed = True
            return _method(handle, 'row_record', cst.Name(index))
        return updated

    def leave_Assign(self, original, updated):
        if len(original.targets) != 1 or not isinstance(original.targets[0].target, cst.Name):
            return updated
        return self._assignment(original, updated, original.targets[0].target.value,
                                original.value, updated.value)

    def leave_AnnAssign(self, original, updated):
        if not isinstance(original.target, cst.Name) or original.value is None:
            return updated
        return self._assignment(original, updated, original.target.value, original.value, updated.value)

    def _assignment(self, original, updated, name, value, new_value):
        found = self.local(name)
        if found is None:
            return updated
        kind = found[0]
        statements = []
        if kind in ('table', 'record'):
            type_name, build, prefix = (('TableHandle', 'table_handle', '_mx_h_') if kind == 'table'
                                        else ('RecordHandle', 'record_handle', '_mx_r_'))
            handle = self.companion(name, type_name, prefix)
            if isinstance(value, cst.Name):
                ref = self.table_ref(value) if kind == 'table' else self.record_ref(value)
                if ref is None:
                    self.failed.setdefault(name, f'{value.value} is not lowered')
                    return updated
                expr = ref.handle
                self.consumed.update(id(node) for node in ref.sites)
            elif id(value) in self._memo:
                ref = self._memo[id(value)]
                if ref.kind != 'table':
                    self.failed.setdefault(name, f'{self._use(value)} is not a table')
                    return updated
                expr = ref.handle
                self.mark(*ref.sites, notes=ref.notes)
            else:
                expr = _call(self.m(build), cst.Name(name) if name in self.escaped else new_value)
            if name in self.escaped:
                statements.append(updated)
            statements.append(cst.Assign([cst.AssignTarget(cst.Name(handle))], expr))
            self.changed = True
        elif kind == 'sub':
            ref = self._memo.get(id(value))
            if ref is None or ref.kind != 'table':
                self.failed.setdefault(name, f'the selection {self._use(value)} is not lowered')
                return updated
            handle = self.companion(name, 'TableRef')
            origin = ref.extra if isinstance(ref.extra, tuple) and ref.extra[0] == 'sub' else None
            if origin is None:      # pragma: no cover - every lowered selection carries its origin
                self.failed.setdefault(name, f'the selection {self._use(value)} has no parent handle')
                return updated
            parent, key_parts = origin[1], origin[2]
            borrowed = _call(self.m(f'sub_ref_{len(key_parts)}'), parent.handle, *key_parts)
            statements.append(cst.Assign([cst.AssignTarget(cst.Name(handle))], borrowed))
            self.mark(*ref.sites, notes=ref.notes)
        elif kind == 'index':
            if name in self.escaped:
                return updated
            ref = self.index_ref(value)
            if ref is None:
                self.failed.setdefault(name, f'{self._use(value)} is not lowered')
                return updated
            self.mark(*ref.sites, notes=ref.notes)
            return cst.Pass()
        elif kind == 'row':
            ref = self._memo.get(id(value))
            if ref is None or ref.kind != 'row':
                self.failed.setdefault(name, f'the row {self._use(value)} is not lowered')
                return updated
            index, handle, shared = self.row_names(name)
            if shared:
                statements.append(cst.Assign([cst.AssignTarget(handle)], ref.handle.handle))
            statements.append(cst.Assign([cst.AssignTarget(cst.Name(index))], ref.extra))
            self.mark(*ref.sites, notes=ref.notes)
        else:       # pragma: no cover
            return updated
        if hasattr(updated, 'semicolon'):
            statements = [s.with_changes(semicolon=cst.MaybeSentinel.DEFAULT) for s in statements]
        return cst.FlattenSentinel(statements)

    def leave_FunctionDef(self, original, updated):
        if original is not self._root:
            return updated
        if not self.changed:
            return updated
        body = updated.body
        if not isinstance(body, cst.IndentedBlock):
            body = cst.IndentedBlock(body=[cst.SimpleStatementLine(body=body.body)])
        statements = list(body.body)
        hoisted = sorted(self._hoisted.values(), key=lambda item: (item[3], item[0]))
        inserts = {}
        for name, receiver, kind, top in hoisted:
            type_name, build = ('TableHandle', 'table_handle') if kind == 'table' else ('RecordHandle', 'record_handle')
            line = cst.SimpleStatementLine([cst.AnnAssign(
                target=cst.Name(name), annotation=cst.Annotation(cst.parse_expression(self.mtype(type_name))),
                value=_call(self.m(build), receiver))])
            inserts.setdefault(top, []).append(line)
        declarations = [cst.SimpleStatementLine([cst.AnnAssign(
            target=cst.Name(name), annotation=cst.Annotation(cst.parse_expression(type_)), value=None)])
            for name, type_ in self._declare]
        result = []
        start = 1 if statements and _is_docstring(statements[0]) else 0
        result.extend(statements[:start])
        result.extend(declarations)
        for index, statement in enumerate(statements[start:], start):
            result.extend(inserts.get(index, ()))
            result.append(statement)
        for index in sorted(inserts):
            if index >= len(statements):
                result.extend(inserts[index])
        return updated.with_changes(body=body.with_changes(body=result))
