"""Reads of a traced pandas DataFrame or Series lowered onto a PreparedTable (runtime ``_tables``).

The formula keeps the user's frame: the sample runs on it and the compiled model reads the same object. A read
whose form the compiler recognises is answered from a PreparedTable prepared from that frame object; every other
operation runs on the frame itself.

Recognised here, the receiver ``R`` a self-rooted zero-argument call of a cached Cell whose every traced value is a
pandas DataFrame or Series with a unique index (``tracer.frame_schema``), or a local assigned once from such a call:
``R.loc[key, "c"]`` / ``R.at[key, "c"]`` of a float64, int64 or object column, the key complete (one part per
level); a Series ``R.loc[k]``, ``R.at[k]``, ``R[k]``.

Where a read is lowered:

- in a class whose instances are items (an ItemSpace): in a formula with parameters (evaluated per step), and in
  a formula without parameters (once per item) when the frame is a Cell of a Space without items (prepared once
  per model, not per item);
- in any other class, only in a formula with parameters: a formula without parameters runs once per model.

Binding: in an item class the receiver is evaluated and its PreparedTable found by identity of the frame object
once per item, when the hold is first needed; the item's fields hold the table (``_mx_tp_<n>``) and whether it is
bound (``_mx_tq_<n>``, 0 until bound), and are reset with its per-policy caches. A read then uses the held table
borrowed. In any other class the receiver is evaluated and its table looked up once per call of the formula.

Each lowered read is ``(native if test else pandas)``: the test is true only when the table prepared from the frame
at run time is of the kinds the read was compiled for (its slot, decided once per table), and each key part that
is not a C number or a text literal names one label at run time (``is_label``: a list, a mask, a slice or an
unhashable value is pandas' own). Otherwise the original pandas expression runs, so its value or its error is
live's. An assignment of a new frame is a new object: its PreparedTable is prepared afresh.

Rule: a recogniser or emitter never raises on a shape it does not recognise: it returns None, and the expression
stays pandas. Every unpacking or indexing of a syntax node is preceded by the check of its shape.
"""

import libcst as cst

from .holds import hold_table

PROVIDER = 'modelx_cython._tables'
COLUMN_KIND = {'f64': 0, 'i64': 1}
READ = ('f64', 'i64')
CTYPE = {'f64': '_mx_cy.double', 'i64': '_mx_cy.longlong'}
MAX_PARTS = 6
# displays whose value as a key part selects several rows, never one label
SEVERAL_ROWS = (cst.List, cst.ListComp, cst.Set, cst.Slice, cst.Tuple, cst.Lambda)
MUTATORS = frozenset(('__setitem__', '__delitem__', 'update', 'insert', 'pop'))
# the native reads of a cell value (pandas returns numpy.float64 / numpy.int64 for them)
VALUE_READS = frozenset(('f64_1', 'f64_c', 'i64_1', 'i64_c'))
# calls whose result does not depend on whether the argument is a numpy scalar or a Python number
NUMPY_NEUTRAL_CALLS = frozenset(('float', 'int', 'bool'))
# calls that return one of their numeric arguments (its type goes on)
NUMPY_PASSING_CALLS = frozenset(('abs', 'min', 'max', 'sum'))


def _several(part):
    """A key part whose display selects several rows (or is unpacked): never one label."""
    return isinstance(part, (*SEVERAL_ROWS, cst.StarredElement))


def _code(node):
    return cst.Module(body=()).code_for_node(node)


def _value_read(read):
    """The native read of a cell value in ``read`` (possibly cast to a C type), else None."""
    if (isinstance(read, cst.Call) and _code(read.func) == '_mx_cy.cast' and len(read.args) == 2
            and _code(read.args[0].value) in CTYPE.values()):
        read = read.args[1].value
    if isinstance(read, cst.Call) and isinstance(read.func, cst.Attribute) and read.func.attr.value in VALUE_READS:
        return read
    return None


def _literal_str(node):
    if isinstance(node, cst.SimpleString):
        value = node.evaluated_value
        return value if type(value) is str else None
    return None


def _cast(type_expr, value):
    return cst.Call(func=cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                    args=[cst.Arg(cst.parse_expression(type_expr)), cst.Arg(value)])


def _parts(key):
    """The parts of a key: the elements of a tuple display, else the key itself."""
    return [e.value for e in key.elements] if isinstance(key, cst.Tuple) else [key]


def _index_parts(subscript):
    parts = []
    for element in subscript.slice:
        if not isinstance(element.slice, cst.Index) or element.slice.star is not None:
            return None
        parts.append(element.slice.value)
    return parts


def item_constant(node, cached_call):
    """A key part whose value is fixed for an item until its caches are cleared: an integer literal,
    ``self.<name>`` (a Space parameter or Reference), or ``self.<cell>()`` of a cached zero-argument Cell."""
    if isinstance(node, cst.Integer):
        return True
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name) and node.value.value == 'self':
        return True
    return (isinstance(node, cst.Call) and not node.args and isinstance(node.func, cst.Attribute)
            and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'
            and cached_call(node))


class _Hold:
    """One (receiver, column, kind) a formula reads natively; ``table`` (and ``state`` in items) are the fields of
    the receiver's binding, shared by every read of that receiver."""

    def __init__(self, table, state, receiver, label, kind):
        self.table, self.state, self.receiver, self.label, self.kind = table, state, receiver, label, kind


class _EveryReceiver(frozenset):
    """The receivers a formula edits when an edit goes through a name that may be any of them: all."""

    def __contains__(self, item):
        return True


_NO_LABEL = object()


class FrameReads:
    """``lower(node)`` returns the formula with recognised reads lowered (the node itself when none is)."""

    def __init__(self, resolver, cls_info, cells, providers, *, builtin_int, builtin_float):
        self.resolver, self.cls_info, self.cells = resolver, cls_info, cells
        self.providers = providers
        self.builtin_int, self.builtin_float = builtin_int, builtin_float
        self.key_count = 0
        self.local_count = 0

    # ------------------------------------------------------------------ facts
    def _facts(self, node):
        """The key-typing facts (is_int, exact_long_long) for this formula."""
        from .key_facts import KeyFacts, _Formula
        facts = KeyFacts(self.resolver, self.cls_info, self.cells, builtin_int=self.builtin_int)
        facts.formula = _Formula(node)
        node.visit(facts.formula)
        facts._params = set(getattr(self.cells, 'params', ()) or ())
        return facts

    def numpy_use(self, node, kind, seen=None):
        """Whether the value of the cell read ``node`` (an original node of this formula) reaches a use where a
        numpy scalar and a C number give different values without raising: ``round`` or a method of it, an
        integer power, a container, an argument of a call other than ``float``/``int``/``bool``, ``math.*`` or a Cell
        of the model, a return of a Cell that is not of a C type. Through ``+ - * / // %`` (equal values; a
        division by zero raises where numpy gives inf or nan), unary minus, ``abs``/``min``/``max``/``sum``,
        conditionals and locals the value keeps its type (live: numpy's); a comparison, a truth test, a key part,
        ``float()``/``int()`` and a C result end it."""
        parents = self.facts.formula.parents
        seen = set() if seen is None else seen
        current = node
        while True:
            parent = parents.get(id(current))
            if isinstance(parent, cst.BinaryOperation):
                op = parent.operator
                if isinstance(op, cst.Power):
                    if kind != 'f64' and not (current is parent.left and isinstance(parent.right, cst.Integer)):
                        return True
                elif not isinstance(op, (cst.Add, cst.Subtract, cst.Multiply, cst.Divide, cst.FloorDivide,
                                         cst.Modulo)):
                    return True     # (a division by zero raises where numpy gives inf or nan: compiled raises)
            elif isinstance(parent, cst.UnaryOperation):
                if isinstance(parent.operator, cst.Not):
                    return False
                if not isinstance(parent.operator, (cst.Minus, cst.Plus)):
                    return True
            elif isinstance(parent, (cst.ComparisonTarget, cst.Comparison)):
                return False
            elif isinstance(parent, (cst.If, cst.While)):
                return False
            elif isinstance(parent, cst.IfExp):
                if current is parent.test:
                    return False
            elif isinstance(parent, cst.BooleanOperation):
                pass
            elif isinstance(parent, (cst.Index, cst.SubscriptElement)):
                return False        # a key part: looked up by its value
            elif isinstance(parent, cst.Arg):
                call = parents.get(id(parent))
                if not isinstance(call, cst.Call) or parent.keyword is not None or parent.star:
                    return True
                func = call.func
                if isinstance(func, cst.Name) and func.value in NUMPY_NEUTRAL_CALLS:
                    return False
                if isinstance(func, cst.Name) and func.value in NUMPY_PASSING_CALLS:
                    parent = call
                elif (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                      and func.value.value == 'math'):
                    return False
                elif (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                      and func.value.value == 'self' and self.cached_call(cst.Call(func=func))):
                    return False    # a Cell of the model: its C parameter, as upstream
                else:
                    return True
            elif isinstance(parent, cst.GeneratorExp) and current is parent.elt:
                arg = parents.get(id(parent))
                call = parents.get(id(arg))
                if not (isinstance(arg, cst.Arg) and isinstance(call, cst.Call) and isinstance(call.func, cst.Name)
                        and call.func.value in NUMPY_PASSING_CALLS):
                    return True
                parent = call
            elif isinstance(parent, cst.Return):
                try:
                    result = self.cells.get_rettype_expr()
                except Exception:
                    result = ''
                return not (result.startswith('_mx_cy.') and result != '_mx_cy.object')
            elif isinstance(parent, cst.Expr):
                return False
            elif isinstance(parent, (cst.Assign, cst.AnnAssign, cst.AugAssign)) and current is parent.value:
                if isinstance(parent, cst.AugAssign) and not isinstance(
                        parent.operator, (cst.AddAssign, cst.SubtractAssign, cst.MultiplyAssign, cst.DivideAssign,
                                          cst.FloorDivideAssign, cst.ModuloAssign)):
                    return True
                targets = ([t.target for t in parent.targets] if isinstance(parent, cst.Assign) else [parent.target])
                if len(targets) != 1 or not isinstance(targets[0], cst.Name):
                    return True
                name = targets[0].value
                if name in seen:
                    return False
                seen.add(name)
                loads = self.facts.formula.loads.get(name) or ()
                return any(nested or self.numpy_use(load, kind, seen) for load, nested in loads)
            else:
                return True
            current = parent

    def _once(self, name):
        """``name`` is bound once by an assignment of the formula (not a parameter, a loop target or another binding)."""
        f = self.facts.formula
        return f.binds.get(name, 0) == 1 and name not in f.other and name not in self.facts._params

    def _value_of(self, name):
        """The value of the local ``name``: the one value of a local bound once, or the call of a local whose
        every assignment is the same zero-argument call (``tbl = data.T()`` in several branches), else None."""
        values = self.facts.formula.values.get(name) or ()
        if len(values) == 1 and self._once(name):
            return values[0]
        f = self.facts.formula
        if (len(values) > 1 and f.binds.get(name, 0) == len(values) and name not in f.other
                and name not in self.facts._params and all(isinstance(v, cst.Call) and not v.args for v in values)
                and len({_code(v) for v in values}) == 1):
            return values[0]
        return None

    def _cell_of_call(self, expr):
        """The fqname of the cached Cell a self-rooted zero-argument call ``expr`` calls, else None."""
        if not (isinstance(expr, cst.Call) and not expr.args):
            return None
        from .usage import RESOLVED, _extract_self_chain
        chain = _extract_self_chain(expr.func)
        if not isinstance(chain, list) or not chain:
            return None
        status, name = self.resolver.resolve_chain(self.cls_info.fqname, chain)
        if status != RESOLVED or name not in getattr(self.resolver, 'cached_cells', ()):
            return None
        return name

    def _schema_of_call(self, expr):
        name = self._cell_of_call(expr)
        return getattr(self.resolver, 'frame_schemas', {}).get(name) if name is not None else None

    def shared(self, call):
        """The frame ``call`` is a Cell of a Space without items: prepared once per model, not per item."""
        name = self._cell_of_call(call)
        return name is not None and name in getattr(self.resolver, 'shared_frames', ())

    def _receiver(self, expr):
        """(cell call, schema) when ``expr`` is a frame receiver, else None. In an item class a frame the item
        computes itself is prepared per item (about 0.3-2M instructions; a pandas read costs 20-100k): it is a
        receiver only for a formula with parameters, evaluated once per step, so that the preparation pays for
        itself. Decided from the text, not from how often the sample evaluated the formula."""
        schema = self._schema_of_call(expr)
        found = (expr, schema) if schema is not None else None
        if found is None:
            value = self._value_of(expr.value) if isinstance(expr, cst.Name) else None
            schema = self._schema_of_call(value) if value is not None else None
            found = (value, schema) if schema is not None else None
        if found is not None and self.per_item and not self.cells.has_args() and not self.shared(found[0]):
            return None
        return found

    # ------------------------------------------------------------------ driver
    def lower(self, node):
        params = node.params.params
        if not params or params[0].name.value != 'self' or not getattr(self.resolver, 'frame_schemas', None):
            return node
        self.per_item = self.cls_info.name in self.cls_info.visitor.itemspace_classes
        if not self.per_item and not self.cells.has_args():
            return node         # evaluated once per model: its reads stay on pandas
        self.facts = self._facts(node)
        from .native_sources import module_name
        self.module = module_name(PROVIDER)
        self.holds = {}         # (receiver code, label, kind) -> _Hold
        self.bindings = {}      # receiver code -> its binding (_Hold of the table field)
        self.code_fields = {}
        self.mutated = self._mutated(node)
        self.declarations = []  # C locals of the formula
        rewriter = _Rewrite(self)
        body = []
        for statement in node.body.body:
            self.new_items = []
            new = statement.visit(rewriter)
            if self.new_items:
                body.extend(self._hold_statements(item) for item in self.new_items)
            body.append(new)
        if not self.holds:      # every lowered read takes a hold: nothing was lowered
            return node
        body = self._drop_unused_locals(body, rewriter.replaced_names | set(rewriter.key_locals))
        if self.declarations:
            from .key_facts import _is_docstring
            head = 1 if body and _is_docstring(body[0]) else 0
            body[head:head] = [cst.parse_statement(f'{name}: _mx_cy.Py_ssize_t') for name in self.declarations]
        self.providers.add(PROVIDER)
        return node.with_changes(body=node.body.with_changes(body=self._flatten(body)))

    def _root(self, expr):
        while isinstance(expr, (cst.Attribute, cst.Subscript)):
            expr = expr.value
        return expr

    def _mutated(self, node):
        """Codes of the receivers the formula itself edits in place (an assignment or ``del`` through them,
        ``__setitem__``/``__delitem__``/``update``/``insert``/``pop``, or any call with ``inplace``): their
        reads stay pandas, so an edit is read as it is made, as live. An edit through a local bound to a receiver
        (``w = df``) edits the receiver; one through a name bound in a way not followed (a loop target, unpacking,
        ``self``) may edit any of them: every read of the formula then stays pandas."""
        owner, found, f = self, set(), self.facts.formula
        unknown = []

        def note(expr, seen=()):
            root = owner._root(expr)
            got = owner._receiver(root)
            if got is not None:
                found.add(_code(got[0]))
            if not isinstance(root, cst.Name) or root.value in seen:
                return
            if root.value == 'self' or root.value in f.other:
                unknown.append(root.value)
            for value in f.values.get(root.value, ()):
                note(value, (*seen, root.value))

        class Scan(cst.CSTVisitor):
            def visit_Assign(self, n):
                for target in n.targets:
                    if isinstance(target.target, (cst.Attribute, cst.Subscript)):
                        note(target.target)

            def visit_AugAssign(self, n):
                if isinstance(n.target, (cst.Attribute, cst.Subscript)):
                    note(n.target)

            def visit_AnnAssign(self, n):
                if isinstance(n.target, (cst.Attribute, cst.Subscript)):
                    note(n.target)

            def visit_Del(self, n):
                note(n.target)

            def visit_Call(self, n):
                if isinstance(n.func, cst.Attribute) and (
                        n.func.attr.value in MUTATORS
                        or any(a.keyword is not None and a.keyword.value == 'inplace' for a in n.args)):
                    note(n.func.value)
        node.visit(Scan())
        return _EveryReceiver() if unknown else found

    @staticmethod
    def _flatten(body):
        out = []
        for item in body:
            out.extend(item if isinstance(item, list) else [item])
        return out

    @staticmethod
    def _simple_assign(statement):
        """(name, value) of a statement ``name = value``, else None."""
        if (isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1
                and isinstance(statement.body[0], cst.Assign) and len(statement.body[0].targets) == 1
                and isinstance(statement.body[0].targets[0].target, cst.Name)):
            return statement.body[0].targets[0].target.value, statement.body[0].value
        return None

    def _drop_unused_locals(self, body, replaced):
        """A frame local whose every use was lowered is no longer evaluated (no Cell call, no reference). Repeated
        until nothing more is dropped."""
        out = self._flatten(body)
        while True:
            loads = {}

            class Loads(cst.CSTVisitor):
                def visit_Name(self, name):
                    loads[name.value] = loads.get(name.value, 0) + 1
            cst.Module(body=out).visit(Loads())
            kept = [statement for statement in out
                    if not ((self._simple_assign(statement) or (None,))[0] in replaced
                            and loads.get(self._simple_assign(statement)[0], 0) == 1)]     # the target itself only
            if len(kept) == len(out):
                return out
            out = kept

    def _hold_statements(self, hold):
        """The statements that resolve one hold, placed before the top-level statement of its first read. That
        statement may not reach the read (a branch not taken): a hold whose evaluation raises is left unresolved (a
        table: one whose every read stays pandas), so that the error is raised by the read's own pandas branch, where
        live raises it, or not at all."""
        m = self.module
        if isinstance(hold, tuple):         # a held code or row, after the hold of its table
            what, field, table, level_or_key, extra = hold
            value = self._held_value(what, table, level_or_key, extra)
            return [cst.parse_statement(f'if self.{field} == 0:\n    try:\n        self.{field} = {value}\n'
                                        f'    except Exception:\n        pass\n')]
        # one binding per receiver and item (or per call outside items): the frame's PreparedTable, or a table whose
        # every slot is -1; each read then takes its slot from the table by its signature id
        receiver = _code(hold.receiver)
        if self.per_item:
            return [cst.parse_statement(
                f'if self.{hold.state} == 0:\n'
                f'    try:\n'
                f'        self.{hold.table} = {m}.bind_ref({receiver})\n'
                f'        self.{hold.state} = 1\n'
                f'    except Exception:\n'
                f'        self.{hold.table} = {m}.bind_ref(None)\n')]
        return [cst.parse_statement(f'try:\n    {hold.table} = {m}.bind_ref({receiver})\n'
                                    f'except Exception:\n    {hold.table} = {m}.bind_ref(None)\n')]

    def _held_value(self, what, table, level_or_key, extra):
        """The value a held field takes: one plus a level code."""
        return f'{self.module}.hold_code(self.{table}, {level_or_key}, {_code(extra)})'

    def read_signature(self, hold):
        """The id of the kinds a read is compiled for (index levels; every column for a row read by field; the
        column read and its kind), whose slot each prepared table decides once."""
        schema = self._schema_of_call(hold.receiver) or {}
        levels = ''.join('i' if kind == 'int' else 'o' for kind in schema.get('levels', ()))
        kinds = {'f64': 0, 'i64': 1, 'obj': 2}
        columns = (tuple((name, kinds[kind]) for name, kind in schema.get('columns', ()))
                   if getattr(hold, 'row_columns', False) else None)
        label = hold.label if hold.label is not _NO_LABEL else None
        return self.signature(levels, columns, label, hold.kind)

    def signature(self, levels, columns, label, kind):
        """The model-wide id of a read signature (written to ``_mx_table_signatures.py`` at the end of the build)."""
        table = self.resolver.__dict__.setdefault('table_signatures', {})
        return table.setdefault((levels, columns, label, kind), len(table))

    def cached_call(self, call):
        """A self-rooted zero-argument call of a cached Cell (as the transformer's ``_cached_call``)."""
        name = call.func.attr.value if isinstance(call.func, cst.Attribute) else None
        return name is not None and any(str(c).endswith('._f_' + name)
                                        for c in getattr(self.resolver, 'cached_cells', ()))

    def item_fixed(self, part, level_kind=None):
        """A key part whose value is fixed per item (held as a code), in a per-step cell of an item class."""
        if isinstance(part, cst.Integer) or not (self.per_item and self.cells.has_args()):
            return False            # an integer literal is passed as it is (a C integer)
        # a text literal and a field of a cached row cell are fixed too, unless the read takes the part as a
        # C integer (key_part): their code is held per item instead of looked up at every read
        return item_constant(part, self.cached_call) or (
            self._fixed_value(part) and not (level_kind == 'int' and self.facts.is_int(part)))

    def _fixed_value(self, part):
        """A text literal, or ``self.R()["f"]`` with R a cached zero-argument cell (a field of a row cell): values
        that do not change while the item's caches hold."""
        if isinstance(part, cst.SimpleString):
            return True
        parts = _index_parts(part) if isinstance(part, cst.Subscript) else None
        return (parts is not None and len(parts) == 1 and _literal_str(parts[0]) is not None
                and isinstance(part.value, cst.Call) and item_constant(part.value, self.cached_call))

    def code_field(self, hold, level, part, emit=None):
        """The item field holding the code of ``part`` (``emit``, the part with its own reads lowered, is what the hold
        evaluates)."""
        key = (hold.table, level, _code(part))
        fields = self.code_fields
        if key not in fields:
            fields[key] = hold_table(self.cls_info).field('prepared_code', key)
            self.new_items.append(('code', fields[key], hold.table, level, part if emit is None else emit))
        return fields[key]

    def hold(self, receiver, label, kind):
        """The read (receiver, column, kind): its table is the receiver's binding, one per receiver."""
        code = _code(receiver)
        key = (code, label, kind)
        found = self.holds.get(key)
        if found is None:
            binding = self.bindings.get(code)
            if binding is None:
                if self.per_item:
                    table = hold_table(self.cls_info).field('prepared', code)
                    state = hold_table(self.cls_info).field('prepared_slot', code)
                else:
                    table, state = f'_mx_pt{self.local_count}', None
                    self.local_count += 1
                binding = self.bindings[code] = _Hold(table, state, receiver, None, None)
                self.new_items.append(binding)
            found = self.holds[key] = _Hold(binding.table, binding.state, receiver, label, kind)
        return found

    def ref(self, hold):
        where = f'self.{hold.table}' if self.per_item else hold.table
        return _cast(f'{self.module}.PreparedRef', cst.parse_expression(where))

    def slot(self, hold):
        """The slot of a read: one C load from the held table by the read's signature id."""
        return cst.Call(func=cst.Attribute(cst.Name(self.module), cst.Name('slot')),
                        args=[cst.Arg(self.ref(hold)), cst.Arg(cst.Integer(str(self.read_signature(hold))))])

    def key_part(self, part, level_kind, emit=None):
        """Emit a key part for typed dispatch on integer levels and object lookup elsewhere.

        The facts apply to the original ``part``. ``emit`` is the same part with its own reads lowered.
        """
        emit = part if emit is None else emit
        if level_kind == 'int':
            # Cython types int() as an object. Give it a C type only when the input fits:
            # casting int() of a C double directly to long long can skip Python's conversion checks.
            inner = self.facts.int_argument(part)
            if inner is not None and self.facts.fits_long_long(inner):
                return _cast('_mx_cy.longlong', emit), 'typed'
            return emit, 'typed'
        return _cast('object', emit), 'object'


# attributes of a pandas object whose subscript is a pandas selection (a bool key is passed as the Python bool)
PANDAS_INDEXERS = frozenset(('index', 'iloc', 'loc', 'at', 'iat', 'columns'))


class _Rewrite(cst.CSTTransformer):
    """Rewrites reads inside one top-level statement; nested functions and lambdas are not entered."""

    def __init__(self, owner):
        super().__init__()
        self.o = owner
        self.replaced_names = set()
        self.key_locals = {}    # name of a key local -> its tuple display (the fallback evaluates the display)
        self.native = {}        # id(lowered read) -> column kind, for float()/int() around it
        self.updated_of = {}    # id(original node) -> its rewritten node (key parts are emitted with their reads lowered)

    def on_leave(self, original, updated):
        result = super().on_leave(original, updated)
        kind = self.native.get(id(result))
        if (kind in CTYPE and result is not updated and isinstance(original, cst.Subscript)
                and isinstance(result, cst.IfExp) and _value_read(result.body) is not None
                and self.o.numpy_use(original, kind)):
            del self.native[id(result)]
            result = self._boxed(result, kind)
        self.updated_of[id(original)] = result
        return result

    def visit_FunctionDef(self, node):
        return False

    def visit_Lambda(self, node):
        return False

    def visit_ClassDef(self, node):
        return False

    def _take(self, receiver, label, kind):
        return self.o.hold(receiver, label, kind)

    def _key(self, key, schema):
        """The parts of a complete key on ``schema``, or None. A key held in a local assigned once to a tuple
        display is that display."""
        levels = schema['levels']
        value = self.o._value_of(key.value) if isinstance(key, cst.Name) and len(levels) > 1 else None
        if isinstance(value, cst.Tuple):
            self.key_locals[key.value] = key = value
        if len(levels) == 1:
            if isinstance(key, cst.Tuple):
                return None
            parts = [key]
        else:
            if not isinstance(key, cst.Tuple) or any(isinstance(e, cst.StarredElement) for e in key.elements):
                return None
            parts = [e.value for e in key.elements]
            if len(parts) != len(levels):
                return None     # a prefix: pandas
        if len(parts) > MAX_PARTS or any(isinstance(part, SEVERAL_ROWS) for part in parts):
            return None
        return parts

    def _read_args(self, hold, parts, levels):
        """(key arguments, extra native conditions) for a read of ``parts`` through ``hold``: one level code per part,
        ``held - 1`` for a part fixed per item (and its condition that the part is present), ``lcode``/``ocode`` of
        the part otherwise."""
        conditions = []
        codes = [self._held(hold, level, part, conditions) if self.o.item_fixed(part, kind)
                 else self.code_of(hold, level, part, kind, conditions)
                 for level, (part, kind) in enumerate(zip(parts, levels))]
        return self.code_args(codes), conditions

    def _detached(self, expr):
        """A key from outside this formula's tree, its reads lowered for a hold to evaluate once per item:
        ``self.R()["f"]``, a field of a row cell. ``expr`` itself when it has no such read."""
        found = []

        class Fields(cst.CSTVisitor):
            def visit_Subscript(inner, node):
                if isinstance(node.value, cst.Call) and self.o._fixed_value(node):
                    found.append(node)
        expr.visit(Fields())
        return expr.visit(self) if found else expr

    def _held(self, hold, level, part, conditions):
        """The code of a key part fixed per item, held per item (``held - 1``); ``conditions`` gets its presence."""
        emit = self.updated_of.get(id(part))
        field = self.o.code_field(hold, level, part, emit if emit is not None else self._detached(part))
        conditions.append(cst.parse_expression(f'self.{field} > 0'))
        return cst.parse_expression(f'self.{field} - 1')

    @staticmethod
    def code_args(codes):
        """(function suffix, arguments) of a read by level codes: ``_1`` with the one code, else ``_c`` with the count
        and six codes (unused ones 0)."""
        if len(codes) == 1:
            return '_1', codes
        return '_c', [cst.Integer(str(len(codes))), *codes, *[cst.Integer('0')] * (MAX_PARTS - len(codes))]

    def code_of(self, hold, level, part, kind, conditions):
        """The level code of one key part computed per read: ``kcode`` of a C typed part on an int level, else
        ``ocode`` of the Python value (looked up as pandas does)."""
        o = self.o
        arg, how = o.key_part(part, kind, self.updated_of.get(id(part)))
        if ((how == 'typed' and not o.facts.is_int(part) and o.facts.int_argument(part) is None)
                or (how == 'object' and _literal_str(part) is None)):
            # not known to be one label from the text (a C integer, a text literal): the read is native only while the
            # part names one label (a list or a mask selects rows, as pandas does whether or not the sample saw one).
            # The test is decided at compile time when Cython gives the part a C type, else it looks at the value;
            # the part is evaluated once, into a local the read then uses
            o.key_count += 1
            name = f'_mx_{"key" if how == "typed" else "okey"}_{o.key_count}'
            conditions.append(cst.parse_expression(f'{o.module}.is_label(({name} := {_code(arg)}))'))
            arg = cst.Name(name)
        return self._m({'typed': 'kcode'}.get(how, 'ocode'), o.ref(hold), cst.Integer(str(level)), arg)

    def _native_test(self, hold, conditions):
        test = self._positive(self.o.slot(hold))
        for condition in conditions:
            test = cst.BooleanOperation(left=test, operator=cst.And(), right=condition)
        return test

    def _m(self, name, *args):
        return cst.Call(func=cst.Attribute(cst.Name(self.o.module), cst.Name(name)), args=[cst.Arg(a) for a in args])

    @staticmethod
    def _next(expr):
        return cst.BinaryOperation(expr, cst.Add(), cst.Integer('1'))

    @staticmethod
    def _positive(expr):
        return cst.Comparison(expr, [cst.ComparisonTarget(cst.GreaterThan(), cst.Integer('0'))])

    @staticmethod
    def _conjunction(conditions):
        test = conditions[0]
        for condition in conditions[1:]:
            test = cst.BooleanOperation(left=test, operator=cst.And(), right=condition)
        return test

    def _boxed(self, lowered, kind):
        """A lowered cell value whose use meets numpy's rules: the native branch boxed as the numpy scalar pandas
        returns, the pandas branch as pandas returns it (not cast); no longer a C value."""
        read, fallback = lowered.body, lowered.orelse
        if (isinstance(fallback, cst.Call) and _code(fallback.func) == '_mx_cy.cast' and len(fallback.args) == 2
                and _code(fallback.args[0].value) in CTYPE.values()):
            fallback = fallback.args[1].value
        if kind == 'f64':
            boxed = self._m('f64_box', read)
        else:
            field = _value_read(read).func.attr.value == 'i64_row'
            boxed = self._m('i64_box', _value_read(read).args[0].value, read, cst.Name('True' if field else 'False'))
        return lowered.with_changes(body=boxed, orelse=fallback)

    def _lowered(self, test, read, fallback, kind=None):
        """``read if test else fallback``: the pandas read cast to the read's C type when the read is typed (both
        branches native), and the lowered read marked as of that kind (``float()``/``int()`` around it dropped). A
        read whose test checks that a text-level key part names one label (``_mx_okey_``) stays a Python object: its
        pandas branch may be a selection (a list key gives a Series)."""
        if '_mx_okey_' in _code(test):
            kind = None
        if kind in CTYPE:
            fallback = _cast(CTYPE[kind], fallback)
        lowered = cst.IfExp(test=test, body=read, orelse=fallback, lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        if kind in CTYPE:
            self.native[id(lowered)] = kind
        return lowered

    @staticmethod
    def _pandas_value(lowered):
        """The pandas branch of a read lowered to a C scalar, not cast to the C type: the numpy scalar live has."""
        pandas = lowered.orelse
        if (isinstance(pandas, cst.Call) and _code(pandas.func) == '_mx_cy.cast' and len(pandas.args) == 2
                and _code(pandas.args[0].value) in CTYPE.values()):
            pandas = pandas.args[1].value
        return pandas.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])

    def _fallback(self, updated, call):
        """The original pandas read on the frame itself: a frame local replaced by its Cell call, a key local by
        its tuple display (so neither local is evaluated on the native path)."""
        frames, keys = self.replaced_names, self.key_locals

        class Swap(cst.CSTTransformer):
            def leave_Name(self, original, new):
                if original.value in frames:
                    return call
                if original.value in keys:
                    return keys[original.value]
                return new
        return updated.visit(Swap()) if (frames or keys) else updated

    def _plain(self, call, schema):
        """The frame ``call`` is read as stored: not edited by the formula, of a unique index."""
        return _code(call) not in self.o.mutated and bool(schema.get('unique'))

    # ------------------------------------------------------------------ reads
    def leave_Subscript(self, original, updated):
        o = self.o
        value = original.value
        if self.native.get(id(updated.value)) in CTYPE and isinstance(updated.value, cst.IfExp):
            # a subscript of a read lowered to a C scalar: the pandas value, which pandas subscripts (or raises on)
            # as live does
            return updated.with_changes(value=self._pandas_value(updated.value))
        parts = _index_parts(original)
        if parts is None:
            return updated
        return self.keyed_read(original, updated, value, parts) or updated

    def keyed_read(self, original, updated, value, parts):
        """``R.loc[key, "c"]`` / ``R.at[key, "c"]`` on a frame; ``R.loc[k]`` / ``R.at[k]`` / ``R[k]`` on a Series."""
        o = self.o
        if isinstance(value, cst.Attribute) and value.attr.value in ('loc', 'at'):
            found = o._receiver(value.value)
            form = value.attr.value
        else:
            found = o._receiver(value)
            form = 'item'
        if found is None:
            return None
        call, schema = found
        if not self._plain(call, schema):
            return None
        if schema['kind'] == 'series':
            if len(parts) != 1:
                return None
            key, label = parts[0], None
            if form == 'item' and schema['levels'] != ('int',) and o.facts.is_int(key):
                return None     # an integer key on a Series without an integer index: by position (pandas)
        else:
            if form == 'item' or len(parts) != 2:
                return None     # a column, a row, or another selection: pandas
            key, label = parts[0], _literal_str(parts[1])
            if label is None:
                return self.named_read(original, updated, call, schema, key, parts[1])
        kind = dict(schema['columns']).get(label)
        receiver = value.value if form != 'item' else value
        if kind == 'obj':
            return self.object_read(updated, receiver, call, schema, key, label)
        if kind not in READ:
            return None
        return self._column_read(updated, call, schema, key, label, kind, receiver)

    def _column_read(self, updated, call, schema, key, label, kind, receiver):
        """A typed read of column ``label`` at a complete ``key`` of the frame ``call``; None when the key is not
        one."""
        o = self.o
        key_parts = self._key(key, schema)
        if key_parts is None:
            return None
        hold = self._take(call, label, COLUMN_KIND[kind])
        if isinstance(receiver, cst.Name):
            self.replaced_names.add(receiver.value)
        (suffix, args), conditions = self._read_args(hold, key_parts, schema['levels'])
        read = self._m(kind + suffix, o.ref(hold), o.slot(hold), *args)
        return self._lowered(self._native_test(hold, conditions), read, self._fallback(updated, call), kind)

    def object_read(self, updated, receiver, call, schema, key, label):
        """``R.loc[key, "c"]`` of an object column (or ``S[k]`` of an object Series): the object pandas returns,
        the stored value (a new reference)."""
        o = self.o
        key_parts = self._key(key, schema)
        if key_parts is None:
            return None
        hold = self._take(call, label, 2)
        if isinstance(receiver, cst.Name):
            self.replaced_names.add(receiver.value)
        (suffix, args), conditions = self._read_args(hold, key_parts, schema['levels'])
        row = self._next(self._m('row' + suffix, o.ref(hold), *args))
        read = self._m('object_field', o.ref(hold), o.slot(hold), row)
        return self._lowered(self._native_test(hold, conditions), read, self._fallback(updated, call))

    def named_read(self, original, updated, call, schema, key, name):
        """``float(R.loc[key, name])`` with the column chosen at run time: the stored value of the column ``name``
        names, as a float (KeyError as pandas for a name that is not a column). Native only while the table has every
        column of the traced kinds (the row is read whole: kind -4)."""
        o = self.o
        up = o.facts.formula.parents.get(id(original))
        consumer = o.facts.formula.parents.get(id(up)) if isinstance(up, cst.Arg) else None
        if not (schema['kind'] == 'frame' and isinstance(consumer, cst.Call) and isinstance(consumer.func, cst.Name)
                and consumer.func.value == 'float' and len(consumer.args) == 1 and o.builtin_float):
            return None
        key_parts = self._key(key, schema)
        if key_parts is None:
            return None
        hold = self._take(call, _NO_LABEL, -4)
        hold.row_columns = True
        (suffix, args), conditions = self._read_args(hold, key_parts, schema['levels'])
        o.key_count += 1
        local = f'_mx_name_{o.key_count}'        # one label at run time, else pandas (a list selects columns)
        label = _cast('object', self.updated_of.get(id(name), name))
        conditions.append(cst.parse_expression(f'{o.module}.is_label(({local} := {_code(label)}))'))
        row = self._next(self._m('row' + suffix, o.ref(hold), *args))
        read = self._m('field_f64', o.ref(hold), row, cst.Name(local))
        return self._lowered(self._native_test(hold, conditions), read, self._fallback(updated, call), 'f64')

    def leave_Call(self, original, updated):
        """``float(...)`` around a float64 read and ``int(...)`` around an int64 read are dropped: the read is
        already that C value, and the builtin would box it into a Python object and back."""
        if (isinstance(updated.func, cst.Name) and updated.func.value in ('isinstance', 'type') and updated.args
                and isinstance(updated.args[0].value, cst.IfExp) and self.native.get(id(updated.args[0].value)) in CTYPE):
            # the type of a read is the numpy scalar's pandas returns, not the C number's
            return updated.with_changes(args=[updated.args[0].with_changes(
                value=self._pandas_value(updated.args[0].value)), *updated.args[1:]])
        inner = self.o.facts.int_argument(original)
        if (isinstance(updated.func, cst.Name) and len(updated.args) == 1 and not updated.args[0].keyword
                and not updated.args[0].star):
            kind = self.native.get(id(updated.args[0].value))
            if (kind == 'f64' and updated.func.value == 'float') or (kind == 'i64' and inner is not None):
                return updated.args[0].value.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        return updated
