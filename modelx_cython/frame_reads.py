"""Reads of a traced pandas DataFrame or Series lowered onto a PreparedTable (T-3; toolkit p2/tables/DESIGN.md).

The formula keeps the user's frame: the sample runs on it (no substitution) and the compiled model reads the same
object. A read whose form the compiler recognises is answered from a PreparedTable bound to that frame object
(runtime ``_tables``); every other operation runs on the frame itself.

Recognised here (the receiver ``R`` a self-rooted zero-argument call of a cached Cell whose traced value is a
pandas DataFrame or Series with a unique index, or a local assigned once from such a call):

- L1 ``R.loc[key, "c"]`` / ``R.at[key, "c"]`` of a float64 or int64 column, the key complete (one part per
  level); a Series ``R.loc[k]``, ``R.at[k]``, ``R[k]``;
- L7 ``key in R.index`` / ``not in`` with a complete key.

Binding, by the kind of the reading class:

- in a class whose instances are items (an ItemSpace), the receiver is evaluated, and its PreparedTable found by
  identity of the frame object, once per item: when the hold is first needed. The item's fields hold the table
  (``_mx_tp_<n>``) and the column slot (``_mx_tq_<n>``; 0 until resolved, -1 when the read stays pandas, else one
  plus the slot) and are
  reset with its per-policy caches. A read then uses the held table borrowed: no reference count of the frame,
  of the Data space's cached value or of the table is touched per read;
- in any other class, the receiver is evaluated and its PreparedTable looked up by identity once per call of the
  formula (the lookup takes references; the reads themselves do not).

Either way an assignment of a new frame is a new object: its PreparedTable is prepared afresh and no prepared copy
of an older frame is used for it. A read the table cannot answer natively (the frame not prepared: a non-unique
index, NaN labels; the column absent or of another kind) runs the original pandas expression.
"""

import libcst as cst

from .holds import hold_table

PROVIDER = 'modelx_cython.runtime._tables'
COLUMN_KIND = {'f64': 0, 'i64': 1}
READ = {'f64': 'read_f64', 'i64': 'read_i64'}
MAX_PARTS = 6
FUSED_PARTS = 3
MUTATORS = frozenset(('__setitem__', '__delitem__', 'update', 'insert', 'pop'))


def _code(node):
    return cst.Module(body=()).code_for_node(node)


def _literal_str(node):
    if isinstance(node, cst.SimpleString):
        value = node.evaluated_value
        return value if type(value) is str else None
    return None


def _cast(type_expr, value):
    return cst.Call(func=cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                    args=[cst.Arg(cst.parse_expression(type_expr)), cst.Arg(value)])


def _index_parts(subscript):
    parts = []
    for element in subscript.slice:
        if not isinstance(element.slice, cst.Index) or element.slice.star is not None:
            return None
        parts.append(element.slice.value)
    return parts


class _Hold:
    """One (receiver, column, kind) a formula reads natively: its fields (item) or locals (other classes)."""

    def __init__(self, table, slot, receiver, label, kind):
        self.table, self.slot, self.receiver, self.label, self.kind = table, slot, receiver, label, kind


class FrameReads:
    """``lower(node)`` returns the formula with recognised reads lowered; ``lowered`` lists what was lowered
    (one dict per read) and ``declined`` why a read of a frame receiver was not."""

    def __init__(self, resolver, cls_info, cells, providers, policy, row_names=None, prefix_cells=None):
        self.row_names = row_names or {}       # code of an inlined row expression -> its row cell (assignments)
        self.prefix_cells = prefix_cells or {}  # zero-argument cell of the class -> its `T.loc[key]` expression
        self.resolver, self.cls_info, self.cells = resolver, cls_info, cells
        self.providers, self.policy = providers, policy
        self.lowered, self.declined = [], []

    # ------------------------------------------------------------------ facts
    def _facts(self, node):
        """The key-typing facts of table_reads (is_int, exact_long_long, enum constants) for this formula."""
        from .table_reads import TableReads, _Formula
        facts = TableReads(self.resolver, self.cls_info, self.cells, set(), self.policy, ())
        facts.formula = _Formula(node)
        node.visit(facts.formula)
        facts._params = set(getattr(self.cells, 'params', ()) or ())
        return facts

    def _schema_of_call(self, expr):
        if not (isinstance(expr, cst.Call) and not expr.args):
            return None
        from .usage import RESOLVED, _extract_self_chain
        chain = _extract_self_chain(expr.func)
        if not isinstance(chain, list) or not chain:
            return None
        status, name = self.resolver.resolve_chain(self.cls_info.fqname, chain)
        if status != RESOLVED or name not in getattr(self.resolver, 'cached_cells', ()):
            return None
        return getattr(self.resolver, 'frame_schemas', {}).get(name)

    def _receiver(self, expr):
        """(cell call, schema) when ``expr`` is a frame receiver, else None."""
        schema = self._schema_of_call(expr)
        if schema is not None:
            return expr, schema
        if isinstance(expr, cst.Name):
            f = self.facts.formula
            values = f.values.get(expr.value) or ()
            if (len(values) == 1 and expr.value not in f.other and expr.value not in self.facts._params
                    and f.binds.get(expr.value, 0) == 1):
                schema = self._schema_of_call(values[0])
                if schema is not None:
                    return values[0], schema
        return None

    def prefix_receiver(self, expr):
        """T-4 (L3 through a cell): ``self.P()`` (or a local assigned once from it) where the formula of P is
        ``return T.loc[prefix]``, T a frame receiver and every prefix part fixed per item: (P call, T call, T's
        schema, prefix parts, P). The keyed reads of P's value read T with the prefix held as codes."""
        call = expr
        if isinstance(expr, cst.Name):
            f = self.facts.formula
            values = f.values.get(expr.value) or ()
            if not (len(values) == 1 and expr.value not in f.other and expr.value not in self.facts._params
                    and f.binds.get(expr.value, 0) == 1):
                return None
            call = values[0]
        if not (isinstance(call, cst.Call) and not call.args and isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name) and call.func.value.value == 'self'):
            return None
        name = call.func.attr.value
        row = self.prefix_cells.get(name)
        if row is None or not self.cached_call(call):
            return None
        found = self._receiver(row.value.value)
        if found is None or found[1]['kind'] != 'frame':
            return None
        key = row.slice[0].slice.value
        parts = [e.value for e in key.elements] if isinstance(key, cst.Tuple) else [key]
        from .row_cells import item_constant
        if len(parts) >= len(found[1]['levels']) or not all(item_constant(p, self.cached_call) for p in parts):
            return None
        # by traced type: every traced value of P is a frame of the parent's remaining levels and of its columns
        mine = self._schema_of_call(call)
        parent = found[1]
        if not (mine and mine['kind'] == 'frame' and mine['levels'] == parent['levels'][len(parts):]
                and dict(mine['columns']).items() <= dict(parent['columns']).items()):
            self.declined.append({'expression': _code(call), 'receiver': _code(call), 'form': 'pandas',
                                  'reason': 'a prefix cell whose traced value is not the sub-frame of its table'})
            return None
        return call, found[0], found[1], parts, name

    # ------------------------------------------------------------------ driver
    def lower(self, node):
        params = node.params.params
        if not params or params[0].name.value != 'self' or not getattr(self.resolver, 'frame_schemas', None):
            return node
        self.facts = self._facts(node)
        plain = self.facts._int_index_read

        def int_index_read(expr):       # T-4 (L5): extremes of a one-level int frame index are C integers
            return plain(expr) or self.extreme(expr) is not None
        self.facts._int_index_read = int_index_read
        from .helpers import module_name
        self.module = module_name(PROVIDER)
        self.per_item = self.cls_info.name in self.cls_info.visitor.itemspace_classes
        self.holds = {}         # (receiver code, label, kind) -> _Hold
        self.code_fields, self.row_fields = {}, {}
        self.mutated = self._mutated(node)
        self.asof_locals = self._asof_locals(node)
        self.local_count = 0
        rewriter = _Rewrite(self)
        body = []
        changed = False
        for statement in node.body.body:
            before = len(self.holds)
            rewriter.new_holds = self.new_items = []
            asof = self._asof_assignment(statement)
            new = rewriter.asof_statement(statement, asof) if asof is not None else statement.visit(rewriter)
            if self.new_items:
                body.extend(self._hold_statements(item) for item in self.new_items)
            changed = changed or new is not statement or len(self.holds) != before
            body.append(new)
        if not changed:
            return node
        body = self._drop_unused_locals(body, rewriter.replaced_names | set(rewriter.key_locals) | set(rewriter.swaps))
        self.providers.add(PROVIDER)
        return node.with_changes(body=node.body.with_changes(body=self._flatten(body)))

    def _asof_assignment(self, statement):
        if (isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1
                and isinstance(statement.body[0], cst.Assign) and len(statement.body[0].targets) == 1
                and isinstance(statement.body[0].targets[0].target, cst.Name)):
            return self.asof_locals.get(statement.body[0].targets[0].target.value)
        return None

    def _asof_locals(self, node):
        """T-4: ``lo = P.index.asof(x)`` at the top level of an item formula, P the value of a prefix cell with one
        remaining int level, and every load of ``lo`` the key of a keyed read ``P.loc[lo, "c"]`` of that value:
        {name: (P name, x)}. ``lo`` is then the row of the label in the parent table, never a Python value."""
        if not self.per_item:
            return {}
        f, found = self.facts.formula, {}
        for statement in node.body.body:
            if not (isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1
                    and isinstance(statement.body[0], cst.Assign) and len(statement.body[0].targets) == 1
                    and isinstance(statement.body[0].targets[0].target, cst.Name)):
                continue
            name, value = statement.body[0].targets[0].target.value, statement.body[0].value
            if not (isinstance(value, cst.Call) and len(value.args) == 1 and not value.args[0].keyword
                    and isinstance(value.func, cst.Attribute) and value.func.attr.value == 'asof'
                    and isinstance(value.func.value, cst.Attribute) and value.func.value.attr.value == 'index'):
                continue
            if f.binds.get(name, 0) != 1 or name in f.other or name in self.facts._params:
                continue
            prefix = self.prefix_receiver(value.func.value.value)
            if prefix is None or len(prefix[2]['levels']) - len(prefix[3]) != 1 or prefix[2]['levels'][-1] != 'int':
                continue
            uses = [use for use, nested in f.loads.get(name, ()) if not nested]
            if len(uses) != len(f.loads.get(name, ())) or not uses or not all(
                    self._keyed_use(use, prefix[4]) for use in uses):
                continue
            found[name] = (prefix, value.args[0].value, value)
        return found

    def _keyed_use(self, use, cell):
        """``use`` (a Name load) is the first key part of ``X.loc[use, "c"]`` with X the value of prefix cell ``cell``."""
        parents = self.facts.formula.parents
        index = parents.get(id(use))
        element = parents.get(id(index)) if isinstance(index, cst.Index) else None
        subscript = parents.get(id(element)) if isinstance(element, cst.SubscriptElement) else None
        if not (isinstance(subscript, cst.Subscript) and len(subscript.slice) == 2
                and subscript.slice[0] is element and _literal_str(subscript.slice[1].slice.value) is not None
                and isinstance(subscript.value, cst.Attribute) and subscript.value.attr.value == 'loc'):
            return False
        prefix = self.prefix_receiver(subscript.value.value)
        return prefix is not None and prefix[4] == cell

    def _root(self, expr):
        while isinstance(expr, (cst.Attribute, cst.Subscript)):
            expr = expr.value
        return expr

    def _mutated(self, node):
        """Codes of the receivers the formula itself edits in place (an assignment or ``del`` through them,
        ``__setitem__``/``__delitem__``/``update``/``insert``/``pop``, or any call with ``inplace``): their
        reads stay pandas, so an edit is read as it is made, as live."""
        owner, found = self, set()

        def note(expr):
            got = owner._receiver(owner._root(expr))
            if got is not None:
                found.add(_code(got[0]))

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
        return found

    @staticmethod
    def _flatten(body):
        out = []
        for item in body:
            out.extend(item if isinstance(item, list) else [item])
        return out

    def _drop_unused_locals(self, body, replaced):
        """A frame local whose every use was lowered is no longer evaluated (no Cell call, no reference)."""
        flat = self._flatten(body)
        module = cst.Module(body=flat)
        loads = {}

        class Loads(cst.CSTVisitor):
            def visit_Name(self, name):
                loads[name.value] = loads.get(name.value, 0) + 1
        module.visit(Loads())
        out = []
        for statement in flat:
            if (isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1
                    and isinstance(statement.body[0], cst.Assign) and len(statement.body[0].targets) == 1
                    and isinstance(statement.body[0].targets[0].target, cst.Name)):
                name = statement.body[0].targets[0].target.value
                if name in replaced and loads.get(name, 0) == 1:     # the target itself only
                    continue
            out.append(statement)
        return out

    def _hold_statements(self, hold):
        """The statements that resolve one hold, placed before the top-level statement of its first read."""
        m = self.module
        if isinstance(hold, tuple):         # T-4: a held code or row, after the hold of its table
            what, field, table, level_or_key, extra = hold
            if what == 'code':
                value = f'{m}.hold_code(self.{table}, {level_or_key}, {_code(extra)})'
            elif what == 'free':
                value = f'{m}.hold_unassigned(self, {level_or_key!r})'
            else:
                value = f'{m}.hold_row(self.{table}, {_code(level_or_key)}, self, {extra!r})'
            return [cst.parse_statement(f'if self.{field} == 0:\n    self.{field} = {value}\n')]
        label = repr(hold.label) if hold.label is not _NO_LABEL else 'None'
        function = 'hold_checked' if getattr(self.policy, 'verify_inputs', False) else 'hold'
        # T-4 item 0: the kinds the read is compiled for; another kind at run time keeps pandas
        schema = self._schema_of_call(hold.receiver) or {}
        levels = repr(''.join('i' if kind == 'int' else 'o' for kind in schema.get('levels', ())))
        kinds = {'f64': 0, 'i64': 1, 'obj': 2}
        columns = (repr(tuple((name, kinds[kind]) for name, kind in schema.get('columns', ())))
                   if getattr(hold, 'row_columns', False) else 'None')
        extra = f', {self.signature(levels, columns)}'
        receiver = _code(hold.receiver)
        if self.per_item:
            return [cst.parse_statement(
                f'if self.{hold.slot} == 0:\n'
                f'    self.{hold.table} = {m}.{function}({receiver}, {label}, {hold.kind}, '
                f'_mx_cy.address(self.{hold.slot}){extra})\n')]
        return [cst.parse_statement(f'{hold.slot}: _mx_cy.Py_ssize_t = 0'),
                cst.parse_statement(f'{hold.table}: object = {m}.{function}({receiver}, {label}, {hold.kind}, '
                                    f'_mx_cy.address({hold.slot}){extra})')]

    def extreme(self, expr):
        """``R.index.max()`` / ``R.index.min()`` on a frame receiver with one unique int level: (R call, 'max'|'min')."""
        if not (isinstance(expr, cst.Call) and not expr.args and isinstance(expr.func, cst.Attribute)
                and expr.func.attr.value in ('max', 'min') and isinstance(expr.func.value, cst.Attribute)
                and expr.func.value.attr.value == 'index'):
            return None
        found = self._receiver(expr.func.value.value)
        if found is None or found[1]['levels'] != ('int',) or not found[1].get('unique') or _code(found[0]) in self.mutated:
            return None
        return found[0], expr.func.attr.value

    def signature(self, levels, columns):
        """The model-wide id of a read signature (written to ``_mx_table_signatures.py`` at the end of the build)."""
        table = self.resolver.__dict__.setdefault('table_signatures', {})
        return table.setdefault((levels, columns), len(table))

    def cached_call(self, call):
        """A self-rooted zero-argument call of a cached Cell (as the transformer's ``_cached_call``)."""
        name = call.func.attr.value if isinstance(call.func, cst.Attribute) else None
        return name is not None and any(str(c).endswith('._f_' + name)
                                        for c in getattr(self.resolver, 'cached_cells', ()))

    def item_fixed(self, part):
        """A key part whose value is fixed per item (held as a code), in a per-step cell of an item class."""
        from .row_cells import item_constant
        if isinstance(part, (cst.Integer, cst.SimpleString)):
            return False            # a literal is passed as it is (a C integer, or an interned str)
        return self.per_item and self.cells.has_args() and item_constant(part, self.cached_call)

    def code_field(self, hold, level, part):
        key = (hold.table, level, _code(part))
        fields = self.code_fields
        if key not in fields:
            fields[key] = hold_table(self.cls_info).field('prepared_code', key)
            self.new_items.append(('code', fields[key], hold.table, level, part))
        return fields[key]

    def assigned_field(self, hold, name):
        """One once the cell ``name`` is known to hold its formula's value for this item, -1 when it holds an
        assigned value (its reads then run on pandas)."""
        key = (hold.table, 'unassigned', name)
        fields = self.code_fields
        if key not in fields:
            fields[key] = hold_table(self.cls_info).field('prepared_code', key)
            self.new_items.append(('free', fields[key], hold.table, name, None))
        return fields[key]

    def row_field(self, hold, key_expr, name):
        key = (hold.table, _code(key_expr))
        fields = self.row_fields
        if key not in fields:
            fields[key] = hold_table(self.cls_info).field('prepared_row', key)
            self.new_items.append(('row', fields[key], hold.table, key_expr, name))
        return fields[key]

    def hold(self, receiver, label, kind):
        key = (_code(receiver), label, kind)
        found = self.holds.get(key)
        if found is None:
            if self.per_item:
                table = hold_table(self.cls_info).field('prepared', key)
                slot = hold_table(self.cls_info).field('prepared_slot', key)
            else:
                n = self.local_count
                self.local_count += 1
                table, slot = f'_mx_pt{n}', f'_mx_ps{n}'
            found = self.holds[key] = _Hold(table, slot, receiver, label, kind)
            self.new_items.append(found)
        return found

    def ref(self, hold):
        where = f'self.{hold.table}' if self.per_item else hold.table
        return _cast(f'{self.module}.PreparedRef', cst.parse_expression(where))

    def slot(self, hold):
        return cst.parse_expression(f'self.{hold.slot}' if self.per_item else hold.slot)

    def key_part(self, part, level_kind):
        """A key part passed to the fused reads: a C long long on an int level when it is an integer, else
        the Python value (the table looks an object key up as pandas does)."""
        if level_kind == 'int' and self.facts.is_int(part):
            constant = self.facts.enum_constant(part)
            if constant is not None:
                part = cst.Name(constant)
            if self.facts.exact_long_long(part):
                return part, 'int'
            return _cast('_mx_cy.longlong', part), 'int'
        return _cast('object', part), 'object'


_NO_LABEL = object()


class _Rewrite(cst.CSTTransformer):
    """Rewrites reads inside one top-level statement; nested functions and lambdas are not entered."""

    def __init__(self, owner):
        super().__init__()
        self.o = owner
        self.new_holds = []
        self.replaced_names = set()
        self.swaps = {}         # frame or prefix local -> the call it stands for (fallbacks, dropped locals)
        self.key_locals = {}    # name of a key local -> its tuple display (the fallback evaluates the display)
        self.asof_rows = {}     # asof local -> (membership hold, prefix cell call, asof expression)
        self.native = {}        # id(lowered read) -> column kind, for float()/int() around it

    def visit_FunctionDef(self, node):
        return False

    def visit_Lambda(self, node):
        return False

    def visit_ClassDef(self, node):
        return False

    def _take(self, receiver, label, kind):
        return self.o.hold(receiver, label, kind)

    def _record(self, node, form, receiver, why=None):
        entry = {'expression': _code(node)[:160], 'receiver': _code(receiver), 'form': form}
        if not why:
            schema = self.o._schema_of_call(receiver) or {}
            entry['pandas_when'] = ('the frame at run time has other index level kinds than '
                                    f"{list(schema.get('levels', ()))}, or the column read is of another kind")
        if why:
            entry['reason'] = why
            self.o.declined.append(entry)
        else:
            self.o.lowered.append(entry)

    def _key(self, key, schema):
        """The parts of a complete key on ``schema``: (parts, None) or (None, why). A key held in a local assigned
        once to a tuple display is that display."""
        levels = schema['levels']
        f = self.o.facts.formula
        if (isinstance(key, cst.Name) and len(levels) > 1 and key.value not in f.other
                and key.value not in self.o.facts._params and f.binds.get(key.value, 0) == 1
                and len(f.values.get(key.value) or ()) == 1 and isinstance(f.values[key.value][0], cst.Tuple)):
            self.key_locals[key.value] = f.values[key.value][0]
            key = f.values[key.value][0]
        if len(levels) == 1:
            if isinstance(key, cst.Tuple):
                return None, 'a tuple key on a single-level index'
            parts = [key]
        else:
            if not isinstance(key, cst.Tuple) or any(isinstance(e, cst.StarredElement) for e in key.elements):
                return None, 'the key is not a tuple display of every level'
            parts = [e.value for e in key.elements]
            if len(parts) != len(levels):
                return None, f'{len(parts)} key parts on an index of {len(levels)} levels (a prefix: pandas)'
        if len(parts) > MAX_PARTS:
            return None, f'{len(parts)} key parts'
        for part in parts:
            if isinstance(part, (cst.List, cst.ListComp, cst.Set, cst.Slice, cst.Tuple, cst.Lambda)):
                return None, 'a key part selects several rows'
        return parts, None

    def _read_args(self, hold, parts, levels):
        """(function suffix, key arguments, extra native conditions) for a read of ``parts`` through ``hold``: the
        fused per-part reads, or (a part fixed per item, or more than FUSED_PARTS parts) one level code per part."""
        o = self.o
        held = [o.item_fixed(part) for part in parts]
        if not any(held) and len(parts) <= FUSED_PARTS:
            return ('' if len(parts) == 1 else f'_{len(parts)}',
                    [o.key_part(part, kind)[0] for part, kind in zip(parts, levels)], [])
        codes, conditions = [], []
        for level, (part, kind, fixed) in enumerate(zip(parts, levels, held)):
            if fixed:
                field = o.code_field(hold, level, part)
                codes.append(cst.parse_expression(f'self.{field} - 1'))
                conditions.append(cst.parse_expression(f'self.{field} > 0'))
            else:
                arg, how = o.key_part(part, kind)
                fn = 'lcode' if how == 'int' else 'ocode'
                codes.append(cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name(fn)),
                                      args=[cst.Arg(o.ref(hold)), cst.Arg(cst.Integer(str(level))), cst.Arg(arg)]))
        codes += [cst.Integer('0')] * (MAX_PARTS - len(codes))
        return '_c', [cst.Integer(str(len(parts))), *codes], conditions

    def _native_test(self, hold, conditions):
        test = cst.Comparison(self.o.slot(hold), [cst.ComparisonTarget(cst.GreaterThan(), cst.Integer('0'))])
        for condition in conditions:
            test = cst.BooleanOperation(left=test, operator=cst.And(), right=condition)
        return test

    def leave_Subscript(self, original, updated):
        o = self.o
        value = original.value
        parts = _index_parts(original)
        if parts is None:
            return updated
        row = self._row_field(original, updated, parts)
        if row is not None:
            return row
        # R.loc[key, "c"] / R.at[key, "c"] on a frame; R.loc[k] / R.at[k] / R[k] on a Series
        if isinstance(value, cst.Attribute) and value.attr.value in ('loc', 'at'):
            prefix = o.prefix_receiver(value.value) if len(parts) == 2 else None
            if prefix is not None:      # before the frame itself: a prefix cell's value is a new frame per item
                return self._prefix_read(original, updated, value, parts, prefix)
            found = o._receiver(value.value)
            form = value.attr.value
        else:
            found = o._receiver(value)
            form = 'item'
        if found is None:
            return updated
        call, schema = found
        if _code(call) in o.mutated:
            self._record(original, 'pandas', call, 'the formula edits this frame in place')
            return updated
        if not schema.get('unique'):
            self._record(original, 'pandas', call, 'the index is not unique (pandas)')
            return updated
        if schema['kind'] == 'series':
            if len(parts) != 1:
                return updated
            key, label = parts[0], None
            if form == 'item' and schema['levels'] != ('int',) and o.facts.is_int(key):
                self._record(original, 'pandas', call, 'an integer key on a Series without an integer index')
                return updated
        else:
            if form == 'item' or len(parts) != 2:
                return updated      # a column, a row, or another selection: pandas (T-4/T-5 forms)
            key, label = parts
            label = _literal_str(label)
            if label is None:
                self._record(original, 'pandas', call, 'the column is not a literal name')
                return updated
        kind = dict(schema['columns']).get(label)
        if kind not in READ:
            self._record(original, 'pandas', call, (
                'the Series values are not float64 or int64 in every trace (a row read by field: form L2)'
                if schema['kind'] == 'series' else f'column {label!r} is not float64 or int64 in every trace'))
            return updated
        key_parts, why = self._key(key, schema)
        if why:
            self._record(original, 'pandas', call, why)
            return updated
        hold = self._take(call, label, COLUMN_KIND[kind])
        if isinstance(value.value if form != 'item' else value, cst.Name):
            self.replaced_names.add((value.value if form != 'item' else value).value)
        suffix, args, conditions = self._read_args(hold, key_parts, schema['levels'])
        name = (READ[kind] if suffix != '_c' else {'f64': 'f64', 'i64': 'i64'}[kind]) + suffix
        read = cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name(name)),
                        args=[cst.Arg(o.ref(hold)), cst.Arg(o.slot(hold)), *[cst.Arg(p) for p in args]])
        # the pandas read converted to the column's C type, so both branches are native (no boxing per read); the
        # value is the same (float64 -> double, int64 -> long long), the boundary types of the probe
        fallback = _cast('_mx_cy.double' if kind == 'f64' else '_mx_cy.longlong', self._fallback(updated, call))
        self._record(original, 'L1', call)
        lowered = cst.IfExp(test=self._native_test(hold, conditions),
                            body=read, orelse=fallback,
                            lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        self.native[id(lowered)] = kind
        return lowered

    def leave_Call(self, original, updated):
        """``float(...)`` around a float64 read and ``int(...)`` around an int64 read are dropped: the read is
        already that C value, and the builtin would box it into a Python object and back. L5: the extremes of a
        one-level int index are read as C integers."""
        found = self.o.extreme(original)
        if found is not None:
            call, which = found
            o = self.o
            hold = self._take(call, _NO_LABEL, -2)
            if isinstance(original.func.value.value, cst.Name):
                self.replaced_names.add(original.func.value.value.value)
            read = cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name(which + '_ll')), args=[cst.Arg(o.ref(hold))])
            self._record(original, 'L5', call)
            lowered = cst.IfExp(test=self._native_test(hold, []), body=read,
                                orelse=_cast('_mx_cy.longlong', self._fallback(updated, call)),
                                lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
            self.native[id(lowered)] = 'i64'
            return lowered
        if (isinstance(updated.func, cst.Name) and len(updated.args) == 1 and not updated.args[0].keyword
                and not updated.args[0].star):
            kind = self.native.get(id(updated.args[0].value))
            if (kind, updated.func.value) in (('f64', 'float'), ('i64', 'int')):
                return updated.args[0].value.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        return updated

    def _prefix_holds(self, hold, prefix_parts, name):
        o = self.o
        codes, conditions = [], []
        for level, part in enumerate(prefix_parts):
            field = o.code_field(hold, level, part)
            codes.append(cst.parse_expression(f'self.{field} - 1'))
            conditions.append(cst.parse_expression(f'self.{field} > 0'))
        conditions.append(cst.parse_expression(f'self.{o.assigned_field(hold, name)} > 0'))
        return codes, conditions

    def asof_statement(self, statement, asof):
        """``lo = P.index.asof(x)`` -> ``_mx_asof_lo = <row of the label in the parent table, or -1>``, through a
        membership hold of the parent (codes of the prefix held per item)."""
        o = self.o
        (pcall, call, schema, prefix_parts, name), x, original = asof
        target = statement.body[0].targets[0].target.value
        hold = self._take(call, _NO_LABEL, -1)
        codes, conditions = self._prefix_holds(hold, prefix_parts, name)
        codes += [cst.Integer('0')] * (5 - len(codes))
        search = cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name('asof_row')),
                          args=[cst.Arg(o.ref(hold)), cst.Arg(cst.Integer(str(len(prefix_parts)))),
                                *[cst.Arg(c) for c in codes], cst.Arg(_cast('_mx_cy.double', x))])
        value = cst.IfExp(test=self._native_test(hold, conditions), body=search, orelse=cst.parse_expression('-1'))
        self.asof_rows[target] = (hold, pcall, original)
        # the fallback of a read evaluates the pandas asof again, on the cell's value
        receiver = original.func.value.value
        self.swaps.setdefault(target, original.with_changes(func=original.func.with_changes(
            value=original.func.value.with_changes(value=pcall))) if isinstance(receiver, cst.Name) else original)
        if isinstance(receiver, cst.Name):
            self.swaps[receiver.value] = pcall
        self._record(original, 'L6', pcall)
        return cst.parse_statement(f'_mx_asof_{target}: _mx_cy.Py_ssize_t = {_code(value)}')

    def _prefix_read(self, original, updated, value, parts, prefix):
        """``P.loc[k, "c"]`` on the value of a prefix cell P (``T.loc[prefix]``): ``T.loc[prefix + k, "c"]`` with
        the prefix parts held as codes per item, while P holds no assigned value (checked once per item)."""
        o = self.o
        pcall, call, schema, prefix_parts, name = prefix
        label = _literal_str(parts[1])
        kind = dict(schema['columns']).get(label)
        if label is None or kind not in READ or not schema.get('unique') or _code(call) in o.mutated:
            self._record(original, 'pandas', pcall, 'a prefix read of a column that is not float64 or int64, '
                         'on a non-unique index, or of a frame the formula edits')
            return updated
        key = parts[0]
        if isinstance(key, cst.Name) and key.value in self.asof_rows:     # a key found by a lowered asof
            return self._asof_read(original, updated, value, label, kind, call, pcall, key.value)
        own = [e.value for e in key.elements] if isinstance(key, cst.Tuple) else [key]
        levels = schema['levels']
        if len(prefix_parts) + len(own) != len(levels) or not o.per_item:
            self._record(original, 'pandas', pcall, 'the prefix and the key do not make a complete key'
                         if o.per_item else 'a prefix cell outside an item space')
            return updated
        hold = self._take(call, label, COLUMN_KIND[kind])
        codes, conditions = self._prefix_holds(hold, prefix_parts, name)
        for level, part in enumerate(own, start=len(prefix_parts)):
            arg, how = o.key_part(part, levels[level])
            codes.append(cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name('lcode' if how == 'int' else 'ocode')),
                                  args=[cst.Arg(o.ref(hold)), cst.Arg(cst.Integer(str(level))), cst.Arg(arg)]))
        codes += [cst.Integer('0')] * (MAX_PARTS - len(codes))
        read = cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name(kind + '_c')),
                        args=[cst.Arg(o.ref(hold)), cst.Arg(o.slot(hold)), cst.Arg(cst.Integer(str(len(levels)))),
                              *[cst.Arg(c) for c in codes]])
        if isinstance(value.value, cst.Name):
            self.swaps[value.value.value] = pcall
        fallback = _cast('_mx_cy.double' if kind == 'f64' else '_mx_cy.longlong', self._fallback(updated, call))
        self._record(original, 'L3', pcall)
        lowered = cst.IfExp(test=self._native_test(hold, conditions), body=read, orelse=fallback,
                            lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        self.native[id(lowered)] = kind
        return lowered

    def _asof_read(self, original, updated, value, label, kind, call, pcall, name):
        o = self.o
        asof_hold = self.asof_rows[name][0]
        hold = self._take(call, label, COLUMN_KIND[kind])
        row = f'_mx_asof_{name}'
        read = cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name(kind + '_row')),
                        args=[cst.Arg(o.ref(hold)), cst.Arg(o.slot(hold)), cst.Arg(cst.parse_expression(f'{row} + 1'))])
        # the row is a row of the table the asof searched: the same prepared table object
        test = self._conjunction([cst.parse_expression(f'{row} >= 0'), self._native_test(hold, []),
                                  cst.parse_expression(f'self.{hold.table} is self.{asof_hold.table}')])
        if isinstance(value.value, cst.Name):
            self.swaps[value.value.value] = pcall
        fallback = _cast('_mx_cy.double' if kind == 'f64' else '_mx_cy.longlong', self._fallback(updated, call))
        self._record(original, 'L3', pcall)
        lowered = cst.IfExp(test=test, body=read, orelse=fallback, lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        self.native[id(lowered)] = kind
        return lowered

    def _row_field(self, original, updated, parts):
        """L2 (T-4): ``R.loc[key]["f"]``, a field of a row of a frame, the key fixed per item (the model-point row,
        also as inlined from its row cell): the row is held per item (``_mx_tw_``) and the field read from the
        prepared table, of the kind pandas gives it: the column's own on a row with an object column, else the
        row's common kind (an int column of a float row reads as a float). None when not this form."""
        o = self.o
        row = original.value
        if not (o.per_item and len(parts) == 1 and isinstance(row, cst.Subscript)
                and isinstance(row.value, cst.Attribute) and row.value.attr.value == 'loc'):
            return None
        label = _literal_str(parts[0])
        key_parts = _index_parts(row)
        found = o._receiver(row.value.value)
        if label is None or key_parts is None or len(key_parts) != 1 or found is None:
            return None
        call, schema = found
        if schema['kind'] != 'frame':
            return None
        if _code(call) in o.mutated or not schema.get('unique'):
            self._record(original, 'pandas', call, 'the formula edits this frame in place'
                         if _code(call) in o.mutated else 'the index is not unique (pandas)')
            return updated
        key = key_parts[0]
        levels = schema['levels']
        fixed = (key.elements if isinstance(key, cst.Tuple) else [key])
        if (isinstance(key, cst.Tuple) and len(key.elements) != len(levels)) or (
                not isinstance(key, cst.Tuple) and len(levels) != 1):
            self._record(original, 'pandas', call, 'the row key is not complete')
            return updated
        from .row_cells import item_constant
        if not all(item_constant(e.value if isinstance(e, cst.Element) else e, o.cached_call) for e in fixed):
            self._record(original, 'pandas', call, 'the row key is not fixed per item')
            return updated
        kinds = dict(schema['columns'])
        if label not in kinds:
            self._record(original, 'pandas', call, f'column {label!r} is not in every traced frame')
            return updated
        mixed = any(k == 'obj' for k in kinds.values())
        kind = kinds[label] if mixed else ('f64' if any(k == 'f64' for k in kinds.values()) else 'i64')
        stored = kinds[label]           # the block the column is stored in
        hold = self._take(call, label, {'f64': 0, 'i64': 1, 'obj': 2}[stored])
        hold.row_columns = True         # the row's dtype depends on every column: all of them are checked
        if isinstance(row.value.value, cst.Name):
            self.replaced_names.add(row.value.value.value)
        name = o.row_names.get(_code(row)) if o.row_names else None
        field = o.row_field(hold, key, name)
        m = o.module
        args = [cst.Arg(o.ref(hold)), cst.Arg(o.slot(hold)), cst.Arg(cst.parse_expression(f'self.{field}'))]
        fn = {'f64': 'f64_row', 'i64': 'i64_row', 'obj': 'object_field'}[stored]
        read = cst.Call(func=cst.Attribute(cst.Name(m), cst.Name(fn)), args=args)
        fallback = self._fallback(updated, call)
        if kind == 'f64':
            if stored == 'i64':
                read = _cast('_mx_cy.double', read)
            fallback = _cast('_mx_cy.double', fallback)
        elif kind == 'i64':
            fallback = _cast('_mx_cy.longlong', fallback)
        self._record(original, 'L2', call)
        lowered = cst.IfExp(test=self._native_test(hold, [cst.parse_expression(f'self.{field} > 0')]),
                            body=read, orelse=fallback, lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        if kind != 'obj':
            self.native[id(lowered)] = kind
        return lowered

    @staticmethod
    def _conjunction(conditions):
        test = conditions[0]
        for condition in conditions[1:]:
            test = cst.BooleanOperation(left=test, operator=cst.And(), right=condition)
        return test

    def _fallback(self, updated, call):
        """The original pandas read on the frame itself: a frame local replaced by its Cell call, a key local by
        its tuple display (so neither local is evaluated on the native path)."""
        frames, keys, swaps = self.replaced_names, self.key_locals, self.swaps

        class Swap(cst.CSTTransformer):
            def leave_Name(self, original, new):
                if original.value in swaps:
                    return swaps[original.value]
                if original.value in frames:
                    return call
                if original.value in keys:
                    return keys[original.value]
                return new
        return updated.visit(Swap()) if (frames or keys or swaps) else updated

    def leave_Comparison(self, original, updated):
        o = self.o
        if len(original.comparisons) != 1:
            return updated
        target = original.comparisons[0]
        if not isinstance(target.operator, (cst.In, cst.NotIn)):
            return updated
        index = target.comparator
        if not (isinstance(index, cst.Attribute) and index.attr.value == 'index'):
            return updated
        found = o._receiver(index.value)
        if found is None:
            return updated
        call, schema = found
        if _code(call) in o.mutated:
            self._record(original, 'pandas', call, 'the formula edits this frame in place')
            return updated
        if not schema.get('unique'):
            self._record(original, 'pandas', call, 'the index is not unique (pandas)')
            return updated
        key_parts, why = self._key(original.left, schema)
        if why:
            self._record(original, 'pandas', call, why)
            return updated
        hold = self._take(call, _NO_LABEL, -1)
        if isinstance(index.value, cst.Name):
            self.replaced_names.add(index.value.value)
        suffix, args, conditions = self._read_args(hold, key_parts, schema['levels'])
        test = cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name('contains' + suffix)),
                        args=[cst.Arg(o.ref(hold)), *[cst.Arg(p) for p in args]])
        if isinstance(target.operator, cst.NotIn):
            test = cst.UnaryOperation(cst.Not(), test)
        self._record(original, 'L7', call)
        # a part fixed per item that is absent from its level: the membership is False without a lookup
        native = self._native_test(hold, [])
        if conditions:
            missing = cst.Name('True' if isinstance(target.operator, cst.NotIn) else 'False')
            test = cst.IfExp(test=self._conjunction(conditions), body=test, orelse=missing,
                             lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        return cst.IfExp(test=native,
                         body=test, orelse=self._fallback(updated, call),
                         lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
