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
MAX_PARTS = 3
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

    def __init__(self, resolver, cls_info, cells, providers, policy):
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

    # ------------------------------------------------------------------ driver
    def lower(self, node):
        params = node.params.params
        if not params or params[0].name.value != 'self' or not getattr(self.resolver, 'frame_schemas', None):
            return node
        self.facts = self._facts(node)
        from .helpers import module_name
        self.module = module_name(PROVIDER)
        self.per_item = self.cls_info.name in self.cls_info.visitor.itemspace_classes
        self.holds = {}         # (receiver code, label, kind) -> _Hold
        self.mutated = self._mutated(node)
        self.local_count = 0
        rewriter = _Rewrite(self)
        body = []
        changed = False
        for statement in node.body.body:
            before = len(self.holds)
            rewriter.new_holds = []
            new = statement.visit(rewriter)
            if rewriter.new_holds:
                body.extend(self._hold_statements(hold) for hold in rewriter.new_holds)
            changed = changed or new is not statement or len(self.holds) != before
            body.append(new)
        if not changed:
            return node
        body = self._drop_unused_locals(body, rewriter.replaced_names)
        self.providers.add(PROVIDER)
        return node.with_changes(body=node.body.with_changes(body=self._flatten(body)))

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
        label = repr(hold.label) if hold.label is not _NO_LABEL else 'None'
        receiver = _code(hold.receiver)
        if self.per_item:
            return [cst.parse_statement(
                f'if self.{hold.slot} == 0:\n'
                f'    self.{hold.table} = {m}.hold({receiver}, {label}, {hold.kind}, '
                f'_mx_cy.address(self.{hold.slot}))\n')]
        return [cst.parse_statement(f'{hold.slot}: _mx_cy.Py_ssize_t = 0'),
                cst.parse_statement(f'{hold.table}: object = {m}.hold({receiver}, {label}, {hold.kind}, '
                                    f'_mx_cy.address({hold.slot}))')]

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
            self.new = found
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
        self.native = {}        # id(lowered read) -> column kind, for float()/int() around it

    def visit_FunctionDef(self, node):
        return False

    def visit_Lambda(self, node):
        return False

    def visit_ClassDef(self, node):
        return False

    def _take(self, receiver, label, kind):
        before = len(self.o.holds)
        hold = self.o.hold(receiver, label, kind)
        if len(self.o.holds) != before:
            self.new_holds.append(hold)
        return hold

    def _record(self, node, form, receiver, why=None):
        entry = {'expression': _code(node)[:160], 'receiver': _code(receiver), 'form': form}
        if why:
            entry['reason'] = why
            self.o.declined.append(entry)
        else:
            self.o.lowered.append(entry)

    def _key(self, key, schema):
        """Key parts for a complete key on ``schema``, or (None, why)."""
        levels = schema['levels']
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
        return [self.o.key_part(part, kind)[0] for part, kind in zip(parts, levels)], None

    def leave_Subscript(self, original, updated):
        o = self.o
        value = original.value
        parts = _index_parts(original)
        if parts is None:
            return updated
        # R.loc[key, "c"] / R.at[key, "c"] on a frame; R.loc[k] / R.at[k] / R[k] on a Series
        if isinstance(value, cst.Attribute) and value.attr.value in ('loc', 'at'):
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
        suffix = '' if len(key_parts) == 1 else f'_{len(key_parts)}'
        read = cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name(READ[kind] + suffix)),
                        args=[cst.Arg(o.ref(hold)), cst.Arg(o.slot(hold)), *[cst.Arg(p) for p in key_parts]])
        # the pandas read converted to the column's C type, so both branches are native (no boxing per read); the
        # value is the same (float64 -> double, int64 -> long long), the boundary types of the probe
        fallback = _cast('_mx_cy.double' if kind == 'f64' else '_mx_cy.longlong', self._fallback(updated, call))
        self._record(original, 'L1', call)
        lowered = cst.IfExp(test=cst.Comparison(o.slot(hold), [cst.ComparisonTarget(
                                cst.GreaterThan(), cst.Integer('0'))]),
                            body=read, orelse=fallback,
                            lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        self.native[id(lowered)] = kind
        return lowered

    def leave_Call(self, original, updated):
        """``float(...)`` around a float64 read and ``int(...)`` around an int64 read are dropped: the read is
        already that C value, and the builtin would box it into a Python object and back."""
        if (isinstance(updated.func, cst.Name) and len(updated.args) == 1 and not updated.args[0].keyword
                and not updated.args[0].star):
            kind = self.native.get(id(updated.args[0].value))
            if (kind, updated.func.value) in (('f64', 'float'), ('i64', 'int')):
                return updated.args[0].value.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        return updated

    def _fallback(self, updated, call):
        """The original pandas read on the frame itself (a frame local replaced by its Cell call)."""
        class Swap(cst.CSTTransformer):
            def __init__(self, names):
                super().__init__()
                self.names = names

            def leave_Name(self, original, new):
                return call if original.value in self.names else new
        return updated.visit(Swap(self.replaced_names)) if self.replaced_names else updated

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
        suffix = '' if len(key_parts) == 1 else f'_{len(key_parts)}'
        test = cst.Call(func=cst.Attribute(cst.Name(o.module), cst.Name('contains' + suffix)),
                        args=[cst.Arg(o.ref(hold)), *[cst.Arg(p) for p in key_parts]])
        if isinstance(target.operator, cst.NotIn):
            test = cst.UnaryOperation(cst.Not(), test)
        self._record(original, 'L7', call)
        return cst.IfExp(test=cst.Comparison(o.slot(hold), [cst.ComparisonTarget(
                             cst.GreaterThan(), cst.Integer('0'))]),
                         body=test, orelse=self._fallback(updated, call),
                         lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
