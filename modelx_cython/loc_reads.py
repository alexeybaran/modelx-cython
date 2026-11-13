"""Type-directed lowering of pandas-spelled table reads.

``R.loc[key, "column"]`` becomes ``R.lookup_float(key, "column")`` when ``R`` is a table
whose traced schema gives ``column`` a float kind. ``R.index.max()`` and
``R.index.min()`` (also inside ``int()``) become the provider's C integer reads of the table's
last and first key when the keys are integers. The rewritten calls are the
declared table operations that later passes already lower; nothing else changes.

``R`` is recognised as a table when it is, after following at most a few steps:

- a call to a Cell whose traced result is a table, or a grouped table (Records of tables);
- ``.slice(...)`` applied to such a value;
- a local name with exactly one assignment in the formula, from such a value (repeating the
  identical assignment counts as one). A local bound to a zero-argument cached Cell call is
  replaced by the call in the rewritten read: the call returns the same cached object.

``K in R.index`` becomes the provider's ``native_contains(R, K)`` (the same membership test).

``float(R.loc[key, "col"])`` loses the ``float()``: the typed read already returns a float.
For the same reason ``float()`` of an integer column read becomes the float read.

``R.loc[key]["col"]`` and ``row["col"]`` with ``row = R.loc[key]`` assigned once (key a local
or a cached Cell call) are read as ``R.loc[key, "col"]``: the same value for a unique index.

``G.loc[(g1, ..., gn, key), "col"]`` on a grouped table (``groups=``) stays an object read; the
same table declared flat, with a multi-part key, is read through the typed keyed path.

The decision uses the receiver's traced type, the traced column kind and a literal column
name. Every other use of ``.loc`` is left as written and runs as an ordinary call on the
table object, with the same meaning as in live modelx and exported Python. The count of
typed and untyped reads, with the reason, goes to the build's admission report.
"""

import os
import libcst as cst

from .scopes import FormulaTransformer
from .usage import RESOLVED, _extract_self_chain


class _SingleAssignments(cst.CSTVisitor):
    def __init__(self):
        self.values, self.counts = {}, {}

    def _target(self, target, value):
        if isinstance(target, cst.Name):
            previous = self.values.get(target.value)
            # A repeated assignment of the identical expression (a Cell call, cached) is one value.
            if previous is not None and value is not None and previous.deep_equals(value):
                return
            self.counts[target.value] = self.counts.get(target.value, 0) + 1
            self.values[target.value] = value
        else:   # tuples, attributes, subscripts: the names involved are not single assignments
            for node in _names(target):
                self.counts[node] = self.counts.get(node, 0) + 2

    def visit_Assign(self, node):
        for target in node.targets:
            self._target(target.target, node.value)

    def visit_AnnAssign(self, node):
        self._target(node.target, node.value)

    def visit_AugAssign(self, node):
        self._target(node.target, None)
        self._target(node.target, None)

    def visit_For(self, node):
        for name in _names(node.target):
            self.counts[name] = self.counts.get(name, 0) + 2

    def visit_NamedExpr(self, node):
        self._target(node.target, None)
        self._target(node.target, None)


def _names(node):
    found = []

    class _Collect(cst.CSTVisitor):
        def visit_Name(self, name):
            found.append(name.value)
    node.visit(_Collect())
    return found


class LocReads(FormulaTransformer):
    def __init__(self, resolver, space, providers=None):
        super().__init__()
        self.providers = providers if providers is not None else set()
        self.resolver = resolver
        self.space = space
        self.typed = 0
        self.untyped = []
        self._aliases = {}
        self._row_locals = set()

    def visit_FunctionDef(self, node):
        if not self._aliases:
            scan = _SingleAssignments()
            node.body.visit(scan)
            self._aliases = {name: value for name, value in scan.values.items()
                             if scan.counts.get(name) == 1 and value is not None}
            self._assign_counts = dict(scan.counts)
            self._row_locals = self._replaceable_rows(node.body)
        return True

    def _replaceable_rows(self, body):
        """Locals ``row = R.loc[key]`` whose every use is ``row["literal"]`` on a traced table.

        Their reads are rewritten to ``R.loc[key, "col"]``, so the row itself is never needed.
        """
        candidates = {}
        for name, value in self._aliases.items():
            if (isinstance(value, cst.Subscript) and isinstance(value.value, cst.Attribute)
                    and value.value.attr.value == 'loc' and len(value.slice) == 1
                    and isinstance(value.slice[0].slice, cst.Index)
                    and self._pure_key(value.slice[0].slice.value)
                    and self._schema(value.value.value) is not None):
                candidates[name] = 0
        if not candidates:
            return set()
        uses, column_reads = dict.fromkeys(candidates, 0), dict.fromkeys(candidates, 0)

        class Uses(cst.CSTVisitor):
            def visit_Name(self, node):
                if node.value in uses:
                    uses[node.value] += 1

            def visit_Subscript(self, node):
                if (isinstance(node.value, cst.Name) and node.value.value in column_reads
                        and len(node.slice) == 1 and isinstance(node.slice[0].slice, cst.Index)
                        and isinstance(node.slice[0].slice.value, cst.SimpleString)):
                    column_reads[node.value.value] += 1
        body.visit(Uses())
        # every Name occurrence except the single assignment target is a literal column read
        return {name for name in candidates if uses[name] == column_reads[name] + 1}

    def leave_SimpleStatementLine(self, original, updated):
        body = [statement for statement in updated.body
                if not (isinstance(statement, cst.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0].target, cst.Name)
                        and statement.targets[0].target.value in self._row_locals)]
        if len(body) == len(updated.body):
            return updated
        self.typed += 0
        return updated.with_changes(body=body) if body else cst.RemovalSentinel.REMOVE

    def _schema(self, receiver, depth=0, sliced=0):
        """The traced schema of the table this expression evaluates to, if any.

        A grouped table (Records of tables) is a table only after ``.slice(...)`` with one
        key per group level; ``.slice(key)`` on a plain table selects a row and is not a table.
        """
        if depth > 4:
            return None
        if isinstance(receiver, cst.Name) and receiver.value in self._aliases:
            return self._schema(self._aliases[receiver.value], depth + 1, sliced)
        if not isinstance(receiver, cst.Call):
            return None
        func = receiver.func
        if isinstance(func, cst.Attribute) and func.attr.value == 'slice':
            if sliced or any(arg.keyword or arg.star for arg in receiver.args):  # also rejects sliced == -1
                return None
            return self._schema(func.value, depth + 1, len(receiver.args))
        chain = _extract_self_chain(func)
        if not isinstance(chain, list) or not chain:     # None, or _Unresolved for a non-self root
            return None
        status, name = self.resolver.resolve_chain(self.space, chain)
        if status != RESOLVED:
            return None
        schema = self.resolver.table_schemas.get(name)
        if schema is None:
            return None
        if sliced == -1:
            return schema if schema.get('__grouped__', 0) else None
        if schema.get('__grouped__', 0) != sliced:
            return None     # a grouped container is a table only when every group level is sliced
        return schema

    def _float_read(self, node, kinds):
        """The typed read for ``R.loc[key, "col"]`` if R is a table and col has one of kinds.

        Returns (call, None) or (None, reason).
        """
        target = node.value
        if not (isinstance(target, cst.Attribute) and target.attr.value == 'loc'):
            return None, None
        elements = node.slice
        if len(elements) != 2 or not all(isinstance(e.slice, cst.Index) for e in elements):
            return None, 'not loc[key, column]'
        key, column = elements[0].slice.value, elements[1].slice.value
        names = self._literal_values(column)
        if names is None:
            return None, 'column is not a string literal'
        receiver = target.value
        schema = self._schema(receiver)
        if schema is None:
            if self._schema(receiver, sliced=-1) is not None:
                # Grouped Records are a layout option; a full-key read is native on a flat table.
                return None, 'full key into a grouped table: declare the table without groups= to read it by full key'
            return None, 'receiver is not a traced table'
        found = {schema.get(name) for name in names}
        if not found <= set(kinds):
            name = sorted(n for n in names if schema.get(n) not in kinds)[0]
            return None, f'column {name!r} has traced kind {schema.get(name)}'
        return cst.Call(func=cst.Attribute(receiver, cst.Name('lookup_float')),
                        args=(cst.Arg(key), cst.Arg(column))), None

    def _literal_values(self, node, depth=0):
        """The possible values of a column expression built only from string literals.

        A literal, a conditional expression between such expressions, or a local assigned
        once from one. None for anything else.
        """
        if depth > 4:
            return None
        if isinstance(node, cst.SimpleString):
            return {node.evaluated_value}
        if isinstance(node, cst.IfExp):
            a, b = self._literal_values(node.body, depth + 1), self._literal_values(node.orelse, depth + 1)
            return a | b if a is not None and b is not None else None
        if isinstance(node, cst.Name) and node.value in self._aliases:
            return self._literal_values(self._aliases[node.value], depth + 1)
        return None

    def _pure_key(self, node):
        """A key expression that can be evaluated again with the same result."""
        if isinstance(node, cst.Name):
            # a parameter or a local assigned at most once keeps its value for the whole formula
            return getattr(self, '_assign_counts', {}).get(node.value, 0) <= 1
        if isinstance(node, (cst.Integer, cst.SimpleString)):
            return True
        if isinstance(node, cst.Tuple):
            return all(self._pure_key(e.value) for e in node.elements)
        if isinstance(node, cst.Call) and not any(a.keyword or a.star for a in node.args):
            chain = _extract_self_chain(node.func)
            if isinstance(chain, list) and chain:
                status, name = self.resolver.resolve_chain(self.space, chain)
                return (status == RESOLVED and name in getattr(self.resolver, 'cached_cells', ())
                        and all(self._pure_key(a.value) for a in node.args))
        return False

    def _row_read(self, node):
        """``R.loc[key]["col"]``, or ``row["col"]`` with ``row = R.loc[key]`` assigned once,
        as the equivalent ``R.loc[key, "col"]``; None for anything else."""
        if not (len(node.slice) == 1 and isinstance(node.slice[0].slice, cst.Index)
                and isinstance(node.slice[0].slice.value, cst.SimpleString)):
            return None
        row = node.value
        if isinstance(row, cst.Name) and row.value in self._aliases:
            row = self._aliases[row.value]
        if not (isinstance(row, cst.Subscript) and isinstance(row.value, cst.Attribute)
                and row.value.attr.value == 'loc' and len(row.slice) == 1
                and isinstance(row.slice[0].slice, cst.Index) and self._pure_key(row.slice[0].slice.value)):
            return None
        return row.with_changes(slice=[cst.SubscriptElement(cst.Index(row.slice[0].slice.value), comma=cst.Comma()),
                                       cst.SubscriptElement(node.slice[0].slice)])

    def leave_Subscript(self, original, updated):
        whole = self._row_read(updated)
        if whole is not None:
            call, reason = self._float_read(whole, ('float',))
            if call is not None:
                self.typed += 1
                return call
            if reason is not None and 'traced kind int' in reason:
                self.untyped.append(reason)
                return whole        # float() around it may still type it as a float read
            if isinstance(updated.value, cst.Name) and updated.value.value in self._row_locals:
                if reason is not None:
                    self.untyped.append(reason)
                return whole        # the row local is removed, so read the cell directly
            return updated
        call, reason = self._float_read(updated, ('float',))
        if call is not None:
            self.typed += 1
            return call
        if reason is not None:
            self.untyped.append(reason)
        return updated

    def _bound(self, node, name):
        """R.index.max() / R.index.min() on an integer-keyed table: the provider's C read."""
        if not (isinstance(node, cst.Call) and not node.args and isinstance(node.func, cst.Attribute)
                and node.func.attr.value == name and isinstance(node.func.value, cst.Attribute)
                and node.func.value.attr.value == 'index'):
            return None
        schema = self._schema(node.func.value.value)
        if schema is None or schema.get('__key__') != 'int':
            return None
        if (name == 'max' and os.environ.get('MX_CY_PLAIN_READS') != '1'
                and os.environ.get('MX_CY_NO_TABLE_MATCHERS') != '1'):
            # The declared spelling of the last key: later passes recognise a clamp on it.
            return node.with_changes(func=node.func.with_changes(attr=cst.Name('max_int')))
        from .helpers import module_name
        provider = 'modelx_cython.runtime._native_inputs'
        self.providers.add(provider)
        function = {'max': 'native_last_int', 'min': 'native_first_int', 'max_int': 'native_max_int'}[name]
        return cst.Call(func=cst.Attribute(cst.Name(module_name(provider)), cst.Name(function)),
                        args=[cst.Arg(node.func.value.value)])

    def leave_Comparison(self, original, updated):
        """``K in R.index`` / ``K not in R.index`` on a traced table: the provider's membership read."""
        if len(updated.comparisons) != 1:
            return updated
        target = updated.comparisons[0]
        if not isinstance(target.operator, (cst.In, cst.NotIn)):
            return updated
        index = target.comparator
        if not (isinstance(index, cst.Attribute) and index.attr.value == 'index'
                and self._schema(index.value) is not None):
            return updated
        from .helpers import module_name
        provider = 'modelx_cython.runtime._native_inputs'
        self.providers.add(provider)
        self.typed += 1
        call = cst.Call(func=cst.Attribute(cst.Name(module_name(provider)), cst.Name('native_contains')),
                        args=[cst.Arg(index.value), cst.Arg(updated.left)])
        if isinstance(target.operator, cst.NotIn):
            return cst.UnaryOperation(operator=cst.Not(), expression=call,
                                      lpar=updated.lpar, rpar=updated.rpar)
        return call.with_changes(lpar=updated.lpar, rpar=updated.rpar)

    def leave_Call(self, original, updated):
        func = original.func
        if (isinstance(func, cst.Name) and func.value == 'int' and len(original.args) == 1
                and not original.args[0].keyword and not original.args[0].star):
            inner = updated.args[0].value
            if (isinstance(inner, cst.Call) and isinstance(inner.func, cst.Attribute)
                    and inner.func.attr.value in ('max_int', 'native_last_int', 'native_first_int',
                                                  'native_max_int')):
                return inner               # int() of the C integer bound is the bound
        # Without the clamp recognisers the declared spelling index.max_int() has no other native route.
        names = ('max', 'min', 'max_int') if os.environ.get('MX_CY_NO_TABLE_MATCHERS') == '1' else ('max', 'min')
        for name in names:
            if original.args or not isinstance(func, cst.Attribute) or func.attr.value != name:
                continue
            bound = self._bound(original, name)
            if bound is not None:
                self.typed += 1
                return bound
        if (isinstance(func, cst.Name) and func.value == 'float' and len(original.args) == 1
                and not original.args[0].keyword and not original.args[0].star
                and isinstance(original.args[0].value, cst.Subscript)
                and isinstance(updated.args[0].value, cst.Call)
                and isinstance(updated.args[0].value.func, cst.Attribute)
                and updated.args[0].value.func.attr.value == 'lookup_float'):
            return updated.args[0].value
        if (isinstance(func, cst.Name) and func.value == 'float' and len(updated.args) == 1
                and not updated.args[0].keyword and not updated.args[0].star
                and isinstance(updated.args[0].value, cst.Subscript)):
            # float() of an integer column read is the float read of that column.
            call, _ = self._float_read(updated.args[0].value, ('int',))
            if call is not None:
                self.untyped.pop()
                self.typed += 1
                return call
        return updated
