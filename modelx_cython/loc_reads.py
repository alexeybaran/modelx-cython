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
  identical assignment counts as one). A local bound to a zero-argument Cell call is replaced
  by the call in the rewritten read: the call returns the same cached object.

``float(R.loc[key, "col"])`` loses the ``float()``: the typed read already returns a float.
For the same reason ``float()`` of an integer column read becomes the float read.

``G.loc[(g1, ..., gn, key), "col"]`` on a grouped table stays an object read: no generic native
grouped read exists yet (backlog C2/C3), and slicing per call was slower than the object read.

The decision uses the receiver's traced type, the traced column kind and a literal column
name. Every other use of ``.loc`` is left as written and runs as an ordinary call on the
table object, with the same meaning as in live modelx and exported Python. The count of
typed and untyped reads, with the reason, goes to the build's admission report.
"""

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

    def visit_FunctionDef(self, node):
        if not self._aliases:
            scan = _SingleAssignments()
            node.body.visit(scan)
            self._aliases = {name: value for name, value in scan.values.items()
                             if scan.counts.get(name) == 1 and value is not None}
        return True

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
                # No generic native grouped read exists yet; slicing per call through the Python
                # Records measured slower than the object .loc, so the read stays as written.
                return None, 'full key into a grouped table: no native grouped read'
            return None, 'receiver is not a traced table'
        receiver = self._direct(receiver)
        found = {schema.get(name) for name in names}
        if not found <= set(kinds):
            name = sorted(n for n in names if schema.get(n) not in kinds)[0]
            return None, f'column {name!r} has traced kind {schema.get(name)}'
        return cst.Call(func=cst.Attribute(receiver, cst.Name('lookup_float')),
                        args=(cst.Arg(key), cst.Arg(column))), None

    def _direct(self, receiver):
        """Replace a local bound once to a zero-argument Cell call by that call.

        A Cell call returns its cached value, so reading through the call or through the
        local is the same object; the direct form is the one later passes lower natively.
        """
        if isinstance(receiver, cst.Name) and receiver.value in self._aliases:
            value = self._aliases[receiver.value]
            if isinstance(value, cst.Call) and not value.args:
                chain = _extract_self_chain(value.func)
                if isinstance(chain, list) and chain:
                    status, _ = self.resolver.resolve_chain(self.space, chain)
                    if status == RESOLVED:
                        return value
        return receiver

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

    def leave_Subscript(self, original, updated):
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
        from .helpers import module_name
        provider = 'modelx_cython.runtime._native_inputs'
        self.providers.add(provider)
        function = {'max': 'table_last_int', 'min': 'table_first_int'}[name]
        return cst.Call(func=cst.Attribute(cst.Name(module_name(provider)), cst.Name(function)),
                        args=[cst.Arg(node.func.value.value)])

    def leave_Call(self, original, updated):
        func = original.func
        if (isinstance(func, cst.Name) and func.value == 'int' and len(original.args) == 1
                and not original.args[0].keyword and not original.args[0].star):
            inner = updated.args[0].value
            if (isinstance(inner, cst.Call) and isinstance(inner.func, cst.Attribute)
                    and inner.func.attr.value in ('table_last_int', 'table_first_int')):
                return inner               # int() of the C integer bound is the bound
        for name in ('max', 'min'):
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
