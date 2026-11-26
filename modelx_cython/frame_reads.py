"""Reads of a traced pandas DataFrame or Series lowered onto a PreparedTable (runtime ``_tables``).

The formula keeps the user's frame: the sample runs on it and the compiled model reads the same object. A read
whose form the compiler recognises is answered from a PreparedTable prepared from that frame object; every other
operation runs on the frame itself.

Recognised here, the receiver ``R`` a self-rooted zero-argument call of a cached Cell whose every traced value is a
pandas DataFrame or Series with a unique index (``tracer.frame_schema``), or a local assigned once from such a call:

- ``R.loc[key, "c"]`` / ``R.at[key, "c"]`` of a float64, int64 or object column, the key complete (one part per
  level); a Series ``R.loc[k]``, ``R.at[k]``, ``R[k]``;
- fields of a row read without building the row: ``R.loc[key]["c"]``; ``row = R.loc[key]`` then ``row["c"]``; and
  ``self.P()["c"]`` where P is a row cell, a cached zero-argument Cell whose formula is ``return R.loc[key]``. A
  key fixed per item has its row held per item (``_mx_tw_<n>``); a field reads as pandas gives it in the row (an
  int64 field of a row pandas widens to float64 is a float);
- index extremes, as C integers where the level is an int level: ``R.index.max()`` / ``.min()`` (one level; any
  other index gives pandas' own extreme label), ``max(R.index)`` / ``min(R.index)``, the extreme of the remaining
  level of a prefix ``R.loc[prefix].index.max()``, ``R.index.get_level_values(L).max()`` (L a level number or a
  name traced on every frame, also through a local) and ``max(a for a, s in R.index)``;
- the ordered lookup ``R.index.asof(x)`` and ``R.loc[prefix].index.asof(x)`` on an int level stored in pandas'
  order (``int(...)`` of it a C integer, any other use the numpy integer pandas returns); x NaN, not one number
  at run time, below the first label, or an index in another order runs pandas' own asof;
- membership ``key in R.index`` / ``not in`` with a complete key. A key part that is not one label at run time
  (a list, a mask, an index, an unhashable value) runs pandas' own membership, which answers or raises as live.
- a view of a prefix held in a local, ``v = R.loc[prefix]`` at the top of the formula: its prefix codes and stored
  rows in C locals (held per item when every part is fixed per item), ``v.loc[k, "c"]``, ``float(v.loc[k, name])``,
  ``v.index.max()`` / ``.min()`` and ``max(v.index)`` / ``min(v.index)`` of one remaining int level. When the view is
  not native the assignment runs on pandas where it stands, so a missing prefix raises where live raises it;
- walks of an index in pandas order: comprehensions and ``for`` loops over ``X.index`` (a view local, a whole frame,
  ``R.loc[prefix]``, a local holding an index), ``list`` / ``tuple`` / ``set`` of it; a label of an int level is a C
  integer, any other the label object pandas iteration gives. A whole frame is walked by pandas position; a prefix
  natively only while its stored rows are in pandas order (decided per table). A generator is walked natively only
  as the argument of a builtin that consumes it at once;
- a float64 or int64 column C of a view or prefix (``v["c"]``, ``R.loc[prefix]["c"]``, ``R.loc[prefix, "c"]``, or a
  local of one): its values and ``C.items()`` walked, ``C.tolist()``, ``list(C)``, a walk of ``C.index``, and
  ``C.iloc[i]`` / ``C.loc[k]``;
- ``for k, row in R.iterrows():`` over a one-level frame, ``row["c"]`` read from the stored row of the kind pandas
  gives it (the Python numbers iteration gives on a row with an object column), every load of ``row`` in the loop;

Where a read is lowered:

- in a class whose instances are items (an ItemSpace): in a formula with parameters (evaluated per step), and in
  a formula without parameters (once per item) when the frame is a Cell of a Space without items (prepared once
  per model, not per item). A frame or Series the item computes itself is prepared per item only for a formula
  with parameters that reads it by a computed key or walks it; reads by literal keys stay on pandas;
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
VALUE_READS = frozenset(('f64_1', 'f64_c', 'i64_1', 'i64_c', 'f64_row', 'i64_row'))
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


def row_cells(module_node, cls_info):
    """{cell: its ``R.loc[key]``} for the row cells of an item class: cached zero-argument Cells whose formula is
    ``return R.loc[key]``, ``R`` a self-rooted zero-argument call and ``self`` the key's only free name, that no
    other class of the module defines (or defines with the same formula, both plain Space classes), so that no
    derived class can give the Cell another value."""
    if cls_info.name not in cls_info.visitor.itemspace_classes:
        return {}
    classes = [n for n in getattr(module_node, 'body', ()) if isinstance(n, cst.ClassDef)]
    own = next((c for c in classes if c.name.value == cls_info.name), None)
    found = {}
    for func in (own.body.body if own is not None else ()):
        if not (isinstance(func, cst.FunctionDef) and func.name.value.startswith('_f_')):
            continue
        expr, name = _row_expression(func), func.name.value[3:]
        if expr is not None and name in cls_info.cells and all(
                _same_row_cell(own, other, name, expr) for other in classes if other is not own and any(
                    isinstance(f, cst.FunctionDef) and f.name.value in ('_f_' + name, name) for f in other.body.body)):
            found[name] = expr
    return found


def _body(func):
    """The statements of a function without its docstring."""
    body = list(func.body.body) if isinstance(func.body, cst.IndentedBlock) else [func.body]
    if (body and isinstance(body[0], cst.SimpleStatementLine) and len(body[0].body) == 1
            and isinstance(body[0].body[0], cst.Expr)
            and isinstance(body[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString))):
        body = body[1:]
    return body


def _row_expression(func):
    """The ``R.loc[key]`` of a formula ``return R.loc[key]`` (``self`` the key's only free name), else None."""
    params = func.params
    if (len(params.params) != 1 or params.params[0].name.value != 'self' or params.star_arg
            not in (None, cst.MaybeSentinel.DEFAULT) or params.kwonly_params or params.star_kwarg
            or params.posonly_params):
        return None
    body = _body(func)
    if len(body) != 1 or not isinstance(body[0], cst.SimpleStatementLine) or len(body[0].body) != 1:
        return None
    ret = body[0].body[0]
    expr = ret.value if isinstance(ret, cst.Return) else None
    if not (isinstance(expr, cst.Subscript) and len(expr.slice) == 1 and isinstance(expr.slice[0].slice, cst.Index)
            and expr.slice[0].slice.star is None and isinstance(expr.value, cst.Attribute)
            and expr.value.attr.value == 'loc' and isinstance(expr.value.value, cst.Call) and not expr.value.value.args):
        return None
    root = expr.value.value.func
    while isinstance(root, cst.Attribute):
        root = root.value
    if not (isinstance(root, cst.Name) and root.value == 'self'):
        return None
    names, plain = set(), [True]

    class Names(cst.CSTVisitor):
        def visit_Attribute(self, node):
            node.value.visit(self)
            return False

        def visit_Name(self, node):
            names.add(node.value)

        def visit_Lambda(self, node):
            plain[0] = False
            return False
        visit_GeneratorExp = visit_ListComp = visit_SetComp = visit_DictComp = visit_NamedExpr = visit_Lambda
    expr.slice[0].slice.value.visit(Names())
    if not plain[0] or names - {'self'}:
        return None
    return expr.with_changes(lpar=(), rpar=())


def _same_row_cell(owner, other, name, expr):
    """Two independent Space classes of the module define the same row cell with the same formula."""
    base = cst.parse_expression('_mx_sys.BaseSpace')
    for cls in (owner, other):
        if (not cls.name.value.startswith('_c_') or len(cls.bases) != 1
                or not cls.bases[0].value.deep_equals(base) or cls.decorators or cls.keywords):
            return False
    formulas = [f for f in other.body.body if isinstance(f, cst.FunctionDef) and f.name.value == '_f_' + name]
    if len(formulas) != 1:
        return False
    found = _row_expression(formulas[0])
    return found is not None and found.deep_equals(expr)


def _walk_extreme(call):
    """``max(a for a, s in X)`` / ``min(...)``, no condition, the element one name of the tuple target:
    (X, the target's names, the element's position), else None."""
    if not (isinstance(call.func, cst.Name) and call.func.value in ('max', 'min') and len(call.args) == 1
            and not call.args[0].keyword and not call.args[0].star and isinstance(call.args[0].value, cst.GeneratorExp)):
        return None
    gen = call.args[0].value
    comp, target = gen.for_in, gen.for_in.target
    if (comp.ifs or comp.inner_for_in is not None or comp.asynchronous is not None or not isinstance(gen.elt, cst.Name)
            or not isinstance(target, cst.Tuple) or not all(isinstance(e, cst.Element) and isinstance(e.value, cst.Name)
                                                            for e in target.elements)):
        return None
    names = [e.value.value for e in target.elements]
    return (comp.iter, names, names.index(gen.elt.value)) if names.count(gen.elt.value) == 1 else None


class _Hold:
    """One (receiver, column, kind) a formula reads natively; ``table`` (and ``state`` in items) are the fields of
    the receiver's binding, shared by every read of that receiver."""

    def __init__(self, table, state, receiver, label, kind):
        self.table, self.state, self.receiver, self.label, self.kind = table, state, receiver, label, kind


class _View:
    """The rows of a prefix of a frame receiver held in a local ``name``: the stored rows [start, stop) of the
    prefix (C locals ``start``, ``stop``; start < 0 when the view is not native), its level codes (``codes``),
    the expression the pandas branches evaluate in place of the local (``source``), and whether a read used its
    rows (``used``). ``untyped``: a prefix part is not known from the text to be one label, so a read through the
    view stays a Python object (its pandas branch may be a selection)."""

    def __init__(self, name, call, schema, depth, codes, start, stop, source, untyped):
        self.name, self.call, self.schema, self.depth, self.codes = name, call, schema, depth, codes
        self.start, self.stop, self.source, self.untyped = start, stop, source, untyped
        self.setup, self.used = None, False


class _Walk:
    """The rows an index walk visits in pandas order: positions [start, stop) of a whole table (``whole``: each
    mapped to its stored row) or the stored rows of a prefix of ``depth`` levels; native while ``test`` holds.
    ``swaps``: the locals the pandas walk replaces by what they stand for."""

    def __init__(self, call, schema, depth, hold, start, stop, test, whole):
        self.call, self.schema, self.depth, self.hold = call, schema, depth, hold
        self.start, self.stop, self.test, self.whole = start, stop, test, whole
        self.swaps = {}
        self.column = None      # (hold, kind) of a column walked by its values
        self.codes = []         # the level codes of a prefix
        self.mode = 'index'     # 'index': the labels; 'values': a column's values; 'items': (label, value)

    def swapped(self, expr):
        """``expr`` with the locals of this walk replaced by what they stand for."""
        swaps = self.swaps

        class Swap(cst.CSTTransformer):
            def leave_Name(self, original, new):
                return swaps[original.value] if original.value in swaps else new
        return expr.visit(Swap()) if swaps else expr


# builtins that consume a generator argument at once: a generator over an index is walked natively only there
WALK_BUILTINS = frozenset(('sorted', 'list', 'tuple', 'set', 'frozenset', 'sum', 'min', 'max', 'any', 'all'))


def _rebinds(comp, names):
    """An inner ``for``, comprehension, lambda or walrus of ``comp`` binds one of ``names``."""
    from .key_facts import _bound_names
    found = []

    class Scan(cst.CSTVisitor):
        def visit_CompFor(self, node):
            if node is not comp.for_in:
                found.extend(n for n in _bound_names(node.target) if n in names)

        def visit_Lambda(self, node):
            found.extend(p.name.value for p in node.params.params if p.name.value in names)
            found.append('lambda')      # a lambda is a scope of its own: not entered by the substitution

        def visit_NamedExpr(self, node):
            found.extend(n for n in _bound_names(node.target) if n in names)
    comp.visit(Scan())
    return bool(found)


class _Substitute(cst.CSTTransformer):
    """Each load of a walked name replaced by its label expression (attribute names and keywords kept)."""

    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def leave_Name(self, original, updated):
        found = self.labels.get(original.value)
        return found.deep_clone() if found is not None else updated

    def leave_Attribute(self, original, updated):
        return updated.with_changes(attr=original.attr)

    def leave_Arg(self, original, updated):
        return updated.with_changes(keyword=original.keyword)


class _EveryReceiver(frozenset):
    """The receivers a formula edits when an edit goes through a name that may be any of them: all."""

    def __contains__(self, item):
        return True


_NO_LABEL = object()


class FrameReads:
    """``lower(node)`` returns the formula with recognised reads lowered (the node itself when none is)."""

    def __init__(self, resolver, cls_info, cells, providers, *, builtin_int, builtin_float, builtin_minmax=False,
                 row_cells=None):
        self.row_cells = row_cells or {}    # row cell of the class -> its ``R.loc[key]`` expression
        self.resolver, self.cls_info, self.cells = resolver, cls_info, cells
        self.providers = providers
        self.builtin_int, self.builtin_float = builtin_int, builtin_float
        self.builtin_minmax = builtin_minmax
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
        plain_int = facts.is_int

        def is_int(expr, depth=0):      # ``max(a for a, s in T.index)`` of an int level
            return plain_int(expr, depth) or (
                'extreme of an integer level' if isinstance(expr, cst.Call) and self._walk_extreme_int(expr) else None)
        facts.is_int = is_int
        facts._int_index_read = lambda expr: (self.extreme(expr) is not None or self.int_label(expr)
                                              or self.level_values(expr) is not None)
        return facts

    def _walk_extreme_int(self, expr):
        """``max(a for a, s in R.index)`` / ``min(...)`` with ``a`` naming an int level of a unique frame."""
        walk = _walk_extreme(expr)
        index = walk[0] if walk else None
        if not (isinstance(index, cst.Attribute) and index.attr.value == 'index'):
            return False
        found = self._receiver(index.value)
        if found is None or not found[1].get('unique'):
            return False
        levels = found[1]['levels']
        return len(levels) == len(walk[1]) and levels[walk[2]] == 'int'

    def extreme(self, expr):
        """``R.index.max()`` / ``R.index.min()`` on a frame receiver with one unique int level: (R call, 'max'|'min')."""
        if not (isinstance(expr, cst.Call) and not expr.args and isinstance(expr.func, cst.Attribute)
                and expr.func.attr.value in ('max', 'min') and isinstance(expr.func.value, cst.Attribute)
                and expr.func.value.attr.value == 'index'):
            return None
        found = self._receiver(expr.func.value.value)
        if (found is None or found[1]['levels'] != ('int',) or not found[1].get('unique')
                or _code(found[0]) in self.mutated):
            return None
        return found[0], expr.func.attr.value

    def int_label(self, expr):
        """``R.loc[prefix].index.max()`` / ``.min()`` whose one remaining level is an int level, also through a view
        local assigned once to ``R.loc[prefix]``: an integer."""
        if not (isinstance(expr, cst.Call) and isinstance(expr.func, cst.Attribute) and expr.func.attr.value in (
                'max', 'min') and isinstance(expr.func.value, cst.Attribute) and expr.func.value.attr.value == 'index'):
            return False
        value = expr.func.value.value
        if isinstance(value, cst.Name) and self._once(value.value):
            value = self._value_of(value.value)
        prefix = self.prefix_of(value)
        if prefix is None:
            return False
        (_, schema), parts = prefix
        return len(schema['levels']) - len(parts) == 1 and schema['levels'][len(parts)] == 'int'

    def prefix_of(self, value):
        """``R.loc[prefix]`` with R a frame receiver (unique index, not edited) of more levels than the prefix, the
        prefix one part or a tuple display of parts: ((R call, schema), prefix parts), else None."""
        if not (isinstance(value, cst.Subscript) and isinstance(value.value, cst.Attribute)
                and value.value.attr.value == 'loc'):
            return None
        parts = _index_parts(value)
        found = self._receiver(value.value.value)
        if (parts is None or len(parts) != 1 or found is None or not found[1].get('unique')
                or _code(found[0]) in self.mutated):
            return None
        parts = _parts(parts[0])
        if not 0 < len(parts) < len(found[1]['levels']) or any(map(_several, parts)):
            return None
        return found, parts

    def level_values(self, expr):
        """``R.index.get_level_values(L).max()`` / ``.min()`` (also through a local assigned once to the level
        values) of an int level, L a level number or a name traced on every frame: (R call, schema, level, name,
        'max'|'min', the local or None), else None."""
        func = expr.func if isinstance(expr, cst.Call) else None
        if not (isinstance(func, cst.Attribute) and func.attr.value in ('max', 'min') and not expr.args):
            return None
        inner, local = func.value, None
        if isinstance(inner, cst.Name):
            local, inner = inner.value, self._value_of(inner.value)
        if not (isinstance(inner, cst.Call) and isinstance(inner.func, cst.Attribute)
                and inner.func.attr.value == 'get_level_values' and len(inner.args) == 1 and not inner.args[0].keyword
                and not inner.args[0].star and isinstance(inner.func.value, cst.Attribute)
                and inner.func.value.attr.value == 'index'):
            return None
        found = self._receiver(inner.func.value.value)
        if found is None or not found[1].get('unique') or _code(found[0]) in self.mutated:
            return None
        call, schema = found
        arg = inner.args[0].value
        names = list(schema.get('names') or ())
        if isinstance(arg, cst.Integer):
            index, name = int(arg.value, 0), None
        elif _literal_str(arg) is not None and names.count(_literal_str(arg)) == 1:
            name = _literal_str(arg)
            index = names.index(name)
        else:
            return None
        if not 0 <= index < len(schema['levels']) or schema['levels'][index] != 'int':
            return None
        return call, schema, index, name, func.attr.value, local

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
            elif isinstance(parent, cst.Element) and isinstance(parents.get(id(parent)), cst.Tuple) and isinstance(
                    parents.get(id(parents.get(id(parent)))), (cst.Index, cst.SubscriptElement)):
                return False        # a part of a tuple key
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
        """(cell call, schema) when ``expr`` is a frame receiver, else None. In an item class a frame or Series the
        item computes itself (its own result frames, a row or column it selects) is prepared per item, at about
        0.1-2M instructions, so it is a receiver only in a formula with parameters that reads it by a computed key
        (a key part that is not a literal or a choice between literals) or walks it: literal keys name a few values
        per item, which pandas reads for less than the preparation. Decided from the text, not from call counts."""
        found = self._receiver_of(expr)
        if found is not None and self.per_item and not self.shared(found[0]) and (
                not self.cells.has_args() or _code(found[0]) in self._literal_reads()):
            return None
        return found

    def _receiver_of(self, expr):
        schema = self._schema_of_call(expr)
        found = (expr, schema) if schema is not None else None
        if found is None:
            value = self._value_of(expr.value) if isinstance(expr, cst.Name) else None
            schema = self._schema_of_call(value) if value is not None else None
            found = (value, schema) if schema is not None else None
        return found

    def _literal_reads(self):
        """Codes of the frames the item computes itself that this formula only reads by literal keys: ``R["c"]``,
        ``R.loc["a", "c"]``, ``R.at[k]`` whose every key part is a literal or a conditional between literals."""
        found = getattr(self, '_literal_codes', None)
        if found is not None:
            return found
        owner, uses = self, {}

        def literal(node):
            if isinstance(node, (cst.SimpleString, cst.Integer, cst.Float)):
                return True
            if isinstance(node, cst.IfExp):
                return literal(node.body) and literal(node.orelse)
            return isinstance(node, cst.Tuple) and all(isinstance(e, cst.Element) and literal(e.value)
                                                       for e in node.elements)

        class Scan(cst.CSTVisitor):
            def visit_FunctionDef(inner, node):
                return node is owner._node

            def visit_Lambda(inner, node):
                return False

            def visit_Call(inner, node):
                inner._use(node)

            def visit_Name(inner, node):
                inner._use(node)

            def _use(inner, node):
                got = owner._receiver_of(node) if isinstance(node, (cst.Call, cst.Name)) else None
                if got is None or owner.shared(got[0]):
                    return
                parents = owner.facts.formula.parents
                up = parents.get(id(node))
                if isinstance(node, cst.Call) and isinstance(up, cst.Assign) and up.value is node:
                    return      # ``r = R()``: the uses of the local are its reads
                if isinstance(up, cst.Attribute) and up.attr.value in ('loc', 'at') and up.value is node:
                    up = parents.get(id(up))
                parts = _index_parts(up) if isinstance(up, cst.Subscript) else None
                code = _code(got[0])
                uses[code] = uses.get(code, True) and parts is not None and all(map(literal, parts))
        self._node.visit(Scan())
        self._literal_codes = {code for code, only in uses.items() if only}
        return self._literal_codes

    # ------------------------------------------------------------------ driver
    def lower(self, node):
        params = node.params.params
        if not params or params[0].name.value != 'self' or not getattr(self.resolver, 'frame_schemas', None):
            return node
        self.per_item = self.cls_info.name in self.cls_info.visitor.itemspace_classes
        self._node = node
        if not self.per_item and not self.cells.has_args():
            return node         # evaluated once per model: its reads stay on pandas
        self.facts = self._facts(node)
        from .native_sources import module_name
        self.module = module_name(PROVIDER)
        self.holds = {}         # (receiver code, label, kind) -> _Hold
        self.bindings = {}      # receiver code -> its binding (_Hold of the table field)
        self.code_fields, self.row_fields, self.range_fields = {}, {}, {}
        self.mutated = self._mutated(node)
        self.row_locals = self._row_locals(node)
        self.views = {}         # view local -> _View
        self.declarations = []  # C locals of the formula
        rewriter = _Rewrite(self)
        body = []
        for statement in node.body.body:
            self.new_items = []
            view = self.view_assignment(statement)
            new = rewriter.view_statement(statement, view) if view is not None else statement.visit(rewriter)
            if self.new_items:
                body.extend(self._hold_statements(item) for item in self.new_items)
            body.append(new)
        self._finish_views(body, rewriter)
        if not self.holds:      # every lowered read takes a hold: nothing was lowered
            return node
        body = self._drop_unused_locals(body, rewriter.replaced_names | set(rewriter.key_locals) | set(rewriter.swaps))
        if self.declarations:
            from .key_facts import _is_docstring
            head = 1 if body and _is_docstring(body[0]) else 0
            body[head:head] = [cst.parse_statement(f'{name}: _mx_cy.Py_ssize_t') for name in self.declarations]
        self.providers.add(PROVIDER)
        return node.with_changes(body=node.body.with_changes(body=self._flatten(body)))

    # ------------------------------------------------------------------ views of a prefix
    def view_assignment(self, statement):
        """``name = R.loc[prefix]`` at the top level of the formula, ``name`` bound once and read in no nested
        function, R a frame receiver of more levels than the prefix (``prefix_of``), at most ``MAX_PARTS`` levels:
        (name, R call, schema, prefix parts), else None."""
        assigned = self._simple_assign(statement)
        if assigned is None:
            return None
        name, value = assigned
        if not self._once(name) or any(nested for _, nested in self.facts.formula.loads.get(name, ())):
            return None
        found = self.prefix_of(value)
        if found is None or found[0][1]['kind'] != 'frame' or len(found[0][1]['levels']) > MAX_PARTS:
            return None
        (call, schema), parts = found
        return name, call, schema, parts

    def _finish_views(self, body, rewriter):
        """Each view's setup, once the formula is rewritten: dropped when no read used the view's rows; else the
        assignment stays as written when a use of the local was not lowered, and otherwise runs only when the view
        is not native (its pandas value, or the error live raises there)."""
        for view in self.views.values():
            setup = view.setup
            assignment = setup[-1]
            if not view.used:
                setup[:] = [assignment]
                rewriter.swaps.pop(view.name, None)     # kept as written (a missing prefix raises there)
                continue
            loads = [0]

            class Loads(cst.CSTVisitor):
                def visit_Name(inner, node):
                    if node.value == view.name:
                        loads[0] += 1
            cst.Module(body=self._flatten(body)).visit(Loads())
            if loads[0] <= 1:       # the assignment's target only
                setup[-1] = cst.If(test=cst.parse_expression(f'{view.start} < 0'), body=cst.IndentedBlock(
                    body=[cst.SimpleStatementLine(body=[cst.Expr(view.source)])]))

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
        if isinstance(hold, tuple) and hold[0] == 'range':      # the stored rows of a prefix held per item
            _, (start, stop), table, args, test = hold
            ref = f'_mx_cy.cast({m}.PreparedRef, self.{table})'
            return [cst.parse_statement(
                f'if self.{start} == 0:\n'
                f'    self.{start} = ({m}.view_start({ref}, {args}) if {test} else -1) + 2\n'
                f'    self.{stop} = {m}.view_stop({ref}, {args}) if self.{start} > 1 else -1\n')]
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
        """The value a held field takes: one plus a level code, or one plus the stored row of a key."""
        if what == 'row':
            return f'{self.module}.hold_row(self.{table}, {_code(level_or_key)})'
        return f'{self.module}.hold_code(self.{table}, {level_or_key}, {_code(extra)})'

    def _row_locals(self, node):
        """``row = R.loc[key]`` (at any depth of the formula's own scope, ``row`` bound once), R a frame receiver of a
        unique index and the key complete, every load of ``row`` a read of a literal column ``row["c"]`` or of a
        computed one under ``float()``: {name: (R call, schema, the assignment)}. Also the row of ``for k, row in
        R.iterrows():`` (R of one level, ``row`` bound by that loop only, read only in its body): {name: (R call,
        schema, the loop)}."""
        f, found = self.facts.formula, {}
        owner = self

        def field_reads(name, schema):
            columns = dict(schema['columns'])
            for use, nested in f.loads.get(name, ()):
                sub = f.parents.get(id(use))
                parts = _index_parts(sub) if isinstance(sub, cst.Subscript) else None
                if nested or parts is None or sub.value is not use or len(parts) != 1:
                    return False
                if _literal_str(parts[0]) in columns:
                    continue
                up = f.parents.get(id(sub))
                call = f.parents.get(id(up)) if isinstance(up, cst.Arg) else None
                if not (isinstance(call, cst.Call) and isinstance(call.func, cst.Name) and call.func.value == 'float'
                        and len(call.args) == 1 and _literal_str(parts[0]) is None and owner.builtin_float):
                    return False
            return True

        class Scan(cst.CSTVisitor):
            def visit_FunctionDef(inner, n):
                return n is node

            def visit_Lambda(inner, n):
                return False

            def visit_Assign(inner, n):
                if not (len(n.targets) == 1 and isinstance(n.targets[0].target, cst.Name)
                        and isinstance(n.value, cst.Subscript) and isinstance(n.value.value, cst.Attribute)
                        and n.value.value.attr.value == 'loc'):
                    return
                name = n.targets[0].target.value
                parts = _index_parts(n.value)
                got = owner._receiver(n.value.value.value)
                if (got is None or parts is None or len(parts) != 1 or got[1]['kind'] != 'frame'
                        or not got[1].get('unique') or _code(got[0]) in owner.mutated or not owner._once(name)):
                    return
                levels, key = got[1]['levels'], parts[0]
                if (len(levels) == 1) == isinstance(key, cst.Tuple) or (
                        isinstance(key, cst.Tuple) and len(key.elements) != len(levels)):
                    return
                if field_reads(name, got[1]):
                    found[name] = (got[0], got[1], n)

            def visit_For(inner, n):     # ``for k, row in R.iterrows():`` over a one-level frame
                it, target = n.iter, n.target
                if not (isinstance(it, cst.Call) and not it.args and isinstance(it.func, cst.Attribute)
                        and it.func.attr.value == 'iterrows' and n.asynchronous is None
                        and isinstance(target, cst.Tuple) and len(target.elements) == 2
                        and all(isinstance(e, cst.Element) and isinstance(e.value, cst.Name) for e in target.elements)):
                    return
                got = owner._receiver(it.func.value)
                key, name = (e.value.value for e in target.elements)
                if (got is None or got[1]['kind'] != 'frame' or not got[1].get('unique') or len(got[1]['levels']) != 1
                        or _code(got[0]) in owner.mutated or key == name or f.binds.get(name, 0) != 1
                        or name in owner.facts._params or name in found):
                    return
                if field_reads(name, got[1]) and all(within(use, n.body) for use, _ in f.loads.get(name, ())):
                    found[name] = (got[0], got[1], n)

        def within(node, block):
            """``node`` is inside ``block`` (a load of a loop's row in its body, never after the loop)."""
            while node is not None and node is not block:
                node = f.parents.get(id(node))
            return node is block
        node.visit(Scan())
        return found

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

    def fixed_part(self, part):
        """A key part fixed per item: ``item_constant``, or one of ``_fixed_value``."""
        return item_constant(part, self.cached_call) or self._fixed_value(part)

    def row_field(self, hold, key_expr, emit=None):
        """The item field holding one plus the stored row of ``key_expr`` (a key fixed per item)."""
        key = (hold.table, _code(key_expr))
        if key not in self.row_fields:
            self.row_fields[key] = hold_table(self.cls_info).field('prepared_row', key)
            self.new_items.append(('row', self.row_fields[key], hold.table, key_expr if emit is None else emit, None))
        return self.row_fields[key]

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
        self.swaps = {}         # row local -> the row cell call it stands for (fallbacks, dropped locals)
        self.native = {}        # id(lowered read) -> column kind, for float()/int() around it
        self.updated_of = {}    # id(original node) -> its rewritten node (key parts are emitted with their reads lowered)
        self.unboxed = set()    # ids of lowered reads whose live value is a Python number (never a numpy scalar)

    def on_leave(self, original, updated):
        result = super().on_leave(original, updated)
        kind = self.native.get(id(result))
        if (kind in CTYPE and result is not updated and isinstance(original, cst.Subscript)
                and id(result) not in self.unboxed
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
        return (self.row_local_read(original, updated, parts) or self.row_cell_field(original, updated, parts)
                or self._row_field(original, updated, parts) or self.view_read(original, updated, parts)
                or self.column_item(original, updated, parts)
                or self.keyed_read(original, updated, value, parts) or updated)

    # ------------------------------------------------------------------ fields of a row
    @staticmethod
    def _branch(test, native, fallback):
        """``if test: native else: fallback`` (statements)."""
        return cst.If(test=test, body=cst.IndentedBlock(body=native),
                      orelse=cst.Else(body=cst.IndentedBlock(body=fallback)))

    @staticmethod
    def _assign(name, value):
        return cst.SimpleStatementLine([cst.Assign([cst.AssignTarget(cst.Name(name))], value)])

    def leave_SimpleStatementLine(self, original, updated):
        """``row = R.loc[key]``: the stored row in a C local (KeyError as pandas); the pandas row only when the
        table is not of the kinds the build traced (every column checked: kind -4) or a key part is not one label."""
        o = self.o
        if not (len(original.body) == 1 and isinstance(original.body[0], cst.Assign)):
            return updated
        name = original.body[0].targets[0].target
        found = o.row_locals.get(name.value) if isinstance(name, cst.Name) else None
        if found is None or found[2] is not original.body[0]:
            return updated
        call, schema, assign = found
        hold = self._take(call, _NO_LABEL, -4)
        hold.row_columns = True
        key = _index_parts(assign.value)[0]
        var = f'_mx_rw_{name.value}'
        if var not in o.declarations:
            o.declarations.append(var)
        fallback = [cst.parse_statement(f'{var} = -1'),
                    original.with_changes(body=[assign.with_changes(value=self._fallback(assign.value, call))])]
        parts = _parts(key)
        if o.per_item and o.cells.has_args() and all(o.fixed_part(p) for p in parts):
            # a key fixed per item (``row = T.loc[cost_id()]`` in a per-step cell): the row held per item, looked
            # up once, not at every step
            field = o.row_field(hold, key, self._detached(key))
            native = self._assign(var, cst.parse_expression(f'self.{field} - 1'))
            return self._branch(self._native_test(hold, [cst.parse_expression(f'self.{field} > 0')]), [native],
                                fallback)
        (suffix, args), conditions = self._read_args(hold, parts, schema['levels'])
        native = self._assign(var, self._m('row' + suffix, o.ref(hold), *args))
        return self._branch(self._native_test(hold, conditions), [native], fallback)

    # ------------------------------------------------------------------ views of a prefix
    def view_statement(self, statement, found):
        """``name = R.loc[prefix]``: the prefix codes and the view's stored rows [start, stop) in C locals before
        the assignment (``_finish_views`` decides what of it stays). A prefix fixed per item has its codes and
        rows held per item. The view is native while start >= 0."""
        o = self.o
        name, call, schema, parts = found
        for part in parts:          # the parts with their own reads lowered (a field of a row cell)
            part.visit(self)
        levels = schema['levels']
        hold = self._take(call, _NO_LABEL, -1)
        value = statement.body[0].value
        if isinstance(value.value.value, cst.Name):
            self.replaced_names.add(value.value.value.value)
        test, statements, codes, untyped = self._native_test(hold, []), [], [], False
        held = o.per_item and o.cells.has_args()
        for level, part in enumerate(parts):
            if o.item_fixed(part, levels[level]):
                conditions = []
                codes.append(self._held(hold, level, part, conditions))
                test = self._conjunction([test, *conditions])
                continue
            held, conditions = False, []
            code = self.code_of(hold, level, part, levels[level], conditions)
            untyped = untyped or any('_mx_okey_' in _code(c) for c in conditions)
            var = f'_mx_vc_{name}_{level}'
            o.declarations.append(var)
            statements.append(self._assign(var, cst.IfExp(
                test=self._conjunction([self._native_test(hold, []), *conditions]), body=code,
                orelse=cst.parse_expression('-1'))))
            codes.append(cst.Name(var))
        args = [cst.Integer(str(len(parts))), *codes, *[cst.Integer('0')] * (MAX_PARTS - 1 - len(codes))]
        start, stop = f'_mx_vs_{name}', f'_mx_ve_{name}'
        o.declarations += [start, stop]
        if held:        # every part fixed per item: the rows are held per item too
            key = (hold.table, _code(cst.Tuple([cst.Element(c) for c in codes])))
            if key not in o.range_fields:
                table = hold_table(o.cls_info)
                o.range_fields[key] = (table.field('prepared_row', ('view start', *key)),
                                       table.field('prepared_row', ('view stop', *key)))
                o.new_items.append(('range', o.range_fields[key], hold.table, ', '.join(map(_code, args)),
                                    _code(test)))
            first, last = o.range_fields[key]
            statements += [self._assign(start, cst.parse_expression(f'self.{first} - 2')),
                           self._assign(stop, cst.parse_expression(f'self.{last}'))]
        else:
            statements += [
                self._assign(start, cst.IfExp(test=test, body=self._m('view_start', o.ref(hold), *args),
                                              orelse=cst.parse_expression('-1'))),
                self._assign(stop, cst.IfExp(test=cst.parse_expression(f'{start} >= 0'),
                                             body=self._m('view_stop', o.ref(hold), *args),
                                             orelse=cst.parse_expression('-1')))]
        source = value.with_changes(value=value.value.with_changes(value=call))
        view = o.views[name] = _View(name, call, schema, len(parts), codes, start, stop, source, untyped)
        view.setup = [*statements, statement]
        self.swaps[name] = source
        return view.setup

    def _view_of(self, expr):
        """The view a view local ``expr`` names, else None."""
        return self.o.views.get(expr.value) if isinstance(expr, cst.Name) else None

    def _own_parts(self, view, key):
        """The parts of a key on the remaining levels of a view (one part, or a tuple display of one part per
        level), else None."""
        remaining = len(view.schema['levels']) - view.depth
        if (remaining == 1) == isinstance(key, cst.Tuple):
            return None
        own = _parts(key)
        if len(own) != remaining or any(map(_several, own)):
            return None
        return own

    def _view_codes(self, view, hold, own, conditions):
        """The level codes of a complete key: the view's prefix codes, then those of ``own``; ``conditions`` gets
        the view's validity and the parts' own conditions."""
        o = self.o
        levels = view.schema['levels']
        view.used = True
        conditions.append(cst.parse_expression(f'{view.start} >= 0'))
        return self.code_args(list(view.codes) + [
            self._held(hold, level, part, conditions) if o.item_fixed(part, levels[level])
            else self.code_of(hold, level, part, levels[level], conditions)
            for level, part in enumerate(own, view.depth)])

    def view_read(self, original, updated, parts):
        """``V.loc[k, "c"]`` / ``V.at[k, "c"]`` of a view local: the prefix codes, then k's; ``float(V.loc[k,
        name])`` with the column chosen at run time."""
        value = original.value
        view = (self._view_of(value.value) if isinstance(value, cst.Attribute) and value.attr.value in ('loc', 'at')
                else None)
        own = self._own_parts(view, parts[0]) if view is not None and len(parts) == 2 else None
        if own is None:
            return None
        label = _literal_str(parts[1])
        if label is None:
            return self.named_read(original, updated, view.call, view.schema, parts[0], parts[1], (view, own))
        kind = dict(view.schema['columns']).get(label)
        if kind not in ('f64', 'i64', 'obj'):
            return None
        o = self.o
        hold = self._take(view.call, label, {'f64': 0, 'i64': 1, 'obj': 2}[kind])
        conditions = []
        suffix, args = self._view_codes(view, hold, own, conditions)
        test, fallback = self._native_test(hold, conditions), self._fallback(updated, view.call)
        if kind == 'obj':
            row = self._next(self._m('row' + suffix, o.ref(hold), *args))
            return self._lowered(test, self._m('object_field', o.ref(hold), o.slot(hold), row), fallback)
        return self._lowered(test, self._m(kind + suffix, o.ref(hold), o.slot(hold), *args), fallback,
                             None if view.untyped else kind)

    def view_extreme(self, original, updated, view, which):
        """The extreme label of the one remaining int level of a view: at its first or last stored row (the rows of
        a prefix are stored in label order of that level), a C integer."""
        o = self.o
        levels = view.schema['levels']
        if len(levels) - view.depth != 1 or levels[view.depth] != 'int':
            return None
        hold = self._take(view.call, _NO_LABEL, -1)
        view.used = True
        row = cst.parse_expression(f'{view.stop} - 1' if which == 'max' else view.start)
        test = self._native_test(hold, [cst.parse_expression(f'{view.start} >= 0')])
        read = self._m('label_ll', o.ref(hold), row, cst.Integer(str(view.depth)))
        return self._lowered(test, read, self._fallback(updated, view.call), None if view.untyped else 'i64')

    # ------------------------------------------------------------------ walks of an index
    def _walk(self, it):
        """What a walk of ``it`` visits in pandas order, else None: the labels of ``X.index`` (X a view local, a whole
        frame receiver, ``R.loc[prefix]`` or a column of one of them; also through a local assigned once to
        ``X.index``); the values of a column C of a view or prefix (``_column``), or its (label, value) pairs
        ``C.items()``."""
        o = self.o
        swaps = {}
        if isinstance(it, cst.Name):
            value = o._value_of(it.value) if o._once(it.value) else None
            if isinstance(value, cst.Attribute) and value.attr.value == 'index':
                swaps[it.value] = value
                self.swaps.setdefault(it.value, value)      # the local is dropped once unread
                it = value
        if isinstance(it, cst.Attribute) and it.attr.value == 'index':
            walk = self._rows(it.value, swaps) or self._column(it.value, swaps)
            if walk is not None:
                walk.mode = 'index'
            return walk
        items = (isinstance(it, cst.Call) and not it.args and isinstance(it.func, cst.Attribute)
                 and it.func.attr.value == 'items')
        walk = self._column(it.func.value if items else it, swaps)
        if walk is not None:
            walk.mode = 'items' if items else 'values'
        return walk

    def _rows(self, target, swaps):
        """The rows of X in pandas order: a view local, a whole frame receiver, or ``R.loc[prefix]``; else None."""
        o = self.o
        view = self._view_of(target)
        whole = o._receiver(target) if view is None else None
        if view is not None:
            hold = self._take(view.call, view.depth, -7)
            view.used = True
            walk = _Walk(view.call, view.schema, view.depth, hold, cst.Name(view.start), cst.Name(view.stop),
                         self._native_test(hold, [cst.parse_expression(f'{view.start} >= 0')]), False)
            walk.codes = list(view.codes)
            swaps[view.name] = view.source
        elif whole is not None:
            if not self._plain(*whole):
                return None
            hold = self._take(whole[0], _NO_LABEL, -1)
            walk = _Walk(whole[0], whole[1], 0, hold, cst.Integer('0'), self._m('nrows', o.ref(hold)),
                         self._native_test(hold, []), True)
            if isinstance(target, cst.Name):
                swaps[target.value] = whole[0]
                self.replaced_names.add(target.value)
        else:
            prefix = o.prefix_of(target)
            if prefix is None or len(prefix[0][1]['levels']) > MAX_PARTS:
                return None
            (call, schema), parts = prefix
            levels = schema['levels']
            hold = self._take(call, len(parts), -7)
            conditions, codes = [], []
            for level, part in enumerate(parts):    # the prefix codes once, into C locals of the test
                if o.item_fixed(part, levels[level]):
                    codes.append(self._held(hold, level, part, conditions))
                    continue
                code = self.code_of(hold, level, part, levels[level], conditions)
                o.key_count += 1
                conditions.append(cst.parse_expression(f'(_mx_vc_{o.key_count} := {_code(code)}) >= 0'))
                codes.append(cst.Name(f'_mx_vc_{o.key_count}'))
            args = [cst.Integer(str(len(parts))), *codes, *[cst.Integer('0')] * (MAX_PARTS - 1 - len(codes))]
            o.key_count += 1
            start = f'_mx_vs_{o.key_count}'
            conditions.append(cst.parse_expression(
                f'({start} := {_code(self._m("view_start", o.ref(hold), *args))}) >= 0'))
            walk = _Walk(call, schema, len(parts), hold, cst.Name(start), self._m('view_stop', o.ref(hold), *args),
                         self._native_test(hold, conditions), False)
            walk.codes = codes
            if isinstance(target.value.value, cst.Name):
                swaps[target.value.value.value] = call
                self.replaced_names.add(target.value.value.value)
        walk.swaps = swaps
        return walk

    def _column(self, expr, swaps):
        """The rows of a float64 or int64 column C of a view or prefix: ``V["c"]``, ``R.loc[prefix]["c"]``,
        ``R.loc[prefix, "c"]``, or a local assigned once to one of them; the walk of those rows with the column's
        hold (``column``), else None."""
        o = self.o
        if isinstance(expr, cst.Name):
            value = o._value_of(expr.value) if o._once(expr.value) else None
            if value is None or any(nested for _, nested in o.facts.formula.loads.get(expr.value, ())):
                return None
            found = self._column(value, swaps)
            if found is not None:
                swaps[expr.value] = found.swapped(value)
                self.swaps.setdefault(expr.value, swaps[expr.value])      # dropped once unread
            return found
        parts = _index_parts(expr) if isinstance(expr, cst.Subscript) else None
        if not parts or _literal_str(parts[-1]) is None:
            return None
        label = _literal_str(parts[-1])
        if len(parts) == 1:
            source = expr.value
        elif isinstance(expr.value, cst.Attribute) and expr.value.attr.value == 'loc':
            source = expr.with_changes(slice=[e.with_changes(comma=cst.MaybeSentinel.DEFAULT) for e in expr.slice[:-1]])
        else:
            return None
        walk = self._rows(source, swaps)
        if walk is None or walk.whole or walk.schema['kind'] != 'frame':
            return None
        kind = dict(walk.schema['columns']).get(label)
        if kind not in READ or len(walk.schema['levels']) - walk.depth != 1:
            return None
        chold = self._take(walk.call, label, COLUMN_KIND[kind])
        walk.column = (chold, kind)
        walk.test = self._conjunction([walk.test, self._positive(o.slot(chold))])
        return walk

    def _labels(self, walk, target, position):
        """{name: its expression at the walked position} for a loop target over ``walk``: one name (the label, or
        the tuple of the labels of several remaining levels), or a tuple of distinct names, one per remaining
        level; None for any other target."""
        levels = walk.schema['levels']
        row = self._m('whole_row', self.o.ref(walk.hold), position) if walk.whole else position

        def label(level):
            return self._m('label_ll' if levels[level] == 'int' else 'label_obj', self.o.ref(walk.hold),
                           row.deep_clone(), cst.Integer(str(level)))
        remaining = range(walk.depth, len(levels))
        if walk.mode != 'index':        # a column: its value, or (label, value)
            chold, kind = walk.column
            value = self._m(kind + '_row', self.o.ref(chold), self.o.slot(chold), self._next(row.deep_clone()))
            if walk.mode == 'values':
                return {target.value: value} if isinstance(target, cst.Name) else None
            if not (isinstance(target, cst.Tuple) and len(target.elements) == 2 and all(
                    isinstance(e, cst.Element) and isinstance(e.value, cst.Name) for e in target.elements)):
                return None
            key, name = (e.value.value for e in target.elements)
            return {key: label(walk.depth), name: value} if key != name else None
        if isinstance(target, cst.Name):
            if len(remaining) == 1:
                return {target.value: label(walk.depth)}
            return {target.value: cst.Tuple([cst.Element(label(level)) for level in remaining])}
        if not (isinstance(target, cst.Tuple) and len(target.elements) == len(remaining)
                and all(isinstance(e, cst.Element) and isinstance(e.value, cst.Name) for e in target.elements)):
            return None
        names = [e.value.value for e in target.elements]
        if len(set(names)) != len(names):
            return None
        return {name: label(level) for name, level in zip(names, remaining)}

    def _walk_fallback(self, updated, walk):
        """The pandas walk: the view, index and frame locals of the walk replaced by what they stand for (so that
        none of them needs evaluating on the native path); every other name as written."""
        swaps = walk.swaps

        class Swap(cst.CSTTransformer):
            def leave_Name(self, original, new):
                return swaps[original.value] if original.value in swaps else new
        return updated.visit(Swap()) if swaps else updated

    def _position(self):
        self.o.key_count += 1
        return cst.Name(f'_mx_p{self.o.key_count}')

    def comprehension(self, original, updated):
        """A comprehension whose first ``for`` walks an index (``_walk``): positions in pandas order, the target names
        replaced by their labels. A generator is walked natively only as the argument of a builtin that consumes it
        at once (its table is read through a borrowed reference)."""
        o = self.o
        comp = original.for_in
        if comp.asynchronous is not None or (isinstance(original, cst.GeneratorExp) and not self._consumed(original)):
            return None
        walk = self._walk(comp.iter)
        if walk is None:
            return None
        position = self._position()
        labels = self._labels(walk, comp.target, position)
        if labels is None or _rebinds(original, set(labels)):
            return None
        sub = _Substitute(labels)
        for_in = updated.for_in
        native = updated.with_changes(for_in=for_in.with_changes(
            target=position, iter=cst.Call(func=cst.Name('range'), args=[cst.Arg(walk.start), cst.Arg(walk.stop)]),
            ifs=[i.visit(sub) for i in for_in.ifs],
            inner_for_in=for_in.inner_for_in.visit(sub) if for_in.inner_for_in is not None else None))
        if isinstance(native, cst.DictComp):
            native = native.with_changes(key=native.key.visit(sub), value=native.value.visit(sub))
        else:
            native = native.with_changes(elt=native.elt.visit(sub))
        fallback = self._walk_fallback(updated, walk)
        if isinstance(original, cst.GeneratorExp):
            parens = {'lpar': [cst.LeftParen()], 'rpar': [cst.RightParen()]}
            native, fallback = native.with_changes(**parens), fallback.with_changes(**parens)
        return self._lowered(walk.test, native, fallback)

    def _consumed(self, gen):
        """A generator that is the one argument of a builtin consuming it at once (``sorted``, ``max``, ...)."""
        o = self.o
        arg = o.facts.formula.parents.get(id(gen))
        call = o.facts.formula.parents.get(id(arg)) if isinstance(arg, cst.Arg) else None
        if isinstance(call, cst.Call) and o.builtin_minmax and o._walk_extreme_int(call):
            return False        # ``max(a for a, s in R.index)``: read as the level's extreme, no walk
        return (isinstance(call, cst.Call) and isinstance(call.func, cst.Name) and call.func.value in WALK_BUILTINS
                and len(call.args) == 1 and call.args[0] is arg and arg.keyword is None and not arg.star
                and call.func.value not in o.facts.formula.binds and call.func.value not in o.facts.formula.other)

    def leave_ListComp(self, original, updated):
        return self.comprehension(original, updated) or updated

    def leave_SetComp(self, original, updated):
        return self.comprehension(original, updated) or updated

    def leave_DictComp(self, original, updated):
        return self.comprehension(original, updated) or updated

    def leave_GeneratorExp(self, original, updated):
        return self.comprehension(original, updated) or updated

    def iterrows_loop(self, original, updated):
        """``for k, row in R.iterrows():``: positions in pandas order, ``k`` the label and ``row`` the stored row
        (``_mx_rw_row``), its fields read as a row local's; the pandas loop (``_mx_rw_row`` -1) when the table is
        not of the traced kinds (every column checked)."""
        o = self.o
        target = original.target
        if not (isinstance(target, cst.Tuple) and len(target.elements) == 2
                and all(isinstance(e, cst.Element) and isinstance(e.value, cst.Name) for e in target.elements)):
            return None
        key, name = (e.value.value for e in target.elements)
        found = o.row_locals.get(name)
        if found is None or found[2] is not original:
            return None
        call, schema, _ = found
        hold = self._take(call, _NO_LABEL, -4)
        hold.row_columns = True
        var = f'_mx_rw_{name}'
        position = self._position()
        o.declarations += [var, position.value]
        walk = _Walk(call, schema, 0, hold, cst.Integer('0'), self._m('nrows', o.ref(hold)), None, True)
        row = self._m('whole_row', o.ref(hold), position)
        first = [self._assign(var, row), *[self._assign(n, e) for n, e in self._labels(
            walk, cst.Name(key), position).items()]]
        body = updated.body
        body = (body.with_changes(body=[*first, *body.body]) if isinstance(body, cst.IndentedBlock)
                else cst.IndentedBlock(body=[*first, cst.SimpleStatementLine(body=body.body)]))
        native = updated.with_changes(body=body, target=position, iter=cst.Call(
            func=cst.Name('range'), args=[cst.Arg(walk.stop)]))
        if isinstance(original.iter.func.value, cst.Name):
            walk.swaps[original.iter.func.value.value] = call
            self.replaced_names.add(original.iter.func.value.value)
        unbound = cst.parse_statement(f'{name} = None')    # read only by the pandas branches of its fields
        return self._branch(self._native_test(hold, []), [unbound, native],
                            [cst.parse_statement(f'{var} = -1'), self._walk_fallback(updated, walk)])

    def leave_For(self, original, updated):
        """``for k in X.index:`` (``_walk``): a loop over positions in pandas order, the target names bound to the
        labels of each position first in the body; the pandas loop when the table cannot answer."""
        o = self.o
        if original.asynchronous is not None:
            return updated
        rows = self.iterrows_loop(original, updated)
        if rows is not None:
            return rows
        walk = self._walk(original.iter)
        if walk is None:
            return updated
        position = self._position()
        labels = self._labels(walk, original.target, position)
        if labels is None:
            return updated
        o.declarations.append(position.value)
        first = [self._assign(name, expr) for name, expr in labels.items()]
        body = updated.body
        body = (body.with_changes(body=[*first, *body.body]) if isinstance(body, cst.IndentedBlock)
                else cst.IndentedBlock(body=[*first, cst.SimpleStatementLine(body=body.body)]))
        native = updated.with_changes(target=position, body=body, iter=cst.Call(
            func=cst.Name('range'), args=[cst.Arg(walk.start), cst.Arg(walk.stop)]))
        return self._branch(walk.test, [native], [self._walk_fallback(updated, walk)])

    def index_builtin(self, original, updated):
        """``list(X.index)``, ``tuple(X.index)``, ``set(X.index)``: the labels in pandas order; ``list(C)`` and
        ``C.tolist()`` of a column: its values (the Python numbers both give)."""
        o = self.o
        func = original.func
        if isinstance(func, cst.Attribute) and func.attr.value == 'tolist' and not original.args:
            walk = self._walk(func.value)
            if walk is None or walk.mode != 'values':
                return None
            func = cst.Name('list')
        elif (isinstance(func, cst.Name) and func.value in ('list', 'tuple', 'set') and len(original.args) == 1
                and original.args[0].keyword is None and not original.args[0].star
                and func.value not in o.facts.formula.binds and func.value not in o.facts.formula.other):
            walk = self._walk(original.args[0].value)
            if walk is None or walk.mode == 'items' or (walk.mode == 'values' and func.value != 'list'):
                return None
        else:
            return None
        position = self._position()
        name = f'_mx_k{o.key_count}'
        labels = self._labels(walk, cst.Name(name), position)
        for_in = cst.CompFor(target=position, iter=cst.Call(
            func=cst.Name('range'), args=[cst.Arg(walk.start), cst.Arg(walk.stop)]))
        native = (cst.SetComp if func.value == 'set' else cst.ListComp)(elt=labels[name], for_in=for_in)
        if func.value == 'tuple':
            native = cst.Call(func=cst.Name('tuple'), args=[cst.Arg(native)])
        return self._lowered(walk.test, native, self._walk_fallback(updated, walk))

    def column_item(self, original, updated, parts):
        """``C.iloc[i]`` (i an integer, from the end when negative) and ``C.loc[k]`` of a column of a view or prefix
        (``_column``): the value at that position of the prefix's rows, or at label k of its remaining level."""
        o = self.o
        value = original.value
        if not (isinstance(value, cst.Attribute) and value.attr.value in ('iloc', 'loc') and len(parts) == 1
                and not _several(parts[0])):
            return None
        part = parts[0]
        if value.attr.value == 'iloc' and (not o.facts.is_int(part) or isinstance(part, cst.Name)):
            return None         # a position known from the text to be an integer (a name may be a bool)
        swaps = {}
        walk = self._column(value.value, swaps)
        if walk is None:
            return None
        walk.swaps = swaps
        chold, kind = walk.column
        emit = self.updated_of.get(id(part), part)
        if value.attr.value == 'iloc':
            row = self._m('position', walk.start, walk.stop, _cast('_mx_cy.Py_ssize_t', emit))
            read, test = self._m(kind + '_row', o.ref(chold), o.slot(chold), self._next(row)), walk.test
        else:
            levels = walk.schema['levels']
            conditions = [walk.test]
            own = (self._held(chold, walk.depth, part, conditions) if o.item_fixed(part, levels[walk.depth])
                   else self.code_of(chold, walk.depth, part, levels[walk.depth], conditions))
            suffix, args = self.code_args([*walk.codes, own])
            read, test = self._m(kind + suffix, o.ref(chold), o.slot(chold), *args), self._conjunction(conditions)
        return self._lowered(test, read, self._walk_fallback(updated, walk), kind)

    def row_local_read(self, original, updated, parts):
        """``row["c"]`` of a row local: the field from the prepared blocks, of the kind pandas gives it."""
        o = self.o
        value = original.value
        found = o.row_locals.get(value.value) if isinstance(value, cst.Name) else None
        if found is None or len(parts) != 1:
            return None
        call, schema, _ = found
        valid = cst.parse_expression(f'_mx_rw_{value.value} >= 0')
        label = _literal_str(parts[0])
        if label is None:           # float(row[name]): the field chosen at run time
            hold = self._take(call, _NO_LABEL, -4)
            hold.row_columns = True
            read = self._m('field_f64', o.ref(hold), cst.parse_expression(f'_mx_rw_{value.value} + 1'),
                           _cast('object', self.updated_of.get(id(parts[0]), parts[0])))
            return self._lowered(valid, read, updated, 'f64')
        read, kind, _ = self._field_read(call, schema, label, lambda hold: (
            cst.parse_expression(f'_mx_rw_{value.value} + 1'), []))
        lowered = self._lowered(valid, read, updated, kind)
        if isinstance(found[2], cst.For) and any(k == 'obj' for _, k in schema['columns']):
            self.unboxed.add(id(lowered))   # iterrows builds such a row from Python scalars, not numpy's
        return lowered

    def _field_read(self, call, schema, label, position):
        """Field ``label`` of a stored row, of the kind pandas gives it: the column's own on a row with an object
        column, else the row's common kind (an int column of a float row reads as a float). ``position(hold)``:
        (the row + 1, its conditions). (read, kind, test)."""
        o = self.o
        kinds = dict(schema['columns'])
        mixed = any(k == 'obj' for k in kinds.values())
        kind = kinds[label] if mixed else ('f64' if any(k == 'f64' for k in kinds.values()) else 'i64')
        stored = kinds[label]           # the block the column is stored in
        hold = self._take(call, label, {'f64': 0, 'i64': 1, 'obj': 2}[stored])
        hold.row_columns = True         # the row's dtype depends on every column: all of them are checked
        row, conditions = position(hold)
        read = self._m({'f64': 'f64_row', 'i64': 'i64_row', 'obj': 'object_field'}[stored], o.ref(hold), o.slot(hold),
                       row)
        if kind == 'f64' and stored == 'i64':
            read = _cast('_mx_cy.double', read)
        return read, (kind if kind in READ else None), self._native_test(hold, conditions)

    def row_cell_field(self, original, updated, parts):
        """``self.P()["f"]`` with P a row cell (``return T.loc[key]``, the key complete and fixed per item): the row
        held per item, as ``T.loc[key]["f"]``; a computed field under ``float()`` reads the stored value as a float.
        Compiled zero-argument Cells cannot be assigned, so P's value is always its formula's."""
        o = self.o
        call = original.value
        local = call.value if isinstance(call, cst.Name) else None
        if local is not None:                   # ``row = self.P()`` then ``row["f"]``
            call = o._value_of(local) or call
        if not (o.per_item and isinstance(call, cst.Call) and not call.args and isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name) and call.func.value.value == 'self'):
            return None
        row = o.row_cells.get(call.func.attr.value)
        if row is None or not o.cached_call(call) or len(parts) != 1:
            return None
        source = o._receiver_of(row.value.value)
        if source is None or not o.shared(source[0]):
            return None     # a row of a frame the item computes: its fields name a few values per item (pandas)
        if _literal_str(parts[0]) is not None:
            if local is not None:       # the local is P() for the pandas branch, and is dropped once unread
                self.swaps.setdefault(local, call)
            return self._row_field(original.with_changes(value=row), updated, parts, held=True)
        up = o.facts.formula.parents.get(id(original))
        consumer = o.facts.formula.parents.get(id(up)) if isinstance(up, cst.Arg) else None
        if not (isinstance(consumer, cst.Call) and isinstance(consumer.func, cst.Name) and consumer.func.value == 'float'
                and len(consumer.args) == 1 and o.builtin_float):
            return None
        found = o._receiver(row.value.value)
        key = _index_parts(row)[0]
        if found is None or found[1]['kind'] != 'frame' or not self._plain(*found):
            return None
        levels = found[1]['levels']
        fixed = _parts(key)
        if len(fixed) != len(levels) or (len(levels) == 1 and isinstance(key, cst.Tuple)) or not all(
                o.fixed_part(part) for part in fixed):
            return None
        hold = self._take(found[0], _NO_LABEL, -4)
        hold.row_columns = True
        field = o.row_field(hold, key, self._detached(key))
        o.key_count += 1
        local_name = f'_mx_name_{o.key_count}'
        label = _cast('object', self.updated_of.get(id(parts[0]), parts[0]))
        test = self._native_test(hold, [cst.parse_expression(f'self.{field} > 0'), cst.parse_expression(
            f'{o.module}.is_label(({local_name} := {_code(label)}))')])
        read = self._m('field_f64', o.ref(hold), cst.parse_expression(f'self.{field}'), cst.Name(local_name))
        return self._lowered(test, read, updated, 'f64')

    def _row_field(self, original, updated, parts, held=False):
        """``R.loc[key]["f"]``, a field of a row of a frame: a key fixed per item has its row held per item (also the
        key of a row cell, ``held``), any other key finds its row per read (KeyError as pandas when it has none); the
        field is read from the prepared table, of the kind pandas gives it. None when not this form."""
        o = self.o
        row = original.value
        if not (len(parts) == 1 and isinstance(row, cst.Subscript)
                and isinstance(row.value, cst.Attribute) and row.value.attr.value == 'loc'):
            return None
        label = _literal_str(parts[0])
        key_parts = _index_parts(row)
        found = o._receiver(row.value.value)
        if label is None or key_parts is None or len(key_parts) != 1 or found is None:
            return None
        call, schema = found
        if schema['kind'] != 'frame' or not self._plain(call, schema) or label not in dict(schema['columns']):
            return None
        key = key_parts[0]
        levels = schema['levels']
        if (isinstance(key, cst.Tuple) and len(key.elements) != len(levels)) or (
                not isinstance(key, cst.Tuple) and len(levels) != 1):
            return None
        per_read = not (o.per_item and all(o.fixed_part(part) for part in _parts(key)))
        if per_read and held:
            return None
        if isinstance(row.value.value, cst.Name):
            self.replaced_names.add(row.value.value.value)

        def position(hold):
            if per_read:
                (suffix, cargs), labels = self._read_args(hold, _parts(key), levels)
                return self._next(self._m('row' + suffix, o.ref(hold), *cargs)), labels
            field = o.row_field(hold, key, self._detached(key))
            return cst.parse_expression(f'self.{field}'), [cst.parse_expression(f'self.{field} > 0')]
        read, kind, test = self._field_read(call, schema, label, position)
        return self._lowered(test, read, self._fallback(updated, call), kind)

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

    def named_read(self, original, updated, call, schema, key, name, through=None):
        """``float(R.loc[key, name])`` with the column chosen at run time: the stored value of the column ``name``
        names, as a float (KeyError as pandas for a name that is not a column). Native only while the table has every
        column of the traced kinds (the row is read whole: kind -4). ``through``: (view, the key's own parts) for a
        read of a view local."""
        o = self.o
        up = o.facts.formula.parents.get(id(original))
        consumer = o.facts.formula.parents.get(id(up)) if isinstance(up, cst.Arg) else None
        if not (schema['kind'] == 'frame' and isinstance(consumer, cst.Call) and isinstance(consumer.func, cst.Name)
                and consumer.func.value == 'float' and len(consumer.args) == 1 and o.builtin_float):
            return None
        key_parts = self._key(key, schema) if through is None else through[1]
        if key_parts is None:
            return None
        hold = self._take(call, _NO_LABEL, -4)
        hold.row_columns = True
        if through is None:
            (suffix, args), conditions = self._read_args(hold, key_parts, schema['levels'])
        else:
            conditions = []
            suffix, args = self._view_codes(through[0], hold, key_parts, conditions)
        o.key_count += 1
        local = f'_mx_name_{o.key_count}'        # one label at run time, else pandas (a list selects columns)
        label = _cast('object', self.updated_of.get(id(name), name))
        conditions.append(cst.parse_expression(f'{o.module}.is_label(({local} := {_code(label)}))'))
        row = self._next(self._m('row' + suffix, o.ref(hold), *args))
        read = self._m('field_f64', o.ref(hold), row, cst.Name(local))
        return self._lowered(self._native_test(hold, conditions), read, self._fallback(updated, call), 'f64')

    # ------------------------------------------------------------------ index extremes and membership
    def _extreme(self, original, updated):
        """The index extremes of the module docstring, else None. An extreme read as a C integer is lowered only
        where its value meets no use in which numpy's integer and a C integer differ (``numpy_use``)."""
        o = self.o
        func = original.func
        found = o.extreme(original)
        if found is not None:               # R.index.max() of one int level: the C integer
            call, which = found
            if o.numpy_use(original, 'i64'):
                return None
            hold = self._take(call, _NO_LABEL, -2)
            if isinstance(original.func.value.value, cst.Name):
                self.replaced_names.add(original.func.value.value.value)
            return self._lowered(self._native_test(hold, []), self._m(which + '_ll', o.ref(hold)),
                                 self._fallback(updated, call), 'i64')
        if (isinstance(func, cst.Name) and func.value in ('max', 'min') and len(original.args) == 1
                and not original.args[0].keyword and not original.args[0].star):
            arg = original.args[0].value        # max(R.index) of one int level: the same C integer
            index = arg.value if isinstance(arg, cst.Attribute) and arg.attr.value == 'index' else None
            view = self._view_of(index)
            if view is not None:                # max(V.index) of a view: its label as the Python int iteration gives
                return self.view_extreme(original, updated, view, func.value) if o.builtin_minmax else None
            got = o._receiver(index) if index is not None else None
            if (got is not None and got[1]['levels'] == ('int',) and self._plain(*got)
                    and o.builtin_minmax and not o.numpy_use(original, 'i64')):
                hold = self._take(got[0], _NO_LABEL, -2)
                if isinstance(index, cst.Name):
                    self.replaced_names.add(index.value)
                return self._lowered(self._native_test(hold, []), self._m(func.value + '_ll', o.ref(hold)),
                                     self._fallback(updated, got[0]), 'i64')
            return self.level_walk_extreme(original, updated)
        if not (isinstance(func, cst.Attribute) and func.attr.value in ('max', 'min') and not original.args):
            return None
        level = o.level_values(original)
        if level is not None:
            return self.level_extreme(original, updated, level)
        index = func.value
        if not (isinstance(index, cst.Attribute) and index.attr.value == 'index'):
            return None
        view = self._view_of(index.value)
        if view is not None:                # V.index.max() of a view local
            return None if o.numpy_use(original, 'i64') else self.view_extreme(original, updated, view, func.attr.value)
        whole = o._receiver(index.value)
        if whole is not None:               # any other index: pandas' own extreme label, prepared once
            if not self._plain(*whole):
                return None
            hold = self._take(whole[0], _NO_LABEL, -2)
            if isinstance(index.value, cst.Name):
                self.replaced_names.add(index.value.value)
            return self._lowered(self._native_test(hold, []), self._m(func.attr.value + '_obj', o.ref(hold)),
                                 self._fallback(updated, whole[0]))
        prefix = o.prefix_of(index.value)
        if prefix is None or not o.int_label(original) or o.numpy_use(original, 'i64'):
            return None
        # R.loc[prefix].index.max(): the label of the remaining int level at the first or last stored row of the
        # prefix (rows are stored in code order, and the codes of an int level are in label order)
        (call, schema), parts = prefix
        hold = self._take(call, _NO_LABEL, -1)
        if isinstance(index.value.value.value, cst.Name):
            self.replaced_names.add(index.value.value.value.value)
        conditions = []
        codes = [self._held(hold, level, part, conditions) if o.item_fixed(part, schema['levels'][level])
                 else self.code_of(hold, level, part, schema['levels'][level], conditions)
                 for level, part in enumerate(parts)]
        args = [o.ref(hold), cst.Integer(str(len(parts))), *codes, *[cst.Integer('0')] * (5 - len(codes))]
        o.key_count += 1
        start = f'_mx_vs_{o.key_count}'
        test = self._conjunction([self._native_test(hold, conditions), cst.parse_expression(
            f'({start} := {_code(self._m("view_" + ("start" if func.attr.value == "min" else "stop"), *args))}) > 0'
            if func.attr.value == 'max' else
            f'({start} := {_code(self._m("view_start", *args))}) >= 0')])
        row = cst.parse_expression(f'{start} - 1' if func.attr.value == 'max' else start)
        read = self._m('label_ll', o.ref(hold), row, cst.Integer(str(len(parts))))
        return self._lowered(test, read, self._fallback(updated, call), 'i64')

    def _asof_parts(self, call):
        """(receiver call, schema, prefix parts) of ``R.index.asof(x)`` / ``R.loc[prefix].index.asof(x)`` whose one
        remaining level is an int level, x one positional argument that is not a text or a display, else None."""
        o = self.o
        func = call.func
        if not (isinstance(func, cst.Attribute) and func.attr.value == 'asof' and isinstance(func.value, cst.Attribute)
                and func.value.attr.value == 'index' and len(call.args) == 1 and not call.args[0].keyword
                and not call.args[0].star and not isinstance(call.args[0].value, (
                    cst.SimpleString, cst.ConcatenatedString, cst.FormattedString, *SEVERAL_ROWS))):
            return None
        receiver = func.value.value
        whole = o._receiver(receiver)
        if whole is not None:
            found, parts = (whole if self._plain(*whole) else None), []
        else:
            prefix = o.prefix_of(receiver)
            found, parts = prefix if prefix is not None else (None, None)
        if found is None:
            return None
        levels = found[1]['levels']
        if len(levels) - len(parts) != 1 or levels[len(parts)] != 'int':
            return None
        return found[0], found[1], parts

    def _asof(self, original, updated):
        """``int(X.index.asof(x))``: the label as a C integer; any other ``X.index.asof(x)``: the numpy integer pandas
        returns. Native while the table's stored order within the prefix is pandas' (kind -3), the prefix has rows, x
        is one number and not NaN (tested at run time), and a label <= x exists; else pandas' own asof, which answers
        or raises as live does (NaN for no label, by pandas' version, is pandas')."""
        o = self.o
        func = original.func
        whole_int = (isinstance(func, cst.Name) and func.value == 'int' and o.facts.int_argument(original) is not None
                     and isinstance(original.args[0].value, cst.Call))
        call = original.args[0].value if whole_int else original
        found = self._asof_parts(call)
        if found is None:
            return None
        if not whole_int:
            up = o.facts.formula.parents.get(id(original))
            outer = o.facts.formula.parents.get(id(up)) if isinstance(up, cst.Arg) else None
            if isinstance(outer, cst.Call) and o.facts.int_argument(outer) is original:
                return None     # int(...) of it: lowered with the call around it
        receiver, schema, parts = found
        depth = len(parts)
        hold = self._take(receiver, depth if parts else None, -3)
        root = call.func.value.value
        root = root.value.value if parts else root
        if isinstance(root, cst.Name):
            self.replaced_names.add(root.value)
        conditions = []
        if parts:
            codes = [self._held(hold, level, part, conditions) if o.item_fixed(part, schema['levels'][level])
                     else self.code_of(hold, level, part, schema['levels'][level], conditions)
                     for level, part in enumerate(parts)]
            args = [o.ref(hold), cst.Integer(str(depth)), *codes, *[cst.Integer('0')] * (5 - len(codes))]
        o.key_count += 1
        n = o.key_count
        x, start, found_at = f'_mx_x_{n}', f'_mx_vs_{n}', f'_mx_at_{n}'
        emit = self.updated_of.get(id(call.args[0].value), call.args[0].value)
        tests = [self._native_test(hold, conditions),
                 cst.parse_expression(f'{o.module}.is_number(({x} := {_code(emit)}))')]
        if parts:
            tests.append(cst.parse_expression(f'({start} := {_code(self._m("view_start", *args))}) >= 0'))
            stop = self._m('view_stop', *args)
        else:
            start, stop = '0', self._m('nrows', o.ref(hold))
        tests.append(cst.parse_expression(
            f'({found_at} := {o.module}.asof_row({_code(o.ref(hold))}, {start}, {_code(stop)}, {depth}, '
            f'_mx_cy.cast(_mx_cy.double, {x}))) >= 0'))
        label = self._m('label_ll', o.ref(hold), cst.Name(found_at), cst.Integer(str(depth)))
        if whole_int:
            return self._lowered(self._conjunction(tests), label, self._fallback(updated, receiver), 'i64')
        boxed = self._m('i64_box', o.ref(hold), label, cst.Name('False'))
        return self._lowered(self._conjunction(tests), boxed, self._fallback(updated, receiver))

    def level_extreme(self, original, updated, found):
        """``R.index.get_level_values(L).max()`` / ``.min()`` of an int level: its largest / smallest label, a C
        integer; native while the level still has that name, is an int level and the table is not empty (kind -6)."""
        o = self.o
        call, schema, index, name, which, local = found
        if o.numpy_use(original, 'i64'):
            return None
        hold = self._take(call, (name, index), -6)
        if local is not None:           # the pandas branch reads the level values; the local goes once unread
            self.swaps.setdefault(local, o._value_of(local))
        inner = o._value_of(local) if local is not None else original.func.value
        if isinstance(inner.func.value.value, cst.Name):
            self.replaced_names.add(inner.func.value.value.value)
        native = self._m(f'level_{which}_ll', o.ref(hold), cst.Integer(str(index)))
        return self._lowered(self._native_test(hold, []), native, self._fallback(updated, call), 'i64')

    def level_walk_extreme(self, original, updated):
        """``max(a for a, s in R.index)`` / ``min(...)``: the element is the part of the tuple target naming an int
        level, with no condition, over a whole frame of unique keys: that level's extreme as a C integer, as
        ``R.index.get_level_values(i).max()`` (kind -6), not a walk of every row per evaluation."""
        o = self.o
        shape = _walk_extreme(original)
        if shape is None or not o.builtin_minmax:
            return None
        it, names, level = shape
        found = o._receiver(it.value) if isinstance(it, cst.Attribute) and it.attr.value == 'index' else None
        if found is None or not self._plain(*found) or o.numpy_use(original, 'i64'):
            return None
        levels = found[1]['levels']
        if len(levels) != len(names) or levels[level] != 'int':
            return None
        hold = self._take(found[0], (None, level), -6)
        if isinstance(it.value, cst.Name):
            self.replaced_names.add(it.value.value)
        native = self._m(f'level_{original.func.value}_ll', o.ref(hold), cst.Integer(str(level)))
        return self._lowered(self._native_test(hold, []), native, self._fallback(updated, found[0]), 'i64')

    def leave_Comparison(self, original, updated):
        """``key in R.index`` / ``not in`` with a complete key: the native test while every key part is one label
        (a part fixed per item and absent from its level gives False without a lookup); pandas' own membership
        for a part that is not one label at run time, so its answer or its error is live's."""
        o = self.o
        if len(original.comparisons) != 1:
            return updated
        target = original.comparisons[0]
        index = target.comparator
        if not (isinstance(target.operator, (cst.In, cst.NotIn)) and isinstance(index, cst.Attribute)
                and index.attr.value == 'index'):
            return updated
        found = o._receiver(index.value)
        if found is None or not self._plain(*found):
            return updated
        call, schema = found
        key_parts = self._key(original.left, schema)
        if key_parts is None:
            return updated
        hold = self._take(call, _NO_LABEL, -1)
        if isinstance(index.value, cst.Name):
            self.replaced_names.add(index.value.value)
        (suffix, args), conditions = self._read_args(hold, key_parts, schema['levels'])
        # a held code is -1 for an absent label (the lookup of code -2 is False), -2 for a part that is not one
        # label and 0 while unresolved (its evaluation raised): those two run pandas
        conditions = [cst.parse_expression(f'({_code(c.left)} > 0 or {_code(c.left)} == -1)')
                      if isinstance(c, cst.Comparison) else c for c in conditions]
        test = self._m('contains' + suffix, o.ref(hold), *args)
        if isinstance(target.operator, cst.NotIn):
            test = cst.UnaryOperation(cst.Not(), test)
        return self._lowered(self._native_test(hold, conditions), test, self._fallback(updated, call))

    def leave_Call(self, original, updated):
        """``float(...)`` around a float64 read and ``int(...)`` around an int64 read are dropped: the read is
        already that C value, and the builtin would box it into a Python object and back."""
        if (isinstance(updated.func, cst.Name) and updated.func.value in ('isinstance', 'type') and updated.args
                and isinstance(updated.args[0].value, cst.IfExp) and self.native.get(id(updated.args[0].value)) in CTYPE
                and id(updated.args[0].value) not in self.unboxed):
            # the type of a read is the numpy scalar's pandas returns, not the C number's
            return updated.with_changes(args=[updated.args[0].with_changes(
                value=self._pandas_value(updated.args[0].value)), *updated.args[1:]])
        inner = self.o.facts.int_argument(original)
        lowered = (self._asof(original, updated) or self._extreme(original, updated)
                   or self.index_builtin(original, updated))
        if lowered is not None:
            return lowered
        if (isinstance(updated.func, cst.Name) and len(updated.args) == 1 and not updated.args[0].keyword
                and not updated.args[0].star):
            kind = self.native.get(id(updated.args[0].value))
            if (kind == 'f64' and updated.func.value == 'float') or (kind == 'i64' and inner is not None):
                return updated.args[0].value.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        return updated
