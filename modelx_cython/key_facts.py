"""Key-typing facts of a formula (T-6 step 0): what the table-read lowering knows of a key part's type.

``is_int`` (why an expression is a non-bool integer), ``exact_long_long`` (statically a C ``long long``),
and ``enum_constant`` (the module constant of an admitted IntEnum member) use the bindings of one formula.
"""

import libcst as cst

from .usage import RESOLVED, _extract_self_chain
from .syntax import dotted_name

INT_TYPES = frozenset(('long long', 'long', 'int', 'short', 'Py_ssize_t', 'size_t'))
SKIPPED_SCOPES = (cst.FunctionDef, cst.Lambda, cst.ClassDef)
COMPREHENSIONS = (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)


def _code(node):
    return cst.Module(body=()).code_for_node(node)


def _plain_args(call):
    return not any(arg.keyword or arg.star for arg in call.args)


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
        # diag-jp: element values of `a, b = x, y` (literal tuples of equal length, plain names,
        # formula level); the names stay in `other`, these values serve `is_int` only
        self.unpacked = {}
        self.binds = {}             # name -> number of bindings outside comprehensions
        # diag-kr: `for k in range(...)` at formula level (the range call per binding of k);
        # the names stay in `other`, these serve `is_int` only
        self.range_loops = {}
        # diag-kr: names declared ``name: _mx_cy.longlong`` (no value) by an earlier pass, at
        # formula level: Cython holds them as C long long whatever is assigned
        self.c_long_long = set()
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
        target, value = node.targets[0].target, node.value
        if (len(node.targets) == 1 and self._depth == 1 and isinstance(target, (cst.Tuple, cst.List))
                and isinstance(value, (cst.Tuple, cst.List)) and len(target.elements) == len(value.elements)
                and all(isinstance(e, cst.Element) and isinstance(e.value, cst.Name) for e in target.elements)
                and all(isinstance(e, cst.Element) for e in value.elements)):
            for name, element in zip(target.elements, value.elements):
                if not any(name.value.value in names for names in self._shadow):
                    self.unpacked.setdefault(name.value.value, []).append(element.value)

    def visit_AnnAssign(self, node):
        self._store(node.target, node.value, plain=node.value is not None and self._depth == 1)
        if (node.value is None and self._depth == 1 and isinstance(node.target, cst.Name)
                and dotted_name(node.annotation.annotation) == '_mx_cy.longlong'):
            self.c_long_long.add(node.target.value)

    def visit_AugAssign(self, node):
        self._store(node.target)

    def visit_For(self, node):
        self._store(node.target)
        if (self._depth == 1 and isinstance(node.target, cst.Name)
                and not any(node.target.value in names for names in self._shadow)
                and isinstance(node.iter, cst.Call) and isinstance(node.iter.func, cst.Name)
                and node.iter.func.value == 'range' and 1 <= len(node.iter.args) <= 3
                and _plain_args(node.iter)):
            self.range_loops.setdefault(node.target.value, []).append(node.iter)

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


    def unshadowed_load(self, node):
        """Whether this read is in the formula's own scope, with no comprehension binding."""
        return any(load is node and not nested for load, nested in self.loads.get(node.value, ()))


class KeyFacts:
    """The facts of one formula: set ``formula`` (a visited ``_Formula``) and ``_params`` before use."""

    def __init__(self, resolver, cls_info, cells, policy, *, builtin_int=True):
        self.resolver, self.cls_info, self.cells, self.policy = resolver, cls_info, cells, policy
        self.builtin_int = builtin_int
        self._key_locals = set()    # loop variables of lowered `for k in T.index:` (C long long)
        self._params = set()

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

    def enum_constant(self, expr):
        """D79: the module constant of ``self.<Enum>.<member>`` (or its definition field
        ``self._mx_definition_<Enum>.<member>``) when the enum Reference is admitted at build
        (``enums.prepare``: an IntEnum definition, enum_safe, value within C int), else None."""
        if not (isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Attribute)
                and isinstance(expr.value.value, cst.Name) and expr.value.value.value == 'self'):
            return None
        ref = expr.value.attr.value
        if ref.startswith('_mx_definition_'):
            ref = ref[len('_mx_definition_'):]
        module = getattr(self.cls_info, 'module', None)
        entry = (getattr(module, 'code_enum_symbols', None) or {}).get(
            (getattr(self.cls_info, 'name', None), ref, expr.attr.value))
        return entry[0] if entry else None

    def int_argument(self, expr):
        """Argument of a one-argument builtin ``int`` call in this formula, or None."""
        if (self.builtin_int and isinstance(expr, cst.Call) and isinstance(expr.func, cst.Name)
                and expr.func.value == 'int' and len(expr.args) == 1 and _plain_args(expr)
                and 'int' not in self._params and 'int' not in self.formula.binds
                and self.formula.unshadowed_load(expr.func)):
            return expr.args[0].value
        return None

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
            # diag-kr: a `for k in range(a, b, c)` target with integer bounds (and any plain
            # integer assignments) is an integer (`for k in range(1, 7): T.loc[k, "c"]`)
            if expr.value in self.formula.c_long_long and expr.value not in self._params:
                return f'local {expr.value} is declared long long'
            loops = self.formula.range_loops.get(expr.value, ())
            if (loops and expr.value not in self._params and 'range' not in self.formula.binds
                    and self.formula.binds.get(expr.value, 0) == len(loops) + len(values or ())
                    and not self.formula.unpacked.get(expr.value)
                    and all(self.is_int(arg.value, depth + 1) for call in loops for arg in call.args)
                    and all(self.is_int(value, depth + 1) for value in (values or ()))):
                return f'local {expr.value} is a range loop variable'
            # diag-jp: bound only by plain and literal-tuple assignments (`x, m = f(k), g(k)`)
            unpacked = self.formula.unpacked.get(expr.value, ())
            every = [*(values or ()), *unpacked]
            if (unpacked and expr.value not in self._params
                    and self.formula.binds.get(expr.value, 0) == len(every)
                    and all(self.is_int(value, depth + 1) for value in every)):
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
        if isinstance(expr, cst.Subscript) and self._int_index_read(expr):
            return 'extreme of an integer level'
        if isinstance(expr, cst.Call) and _plain_args(expr):
            func = expr.func
            from .vectors import ALIAS as _VECTORS
            if (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                    and func.value.value == _VECTORS and func.attr.value == 'ivec_get'):
                return 'element of an integer vector'      # diag-kr: the vector pass's typed read
            if isinstance(func, cst.Name) and func.value in ('min', 'max', 'abs') and expr.args:
                return ('integer min/max' if all(self.is_int(arg.value, depth + 1) for arg in expr.args)
                        else None)
            if isinstance(func, cst.Name) and func.value == 'len' and len(expr.args) == 1:
                return 'len()'
            inner = self.int_argument(expr)
            if inner is not None:
                if self.is_int(inner, depth + 1) or self._int_index_read(inner):
                    return 'int() of an integer'
                return None
            kind, cell = self.call_rettype(expr)
            if kind in INT_TYPES:
                return f'cell {cell} returns {kind}'
        return None

    def _int_index_read(self, expr):
        return False  # Prepared pandas frames supply this fact during lowering.

    def exact_long_long(self, expr):
        """``expr`` is statically a C ``long long`` (a cell result or a parameter)."""
        if isinstance(expr, cst.Name):
            return (self.param_type(expr.value) == 'long long' or expr.value in self._key_locals
                    or (expr.value in self.formula.c_long_long and expr.value not in self._params))
        return isinstance(expr, cst.Call) and self.call_rettype(expr)[0] == 'long long'

    def fits_long_long(self, expr, depth=0):
        """Prove a signed 64-bit integer value from the expression and its local bindings."""
        if depth > 6:
            return False
        if isinstance(expr, cst.Integer):
            return expr.evaluated_value < 2**63
        if (isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, cst.Minus)
                and isinstance(expr.expression, cst.Integer)):
            return expr.expression.evaluated_value <= 2**63
        if isinstance(expr, cst.Name):
            if not self.formula.unshadowed_load(expr):
                return False
            if self.exact_long_long(expr):
                return True
            name = expr.value
            values = self.formula.values.get(name, ())
            unpacked = self.formula.unpacked.get(name, ())
            sources = (*values, *unpacked)
            return (bool(sources) and name not in self._params
                    and self.formula.binds.get(name, 0) == len(sources)
                    and all(self.fits_long_long(value, depth + 1) for value in sources))
        if isinstance(expr, cst.Call):
            inner = self.int_argument(expr)
            if inner is not None:
                return self.fits_long_long(inner, depth + 1)
            receiver = expr.func
            while isinstance(receiver, cst.Attribute):
                receiver = receiver.value
            return (isinstance(receiver, cst.Name) and receiver.value == 'self'
                    and 'self' not in self.formula.binds and self.formula.unshadowed_load(receiver)
                    and self.exact_long_long(expr))
        return False
