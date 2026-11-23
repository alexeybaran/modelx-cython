"""Lower bounded right-associated sums while preserving each cached suffix."""

import ast

import libcst as cst

from .consts import CY_MOD
from .factory_reuse import detached_factory_graph
from .scopes import Names
from .syntax import code


def _pattern(body, key, recursive_entry):
    if len(body) != 3 or not isinstance(body[0], ast.If) or not isinstance(body[1], ast.If):
        return None
    stop, exact, _ = body
    if (stop.orelse or len(stop.body) != 1 or not isinstance(stop.body[0], ast.Return)
            or not isinstance(stop.body[0].value, ast.Constant) or stop.body[0].value.value != 0.0
            or not isinstance(stop.test, ast.Compare) or len(stop.test.ops) != 1
            or not isinstance(stop.test.ops[0], ast.GtE)
            or not isinstance(stop.test.left, ast.Name) or stop.test.left.id != key
            or not isinstance(exact.test, ast.Name) or exact.test.id != '_mx_exact_receiver'
            or exact.orelse or len(exact.body) != 1 or not isinstance(exact.body[0], ast.Return)):
        return None
    total = exact.body[0].value
    if (not isinstance(total, ast.BinOp) or not isinstance(total.op, ast.Add)
            or not isinstance(total.right, ast.Call) or total.right.keywords
            or ast.unparse(total.right) != f'{recursive_entry}(self, {key} + 1)'):
        return None
    horizon = stop.test.comparators[0]
    if any(isinstance(part, ast.Name) and part.id == key for part in ast.walk(horizon)):
        return None
    return horizon, total.left


class _Index(ast.NodeTransformer):
    def __init__(self, parameter, index):
        self.parameter = parameter
        self.index = index

    def visit_Name(self, node):
        return ast.Name(id=self.index, ctx=node.ctx) if node.id == self.parameter else node


def _native_term(node, cls, key):
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, float)
    if isinstance(node, ast.Name):
        return node.id == key
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub)) and _native_term(node.operand, cls, key)
    if isinstance(node, ast.BinOp):
        return (isinstance(node.op, (ast.Add, ast.Sub, ast.Mult))
                and _native_term(node.left, cls, key) and _native_term(node.right, cls, key))
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
            or not isinstance(node.func.value, ast.Name) or node.func.value.id != 'self' or node.keywords):
        return False
    cell = cls.cells.get(node.func.attr)
    return (cell is not None and not cell.body_has_closure and len(cell.params) == len(node.args)
            and cell.get_rettype_expr(c_style=True) in ('double', 'long long', 'bint')
            and all(_native_term(arg, cls, key) for arg in node.args))


def lower_suffix_sums(state, module_info, layouts):
    """Evaluate terms in order and add in reverse on an already cached bound."""
    sources = {node.name.value: node for node in module_info.visitor.wrapper.module.body
               if isinstance(node, cst.ClassDef)}
    replacements = {}
    malloc = state.names.reserve('_mx_suffix_malloc')
    free = state.names.reserve('_mx_suffix_free')
    for owner, cls in module_info.classes.items():
        for name, cell in cls.cells.items():
            if not cell._spec.get('iterative_suffix_sum', False):
                continue
            entry = getattr(cell, 'inline_entry', None)
            layout = layouts.get(entry)
            if (layout is None or len(layout.axes) != 1 or cell.is_locked or cell.has_dense_spill
                    or cell.get_rettype_expr(c_style=True) != 'double'):
                continue
            axis = layout.axes[0]
            capacity = axis.stop - axis.lower
            if axis.c_type != 'long long' or not 1 <= capacity <= 4096:
                continue
            method = state.methods[owner, '_f_' + name]
            wrapper = ast.parse(code(method).lstrip()).body[0]
            if len(wrapper.body) != 1 or not isinstance(wrapper.body[0], ast.Return):
                continue
            call = wrapper.body[0].value
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            formula = state.functions.get(call.func.id)
            if formula is None or state.function_origins.get(call.func.id) != (owner, method.name.value):
                continue
            pattern = _pattern(ast.parse(code(formula)).body[0].body, axis.key, entry)
            if pattern is None:
                continue
            original = ast.parse(code(sources[owner])).body[0]
            authored = next(f for f in original.body if isinstance(f, ast.FunctionDef)
                            and f.name == method.name.value)
            authored_sum = authored.body[-1].value if isinstance(authored.body[-1], ast.Return) else None
            if (not isinstance(authored_sum, ast.BinOp) or not isinstance(authored_sum.op, ast.Add)
                    or not _native_term(authored_sum.left, cls, axis.key)):
                continue
            reads = {n.attr for n in ast.walk(authored) if isinstance(n, ast.Attribute)
                     and isinstance(n.value, ast.Name) and n.value.id == 'self'
                     and n.attr in cls.cells and n.attr != name}
            if not detached_factory_graph(sources[owner], cls, reads, forbidden_cells={method.name.value},
                                          include_factory=False):
                continue
            horizon, term = pattern
            # The exact arm of the generated guard reads a cached int64 Cell.
            # If that Cell is cold, enter normal recursion first. Never hoist or
            # truncate an authored bound expression.
            if (not isinstance(horizon, ast.IfExp)
                    or ast.unparse(horizon.test) != '_mx_exact_receiver'
                    or not isinstance(horizon.body, ast.IfExp)
                    or not isinstance(bound := horizon.body.body, ast.Attribute)
                    or not bound.attr.startswith('_v_')):
                continue
            bound_name = bound.attr[3:]
            bound_cell = cls.cells.get(bound_name)
            if (bound_cell is None or bound_cell.has_args()
                    or bound_cell.get_rettype_expr(c_style=True) != 'long long'
                    or ast.unparse(horizon.body.test) != f'self._has_{bound_name}'
                    or ast.unparse(bound) != f'self._v_{bound_name}'):
                continue
            key = axis.key
            names = Names(formula)
            index, stop, value, terms = (names.reserve('_mx_suffix_' + label)
                                         for label in ('index', 'stop', 'value', 'terms'))
            offset = f'{index} - ({axis.lower})'
            flags = f'self._has_{name}[{offset}]'
            cached = f'self._v_{name}[{offset}]'
            term = ast.unparse(_Index(key, index).visit(term))
            branch = cst.parse_statement(f"""\
if _mx_exact_receiver and self._has_{bound_name} and {axis.lower} <= {key} < {axis.stop}:
    {stop}: {CY_MOD}.longlong = self._v_{bound_name}
    if {key} < {stop} < {axis.stop}:
        {terms}: {CY_MOD}.p_double = {CY_MOD}.cast(
            {CY_MOD}.p_double, {malloc}(({stop} - {key}) * {CY_MOD}.sizeof({CY_MOD}.double)))
        if {terms} != {CY_MOD}.NULL:
            {index}: {CY_MOD}.longlong = {key}
            {value}: {CY_MOD}.double
            try:
                while {index} < {stop} and not {flags}:
                    {terms}[{index} - {key}] = {term}
                    {index} += 1
                if {flags}:
                    {value} = {cached}
                else:
                    {value} = 0.0
                    {cached} = {value}
                    {flags} = True
                while {index} > {key}:
                    {index} -= 1
                    {value} = {terms}[{index} - {key}] + {value}
                    {cached} = {value}
                    {flags} = True
                return {value}
            finally:
                {free}({terms})
""")
            replacements[formula] = formula.with_changes(body=formula.body.with_changes(
                body=(branch, *formula.body.body)))
    if not replacements:
        return state
    imports = cst.parse_module(
        f'from cython.cimports.libc.stdlib import malloc as {malloc}, free as {free}\n').body
    # Append C imports before the first affected function. The module can
    # otherwise start with directives, a module docstring or future imports.
    body = []
    for item in state.syntax.body:
        if item in replacements and imports:
            body.extend(imports)
            imports = ()
        body.append(replacements.get(item, item))
    return state.replace(state.syntax.with_changes(body=tuple(body)))
