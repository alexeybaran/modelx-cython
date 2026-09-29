"""Lower statically numeric fractional powers in native frame formulas.

The fast path handles positive, finite real operands. The generated helper
executes the original Cython power expression for every other case, so a
formula never has to assume a narrower input domain to use this lowering.
"""
import ast
import inspect

from .lookups import Curve, CurveTable, IndexedFloat1, IndexedFloat2, IndexedInt1, IndexedInt2


INT = 'int'
FLOAT = 'float'
UNKNOWN = 'unknown'


def _kind(typ):
    if typ in (int, bool):
        return INT
    if typ is float:
        return FLOAT
    return UNKNOWN


class _NumericKind:
    """Classify only scalar expressions whose types the frame already declares."""

    def __init__(self, cell_types, input_types, parameters, local_names):
        self.cells = cell_types
        self.inputs = input_types
        self.parameters = parameters
        self.local_names = local_names

    def of(self, node):
        if isinstance(node, ast.Constant):
            return _kind(type(node.value))
        if isinstance(node, ast.Name):
            if node.id in self.local_names and node.id not in self.parameters:
                return UNKNOWN
            kind = self.parameters.get(node.id, self.inputs.get(node.id, UNKNOWN))
            return kind if kind in (INT, FLOAT) else UNKNOWN
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self.of(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = self.of(node.left), self.of(node.right)
            if UNKNOWN in (left, right):
                return UNKNOWN
            if isinstance(node.op, ast.Div):
                return FLOAT
            if isinstance(node.op, ast.Pow):
                if right == FLOAT:
                    return FLOAT
                if (isinstance(node.right, ast.Constant)
                        and type(node.right.value) is int and node.right.value >= 0):
                    return FLOAT if left == FLOAT else INT
                return UNKNOWN
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv,
                                    ast.Mod)):
                return FLOAT if FLOAT in (left, right) else INT
        if isinstance(node, ast.IfExp):
            left, right = self.of(node.body), self.of(node.orelse)
            if UNKNOWN in (left, right):
                return UNKNOWN
            return FLOAT if FLOAT in (left, right) else INT
        if isinstance(node, ast.Call) and not node.keywords:
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in self.cells and name not in self.local_names:
                    return self.cells[name]
                if name in self.local_names:
                    return UNKNOWN
                if name in ('float', 'power'):
                    return FLOAT
                if name == 'int':
                    return INT
                if name in ('abs', 'min', 'max') and node.args:
                    kinds = [self.of(arg) for arg in node.args]
                    if UNKNOWN not in kinds:
                        return FLOAT if FLOAT in kinds else INT
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.attr in ('get', 'at', 'previous', 'linear')
                    and node.func.value.id not in self.local_names):
                source = self.inputs.get(node.func.value.id, UNKNOWN)
                if source in (IndexedFloat1, IndexedFloat2, Curve, CurveTable):
                    return FLOAT
                if source in (IndexedInt1, IndexedInt2):
                    return INT
        return UNKNOWN


class _Lower(ast.NodeTransformer):
    def __init__(self, kinds):
        self.kinds = kinds
        self.rewritten = 0

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            base = self.kinds.of(node.left)
            exponent = self.kinds.of(node.right)
            if base != UNKNOWN and exponent == FLOAT:
                updated = ast.Call(ast.Name('_power_compatible', ast.Load()),
                                   [self.visit(node.left), self.visit(node.right)], [])
                self.rewritten += 1
                return ast.copy_location(updated, node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Nested scopes can rebind names that had a declared numeric type in
        # the outer Cell. They are not native-frame eligible today.
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        return node

    def visit_GeneratorExp(self, node):
        return node

    visit_ListComp = visit_GeneratorExp
    visit_SetComp = visit_GeneratorExp
    visit_DictComp = visit_GeneratorExp


def lower_numeric_powers(node, cells, inputs):
    """Return a rewritten formula AST and the number of eligible powers."""
    signatures = {name: _kind(inspect.signature(fn).return_annotation)
                  for name, fn in cells.items()}
    input_types = {name: _kind(typ) if typ in (int, float, bool) else typ
                   for name, typ in inputs.items()}
    parameters = {name: _kind(p.annotation)
                  for name, p in inspect.signature(cells[node.name]).parameters.items()}
    local_names = {part.id for part in ast.walk(node)
                   if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)}
    visitor = _Lower(_NumericKind(signatures, input_types, parameters, local_names))
    for index, statement in enumerate(node.body):
        node.body[index] = visitor.visit(statement)
    return ast.fix_missing_locations(node), visitor.rewritten
