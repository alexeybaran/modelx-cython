"""Lexical boundaries shared by formula-local specialization passes."""
import ast

import libcst as cst


OPAQUE_SCOPES = (
    cst.FunctionDef, cst.ClassDef, cst.Lambda, cst.GeneratorExp,
    cst.ListComp, cst.SetComp, cst.DictComp,
)


def expression_ast(expression):
    """Parse detached expressions with their implicit parenthesized context."""
    source = expression if isinstance(expression, str) else cst.Module([]).code_for_node(expression)
    return ast.parse('(\n' + source + '\n)', mode='eval').body


class Names(cst.CSTVisitor):
    """Allocate identifiers from syntax, never comments or literal text."""
    def __init__(self, node=None, used=()):
        self.names = set(used)
        if node is not None:
            node.visit(self)

    def visit_Name(self, node):
        self.names.add(node.value)

    def reserve(self, base, family=False):
        name = base
        while name in self.names or (family and any(
                item.startswith(name + '_') for item in self.names)):
            name += '_'
        self.names.add(name)
        return name


class FormulaTransformer(cst.CSTTransformer):
    """Visit one formula at a time without borrowing facts into child scopes."""
    def __init__(self):
        self._formula_root = None

    def on_visit(self, node):
        if isinstance(node, cst.FunctionDef):
            if self._formula_root is not None:
                return False
            self._formula_root = node
        elif isinstance(node, OPAQUE_SCOPES):
            return False
        return super().on_visit(node)

    def on_leave(self, original_node, updated_node):
        result = super().on_leave(original_node, updated_node)
        if original_node is self._formula_root:
            self._formula_root = None
        return result


def lexical_nodes(statements):
    """Walk statements in the current scope, excluding child-scope contents."""
    opaque = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
              ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)
    def walk(node):
        yield node
        if not isinstance(node, opaque):
            for child in ast.iter_child_nodes(node):
                yield from walk(child)
    for statement in statements:
        yield from walk(statement)


def binding_names(nodes):
    """Include bindings represented by AST strings as well as Store/Del nodes."""
    names = set()
    for node in nodes:
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split('.')[0])
        elif isinstance(node, (ast.ExceptHandler, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name:
                names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif type(node).__name__ in ('MatchAs', 'MatchStar', 'MatchMapping'):
            name = getattr(node, 'name', None) or getattr(node, 'rest', None)
            if name:
                names.add(name)
    return names


def has_scope_effects(statements):
    """Value-only rewrites must not remove or relocate observable bindings."""
    effects = (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom,
               ast.ExceptHandler, ast.Delete, ast.NamedExpr, ast.FunctionDef,
               ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    return any(isinstance(node, effects) or type(node).__name__ == 'Match'
               for node in lexical_nodes(statements))
