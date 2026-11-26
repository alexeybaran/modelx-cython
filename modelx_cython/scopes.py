"""Lexical boundaries shared by formula-local specialization passes."""

import libcst as cst
from collections import Counter
from contextlib import contextmanager
from libcst.metadata import ExpressionContext, ExpressionContextProvider, MetadataWrapper

from .syntax import children, dotted_name, kept_children, walk


OPAQUE_SCOPES = (
    cst.FunctionDef, cst.ClassDef, cst.Lambda, cst.GeneratorExp,
    cst.ListComp, cst.SetComp, cst.DictComp,
)


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


def lexical_nodes(body):
    def current(node):
        yield node
        if not isinstance(node, OPAQUE_SCOPES):
            for child in children(node):
                yield from current(child)
    for statement in body:
        yield from current(statement)


def has_scope_effects(body):
    effects = (cst.Global, cst.Nonlocal, cst.Import, cst.ImportFrom, cst.ExceptHandler,
               cst.Del, cst.NamedExpr, cst.FunctionDef, cst.ClassDef, cst.Lambda)
    return any(isinstance(node, effects) or isinstance(node, getattr(cst, 'Match', ()))
               for node in lexical_nodes(body))


_SHARED = None     # while shared_facts() is open: {key of the nodes: (the nodes, CstFacts)}


@contextmanager
def shared_facts():
    """Within the block, CstFacts of the same nodes (the same node, or a sequence of the same
    statements) is computed once, and so is each node's ``children``: the nodes are immutable,
    so both are the same."""
    global _SHARED
    outer = _SHARED
    if outer is None:
        _SHARED = {}
    try:
        with kept_children():
            yield
    finally:
        _SHARED = outer


class CstFacts:
    """Read and binding contexts on existing CST nodes; no source reparse."""
    def __init__(self, node):
        shared = _SHARED
        if shared is not None:
            key = tuple(map(id, node)) if isinstance(node, (list, tuple)) else id(node)
            found = shared.get(key)
            if found is not None:
                self.__dict__.update(found[1].__dict__)
                return
            shared[key] = node, self        # the nodes are held, so that their ids stay theirs
        if isinstance(node, cst.Module):
            module = node
        elif isinstance(node, cst.BaseExpression):
            module = cst.Module([cst.SimpleStatementLine([cst.Expr(node)])])
        else:
            module = cst.Module(tuple(cst.SimpleStatementLine([item])
                                      if isinstance(item, cst.BaseSmallStatement) else item for item in node))
        self.contexts = MetadataWrapper(module, unsafe_skip_copy=True).resolve(ExpressionContextProvider)
        self.reads = frozenset(node.value for node, context in self.contexts.items()
                               if isinstance(node, cst.Name) and context is ExpressionContext.LOAD)
        self.stores = Counter(node.value for node, context in self.contexts.items()
                              if isinstance(node, cst.Name) and context is ExpressionContext.STORE)
        self.deleted = frozenset(node.value for node, context in self.contexts.items()
                                if isinstance(node, cst.Name) and context is ExpressionContext.DEL)

    def bindings(self, nodes):
        names = set()
        for node in nodes:
            if isinstance(node, cst.Name) and self.contexts.get(node) in (
                    ExpressionContext.STORE, ExpressionContext.DEL):
                names.add(node.value)
            elif isinstance(node, cst.ImportAlias):
                name = node.asname.name.value if node.asname else dotted_name(node.name).split('.')[0]
                names.add(name)
            elif isinstance(node, (cst.FunctionDef, cst.ClassDef)):
                names.add(node.name.value)
            elif isinstance(node, cst.ExceptHandler) and node.name:
                names.update(item.value for item in walk(node.name.name) if isinstance(item, cst.Name))
            elif isinstance(node, (cst.Global, cst.Nonlocal)):
                names.update(item.name.value for item in node.names)
            elif type(node).__name__ in ('MatchAs', 'MatchStar', 'MatchMapping'):
                name = getattr(node, 'name', None) or getattr(node, 'rest', None)
                if isinstance(name, cst.Name):
                    names.add(name.value)
        return names
