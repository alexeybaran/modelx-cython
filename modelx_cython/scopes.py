"""Lexical boundaries shared by formula-local specialization passes."""

import libcst as cst
from collections import Counter
from collections.abc import Sequence
from libcst.metadata import ExpressionContext

from .syntax import children, dotted_name, visit, walk


OPAQUE_SCOPES = (
    cst.FunctionDef, cst.ClassDef, cst.Lambda, cst.GeneratorExp,
    cst.ListComp, cst.SetComp, cst.DictComp,
)


class Names(cst.CSTVisitor):
    """Allocate identifiers from syntax, never comments or literal text."""
    def __init__(self, node=None, used=()):
        self.names = set(used)
        if node is not None:
            visit(node, self)

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
    """Every node of ``body`` (a sequence of statements), depth first in source order, not
    descending into a nested scope (the scope's own node is included)."""
    for statement in body:
        stack = [statement]
        while stack:
            node = stack.pop()
            yield node
            if not isinstance(node, OPAQUE_SCOPES):
                found = children(node)
                found.reverse()
                stack.extend(found)


class _Contexts(cst.CSTVisitor):
    """LibCST's ``ExpressionContextProvider`` (its ``ExpressionContextVisitor``, rule for
    rule) recorded into one mapping, on the shared traversal that rebuilds no node."""
    def __init__(self, contexts, context):
        self.contexts = contexts
        self.context = context

    def _visit(self, node, context=None):
        visit(node, self if context is None else _Contexts(self.contexts, context))

    def visit_Assign(self, node):
        for target in node.targets:
            self._visit(target, ExpressionContext.STORE)
        self._visit(node.value)
        return False

    def visit_AnnAssign(self, node):
        self._visit(node.target, ExpressionContext.STORE)
        self._visit(node.annotation)
        if node.value:
            self._visit(node.value)
        return False

    def visit_AugAssign(self, node):
        self._visit(node.target, ExpressionContext.STORE)
        self._visit(node.value)
        return False

    def visit_NamedExpr(self, node):
        self._visit(node.target, ExpressionContext.STORE)
        self._visit(node.value)
        return False

    def visit_Name(self, node):
        self.contexts[node] = self.context
        return False

    def visit_AsName(self, node):
        self._visit(node.name, ExpressionContext.STORE)
        return False

    def visit_CompFor(self, node):
        self._visit(node.target, ExpressionContext.STORE)
        self._visit(node.iter)
        for item in node.ifs:
            self._visit(item)
        if node.inner_for_in:
            self._visit(node.inner_for_in)
        return False

    def visit_For(self, node):
        self._visit(node.target, ExpressionContext.STORE)
        self._visit(node.iter)
        self._visit(node.body)
        if node.orelse:
            self._visit(node.orelse)
        return False

    def visit_Del(self, node):
        self._visit(node.target, ExpressionContext.DEL)
        return False

    def visit_Attribute(self, node):
        self.contexts[node] = self.context
        self._visit(node.value, ExpressionContext.LOAD)
        return False

    def visit_Subscript(self, node):
        self.contexts[node] = self.context
        self._visit(node.value, ExpressionContext.LOAD)
        if isinstance(node.slice, Sequence):
            for item in node.slice:
                self._visit(item, ExpressionContext.LOAD)
        else:
            self._visit(node.slice, ExpressionContext.LOAD)
        return False

    def visit_Tuple(self, node):
        self.contexts[node] = self.context

    def visit_List(self, node):
        self.contexts[node] = self.context

    def visit_StarredElement(self, node):
        self.contexts[node] = self.context

    def visit_ClassDef(self, node):
        self._visit(node.name, ExpressionContext.STORE)
        self._visit(node.body)
        for item in (*node.bases, *node.keywords, *node.decorators):
            self._visit(item)
        return False

    def visit_FunctionDef(self, node):
        self._visit(node.name, ExpressionContext.STORE)
        self._visit(node.params)
        self._visit(node.body)
        for decorator in node.decorators:
            self._visit(decorator)
        if node.returns:
            self._visit(node.returns)
        return False

    def visit_Param(self, node):
        self._visit(node.name, ExpressionContext.STORE)
        if node.annotation:
            self._visit(node.annotation)
        if node.default:
            self._visit(node.default)
        return False


def expression_contexts(module):
    """What ``MetadataWrapper(module, unsafe_skip_copy=True).resolve(ExpressionContextProvider)``
    holds: the context of each Name, Attribute, Subscript, Tuple, List and StarredElement."""
    contexts = {}
    visit(module, _Contexts(contexts, ExpressionContext.LOAD))
    return contexts


def has_scope_effects(body):
    effects = (cst.Global, cst.Nonlocal, cst.Import, cst.ImportFrom, cst.ExceptHandler,
               cst.Del, cst.NamedExpr, cst.FunctionDef, cst.ClassDef, cst.Lambda)
    return any(isinstance(node, effects) or isinstance(node, getattr(cst, 'Match', ()))
               for node in lexical_nodes(body))


class CstFacts:
    """Read and binding contexts on existing CST nodes; no source reparse."""
    def __init__(self, node):
        if isinstance(node, cst.Module):
            module = node
        elif isinstance(node, cst.BaseExpression):
            module = cst.Module([cst.SimpleStatementLine([cst.Expr(node)])])
        else:
            module = cst.Module(tuple(cst.SimpleStatementLine([item])
                                      if isinstance(item, cst.BaseSmallStatement) else item for item in node))
        self.contexts = expression_contexts(module)
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
