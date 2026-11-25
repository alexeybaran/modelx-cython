"""Lexical boundaries shared by formula-local specialization passes."""

import libcst as cst
from collections import Counter
from collections.abc import Sequence
from types import MappingProxyType

from libcst.metadata import (ExpressionContext, ExpressionContextProvider, MetadataWrapper, ParentNodeProvider,
                             ScopeProvider)
try:
    from libcst.metadata.scope_provider import ScopeVisitor as _ScopeVisitor
except ImportError:     # pragma: no cover - ScopeProvider then resolves as LibCST does
    _ScopeVisitor = None

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
            opaque = _OPAQUE.get(type(node))
            if opaque is None:
                opaque = _OPAQUE[type(node)] = issubclass(type(node), OPAQUE_SCOPES)
            if not opaque:
                found = children(node)
                found.reverse()
                stack.extend(found)


_OPAQUE = {}    # node class -> whether it is one of OPAQUE_SCOPES


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


class _Parents(cst.CSTVisitor):
    """LibCST's ``ParentNodeProvider`` (its ``ParentNodeVisitor``) on the shared traversal."""
    def __init__(self, parents):
        self.parents = parents

    def on_leave(self, original_node):
        for child in children(original_node):
            self.parents[child] = original_node


def parent_nodes(module):
    """What ``MetadataWrapper(module, unsafe_skip_copy=True).resolve(ParentNodeProvider)`` holds."""
    parents = {}
    visit(module, _Parents(parents))
    return parents


def metadata_wrapper(module, copy=False, parents=False):
    """``MetadataWrapper(module, unsafe_skip_copy=not copy)`` with its ExpressionContextProvider,
    ScopeProvider and, with ``parents``, ParentNodeProvider metadata already resolved.

    LibCST resolves ExpressionContextProvider and ParentNodeProvider with two traversals each
    that rebuild every node; they come from the shared traversal instead. ScopeProvider runs
    its own ScopeVisitor as LibCST does, without the batched traversal after it that does
    nothing for it. Every other provider resolves as LibCST does.
    """
    wrapper = MetadataWrapper(module, unsafe_skip_copy=not copy)
    resolved = getattr(wrapper, '_metadata', None)
    if type(resolved) is dict and not resolved:
        resolved[ExpressionContextProvider] = MappingProxyType(expression_contexts(wrapper.module))
        if parents:
            resolved[ParentNodeProvider] = MappingProxyType(parent_nodes(wrapper.module))
        if _ScopeVisitor is not None:
            # what ScopeProvider.visit_Module does when LibCST resolves the provider
            provider = (ScopeProvider(wrapper._cache.get(ScopeProvider)) if ScopeProvider.gen_cache
                        else ScopeProvider())
            with provider.resolve(wrapper):
                visitor = _ScopeVisitor(provider)
                wrapper.module.visit(visitor)
                visitor.infer_accesses()
            resolved[ScopeProvider] = MappingProxyType(dict(provider._computed))
    return wrapper


# Nodes that bind a name of their own (``name`` field) without an expression context
_NAMED_BINDERS = tuple(getattr(cst, kind) for kind in (
    'FunctionDef', 'ClassDef', 'Param', 'TypeAlias', 'TypeVar', 'TypeVarTuple', 'ParamSpec')
    if hasattr(cst, kind))
# Nodes whose names may bind although LibCST reads them as loads
_NAME_BINDERS = tuple(getattr(cst, kind) for kind in (
    'Import', 'ImportFrom', 'Global', 'Nonlocal', 'Match') if hasattr(cst, kind))


def binds_none_of(module, names):
    """True when no node of ``module`` can bind any of ``names``: no store of one, no def,
    class, parameter, type parameter or import of one, no global, nonlocal or match naming
    one. ScopeProvider then gives each of ``names`` its builtin binding alone, in every scope.
    False when one may be bound (ask ScopeProvider)."""
    names = frozenset(names)
    contexts = expression_contexts(module)
    roles = {}      # node class -> 1 a Name, 2 a named binder, 3 a binder of inner names, 0 none
    for node in walk(module):
        kind = type(node)
        role = roles.get(kind)
        if role is None:
            role = roles[kind] = (1 if issubclass(kind, cst.Name) else 2 if issubclass(kind, _NAMED_BINDERS)
                                  else 3 if issubclass(kind, _NAME_BINDERS) else 0)
        if role == 1:
            if node.value in names and contexts.get(node, ExpressionContext.LOAD) is not ExpressionContext.LOAD:
                return False
        elif role == 2:
            if node.name.value in names:
                return False
        elif role == 3:
            if any(isinstance(item, cst.Name) and item.value in names for item in walk(node)):
                return False
    return True


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
