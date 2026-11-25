"""Lexical boundaries shared by formula-local specialization passes."""

import libcst as cst
from collections import Counter
from contextlib import contextmanager
from libcst.metadata import ExpressionContext, ExpressionContextProvider, MetadataWrapper

from .syntax import dotted_name, walk


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
            for child in node.children:
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
    statements) is computed once: the nodes are immutable, so the facts are the same."""
    global _SHARED
    outer = _SHARED
    if outer is None:
        _SHARED = {}
    try:
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


class _Unknown(Exception):
    """A binding construct the definite-assignment analysis does not follow."""


_UNREACHABLE = 'unreachable'


def _join(*states):
    reachable = [state for state in states if state != _UNREACHABLE]
    return all(reachable) if reachable else _UNREACHABLE


def _binds(target, name):
    """Whether an assignment target binds ``name`` (a name, or one inside a tuple or list target)."""
    if isinstance(target, cst.Name):
        return target.value == name
    if isinstance(target, (cst.Tuple, cst.List)):
        return any(_binds(element.value, name) for element in target.elements)
    return False


def maybe_unbound(body, name, facts):
    """Whether a read of the local ``name`` can come before any assignment to it, on some path of the
    function ``body`` (or the body binds it in a way not followed here). Python raises then; a C local
    has no unbound state, so such a local must stay a Python object."""
    reads = {node for node, context in facts.contexts.items()
             if isinstance(node, cst.Name) and node.value == name and context is ExpressionContext.LOAD}
    for part in walk(cst.Module(list(body))):
        if isinstance(part, (cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp)):
            comprehension = part.for_in
            while comprehension is not None:
                if _binds(comprehension.target, name):
                    reads.difference_update(walk(part))     # the comprehension's own variable
                comprehension = comprehension.inner_for_in

    class Found(Exception):
        pass

    def read(node, state):
        if state is False and any(part in reads for part in walk(node)):
            raise Found
        if any(isinstance(part, cst.NamedExpr) and _binds(part.target, name) for part in walk(node)):
            raise _Unknown

    def block(lines, state):
        for line in lines:
            if state == _UNREACHABLE:
                break
            state = statement(line, state)
        return state

    def suite(node, state):
        if isinstance(node, cst.SimpleStatementSuite):
            for small in node.body:
                if state == _UNREACHABLE:
                    break
                state = small_statement(small, state)
            return state
        return block(node.body, state)

    def statement(line, state):
        if isinstance(line, cst.SimpleStatementLine):
            for small in line.body:
                state = small_statement(small, state)
            return state
        if isinstance(line, cst.If):
            read(line.test, state)
            taken = suite(line.body, state)
            if isinstance(line.orelse, cst.If):
                other = statement(line.orelse, state)
            elif line.orelse is not None:
                other = suite(line.orelse.body, state)
            else:
                other = state
            return _join(taken, other)
        if isinstance(line, (cst.For, cst.While)):
            if isinstance(line, cst.For):
                read(line.iter, state)
                inside = True if _binds(line.target, name) else state
                if not _binds(line.target, name):
                    read(line.target, state)
            else:
                read(line.test, state)
                inside = state
            suite(line.body, inside)
            # the body may run no time; the else clause follows it
            return _join(state, suite(line.orelse.body, state)) if line.orelse is not None else state
        if isinstance(line, cst.Try):
            body_end = suite(line.body, state)
            ends = [suite(line.orelse.body, body_end) if line.orelse is not None else body_end]
            for handler in line.handlers:
                if handler.name is not None and _binds(handler.name.name, name):
                    raise _Unknown
                if handler.type is not None:
                    read(handler.type, state)
                ends.append(suite(handler.body, state))
            end = _join(*ends)
            if line.finalbody is not None:
                suite(line.finalbody.body, state)
                end = suite(line.finalbody.body, end) if end != _UNREACHABLE else end
            return end
        if isinstance(line, cst.With):
            for item in line.items:
                read(item.item, state)
                if item.asname is not None and _binds(item.asname.name, name):
                    state = True if state != _UNREACHABLE else state
            return suite(line.body, state)
        if isinstance(line, (cst.FunctionDef, cst.ClassDef)):
            read(line, state)
            return True if line.name.value == name else state
        raise _Unknown

    def small_statement(small, state):
        if isinstance(small, cst.Assign):
            read(small.value, state)
            for target in small.targets:
                if _binds(target.target, name):
                    state = True
                else:
                    read(target.target, state)
            return state
        if isinstance(small, cst.AnnAssign):
            if small.value is None:
                return state
            read(small.value, state)
            return True if _binds(small.target, name) else (read(small.target, state) or state)
        if isinstance(small, cst.AugAssign):
            read(small.target, state)
            read(small.value, state)
            return state
        if isinstance(small, (cst.Return, cst.Raise)):
            read(small, state)
            return _UNREACHABLE
        if isinstance(small, (cst.Break, cst.Continue)):
            return _UNREACHABLE
        if isinstance(small, (cst.Expr, cst.Assert, cst.Pass)):
            read(small, state)
            return state
        if isinstance(small, cst.Del) and not any(isinstance(part, cst.Name) and part.value == name
                                                  for part in walk(small)):
            return state
        raise _Unknown          # del, global, nonlocal, import: not followed

    try:
        block(body, False)
    except (Found, _Unknown):
        return True
    return False
