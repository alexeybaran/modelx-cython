"""Small operations on the compiler's existing LibCST representation."""
from dataclasses import fields

import libcst as cst


def code(node):
    return cst.Module([]).code_for_node(node)


def statements(suite):
    """Expose statements without formatting-only line/suite wrappers."""
    body = suite.body if isinstance(suite, (cst.BaseSuite, cst.Module)) else suite
    return tuple(small for item in body for small in
                 (item.body if isinstance(item, cst.SimpleStatementLine) else (item,)))


def single_statement(line):
    if isinstance(line, cst.SimpleStatementLine) and len(line.body) == 1:
        return line.body[0]
    return None


def local_assignment(node):
    if (isinstance(node, cst.Assign) and len(node.targets) == 1
            and isinstance(target := node.targets[0].target, cst.Name)):
        return target
    return None


# The fields each node class visits, in LibCST's own order, learned from LibCST the first
# time a class is met. ``CSTNode.children`` and ``CSTNode.visit`` rebuild a node (with its
# validation) to list its children; reading the learned fields lists the same children, in
# the same order, without building anything. A class whose visiting depends on a value
# (one field visited by different helpers, or not at all) keeps LibCST's own traversal.
_FIELDS = {}
_CONDITIONAL = frozenset(('FormattedStringExpression', 'TemplatedStringExpression',
                          'ImportFrom', 'LazyImportFrom'))


class _FieldRecorder(cst.CSTVisitor):
    def __init__(self, node):
        self.node = node
        self.fields = []
        self.children = []

    def on_visit_attribute(self, node, attribute):
        if node is self.node:
            self.fields.append(attribute)

    def on_leave_attribute(self, original_node, attribute):
        pass

    def on_visit(self, node):
        self.children.append(node)
        return False

    def on_leave(self, original_node):
        pass


def _learn(node):
    recorder = _FieldRecorder(node)
    node._visit_and_replace_children(recorder)
    kind = type(node)
    if kind.__name__ not in _CONDITIONAL:
        _FIELDS[kind] = tuple(recorder.fields)
    return recorder.children


_NODE, _NONE, _MANY = 0, 1, 2
_VALUE_KINDS = {type(None): _NONE, cst.MaybeSentinel: _NONE, tuple: _MANY, list: _MANY}


def _value_kind(value):
    """How a visited field's value holds nodes: one node, none, or a sequence of them (the
    tests LibCST's visit helpers make, by type: ``isinstance`` on its abstract classes is slow)."""
    kind = _VALUE_KINDS.get(type(value))
    if kind is None:
        kind = (_NODE if isinstance(value, cst.CSTNode) else
                _NONE if value is None or isinstance(value, cst.MaybeSentinel) else _MANY)
        if type(value).__module__.startswith('libcst'):     # a node class or a sentinel
            _VALUE_KINDS[type(value)] = kind
    return kind


def children(node):
    """``node.children`` (the same nodes in the same order), read from the node's fields."""
    names = _FIELDS.get(type(node))
    if names is None:
        return _learn(node)
    found = []
    for name in names:
        value = getattr(node, name)
        kind = _value_kind(value)
        if kind == _NODE:
            found.append(value)
        elif kind == _MANY:
            found.extend(value)
    return found


_ATTRIBUTE_HOOKS = {}    # visitor class -> whether its attribute callbacks can do anything


def _attribute_hooks(kind):
    """Whether a visitor class reacts to ``on_visit_attribute``/``on_leave_attribute``: it
    overrides them, or defines a ``visit_<Node>_<field>``/``leave_<Node>_<field>`` method
    (LibCST's own are no-ops)."""
    found = _ATTRIBUTE_HOOKS.get(kind)
    if found is None:
        base = cst.CSTVisitor
        found = _ATTRIBUTE_HOOKS[kind] = (
            kind.on_visit_attribute is not base.on_visit_attribute
            or kind.on_leave_attribute is not base.on_leave_attribute
            or any(name.startswith(('visit_', 'leave_')) and name.count('_') > 1
                   and getattr(kind, name) is not getattr(base, name, None) for name in dir(kind)))
    return found


def visit(node, visitor):
    """``node.visit(visitor)`` for a :class:`libcst.CSTVisitor`, without rebuilding nodes.

    The visitor's callbacks run on the same nodes, attributes and order as LibCST's own
    traversal (attribute callbacks only for a visitor that has some); a visitor that
    transforms (a ``CSTTransformer``) must use ``node.visit``.
    """
    if _attribute_hooks(type(visitor)):
        _visit_attributes(node, visitor)
    else:
        _visit(node, visitor)


def _visit(node, visitor):
    if visitor.on_visit(node):
        names = _FIELDS.get(type(node))
        if names is None:
            if type(node).__name__ in _CONDITIONAL:
                node._visit_and_replace_children(visitor)
                visitor.on_leave(node)
                return
            _learn(node)
            names = _FIELDS[type(node)]
        for name in names:
            value = getattr(node, name)
            kind = _value_kind(value)
            if kind == _NODE:
                _visit(value, visitor)
            elif kind == _MANY:
                for item in value:
                    _visit(item, visitor)
    visitor.on_leave(node)


def _visit_attributes(node, visitor):
    if visitor.on_visit(node):
        names = _FIELDS.get(type(node))
        if names is None:
            if type(node).__name__ in _CONDITIONAL:
                node._visit_and_replace_children(visitor)
                visitor.on_leave(node)
                return
            _learn(node)
            names = _FIELDS[type(node)]
        for name in names:
            visitor.on_visit_attribute(node, name)
            value = getattr(node, name)
            kind = _value_kind(value)
            if kind == _NODE:
                _visit_attributes(value, visitor)
            elif kind == _MANY:
                for item in value:
                    _visit_attributes(item, visitor)
            visitor.on_leave_attribute(node, name)
    visitor.on_leave(node)


def walk(node):
    """``node`` and every node below it, depth first in source order."""
    stack = [node]
    while stack:
        node = stack.pop()
        yield node
        found = children(node)
        found.reverse()
        stack.extend(found)


def dotted_name(node):
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        parent = dotted_name(node.value)
        return parent + '.' + node.attr.value if parent else None
    return None


def arguments(node, name=None, count=None):
    if (not isinstance(node, cst.Call) or name is not None and dotted_name(node.func) != name
            or count is not None and len(node.args) != count
            or any(arg.keyword is not None or arg.star for arg in node.args)):
        return None
    return tuple(arg.value for arg in node.args)


def literal_value(node):
    """Decode CST literals, never execute names, calls or expressions.

    Decode numeric tokens directly and let LibCST decode string escapes.
    """
    if isinstance(node, cst.Integer):
        return int(node.value, 0)
    if isinstance(node, cst.Float):
        return float(node.value)
    if isinstance(node, cst.Imaginary):
        return complex(node.value)
    if isinstance(node, (cst.SimpleString, cst.ConcatenatedString)):
        value = node.evaluated_value
        if value is None:
            raise ValueError('formatted strings are not literals')
        return value
    if isinstance(node, cst.Name) and node.value in ('True', 'False', 'None'):
        return {'True': True, 'False': False, 'None': None}[node.value]
    if isinstance(node, cst.Ellipsis):
        return Ellipsis
    if isinstance(node, (cst.Tuple, cst.List, cst.Set)):
        if any(not isinstance(item, cst.Element) for item in node.elements):
            raise ValueError('starred values are not specification literals')
        values = (literal_value(item.value) for item in node.elements)
        return (tuple if isinstance(node, cst.Tuple) else
                list if isinstance(node, cst.List) else set)(values)
    if isinstance(node, cst.Dict):
        if any(not isinstance(item, cst.DictElement) for item in node.elements):
            raise ValueError('dictionary unpacking is not a specification literal')
        return {literal_value(item.key): literal_value(item.value) for item in node.elements}
    if (isinstance(node, cst.UnaryOperation) and isinstance(node.operator, (cst.Plus, cst.Minus))
            and isinstance(node.expression, (cst.Integer, cst.Float, cst.Imaginary))):
        value = literal_value(node.expression)
        return -value if isinstance(node.operator, cst.Minus) else value
    if (isinstance(node, cst.BinaryOperation) and isinstance(node.operator, (cst.Add, cst.Subtract))
            and isinstance(node.right, cst.Imaginary)):
        left, right = literal_value(node.left), literal_value(node.right)
        if type(left) in (int, float) and type(right) is complex:
            return left + right if isinstance(node.operator, cst.Add) else left - right
    if arguments(node, 'set', 0) is not None:
        return set()
    raise ValueError('expected a Python literal')


def read_literal(source):
    module = cst.parse_module(source.lstrip(' \t'))
    body = statements(module)
    if (len(body) != 1 or not isinstance(body[0], cst.Expr)
            or body[0].semicolon is not cst.MaybeSentinel.DEFAULT):
        raise ValueError('expected one literal expression')
    return literal_value(body[0].value)


_FORMATTING = frozenset((
    'lpar', 'rpar', 'comma', 'semicolon', 'leading_lines', 'lines_after_decorators',
    'header', 'footer', 'indent', 'trailing_whitespace', 'default_indent',
    'default_newline', 'encoding', 'has_trailing_newline', 'equal', 'dot',
))


def semantic_key(node):
    """Compare existing CST structure independently of comments and layout."""
    if isinstance(node, (cst.Integer, cst.Float, cst.Imaginary,
                         cst.SimpleString, cst.ConcatenatedString)):
        value = node.evaluated_value
        if value is not None:
            return 'literal', type(value).__name__, value
    if isinstance(node, cst.CSTNode):
        return type(node).__name__, tuple(
            (field.name, semantic_key(getattr(node, field.name)))
            for field in fields(node)
            if field.name not in _FORMATTING and not field.name.startswith('whitespace'))
    if isinstance(node, (tuple, list)):
        return tuple(semantic_key(item) for item in node)
    return node


def same_expression(left, right):
    return semantic_key(left) == semantic_key(right)
