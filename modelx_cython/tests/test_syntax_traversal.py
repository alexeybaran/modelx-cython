"""The shared traversal reads the nodes LibCST's own traversal visits, in its order."""
from pathlib import Path

import libcst as cst
import pytest

from modelx_cython import syntax
from modelx_cython.scopes import lexical_nodes

SNIPPETS = '''
from . import a as b, c
from .x import (y as z, w)
from m import *
import p.q as r, s
global g
x: int = (y := 1) + -z ** 2 // 3 @ m
del x[1:2, ::3], y.z
f"{a!r:>{w}} {b=}" "plain"
b"x" rb'y'
lambda a, /, b=1, *c, d, **e: (yield)
async def f[T, *U, **V](x: T = 0) -> "U":
    async with a as (b, c), d:
        async for [e, *f] in g:
            await h
    nonlocal_value = [i for i in j if i for k in i]
    return {k: v for k, v in m.items()}, {*s}, (t for t in u)
class C(B, metaclass=M):
    @dec(1)
    def m(self): pass
try:
    pass
except* (A, B) as e:
    raise X from e
else:
    pass
finally:
    pass
match command:
    case [Point(x=0, y=0), *rest] | {"k": v, **kw} if v > 0:
        pass
    case str() as s:
        pass
type Alias[T] = list[T]
while not a and b or c: break
if a is not b in c != d: continue
assert x, "message"
'''


def sources():
    package = Path(syntax.__file__).parent
    texts = [SNIPPETS]
    texts.extend(path.read_text(encoding='utf-8') for path in sorted(package.glob('*.py')))
    return texts


def libcst_children(node):
    return list(node.children)


class Record(cst.CSTVisitor):
    def __init__(self):
        self.log = []

    def on_visit(self, node):
        self.log.append(('visit', node))
        return not isinstance(node, cst.Lambda)

    def on_leave(self, node):
        self.log.append(('leave', node))

    def on_visit_attribute(self, node, attribute):
        self.log.append(('visit', node, attribute))

    def on_leave_attribute(self, node, attribute):
        self.log.append(('leave', node, attribute))


@pytest.mark.parametrize('index', range(len(sources())))
def test_children_walk_and_visit_match_libcst(index):
    module = cst.parse_module(sources()[index])
    expected_walk = []
    stack = [module]
    while stack:
        node = stack.pop()
        expected_walk.append(node)
        found = libcst_children(node)
        assert [id(child) for child in syntax.children(node)] == [id(child) for child in found]
        stack.extend(reversed(found))
    assert [id(node) for node in syntax.walk(module)] == [id(node) for node in expected_walk]

    expected, actual = Record(), Record()
    module.visit(expected)
    syntax.visit(module, actual)
    assert [(kind, id(node), *rest) for kind, node, *rest in actual.log] == \
        [(kind, id(node), *rest) for kind, node, *rest in expected.log]


def test_lexical_nodes_stop_at_nested_scopes():
    body = cst.parse_module('a = 1\ndef f():\n    b = 2\nc = [d for d in e]\n').body
    names = [node.value for node in lexical_nodes(body) if isinstance(node, cst.Name)]
    assert names == ['a', 'c']


@pytest.mark.parametrize('index', range(len(sources())))
def test_expression_contexts_match_libcst(index):
    from libcst.metadata import ExpressionContextProvider, MetadataWrapper
    from modelx_cython.scopes import expression_contexts
    module = cst.parse_module(sources()[index])
    expected = MetadataWrapper(module, unsafe_skip_copy=True).resolve(ExpressionContextProvider)
    actual = expression_contexts(module)
    assert [(id(node), context) for node, context in actual.items()] == \
        [(id(node), context) for node, context in expected.items()]


@pytest.mark.parametrize('source, bound', [
    ('x = int(y)\nisinstance(x, bool)\nz: int = 1\n', False),
    ('int = 3\n', True),
    ('def f():\n    global int\n    int = 1\n', True),
    ('from m import bool\n', True),
    ('import int.x\n', True),
    ('def isinstance(a, b): pass\n', True),
    ('class C:\n    def m(self, int): pass\n', True),
    ('for bool in y: pass\n', True),
    ('[(int := 1) for _ in y]\n', True),
    ('with a as int: pass\n', True),
    ('try:\n    pass\nexcept E as bool:\n    pass\n', True),
    ('match v:\n    case int(): pass\n', True),
    ('def f[int](): pass\n', True),
    ('del bool\n', True),
])
def test_binds_none_of_agrees_with_scope_provider(source, bound):
    from libcst.metadata import MetadataWrapper, ScopeProvider
    from libcst.metadata.scope_provider import BuiltinAssignment
    from modelx_cython.scopes import binds_none_of
    names = ('isinstance', 'int', 'bool')
    module = cst.parse_module(source)
    scope = MetadataWrapper(module, unsafe_skip_copy=True).resolve(ScopeProvider)[module]
    builtin = all(isinstance(binding, BuiltinAssignment) for name in names for binding in scope[name])
    assert binds_none_of(module, names) is not bound
    if not bound:
        assert builtin


class RecordNodes(cst.CSTVisitor):
    """No attribute callbacks: the traversal leaves them out."""
    def __init__(self):
        self.log = []

    def on_visit(self, node):
        self.log.append(('visit', id(node)))
        return not isinstance(node, cst.Lambda)

    def on_leave(self, node):
        self.log.append(('leave', id(node)))


@pytest.mark.parametrize('index', range(len(sources())))
def test_visit_without_attribute_callbacks_matches_libcst(index):
    module = cst.parse_module(sources()[index])
    assert not syntax._attribute_hooks(RecordNodes)
    expected, actual = RecordNodes(), RecordNodes()
    module.visit(expected)
    syntax.visit(module, actual)
    assert actual.log == expected.log
