"""Expression composability and lexical refusal of contextual conversion."""
from types import SimpleNamespace as NS
import libcst as cst
import pytest
from modelx_cython.numeric_helpers import ContextualFloat


def rewrite(body, builtins=('float',), kind='long long', contexts=None):
    helper=NS(helper=('lifelib._native_inputs','lookup'),float_numeric_helper='lookup_float_f64')
    cls=NS(refs={'lookup':helper,'ops':NS(helper_members={'lookup':helper})},cells={})
    cell=NS(name='rate',params=('t',),get_argtype_expr=lambda *a,**k:kind)
    transform=ContextualFloat(cls,cell,set(),builtins,contexts)
    node=cst.parse_module('def rate(self, t):\n'+''.join('    '+line+'\n' for line in body.splitlines()))
    return node.visit(transform).code,transform


@pytest.mark.parametrize('expression',[
    'return float(self.lookup(self.table, t, "rate"))',
    'x = 2 * float(self.ops.lookup(self.table, t, "rate"))\nreturn x + 1',
    'if t:\n    return float(self.ops.lookup(self.table, t, "rate", "2.5"))\nreturn 0.0',
    'sink.add(float(self.ops.lookup(self.table, t, "rate")))\nreturn 1.0'])
def test_expression_contexts_share_one_lowering(expression):
    code,transform=rewrite(expression)
    assert '.lookup_float_f64(float, self.' in code
    assert transform.operations=={'lifelib._native_inputs.lookup'}
    assert transform.contexts=={'_v_mx_context_rate':'lifelib._native_inputs'}


@pytest.mark.parametrize('expression',[
    'return float(self.ops.lookup(self.table, t, column="rate"))',
    'return float(self.ops.lookup(self.table, t, "rate"), other=1)',
    'return float(x) + 1',
    'return [float(self.ops.lookup(self.table, t, "rate")) for t in self.values]',
    'return (float(self.ops.lookup(self.table, t, "rate")) for t in self.values)',
    'return lambda: float(self.ops.lookup(self.table, t, "rate"))',
    'def inner():\n    return float(self.ops.lookup(self.table, t, "rate"))\nreturn inner()',
    'global float\nreturn float(self.ops.lookup(self.table, t, "rate"))'])
def test_declines_opaque_or_unproven_semantics(expression):
    code,transform=rewrite(expression)
    assert 'lookup_float_f64' not in code and not transform.operations


def test_declines_shadowed_builtin_or_boxed_key():
    body='return float(self.lookup(self.table, t, "rate"))'
    assert not rewrite(body,builtins=())[1].operations
    assert not rewrite(body,kind='object')[1].operations


def test_context_collision_declines_without_changing_existing_owner():
    contexts={'_v_mx_context_rate':'other.provider'}
    assert not rewrite('return float(self.lookup(self.table, t, "rate"))',contexts=contexts)[1].operations
    assert contexts=={'_v_mx_context_rate':'other.provider'}
