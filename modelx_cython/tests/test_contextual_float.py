"""Expression composability and lexical refusal of contextual conversion."""
from types import SimpleNamespace as NS
import libcst as cst
import pytest
from modelx_cython.numeric_helpers import ContextualFloat


def rewrite(body, builtins=('float',), kind='long long', contexts=None, scope_safe=None):
    helper=NS(helper=('lifelib._native_inputs','lookup'),float_numeric_helper='lookup_float_f64',
              keyed_float_numeric_helper=tuple(f'lookup_key{i}_float_f64' for i in range(5)))
    cls=NS(refs={'lookup':helper,'ops':NS(helper_members={'lookup':helper})},cells={})
    cell=NS(name='rate',params=('t',),get_argtype_expr=lambda *a,**k:kind)
    transform=ContextualFloat(cls,cell,set(),builtins,contexts,scope_safe=scope_safe)
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


@pytest.mark.parametrize('key,arity,flag', [
    ('()',0,'True'), ('("M",)',1,'True'), ('"M"',1,'False'),
    ('("M", t)',2,'True'), ('("M", -3, t)',3,'True'),
    ('("M", t, True, "Y")',4,'True')])
def test_native_complete_keys_do_not_construct_query_tuples(key,arity,flag):
    code,transform=rewrite(f'return float(self.ops.lookup(self.table, {key}, "rate"))')
    assert f'.lookup_key{arity}_float_f64(float, self.ops.lookup, ' in code
    assert f'{flag}, "rate")' in code
    assert transform.operations=={'lifelib._native_inputs.lookup'}


@pytest.mark.parametrize('key', ['("M", t, 1, 2, 3)', '("M", *other)',
                                     '("M", unknown)', '("M", 0.5)', '(b"M", t)',
                                     '(-9223372036854775809, t)'])
def test_unsupported_complete_key_domain_remains_ordinary(key):
    code,transform=rewrite(f'return float(self.ops.lookup(self.table, {key}, "rate"))')
    assert not transform.operations and 'lookup_key' not in code


def test_boolean_scalar_uses_type_preserving_entry():
    body='return float(self.lookup(self.table, t, "rate"))'
    code,transform=rewrite(body,kind='bint')
    assert '.lookup_key1_float_f64(' in code
    assert 't, False, "rate")' in code


def test_mixed_minimum_cannot_erase_boolean_key_identity():
    body='return float(self.lookup(self.table, min(t, True), "rate"))'
    assert not rewrite(body,builtins=('float','min'))[1].operations


def test_original_scope_facts_survive_generated_exception_handlers():
    body='try:\n    return float(self.lookup(self.table, t, "rate"))\nexcept StopIteration as generated_error:\n    raise RuntimeError("generator raised StopIteration") from generated_error'
    assert not rewrite(body)[1].operations
    assert rewrite(body,scope_safe=True)[1].operations
    assert not rewrite(body,scope_safe=False)[1].operations


def test_existing_native_loop_declaration_supplies_key_kind():
    body='i: _mx_cy.longlong\nfor i in range(3):\n    consume(float(self.ops.lookup(self.table, ("M", i), "rate")))\nreturn 1.0'
    code,transform=rewrite(body)
    assert '.lookup_key2_float_f64(float,' in code
    assert transform.native_locals=={'i':'long long'}
    assert not rewrite(body.replace('_mx_cy.longlong','int'))[1].operations


def test_classifier_is_shared_side_effect_free_and_exposes_arithmetic_checks():
    import ast
    code,visitor=rewrite('return 1.0',scope_safe=True)
    node=ast.parse('float(self.ops.lookup(self.table, ("M", i), "rate"))',mode='eval').body
    match=visitor.classify_call(node,{'i':'longlong'})
    assert match is not None and match.result_kind=='double' and match.checked_arguments==()
    assert not visitor.operations and not visitor.contexts and not visitor.providers
    assert visitor.native_locals=={}
    assert visitor.classify_call(node,{'i':'double'}) is None


def test_classifier_leaves_cell_argument_arithmetic_to_generator_proof():
    import ast
    _,visitor=rewrite('return 1.0',scope_safe=True)
    visitor.cls.cells['age']=NS(is_special=lambda:False,called_with_kwargs=False,
                               body_has_closure=False,get_rettype_expr=lambda **k:'long long')
    node=ast.parse('float(self.ops.lookup(self.table, ("M", self.age(i+1)), "rate"))',mode='eval').body
    match=visitor.classify_call(node,{'i':'longlong'})
    assert match is not None
    assert tuple(ast.unparse(x) for x in match.checked_arguments)==('i + 1',)
    assert not visitor.operations and not visitor.contexts


def test_classifier_refuses_object_argument_arithmetic():
    import ast
    _,visitor=rewrite('return 1.0',scope_safe=True)
    node=ast.parse('float(self.ops.lookup(self.table[i+1], ("M", i), "rate"))',mode='eval').body
    assert visitor.classify_call(node,{'i':'longlong'}) is None
