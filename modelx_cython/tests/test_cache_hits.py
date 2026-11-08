"""Safety boundaries for the native field-read optimization."""
import ast
import textwrap
import pytest
import libcst as cst
from modelx_cython import cache_hits as hits


def entry(key='t', guard=None, index=None, flag=False):
    index=index or key
    arguments=f'self: Receiver, {key}: _mx_cy.longlong'
    if flag:arguments+=', exact: _mx_cy.bint'
    body=f'''if self._has_value[{index}]:
    return self._v_value[{index}]
else:
    return calculate(self, {key})
'''
    if guard:body=f'if {guard}:\n'+textwrap.indent(body,'    ')+'else:\n    raise IndexError()\n'
    return f'def _mx_cache_entry_0({arguments}) -> _mx_cy.double:\n'+textwrap.indent(body,'    ')


def rewrite(cache, call, args='self: Receiver, t: _mx_cy.longlong', prelude=''):
    source=cache+f'\ndef _mx_dense_formula_0({args}) -> _mx_cy.double:\n'+textwrap.indent(prelude+f'return {call}\n','    ')
    functions={fn.name:fn for fn in ast.parse(source).body if isinstance(fn,ast.FunctionDef)}
    layout=hits._layout(functions['_mx_cache_entry_0']);assert layout
    return cst.parse_module(source).visit(hits._Calls(functions,{'_mx_cache_entry_0':layout})).code


def test_checked_hit_keeps_checked_miss():
    result=rewrite(entry(guard='0 <= t < 4'),'_mx_cache_entry_0(self, t)')
    fn=ast.parse(result).body[-1]
    expr=fn.body[-1].value
    assert isinstance(expr,ast.IfExp)
    assert ast.unparse(expr.orelse)=='_mx_cache_entry_0(self, t)'
    assert ast.unparse(expr.test)=='0 <= t < 4 and self._has_value[t]'


def test_proven_entry_inherits_existing_hit_without_new_domain():
    result=rewrite(entry(flag=True),'_mx_cache_entry_0(self, t, True)')
    expr=ast.parse(result).body[-1].body[-1].value
    assert isinstance(expr,ast.IfExp) and ast.unparse(expr.test)=='self._has_value[t]'
    assert ast.unparse(expr.orelse)=='_mx_cache_entry_0(self, t, True)'


@pytest.mark.parametrize('call,args',[
    ('_mx_cache_entry_0(self, effect())','self: Receiver, t: _mx_cy.longlong'),
    ('_mx_cache_entry_0(self, t+1)','self: Receiver, t: _mx_cy.longlong'),
    ('_mx_cache_entry_0(self, (t:=1))','self: Receiver, t: _mx_cy.longlong'),
    ('_mx_cache_entry_0(self, t)','self: Receiver, t: _mx_cy.double'),
    ('_mx_cache_entry_0(self, True)','self: Receiver, t: _mx_cy.longlong'),
    ('_mx_cache_entry_0(self, 9223372036854775808)','self: Receiver, t: _mx_cy.longlong'),
    ('_mx_cache_entry_0(other, t)','self: Receiver, other: Receiver, t: _mx_cy.longlong'),
    ('_mx_cache_entry_0(self, t=t)','self: Receiver, t: _mx_cy.longlong'),
    ('_mx_cache_entry_0(self, t)','self: OtherReceiver, t: _mx_cy.longlong'),
])
def test_arguments_with_effects_coercions_or_unknown_receiver_are_unchanged(call,args):
    result=rewrite(entry(guard='0 <= t < 4'),call,args)
    assert isinstance(ast.parse(result).body[-1].body[-1].value,ast.Call)


@pytest.mark.parametrize('guard',['check(t)','self.limit > t','0 <= t / 2 < 4','0 <= other < 4'])
def test_effectful_or_unknown_domain_is_not_copied(guard):
    assert hits._layout(ast.parse(entry(guard=guard)).body[0]) is None


def test_mismatched_value_index_is_rejected():
    source=entry().replace('return self._v_value[t]','return self._v_value[t + 1]')
    assert hits._layout(ast.parse(source).body[0]) is None


@pytest.mark.parametrize('prelude',[
    '_mx_cy = other\n',
    'import other as _mx_cy\n',
    'try:\n    pass\nexcept RuntimeError as _mx_cy:\n    pass\n',
    'match t:\n    case _mx_cy:\n        pass\n',
])
def test_cython_alias_rebinding_is_rejected(prelude):
    result=rewrite(entry(guard='0 <= t < 4'),'_mx_cache_entry_0(self, t)',prelude=prelude)
    assert isinstance(ast.parse(result).body[-1].body[-1].value,ast.Call)


def test_attribute_names_are_not_parameter_references():
    cache=entry(key='cast',guard='(_mx_cy.cast(_mx_cy.ulonglong, cast) - _mx_cy.cast(_mx_cy.ulonglong, 0)) < 4')
    result=rewrite(cache,'_mx_cache_entry_0(self, t)')
    expr=ast.parse(result).body[-1].body[-1].value
    assert isinstance(expr,ast.IfExp)
    assert '_mx_cy.cast(_mx_cy.ulonglong, t)' in ast.unparse(expr.test)
    assert '_mx_cy.t' not in ast.unparse(expr.test)


def test_python_without_pattern_ast_nodes_still_optimizes(monkeypatch):
    monkeypatch.delattr(ast,'Match')
    result=rewrite(entry(guard='0 <= t < 4'),'_mx_cache_entry_0(self,t)')
    assert isinstance(ast.parse(result).body[-1].body[-1].value,ast.IfExp)


@pytest.mark.parametrize('expression',['lambda: _mx_cache_entry_0(self,t)',
    '[_mx_cache_entry_0(self,t) for t in range(4)]',
    '(_mx_cache_entry_0(self,t) for t in range(4))'])
def test_deferred_and_rebound_scopes_are_unchanged(expression):
    result=rewrite(entry(guard='0 <= t < 4'),expression)
    assert not any(isinstance(n,ast.IfExp) for n in ast.walk(ast.parse(result).body[-1]))
