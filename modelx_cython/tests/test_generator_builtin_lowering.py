"""Scope admission and semantic differential tests before pipeline integration."""
import builtins
from types import SimpleNamespace

import cython
import libcst as cst
import pytest

from modelx_cython.helpers import module_name
from modelx_cython.reductions import NativeReductions


class Receiver:
    def __init__(self):
        self.events = []
        self.throw_at = None
        self.error = StopIteration('sentinel')

    def predicate(self, i):
        self.events.append(i)
        if i == self.throw_at:
            raise self.error
        return i != 2


def metadata(result='bint', params=()):
    return SimpleNamespace(params=params, has_formula_def=True,
        has_typeinfo=lambda: True, called_with_kwargs=False, body_has_closure=False,
        formula_allows_exact_receiver=True,
        get_rettype_expr=lambda **kw: result,
        get_argtype_expr=lambda *args, **kw: 'long long' if kw.get('c_style') else '_mx_cy.longlong')


def lower(source, names):
    formula = cst.parse_module(source).body[0]
    info = SimpleNamespace(name='_c_Test', refs={}, cells={'predicate': metadata(params=('i',))})
    pass_ = NativeReductions(info, metadata(params=('n',)), set(), formula,
                             lambda: '_cold', True, builtin_reductions=names)
    rewritten = cst.Module([formula.visit(pass_)]).code
    namespace = {'_mx_cy': cython, '_c_Test': Receiver,
        module_name('modelx_cython._native_iterables'): SimpleNamespace(
            all_accepts=lambda f: f is builtins.all,
            any_accepts=lambda f: f is builtins.any)}
    for fallback in pass_.fallbacks:
        exec(cst.Module([fallback]).code, namespace)
    exec(rewritten, namespace)
    return namespace, pass_


@pytest.mark.parametrize('operation', ['all', 'any'])
def test_builtin_reduction_keeps_short_circuit_and_pep479(operation):
    source = f'def formula(self, n):\n    return {operation}(self.predicate(i) for i in range(n) if i != 1)\n'
    original = {}
    exec(source, original)
    lowered, pass_ = lower(source, (operation,))
    assert pass_.count == 1
    for n in (-3, 0, 1, 2, 3, 8):
        left, right = Receiver(), Receiver()
        assert original['formula'](left, n) == lowered['formula'](right, n)
        assert left.events == right.events
    receiver = Receiver()
    receiver.throw_at = 0
    with pytest.raises(RuntimeError) as actual:
        lowered['formula'](receiver, 3)
    assert actual.value.__cause__ is receiver.error
    assert actual.value.__context__ is receiver.error


def test_builtin_wrapper_needs_independent_bool_binding_proof():
    source = 'def formula(self, n):\n    return bool(all(self.predicate(i) for i in range(n)))\n'
    assert lower(source, ('all',))[1].count == 0
    namespace, pass_ = lower(source, ('all', 'bool'))
    assert pass_.count == 1
    assert namespace['formula'](Receiver(), 3) is False


def test_unproved_builtin_and_integer_overflow_expression_keep_generators():
    source = 'def formula(self, n):\n    return all(self.predicate(i) for i in range(n))\n'
    assert lower(source, ())[1].count == 0
    overflow = 'def formula(self, n):\n    return all(self.predicate(i + 1) for i in range(n))\n'
    assert lower(overflow, ('all',))[1].count == 0
