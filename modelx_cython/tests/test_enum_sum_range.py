from types import SimpleNamespace

import libcst as cst
import pytest

from modelx_cython.reductions import NativeReductions
from modelx_cython.typedefs import INT64_MAX


def lower_sum(members, argument_type='long long', *, exact_receiver=True, term='self.term(t, kind)'):
    formula = cst.parse_module(
        f'def formula(self, t):\n    return sum({term} for kind in ({members}))\n'
    )
    cell = SimpleNamespace(
        params=('t', 'kind'),
        has_typeinfo=lambda: True,
        called_with_kwargs=False,
        body_has_closure=False,
        get_argtype_expr=lambda name, **kwargs: 'long long' if name == 't' else argument_type,
        get_rettype_expr=lambda **kwargs: 'double',
    )
    formula_cell = SimpleNamespace(
        params=('t',),
        formula_allows_exact_receiver=exact_receiver,
        get_rettype_expr=lambda **kwargs: 'double',
    )
    owner = SimpleNamespace(name='_c_Projection', refs={}, cells={'term': cell})
    members_by_name = (('A', -1), ('B', 0), ('C', 1), ('D', 3),
                       ('E', INT64_MAX - 1), ('F', INT64_MAX))
    symbols = {
        ('_c_Projection', 'Kind', name): (f'_mx_enum_const_{index}', value)
        for index, (name, value) in enumerate(members_by_name)
    }
    lowering = NativeReductions(owner, formula_cell, set(), formula.body[0], True, {'sum'}, symbols)
    return formula.visit(lowering).code


def test_consecutive_enum_sum_has_an_exact_receiver_range_and_keeps_the_unroll_fallback():
    result = lower_sum('self.Kind.A, self.Kind.B, self.Kind.C')
    assert '_mx_reduce_0_v_kind: _mx_cy.longlong' in result
    assert 'if type(self) is _c_Projection:\n            for _mx_reduce_0_v_kind in range(-1, 2):' in result
    assert 'self.term(t, _mx_reduce_0_v_kind)' in result
    fallback = result.split('else:', 1)[1]
    assert fallback.index('self.term(t, self.Kind.A)') < fallback.index('self.term(t, self.Kind.B)')
    assert fallback.index('self.term(t, self.Kind.B)') < fallback.index('self.term(t, self.Kind.C)')
    assert '.F64Sum' in result and '.f64_result(' in result
    assert 'except StopIteration' in result


@pytest.mark.parametrize(('members', 'argument_type', 'exact_receiver', 'term'), [
    ('self.Kind.A, self.Kind.B, self.Kind.D', 'long long', True, 'self.term(t, kind)'),
    ('self.Kind.B, self.Kind.A', 'long long', True, 'self.term(t, kind)'),
    ('self.Kind.A, self.Kind.B', 'object', True, 'self.term(t, kind)'),
    ('self.Kind.A, self.Kind.B', 'long long', False, 'self.term(t, kind)'),
    ('self.Kind.A, self.Kind.B', 'long long', True, 'self.term(t, kind + 1)'),
    ('self.Kind.E, self.Kind.F', 'long long', True, 'self.term(t, kind)'),
])
def test_enum_sum_keeps_the_unroll_when_the_range_cannot_preserve_the_calls(
    members, argument_type, exact_receiver, term,
):
    result = lower_sum(members, argument_type, exact_receiver=exact_receiver, term=term)
    assert ' in range(' not in result
    assert all(member.strip() in result for member in members.split(','))
    assert result.count('self.term(') == len(members.split(','))
    assert '.F64Sum' in result
