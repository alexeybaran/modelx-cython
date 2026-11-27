"""The shared scalar passes keep both existing math dispatch routes."""
import math
from types import SimpleNamespace

import libcst as cst

from modelx_cython import native_math
from modelx_cython.helper_members import MATH_PROVIDER, module_helpers
from modelx_cython.helpers import HelperCalls
from modelx_cython.rounding import NativeRounding


def test_native_math_provider_and_inherited_unary_calls():
    refs = {'math': SimpleNamespace(is_math=True, helper_members=module_helpers(math))}
    plain_unary = ', '.join(f'self.math.{name}(x)' for name in native_math.NAMES)
    source = f'''
def formula(self, x):
    return ({plain_unary}, self.math.floor(x),
            self.math.ceil(self.other(x)), self.math.inf,
            self.math.isclose(x, 0.0, abs_tol=1e-9), self.math.log(x, 2.0))
'''
    cell = SimpleNamespace(
        params=('x',), has_typeinfo=lambda: True, get_argtype_expr=lambda *args, **kwargs: 'double',
        model_path='Values.formula',
    )
    used = set()
    binder = HelperCalls(refs, used, current=cell)
    formula = binder.rewrite(cst.parse_module(source).body[0])
    expected = {'m_floor_ll', 'm_ceil', 'm_constant', 'm_isclose'}
    assert binder.operations == {f'{MATH_PROVIDER}.{name}' for name in expected}
    assert used == {MATH_PROVIDER}
    result, count = native_math.lower(formula, SimpleNamespace(refs=refs), cell)
    assert count == len(native_math.NAMES)
    code = cst.Module(body=(result,)).code
    for name in native_math.NAMES:
        assert f'_mx_sys._mx_math_{name}(' in code
    assert 'self.math.log(x, 2.0)' in code
    assert any('2 arguments' in entry['reason'] for entry in binder.math_untyped)


def test_native_rounding_keeps_the_authored_callables_and_nested_scope():
    source = '''
def formula(self, x, digits):
    return (round(x), round(x, digits), lambda round: round(x))
'''
    formula = cst.parse_module(source).body[0]
    rounding = NativeRounding()
    result = formula.with_changes(body=formula.body.visit(rounding))
    code = cst.Module(body=(result,)).code
    assert rounding.used
    assert '_mx_sys._mx_round_one(round, x, _mx_canonical_round)' in code
    assert '_mx_sys._mx_round_digits(round, x, digits, _mx_canonical_round)' in code
    assert 'lambda round: round(x)' in code


def test_native_isclose_records_the_authored_keyword_order():
    refs = {'math': SimpleNamespace(is_math=True, helper_members=module_helpers(math))}
    for keywords, mask in (
        ('rel_tol=1e-8, abs_tol=1e-9', 3),
        ('abs_tol=1e-9, rel_tol=1e-8', 7),
    ):
        source = f'def formula(self, a, b):\n    return self.math.isclose(a, b, {keywords})\n'
        binder = HelperCalls(refs, set())
        formula = binder.rewrite(cst.parse_module(source).body[0])
        code = cst.Module(body=(formula,)).code
        assert f'.m_isclose(_mx_cy.cast(_mx_cy.p_void, self.math), a, b, 1e-8, 1e-9, {mask})' in code
