import ast

import libcst as cst

from modelx_cython.ordered_arguments import preserve_argument_order


def rewrite(source, native_result=True):
    return cst.Module(body=(preserve_argument_order(cst.parse_module(source).body[0],
                                                 native_result=native_result),)).code


def test_builtin_extrema_remain_eligible_for_cython_intrinsics():
    assert 'max(first(), second(), third(), key=key())' in rewrite(
        'def formula():\n    return max(first(), second(), third(), key=key())\n')
    assert 'min(first(), second())' in rewrite(
        'def formula():\n    return min(first(), second())\n')


def test_shadowed_and_iterable_extrema_are_unchanged():
    for source in ['def formula(max):\n    return max(first(), second())\n',
                   'def formula():\n    return max(values, key=key())\n',
                   'def formula():\n    return max(*values)\n',
                   'def formula():\n    return self.max(first(), second())\n']:
        assert rewrite(source) == source


def test_binding_is_lexical_inside_callbacks_and_never_parses_ast(monkeypatch):
    source = cst.parse_module('def formula():\n    return lambda max: (max(first(), second()), min(first(), second()))\n').body[0]
    def forbidden(*args, **kwargs): raise AssertionError('AST reduction argument parsing')
    with monkeypatch.context() as patch:
        patch.setattr(ast, 'parse', forbidden)
        result = cst.Module(body=(preserve_argument_order(source),)).code
    assert 'max(first(), second())' in result
    assert 'min(*(first(), second()))' in result


def test_object_result_keeps_selected_mixed_numeric_scalar_class():
    source = rewrite('def formula(a, b):\n    return min(a, b)\n', native_result=False)
    assert 'min(*(a, b))' in source


def test_remaining_power_keeps_python_types_and_ignores_pow_binding():
    import cython
    source = rewrite('def formula(n):\n    return 2 ** -n\n')
    namespace = {'_mx_cy': cython, 'pow': lambda *args: object()}
    exec(source, namespace)
    for n in (-3, 0, 1, 10):
        expected = 2 ** -n
        actual = namespace['formula'](n)
        assert actual == expected and type(actual) is type(expected)


def test_power_does_not_capture_shadowed_generated_or_object_names():
    for source in ['def formula(object, n):\n    return 2 ** -n\n',
                   'def formula(_mx_cy, n):\n    return 2 ** -n\n']:
        assert rewrite(source) == source


def test_native_result_boxes_unproved_fractional_power_and_preserves_integer_exponents():
    from modelx_cython.powers import OperandKind, KIND_INT
    import cython
    classifier = OperandKind(None, {}, {}, '', {'n': KIND_INT})
    source = 'def formula(x, n):\n    q = x ** 0.5\n    return min(1.0, q) + x ** n\n'
    formula = preserve_argument_order(cst.parse_module(source).body[0],
                                     native_result=True, operand_kind=classifier)
    code = cst.Module(body=(formula,)).code
    assert '_mx_cy.cast(object, x) ** _mx_cy.cast(object, 0.5)' in code
    assert 'x ** n' in code
    namespace = {'_mx_cy': cython}
    exec(code, namespace)
    assert namespace['formula'](4.0, -1) == 1.25
    import pytest
    with pytest.raises(TypeError):
        namespace['formula'](-4.0, 1)


def test_native_return_keeps_existing_cython_conversion_boundary():
    from modelx_cython.powers import OperandKind
    classifier = OperandKind(None, {}, {}, '', {})
    source = 'def formula(x, y):\n    return -2.0 * (x - 1.0) ** y\n'
    result = preserve_argument_order(cst.parse_module(source).body[0],
                                    native_result=True, operand_kind=classifier)
    assert cst.Module(body=(result,)).code == source
