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


def test_extrema_in_object_scopes_stay_the_builtin_call():
    """min/max in object-result scopes call the built-in (the selected operand keeps its class);
    a native-result Cell keeps Cython's intrinsic, as before."""
    from types import SimpleNamespace
    from modelx_cython.powers import OperandKind, KIND_FLOAT, KIND_INT
    kinds = OperandKind(None, {}, {}, '', {'x': KIND_FLOAT, 'y': KIND_FLOAT, 'i': KIND_INT})

    def lowered(source, native_result=False):
        cells, providers = SimpleNamespace(), set()
        node = preserve_argument_order(cst.parse_module(source).body[0], native_result=native_result,
                                       operand_kind=kinds, cells=cells, providers=providers, math_refs={'math'})
        return cst.Module(body=(node,)).code, cells, providers

    code, cells, providers = lowered('def formula(x, y):\n    return max(0.0, x, y)\n')
    assert 'max(*(0.0, x, y))' in code and not providers
    assert [(e['form'], e['line']) for e in cells.minmax_untyped] == [('max', 2)]
    code, cells, _ = lowered('def formula(i, x):\n    return min(i, x)\n')
    assert 'min(*(i, x))' in code
    code, cells, _ = lowered('def formula(x):\n    return min(x, 1.0 - 1e-12)\n')
    assert 'return min(*(x, 1.0 - 1e-12))' in code
    code, cells, _ = lowered('def formula(self, x):\n    return [self.math.log(1.0 - min(x, 1.0 - 1e-12))]\n')
    assert '1.0 - min(x, 1.0 - 1e-12)' in code and cells.minmax_untyped == []
    code, cells, _ = lowered('def formula(i):\n    return min(i, 1.0)\n')
    assert 'min(*(i, 1.0))' in code
    code, cells, _ = lowered('def formula(x, y):\n    return [max(x, y) for _ in range(2)]\n', True)
    assert 'max(*(x, y))' in code           # a child scope of a native-result cell
    code, cells, _ = lowered('def formula(x, y):\n    return max(x, y)\n', True)
    assert 'return max(x, y)' in code and cells.minmax_untyped == []   # Cython's intrinsic, as before


def test_integer_local_exponent_keeps_the_native_power():
    """diag-us (SPIA/DIA mort_rate): ``k = calendar_year(t) - year`` then ``b ** k``: with the
    formula's local kinds the exponent is an integer, as the inline expression was."""
    from modelx_cython.powers import OperandKind, KIND_INT, KIND_FLOAT
    source = 'def formula(x, n):\n    k = n - 3\n    q = x * (1.0 - x) ** k\n    return min(1.0, q)\n'
    boxed = preserve_argument_order(cst.parse_module(source).body[0], native_result=True,
                                    operand_kind=OperandKind(None, {}, {}, '', {'n': KIND_INT}))
    assert '_mx_cy.cast(object, (1.0 - x)) ** _mx_cy.cast(object, k)' in cst.Module(body=(boxed,)).code
    native = preserve_argument_order(cst.parse_module(source).body[0], native_result=True,
                                     operand_kind=OperandKind(None, {}, {}, '', {'n': KIND_INT, 'k': KIND_INT,
                                                                                'x': KIND_FLOAT}))
    assert '(1.0 - x) ** k' in cst.Module(body=(native,)).code


def test_int_of_a_proven_integer_is_the_integer():
    """diag-us (DIA_US_S annuity_factor): ``n = int((last - start) // step)`` keeps ``n`` a C
    integer; a float, an unknown operand, a shadowed ``int`` or a comprehension keep the call."""
    from modelx_cython.powers import OperandKind, KIND_INT, KIND_FLOAT
    kinds = OperandKind(None, {}, {}, '', {'a': KIND_INT, 'b': KIND_INT, 'x': KIND_FLOAT})

    def code(source):
        formula = preserve_argument_order(cst.parse_module(source).body[0], native_result=False, operand_kind=kinds)
        return cst.Module(body=(formula,)).code

    out = code('def f(a, b, x, y):\n    n = int((a - b) // 3) + 1\n    return n\n')
    assert 'n = ((a - b) // 3) + 1' in out, out
    assert 'int(x)' in code('def f(a, b, x, y):\n    return int(x)\n')
    assert 'int(y)' in code('def f(a, b, x, y):\n    return int(y)\n')
    assert 'int(a)' in code('def f(a, b, x, y):\n    int = abs\n    return int(a)\n')
    assert 'int(a)' in code('def f(a, b, x, y):\n    return [int(a) for a in y]\n')
    namespace = {}
    exec(out, namespace)
    assert namespace['f'](10, 2, 0.5, None) == 3 and type(namespace['f'](10, 2, 0.5, None)) is int
