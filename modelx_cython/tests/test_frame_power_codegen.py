"""Eligibility tests for numeric power lowering in the native frame."""
import ast

import pytest

from modelx_cython._frame_runtime.lookups import IndexedFloat1
from modelx_cython._frame_runtime.power_codegen import lower_numeric_powers


def _target(t: int) -> float:
    return 0.0


def _rate(t: int) -> float:
    return 0.0


def _object(t: int) -> object:
    return None


def _lower(source, inputs=None):
    node = ast.parse(source).body[0]
    return lower_numeric_powers(node, {'target': _target, 'rate': _rate,
                                        'object_value': _object}, inputs or {})


@pytest.mark.parametrize('expression', [
    '(1 + rate(t)) ** (1 / 12)',
    '(1 + rates.get(t // 12)) ** (1 / 12)',
    '(1 - rate(t)) ** (t / 12)',
    '(1 + rate(t)) ** 0.5',
])
def test_numeric_fractional_power_is_lowered(expression):
    node, count = _lower('def target(t):\n    return ' + expression,
                         {'rates': IndexedFloat1})
    assert count == 1
    assert '_power_compatible(' in ast.unparse(node)


@pytest.mark.parametrize('expression', [
    '(1 + rate(t)) ** -t',
    'object_value(t) ** (1 / 12)',
    'unknown ** (1 / 12)',
])
def test_integer_exponent_and_unknown_operands_stay_unchanged(expression):
    node, count = _lower('def target(t):\n    return ' + expression)
    assert count == 0
    assert '**' in ast.unparse(node)


def test_shadowed_cell_name_is_not_treated_as_a_typed_cell():
    node, count = _lower('def target(t):\n    rate = object_value(t)\n'
                         '    return rate ** (1 / 12)')
    assert count == 0
    assert '**' in ast.unparse(node)


def test_nested_numeric_powers_are_rewritten_once_each():
    node, count = _lower('def target(t):\n'
                         '    return ((1 + rate(t)) ** 0.5) ** (1 / 12)')
    assert count == 2
    assert ast.unparse(node).count('_power_compatible(') == 2
