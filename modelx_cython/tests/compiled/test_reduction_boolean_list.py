"""all([...]) and any([...]) evaluate every element and then test them in order, stopping at the deciding one."""
import numpy as np

from reduction_cases import assert_equal_live

CELLS = {
    'n': 'lambda t: t',
    'boom': 'lambda t: {}[t]',
    # an array of several items has no truth value: live never asks for it after the deciding element
    'all_stops': 'lambda t: all([n(t) > 1, arr * t])',
    'any_stops': 'lambda t: any([n(t) > 1, arr * t])',
    'all_tuple': 'lambda t: all([v for v in (n(t) - 2, arr)])',
    'any_filter': 'lambda t: any([arr if i else n(t) for i in range(2)])',
    # every element is still evaluated before any is tested
    'all_evaluates': 'lambda t: all([n(i) < 2 or boom(i) for i in range(t)])',
    'all_generator': 'lambda t: all(v for v in (n(t) - 2, arr))',
}


def test_boolean_reductions_over_a_list(tmp_path):
    live = assert_equal_live(tmp_path, CELLS, list(CELLS)[2:], range(5), refs={'arr': np.array([1.0, 2.0])})
    assert live['all_stops', 0] == ('bool', 'False') and live['all_stops', 3][0] == 'raise'
    assert live['any_stops', 3] == ('bool', 'True') and live['any_stops', 0][0] == 'raise'
