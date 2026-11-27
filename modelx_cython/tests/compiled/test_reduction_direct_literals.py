"""A literal tuple or a range given directly to a reduction."""
from reduction_cases import assert_equal_live

CELLS = {
    'n': 'lambda t: t',
    'fl': 'lambda t: t * 0.5 + 0.25',
    'max_pair': 'lambda t: max((n(t), 2))',
    'min_three': 'lambda t: min((fl(t), 1.0, fl(t + 1)))',
    'sum_pair': 'lambda t: sum((fl(t), n(t)))',
    'any_pair': 'lambda t: any((n(t) > 3, n(t) == 1))',
    'sum_range': 'lambda t: sum(range(t))',
    'max_nested': 'lambda t: n(t) + max((fl(t), 2.0))',
}


def test_direct_literal_iterables(tmp_path):
    live = assert_equal_live(tmp_path, CELLS, list(CELLS)[2:], range(6))
    assert live['max_pair', 5] == ('int', '5')
