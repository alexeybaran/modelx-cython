"""A keyword argument named as the generator variable keeps its name in a lowered reduction."""
from reduction_cases import assert_equal_live

CELLS = {
    'fl': 'lambda t: t * 0.5 + 0.25',
    'over_range': 'lambda t: sum(fl(t=t) for t in range(t))',
    'over_tuple': 'lambda t: sum(fl(t=t) for t in (1, 2))',
    'in_minimum': 'lambda t: min(fl(t=i) + fl(t=t) for i in range(t + 1))',
    'in_all': 'lambda t: all(fl(t=t) > 0.0 for t in (t, 3))',
}


def test_keyword_named_as_the_loop_variable(tmp_path):
    live = assert_equal_live(tmp_path, CELLS, [name for name in CELLS if name != 'fl'], range(5))
    assert live['over_range', 3] == ('float', '2.25')
    assert live['over_tuple', 0] == ('float', '2.0')
