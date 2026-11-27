"""A reduction inside a call an earlier lowering put in (a typed vector read) is not lifted out of it."""
from reduction_cases import assert_equal_live

CELLS = {
    'fl': 'lambda t: t * 0.5 + 0.25',
    'n': 'lambda t: t',
    'vec': 'lambda t: [0.1 * i for i in range(t + 2)]',
    'index_is_a_reduction': 'lambda t: vec(t + 1)[min(i for i in range(t, t + 2))]',
    'argument_is_a_reduction': 'lambda t: vec(sum(1 for i in range(t)))[0 if t < 2 else 1]',
    'beside_a_vector_read': 'lambda t: vec(t + 1)[t] * sum(fl(i) for i in range(t))',
    'two_in_one_call': 'lambda t: divmod(sum(n(i) for i in range(t)), max(n(i) for i in range(1, t + 2)))[1]',
}


def test_reduction_inside_a_typed_vector_read(tmp_path):
    live = assert_equal_live(tmp_path, CELLS, list(CELLS)[3:], range(6))
    assert live['index_is_a_reduction', 1] == ('float', '0.1')
