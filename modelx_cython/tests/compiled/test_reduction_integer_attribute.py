"""An integer literal substituted for a generator variable keeps its meaning before an attribute."""
from reduction_cases import assert_equal_live

CELLS = {
    'n': 'lambda t: t',
    'fl': 'lambda t: t * 0.5 + 0.25',
    'attribute_of_integer': 'lambda t: sum(x.real for x in (1, 2, t))',
    'attribute_of_negative': 'lambda t: max(x.real for x in (-1, t))',
    'method_of_integer': 'lambda t: sum(x.bit_length() for x in (1, 255, t))',
}


def test_attribute_of_a_substituted_integer(tmp_path):
    live = assert_equal_live(tmp_path, CELLS, list(CELLS)[2:], range(6))
    assert live['attribute_of_integer', 4] == ('int', '7')
