"""An ordered lookup (``searchsorted``, ``asof``) whose argument holds several values is pandas' own and returns an
array as live; the native lookup takes a single number. A C integer argument needs no test; any other is looked up
natively only while it is one label at run time."""
from forms_probe import compare, lowered

FRAMES = {'T': 'pd.DataFrame({"q": [0.1, 0.2, 0.3, 0.4]}, index=pd.Index([18, 19, 20, 22]))'}
CELLS = {
    'many': ('t', 'x = np.array([19, 20 + t])\nreturn data.T().index.searchsorted(x).tolist()'),
    'one': ('t', 'x = 18.0 + t if policy_id else None\nreturn int(data.T().index.searchsorted(x))'),
    'below': ('t', 'x = 18.5 + t if policy_id else None\nreturn int(data.T().index.asof(x))'),
}


def test_an_ordered_lookup_of_several_values_is_pandas(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 3))
    assert not failures, '\n'.join(failures)
    assert lowered(reads, 'one', 'searchsorted(x)'), reads
    assert lowered(reads, 'below', 'asof(x)'), reads
