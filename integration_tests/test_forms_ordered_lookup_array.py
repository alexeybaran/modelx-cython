"""An ordered lookup (``searchsorted``, ``asof``) whose argument holds several values is pandas' own and returns an
array as live; the native lookup takes a single number. A C integer argument needs no test; any other is looked up
natively only while it is one label at run time."""
from forms_probe import compare, lowered

FRAMES = {
    'T': 'pd.DataFrame({"q": [0.1, 0.2, 0.3, 0.4]}, index=pd.Index([18, 19, 20, 22]))',
    'G': 'pd.DataFrame({"q": [1.0, 2.0, 3.0]}, '
         'index=pd.MultiIndex.from_product([["A"], [2**53, 2**53 + 1, 2**53 + 3]]))',
}
CELLS = {
    'many': ('t', 'x = np.array([19, 20 + t])\nreturn data.T().index.searchsorted(x).tolist()'),
    'one': ('t', 'x = 18.0 + t if policy_id else None\nreturn int(data.T().index.searchsorted(x))'),
    'below': ('t', 'x = 18.5 + t if policy_id else None\nreturn int(data.T().index.asof(x))'),
    'typed': ('t', 'return int(data.T().index.searchsorted(18 + t))'),
    'typed_key': ('t', 'return 2**53 + t if policy_id < 4 else 2**53 - 1'),
    'view_typed': ('t', 'sub = data.G().loc["A"]\n'
                       'return float(sub.loc[sub.index.asof(typed_key(t)), "q"])'),
    'search_pop': ('t', 'values = [22.0, np.array([19, 20 + t])] if policy_id >= 3 else [22.0, 18.0 + t]\n'
                       'result = data.T().index.searchsorted(values.pop())\n'
                       'return len(values), str(result)'),
    'asof_pop': ('t', 'values = [22.0, 17.0 if policy_id >= 3 else 18.5 + t]\n'
                     'result = data.T().index.asof(values.pop())\n'
                     'return len(values), str(result)'),
    'int_asof_pop': ('t', 'values = [22.0, float("nan") if policy_id >= 3 else 18.5 + t]\n'
                         'try:\n'
                         '    result = int(data.T().index.asof(values.pop()))\n'
                         'except (TypeError, ValueError):\n'
                         '    result = None\n'
                         'return len(values), result'),
}


def test_an_ordered_lookup_of_several_values_is_pandas(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 3))
    assert not failures, '\n'.join(failures)
    assert lowered(reads, 'one', 'searchsorted(x)'), reads
    assert lowered(reads, 'below', 'asof(x)'), reads
    assert lowered(reads, 'typed', 'searchsorted(18 + t)'), reads
    assert lowered(reads, 'view_typed', 'sub.index.asof(self.typed_key(t))'), reads
    assert not lowered(reads, 'search_pop', 'searchsorted('), reads
    assert not lowered(reads, 'asof_pop', 'asof('), reads
    assert not lowered(reads, 'int_asof_pop', 'asof('), reads
