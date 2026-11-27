"""``k in sub.index`` of a prefix view is native only while each key part names one label: a Series, an array or a
list there is pandas' own membership, which raises (unhashable) or answers as live."""
from forms_probe import compare, lowered

FRAMES = {
    'G': 'pd.DataFrame({"q": [0.01 * n for n in range(8)]}, '
         'index=pd.MultiIndex.from_product([["A", "B"], [20, 21, 22, 23]], names=["plan", "age"]))',
    'H': 'pd.DataFrame({"q": range(8)}, '
         'index=pd.MultiIndex.from_product([["A", "B"], [20, 21], [0, 1]]))',
}
CELLS = {
    'plain': ('t', 'sub = data.G().loc[("A", "B")[t % 2]]\nreturn 1.0 if 20 + t in sub.index else 0.0'),
    'odd': ('t', 'sub = data.G().loc[("A", "B")[t % 2]]\nk = 20 + t if policy_id < 3 else np.array([20, 21])\n'
                 'return 1.0 if k in sub.index else 0.0'),
    'pop_key': ('t', 'sub = data.G().loc["A"]\n'
                     'keys = [21, np.array([20, 21])] if policy_id >= 3 else [21, 20 + t]\n'
                     'try:\n'
                     '    member = keys.pop() in sub.index\n'
                     'except TypeError:\n'
                     '    member = None\n'
                     'return len(keys), member'),
    'tuple_key': ('t', 'sub = data.H().loc["A"]\n'
                       'keys = [21, 20]\n'
                       'key = (keys.pop(), t % 2)\n'
                       'member = key in sub.index\n'
                       'return len(keys), member'),
}


def test_a_view_membership_of_several_values_is_pandas(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2))
    assert not failures, '\n'.join(failures)
    assert lowered(reads, 'plain', 'in sub.index'), reads
    assert lowered(reads, 'odd', 'in sub.index'), reads
    assert not lowered(reads, 'pop_key', 'in sub.index'), reads
    assert not lowered(reads, 'tuple_key', 'in sub.index'), reads
