"""``k in sub.index`` of a prefix view is native only while each key part names one label: a Series, an array or a
list there is pandas' own membership, which raises (unhashable) or answers as live."""
from forms_probe import compare, lowered

FRAMES = {'G': 'pd.DataFrame({"q": [0.01 * n for n in range(8)]}, '
               'index=pd.MultiIndex.from_product([["A", "B"], [20, 21, 22, 23]], names=["plan", "age"]))'}
CELLS = {
    'plain': ('t', 'sub = data.G().loc[("A", "B")[t % 2]]\nreturn 1.0 if 20 + t in sub.index else 0.0'),
    'odd': ('t', 'sub = data.G().loc[("A", "B")[t % 2]]\nk = 20 + t if policy_id < 3 else np.array([20, 21])\n'
                 'return 1.0 if k in sub.index else 0.0'),
}


def test_a_view_membership_of_several_values_is_pandas(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2))
    assert not failures, '\n'.join(failures)
    assert lowered(reads, 'plain', 'in sub.index'), reads
    assert lowered(reads, 'odd', 'in sub.index'), reads
