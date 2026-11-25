"""A view local whose every read is lowered keeps live's KeyError for an absent prefix.

``v = T.loc[policy_id]`` raises KeyError in live for a policy the table does not hold, even when no read of ``v``
runs afterwards. The assignment is dropped once every use of ``v`` is lowered; the compiled formula must still raise.
"""
from forms_probe import compare, lowered

FRAMES = {'M': 'pd.DataFrame({"c": np.arange(9, dtype=float) + 0.5}, '
               'index=pd.MultiIndex.from_product([[1, 2, 3], range(3)], names=["pid", "t"]))'}
CELLS = {'view_cond': ('', 'v = data.M().loc[policy_id]\nif policy_id < 3:\n    return v.loc[1, "c"]\nreturn 0.0')}


def test_absent_prefix_of_a_dropped_view_raises_as_live(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2, 3))
    assert lowered(reads, 'view_cond', 'v.loc[1, "c"]'), reads
    assert not failures, '\n'.join(failures)
