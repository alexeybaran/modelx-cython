"""A key part held per item is evaluated only where live evaluates it: a hold that raises does not raise on a branch
that does not read the table."""
from forms_probe import compare, lowered

FRAMES = {'T2': 'pd.DataFrame({"rate": np.arange(12, dtype=float) / 100}, '
                'index=pd.MultiIndex.from_product([["a", "b", "c"], range(4)]))'}
CELLS = {
    'gcode': ('', 'if policy_id == 3:\n    raise ValueError("no code")\nreturn "b"'),
    'rate': ('t', 'if policy_id != 3:\n    return data.T2().loc[(gcode(), t % 4), "rate"]\nreturn 0.0'),
}


def test_hold_raising_on_a_branch_not_taken(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2))
    assert lowered(reads, 'rate', 'loc[(self.gcode(), t % 4), "rate"]'), reads
    assert not failures, '\n'.join(failures)
