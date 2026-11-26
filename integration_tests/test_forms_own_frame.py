"""A frame an item computes itself is read natively only by a formula with parameters that reads it by a computed key,
whatever the number of times the sample evaluated it; by literal keys (or a choice between literals), and in a formula
without parameters, it is read on pandas (decided from the text, not from call counts). A frame of a Space without
items is prepared once per model."""
from forms_probe import compare, lowered

FRAMES = {'T1': 'pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0]}, index=pd.Index([1, 2, 3, 4], name="pid"))'}
CELLS = {
    'own': ('', 'return data.T1() * policy_id'),
    'step': ('t', 'return own().loc[policy_id, "sa"] * (t + 1)'),
    'once': ('', 'return own().loc[policy_id, "sa"]'),
    'shared': ('t', 'return data.T1().loc[policy_id, "sa"] * (t + 1)'),
}


def test_own_frame_is_prepared_by_text_not_by_sample_counts(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert lowered(reads, 'step', 'own().loc[self.policy_id, "sa"]'), reads
    assert not lowered(reads, 'once', 'own().loc[self.policy_id, "sa"]'), reads
    assert lowered(reads, 'shared', 'T1().loc[self.policy_id, "sa"]'), reads
    assert not failures, '\n'.join(failures)


CELLS_KEYED = {
    'own': ('', 'return data.T1() * policy_id'),
    'ownrow': ('', 'return data.T1().loc[policy_id] * 2'),
    'chosen': ('t', 'return ownrow()["sa" if t % 2 else "sa"] * t'),
    'literal': ('t', 'return own().loc[1, "sa"] * t'),
    'computed': ('t', 'return own().loc[1 + t % 4, "sa"] * (t + 1)'),
    'local': ('t', 'tbl = own()\nreturn tbl.loc[1 + t % 4, "sa"] - tbl.loc[1, "sa"]'),
}


def test_own_frame_prepared_only_for_computed_keys(tmp_path):
    """Literal keys (or a choice between literals) read a few values per item: pandas, with no preparation of the
    item's own frame. A key computed in a formula with parameters ranges over the steps: the frame is prepared."""
    reads, failures = compare(tmp_path, FRAMES, CELLS_KEYED, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    assert not lowered(reads, 'chosen', 'ownrow()["sa" if t % 2 else "sa"]'), reads
    assert not lowered(reads, 'literal', 'own().loc[1, "sa"]'), reads
    assert lowered(reads, 'computed', 'own().loc[1 + t % 4, "sa"]'), reads
    assert lowered(reads, 'local', 'tbl.loc[1 + t % 4, "sa"]'), reads
