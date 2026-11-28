"""A column part that selects several columns is pandas' own selection, never a read of one column: a list display
(``T.loc[k, ['q']]``) stays on pandas from its text, and a column name computed at run time is read natively only
while it names one label (a list held in a local is pandas' selection, as live). The sample reaches only one-label names; the list is read by the run-time guard."""
from forms_probe import compare, lowered

FRAMES = {'F': 'pd.DataFrame({"q": [0.1, 0.2, 0.3, 0.4], "r": [1.0, 2.0, 3.0, 4.0]}, '
               'index=pd.Index([20, 21, 22, 23]))'}
CELLS = {
    'shown': ('t', 'return data.F().loc[20 + t % 4, ["q"]]'),
    'held': ('t', 'c = "q" if policy_id < 3 else ["q", "r"]\nreturn (data.F().loc[20 + t % 4, c], 0)'),
    'named': ('t', 'c = "q" if t % 2 else "r"\nreturn data.F().loc[20 + t % 4, c]'),
}


def test_a_column_selection_is_pandas(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2))
    assert not failures, '\n'.join(failures)
    assert not lowered(reads, 'shown', 'loc[20 + t % 4, ["q"]]'), reads
    assert lowered(reads, 'named', 'loc[20 + t % 4, c]'), reads
    assert lowered(reads, 'held', 'loc[20 + t % 4, c]'), reads     # native while c is one label
