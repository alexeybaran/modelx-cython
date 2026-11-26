"""A frame an item computes itself is read natively only by a formula with parameters that reads it by a computed key,
whatever the number of times the sample evaluated it; by literal keys, and in a formula without parameters, it is read
on pandas (decided from the text, not from call counts). A frame of a Space without items is prepared once per model."""
from forms_probe import compare, native_reads

FRAMES = {'T1': 'pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0]}, index=pd.Index([1, 2, 3, 4], name="pid"))'}
CELLS = {
    'own': ('', 'return data.T1() * policy_id'),
    'step': ('t', 'return own().loc[policy_id, "sa"] * (t + 1)'),
    'once': ('', 'return own().loc[policy_id, "sa"]'),
    'shared': ('t', 'return data.T1().loc[policy_id, "sa"] * (t + 1)'),
    'row': ('', 'return own().loc[policy_id]'),
    'field': ('t', 'return row()["sa"] * t if t == 0 else 0.0'),
}


def test_own_frame_is_prepared_by_text_not_by_sample_counts(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    assert native_reads(source, 'step') == ['f64_1'], source     # a key computed per item, in a formula of t
    assert native_reads(source, 'once') == [], source
    assert native_reads(source, 'shared') == ['f64_1'], source     # a Data frame: prepared once per model
    assert native_reads(source, 'field') == [], source     # a row Series the item computes: not prepared per item


CELLS_KEYED = {
    'own': ('', 'return data.T1() * policy_id'),
    'row': ('', 'return own().loc[policy_id]'),
    'field': ('t', 'return row()["sa"] * t if t == 0 else 0.0'),
    'chosen': ('t', 'return row()["sa" if t % 2 else "sa"] * t'),
    'computed': ('t', 'return own().loc[1 + t % 4, "sa"] * (t + 1)'),
    'local': ('t', 'tbl = own()\nreturn tbl.loc[1 + t % 4, "sa"] - tbl.loc[1, "sa"]'),
}


def test_own_frame_prepared_only_for_computed_keys(tmp_path):
    """Literal keys (or a choice between literals) read a few values per item: pandas. A key computed in a formula
    with parameters ranges over the steps: the item's own frame is prepared for it."""
    source, failures = compare(tmp_path, FRAMES, CELLS_KEYED, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    assert native_reads(source, 'field') == [], source
    assert native_reads(source, 'chosen') == [], source
    assert native_reads(source, 'computed') == ['f64_1'], source
    assert native_reads(source, 'local') == ['f64_1', 'f64_1'], source
