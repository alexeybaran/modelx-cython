"""A frame an item computes itself is read natively by a formula with parameters, whatever the number of times the
sample evaluated it, and on pandas by a formula without parameters (decided from the text, not from call counts)."""
from forms_probe import compare, native_reads

FRAMES = {'T1': 'pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0]}, index=pd.Index([1, 2, 3, 4], name="pid"))'}
CELLS = {
    'own': ('', 'return data.T1() * policy_id'),
    'step': ('t', 'return own().loc[policy_id, "sa"] * (t + 1)'),
    'once': ('', 'return own().loc[policy_id, "sa"]'),
}


def test_own_frame_is_prepared_by_text_not_by_sample_counts(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert native_reads(source, 'step') == ['f64_1'], source
    assert native_reads(source, 'once') == [], source
    assert not failures, '\n'.join(failures)
