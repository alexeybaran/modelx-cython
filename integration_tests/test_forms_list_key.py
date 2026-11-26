"""A key part on a text level that is a list at run time (not in the sample) is pandas' selection, never a lookup."""
from forms_probe import compare, native_reads

FRAMES = {'TS': 'pd.DataFrame({"sa": [1.0, 2.0, 3.0]}, index=pd.Index(["a", "b", "c"]))'}
CELLS = {'pick': ('', 'k = ["a", "b"] if policy_id > 1 else "a"\nreturn float(np.sum(data.TS().loc[k, "sa"]))')}


def test_unsampled_list_key_on_a_text_level(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2), sample_ids=(1,))
    assert native_reads(source, 'pick') == ['f64_1'], source
    assert '.is_label(' in source
    assert not failures, '\n'.join(failures)
