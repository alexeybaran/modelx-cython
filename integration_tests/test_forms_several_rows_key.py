"""A key part that is a list, a mask or a slice at run time is pandas' selection, never a lookup: whether the sample
saw it or not, held in a name or chosen by a condition. The read is lowered with a run-time test that the part is one
label; any other value runs the original pandas expression."""
from forms_probe import compare, native_reads

FRAMES = {'TI': 'pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0]}, index=pd.Index([1, 2, 3, 4], name="pid"))'}
CELLS = {
    # a label or a mask by one expression, both reached by the sample (t = 0..2)
    'masked': ('t', 'k = data.TI()["sa"] > 150.0 if t == 1 else policy_id\n'
                    'return float(np.sum(data.TI().loc[k, "sa"]))'),
    # a mask only for policies the sample does not run
    'late_mask': ('', 'k = (data.TI()["sa"] > 150.0) if policy_id > 2 else policy_id\n'
                      'return float(np.sum(data.TI().loc[k, "sa"]))'),
    # a list, a slice and a mask held in a name
    'listed': ('t', 'k = [1, policy_id]\nreturn float(np.sum(data.TI().loc[k, "sa"])) + t'),
    'sliced': ('t', 'k = slice(1, policy_id)\nreturn float(np.sum(data.TI().loc[k, "sa"])) + t'),
    'mask_name': ('t', 'k = data.TI()["sa"] < 100.0 * policy_id\nreturn float(np.sum(data.TI().loc[k, "sa"])) + t'),
}


def test_several_rows_keys_are_pandas_selections(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2))
    assert not failures, '\n'.join(failures)
    for cell in CELLS:
        assert native_reads(source, cell) == ['f64_1'], (cell, source)
    assert source.count('.is_label(') >= len(CELLS)
