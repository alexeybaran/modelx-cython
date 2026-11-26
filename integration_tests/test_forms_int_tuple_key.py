"""A key that Cython types as a C integer is never a key tuple: on an index of several levels its read is pandas'
own (which raises or selects as live), and no tuple test subscripts it, so the formula compiles."""
from forms_probe import compare, lowered

FRAMES = {'G': 'pd.DataFrame({"n": np.arange(8, dtype=float)}, '
               'index=pd.MultiIndex.from_product([[21, 22], [1, 2, 3, 4]]))'}
CELLS = {
    'whole': ('t', 'k1 = 21 + t % 2\nreturn data.G().at[(k1, 1 + t % 4), "n"]'),
    'single': ('t', 'k1 = 21 + t % 2\nreturn float(data.G().loc[k1, "n"].sum())'),
}


def test_a_c_integer_key_on_several_levels_compiles_and_reads_as_live(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2), sample_ids=(1, 2))
    assert not failures, '\n'.join(failures)
    assert lowered(reads, 'whole', 'at[(k1, 1 + t % 4), "n"]'), reads
    assert not lowered(reads, 'single', 'loc[k1, "n"]'), reads
