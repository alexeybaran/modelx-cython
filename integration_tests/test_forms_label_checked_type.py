"""A read whose key part is checked per call for one label (a text key, a column named at run time) keeps the type
live has where the type shows: in a container, under ``type()``, ``round()``. Its native value is the numpy scalar
pandas returns there, as for any other native read; under ``float()`` it stays a C double."""
from forms_probe import compare, lowered

FRAMES = {'TS': 'pd.DataFrame({"sa": [1.25, 2.5, 3.75], "n": [1, 2, 3]}, index=pd.Index(["a", "b", "c"]))'}
CELLS = {
    'code': ('', 'return "abc"[policy_id % 3]'),
    'pair': ('', 'k = code()\nreturn (data.TS().loc[k, "sa"], 0)'),
    'kind': ('', 'k = code()\nreturn type(data.TS().loc[k, "n"]).__name__'),
    'plain': ('', 'k = code()\nreturn float(data.TS().loc[k, "sa"]) * 2'),
}


def test_a_label_checked_read_has_numpy_type_where_it_shows(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3), sample_ids=(1, 2, 3))
    assert not failures, '\n'.join(failures)
    for cell, text in (('pair', 'loc[k, "sa"]'), ('kind', 'loc[k, "n"]'), ('plain', 'loc[k, "sa"]')):
        assert lowered(reads, cell, text), (cell, reads)
