"""Ordered lookups ``X.index.asof(x)`` read per step on an int level: as a key, under ``int()`` and as a bare value
(the numpy integer pandas returns), on a whole table and on a prefix. Equal to live, also where pandas answers
itself: x below the first label (NaN), x NaN, x an array (pandas raises its own error), a prefix without rows, an index that is not
in increasing order (pandas raises ValueError)."""
from forms_probe import compare, native_reads

FRAMES = {
    'T1': 'pd.DataFrame({"q": [0.1, 0.2, 0.3, 0.4]}, index=pd.Index([18, 19, 20, 22], name="age"))',
    'TU': 'pd.DataFrame({"q": [0.1, 0.2, 0.3]}, index=pd.Index([22, 18, 20], name="age"))',
    'T2': 'pd.DataFrame({"r": np.arange(8, dtype=float)}, '
          'index=pd.MultiIndex.from_product([["M", "F"], [1, 2, 3, 5]], names=["sex", "year"]))',
}
CELLS = {
    'sex': ('', 'return "M" if policy_id % 2 else "F"'),
    'as_key': ('t', 'tbl = data.T1()\nreturn tbl.loc[tbl.index.asof(17.5 + 1.5 * t), "q"]'),
    'as_int': ('t', 'return int(data.T1().index.asof(19 + t)) * 2'),
    'bare': ('t', 'k = data.T1().index.asof(16.0 + 2.5 * t)\nreturn float(k) if k == k else -1.0'),
    'nan_x': ('t', 'k = data.T1().index.asof(float("nan") if t == 1 else 21.0)\nreturn float(k) if k == k else -1.0'),
    'array_x': ('t', 'x = np.array([19.5, 21.0]) if policy_id > 2 else 20.5\ntry:\n'
                     '    return float(np.sum(data.T1().index.asof(x)))\nexcept Exception as error:\n'
                     '    return -float(len(type(error).__name__))'),
    'prefix': ('t', 'return int(data.T2().loc[sex()].index.asof(t + 1.5))'),
    'prefix_key': ('t', 'tbl = data.T2()\nreturn tbl.loc[(sex(), tbl.loc[sex()].index.asof(t + 1.5)), "r"]'),
    'unsorted': ('t', 'return int(data.TU().index.asof(20.5 + t))'),
}


def test_ordered_lookups_per_step(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    for cell in ('as_key', 'as_int', 'bare', 'nan_x', 'array_x', 'prefix', 'prefix_key', 'unsorted'):
        assert 'label_ll' in native_reads(source, cell), (cell, source)
    assert source.count('.asof_row(') >= 8 and source.count('.is_number(') >= 8    # (a key's twice: both branches)
