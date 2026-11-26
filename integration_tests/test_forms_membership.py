"""``key in T.index`` read per step: a native test while every key part is one label, pandas' own membership
otherwise, so a several-rows or unhashable key answers or raises as live does (``T.index[mask] in T.index``
raises TypeError)."""
from forms_probe import compare, lowered

FRAMES = {
    'T1': 'pd.DataFrame({"q": [0.1, 0.2, 0.3, 0.4]}, index=pd.Index([18, 19, 20, 22], name="age"))',
    'T2': 'pd.DataFrame({"r": np.arange(8, dtype=float)}, '
          'index=pd.MultiIndex.from_product([["M", "F"], [1, 2, 3, 5]], names=["sex", "year"]))',
    'TS': 'pd.DataFrame({"q": [1.0, 2.0]}, index=pd.Index(["a", "b"]))',
}
CELLS = {
    'code': ('', 'return "a" if policy_id < 3 else "z"'),
    'has_age': ('t', 'return 1.0 if 17 + t in data.T1().index else 0.0'),
    'lacks_pair': ('t', 'return 1.0 if ("M", t + 2) not in data.T2().index else 0.0'),
    'held_code': ('t', 'return 1.0 if code() in data.TS().index else float(t)'),
    'mask_key': ('t', 'k = data.T1().index[data.T1()["q"] > 0.15] if policy_id > 2 else 18 + t\n'
                      'return 1.0 if k in data.T1().index else 0.0'),
    'list_key': ('t', 'k = [18, 19] if policy_id > 3 else 19\nreturn 1.0 if k in data.T1().index else 0.0'),
    'self_mask': ('t', 'return 1.0 if data.T1().index[data.T1()["q"] > 0.1 * t] in data.T1().index else 0.0'),
}


def test_membership_native_and_pandas_by_key(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    for cell in ('has_age', 'held_code', 'mask_key', 'list_key', 'lacks_pair'):
        assert lowered(reads, cell, ' in self.data.'), (cell, reads)
