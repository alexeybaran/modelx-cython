"""Index extremes read per step: a C integer on an int level, pandas' own label elsewhere; equal to live, also on
policies and steps the sample did not reach, and on a frame whose index is not of the traced kinds."""
from forms_probe import compare, native_reads

FRAMES = {
    'T1': 'pd.DataFrame({"q": [0.1, 0.2, 0.3, 0.4]}, index=pd.Index([18, 19, 20, 22], name="age"))',
    'T2': 'pd.DataFrame({"r": np.arange(8, dtype=float)}, '
          'index=pd.MultiIndex.from_product([["M", "F"], [1, 2, 3, 5]], names=["sex", "year"]))',
    'TS': 'pd.DataFrame({"q": [1.0, 2.0]}, index=pd.Index(["a", "b"]))',
}
CELLS = {
    'sex': ('', 'return "M" if policy_id % 2 else "F"'),
    'upper': ('t', 'return data.T1().loc[min(18 + t, int(data.T1().index.max())), "q"]'),
    'bottom': ('t', 'return t + data.T1().index.min()'),
    'builtin': ('t', 'return min(18 + t, max(data.T1().index))'),
    'prefix_top': ('t', 'tbl = data.T2()\nreturn tbl.loc[(sex(), min(t + 1, tbl.loc[sex()].index.max())), "r"]'),
    'prefix_bottom': ('t', 'return data.T2().loc[sex()].index.min() + t'),
    'level_top': ('t', 'return min(t, int(data.T2().index.get_level_values("year").max()))'),
    'level_local': ('t', 'years = data.T2().index.get_level_values(1)\nreturn years.min() * t'),
    'walk_top': ('t', 'return max(y for s, y in data.T2().index) + t'),
    'text_top': ('t', 'return data.TS().index.max() * (t + 1)'),
    'pair_top': ('t', 'return str(data.T2().index.max()) + str(t)'),
    'powered': ('t', 'return data.T1().index.max() ** -1 if t else 0.0'),
}


def test_extremes_read_per_step(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    assert set(native_reads(source, 'upper')) == {'max_ll', 'f64_1'}, source     # an extreme inside a key
    assert native_reads(source, 'bottom') == ['min_ll'], source
    assert native_reads(source, 'builtin') == ['max_ll'], source
    assert set(native_reads(source, 'prefix_top')) == {'label_ll', 'f64_c'}, source
    assert native_reads(source, 'prefix_bottom') == ['label_ll'], source
    assert native_reads(source, 'level_top') == ['level_max_ll'], source
    assert native_reads(source, 'level_local') == ['level_min_ll'], source
    assert native_reads(source, 'walk_top') == ['level_max_ll'], source
    assert native_reads(source, 'text_top') == ['max_obj'], source
    assert native_reads(source, 'pair_top') == ['max_obj'], source
    assert native_reads(source, 'powered') == [], source        # numpy's int64 ** -1 raises: left to pandas
