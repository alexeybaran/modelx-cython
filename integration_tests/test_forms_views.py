"""Views of a prefix held in a local (``sub = T.loc[prefix]``), then read per step: keyed reads of float64, int64
and object columns, a column chosen at run time, the extremes of the remaining level. Equal to live, also on
policies and steps the sample did not reach, where a prefix or key is missing, and where a prefix part is not one
label at run time."""
import re

from forms_probe import compare, native_reads as all_reads


def native_reads(source, cell):
    """The reads of a cell without the setup of a view computed per call (its prefix codes and rows)."""
    return [read for read in all_reads(source, cell) if read not in ('view_start', 'kcode', 'ocode')]


def formula_text(source, cell):
    """The text of a cell's compiled formula, and of the module function it delegates to."""
    start = source.index(f'def _f_{cell}(')
    text = source[start:source.index('\n    @', start)]
    called = re.search(r'\b(_mx_\w*formula_\d+)\(', text)
    if called:
        begin = source.index(f'\ndef {called.group(1)}(')
        end = re.search(r'\n(def |class |@|_mx_)', source[begin + 1:])
        text += source[begin:begin + 1 + end.start()] if end else source[begin:]
    return text

FRAMES = {
    'T2': 'pd.DataFrame({"r": np.arange(8, dtype=float) / 10, "n": np.arange(8), "s": list("abcdefgh")}, '
          'index=pd.MultiIndex.from_product([["M", "F"], [1, 2, 3, 5]], names=["sex", "year"]))',
    'T3': 'pd.DataFrame({"r": np.arange(16, dtype=float) / 7}, '
          'index=pd.MultiIndex.from_product([["M", "F"], [0, 1], [1, 2, 3, 4]], names=["sex", "s", "m"]))',
}
CELLS = {
    'sex': ('', 'return "M" if policy_id % 2 else "F"'),
    'v_read': ('t', 'sub = data.T2().loc[sex()]\nreturn float(sub.loc[t + 1, "r"])'),
    'v_missing_key': ('t', 'sub = data.T2().loc[sex()]\nreturn float(sub.loc[t + 3, "r"])'),
    'v_extreme': ('t', 'grid = data.T2().loc[sex()]\nyears = min(t + 1, int(grid.index.max()))\n'
                       'return float(grid.loc[years, "r"])'),
    'v_int': ('t', 'sub = data.T2().loc[sex()]\nreturn sub.loc[t + 1, "n"] * 2'),
    'v_obj': ('t', 'sub = data.T2().loc[sex()]\nreturn sub.at[t + 1, "s"] + "!"'),
    'v_named': ('t', 'sub = data.T2().loc[sex()]\nname = "r" if t % 2 else "n"\nreturn float(sub.loc[t + 1, name])'),
    'v_step_part': ('t', 'tbl = data.T3()\nsub = tbl.loc[(sex(), t % 2)]\nreturn float(sub.loc[t + 1, "r"])'),
    'v_missing_prefix': ('t', 'sub = data.T2().loc["X" if t == 2 else sex()]\nreturn float(sub.loc[1, "r"])'),
    'v_order': ('t', 'sub = data.T2().loc["X" if t == 1 else sex()]\nx = 1 / (t - 1) if t else 0\n'
                     'return float(sub.loc[1, "r"]) + x'),
    'v_list_part': ('t', 'keys = ["M"] if t == 1 else sex()\nsub = data.T2().loc[keys]\nreturn float(sub.loc[2, "r"])'),
    'v_kept': ('t', 'sub = data.T2().loc[sex()]\nreturn float(sub.loc[t + 1, "r"]) + float(sub["r"].sum())'),
    'v_min_builtin': ('t', 'sub = data.T2().loc[sex()]\nreturn min(sub.index) + t'),
    'v_bottom': ('t', 'sub = data.T2().loc[sex()]\nreturn sub.index.min() + t'),
    'v_numpy': ('t', 'sub = data.T2().loc[sex()]\nreturn sub.index.max() ** -1 if t else 0.0'),
    'v_item': ('', 'sub = data.T2().loc[sex()]\nreturn float(sub.loc[2, "r"])'),
}


def test_views_read_per_step(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    assert native_reads(source, 'v_read') == ['f64_c'], source
    assert native_reads(source, 'v_missing_key') == ['f64_c'], source
    assert native_reads(source, 'v_extreme') == ['label_ll', 'f64_c'], source
    assert native_reads(source, 'v_int') == ['i64_c'], source
    assert native_reads(source, 'v_obj') == ['object_field'], source
    assert native_reads(source, 'v_named') == ['field_f64'], source
    assert native_reads(source, 'v_step_part') == ['f64_c'], source
    assert native_reads(source, 'v_missing_prefix') == ['f64_c'], source
    assert native_reads(source, 'v_order') == ['f64_c'], source
    assert native_reads(source, 'v_list_part') == ['f64_c'], source
    assert native_reads(source, 'v_kept') == ['f64_c'], source
    assert native_reads(source, 'v_min_builtin') == ['label_ll'], source
    assert native_reads(source, 'v_bottom') == ['label_ll'], source
    assert native_reads(source, 'v_numpy') == [], source        # numpy's int64 ** -1 raises: left to pandas
    assert native_reads(source, 'v_item') == ['f64_c'], source
    body = formula_text(source, 'v_read')
    assert not re.search(r'^ +sub = ', body, re.M) and 'if _mx_vs_sub < 0:' in body, body   # every use lowered: pandas only off the view
    assert re.search(r'_mx_vs_sub = self\._mx_tw_\d+ - 2', body), body     # a prefix fixed per item: rows held per item
    kept = formula_text(source, 'v_kept')
    assert re.search(r'^ +sub = ', kept, re.M), kept       # a use left on pandas reads the local
