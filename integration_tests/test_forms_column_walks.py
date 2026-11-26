"""A column of a view or prefix walked or read by position per step: its values, ``.items()``, ``.tolist()``,
``list()``, a walk of its index, ``C.iloc[i]`` and ``C.loc[k]``. Equal to live (Python numbers for walks, numpy
scalars for ``iloc``/``loc``), also where a position is out of bounds or a label missing, and on policies and steps
the sample did not reach."""
import re

from forms_probe import compare

FRAMES = {
    'T3': 'pd.DataFrame({"r": np.arange(8, dtype=float) / 10, "n": np.arange(8) * 3}, '
          'index=pd.MultiIndex.from_product([["M", "F"], [10, 20, 30, 40]], names=["sex", "age"]))',
    'T4': 'pd.DataFrame({"tec": np.arange(12, dtype=float) / 100}, '
          'index=pd.MultiIndex.from_product([[1, 2], [2020, 2021], [1, 3, 5]], names=["scen", "year", "mat"]))',
}
CELLS = {
    'sex': ('', 'return "M" if policy_id % 2 else "F"'),
    'col_items': ('t', 's = data.T3().loc[sex()]["r"]\nreturn sum(v * a for a, v in s.items() if a > 10 * t)'),
    'col_values': ('t', 'sub = data.T3().loc[sex()]\nreturn sum(sub["r"].tolist()) + t'),
    'col_list': ('t', 'return list(data.T3().loc[sex()]["n"])[t] * 1.0'),
    'col_walk': ('t', 'return sum(float(p) for p in data.T3().loc[sex()]["r"]) * t'),
    'col_curve': ('t', 'curve = data.T4().loc[(1 + t % 2, 2020), "tec"]\nms = [int(v) for v in curve.index]\n'
                       'return float(curve.iloc[0]) + float(curve.iloc[-1]) + float(curve.loc[ms[1]])'),
    'col_iloc_out': ('t', 'curve = data.T4().loc[(1, 2020), "tec"]\nreturn float(curve.iloc[t * 2])'),
    'col_loc_missing': ('t', 'curve = data.T4().loc[(2, 2021), "tec"]\nreturn float(curve.loc[1 + t])'),
    'col_numpy': ('t', 'curve = data.T4().loc[(1, 2021), "tec"]\nreturn type(curve.iloc[-1]).__name__ + str(t)'),
    'col_loop': ('t', 'out = 0.0\nfor a, v in data.T3().loc[sex()]["r"].items():\n    out += a * v\nreturn out + t'),
}


def test_column_walks_read_per_step(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    for name in CELLS:
        if name != 'sex':
            assert re.search(r'\.(f64|i64)_(row|c)\(', _formula(source, name)), name
    assert 'position(' in _formula(source, 'col_curve') and 'f64_box' in _formula(source, 'col_numpy')


def _formula(source, cell):
    start = source.index(f'def _f_{cell}(')
    text = source[start:source.index('\n    @', start)]
    called = re.search(r'\b(_mx_\w*formula_\d+)\(', text)
    if called:
        begin = source.index(f'\ndef {called.group(1)}(')
        end = re.search(r'\n(def |class |@|_mx_)', source[begin + 1:])
        text += source[begin:begin + 1 + end.start()] if end else source[begin:]
    return text
