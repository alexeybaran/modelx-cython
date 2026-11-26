"""Reductions over the walks of a table index read per step: ``sum``/``min``/``max``/``any``/``all`` of a generator
over the index of a view local, of a whole frame or of an inline prefix (walked in pandas order on the prepared
table), and over a list built from such a walk. Equal to live, also on policies and steps the sample did not reach,
where a prefix is missing (KeyError) and where nothing is left to reduce (ValueError)."""
import re

from forms_probe import compare

FRAMES = {
    'T2': 'pd.DataFrame({"r": np.arange(8, dtype=float) / 10}, '
          'index=pd.MultiIndex.from_tuples([("M", 3), ("M", 1), ("M", 5), ("M", 2), ("F", 1), ("F", 2), ("F", 3), '
          '("F", 5)], names=["sex", "year"]))',
    'T3': 'pd.DataFrame({"r": np.arange(8, dtype=float) * 0.1}, '
          'index=pd.MultiIndex.from_product([["M", "F"], [10, 20, 30, 40]], names=["sex", "age"]))',
    'TO': 'pd.DataFrame({"share": [0.5, 0.3, 0.2]}, index=pd.Index(["c", "a", "b"], name="cause"))',
}
CELLS = {
    'sex': ('', 'return "M" if policy_id % 2 else "F"'),
    'view_max': ('t', 'sub = data.T3().loc[sex()]\nreturn max(a for a in sub.index if a <= 10 * (t + 1)) * 1.0'),
    'view_sum': ('t', 'sub = data.T2().loc[sex()]\nreturn sum(y * 2 for y in sub.index if y > t)'),
    'view_any': ('t', 'sub = data.T2().loc[sex()]\nreturn any(y == t + 3 for y in sub.index)'),
    'whole_text_sum': ('t', 'tbl = data.TO()\nreturn sum(tbl.loc[c, "share"] for c in tbl.index if c != "abc"[t])'),
    'interpolated': ('t', 'sub = data.T3().loc[sex()]\nages = sorted(int(a) for a in sub.index)\nx = 15 + 10 * t\n'
                          'lo = max(a for a in ages if a <= x)\nhi = min(a for a in ages if a > x)\n'
                          'return float(sub.loc[lo, "r"]) * (hi - x) + float(sub.loc[hi, "r"]) * (x - lo)'),
    'missing': ('t', 'return max(a for a in data.T3().loc["MF"[t % 2] if t < 2 else "X"].index) * 1.0'),
    'nothing_left': ('t', 'sub = data.T3().loc[sex()]\nreturn min(a for a in sub.index if a > 25 + 10 * t) * 1.0'),
}


def test_reductions_over_walks(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    lowered = {name for name in CELLS if '_mx_reduce_' in _formula(source, name)}
    assert lowered >= {'view_max', 'view_sum', 'view_any', 'whole_text_sum', 'interpolated', 'missing',
                       'nothing_left'}, lowered
    # the generator over the index is a loop over positions
    assert 'range(' in _formula(source, 'view_max')
    assert not re.search(r'\bmax\(a for a in', _formula(source, 'view_max').split('else:')[0])


def _formula(source, cell):
    start = source.index(f'def _f_{cell}(')
    text = source[start:source.index('\n    @', start)]
    called = re.search(r'\b(_mx_\w*formula_\d+)\(', text)
    if called:
        begin = source.index(f'\ndef {called.group(1)}(')
        end = re.search(r'\n(def |class |@|_mx_)', source[begin + 1:])
        text += source[begin:begin + 1 + end.start()] if end else source[begin:]
    return text
