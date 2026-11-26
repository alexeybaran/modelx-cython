"""Walks of an index read per step: comprehensions and ``for`` loops over the index of a view local, of a whole frame,
of an inline prefix and of a local holding an index, and ``list``/``tuple``/``set`` of an index. The labels are walked
in pandas order, also on a frame not stored in label order. Equal to live, also on policies and steps the sample did
not reach, where a prefix is missing, and where the rows of a prefix are not in pandas order."""
from forms_probe import compare, native_reads

FRAMES = {
    # rows of each prefix stored out of label order: walked by pandas position
    'T2': 'pd.DataFrame({"r": np.arange(8, dtype=float) / 10}, '
          'index=pd.MultiIndex.from_tuples([("M", 3), ("M", 1), ("M", 5), ("M", 2), ("F", 1), ("F", 2), ("F", 3), '
          '("F", 5)], names=["sex", "year"]))',
    'T1': 'pd.DataFrame({"q": [0.4, 0.1, 0.3, 0.2]}, index=pd.Index([22, 18, 20, 19], name="age"))',
    'TO': 'pd.DataFrame({"share": [0.5, 0.3, 0.2]}, index=pd.Index(["c", "a", "b"], name="cause"))',
    'T3': 'pd.DataFrame({"r": np.arange(8, dtype=float)}, '
          'index=pd.MultiIndex.from_product([["M", "F"], [10, 20, 30, 40]], names=["sex", "age"]))',
}
CELLS = {
    'sex': ('', 'return "M" if policy_id % 2 else "F"'),
    'view_list': ('t', 'sub = data.T3().loc[sex()]\nages = [a for a in sub.index if a <= 10 * (t + 1)]\n'
                       'return float(sub.loc[max(ages), "r"])'),
    'view_sorted': ('t', 'sub = data.T3().loc[sex()]\nages = sorted(int(a) for a in sub.index)\nreturn ages[t] * 1.5'),
    'view_unordered': ('t', 'sub = data.T2().loc[sex()]\nreturn sum(y * 2 for y in sub.index if y > t)'),
    'whole_sorted': ('t', 'tbl = data.T1()\nages = sorted(int(a) for a in tbl.index)\nreturn ages[t] + 0.5'),
    'whole_order': ('t', 'return [a for a in data.T1().index][t] * 1.0'),
    'whole_pairs': ('t', 'tbl = data.T2()\nreturn max(a for s, a in tbl.index if s == sex()) - t'),
    'whole_text': ('t', 'tbl = data.TO()\nreturn sum(tbl.loc[c, "share"] for c in tbl.index if c != "abc"[t])'),
    'index_local': ('t', 'idx = data.T1().index\nreturn int(max(b for b in idx if b <= 19 + t))'),
    'inline_prefix': ('t', 'return tuple(int(d) for d in data.T3().loc[sex()].index)[t] * 1.0'),
    'inline_missing': ('t', 'return len([d for d in data.T3().loc["MF"[t % 2] if t < 2 else "X"].index])'),
    'dict_walk': ('t', 'sub = data.T3().loc[sex()]\nd = {a: a * t for a in sub.index}\nreturn float(d[20])'),
    'loop': ('t', 'tab = data.T1()\nfor a in tab.index:\n    if a >= 19 + t:\n        return float(tab.loc[a, "q"])\n'
                  'return -1.0'),
    'tuple_target': ('t', 'out = 0\nfor k in data.T2().index:\n    out += k[1] * (k[0] == sex())\nreturn out * 1.0'),
    'as_list': ('t', 'sub = data.T3().loc[sex()]\nreturn list(sub.index)[t] * 1.0'),
    'as_set': ('t', 'return len(set(data.T2().index)) + t'),
    'lambda_inside': ('t', 'sub = data.T3().loc[sex()]\nreturn sum(map(lambda a: a, [a for a in sub.index])) * 1.0 + t'),
}


def test_walks_read_per_step(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    walked = {name for name in CELLS if 'range(' in _formula(source, name)}
    assert walked >= {'view_list', 'view_sorted', 'view_unordered', 'whole_sorted', 'whole_order', 'whole_pairs',
                      'whole_text', 'index_local', 'inline_prefix', 'inline_missing', 'dict_walk', 'loop',
                      'tuple_target', 'as_list', 'as_set', 'lambda_inside'}, walked
    assert 'label_obj' in _formula(source, 'whole_text') and 'whole_row' in _formula(source, 'whole_text')
    assert 'f64_1' in native_reads(source, 'whole_text'), source     # the read keyed by the walked label
    assert re.search(r'^ +idx = ', _formula(source, 'index_local'), re.M) is None     # the index local is dropped


def _formula(source, cell):
    start = source.index(f'def _f_{cell}(')
    text = source[start:source.index('\n    @', start)]
    called = re.search(r'\b(_mx_\w*formula_\d+)\(', text)
    if called:
        begin = source.index(f'\ndef {called.group(1)}(')
        end = re.search(r'\n(def |class |@|_mx_)', source[begin + 1:])
        text += source[begin:begin + 1 + end.start()] if end else source[begin:]
    return text


import re  # noqa: E402
