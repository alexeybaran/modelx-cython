"""``for k, row in T.iterrows():`` read per step and per item: the stored rows in pandas order, ``row["c"]`` of the
kind pandas gives it (numpy scalars on a row of number columns, the Python scalars iteration gives on a row with an
object column). Equal to live, also on policies and steps the sample did not reach."""
import re

from forms_probe import compare, native_reads

FRAMES = {
    'TN': 'pd.DataFrame({"lo": [0.0, 1.0, 2.5], "hi": [1.0, 2.5, 9.0], "v": [0.1, 0.2, 0.3], "n": [1, 2, 3]})',
    'TM': 'pd.DataFrame({"basis": ["a", "b", "a"], "lo": [0, 1, 2], "hi": [1, 3, 9], "v": [0.5, 1.5, 2.5]}, '
          'index=pd.Index([30, 10, 20]))',
}
CELLS = {
    'rows_numeric': ('t', 'for _, r in data.TN().iterrows():\n    if r["lo"] < t + 0.5 <= r["hi"]:\n'
                          '        return float(r["v"])\nreturn -1.0'),
    'rows_mixed': ('t', 'for k, row in data.TM().iterrows():\n'
                        '    if str(row["basis"]) == "ab"[t % 2] and int(row["lo"]) <= t < int(row["hi"]):\n'
                        '        return float(row["v"]) + k\nreturn 0.0'),
    'rows_types': ('t', 'out = []\ntbl = data.TM()\nfor k, row in tbl.iterrows():\n'
                        '    out.append(type(row["lo"]).__name__ + type(row["v"]).__name__ + str(k))\n'
                        'return ",".join(out) + str(t)'),
    'rows_numeric_types': ('t', 'out = []\nfor k, row in data.TN().iterrows():\n'
                                '    out.append(type(row["n"]).__name__ + str(row["n"] * 2))\nreturn ",".join(out) + str(t)'),
    'rows_order': ('t', 'out = 0\nfor k, r in data.TM().iterrows():\n    out = out * 100 + k\nreturn out * 1.0 + t'),
    'rows_after': ('t', 'for k, r in data.TN().iterrows():\n    pass\nreturn float(r["v"]) * t'),
    'rows_item': ('', 'for _, r in data.TN().iterrows():\n    if r["n"] == 2:\n        return float(r["v"])\nreturn 0.0'),
}


def test_iterrows_read_per_step(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2), sample_ids=(1,))
    assert not failures, '\n'.join(failures)
    for name in ('rows_numeric', 'rows_mixed', 'rows_types', 'rows_numeric_types', 'rows_order', 'rows_item'):
        assert 'whole_row' in _formula(source, name), name
    assert 'f64_row' in native_reads(source, 'rows_numeric'), source
    assert 'object_field' in native_reads(source, 'rows_mixed'), source
    assert 'iterrows' in _formula(source, 'rows_after') and 'whole_row' not in _formula(source, 'rows_after')


def _formula(source, cell):
    start = source.index(f'def _f_{cell}(')
    text = source[start:source.index('\n    @', start)]
    called = re.search(r'\b(_mx_\w*formula_\d+)\(', text)
    if called:
        begin = source.index(f'\ndef {called.group(1)}(')
        end = re.search(r'\n(def |class |@|_mx_)', source[begin + 1:])
        text += source[begin:begin + 1 + end.start()] if end else source[begin:]
    return text
