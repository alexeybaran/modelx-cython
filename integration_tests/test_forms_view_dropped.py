"""A view local whose every read is lowered is not evaluated on pandas while its rows are found.

``v = T.loc[prefix]`` runs as written only when the prefix's stored rows are not found (live's KeyError). Once every
use of ``v`` is lowered, the compiled formula holds no unconditional copy of that assignment: the pandas ``.loc``
would otherwise run on every call.
"""
import re

from forms_probe import compare, lowered

FRAMES = {'M': 'pd.DataFrame({"c": np.arange(9, dtype=float) + 0.5}, '
               'index=pd.MultiIndex.from_product([[1, 2, 3], range(3)], names=["pid", "t"]))'}
CELLS = {'view_all': ('t', 'v = data.M().loc[policy_id]\nreturn v.loc[min(t, 2), "c"]'),
         'view_cond': ('', 'v = data.M().loc[policy_id]\nif policy_id < 3:\n    return v.loc[1, "c"]\nreturn 0.0')}


def test_a_view_whose_reads_are_lowered_is_assigned_only_where_its_rows_are_not_found(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2, 3))
    assert not failures, '\n'.join(failures)
    source = (tmp_path / 'probe_nomx_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    for cell, read in (('view_all', 'v.loc[min(t, 2), "c"]'), ('view_cond', 'v.loc[1, "c"]')):
        assert lowered(reads, cell, read), reads
    lines = source.splitlines()
    assigned = [n for n, line in enumerate(lines) if re.match(r'\s*v = .*\.loc\[', line)]
    assert assigned, source
    for n in assigned:      # each assignment of v is the guarded copy: the line before it tests the start
        assert re.search(r'_mx_vs_v\w* < 0:\s*$', lines[n - 1]), '\n'.join(lines[max(0, n - 8):n + 2])
