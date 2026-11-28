"""A prefix view ``s = T.loc[(cause, sex())]`` read by a comprehension the vector pass turns into a typed loop stays
native. The loop fills a vector local the compiler generates (``_mx_v<n>_r.data[i] = ...``); that store is no edit
of a frame, so the formula's reads are not sent to pandas as if it edited one in place."""
import re

from forms_probe import compare, lowered

FRAMES = {'T3': 'pd.DataFrame({"rate": np.arange(16, dtype=float) / 100}, '
                'index=pd.MultiIndex.from_product([[1, 2], [0, 1], [10, 20, 30, 40]], names=["cause", "sex", "age"]))'}
CELLS = {
    'sex': ('', 'return policy_id % 2'),
    'ages': ('cause', 's = data.T3().loc[(cause, sex())]\nreturn [int(s.index[i]) for i in range(len(s.index))]'),
    'rates': ('cause', 's = data.T3().loc[(cause, sex())]\n'
                       'return [float(s.loc[s.index[i], "rate"]) for i in range(len(s.index))]'),
}


def test_a_typed_loop_over_a_prefix_view_stays_native(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3), sample_ids=(1, 2))
    assert not failures, '\n'.join(failures)
    assert lowered(reads, 'ages', 's.index['), reads
    assert lowered(reads, 'rates', 's.loc[s.index['), reads
    assert not any('edits this frame' in (e.get('reason') or '') for es in reads.values() if isinstance(es, list)
                   for e in es if isinstance(e, dict)), reads
    lines = (tmp_path / 'probe_nomx_cy' / '_mx_classes.py').read_text(encoding='utf-8').splitlines()
    assert any('.data[' in line for line in lines)          # the comprehensions are typed loops
    for n, line in enumerate(lines):                          # the view is assigned only where its rows are not found
        if re.match(r'\s*s = .*\.loc\[', line):
            assert re.search(r'_mx_vs_s\w* < 0:\s*$', lines[n - 1]), '\n'.join(lines[max(0, n - 6):n + 2])
