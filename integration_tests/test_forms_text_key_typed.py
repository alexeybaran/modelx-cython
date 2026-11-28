"""``float(T.loc[k, c])`` with a non-literal key part on a text level stays a C double.

The read is native only while ``k`` names one label; otherwise pandas selects (a list gives a Series). Under
``float()`` the pandas branch is converted as the formula converts it, so the read keeps its C type and its native
value is not boxed. Live's values and errors are kept: a one-label list gives that label's value, a longer one raises.
"""
import re

from forms_probe import compare, lowered

FRAMES = {'TS': 'pd.DataFrame({"sa": [1.0, 2.0, 3.0]}, index=pd.Index(["a", "b", "c"]))'}
CELLS = {'pick': ('', 'k = "abc"[policy_id - 1] if policy_id < 4 else (["b"] if policy_id == 4 else ["a", "b"])\n'
                      'return float(data.TS().loc[k, "sa"])')}


def test_float_of_a_text_key_read_is_a_c_double(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4, 5), sample_ids=(1, 2))
    assert lowered(reads, 'pick', 'loc[k, "sa"]'), reads
    assert not failures, '\n'.join(failures)
    source = (tmp_path / 'probe_nomx_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    returns = [line for line in source.splitlines() if 'is_label((_mx_okey_' in line]
    assert returns, source
    for line in returns:    # the native branch is not boxed: float() is around the pandas branch only
        assert re.search(r'return \(', line) and 'else float(' in line, line
