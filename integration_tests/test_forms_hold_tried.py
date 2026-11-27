"""A hold whose evaluation raises is remembered as tried: its reads run on pandas for the item, and the hold is not
evaluated again (and raised again) on every call of the formula. Live's values and errors are kept: the error is the
read's own, raised by its pandas branch where live raises it."""
import re

from forms_probe import compare, lowered

FRAMES = {'T2': 'pd.DataFrame({"rate": np.arange(12, dtype=float) / 100}, '
                'index=pd.MultiIndex.from_product([["a", "b", "c"], range(4)]))'}
CELLS = {
    'gcode': ('', 'if policy_id == 3:\n    raise ValueError("no code")\nreturn "b"'),
    'rate': ('t', 'if policy_id != 3:\n    return data.T2().loc[(gcode(), t % 4), "rate"]\nreturn 0.0'),
    'view': ('t', 'v = data.T2().loc[gcode()]\nreturn v.loc[t % 4, "rate"]'),
}


def test_a_hold_that_raises_is_tried_once(tmp_path):
    reads, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4), sample_ids=(1, 2))
    assert not failures, '\n'.join(failures)
    assert lowered(reads, 'rate', 'loc[(self.gcode(), t % 4), "rate"]'), reads
    source = (tmp_path / 'probe_nomx_cy' / '_mx_classes.py').read_text(encoding='utf-8').splitlines()
    holds = 0
    for n, line in enumerate(source):
        tested = re.match(r'\s*if self\.(_mx_t[a-z]_\d+) == 0:\s*$', line)
        if tested is None:
            continue
        block = source[n + 1:n + 8]
        handler = next(i for i, text in enumerate(block) if text.strip() == 'except Exception:')
        after = '\n'.join(block[handler + 1:handler + 3])
        # the handler records the miss in the field its test reads (a table: its state, set to 1)
        assert re.search(rf'self\.{tested[1]} = (-?1)\b', after), '\n'.join(source[n:n + 8])
        holds += 1
    assert holds >= 3, holds
