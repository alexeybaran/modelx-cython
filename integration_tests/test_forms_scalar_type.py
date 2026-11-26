"""``isinstance`` and ``type`` of a lowered read see the numpy scalar pandas returns, as live does."""
from forms_probe import compare, native_reads

FRAMES = {'T1': 'pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0], "n": [7, 8, 9, 10]}, '
                'index=pd.Index([1, 2, 3, 4], name="pid"))'}
CELLS = {
    'is_int': ('', 'return 1.0 if isinstance(data.T1().loc[policy_id, "n"], int) else 0.0'),
    'is_float': ('', 'return 1.0 if type(data.T1().at[policy_id, "sa"]) is float else 0.0'),
    'plain': ('', 'return data.T1().loc[policy_id, "sa"] * 2'),
}


def test_type_of_a_lowered_read_is_live_type(tmp_path):
    source, failures = compare(tmp_path, FRAMES, CELLS, ids=(1, 2, 3, 4))
    assert native_reads(source, 'plain') == ['f64_1'], source
    assert native_reads(source, 'is_int') == ['i64_1'] and native_reads(source, 'is_float') == ['f64_1'], source
    assert not failures, '\n'.join(failures)
