"""diag-run8: ``T.index.max()[k]`` / ``min()[k]`` on an integer level k of several is read as a C integer
(``H.last_part_int(k)`` / ``first_part_int(k)``), so a per-step read does not take references to the
shared extreme key tuple or its part. A label level, a variable index and a single level keep their forms."""

from modelx_cython.tests.test_table_reads import RUNTIME, _run, base_model, lower

CASES = {
    'max part as a clamp': ('def _f_x(self, t):\n'
                            '    tbl = self.r_table()\n'
                            '    top = tbl.index.max()[0]\n'
                            '    return float(tbl.loc[(min(t, top), 20), "q"])\n', 'last_part_int(0)'),
    'min part in arithmetic': ('def _f_x(self, t):\n'
                               '    low = self.r_table().index.min()[1]\n'
                               '    return low + t\n', 'first_part_int(1)'),
    'max part inline': ('def _f_x(self, t):\n'
                        '    return self.r_table().index.max()[1] - t\n', 'last_part_int(1)'),
}


def test_integer_level_parts_are_c_integer_reads_with_live_values():
    receiver = base_model().receiver()
    for name, (source, spelled) in CASES.items():
        text, reads, _ = lower(base_model(), source)
        assert spelled in text and 'last_key()' not in text and 'first_key()' not in text, (name, text)
        assert any(key.startswith('index extreme part') for key in reads.keys), (name, reads.keys)
        for t in (0, 1, 2, 5, 40):
            assert _run(text, receiver, (t,), RUNTIME) == _run(source, receiver, (t,), {}), (name, t)


def test_label_level_variable_index_and_single_level_keep_their_forms():
    declined = ['def _f_x(self, t):\n    p = self.multi_table().index.max()[1]\n    return p\n',
                'def _f_x(self, t):\n    k = 0\n    return self.r_table().index.max()[k]\n',
                'def _f_x(self, t):\n    return self.mort_table().index.max() + t\n']
    for source in declined:
        text, _, _ = lower(base_model(), source)
        assert 'part_int' not in text, text
