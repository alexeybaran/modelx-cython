"""diag-us2 (D91): ``T.index.max()[k]`` / ``min()[k]`` on several levels whose level k is an integer is an
integer key part (VA_US_S mort_rate: ``top = table.index.max()[0]``), so the read is not boxed."""

from modelx_cython.tests.test_table_reads import RUNTIME, _run, base_model, lower

SOURCE = ('def _f_x(self, t):\n'
          '    tbl = self.r_table()\n'
          '    top = tbl.index.max()[0]\n'
          '    return float(tbl.loc[(min(t, top), 20), "q"])\n')


def test_integer_level_part_of_an_extreme_key_is_an_integer():
    text, reads, _ = lower(base_model(), SOURCE)
    assert '.last_key()[0]' in text, text
    assert 'row_2(_mx_cy.cast(_mx_cy.longlong, min(t, top)), _mx_cy.cast(_mx_cy.longlong, 20))' in text, text
    assert 'cast(object' not in text, text
    assert reads.keys[-1].startswith('loc[key, column], 2 parts: int (integer min/max)'), reads.keys
    receiver = base_model().receiver()
    for t in (0, 1, 2, 5):
        assert _run(text, receiver, (t,), RUNTIME) == _run(SOURCE, receiver, (t,), {})


def test_a_label_level_part_is_not_an_integer():
    source = ('def _f_x(self, t):\n'
              '    tbl = self.multi_table()\n'
              '    p = tbl.index.max()[1]\n'
              '    return float(tbl.loc[(self.sex(), p, 1), "v"])\n')
    text, reads, _ = lower(base_model(), source)
    assert '_mx_cy.cast(_mx_cy.longlong, p)' not in text, text
