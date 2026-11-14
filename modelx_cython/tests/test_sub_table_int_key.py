"""diag-us2: ``int(sub.index.max())`` on a prefix-selected sub-table is a C integer key part."""

from modelx_cython.tests.test_table_reads import M, base_model, lower


def test_sub_table_index_extreme_is_an_integer_key():
    text, reads, _ = lower(base_model(), 'def _f_x(self, t):\n'
                           '    sub = self.multi_table().loc[(self.sex(), self.plan())]\n'
                           '    m = min(t, int(sub.index.max()))\n'
                           '    return float(sub.loc[m, "v"])\n')
    sub = f'_mx_cy.cast({M}.TableHandle, _mx_h_sub)'
    assert f'm = min(t, {sub}.last_int())' in text
    # the key is passed as a C integer, not boxed
    assert f'{sub}.row_1(_mx_cy.cast(_mx_cy.longlong, m))' in text, text
    assert 'cast(object, m)' not in text
    assert reads.keys[-1].endswith('int (local m is assigned integers)'), reads.keys


def test_a_full_key_row_is_not_a_sub_table():
    # a `.loc` with as many parts as levels is a row, not a sub-table: no integer index there
    text, _, _ = lower(base_model(), 'def _f_x(self, t):\n'
                       '    sub = self.multi_table().loc[(self.sex(), self.plan(), 3)]\n'
                       '    m = min(t, int(sub.index.max()))\n'
                       '    return m\n')
    assert 'last_int' not in text
