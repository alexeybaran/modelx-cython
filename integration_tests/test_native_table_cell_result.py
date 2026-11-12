"""Verify native table dispatch through an ordinary parameterized Cell."""

import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd
from modelx_helpers import pdn, typed_inputs
from modelx_helpers.immutable_inputs import Record, Table


def test_parameterized_table_result(tmp_path):
    model = mx.new_model('ParameterizedTable')
    try:
        space = model.new_space('Policy', formula='lambda policy: None')
        space.tables = typed_inputs
        space.pdn = pdn
        space.schedules = Record({
            2: Table((0, 1), ((2.0,), (5.0,)), ('rate',)),
            7: Table((0, 1, 2), ((11.0,), (13.0,), (17.0,)), ('rate',)),
        })
        space.rows = pdn.table(pd.DataFrame({'rate': [2.0, 5.0]}, index=[4, 9]), rows=True)
        index = pd.MultiIndex.from_tuples([('a', 4), ('a', 9), ('b', 3)])
        space.compound_rows = pdn.table(pd.DataFrame({'rate': [2.0, 5.0, 1.0]}, index=index))
        space.new_cells('row_table', '''def row_table() -> "modelx_helpers.immutable_inputs.Table":
    return rows
''')
        space.new_cells('row_rate', 'lambda key: row_table().slice(key)["rate"]')
        space.new_cells('compound_table', '''def compound_table() -> "modelx_helpers.immutable_inputs.Table":
    return compound_rows
''')
        space.new_cells('has_row', 'lambda group, key: (group, key) in compound_table().index')
        space.new_cells('missing', '''def missing(value: object) -> object:
    return pdn.isna(value)
''').is_cached = False
        space.new_cells('schedule', textwrap.dedent('''\
            def schedule(account) -> "modelx_helpers.immutable_inputs.Table":
                return schedules[account]
            '''))
        space.new_cells('amount', textwrap.dedent('''\
            def amount(account, month):
                """Read the requested account's rate and repeat its last month."""
                table = schedule(account)
                key = min(month, table.index.max_int())
                return table.lookup_float(key, "rate")
            '''))
        space.new_cells('previous_amount', '''def previous_amount(account, month):
    return pdn.previous(schedule(account), month, "rate", -2.0)
''')
        space.new_cells('fixed_schedule', '''def fixed_schedule() -> "modelx_helpers.immutable_inputs.Table":
    return schedules[7]
''')
        space.new_cells('previous_method', 'lambda month: fixed_schedule().previous(month, "rate")')
        space.new_cells('previous_column', 'lambda month, column: fixed_schedule().previous(month, column)')
        space.new_cells('bounded_method', '''def bounded_method(month):
    table = fixed_schedule()
    key = month_key(month)
    lo = int(min(table.index))
    hi = table.index.max_int()
    if key < lo:
        return 2.5
    return table.lookup_float(min(key, hi), "rate")
''')
        space.new_cells('month_key', 'lambda month: month')
        space.new_cells('interpolated_amount', 'lambda account, month: schedule(account).interpolate_float(month, "rate")')
        space.new_cells('policy_row', 'lambda: row_table().slice(4)')
        space.new_cells('float_field', 'lambda t: float(policy_row()["rate"])')
        space.new_cells('raw_field', '''def raw_field(t):
    field = policy_row()["rate"]
    return field
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        's = mx_model.Policy[0]\n'
        's.amount(2, 0); s.amount(7, 1); s.row_rate(4); s.missing(float("nan"))\n'
        's.has_row("a", 4); s.previous_amount(2, 0); s.interpolated_amount(2, 0.5)\n'
        's.float_field(0); s.raw_field(0); s.previous_method(1); s.previous_column(1, 0); s.bounded_method(1)\n',
        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    generated = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert '.bind_native_table(self._f_schedule(account),' in generated
    assert '.read_clamped_table_float(' in generated
    assert '.isna(self.pdn.isna, value)' in generated
    assert '.previous_f64(self.pdn.previous,' in generated
    assert '.prepare_previous_table_method(' in generated
    assert '.read_previous_table_int(' in generated
    assert '.read_bounded_table_float(' in generated
    assert '.read_cast_record_field_float(' in generated
    assert '.read_record_item(' in generated
    probe = textwrap.dedent('''\
        from exported_cy import mx_model
        s = mx_model.Policy[0]
        assert type(s.schedule(2)).__name__ == 'NativeTable'
        assert list(s.compound_table().index) == [('a', 4), ('a', 9), ('b', 3)]
        assert s.row_table().index.max_int() == 9
        assert s.row_table().index.max_float() == 9.0
        assert s.schedule(2) is s.schedule(2)
        assert s.schedule(2) is not s.schedule(7)
        from modelx_helpers.immutable_inputs import Table
        from modelx_helpers import pdn
        import numpy as np
        import pandas as pd
        def check(frame, event, arg):
            if event == 'call':
                assert frame.f_code not in (Table.row.__code__, Table.contains.__code__, Table.previous.__code__, pdn.previous.__code__,
                                            pdn.isna.__code__, pd.isna.__code__)
        import sys
        sys.setprofile(check)
        try:
            assert s.row_rate(4) == 2.0 and s.row_rate(9) == 5.0
            assert s.missing(float('nan')) is True
            assert s.missing(0.0) is False
            assert s.has_row('a', 4) is True and s.has_row('a', 3) is False
            assert s.has_row('b', 3) is True and s.has_row('c', 3) is False
            assert s.previous_amount(2, -1) == -2.0
            assert s.previous_amount(2, 8) == 5.0
            assert s.previous_amount(7, 1) == 13.0
            assert s.previous_method(8) == 17.0
            assert s.previous_column(8, -1) == s.previous_column(8, 0) == 17.0
            assert s.bounded_method(-1) == 2.5 and s.bounded_method(8) == 17.0
            assert s.interpolated_amount(2, 0.5) == 3.5
            assert s.interpolated_amount(7, 8.5) == 17.0
            assert s.float_field(1) == s.raw_field(1) == 2.0
        finally:
            sys.setprofile(None)
        try:
            s.previous_method(-1)
        except ValueError as error:
            assert str(error) == 'No key at or before -1'
        else:
            raise AssertionError('preceding row before the first key')
        previous = pdn.previous
        pdn.previous = lambda table, key, column, default: 23.0
        try:
            assert s.previous_amount(2, 9) == 23.0
        finally:
            pdn.previous = previous
        assert np.array_equal(s.missing(np.array([np.nan, 0.])), [True, False])
        original_isna = pd.isna
        pd.isna = lambda value: value
        try:
            assert s.missing('replacement') == 'replacement'
        finally:
            pd.isna = original_isna
        try:
            s.row_rate(8)
        except KeyError as error:
            assert error.args == (8,)
        else:
            raise AssertionError('missing row key was accepted')
        for month in range(8):
            assert s.amount(2, month) == (2.0, 5.0)[min(month, 1)]
            assert s.amount(7, month) == (11.0, 13.0, 17.0)[min(month, 2)]
        for account, month in ((2, -1), (12, 0)):
            try:
                s.amount(account, month)
            except KeyError:
                pass
            else:
                raise AssertionError('missing input key was accepted')
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
