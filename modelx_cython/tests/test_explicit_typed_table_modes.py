"""Explicit .pxd helper signatures through the normal three execution modes."""
import os
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd

from lifelib import pandas_inputs, typed_inputs


def test_explicit_typed_table_helpers_modes(tmp_path):
    model = mx.new_model('ExplicitTables')
    try:
        source = pd.DataFrame(
            {'rate': [1.5, 2.5], 'age': [42, 43],
             'flag': [True, False], 'label': ['M', 'F']},
            index=pd.MultiIndex.from_tuples([('M', 42), ('F', 43)]))
        refs = {'source': source, 'table_ctor': pandas_inputs.table,
                'record_ctor': pandas_inputs.record, 'predicate': pd.isna,
                'is_missing': typed_inputs.is_missing, 'empty_value': None}
        refs.update({name: getattr(typed_inputs, name) for name in
                     ('lookup_float', 'lookup_int', 'lookup_bool',
                      'lookup_str', 'lookup_object')})
        space = model.new_space('Inputs', refs=refs)
        space.new_cells('raw', 'lambda: source')
        space.new_cells('table', 'lambda: table_ctor(raw(), indexing="at")')
        space.new_cells('row', 'lambda: record_ctor(raw().iloc[0])')
        for name, column in [('float', 'rate'), ('int', 'age'),
                             ('bool', 'flag'), ('str', 'label'),
                             ('object', 'label')]:
            space.new_cells(name + '_value',
                            f'lambda sex, age: lookup_{name}(table(), (sex, age), "{column}")')
        space.new_cells('record_value', 'lambda: lookup_int(row(), "age")')
        space.new_cells('missing_cached', 'lambda: is_missing(predicate, empty_value)')
        space.new_cells('missing_fresh', 'lambda: is_missing(predicate, empty_value)').is_cached = False
        space.new_cells('defaulted',
                        'lambda sex, age: lookup_float(table(), (sex, age), "rate", 7.5)')
        assert space.float_value('M', 42) == 1.5
        assert type(space.int_value('M', 42)) is int
        assert space.bool_value('M', 42) is True
        assert space.str_value('M', 42) == 'M'
        assert space.object_value('F', 43) == 'F'
        assert space.record_value() == 42
        assert space.missing_cached() is True and space.missing_fresh() is True
        assert space.defaulted('X', 1) == 7.5
        space.source = source.assign(rate=[3.5, 4.5])
        assert space.float_value('M', 42) == 3.5
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text('''from exported import mx_model as m
for sex, age in [('M', 42), ('F', 43)]:
    m.Inputs.float_value(sex, age)
    m.Inputs.int_value(sex, age)
    m.Inputs.bool_value(sex, age)
    m.Inputs.str_value(sex, age)
    m.Inputs.object_value(sex, age)
    m.Inputs.defaulted(sex, age)
m.Inputs.record_value()
m.Inputs.missing_cached(); m.Inputs.missing_fresh()
''')
    build = subprocess.run(
        [sys.executable, '-m', 'modelx_cython', 'exported',
         '--sample', 'sample.py', '--no-spec'], cwd=tmp_path,
        capture_output=True, text=True, env=dict(os.environ))
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text()
    for suffix in ('float', 'int', 'bool', 'str', 'object'):
        assert f'.lookup_{suffix}(' in source
    assert '.lookup_key2_float_f64(' not in source
    assert '.lookup_float_f64(' not in source
    assert '.is_missing(' in source

    probe = textwrap.dedent('''
        import importlib, pickle, sys
        from lifelib import typed_inputs as h
        from lifelib.immutable_inputs import Record, Table, freeze
        m = importlib.import_module(sys.argv[1]).mx_model
        s = m.Inputs
        # Preparation is an explicit boundary; profile the admitted reads.
        assert isinstance(s.raw(), __import__('pandas').DataFrame)
        s.table(); s.row()
        forbidden = {getattr(h, 'lookup_' + suffix).__code__
                     for suffix in ('float', 'int', 'bool', 'str', 'object')}
        forbidden.update([h._typed_value.__code__, h.lookup.__code__,
                          h.is_missing.__code__, __import__('pandas').isna.__code__,
                          Table.lookup.__code__, Table.__getitem__.__code__,
                          Record.__getitem__.__code__, freeze.__code__])
        calls = []
        def profile(frame, event, arg):
            if event == 'call' and frame.f_code in forbidden:
                calls.append(frame.f_code.co_name)
        if sys.argv[1].endswith('_cy'):
            sys.setprofile(profile)
        try:
            assert s.float_value('M', 42) == 3.5
            assert type(s.int_value('M', 42)) is int
            assert s.bool_value('M', 42) is True
            assert s.bool_value('F', 43) is False
            assert type(s.str_value('M', 42)) is str
            assert s.object_value('F', 43) == 'F'
            assert s.record_value() == 42
            assert s.missing_cached() is True and s.missing_fresh() is True
            assert s.defaulted('X', 1) == 7.5
            try:
                s.float_value('M', 43)
            except KeyError as error:
                assert error.args == (15,)
            else:
                raise AssertionError('missing sparse .at key must raise')
        finally:
            sys.setprofile(None)
        assert calls == []
        restored = pickle.loads(pickle.dumps(m))
        assert restored.Inputs.float_value('M', 42) == 3.5
        replacement = s.source.assign(rate=[8.5, 9.5])
        s.source = replacement
        assert s.float_value('M', 42) == 3.5  # explicit cleanup required
        fresh = type(m)()
        fresh.Inputs.source = replacement
        assert fresh.Inputs.float_value('M', 42) == 8.5
        events = []
        def current(table, key, column):
            events.append(key)
            return 12.5
        fresh.Inputs.lookup_float = current
        assert fresh.Inputs.float_value('F', 43) == 12.5
        assert events == [('F', 43)]
        seen = []
        def current_predicate(value):
            seen.append(value)
            return False
        s.predicate = current_predicate
        assert s.missing_cached() is True  # exported/compiled cache is retained
        assert s.missing_fresh() is False
        assert seen == [None]
    ''')
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             capture_output=True, text=True, env=dict(os.environ))
        (tmp_path / f'{package}-validation.log').write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
