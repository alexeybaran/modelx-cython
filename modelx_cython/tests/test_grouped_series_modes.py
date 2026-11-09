"""Native column selection and explicit public Series output in all modes."""
import os
import subprocess
import sys
import textwrap

import modelx as mx
import numpy as np
import pandas as pd

from lifelib import pandas_inputs as inputs, table_operations as ops, typed_inputs as h


def test_grouped_column_public_series_modes(tmp_path):
    index = pd.MultiIndex.from_tuples([('M', 42), ('F', 40), ('M', 40)], names=['sex', 'age'])
    frame = pd.DataFrame({'mort_rate': [0.1, 0.2, np.nan], 'count': [1, 2, 3]}, index=index)
    frame.attrs = {'basis': ['original']}
    frame.flags.allows_duplicate_labels = False
    model = mx.new_model('GroupedPublicSeries')
    try:
        p = model.new_space('Projection', refs={'source': frame, 'prepare': inputs.table,
            'subtable': h.subtable, 'column': ops.column, 'to_pandas': ops.to_pandas,
            'lookup_float': h.lookup_float})
        p.new_cells('public_input', 'lambda: source')
        p.new_cells('owned', 'lambda: prepare(public_input(), groups=("sex",), report=True, preserve_scalar_types=True)')
        p.new_cells('selected', 'lambda sex: subtable(owned(), sex)')
        p.new_cells('column_report', 'lambda sex: column(selected(sex), "mort_rate", report=True)')
        p.new_cells('report', 'lambda sex: to_pandas(column_report(sex))')
        p.new_cells('rate', 'lambda sex, age: lookup_float(selected(sex), age, "mort_rate")')
        pd.testing.assert_series_equal(p.report('M'), frame.loc['M']['mort_rate'], check_exact=True)
        p.source = frame.assign(mort_rate=[0.3, 0.4, 0.5])
        assert p.rate('M', 42) == 0.3
        p.source = frame
        model.export(tmp_path/'exported')
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model as m
p=m.Projection
p.report('M');p.report('F');p.rate('M',42);p.rate('F',40)
''')
    run = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                          '--sample', 'sample.py', '--no-spec'], cwd=tmp_path,
                         env=dict(os.environ), capture_output=True, text=True)
    (tmp_path/'build.log').write_text(run.stdout+run.stderr)
    assert run.returncode == 0, run.stdout+run.stderr
    generated = (tmp_path/'exported_cy/_mx_classes.py').read_text()
    assert '.column(' in generated and 'report=True' in generated
    probe = textwrap.dedent('''
        import importlib,pickle,sys
        import pandas as pd
        from lifelib import table_operations as ops,typed_inputs as h
        from lifelib.immutable_inputs import Table,Record
        m=importlib.import_module(sys.argv[1]).mx_model
        p=m.Projection
        expected=p.source.loc['M']['mort_rate']
        p.owned()  # Grouped input preparation remains an inventoried boundary.
        forbidden={ops.column.__code__,h.subtable.__code__,h.lookup_float.__code__,
                   Table.__init__.__code__,Record.__init__.__code__,Record.__getitem__.__code__}
        calls=[]
        def profile(frame,event,arg):
            if event=='call' and (frame.f_code in forbidden or
                    frame.f_globals.get('__name__','').startswith('pandas')):
                calls.append(frame.f_code.co_name)
        if sys.argv[1].endswith('_cy'):sys.setprofile(profile)
        try:
            selected=p.column_report('M')
            assert p.rate('M',42)==0.1
        finally:sys.setprofile(None)
        assert calls==[],calls
        pd.testing.assert_series_equal(ops.to_pandas(selected),expected,check_exact=True)
        pd.testing.assert_series_equal(p.report('M'),expected,check_exact=True)
        pd.testing.assert_series_equal(ops.to_pandas(pickle.loads(pickle.dumps(selected))),expected,check_exact=True)
        assert p.report('M').flags.allows_duplicate_labels is False
        try:p.report('absent')
        except KeyError as error:assert error.args==('absent',)
        else:raise AssertionError('missing group did not raise')
        p.source=p.source.assign(mort_rate=[0.3,0.4,0.5])
        assert p.rate('M',42)==0.1
        fresh=type(m)().Projection
        fresh.source=p.source
        assert fresh.rate('M',42)==0.3
        def current(table,name,*,report):
            assert report
            return ops.column(table,name,report=True)
        replacement=type(m)().Projection
        replacement.column=current
        pd.testing.assert_series_equal(replacement.report('M'),expected,check_exact=True)
    ''')
    for package in ['exported', 'exported_cy']:
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             env=dict(os.environ), capture_output=True, text=True)
        (tmp_path/(package+'-validation.log')).write_text(run.stdout+run.stderr)
        assert run.returncode == 0, run.stdout+run.stderr
