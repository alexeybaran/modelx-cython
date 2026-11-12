"""Explicit row/column providers preserve public contracts in all three modes."""
import os
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd
import pytest

from modelx_helpers import pandas_inputs, table_operations, typed_inputs
from modelx_helpers.table_keys import keys


@pytest.mark.parametrize('raw', [False, True])
def test_row_column_public_contract_and_cleanup_modes(tmp_path, raw):
    frame = pd.DataFrame({'age': [42, 43], 'rate': [0.1, 0.2],
                          'sex': ['M', 'F'], 'active': [True, False]},
                         index=pd.Index([9, 2], name='policy'))
    frame.columns.name = 'field'
    model = mx.new_model('SelectedTables')
    try:
        p = model.new_space('Projection', refs={
            'source': frame, 'prepare': pandas_inputs.table,
            'row': table_operations.row, 'column': table_operations.column,
            'to_pandas': table_operations.to_pandas, 'keys': keys,
            'lookup_float': typed_inputs.lookup_float})
        p.new_cells('public_input', 'lambda: source')
        p.new_cells('owned', f'lambda: prepare(public_input(), rows=True, preserve_scalar_types={raw})')
        p.new_cells('selected', 'lambda key: row(key=key, table=owned())')
        p.new_cells('rate', 'lambda key: lookup_float(selected(key), "rate")')
        p.new_cells('values', 'lambda: column(owned(), "rate")')
        p.new_cells('fields', 'lambda key: keys(selected(key))')
        p.new_cells('public_row', 'lambda key: to_pandas(selected(key))')
        p.new_cells('public_report', 'lambda: to_pandas(owned())')
        assert p.rate(9) == 0.1 and p.values() == (0.1, 0.2)
        assert p.fields(9) == ('age', 'rate', 'sex', 'active')
        pd.testing.assert_series_equal(p.public_row(9), frame.loc[9], check_exact=True)
        pd.testing.assert_frame_equal(p.public_report(), frame, check_exact=True)
        replacement = frame.assign(rate=[0.3, 0.4])
        p.source = replacement
        assert p.rate(9) == 0.3  # Whole-value replacement is traced live.
        p.source = frame
        assert p.rate(9) == 0.1
        prior = p.owned()
        p.public_input[()] = replacement
        assert p.rate(9) == 0.3 and p.owned() is not prior
        p.public_input.clear_all()
        assert p.rate(9) == 0.1
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('''from exported import mx_model as m
p=m.Projection
p.owned();p.rate(9);p.rate(2);p.values();p.fields(9)
p.public_row(9);p.public_report()
''')
    run = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                          '--sample', 'sample.py', '--no-spec'], cwd=tmp_path,
                         env=dict(os.environ), capture_output=True, text=True)
    (tmp_path / 'build.log').write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    generated = (tmp_path / 'exported_cy/_mx_classes.py').read_text()
    assert '.row(' in generated and '.column(' in generated
    probe = textwrap.dedent('''
        import importlib,pickle,sys
        import pandas as pd
        from modelx_helpers.table_operations import row,column
        from modelx_helpers import pandas_inputs
        from modelx_helpers.immutable_inputs import freeze
        from modelx_helpers.immutable_inputs import Record,Table
        from modelx_helpers.table_keys import keys
        m=importlib.import_module(sys.argv[1]).mx_model
        p=m.Projection
        original=p.source.copy(deep=True)
        forbidden={row.__code__,column.__code__,keys.__code__,
                   pandas_inputs.table.__code__,pandas_inputs._scalar.__code__,freeze.__code__,
                   Record.__getitem__.__code__,Record.__init__.__code__,
                   Table.__getitem__.__code__,Table.__init__.__code__}
        calls=[]
        preparing=True
        def profile(frame,event,arg):
            if event=='call' and (frame.f_code in forbidden or
                not preparing and frame.f_globals.get('__name__','').startswith('pandas')):
                calls.append(frame.f_code.co_name)
        if sys.argv[1].endswith('_cy'):sys.setprofile(profile)
        try:
            p.owned()  # Native preparation uses public pandas accessors.
            preparing=False
            assert p.rate(9)==0.1 and p.rate(2)==0.2
            assert p.values()==(0.1,0.2)
            assert p.fields(9)==('age','rate','sex','active')
        finally:sys.setprofile(None)
        assert calls==[],calls
        # Public pandas materialization is outside the calculation guard.
        pd.testing.assert_frame_equal(p.public_report(),original,check_exact=True)
        pd.testing.assert_series_equal(p.public_row(9),original.loc[9],check_exact=True)
        p.source=original.assign(rate=[0.3,0.4])
        assert p.rate(9)==0.1  # Exported/native cache rules require cleanup.
        fresh=type(m)().Projection
        fresh.source=p.source
        assert fresh.rate(9)==0.3
        pd.testing.assert_frame_equal(fresh.public_report(),p.source,check_exact=True)
        restored=pickle.loads(pickle.dumps(m))
        pd.testing.assert_frame_equal(restored.Projection.public_report(),original,check_exact=True)
        replacement=type(m)().Projection
        seen=[]
        def alternate(*,key,table):
            seen.append(key)
            return row(table,key)
        replacement.row=alternate
        assert replacement.rate(9)==0.1 and seen==[9]
    ''')
    for package in ['exported', 'exported_cy']:
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             env=dict(os.environ), capture_output=True, text=True)
        (tmp_path / (package + '-validation.log')).write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
