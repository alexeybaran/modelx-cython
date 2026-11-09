"""Direct grouped-report helper across live/exported/compiled execution."""
import os
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd

from lifelib import pandas_inputs, table_operations


def test_grouped_public_report_and_empty_metadata_modes(tmp_path):
    frame=pd.DataFrame({'flow':[1e16,1.0,-1e16,3.0],
                        'opening':[1.0,2.0,3.0,4.0],
                        'closing':[5.0,6.0,7.0,8.0]})
    frame.columns.name='cashflow'
    model=mx.new_model('GroupedReport')
    try:
        p=model.new_space('Projection',refs={'source':frame,'labels':[1,1,1,1],
                'prepare':pandas_inputs.table,'group_sum':table_operations.group_sum,
                'to_pandas':table_operations.to_pandas})
        p.new_cells('result_cf','lambda: source')
        p.new_cells('owned','lambda: prepare(result_cf(), report=True)')
        p.new_cells('grouped','lambda: group_sum(owned(), labels, first=("opening",), last=("closing",))')
        p.new_cells('report','lambda: to_pandas(grouped(), index_name="policy_year")')
        expected=pd.DataFrame({'opening':[1.0],'closing':[8.0],'flow':[4.0]},index=pd.Index([1],name='policy_year'))
        expected.columns.name='cashflow'
        pd.testing.assert_frame_equal(p.report(),expected,check_exact=True)
        p.source=frame.assign(flow=[1.0,2.0,3.0,4.0])
        assert p.report()['flow'].iloc[0]==10.0
        p.source=frame
        pd.testing.assert_frame_equal(p.report(),expected,check_exact=True)
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model as m
m.Projection.report()
fresh=type(m)().Projection
fresh.source=fresh.source[:0];fresh.labels=[]
fresh.report()
''')
    run=subprocess.run([sys.executable,'-m','modelx_cython','exported','--sample','sample.py','--no-spec'],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ))
    (tmp_path/'build.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
    assert '.group_sum(' in (tmp_path/'exported_cy/_mx_classes.py').read_text()
    probe=textwrap.dedent('''
        import importlib,pickle,sys
        import pandas as pd
        from lifelib.table_operations import group_sum,to_pandas
        from lifelib.pandas_inputs import table
        m=importlib.import_module(sys.argv[1]).mx_model
        p=m.Projection
        p.owned()  # Explicit preparation boundary.
        calls=[]
        def profile(frame,event,arg):
            if event=='call' and (frame.f_code in {group_sum.__code__,to_pandas.__code__,table.__code__}
                    or frame.f_globals.get('__name__','').startswith('pandas')):
                calls.append(frame.f_code.co_name)
        if sys.argv[1].endswith('_cy'):sys.setprofile(profile)
        try:owned=p.grouped()
        finally:sys.setprofile(None)
        assert calls==[],calls
        expected=pd.DataFrame({'opening':[1.0],'closing':[8.0],'flow':[4.0]},index=pd.Index([1],name='policy_year'))
        expected.columns.name='cashflow'
        pd.testing.assert_frame_equal(p.report(),expected,check_exact=True)
        restored=pickle.loads(pickle.dumps(m))
        pd.testing.assert_frame_equal(restored.Projection.report(),expected,check_exact=True)
        p.source=p.source.assign(flow=[1.0,2.0,3.0,4.0])
        assert p.report()['flow'].iloc[0]==4.0  # Explicit cleanup is required.
        fresh=type(m)().Projection
        fresh.source=p.source
        assert fresh.report()['flow'].iloc[0]==10.0
        empty=type(m)().Projection
        empty.source=empty.source[:0];empty.labels=[]
        actual=empty.report()
        expected=expected[:0]
        expected.index=pd.Index([],dtype='object',name='policy_year')
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)
        replaced=type(m)().Projection
        def current(table,labels,*,first,last):
            return group_sum(table,labels,first=first,last=last)
        replaced.group_sum=current
        assert replaced.report()['flow'].iloc[0]==4.0
    ''')
    for name in ['exported','exported_cy']:
        run=subprocess.run([sys.executable,'-c',probe,name],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ))
        (tmp_path/(name+'-validation.log')).write_text(run.stdout+run.stderr)
        assert run.returncode==0,run.stdout+run.stderr
