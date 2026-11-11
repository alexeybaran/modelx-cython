"""Raw table scalars retain NumPy arithmetic and defer unused object errors."""
import os
import subprocess
import sys
import textwrap

import modelx as mx
import numpy as np
import pandas as pd

from modelx_helpers import pandas_inputs as inputs, table_operations as ops, typed_inputs as h


def test_raw_table_selection_all_modes(tmp_path):
    frame = pd.DataFrame({'rate': pd.Series([np.float64(0.1), object()], dtype=object),
                          'opaque': pd.Series([['original'], object()], dtype=object)})
    model = mx.new_model('RawTableScalars')
    try:
        p = model.new_space('Projection', refs={'source': frame, 'prepare': inputs.table,
            'row': ops.row, 'lookup_float': h.lookup_float, 'lookup_object': h.lookup_object})
        p.new_cells('public_input', 'lambda: source')
        p.new_cells('owned', 'lambda: prepare(public_input(), rows=True, preserve_scalar_types=True)')
        p.new_cells('rate', 'lambda key: lookup_float(owned(), key, "rate")')
        p.new_cells('scalar', '''def scalar(key) -> object:
    return lookup_object(row(owned(), key), "rate")
''')
        p.new_cells('opaque', 'lambda key: lookup_object(owned(), key, "opaque")')
        assert p.rate(0) == 0.1 and type(p.scalar(0)) is np.float64
        p.source = frame.assign(rate=[0.3, 0.4])
        assert p.rate(0) == 0.3
        p.source = frame
        assert p.rate(0) == 0.1
        model.export(tmp_path/'exported')
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model as m
p=m.Projection
p.owned();p.rate(0);p.scalar(0);p.opaque(0)
try:p.rate(1)
except TypeError:pass
else:raise AssertionError('selected opaque value did not raise while sampling')
''')
    run = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                          '--sample', 'sample.py', '--no-spec'], cwd=tmp_path,
                         env=dict(os.environ), capture_output=True, text=True)
    (tmp_path/'build.log').write_text(run.stdout+run.stderr)
    assert run.returncode == 0, run.stdout+run.stderr
    generated = (tmp_path/'exported_cy/_mx_classes.py').read_text()
    assert '.table(' in generated and 'preserve_scalar_types=True' in generated
    probe = textwrap.dedent('''
        import importlib,pickle,sys
        import numpy as np
        from modelx_helpers import pandas_inputs as inputs,table_operations as ops,typed_inputs as h
        from modelx_helpers.immutable_inputs import Table,Record
        m=importlib.import_module(sys.argv[1]).mx_model
        p=m.Projection
        p.owned()  # Mixed input parsing is an explicit compatibility boundary.
        original_opaque=p.source.loc[0,'opaque']
        forbidden={h.lookup_float.__code__,h.lookup_object.__code__,ops.row.__code__,
                   Table.lookup.__code__,Table.__getitem__.__code__,Record.__getitem__.__code__}
        calls=[]
        def profile(frame,event,arg):
            if event=='call' and (frame.f_code in forbidden or
                    frame.f_globals.get('__name__','').startswith('pandas')):
                calls.append(frame.f_code.co_name)
        if sys.argv[1].endswith('_cy'):sys.setprofile(profile)
        try:
            assert p.rate(0)==0.1
            assert type(p.scalar(0))is np.float64
            assert p.opaque(0)is original_opaque
        finally:sys.setprofile(None)
        assert calls==[],calls
        try:p.rate(1)
        except TypeError:pass
        else:raise AssertionError('selected opaque value did not raise')
        restored=pickle.loads(pickle.dumps(m))
        assert restored.Projection.rate(0)==0.1
        assert type(restored.Projection.scalar(0))is np.float64
        p.source=p.source.assign(rate=[0.3,0.4])
        assert p.rate(0)==0.1  # Exported/native cached values need explicit cleanup.
        fresh=type(m)().Projection
        fresh.source=p.source
        assert fresh.rate(0)==0.3
        def current(frame,*,preserve_scalar_types,rows):
            assert preserve_scalar_types and rows
            return inputs.table(frame,rows=rows,preserve_scalar_types=preserve_scalar_types)
        replacement=type(m)().Projection
        replacement.prepare=current
        assert replacement.rate(0)==0.1
    ''')
    for package in ['exported', 'exported_cy']:
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             env=dict(os.environ), capture_output=True, text=True)
        (tmp_path/(package+'-validation.log')).write_text(run.stdout+run.stderr)
        assert run.returncode == 0, run.stdout+run.stderr
