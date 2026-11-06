"""The public helper contract must survive export, compilation and rebinding."""
import os
import json
import subprocess
import sys
import textwrap

import modelx as mx
from lifelib.immutable_inputs import Record, Table
from lifelib.typed_inputs import lookup, previous


def test_helper_reference_assignment(tmp_path):
    model = mx.new_model('HelperContract')
    try:
        space = model.new_space('Inputs', refs={
            'record': Record({'rate': 0.1}),
            'table': Table((9, -2, 5), ((90.0,), (-20.0,), (50.0,)), ('value',)),
            'lookup': lookup, 'previous': previous, 'floor': -2, 'ceiling': 8})
        space.new_cells('row', 'lambda: record')
        space.new_cells('value', 'lambda key: lookup(table, key, "value")')
        space.new_cells('preceding', 'lambda key: previous(table, key, "value")')
        space.new_cells('rate', 'lambda: record["rate"]').is_cached = False
        space.new_cells('fresh', 'lambda key: lookup(table, key, "value")').is_cached = False
        space.new_cells('floored', 'lambda key: lookup(table, max(key, floor), "value")').is_cached = False
        space.new_cells('capped', 'lambda key: previous(table, min(key, ceiling), "value", 42.0)').is_cached = False
        assert space.value(-2) == -20.0
        assert space.preceding(8) == 50.0
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'm.Inputs.row(); m.Inputs.rate(); m.Inputs.value(-2); m.Inputs.preceding(8); m.Inputs.fresh(-2)\n'
        'm.Inputs.floored(-10); m.Inputs.capped(20)\n',
        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    assert admission['exported_cy._mx_classes._c_Inputs._f_value']['native_helpers'] == [
        'lifelib._native_inputs.lookup']
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        from lifelib.immutable_inputs import Record, Table
        s = import_module(sys.argv[1]).mx_model.Inputs
        row = s.row()
        assert row is s.row() is s.record
        assert s.value(-2) == -20.0
        assert s.floored(-10) == -20.0
        assert s.capped(20) == 50.0
        s.record = Record({'rate': 0.2})
        assert s.row() is row
        assert s.rate() == 0.2
        s.table = Table((9, -2, 5), ((9.0,), (-2.0,), (5.0,)), ('value',))
        assert s.value(-2) == -20.0
        assert s.value(9) == 9.0
        assert s.preceding(8) == 5.0
        assert s.floored(-10) == -2.0
        s.floor = 5
        assert s.floored(-10) == 5.0
        try:
            s.value(100)
        except KeyError:
            pass
        else:
            raise AssertionError('missing key must raise')
        s.table = Table((100,), ((123.0,),), ('value',))
        assert s.value(100) == 123.0
        assert s.fresh(100) == 123.0
        assert s.capped(20) == 42.0
        s.lookup = lambda table, key, column: 456.0
        assert s.fresh(100) == 456.0
        assert s.value(100) == 123.0
        from lifelib.typed_inputs import lookup
        s.lookup = lookup
        s.table = {'keys': [100], 'values': [[789.0]], 'columns': ['value']}
        assert s.fresh(100) == 789.0
        s.table['values'][0][0] = 987.0
        assert s.fresh(100) == 987.0
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                       env=dict(os.environ), check=True)
