"""Temporary explicit-sum lowering preserves the exported helper boundary."""
import json
import subprocess
import sys
import textwrap

import modelx as mx
from lifelib.typed_inputs import sum_f64


def test_explicit_numeric_sum(tmp_path):
    model = mx.new_model('NumericReduction')
    try:
        s = model.new_space('Projection', refs={'sum_f64': sum_f64,
                           'values': (1e16, 1.0, -1e16)})
        s.new_cells('count', 'lambda: 3')
        s.new_cells('term', 'lambda t, k: values[k] + t * 0.0')
        s.new_cells('total', 'lambda t: sum_f64(term(t, k) for k in range(count()))')
        s.new_cells('nested', 'lambda t: sum_f64(term(t, fund) for k in range(count()) for fund in (k,))')
        s.new_cells('filtered', 'lambda t: sum_f64(term(t, k) for k in range(count()) if k != 1)')
        assert s.total(0) == s.nested(0) == 1.0
        model.export(tmp_path/'exported')
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model as m\n'
        'for t in range(5):\n    m.Projection.total(t); m.Projection.nested(t); m.Projection.filtered(t)\n')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    for name in ('total', 'nested'):
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 1
    assert report['exported_cy._mx_classes._c_Projection._f_filtered']['native_reductions'] == 0
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        from lifelib.typed_inputs import sum_f64
        s = import_module(sys.argv[1]).mx_model.Projection
        assert s.total(0) == s.nested(0) == 1.0
        assert s.filtered(0) == 0.0
        seen = []
        def replacement(values):
            seen.extend(values)
            return 42.0
        s.sum_f64 = replacement
        assert s.total(0) == 1.0
        assert s.total(1) == s.nested(1) == 42.0
        assert seen == [1e16, 1.0, -1e16] * 2
        error = ValueError('replacement')
        def broken(values):
            next(values)
            raise error
        s.sum_f64 = broken
        try:
            s.total(2)
        except ValueError as actual:
            assert actual is error
        else:
            raise AssertionError('exception swallowed')
        s.sum_f64 = sum_f64
        assert s.total(2) == 1.0
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
