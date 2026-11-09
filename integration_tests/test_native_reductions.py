"""Explicit native reductions preserve the exported helper boundary."""
import json
import subprocess
import sys
import textwrap

import modelx as mx
from lifelib.typed_inputs import all_true, max_value, sum_f64


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
    for name in ('total', 'nested', 'filtered'):
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 0
    from compiled_evidence import assert_compiled_generator_formulas
    assert_compiled_generator_formulas(tmp_path, ('total','nested','filtered'))
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


def test_explicit_all_and_max(tmp_path):
    model = mx.new_model('PredicateAndMaximum')
    try:
        s = model.new_space('Projection', refs={
            'all_true': all_true, 'max_value': max_value, 'tolerance': 1.0,
            'values': (2.0, 0.0, 1.0, float('nan'), -0.0, 2.0, 4.0, 3.0),
        })
        s.new_cells('one', 'lambda: 1')
        s.new_cells('two', 'lambda: 2')
        s.new_cells('zero', 'lambda: 0')
        s.new_cells('first', 'lambda t: values[0] if t == 0 else values[100]')
        s.new_cells('term', 'lambda t: values[t % 8]')
        s.new_cells('all_short', 'lambda: all_true(abs(first(t)) < tolerance for t in range(two()))')
        s.new_cells('all_empty', 'lambda: all_true(abs(first(t)) <= tolerance for t in range(zero()))')
        s.new_cells('all_replace', 'lambda: all_true(abs(term(t)) < tolerance for t in range(one()))')
        s.new_cells('maximum', textwrap.dedent('''\
            def maximum(t):
                if t == -100000:
                    return 0.0
                return max_value(term(u) for u in range(t, max(-1, t - 12), -3))
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(textwrap.dedent('''\
        from exported import mx_model
        p = mx_model.Projection
        p.all_short(); p.all_empty(); p.all_replace()
        p.zero(); p.term(-100000); p.maximum(-100000); p.maximum(0); p.maximum(5)
        '''), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name in ('all_short', 'all_empty', 'all_replace', 'maximum'):
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 0
    probe = textwrap.dedent('''\
        import math
        import sys
        from importlib import import_module

        s = import_module(sys.argv[1]).mx_model.Projection
        assert s.all_short() is False
        assert s.all_empty() is True
        assert math.isnan(s.maximum(3))
        assert s.maximum(6) == 4.0
        assert math.copysign(1.0, s.maximum(4)) == -1.0
        upper = (1 << 63) - 1
        assert s.maximum(upper) == 4.0
        try:
            max(())
        except ValueError as error:
            empty_message = str(error)
        for negative in (-1, -(1 << 63)):
            try:
                s.maximum(negative)
            except ValueError as error:
                assert str(error) == empty_message
            else:
                raise AssertionError('expected empty maximum to fail')

        def replace_all(values):
            assert iter(values) is values
            assert list(values) == [False]
            return True

        def replace_max(values):
            assert iter(values) is values
            assert list(values) == [2.0, 1.0]
            return 9.0

        s.all_true = replace_all
        s.max_value = replace_max
        assert s.all_replace() is True
        assert s.maximum(5) == 9.0
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
