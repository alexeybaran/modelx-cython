"""Standard compiler path for module helpers, aliases and builtin reductions."""
import json
import subprocess
import sys
import textwrap

import modelx as mx
from lifelib import typed_inputs


def test_module_helpers_compile_through_shared_metadata(tmp_path):
    model = mx.new_model('ModuleHelpers')
    try:
        p = model.new_space('Projection', refs={'ops': typed_inputs, 'alias_ops': typed_inputs,
            'values': (1e16, 1.0, -1e16), 'count_ref': 3})
        p.new_cells('count', 'lambda: count_ref')
        p.new_cells('term', 'lambda n, i: values[i] + n * 0.0')
        p.new_cells('total', 'lambda n: ops.sum_f64(term(n, i) for i in range(count()))')
        p.new_cells('alias_total', 'lambda n: alias_ops.sum_f64(term(n, i) for i in range(count()))')
        p.new_cells('every', 'lambda n: ops.all_true(term(n, i) > 0.0 for i in range(count()))')
        p.new_cells('wrapped_every', 'lambda n: bool(ops.all_true(term(n, i) > 0.0 for i in range(count())))')
        p.new_cells('some', 'lambda n: ops.any_true(term(n, i) > 0.0 for i in range(count()))')
        p.new_cells('builtin_every', 'lambda n: bool(all(term(n, i) > 0.0 for i in range(count())))')
        p.new_cells('minimum', 'lambda n: min(i for i in range(count()) if i != 1)')
        p.new_cells('maximum', 'lambda n: max(i for i in range(count()) if i != 1)')
        p.new_cells('scaled_every', 'lambda n: ops.all_true(abs(term(n, i)) <= 1e-10 * max(term(n, 1), 1.0) for i in range(count()))')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    methods = ('total', 'alias_total', 'every', 'wrapped_every', 'some', 'builtin_every', 'minimum', 'maximum', 'scaled_every')
    (tmp_path / 'sample.py').write_text('from exported import mx_model\np = mx_model.Projection\n'
        + ''.join(f'for n in range(3): p.{name}(n)\n' for name in methods))
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name in methods:
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 1
    probe = textwrap.dedent('''\
        from importlib import import_module
        import sys
        from types import SimpleNamespace
        model = import_module(sys.argv[1]).mx_model
        def fresh(): return type(model)().Projection
        p = fresh()
        assert p.total(0) == p.alias_total(0) == 1.0
        assert p.every(0) is False and p.some(0) is True
        assert p.scaled_every(0) is False
        assert p.wrapped_every(0) is False
        assert p.builtin_every(0) is False
        assert p.minimum(0) == 0 and p.maximum(0) == 2
        seen = []
        def replacement(values):
            assert iter(values) is values
            seen.extend(values)
            assert list(values) == []
            return 42.0
        p.ops = SimpleNamespace(sum_f64=replacement)
        assert p.total(1) == 42.0
        assert seen == [1e16, 1.0, -1e16]
        assert p.alias_total(1) == 1.0
        truth_seen = []
        class Truth:
            def __bool__(self):
                truth_seen.append('truth')
                return True
        p.ops = SimpleNamespace(all_true=lambda values: Truth())
        assert p.wrapped_every(1) is True
        assert truth_seen == ['truth']
        error = StopIteration('consumer')
        def broken(values): raise error
        p.ops = SimpleNamespace(all_true=broken)
        try: p.wrapped_every(2)
        except StopIteration as actual: assert actual is error
        else: raise AssertionError('consumer StopIteration changed')
        p = fresh()
        p.count_ref = 0
        assert p.total(0) == 0.0 and type(p.total(0)) is float
        assert p.every(0) is True and p.some(0) is False
        assert p.builtin_every(0) is True
        for name in ('minimum', 'maximum'):
            try: getattr(p, name)(0)
            except ValueError: pass
            else: raise AssertionError('empty extremum did not raise')
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
