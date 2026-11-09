"""Explicit helper ABI through ordinary model export and compiler entry."""
import json
import os
import subprocess
import sys
import textwrap

import modelx as mx
from lifelib import expression_streams as helpers
from lifelib import expressions, pandas_inputs, typed_inputs
import pandas as pd


def test_explicit_streams_without_legacy_ast_generator_passes(tmp_path):
    model = mx.new_model('ExplicitStreams')
    try:
        p = model.new_space('Projection', refs={name: getattr(helpers, name) for name in
            ('periods', 'over', 'flatten', 'sum_float')})
        p.expressions = expressions
        p.rates = pandas_inputs.table(pd.DataFrame({'rate': [1.0, 0.5]}, index=[0, 1]))
        p.new_cells('cf', 'lambda t: 1.0 + 0.5*t')
        p.new_cells('disc', 'lambda t: 1.0 / (1.0 + 0.01*t)')
        p.new_cells('eligible', 'lambda t: t % 2 == 0')
        p.new_cells('total', 'lambda n: sum_float(periods(n), lambda t: cf(t)*disc(t), where=lambda t: eligible(t))')
        p.new_cells('composed', 'lambda n: sum_float(over(periods(n), lambda t: cf(t)*disc(t)))')
        p.new_cells('generated', 'lambda n: sum_float(cf(t)*disc(t) for t in range(n))')
        p.new_cells('facade_generated', 'lambda n: expressions.sum_float(cf(t) * expressions.lookup_float(rates, t % 2, "rate") for t in range(n))')
        p.new_cells('filtered_generated', 'lambda n: sum_float(cf(t)*disc(t) for t in range(n) if eligible(t))')
        p.new_cells('bound_callback', 'lambda n: sum_float(periods(n), cf)')
        p.new_cells('method_pairs', '''def method_pairs(n):
    total = 0.0
    for first, second in ((cf, disc), (cf, disc)):
        total += first(n) + second(n)
    return total
''')
        p.new_cells('where_only', 'lambda n: sum_float(periods(n), where=lambda t: eligible(t))')
        p.new_cells('nested', 'lambda n: sum_float(flatten(over(periods(n), lambda g: over(periods(g), lambda z: cf(z)))))')
        p.new_cells('pairs', 'lambda: tuple(over(((1, 2), (3, 4)), lambda a, b: (a, b), unpack=2))')
        expected = {}
        for n in [0, 1, 3, 10]:
            expected[str(n)] = {name: getattr(p, name)(n) for name in
                               ('total', 'composed', 'generated', 'facade_generated', 'filtered_generated', 'bound_callback', 'method_pairs', 'where_only', 'nested')}
        assert p.pairs() == ((1, 2), (3, 4))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'expected.json').write_text(json.dumps(expected))
    (tmp_path / 'sample.py').write_text('''from exported import mx_model
p = mx_model.Projection
for n in [0,1,3,10]:
    p.total(n); p.composed(n); p.generated(n); p.facade_generated(n); p.filtered_generated(n); p.bound_callback(n); p.method_pairs(n); p.where_only(n); p.nested(n)
p.pairs()
''')
    runner = textwrap.dedent('''
        import sys
        from modelx_cython import cli, reductions, numeric_helpers
        def forbidden(*args, **kwargs):
            raise AssertionError('legacy AST generator/classifier activation')
        reductions.NativeReductions = forbidden
        numeric_helpers.ContextualFloat = forbidden
        raise SystemExit(cli.main(['exported','--sample','sample.py','--no-spec'],sys.stdout,sys.stderr))
    ''')
    run = subprocess.run([sys.executable, '-c', runner], cwd=tmp_path,
                         capture_output=True, text=True, env=dict(os.environ))
    (tmp_path / 'build.log').write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text()
    assert 'sum_float(' in source and '.omitted_argument' in source
    assert '_mx_cold_sum_' not in source
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    assert all(cell['native_reductions'] == 0 for cell in report.values())
    probe = textwrap.dedent('''
        import importlib, json, sys
        from lifelib import expression_streams as h, expressions as e, typed_inputs as tables
        model = importlib.import_module(sys.argv[1]).mx_model
        expected = json.load(open('expected.json'))
        forbidden = {getattr(h, name).__code__ for name in
                     ('periods','over','flatten','sum_float')}
        forbidden |= {h._Over.__init__.__code__, h._Over.__next__.__code__,
                      h._Flatten.__init__.__code__, h._Flatten.__next__.__code__}
        forbidden |= {tables.lookup_float.__code__, tables._typed_value.__code__}
        calls = []
        def profile(frame,event,arg):
            if event == 'call' and frame.f_code in forbidden:
                calls.append(frame.f_code.co_name)
        if sys.argv[1].endswith('_cy'): sys.setprofile(profile)
        try:
            p = type(model)().Projection
            for n, values in expected.items():
                for name, value in values.items():
                    assert getattr(p,name)(int(n)) == value
            assert p.pairs() == ((1,2),(3,4))
        finally: sys.setprofile(None)
        assert calls == [], calls
        events=[]
        def current(source, function, *, where):
            events.append((source,function,where))
            return 42.5
        p = type(model)().Projection
        p.sum_float = current
        assert p.total(3) == 42.5 and len(events) == 1
        p = type(model)().Projection
        def consume_generator(values):
            first = values.send(None)
            return first + sum(values)
        p.sum_float = consume_generator
        assert p.generated(3) == expected['3']['generated']
        original = ValueError('injected')
        def throw_into_generator(values):
            try: values.throw(original)
            except ValueError as raised: assert raised is original
            else: raise AssertionError('generator throw did not propagate')
            assert list(values) == []
            return 42.5
        p = type(model)().Projection
        p.sum_float = throw_into_generator
        assert p.generated(3) == 42.5
        original_sum = e.sum_float
        p = type(model)().Projection
        try:
            e.sum_float = lambda values: 17.5 + next(values)
            assert p.facade_generated(3) == 18.5
        finally:
            e.sum_float = original_sum
        root = type(model)()
        class Override(type(root.Projection)):
            def cf(self,t): return 7.5
        p = Override(root)
        p._mx_copy_refs(root.Projection, root.Projection)
        assert p.total(3) == sum(7.5/(1.0+0.01*t) for t in [0,2])
    ''')
    for package in ['exported', 'exported_cy']:
        result = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                                capture_output=True, text=True, env=dict(os.environ))
        (tmp_path / (package+'-validation.log')).write_text(result.stdout+result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
