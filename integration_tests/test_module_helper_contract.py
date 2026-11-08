"""Standard compiler path for module helpers, aliases and builtin reductions."""
import ast
import json
import re
import subprocess
import sys
import textwrap

import modelx as mx
from lifelib import typed_inputs
from lifelib.immutable_inputs import Table


def test_module_helpers_compile_through_shared_metadata(tmp_path):
    model = mx.new_model('ModuleHelpers')
    try:
        p = model.new_space('Projection', refs={'ops': typed_inputs, 'alias_ops': typed_inputs,
            'values': (1e16, 1.0, -1e16), 'count_ref': 3,
            'step_ref': 2, 'state_table': Table(('AFT_INT', 'BEF_INT'), ((1.0,), (2.0,)), ('value',)),
            'key_table': Table((0, 1, 2), ((0.0,), (1.0,), (2.0,)), ('value',))})
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
        p.new_cells('scaled_every', 'lambda n: ops.all_true(abs(term(n, i)) <= 1e-10 * max(term(n, 1), 1.0, 0.0) for i in range(count()))')
        p.new_cells('key_total', 'lambda n: ops.sum_f64(term(n, i) for i in ops.keys(key_table))')
        p.new_cells('key_every', 'lambda n: all(i >= 0 for i in ops.keys(key_table))')
        p.new_cells('key_minimum', 'lambda n: min(i for i in ops.keys(key_table) if i != 1)')
        p.new_cells('key_local', 'def key_local(n):\n    i = key_table\n    return all(i >= 0 for i in ops.keys(i))\n')
        p.new_cells('captured_every', 'def captured_every(n):\n    scale = max(term(n, 1), 1.0)\n    return all(abs(term(n, i)) <= 1e-10 * scale for i in range(count()))\n')
        p.new_cells('stepped_some', 'lambda n: any(term(n, i) > 0.0 for i in range(0, count(), step_ref))')
        p.new_cells('stepped_minimum', 'lambda n: min(i for i in range(0, count(), step_ref))')
        p.new_cells('term_at', 'def term_at(n, i, timing):\n    if timing == \"AFT_INT\":\n        return term(n, i)\n    return 0.0\n')
        p.new_cells('timed_some', 'lambda n: any(term_at(n, i, \"AFT_INT\") > 0.0 for i in range(count()))')
        p.new_cells('state_total', 'lambda n: ops.sum_f64(term_at(n, i, \"AFT_INT\") for state in (\"DEATH\", \"LAPSE\") for i in range(count()))')
        p.new_cells('state_some', 'lambda n: any(term_at(n, 1, state) > 0.0 for state in (\"AFT_INT\", \"BEF_INT\"))')
        p.new_cells('tuple_minimum', 'lambda n: min(i for i in (-9223372036854775808, 0, 9223372036854775807))')
        p.new_cells('lookup_total', 'lambda n: ops.sum_f64(float(ops.lookup(key_table, i, \"value\")) for i in ops.keys(key_table))')
        p.new_cells('lookup_filtered', 'lambda n: all(term(n, i) > 0.0 for i in range(count()) if float(ops.lookup(key_table, i, \"value\")) > 0.0)')
        p.new_cells('lookup_states', 'lambda n: ops.sum_f64(float(ops.lookup(state_table, state, \"value\")) for state in (\"AFT_INT\", \"BEF_INT\"))')
        p.new_cells('lookup_bounded', 'lambda n: ops.sum_f64(float(ops.lookup(key_table, max(i, 0), \"value\")) for i in range(count()))')
        p.new_cells('mx_context_lookup_declined', 'lambda: 42')
        p.new_cells('lookup_declined', 'lambda n: ops.sum_f64(float(ops.lookup(key_table, i, \"value\")) for i in range(count()))')
        p.new_cells('assigned_check', 'def assigned_check(n):\n    ok = ops.all_true(term(n, i) > 0.0 for i in range(count()))\n    return (ok,)\n')
        p.new_cells('conditional_check', 'def conditional_check(n):\n    if ops.all_true(term(n, i) > 0.0 for i in range(count())):\n        return 1\n    return 2\n')
        p.new_cells('arithmetic_total', 'lambda n: 1.0 + ops.sum_f64(term(n, i) for i in range(count()))')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    methods = ('total', 'alias_total', 'every', 'wrapped_every', 'some', 'builtin_every', 'minimum', 'maximum', 'scaled_every', 'key_total', 'key_every', 'key_minimum', 'key_local', 'captured_every', 'stepped_some', 'stepped_minimum', 'timed_some', 'state_total', 'state_some', 'tuple_minimum', 'lookup_total', 'lookup_filtered', 'lookup_states', 'lookup_bounded', 'assigned_check', 'conditional_check', 'arithmetic_total')
    (tmp_path / 'sample.py').write_text('from exported import mx_model\np = mx_model.Projection\n'
        + ''.join(f'for n in range(3): p.{name}(n)\n' for name in methods)
        + 'for n in range(3): p.lookup_declined(n)\n')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name in methods:
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 1
    declined = report['exported_cy._mx_classes._c_Projection._f_lookup_declined']
    assert declined['native_reductions'] == 0
    assert declined['native_numeric_helpers'] == []
    syntax = ast.parse((tmp_path / 'exported_cy/_mx_classes.py').read_text())
    c_source = (tmp_path / 'exported_cy/_mx_classes.c').read_text()
    for name in ('key_total', 'key_every', 'key_minimum', 'key_local', 'captured_every',
                 'state_total', 'state_some', 'tuple_minimum', 'timed_some',
                 'lookup_total', 'lookup_filtered', 'lookup_states', 'lookup_bounded'):
        method = next(node for node in ast.walk(syntax)
                      if isinstance(node, ast.FunctionDef) and node.name == '_f_' + name)
        native = method.body[0].value.func.id
        signatures = list(re.finditer(r'^static [^\n]*__pyx_f_[^\n]*' + re.escape(native)
                                      + r'\([^\n]*\) \{', c_source, re.MULTILINE))
        assert signatures, native
        for signature in signatures:
            body = c_source[signature.start():c_source.index('\n}\n', signature.end())]
            assert 'for (' in body
            if name.startswith('key_'):
                assert 'key_span_at(' in body
            if name.startswith('lookup_'):
                operation = (typed_inputs.lookup.__modelx_cython_keyed_float_numeric__[1]
                             if name == 'lookup_states' else typed_inputs.lookup.__modelx_cython_float_numeric__)
                assert operation + '(' in body
            assert not any(token in body for token in
                           ('PyObject_GetIter(', '__Pyx_PyObject_GetIterNextFunc(', 'PyIter_Next('))
    probe = textwrap.dedent('''\
        from importlib import import_module
        import sys
        from types import SimpleNamespace
        from lifelib import typed_inputs
        model = import_module(sys.argv[1]).mx_model
        def fresh(): return type(model)().Projection
        p = fresh()
        assert p.total(0) == p.alias_total(0) == p.key_total(0) == 1.0
        assert p.key_every(0) is True and p.key_minimum(0) == 0
        assert p.key_local(0) is True
        assert p.every(0) is False and p.some(0) is True
        assert p.scaled_every(0) is False
        assert p.captured_every(0) is False
        assert p.stepped_some(0) is True and p.stepped_minimum(0) == 0
        assert p.timed_some(0) is True
        assert p.state_total(0) == 2.0
        assert p.state_some(0) is True
        assert p.tuple_minimum(0) == -(1 << 63)
        assert p.lookup_total(0) == p.lookup_states(0) == p.lookup_bounded(0) == 3.0
        assert p.lookup_filtered(0) is False
        assert p.lookup_declined(0) == 3.0
        assert p.assigned_check(0) == (False,)
        assert p.conditional_check(0) == 2 and p.arithmetic_total(0) == 2.0
        assert p.wrapped_every(0) is False
        assert p.builtin_every(0) is False
        assert p.minimum(0) == 0 and p.maximum(0) == 2
        q = fresh()
        q.step_ref = 0
        try: q.stepped_some(0)
        except ValueError as error: assert str(error) == 'range() arg 3 must not be zero'
        else: raise AssertionError('zero step did not raise eagerly')
        q = fresh()
        q.step_ref = -2
        assert q.stepped_some(0) is False
        try: q.stepped_minimum(0)
        except ValueError: pass
        else: raise AssertionError('empty stepped minimum did not raise')
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
        p = fresh()
        key_seen = []
        def replacement_keys(table):
            key_seen.append(table)
            return iter((2, 0))
        p.ops = SimpleNamespace(keys=replacement_keys, sum_f64=typed_inputs.sum_f64)
        assert p.key_total(1) == 0.0
        assert key_seen == [p.key_table]
        error = StopIteration('outer iterator')
        def broken_keys(table): raise error
        p.ops = SimpleNamespace(keys=broken_keys)
        try: p.key_every(1)
        except StopIteration as actual: assert actual is error
        else: raise AssertionError('outer StopIteration changed')
        p = fresh()
        lookup_seen = []
        def replacement_lookup(table, key, column):
            lookup_seen.append((key, column))
            return '2.5'
        p.ops = SimpleNamespace(sum_f64=typed_inputs.sum_f64, keys=typed_inputs.keys, lookup=replacement_lookup)
        assert p.lookup_total(1) == 7.5
        assert lookup_seen == [(0, 'value'), (1, 'value'), (2, 'value')]
        error = StopIteration('lookup body')
        def broken_lookup(table, key, column): raise error
        p.ops = SimpleNamespace(sum_f64=typed_inputs.sum_f64, keys=typed_inputs.keys, lookup=broken_lookup)
        try: p.lookup_total(2)
        except RuntimeError as actual:
            assert actual.__cause__ is error and actual.__context__ is error
        else: raise AssertionError('lookup StopIteration not converted')
        p = fresh()
        marker = object()
        p.ops = SimpleNamespace(all_true=lambda values: marker)
        assert p.assigned_check(1)[0] is marker
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
