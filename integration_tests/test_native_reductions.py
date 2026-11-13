"""Explicit native reductions preserve the exported helper boundary."""
import json
import subprocess
import sys
import textwrap

import modelx as mx
from modelx_helpers.typed_inputs import all_true, max_value, sum_f64


def test_ordinary_sum_at_double_boundary(tmp_path):
    from modelx_helpers import expression_streams as streams
    model = mx.new_model('OrdinaryDoubleReduction')
    try:
        s = model.new_space('Projection', refs={'streams': streams,
            'values': {'A': 1e16, 'B': 1.0, 'C': -1e16, None: 2.0, 1: 3.0}})
        s.new_cells('term', 'lambda t, k: values[k] + t * 0.0')
        for name, domain in (('labels', '("A", "B", "C")'),
                             ('single', '("B",)'), ('mixed', '(None, 1, "B")')):
            s.new_cells(name, f'lambda t: streams.sum_values(term(t, k) for k in {domain})')
        for variable in ('state', 'function', 'iterable', 'error', 'bound_0'):
            s.new_cells('named_' + variable,
                f'lambda t: streams.sum_values(term(t, {variable}) for {variable} in ("A", "B", "C"))')
        s.new_cells('captured_state', '''def captured_state(t):
    state = 0.5
    return streams.sum_values(term(t, k) * state for k in ("A", "B", "C"))
''')
        s.new_cells('empty', '''def empty(t):
    if t == -1:
        return 0.0
    return streams.sum_values(term(t, k) for k in ())
''')
        s.new_cells('assigned', '''def assigned(t):
    result = streams.sum_values(term(t, k) for k in ("A", "B", "C"))
    return result
''')
        s.new_cells('integers', 'lambda t: streams.sum_values(k for k in (1, 2, 3))')
        s.new_cells('large', 'lambda t: streams.sum_values(k for k in (2**100, 2**100))')
        s.new_cells('object_total', '''def object_total(t) -> object:
    return streams.sum_values(term(t, k) for k in ("A", "B", "C"))
''')
        s.new_cells('nested_labels', 'lambda t: streams.sum_values(term(t, k) for j in ("outer",) for k in ("A", "B", "C"))')
        s.new_cells('filtered_labels', 'lambda t: streams.sum_values(term(t, k) for k in ("A", "B", "C") if t >= 0)')
        s.new_cells('label_filter', 'lambda t: streams.sum_values(term(t, k) for k in ("A", "B", "C") if k != "B")')
        s.new_cells('literal_key', 'lambda: 1000')
        s.new_cells('identity_term', 'lambda t, k: 1.0 if k is literal_key() else 0.0')
        s.new_cells('identity_sum', 'lambda t: streams.sum_values(identity_term(t, k) for k in (1000, 1001))')
        s.new_cells('identity_assigned', '''def identity_assigned(t):
    result = streams.sum_values(identity_term(t, k) for k in (1000, 1001))
    return result
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\n'
        'for t in (-1, 0):\n'
        '    for name in ("labels", "single", "mixed", "empty", "assigned", "integers", "large",\n'
        '                 "object_total", "nested_labels", "filtered_labels", "label_filter", "identity_sum", "identity_assigned"):\n'
        '        getattr(m.Projection, name)(t)\n'
        '    for variable in ("state", "function", "iterable", "error", "bound_0"):\n'
        '        getattr(m.Projection, "named_" + variable)(t)\n'
        '    m.Projection.captured_state(t)\n'
        'm.Projection.labels(11)\n')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name in ('labels', 'single', 'mixed', 'empty', 'nested_labels', 'filtered_labels', 'identity_sum',
                 'named_state', 'named_function', 'named_iterable', 'named_error', 'named_bound_0', 'captured_state'):
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 1
    for name in ('assigned', 'integers', 'large', 'object_total', 'label_filter', 'identity_assigned'):
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 0
    probe = textwrap.dedent('''\
        import math, sys
        from importlib import import_module
        from types import SimpleNamespace
        from modelx_helpers import expression_streams
        model = import_module(sys.argv[1]).mx_model
        s = model.Projection
        assert s.labels(0) == s.assigned(0) == 1.0
        assert s.single(0) == 1.0 and s.mixed(0) == 6.0
        assert s.empty(0) == 0.0
        assert type(s.integers(0)) is int and s.integers(0) == 6
        assert type(s.large(0)) is int and s.large(0) == 2**101
        assert s.nested_labels(0) == s.filtered_labels(0) == 1.0
        assert s.filtered_labels(-1) == s.label_filter(0) == 0.0
        assert s.identity_sum(0) == s.identity_assigned(0)
        for variable in ('state', 'function', 'iterable', 'error', 'bound_0'):
            assert getattr(s, 'named_' + variable)(0) == 1.0
        assert s.captured_state(0) == 0.5
        class Override(type(s)):
            def term(self, t, k): return 7.5
        override = Override(model)
        override._mx_copy_refs(s, s)
        assert override.labels(0) == 22.5
        # Native arithmetic must retain the interpreter's compensated result.
        for t, values in enumerate(((1e16, 1., -1e16), (float('inf'), 1., float('-inf')),
                                    (-0., -0., -0.), (float('nan'), 1., 2.)), 1):
            s.values = dict(zip(('A', 'B', 'C'), values))
            expected = sum(v + t * 0.0 for v in values)
            actual = s.labels(t)
            assert math.isnan(actual) if math.isnan(expected) else actual == expected
            if actual == 0.: assert math.copysign(1., actual) == math.copysign(1., expected)
        held = []
        def replacement(source):
            assert iter(source) is source
            held.append(source)
            return 42.0
        s.values = {'A': 1e16, 'B': 1., 'C': -1e16}
        s.streams = SimpleNamespace(sum_values=replacement)
        assert s.labels(10) == 42.0
        assert list(held.pop()) == [1e16, 1., -1e16]
        s.streams = expression_streams
        error = StopIteration('term')
        class Broken(dict):
            def __getitem__(self, key): raise error
        s.values = Broken()
        try: s.labels(11)
        except RuntimeError as actual:
            assert actual.__cause__ is error and actual.__context__ is error
        else: raise AssertionError('generator StopIteration was not wrapped')
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)


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
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 1
    from compiled_evidence import assert_compiled_generator_formulas
    assert_compiled_generator_formulas(tmp_path, ('total','nested','filtered'))
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        from modelx_helpers.typed_inputs import sum_f64
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


def test_builtin_minimum_of_typed_range(tmp_path):
    model = mx.new_model('BuiltinMinimum')
    try:
        s = model.new_space('Projection', refs={'values': (1.0, 2.0)})
        s.new_cells('value', 'lambda t, case: values[t]')
        s.new_cells('minimum', 'lambda n, case: min(value(t, case) for t in range(n))')
        s.new_cells('count', 'lambda: 2')
        s.new_cells('positive', 'lambda t: value(t, 0) > 0')
        s.new_cells('all_positive', 'lambda: all(positive(t) for t in range(count()))')
        s.new_cells('first', 'lambda t: t')
        s.new_cells('any_positive', '''def any_positive(t):
    first_index = first(t)
    return any(positive(u) for u in range(max(0, first_index), min(first_index + 2, count())))
''')
        s.new_cells('term', 'lambda t: value(t, 0)')
        s.new_cells('maximum', 'lambda t: max(term(u) for u in range(t, max(-1, t - 12), -3))')
        s.new_cells('total', 'lambda n: sum(value(t, 0) for t in range(n))')
        s.new_cells('float_total', '''def float_total(n):
    total = sum((value(t, 0) for t in range(n)), 0.0)
    return total
''')
        s.new_cells('shadowed', '''def shadowed(n):
    min = max
    return min(value(t, 0) for t in range(n))
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\n'
        'mx_model.Projection.minimum(2, 0)\nmx_model.Projection.shadowed(2)\n'
        'mx_model.Projection.all_positive()\nmx_model.Projection.total(2)\n'
        'mx_model.Projection.any_positive(0)\nmx_model.Projection.any_positive(2)\n'
        'mx_model.Projection.maximum(1)\nmx_model.Projection.minimum(2, 6)\n'
        'mx_model.Projection.float_total(2)\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name, count in (('minimum', 1), ('shadowed', 0), ('all_positive', 1), ('total', 1),
                        ('any_positive', 1), ('maximum', 1), ('float_total', 1)):
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == count
    probe = textwrap.dedent('''\
        import math
        import sys
        from importlib import import_module

        model = import_module(sys.argv[1]).mx_model
        s = model.Projection
        assert s.minimum(2, 0) == 1.0 and s.shadowed(2) == 2.0
        assert s.all_positive() is True and s.total(2) == 3.0
        assert s.any_positive(1) is True and s.any_positive(2) is False
        assert s.maximum(0) == 1.0 and s.maximum(1) == 2.0
        assert s.float_total(2) == 3.0 and type(s.float_total(0)) is float
        values_by_case = ((-0., 0.), (0., -0.), (float('nan'), 1.), (1., float('nan')),
                          (float('inf'), float('-inf')))
        for case, values in enumerate(values_by_case, 1):
            s.values = values
            expected = min(values)
            actual = s.minimum(2, case)
            assert math.isnan(actual) if math.isnan(expected) else actual == expected
            if actual == 0.:
                assert math.copysign(1., actual) == math.copysign(1., expected)
        try:
            min(())
        except ValueError as error:
            empty_message = str(error)
        try:
            s.minimum(0, 0)
        except ValueError as error:
            assert str(error) == empty_message
        else:
            raise AssertionError('expected empty minimum to fail')
        error = StopIteration('value')
        class Broken:
            def __getitem__(self, index):
                raise error
        s.values = Broken()
        try:
            s.minimum(2, 6)
        except RuntimeError as actual:
            assert actual.__cause__ is error
        else:
            raise AssertionError('generator StopIteration was not wrapped')
        class Override(type(s)):
            def value(self, t, case):
                return 7.5 - t
        override = Override(model)
        override._mx_copy_refs(s, s)
        assert override.minimum(2, 0) == 6.5
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
        p.zero(); p.maximum(-100000); p.maximum(0); p.maximum(5); p.maximum(6)
        upper = (1 << 63) - 1
        p.maximum(upper); p.term(-(1 << 63))
        for key in (-1, -(1 << 63)):
            try:
                p.maximum(key)
            except ValueError:
                pass
        '''), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name in ('all_short', 'all_empty', 'all_replace', 'maximum'):
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 1
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
