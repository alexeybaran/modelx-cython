"""Differential boundaries for composed native boolean reductions."""
import ast
import json
import re
import subprocess
import sys
import textwrap

import modelx as mx
from lifelib.typed_inputs import all_true, any_true


def test_nested_filtered_boolean_ranges(tmp_path):
    model = mx.new_model('BooleanRanges')
    try:
        s = model.new_space('Projection', refs={
            'all_true': all_true, 'any_true': any_true, 'events': [],
            'throw_at': -1, 'false_at': 2, 'error': StopIteration('sentinel')})
        s.new_cells('bound', '''def bound(n):
    events.append(('bound', n))
    if n == -9:
        raise error
    return n
''')
        s.new_cells('inner', '''def inner(i):
    events.append(('inner', i))
    return 3
''')
        s.new_cells('predicate', '''def predicate(i):
    events.append(('term', i))
    if i == throw_at:
        raise error
    return i != false_at
''')
        s.new_cells('every', 'lambda n: all_true(predicate(i) for i in range(bound(n)) if i != 1)')
        s.new_cells('some', 'lambda n: any_true(predicate(j) for i in range(bound(n)) if i != 1 for j in range(inner(i)) if j != 0)')
        s.new_cells('reverse', 'lambda n: all_true(predicate(i) for i in range(bound(n), -1, -1) if i != 1)')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\np = mx_model.Projection\n'
        'p.every(8); p.some(8); p.reverse(8)\n'
        'p.every(-10); p.some(-10); p.reverse(-10)\n')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name in ('every', 'some', 'reverse'):
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 1
    generated = tmp_path / 'exported_cy/_mx_classes.py'
    c_source = generated.with_suffix('.c').read_text()
    native_bodies = 0
    for function in ast.parse(generated.read_text()).body:
        if not isinstance(function, ast.FunctionDef):
            continue
        if not any(isinstance(node, ast.Attribute) and node.attr in
                   ('all_true_accepts', 'any_true_accepts') for node in ast.walk(function)):
            continue
        pattern = r'^static [^\n]*' + re.escape(function.name) + r'\([^\n]*\) \{'
        for signature in re.finditer(pattern, c_source, re.MULTILINE):
            body = c_source[signature.start():c_source.index('\n}\n', signature.end())]
            assert 'for (' in body
            assert not any(token in body for token in
                           ('PyObject_GetIter(', '__Pyx_PyObject_GetIterNextFunc(', 'PyIter_Next('))
            native_bodies += 1
    assert native_bodies == 3
    probe = textwrap.dedent('''\
        import sys
        import pickle
        from importlib import import_module
        model = import_module(sys.argv[1]).mx_model
        def fresh():
            return pickle.loads(pickle.dumps(model)).Projection
        s = fresh()
        s.events.clear()
        assert s.every(4) is False
        assert s.events == [('bound', 4), ('term', 0), ('term', 2)]
        s = fresh()
        assert s.some(4) is True
        assert s.events == [('bound', 4), ('inner', 0), ('term', 1)]
        s = fresh()
        assert s.every(0) is True
        assert s.events == [('bound', 0)]
        s = fresh()
        assert s.some(0) is False
        assert s.events == [('bound', 0)]
        s = fresh()
        assert s.reverse(4) is False
        assert s.events == [('bound', 4), ('term', 4), ('term', 3), ('term', 2)]
        s = fresh()
        s.throw_at = 0
        try:
            s.every(5)
        except RuntimeError as error:
            assert str(error) == 'generator raised StopIteration'
            assert error.__cause__ is s.error
            assert error.__context__ is s.error
        else:
            raise AssertionError('body StopIteration not converted')
        try:
            s.every(-9)
        except StopIteration as error:
            assert error is s.error
        else:
            raise AssertionError('outer bound StopIteration changed')
        s = fresh()
        def replacement(values):
            assert iter(values) is values
            assert s.events == [('bound', 6)]
            assert next(values) is True
            assert next(values) is False
            assert next(values) is True
            assert next(values) is True
            assert next(values) is True
            assert list(values) == []
            assert list(values) == []
            return True
        s.all_true = replacement
        assert s.every(6) is True
        assert s.events == [('bound', 6), ('term', 0), ('term', 2),
                            ('term', 3), ('term', 4), ('term', 5)]
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
