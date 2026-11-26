"""A Cell with integer parameters is reached from Python as upstream: no entry object, no gate.

A per-Cell Python entry (and before it a gate object) once stood between a Python call and the compiled
method of a Cell with integer parameters: a ``__getattribute__`` on the Space class and an extra call on
every read from Python. The Space class must have neither, its Cell must be the compiled method itself, and
its values equal live.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx

PROBE = textwrap.dedent('''\
    import json
    from exported_cy import mx_model as m
    s = m.S
    space_type = type(s)
    print(json.dumps({
        'getattribute': '__getattribute__' in vars(space_type),
        'method': type(s.f).__name__,
        'values': [s.f(3), s.g(3, 4), s.g(5)],
    }))
    ''')


def test_a_cell_with_integer_parameters_has_no_python_entry(tmp_path):
    model = mx.new_model('NoEntry')
    try:
        s = model.new_space('S')
        s.new_cells('f', 'lambda t: t * 1.5')
        s.new_cells('g', 'lambda t, u=2: t * 1.5 + u')
        live = [s.f(3), s.g(3, 4), s.g(5)]
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\nm.S.f(3); m.S.g(3, 4); m.S.g(5)\n',
                                        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True, capture_output=True)
    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    assert '_mx_python_entry_' not in source and 'def __getattribute__' not in source
    out = json.loads(subprocess.run([sys.executable, '-c', PROBE], cwd=tmp_path, check=True,
                                    capture_output=True, text=True).stdout)
    assert not out['getattribute'], 'the Space class overrides __getattribute__'
    assert out['method'] in ('builtin_function_or_method', 'method', 'cython_function_or_method'), out['method']
    assert out['values'] == live
