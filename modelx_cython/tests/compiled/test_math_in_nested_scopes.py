"""A math call inside a lambda or comprehension of a formula reads the names of that inner scope.

``math.ceil(x)`` / ``math.floor(x)`` take a C-integer form when ``x`` is a typed parameter of the
formula. Inside ``lambda x: math.ceil(x)`` the ``x`` is the lambda's own parameter, of any type: the
call must not take the formula parameter's type."""
import math
import subprocess
import sys

import modelx as mx


def test_lambda_parameter_shadowing_a_typed_formula_parameter(tmp_path):
    model = mx.new_model('NestedMath')
    try:
        space = model.new_space('Sp', refs={'math': math})
        space.new_cells('up', 'def up(x):\n    f = lambda x: math.ceil(x)\n    return f(x / 2) + math.ceil(x)\n')
        space.new_cells('down', 'def down(x):\n    return sum(math.floor(x) for x in (x * 0.5, x * 1.5))\n')
        calls = [('up', 3), ('up', 7), ('down', 3), ('down', 5)]
        live = repr([getattr(space, name)(x) for name, x in calls])
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        f'for name, x in {calls!r}: getattr(m.Sp, name)(x)\n', encoding='utf-8')
    build = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                            '--no-spec'], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([sys.executable, '-B', '-c',
                          'from exported_cy import mx_model as m\n'
                          f'print(repr([getattr(m.Sp, name)(x) for name, x in {calls!r}]))'],
                         cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip().splitlines()[-1] == live
