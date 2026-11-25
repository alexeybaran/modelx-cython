"""Enum definitions are selected by traced type.

A Reference whose value is an enum-safe IntEnum class is an enum definition: fixed in the compiled
model and its members are integer literals. No declaration is needed.
"""
from http import HTTPStatus
import json
import subprocess
import sys

import modelx as mx

FORMULAS = {
    'kind': 'lambda t: 200 + t',
    'matches': 'lambda t: kind(t) == Status.OK',
    'code': 'lambda t: 1.0 if kind(t) == Status.NOT_FOUND else 0.5',
}

CHECK = '''\
import sys
from types import SimpleNamespace
from http import HTTPStatus
from exported import mx_model as original
from exported_cy import mx_model as compiled
o, c = original.Projection, compiled.Projection
for t in range(0, 300, 7):
    for name in ('matches', 'code'):
        assert getattr(o, name)(t) == getattr(c, name)(t), (name, t)
assert c.Status is HTTPStatus
try:
    c.Status = SimpleNamespace(OK=201)
except AttributeError:
    pass
else:
    raise AssertionError('an IntEnum Reference was rebindable in the compiled model')
print('enum by type ok')
'''


def test_enum_selected_by_type(tmp_path):
    model = mx.new_model('EnumByType')
    try:
        s = model.new_space('Projection', refs={'Status': HTTPStatus, 'scale': 2})
        for name, formula in FORMULAS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nfor t in range(0, 300, 7): m.Projection.matches(t); m.Projection.code(t)\n',
        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    generated = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    # members are integer literals in the formulas
    assert '== 200' in generated and '== 404' in generated, generated
    assert 'Status.OK' not in generated and 'Status.NOT_FOUND' not in generated, generated
    (tmp_path / 'check.py').write_text(CHECK, encoding='utf-8')
    result = subprocess.run([sys.executable, 'check.py'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'enum by type ok' in result.stdout
