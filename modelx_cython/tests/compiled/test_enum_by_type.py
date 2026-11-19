"""Enum definitions are selected by traced type; the former declaration marker is accepted and ignored.

A Reference whose value is an enum-safe IntEnum class is an enum definition: fixed in the compiled
model and its members are C constants. No declaration is needed. A model saved with the former
``EnumDefinitions`` Reference still builds: the marker is an ordinary Reference, and a name it lists
that is not an IntEnum Reference is no longer an error. Both builds select the same enums.
"""
from http import HTTPStatus
import json
import subprocess
import sys

import pytest

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


@pytest.mark.parametrize('legacy_marker', [False, True])
def test_enum_selected_by_type(tmp_path, legacy_marker):
    model = mx.new_model('EnumByType')
    try:
        s = model.new_space('Projection', refs={'Status': HTTPStatus, 'scale': 2})
        if legacy_marker:
            from modelx_helpers.typed_inputs import EnumDefinitions
            s.enum_definitions = EnumDefinitions(('Status', 'scale'))   # 'scale' is not an enum: ignored
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
    report = json.loads((tmp_path / 'exported_cy' / '_mx_admission.json').read_text(encoding='utf-8'))
    constants = {key.rsplit('._f_', 1)[-1]: entry['native_enum_constants'] for key, entry in report.items()
                 if isinstance(entry, dict) and 'native_enum_constants' in entry}
    assert constants['matches'] >= 1 and constants['code'] >= 1, constants
    (tmp_path / 'check.py').write_text(CHECK, encoding='utf-8')
    result = subprocess.run([sys.executable, 'check.py'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'enum by type ok' in result.stdout
