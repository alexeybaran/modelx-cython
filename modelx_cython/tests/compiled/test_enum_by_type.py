"""Enum definitions are selected by traced type.

A Reference whose value is an enum family (here an IntEnum whose values are 0..n-1) is an enum definition:
fixed in the compiled model, and its members are positions where a parameter held for the class is compared with
them. No declaration is needed.
"""
import json
import subprocess
import sys

import modelx as mx

FORMULAS = {
    'kind': 'lambda t: 200 + t',
    'matches': 'lambda t: kind(t) == Status.OK',
    'code': 'lambda t, s: 1.0 if s == Status.NOT_FOUND else 0.5',
}
STATUS = 'import enum\n\nclass Status(enum.IntEnum):\n    OK = 0\n    NOT_FOUND = 1\n'

CHECK = '''\
import sys
from types import SimpleNamespace
from bytype_status import Status
from exported import mx_model as original
from exported_cy import mx_model as compiled
o, c = original.Projection, compiled.Projection
for t in range(0, 300, 7):
    assert o.matches(t) == c.matches(t) and o.code(t, Status(t % 2)) == c.code(t, Status(t % 2)), t
assert c.Status is Status
# a family's members are in the code: a class whose definition changed after compilation is refused at load
for parent in (type(c), type('Middle', (type(c),), {})):
    invalid = type('ChangedSchema', (parent,), {'Status': object()})
    try:
        invalid(compiled)
    except TypeError as actual:
        assert 'Status' in str(actual) and 'rebuild' in str(actual), actual
    else:
        raise AssertionError('changed compiled definition was accepted')
try:
    c.Status = SimpleNamespace(OK=201)
except AttributeError:
    pass
else:
    raise AssertionError('an IntEnum Reference was rebindable in the compiled model')
print('enum by type ok')
'''


def test_enum_selected_by_type(tmp_path, monkeypatch):
    (tmp_path / 'bytype_status.py').write_text(STATUS, encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    from bytype_status import Status
    model = mx.new_model('EnumByType')
    try:
        s = model.new_space('Projection', refs={'Status': Status, 'scale': 2})
        for name, formula in FORMULAS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nfor t in range(0, 300, 7): m.Projection.matches(t); m.Projection.code(t, m.Projection.Status(t % 2))\n',
        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    generated = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    # a member compared with a held parameter is its position; compared with any other value it stays a member
    assert '1.0 if s == 1 else 0.5' in generated and ') == self.Status.OK' in generated, generated
    (tmp_path / 'check.py').write_text(CHECK, encoding='utf-8')
    result = subprocess.run([sys.executable, 'check.py'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'enum by type ok' in result.stdout

