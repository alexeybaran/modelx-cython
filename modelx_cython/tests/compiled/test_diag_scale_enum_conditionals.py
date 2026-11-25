"""diag-scale P2: enum members through conditional expressions.

A conditional of members (nested, or with one branch another integer or an object) is a Cell result or a
local, so it stays members (enums.plan); given to a parameter held as a position it is converted there.
Compiled equals the exported model.
"""
import os
import subprocess
import sys
import textwrap

import modelx as mx


CELLS = {
    'kind_of': '''\
def kind_of(t):
    return Kind.A if t % 2 == 0 else Kind.B
''',
    'nested': '''\
def nested(t):
    return Kind.A if t % 3 == 0 else (Kind.B if t % 3 == 1 else Kind.C)
''',
    'int_branch': '''\
def int_branch(t):
    return Kind.C if t < 3 else t
''',
    'object_branch': '''\
def object_branch(t):
    return Kind.A if t < 3 else len(names())
''',
    'names': '''\
def names():
    return [Kind.B, Kind.C]
''',
    'by_arg': '''\
def by_arg(t, k):
    return 10.0 * k + t
''',
    'caller': '''\
def caller(t):
    return by_arg(t, Kind.A if t % 2 else Kind.C)
''',
    'compare': '''\
def compare(t):
    return 1.0 if kind_of(t) == (Kind.A if t > 4 else Kind.B) else 0.0
''',
    'local': '''\
def local(t):
    k = kind_of(t) if t % 3 else kind_of(t + 1)
    return 2.0 if k == Kind.A else 3.0
''',
    'local_object': '''\
def local_object(t):
    k = kind_of(t) if t % 3 else names()[0]
    return 2.0 if k == Kind.B else 3.0
''',
}


def _model(tmp_path, monkeypatch):
    (tmp_path / 'p2_kinds.py').write_text(
        'import enum\n\nclass Kind(enum.IntEnum):\n    A = 0\n    B = 1\n    C = 2\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    from p2_kinds import Kind
    model = mx.new_model('EnumConditionals')
    try:
        s = model.new_space('Projection', formula='lambda policy: None', refs={'Kind': Kind})
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    calls = ('kind_of', 'nested', 'int_branch', 'object_branch', 'caller', 'compare', 'local', 'local_object')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'p = m.Projection\n'
        'for t in range(9):\n'
        + ''.join(f'    p.{name}(t)\n' for name in calls), encoding='utf-8')
    return calls


PROBE = textwrap.dedent('''\
    import sys
    from importlib import import_module
    p = import_module(sys.argv[1]).mx_model.Projection
    names = sys.argv[2].split(',')
    print(repr([[repr(getattr(p, name)(t)) if name not in ('kind_of', 'nested', 'int_branch', 'object_branch')
                 else int(getattr(p, name)(t)) for name in names] for t in range(9)]))
    ''')


def test_enum_members_through_conditional_expressions(tmp_path, monkeypatch):
    calls = _model(tmp_path, monkeypatch)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    out = {}
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + ':' + os.environ.get('PYTHONPATH', '')}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', PROBE, package, ','.join(calls)], cwd=tmp_path,
                             capture_output=True, text=True, env=env)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout
    assert out['exported_cy'] == out['exported'], out

    # a result keeps its members; a conditional given to a held parameter (used in arithmetic) is converted there
    generated = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert 'self.Kind.A if t % 2 == 0 else self.Kind.B' in generated, generated
    assert ("_mx_enum_code(self.Kind, self.Kind.A if t % 2 else self.Kind.C, False, 'Projection.by_arg')"
            in generated), generated
    assert 'return [self.Kind.B, self.Kind.C]' in generated, generated     # a list display keeps its members
