"""``x is E.M`` / ``x is not E.M`` gives what the model gives.

An identity test needs the member object: a member written in the test stays a member, and a Cell result or
parameter that compiled code holds as an integer is converted through its class (enums.plan). This was a build
error of the first literal pass.
"""
import subprocess
import sys

import modelx as mx


def _model(tmp_path, monkeypatch):
    (tmp_path / 'd98_kinds.py').write_text(
        'import enum\n\nclass Kind(enum.IntEnum):\n    A = 1\n    B = 2\n    C = 3\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    from d98_kinds import Kind
    model = mx.new_model('EnumIdentity')
    try:
        s = model.new_space('Projection', formula='lambda policy: None', refs={'Kind': Kind})
        s.new_cells('kind', '''\
def kind(t):
    return Kind.A if t % 3 == 0 else (Kind.B if t % 3 == 1 else Kind.C)
''')
        s.new_cells('direct', '''\
def direct(t):
    if kind(t) is Kind.A:
        return 1.0
    elif kind(t) is not Kind.B:
        return 3.0
    return 2.0
''')
        s.new_cells('reversed_', '''\
def reversed_(t):
    if Kind.A is kind(t):
        return 10.0
    if Kind.C is not kind(t):
        return 20.0
    return 30.0
''')
        s.new_cells('local', '''\
def local(t):
    k = kind(t)
    a = 100.0 if k is Kind.B else 0.0
    b = 1000.0 if Kind.C is not k else 0.0
    return a + b + t
''')
        s.new_cells('by_arg', '''\
def by_arg(t, k):
    if k is Kind.C:
        return 7.0 * t
    return 5.0 * t if Kind.A is not k else 3.0 * t
''')
        s.new_cells('caller', '''\
def caller(t):
    return by_arg(t, kind(t)) + by_arg(t, Kind.A)
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'p = m.Projection\n'
        'for t in range(9):\n'
        '    p.direct(t); p.reversed_(t); p.local(t); p.caller(t)\n', encoding='utf-8')


def test_enum_identity_equals_live(tmp_path, monkeypatch):
    _model(tmp_path, monkeypatch)
    env = {**__import__('os').environ,
           'PYTHONPATH': str(tmp_path) + ':' + __import__('os').environ.get('PYTHONPATH', '')}
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    probe = ('import sys\nfrom importlib import import_module\n'
             'p = import_module(sys.argv[1]).mx_model.Projection\n'
             'print([(p.direct(t), p.reversed_(t), p.local(t), p.caller(t)) for t in range(9)])\n')
    out = [subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True,
                          env=env).stdout for package in ('exported', 'exported_cy')]
    assert out[0] == out[1] and out[0].startswith('[(1.0, 10.0, 1000.0, ') and '(2.0, 20.0, 1101.0, ' in out[0], out
