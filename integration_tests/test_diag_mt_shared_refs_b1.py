"""diag-mt, the IntEnum case on the direct route: a member local is a C integer and its ``.name`` is read by
value. Direct-route twin of the second test of ``test_diag_mt_shared_refs_join``."""
import subprocess
import sys
import textwrap

import modelx as mx


def test_enum_member_local_is_a_c_integer_and_results_stay(tmp_path, monkeypatch):
    """``kind = Kind.A if ... else Kind.B`` is a conditional of integer literals (no member object read per
    call), and ``kind.name`` its name by value; both give the exported model's values."""
    (tmp_path / 'diag_mt_kinds.py').write_text(
        'import enum\n\nclass Kind(enum.IntEnum):\n    A = 1\n    B = 2\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    from diag_mt_kinds import Kind
    keys = tuple(range(1, 41))
    model = mx.new_model('EnumLocal')
    try:
        portfolio = model.new_space('Portfolio')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'keys': keys, 'Kind': Kind})
        projection.new_cells('pay', '''\
def pay(t, kind):
    return (t + key) * (1.5 if kind == Kind.A else 0.25)
''')
        projection.new_cells('chosen', '''\
def chosen(t):
    kind = Kind.A if (t + key) % 3 == 0 else Kind.B
    if kind == Kind.B:
        return pay(t, kind) + 1.0
    return pay(t, kind)
''')
        projection.new_cells('named', '''\
def named(t):
    kind = Kind.A if t % 2 == 0 else Kind.B
    return kind.name
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(30):
        total += chosen(t)
    return total
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'for key in (1, 2, 3):\n'
        '    [p[key].chosen(t) for t in range(30)]; p[key].value(); [p[key].named(t) for t in range(3)]\n',
        encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    generated = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    # members are integer literals (enums.literals); Cython types the local as it types any integer local
    assert '= 1 if (t + self.key) % 3 == 0 else 2' in generated, generated
    assert 'kind = 1 if t % 2 == 0 else 2' in generated                     # named: the same integers
    assert "{1: 'A', 2: 'B'}.get(kind) or getattr(kind, 'name')" in generated   # and its .name by value
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        m = import_module(sys.argv[1]).mx_model
        p = m.Portfolio.Projection
        print(repr(sum(p[key].value() for key in range(1, 41))))
        item = p[5]
        print(repr([item.chosen(t) for t in range(30)]), [item.named(t) for t in range(4)])
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    assert out['exported_cy'] == out['exported'], out
