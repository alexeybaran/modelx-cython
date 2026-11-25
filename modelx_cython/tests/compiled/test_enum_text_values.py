"""Text-valued enum classes (``enum.StrEnum``): a member is held as its position where that makes the speed.

A parameter given members of one text-safe class, or their texts, is a C integer (the position, with a cache of
the size of the class); comparisons with members or their texts and loops over displays of members are integer
operations. Every other use sees the member. A caller passes the member or its text, from Python and from a
formula (a text Reference, a text result), to a Cell that only tells members apart. Compiled equals the exported
model. A text that is no member's value, or a text given to a Cell that uses the member itself, raises
``ValueError`` at the Cell.
"""
import enum
import os
import subprocess
import sys
import textwrap

import pytest

import modelx as mx

pytestmark = pytest.mark.skipif(not hasattr(enum, 'StrEnum'), reason='enum.StrEnum is of Python 3.11')

ENUMS = textwrap.dedent('''\
    import enum

    class Kind(enum.StrEnum):
        DEATH = "DEATH"
        LAPSE = "LAPSE"
        MAT = "MATURITY"

    class Sex(enum.StrEnum):
        M = "M"
        F = "F"
    ''')
CELLS = {
    'claims': textwrap.dedent('''\
        def claims(t, kind):
            if kind == Kind.DEATH:
                return 1.0 * t
            if kind == "LAPSE":
                return 10.0 * t + claims(t, Kind.DEATH)
            if kind in (Kind.MAT, "NOPE"):
                return 100.0 if t == 2 else 0.0
            raise ValueError("bad kind")
        '''),
    'total': 'lambda t: sum(claims(t, k) for k in (Kind.DEATH, Kind.LAPSE, Kind.MAT)) + claims(t, "DEATH")'
             ' + claims(t, default_kind)',
    'sex': 'lambda: "M"',
    'rate': 'lambda t, s: (0.1 if s == Sex.M else 0.2) * t',
    'use': 'lambda t: rate(t, sex()) + rate(t, Sex.F) + (1.0 if sex() == Sex.M else 0.0)',
    'nm': 'lambda kind: kind.name + "|" + str(kind) + "|" + kind + f"|{kind}|{kind!r}" + repr([kind])',
    'tab': 'lambda: {"DEATH": 1.5, "LAPSE": 2.5, "MATURITY": 3.5}',
    'lookup': 'lambda kind: tab()[kind]',
    'ident': 'lambda kind: (kind is Kind.DEATH, kind == "DEATH", kind in ("LAPSE", "MATURITY"), kind < "E", len(kind))',
}
SAMPLE = '''
for t in range(3):
    p.total(t); p.use(t); p.claims(t, "MATURITY"); p.claims(t, p.Kind.MAT)
for k in p.Kind:
    p.nm(k); p.lookup(k); p.ident(k)
p.rate(1, "F")
'''
PROBES = ['p.total(2)', 'p.use(2)', 'p.claims(2, p.Kind.MAT)', 'p.claims(2, "MATURITY")', 'p.claims(1, "LAPSE")',
          'p.claims(kind="DEATH", t=2)', 'p.rate(2, "F")', 'p.rate(2, p.Sex.M)', 'p.nm(p.Kind.MAT)',
          'p.lookup(p.Kind.LAPSE)', 'p.ident(p.Kind.DEATH)', 'p.sex()', 'p.claims(1, "XYZ")']
DIFFERENT = ['p.nm("LAPSE")', 'p.claims(1, "NOPE")', 'p.lookup("DEATH")', 'p.ident("MATURITY")']


def test_text_valued_members_equal_live(tmp_path, monkeypatch):
    (tmp_path / 'text_enums.py').write_text(ENUMS, encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    import text_enums as e
    model = mx.new_model('EnumTextValues')
    try:
        s = model.new_space('P', refs={'Kind': e.Kind, 'Sex': e.Sex, 'default_kind': 'LAPSE'})
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\np = m.P\n' + SAMPLE, encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                          cwd=tmp_path, capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        p = import_module(sys.argv[1]).mx_model.P
        for probe in sys.argv[2:]:
            try:
                print(probe, repr(eval(probe)))
            except Exception as error:
                print(probe, type(error).__name__)
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package, *PROBES, *DIFFERENT], cwd=tmp_path,
                             capture_output=True, text=True, env=env)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    live, compiled = out['exported'], out['exported_cy']
    assert compiled[:len(PROBES)] == live[:len(PROBES)], '\n'.join(
        f'{a}  |  {b}' for a, b in zip(live, compiled) if a != b)
    assert 'p.total(2) 148.0' in compiled and 'p.claims(1, "XYZ") ValueError' in compiled
    assert "p.ident(p.Kind.DEATH) (True, True, False, True, 5)" in compiled
    # where compiled is not live it raises at the Cell: a text that is no member's value; a text given to a Cell
    # that uses the member
    assert live[len(PROBES):] == ['p.nm("LAPSE") AttributeError', 'p.claims(1, "NOPE") 0.0', 'p.lookup("DEATH") 1.5',
                                  'p.ident("MATURITY") (False, False, True, False, 8)']
    assert [line.split()[-1] for line in compiled[len(PROBES):]] == ['ValueError'] * 4
    pxd = (tmp_path / 'exported_cy/_mx_classes.pxd').read_text(encoding='utf-8')
    assert 'double[3][3] _v_claims' in pxd and 'long long kind' in pxd, pxd        # a cache of the size of the class
