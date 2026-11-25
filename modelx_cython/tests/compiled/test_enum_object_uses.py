"""IntEnum members as integer literals (enums.literals): where the member's class shows in the result, compiled
equals the exported model, or the build stops with an error naming the Cell.

A member written in place stays a member object where its class shows (``repr``, ``type``, ``isinstance``,
``getattr``, ``!r``, ``%``), and a class with its own ``__str__``/``__format__`` is not substituted. ``.name`` of an
integer that is not a member raises AttributeError, as live. A member stored as a dict key, ``getattr(member,
"name")`` and ``.name`` of a child Space's Cell result work. A sum over a literal tuple whose loop variable is an
operand of arithmetic is exact beyond 64 bits. Identity tests, a member of another class given to an enum
parameter, and an enum value held as an integer where its class would show are build errors.
"""
import os
import subprocess
import sys
import textwrap

import pytest

import modelx as mx

ENUMS = textwrap.dedent('''\
    import enum

    class Leg(enum.IntEnum):
        CALL = 0
        PUT = 1
        BOTH = 2

    class Kind(enum.IntEnum):
        X = 0
        Y = 1

    class Gap(enum.IntEnum):
        A = 1
        B = 5

    class Pretty(enum.IntEnum):
        M = 0
        F = 1

        def __str__(self):
            return self.name

        def __format__(self, spec):
            return self.name
    ''')

CELLS = {
    'r_repr': 'lambda: repr(Leg.CALL)',
    'r_fstr': 'lambda: f"{Leg.CALL}|{Leg.CALL!r}|{Leg.PUT:>3}"',
    'r_pct': 'lambda: "%r" % Leg.CALL',
    'r_type': 'lambda: (type(Leg.CALL).__name__, isinstance(Leg.CALL, Leg))',
    'pretty': 'lambda: (str(Pretty.F), f"{Pretty.F}-{Pretty.M:>3}")',
    'pretty_nm': 'lambda s: s.name',
    'pick2': 'lambda t: Leg.PUT if t != -1 else Leg.CALL - 1',
    'desc2': 'lambda t: pick2(t).name',
    'gnm': 'lambda g: g.name',
    'incr_dict': 'def incr_dict():\n    d = {}\n    d[Leg.CALL] = 1.0\n    d[Leg.PUT] = 2.0\n'
                 '    return [k.name for k in d]\n',
    'r_getattr': 'lambda: getattr(Leg.PUT, "name")',
    'x_name': 'lambda: O.pick().name',
    'x_local': 'def x_local():\n    x = O.pick()\n    return x.name\n',
    'g': 'lambda x: x / 3.0',
    'cube': 'lambda: sum(g(k * k * k) for k in (3000000, 3000001))',
}
SAMPLE = ('p = m.P\n' + ''.join(f'p.{name}()\n' for name in (
    'r_repr', 'r_fstr', 'r_pct', 'r_type', 'pretty', 'incr_dict', 'r_getattr', 'x_name', 'x_local', 'cube'))
    + 'p.pretty_nm(p.Pretty.F); p.desc2(0); p.desc2(-5)\nfor g in p.Gap: p.gnm(g)\n')
PROBES = [f'p.{name}()' for name in ('r_repr', 'r_fstr', 'r_pct', 'r_type', 'pretty', 'incr_dict', 'r_getattr',
                                      'x_name', 'x_local', 'cube')] + [
    'p.pretty_nm(p.Pretty.F)', 'p.desc2(-5)', 'p.desc2(-1)', 'p.gnm(3)', 'p.gnm(p.Gap.B)']


def build(tmp_path, monkeypatch, cells, sample):
    (tmp_path / 'review_enums.py').write_text(ENUMS, encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    import review_enums as e
    model = mx.new_model('EnumObjectUses')
    try:
        s = model.new_space('P', refs={'Leg': e.Leg, 'Kind': e.Kind, 'Gap': e.Gap, 'Pretty': e.Pretty})
        s.new_space('O', refs={'Leg': e.Leg}).new_cells('pick', 'lambda: Leg.PUT')
        for name, formula in cells.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\n' + sample, encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                          cwd=tmp_path, capture_output=True, text=True, env=env)
    return done, env


def test_results_where_the_class_shows_equal_live(tmp_path, monkeypatch):
    done, env = build(tmp_path, monkeypatch, CELLS, SAMPLE)
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
        run = subprocess.run([sys.executable, '-c', probe, package, *PROBES], cwd=tmp_path, capture_output=True,
                             text=True, env=env)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    assert out['exported_cy'] == out['exported'], '\n'.join(
        f'{live}  |  {compiled}' for live, compiled in zip(out['exported'], out['exported_cy']) if live != compiled)
    assert "p.r_repr() '<Leg.CALL: 0>'" in out['exported_cy'] and 'p.desc2(-1) AttributeError' in out['exported_cy']


@pytest.mark.parametrize('cells, sample, message', [
    ({'nm': 'lambda k: k.name', 'other': 'lambda: nm(Kind.Y)'}, 'm.P.nm(m.P.Leg.CALL)\n',
     'P.other: `self.nm(self.Kind.Y)` passes a value of one IntEnum class to a parameter traced as another'),
    ({'is_tuple': 'lambda: tuple(x for x in (Leg.CALL,))[0] is (Leg.CALL, Leg.PUT)[0]'}, 'm.P.is_tuple()\n',
     'P.is_tuple: `tuple(x for x in (self.Leg.CALL,))[0] is (self.Leg.CALL, self.Leg.PUT)[0]` tests by identity'),
    ({'pretty_param': 'lambda s: "sex=" + str(s)'}, 'm.P.pretty_param(m.P.Pretty.M)\n',
     'P.pretty_param: `s` holds an IntEnum value as an integer'),
    ({'rp': 'lambda k: repr(k)'}, 'm.P.rp(m.P.Leg.CALL)\n', 'P.rp: `k` holds an IntEnum value as an integer'),
])
def test_values_that_would_differ_are_build_errors(tmp_path, monkeypatch, cells, sample, message):
    done, _ = build(tmp_path, monkeypatch, cells, sample)
    assert done.returncode != 0 and message in done.stdout + done.stderr, done.stdout[-3000:] + done.stderr[-3000:]
