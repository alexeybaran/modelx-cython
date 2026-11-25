"""Enum families from a module stored with the model, as the lifelib text holds them (shapes of KLV_DE_S and IP_UK_S).

Such a class is of no importable module (``'<unnamed module>'``): compiled code reaches it through a Reference of
the Space, from formulas and from the gate of calls from Python. A parameter whose default is no member
(``kind=None``) is an object. Compiled equals the exported model; where it cannot, it raises at the named Cell.
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

    class Leg(enum.IntEnum):
        CALL = 0
        PUT = 1
    ''')
CELLS = {
    'claims': textwrap.dedent('''\
        def claims(t, kind=None):
            if kind is None:
                return sum(claims(t, k) for k in (Kind.DEATH, Kind.LAPSE))
            if kind == Kind.DEATH:
                return 1.0 * t
            return 10.0 * t
        '''),
    'net': 'lambda t: claims(t, Kind.DEATH) + claims(t, Kind.LAPSE) + rate(t, Kind.LAPSE) + tagged(t, Kind.DEATH) + untagged(t, Kind.DEATH)',
    'rate': 'lambda t, kind: (2.0 if kind == "DEATH" else 3.0) * t',
    'price': 'lambda t, leg: (1.0 if leg == Leg.CALL else 2.0) * t + len(leg.name)',
    'both': 'lambda t: price(t, Leg.CALL) + price(t, Leg.PUT) + scaled(t, Leg.PUT)',
    'scaled': 'lambda t, leg: sum(map(lambda x: x * t, (1.0, 2.0))) if leg == Leg.PUT else 0.0',    # a closure
    'tagged': textwrap.dedent('''\
        def tagged(t, kind):
            def scale(x):
                return x * t
            return scale(3.0) if kind == Kind.DEATH else 0.5
        '''),                                                   # a closure: the Cell's public method is Python
    'untagged': textwrap.dedent('''\
        def untagged(t, kind):
            def scale(x):
                return x * t
            return scale(3.0) if kind == Kind.DEATH else 0.5
        '''),                                                   # uncached with a closure: a Python method
    'named': 'lambda t, kind: kind.name + str(t)',
    'call_named': 'lambda t: named(t, Kind.LAPSE)',
}
SAMPLE = 'for t in range(3):\n    p.net(t); p.both(t); p.call_named(t)\n'
PROBES = ['p.net(2)', 'p.claims(2)', 'p.claims(2, p.Kind.LAPSE)', 'p.claims(2, "LAPSE")', 'p.rate(2, p.Kind.DEATH)',
          'p.rate(2, "DEATH")', 'p.both(2)', 'p.price(2, p.Leg.PUT)', 'p.scaled(2, p.Leg.PUT)', 'p.scaled(2, 1)',
          'p.named(2, p.Kind.DEATH)', 'p.call_named(1)', 'p.tagged(2, p.Kind.DEATH)', 'p.tagged(2, "LAPSE")',
          'p.tagged(1, p.Kind.LAPSE)', 'p.untagged(2, p.Kind.DEATH)', 'p.untagged(2, "DEATH")']
RAISE = ['p.rate(2, "NOPE")', 'p.named(2, "DEATH")', 'p.price(2, 1.0)', 'p.price(2, 1)']


def test_a_family_stored_with_the_model_equals_live(tmp_path):
    (tmp_path / 'src_enums.py').write_text(ENUMS, encoding='utf-8')
    model = mx.new_model('EnumStored')
    try:
        model.new_module('enums', 'enums.py', tmp_path / 'src_enums.py')
        s = model.new_space('P', refs={'Kind': model.enums.Kind, 'Leg': model.enums.Leg})
        for name, formula in CELLS.items():
            s.new_cells(name, formula).is_cached = name != 'untagged'
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'src_enums.py').unlink()            # nothing to import: the class lives in the model
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
                print(probe, type(error).__name__, str(error))
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package, *PROBES, *RAISE], cwd=tmp_path,
                             capture_output=True, text=True, env=env)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    live, compiled = out['exported'], out['exported_cy']
    assert compiled[:len(PROBES)] == live[:len(PROBES)], '\n'.join(
        f'{a}  |  {b}' for a, b in zip(live, compiled) if a != b)
    assert 'p.claims(2) 22.0' in compiled and "p.named(2, p.Kind.DEATH) 'DEATH2'" in compiled
    # where compiled is not live, it raises at the named Cell
    assert compiled[len(PROBES):] == [
        "p.rate(2, \"NOPE\") ValueError P.rate: 'NOPE' is not a member of Kind",
        "p.named(2, \"DEATH\") ValueError P.named: 'DEATH' is not a member of Kind, which this Cell uses as a member",
        'p.price(2, 1.0) ValueError P.price: 1.0 is not a member of Leg, which this Cell uses as a member',
        'p.price(2, 1) ValueError P.price: 1 is not a member of Leg, which this Cell uses as a member']
    pxd = (tmp_path / 'exported_cy/_mx_classes.pxd').read_text(encoding='utf-8')
    assert 'object kind=*' in pxd and 'long long leg' in pxd and 'double[3][2] _v_rate' in pxd, pxd
