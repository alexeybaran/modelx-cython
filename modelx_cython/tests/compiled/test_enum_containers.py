"""IntEnum members in a container display stay members (enums.plan) and give live's results.

A tuple of members returned by a Cell keeps its members, so ``leg.name`` works after the tuple is iterated
elsewhere (the RILA_US_S ``result`` shape). Such a member compared with a member written in place, passed as the
label argument of a Cell with a dense cache (converted to its position there), or used as a dict key looked up
with a member gives the value live gives. Compiled equals the exported model.
"""
import os
import subprocess
import sys
import textwrap

import modelx as mx

CELLS = {
    'legs': 'lambda: (Leg.CALL, Leg.PUT) if cap else (Leg.CALL,)',
    'component': 'lambda t, leg: 10.0 * t + leg',
    'result': textwrap.dedent('''\
        def result():
            out = {}
            for leg in legs():
                out[leg.name.lower()] = [component(t, leg) for t in range(3)]
            return out
        '''),
    'compared': 'lambda: [leg == Leg.PUT for leg in legs()]',
    'weights': 'lambda: {Leg.CALL: 1.5, Leg.PUT: 2.5}',
    'weight': 'lambda: weights()[Leg.PUT]',
}


def test_members_in_containers_agree_with_literals(tmp_path, monkeypatch):
    (tmp_path / 'legs_enum.py').write_text(
        'import enum\n\nclass Leg(enum.IntEnum):\n    CALL = 0\n    PUT = 1\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    from legs_enum import Leg
    model = mx.new_model('EnumContainers')
    try:
        s = model.new_space('Projection', refs={'Leg': Leg, 'cap': True})
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\np = m.Projection\np.result(); p.compared(); p.weight()\n',
        encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                          cwd=tmp_path, capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    generated = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert "_mx_enum_code(self.Leg, leg, False, 'Projection.component')" in generated, generated
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        p = import_module(sys.argv[1]).mx_model.Projection
        print(repr((p.result(), p.compared(), p.weight(), p.legs(), [type(x).__name__ for x in p.legs()])))
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True,
                             env=env)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout
    assert out['exported_cy'] == out['exported'], out
    assert "'call': [0.0, 10.0, 20.0]" in out['exported_cy'] and "'Leg', 'Leg'" in out['exported_cy'], out
