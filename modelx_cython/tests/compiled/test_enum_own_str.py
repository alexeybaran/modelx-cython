"""An IntEnum class with its own ``__str__`` / ``__format__`` is not an enum definition: compiled equals live.

Such a class shows differently from its integers, so its Reference stays an ordinary Reference: ``str`` and
f-strings of its members and of a Cell's result give the class's own text, and comparisons give
live's results.
"""
import os
import subprocess
import sys
import textwrap

import modelx as mx

ENUMS = textwrap.dedent('''\
    import enum


    class Pretty(enum.IntEnum):
        M = 0
        F = 1

        def __str__(self):
            return self.name

        def __format__(self, spec):
            return format(self.name, spec)
    ''')

CELLS = {
    'kind': 'lambda t: Pretty.F if t % 2 else Pretty.M',
    'text': 'lambda: str(Pretty.F) + "|" + f"{Pretty.M:>3}"',
    'label': 'lambda t: str(kind(t))',
    'rate': 'lambda t: 1.5 if kind(t) == Pretty.F else 2.5',
    'within': 'lambda t: 1.0 if kind(t) in (Pretty.F,) else 0.0',
}
PROBES = ['p.text()', *[f'p.label({t})' for t in range(4)],
          *[f'p.rate({t})' for t in range(4)], *[f'p.within({t})' for t in range(4)], 'p.kind(1)', 'p.kind(2)']


def test_a_class_with_its_own_str_shows_as_live(tmp_path, monkeypatch):
    (tmp_path / 'own_str_enums.py').write_text(ENUMS, encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    import own_str_enums
    model = mx.new_model('EnumOwnStr')
    try:
        s = model.new_space('P', formula='lambda policy: None', refs={'Pretty': own_str_enums.Pretty})
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\np = m.P\np.text()\n'
        'for t in range(4):\n    p.label(t); p.rate(t); p.within(t)\n', encoding='utf-8')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                          cwd=tmp_path, capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    out = {}
    for package in ('exported', 'exported_cy'):
        code = f'from importlib import import_module\np = import_module({package!r}).mx_model.P\n' + ''.join(
            f'print({probe!r}, repr({probe}))\n' for probe in PROBES)
        run = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, capture_output=True, text=True, env=env)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    assert out['exported_cy'] == out['exported'], '\n'.join(
        f'{live}  |  {compiled}' for live, compiled in zip(out['exported'], out['exported_cy']) if live != compiled)
