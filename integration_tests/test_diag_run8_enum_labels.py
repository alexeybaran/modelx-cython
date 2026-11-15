"""diag-run8: ``E[label]`` of a declared IntEnum in a C-integer position (the value of a Cell typed
long long, a C-integer argument, an ``==`` operand with a typed integer) is a C integer: an exact ``str``
label equal to a member name gives the member's constant without touching the enum class or its members.
Every other label runs ``E[label]`` as written: an alias name, an unknown name (same KeyError and
message), a non-string operand. The label may come from a Cell or a scalar Reference.
Compiled equals the exported model, values and errors."""
import json
import os
import subprocess
import sys
import textwrap

import modelx as mx


CELLS = {
    'label_of': 'def label_of(t):\n    return ("A", "B", "C", "D", "Z")[t % 5]\n',
    'from_cell': 'def from_cell(t):\n    return Kind[label_of(t)]\n',
    'from_ref': 'def from_ref(t):\n    return Kind[default_label]\n',
    'int_label': 'def int_label(t):\n    return Kind[t % 2]\n',
    'compare': 'def compare(t):\n    return 1.0 if Kind[label_of(t)] == Kind.B else 0.0\n',
    'by_arg': 'def by_arg(t, k):\n    return 10.0 * k + t\n',
    'caller': 'def caller(t):\n    return by_arg(t, Kind[label_of(t)])\n',
}
CALLS = ('from_cell', 'from_ref', 'int_label', 'compare', 'caller')


def _model(tmp_path, monkeypatch):
    from modelx_helpers.typed_inputs import EnumDefinitions
    (tmp_path / 'r8_kinds.py').write_text(
        'import enum\n\nclass Kind(enum.IntEnum):\n    A = 1\n    B = 2\n    C = 3\n    D = 1\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    from r8_kinds import Kind
    model = mx.new_model('EnumLabels')
    try:
        s = model.new_space('Projection', formula='lambda policy: None',
                            refs={'Kind': Kind, 'default_label': 'C'})
        s.enum_definitions = EnumDefinitions(('Kind',))
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    # an unknown label raises in the sample too; it is caught there
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\np = m.Projection\n'
        'for t in range(10):\n'
        + ''.join(f'    try:\n        p.{name}(t)\n    except KeyError:\n        pass\n'
                  for name in CALLS if name != 'int_label'), encoding='utf-8')


PROBE = textwrap.dedent('''\
    import sys
    from importlib import import_module
    p = import_module(sys.argv[1]).mx_model.Projection
    out = []
    for name in sys.argv[2].split(','):
        for t in range(10):
            try:
                v = getattr(p, name)(t)
                out.append((name, t, 'value', float(v)))
            except Exception as e:
                out.append((name, t, 'raise', type(e).__name__, str(e)))
    print(repr(out))
    ''')


def test_enum_labels_lower_to_c_integers_with_live_values_and_errors(tmp_path, monkeypatch):
    _model(tmp_path, monkeypatch)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    out = {}
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + ':' + os.environ.get('PYTHONPATH', '')}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', PROBE, package, ','.join(CALLS)], cwd=tmp_path,
                             capture_output=True, text=True, env=env)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout
    assert out['exported_cy'] == out['exported'], out
    assert "'raise', 'KeyError', \"'Z'\"" in out['exported'], out['exported']      # the unknown label

    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    cells = {key.rsplit('._f_', 1)[-1]: value for key, value in report.items()
             if isinstance(value, dict) and '._f_' in key}
    labels = {name: cells[name].get('native_enum_labels', 0) for name in CALLS if name in cells}
    assert labels['from_cell'] == 1 and labels['from_ref'] == 1 and labels['caller'] == 1, labels
    assert labels['compare'] == 1, labels
    pxd = (tmp_path / 'exported_cy/_mx_classes.pxd').read_text(encoding='utf-8')
    assert 'cdef inline long long _mx_enum_label_0(object receiver, object label)' in pxd, pxd
