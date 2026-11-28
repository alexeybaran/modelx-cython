"""Small models for the vector reproductions: built from formula text, compiled, and compared with live.

``build`` exports a model of the given Spaces, samples it with ``sample`` (statements using ``m``, the
exported model) and compiles it. ``probe`` evaluates expressions in a fresh process on one package and
gives, per expression, ``('value', repr)`` or ``('raises', exception type name)``.
"""
import json
import subprocess
import sys

import modelx as mx

PROBE = '''\
import json, sys
from importlib import import_module
m = import_module(sys.argv[1]).mx_model
out = []
for expr in json.loads(sys.argv[2]):
    try:
        out.append(['value', repr(eval(expr, {'m': m}))])
    except Exception as exc:
        out.append(['raises', type(exc).__name__])
print(json.dumps(out))
'''


def build(tmp_path, spaces, sample, name='VectorRepro'):
    """``spaces``: {Space name: ({cell name: formula text}, {ref name: Space name or value})}, in order."""
    model = mx.new_model(name)
    try:
        made = {}
        for space_name, (formulas, refs) in spaces.items():
            space = model.new_space(space_name, refs={k: made.get(v, v) if isinstance(v, str) else v
                                                      for k, v in refs.items()})
            for formula in formulas.values():
                space.new_cells(formula=formula)
            made[space_name] = space
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\n' + sample, encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                             '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    (tmp_path / 'probe.py').write_text(PROBE, encoding='utf-8')


def probe(tmp_path, package, exprs):
    result = subprocess.run([sys.executable, 'probe.py', package, json.dumps(exprs)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return [tuple(item) for item in json.loads(result.stdout)]


def live_and_compiled(tmp_path, exprs):
    return probe(tmp_path, 'exported', exprs), probe(tmp_path, 'exported_cy', exprs)


def reports(tmp_path):
    """{formula name: its admission entry} of the compiled model."""
    found = {}

    def walk(d):
        for key, value in d.items():
            if isinstance(value, dict) and ('vectors' in value or 'cache' in value):
                found[key.rsplit('._f_', 1)[-1].rsplit('.', 1)[-1]] = value
            elif isinstance(value, dict):
                walk(value)
    walk(json.loads((tmp_path / 'exported_cy' / '_mx_admission.json').read_text(encoding='utf-8')))
    return found
