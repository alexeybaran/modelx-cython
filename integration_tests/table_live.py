"""Cells of a small model compared with live modelx, cell by cell (table reviews r4).

``differences(tmp_path, frames, data, cells)`` builds a model with a locked Space ``Data`` (its References from
the dict ``REFS`` that the text ``frames`` defines, its cells from ``data``) and an item Space ``Projection`` over
``policy_id`` 1 to 4 with the cells ``cells`` ({name: (parameters, body)}; a cell with a parameter is evaluated
for t in 0 to 9). It compiles the model on a sample that calls every cell not named in ``unsampled``, and returns
{cell: (call, live outcome, compiled outcome)} for the cells whose outcome differs: ('ok', value) or
('raise', exception type). ``frames`` may define ``RUN_REFS``, References of ``Data`` set on both models before
they run. A float is compared rounded to nine places, any other value by its repr.
"""
import subprocess
import sys
import textwrap

MODEL = '''\
import modelx as mx
import numpy as np
import pandas as pd
{frames}
IDS = (1, 2, 3, 4)
TS = tuple(range(10))
CELLS = {cells!r}
DATA = {data!r}


def _cell(space, name, body, params=""):
    space.new_cells(name, f"def {{name}}({{params}}):\\n" + "".join("    " + line + "\\n" for line in body.split("\\n")))


def build():
    model = mx.new_model("Probe")
    data = model.new_space("Data")
    for name, value in REFS.items():
        setattr(data, name, value)
    for name, (params, body) in DATA.items():
        _cell(data, name, body, params)
    proj = model.new_space("Projection", formula="lambda policy_id: None", refs={{"data": data, "np": np, "pd": pd}})
    for name, (params, body) in CELLS.items():
        _cell(proj, name, body, params)
    return model
'''

SAMPLE = '''\
from probe_nomx import mx_model
from probe_model import CELLS, IDS, TS
for k in IDS:
    for name, (params, body) in CELLS.items():
        if name in {unsampled!r}:
            continue
        for t in (TS if params else (None,)):
            try:
                getattr(mx_model.Projection[k], name)(*(() if t is None else (t,)))
            except Exception:
                pass
'''

CHECKS = '''\
import json, math, re, warnings
warnings.simplefilter("ignore")
from importlib import import_module
import probe_model
from probe_model import CELLS, IDS, TS, build

Model = type(import_module("probe_nomx_cy").mx_model)


def outcome(call):
    try:
        value = call()
    except Exception as error:        # live wraps the formula's error: compare the original type
        text = type(error).__name__
        if text == "FormulaError":
            found = re.search(r"\\n(\\w+): ", str(error))
            text = found.group(1) if found else text
        return ["raise", text]
    if isinstance(value, float):
        return ["ok", "nan" if math.isnan(value) else round(float(value), 9)]
    return ["ok", repr(value)]


def values(m):
    out = {}
    for name, (params, body) in CELLS.items():
        for i in IDS:
            for t in (TS if params else (None,)):
                args = () if t is None else (t,)
                out[f"{name}[{i}]{args}"] = (name, outcome(lambda: getattr(m.Projection[i], name)(*args)))
    return out


compiled, live = Model(), build()
for name, value in getattr(probe_model, "RUN_REFS", {}).items():
    setattr(compiled.Data, name, value)
    setattr(live.Data, name, value)
want, got = values(live), values(compiled)
found = {}
for call, (name, expected) in want.items():
    if got[call][1] != expected and name not in found:
        found[name] = [call, expected, got[call][1]]
print(json.dumps(found))
'''


def differences(tmp_path, frames, data, cells, unsampled=(), spec=None):
    import json
    (tmp_path / 'probe_model.py').write_text(
        MODEL.format(frames=textwrap.dedent(frames), cells=cells, data=data), encoding='utf-8')
    (tmp_path / 'build_model.py').write_text(
        'from probe_model import build\nm = build()\nm.export("probe_nomx", locked_spaces=["Data"])\nm.close()\n',
        encoding='utf-8')
    (tmp_path / 'sample.py').write_text(SAMPLE.format(unsampled=tuple(unsampled)), encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr(dict({'parallel_workers': 1, 'strict_lowering': True}, **(spec or {}))),
                                      encoding='utf-8')
    (tmp_path / 'checks.py').write_text(CHECKS, encoding='utf-8')

    def run(*args):
        done = subprocess.run([sys.executable, *args], cwd=tmp_path, timeout=1800, capture_output=True, text=True)
        assert done.returncode == 0, f'{args}\n' + done.stdout[-4000:] + done.stderr[-8000:]
        return done
    run('build_model.py')
    run('-m', 'modelx_cython', 'probe_nomx', '--sample', 'sample.py', '--spec', 'spec.py')
    found = json.loads(run('checks.py').stdout.strip().splitlines()[-1])
    return {name: tuple(tuple(part) if isinstance(part, list) else part for part in entry)
            for name, entry in found.items()}
