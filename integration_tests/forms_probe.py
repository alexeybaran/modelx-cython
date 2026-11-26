"""A small item model whose cells read pandas frames, compiled and compared with live modelx, cell by cell.

``compare(tmp_path, frames, cells, ids)`` builds the model: a locked Space ``Data`` with one Reference
``raw_<name>`` and one Cell ``<name>()`` returning it per frame (``frames``: name -> Python source of a DataFrame),
and an item Space ``P`` on ``policy_id`` with ``cells`` (name -> (parameters, body)) and the References ``data``
and ``np``. The sample calls every cell of the ids ``sample_ids`` (every one by default), at ``t`` = 0..2 for a cell of
``t``. It returns the generated source of ``P`` and the list of differences from live over ``ids``: a value (rel
1e-10, abs 1e-7), or the type of the error raised (live's FormulaError gives its original error's type).
``native_reads(source, cell)`` lists the prepared-table reads compiled into a cell's formula.
"""
import json
import re

import libcst as cst
import subprocess
import sys
import textwrap

MODEL = '''\
import modelx as mx
import numpy as np
import pandas as pd
import textwrap

FRAMES = {{{frames}}}
CELLS = {cells!r}
IDS = {ids!r}


def build():
    model = mx.new_model("Probe")
    data = model.new_space("Data")
    for name, frame in FRAMES.items():
        setattr(data, "raw_" + name, frame)
        data.new_cells(name, f"def {{name}}():\\n    return raw_{{name}}\\n")
    space = model.new_space("P", formula="lambda policy_id: None", refs={{"data": data, "np": np}})
    for name, (params, body) in CELLS.items():
        space.new_cells(name, f"def {{name}}({{params}}):\\n" + textwrap.indent(body, "    ") + "\\n")
    return model
'''

CHECKS = '''\
import json, math, re, sys
from importlib import import_module
from probe_model import CELLS, IDS, build

Model = type(import_module(sys.argv[1]).mx_model)


def outcome(call):
    try:
        value = call()
    except Exception as error:
        text = type(error).__name__
        if text == "FormulaError":
            found = re.search(r"\\n(\\w+): ", str(error))
            text = found.group(1) if found else text
        return ["raise", text]
    try:
        return ["ok", float(value)]
    except (TypeError, ValueError):
        return ["ok", repr(value)]


def same(a, b):
    if a[0] != b[0] or type(a[1]) is not type(b[1]):
        return False
    if isinstance(a[1], float):
        if math.isnan(a[1]) or math.isnan(b[1]):
            return math.isnan(a[1]) and math.isnan(b[1])
        return abs(a[1] - b[1]) <= max(1e-7, 1e-10 * abs(b[1]))
    return a[1] == b[1]


compiled, live = Model(), build()
out = []
for i in IDS:
    for name, (params, _) in CELLS.items():
        for args in ([(t,) for t in range(3)] if params else [()]):
            got = outcome(lambda: getattr(compiled.P[i], name)(*args))
            want = outcome(lambda: getattr(live.P[i], name)(*args))
            if not same(got, want):
                out.append(f"P[{{i}}].{{name}}{{args}}: compiled {{got}}, live {{want}}")
print(json.dumps(out))
'''


def compare(tmp_path, frames, cells, ids, sample_ids=None, spec=None):
    sample_ids = ids if sample_ids is None else sample_ids
    source = ', '.join(f'{name!r}: {text}' for name, text in frames.items())
    (tmp_path / 'probe_model.py').write_text(MODEL.format(frames=source, cells=cells, ids=tuple(ids)),
                                             encoding='utf-8')
    (tmp_path / 'build_model.py').write_text(
        'from probe_model import build\nm = build()\nm.export("probe_nomx", locked_spaces=["Data"])\nm.close()\n',
        encoding='utf-8')
    (tmp_path / 'sample.py').write_text(textwrap.dedent(f'''\
        from probe_nomx import mx_model
        from probe_model import CELLS
        for i in {tuple(sample_ids)!r}:
            for name, (params, _) in CELLS.items():
                for args in ([(t,) for t in range(3)] if params else [()]):
                    try:
                        getattr(mx_model.P[i], name)(*args)
                    except Exception:
                        pass
        '''), encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'strict_lowering': True, **(spec or {})}),
                                      encoding='utf-8')
    (tmp_path / 'checks.py').write_text(CHECKS.format(), encoding='utf-8')

    def run(*args):
        done = subprocess.run([sys.executable, *args], cwd=tmp_path, timeout=1200, capture_output=True, text=True)
        assert done.returncode == 0, f'{args}\n' + done.stdout[-4000:] + done.stderr[-8000:]
        return done
    run('build_model.py')
    run('-m', 'modelx_cython', 'probe_nomx', '--sample', 'sample.py', '--spec', 'spec.py')
    source = (tmp_path / 'probe_nomx_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    failures = json.loads(run('checks.py', 'probe_nomx_cy').stdout.strip().splitlines()[-1])
    return source, failures


def native_reads(source, cell):
    """The prepared-table reads compiled into the formula of ``cell`` (its own function and the module functions it
    calls), one runtime function name per read: each lowered read is ``(native if <slot test> else pandas)``, a
    field of a row local ``(native if <the row is stored> else pandas)``."""
    module = cst.parse_module(source)
    functions = {}
    for node in module.body:
        if isinstance(node, cst.ClassDef):
            for item in node.body.body:
                if isinstance(item, cst.FunctionDef) and item.name.value == '_f_' + cell:
                    functions['_f_' + cell] = item
        elif isinstance(node, cst.FunctionDef):
            functions[node.name.value] = node
    own = functions.get('_f_' + cell)
    if own is None:
        return []
    text = module.code_for_node(own)
    bodies = [own] + [functions[name] for name in sorted(set(re.findall(r'\b(_mx_\w*formula_\d+)\(', text)))
                      if name in functions]
    reads = []

    class Reads(cst.CSTVisitor):
        def visit_IfExp(self, node):
            # the native branch is taken on a slot of the prepared table (or on the stored row of a row local)
            if re.match(r'\(*(_mx_helper_\w+\.slot\(|_mx_rw_\w+ >= 0)', module.code_for_node(node.test)):
                # the read itself (a value kept as the numpy scalar pandas returns is boxed around it)
                names = [n for n in re.findall(r'_mx_helper_\w+\.(\w+)\(', module.code_for_node(node.body))
                         if not n.endswith('_box')]
                reads.append(names[0] if names else '?')
    for body in bodies:
        body.visit(Reads())
    return reads
