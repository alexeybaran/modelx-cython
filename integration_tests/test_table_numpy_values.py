"""A cell value read natively keeps numpy's rules where they give another value than a C number.

pandas returns ``numpy.float64`` / ``numpy.int64`` for a cell of a float64 / int64 column. A native read is a C
number; where its value reaches ``round``, a method, ``type``/``isinstance``, an integer power or a container, the
read is boxed as the numpy scalar (live: numpy's rounding, ``int64 ** -1`` raises, the type is numpy's), also
through arithmetic and a local. Elsewhere it stays a C number (checked on the generated source). One small model,
built once; every case is compared with live modelx.
"""
import json
import re
import subprocess
import sys

import pytest

MODEL = '''\
import modelx as mx
import numpy as np
import pandas as pd

OPS = {
    "round_read": "round(T.loc[20 + t, 'q'], 4)",
    "round_arith": "round(T.loc[20 + t, 'q'] * 1.0 + 0.0, 4)",
    "round_local": "x = T.loc[20 + t, 'q']\\n    y = x * 1.0\\n    return round(y, 4)",
    "round_method": "T.loc[20 + t, 'q'].round(4)",
    "item": "T.loc[20 + t, 'n'].item() + 0.5",
    "type_name": "len(type(T.loc[20 + t, 'q']).__name__) + len(type(T.loc[20 + t, 'n']).__name__)",
    "is_numpy": "isinstance(T.loc[20 + t, 'q'], np.floating) and isinstance(T.at[20 + t, 'n'], np.integer)",
    "int_power": "T.loc[20 + t, 'n'] ** -1",
    "row_field": "row = W.loc[20 + t]\\n    return len(type(row['n']).__name__) + round(row['q'], 4)",
    "plain": "T.loc[20 + t, 'q'] * 2.0 + T.loc[20 + t, 'n']",
    "divide": "T.loc[20 + t, 'q'] / T.loc[20 + t, 'n']",
}


def build():
    m = mx.new_model("NumpyValues")
    d = m.new_space("Data")
    d.pd, d.np = pd, np
    d.new_cells("tab", "def tab():\\n    return pd.DataFrame({'q': [0.02545, 0.55645, 0.85995, 0.12345], "
                "'n': [1, 2, 3, 4], 'tag': ['a', 'b', 'c', 'd']}, index=pd.Index([20, 21, 22, 23]))\\n")
    d.new_cells("wide", "def wide():\\n    return pd.DataFrame({'q': [0.02545, 0.55645, 0.85995, 0.12345], "
                "'n': [1, 2, 3, 4]}, index=pd.Index([20, 21, 22, 23]))\\n")
    c = m.new_space("Calc")
    c.data, c.np = d, np
    for name, body in OPS.items():
        body = body if "return" in body else "return " + body
        c.new_cells(name, f"def {name}(t):\\n    T = data.tab()\\n    W = data.wide()\\n    {body}\\n")
    return m


if __name__ == "__main__":
    build().export("numpyv_nomx")
'''

SAMPLE = '''\
import warnings
warnings.simplefilter("ignore")
from numpyv_nomx import mx_model as m
from numpyv_model import OPS
for name in OPS:
    for t in range(4):
        try:
            getattr(m.Calc, name)(t)
        except Exception:
            pass
'''

CHECKS = '''\
import json
import re
import warnings
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
warnings.simplefilter("ignore")
from numpyv_model import build, OPS

assert isinstance(import_module("numpyv_nomx_cy._mx_classes").__loader__, ExtensionFileLoader)
compiled, live = import_module("numpyv_nomx_cy").mx_model, build()


def outcome(space, name, t):
    try:
        return ["ok", float(getattr(space, name)(t))]
    except Exception as error:
        named = re.search(r"\\n(\\w+Error): ", str(error))
        return ["raise", named.group(1) if named else type(error).__name__]


print(json.dumps({f"{name}({t})": [outcome(live.Calc, name, t), outcome(compiled.Calc, name, t)]
                  for name in OPS for t in range(4)}))
'''


def _run(folder, *args):
    return subprocess.run([sys.executable, *args], cwd=folder, check=True, timeout=2400, capture_output=True, text=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    work = tmp_path_factory.mktemp("numpyv")
    (work / "numpyv_model.py").write_text(MODEL, encoding="utf-8")
    (work / "sample.py").write_text(SAMPLE, encoding="utf-8")
    (work / "checks.py").write_text(CHECKS, encoding="utf-8")
    (work / "spec.py").write_text(repr({"parallel_workers": 1}), encoding="utf-8")
    _run(work, "numpyv_model.py")
    _run(work, "-m", "modelx_cython", "numpyv_nomx", "--sample", "sample.py", "--spec", "spec.py")
    return work, json.loads(_run(work, "checks.py").stdout)


def test_compiled_equals_live(built):
    _, verdicts = built
    differing = {case: pair for case, pair in verdicts.items() if pair[0] != pair[1]}
    assert not differing, "live against compiled:\n" + "\n".join(f"{c}: {p[0]} | {p[1]}" for c, p in differing.items())


def test_neutral_reads_stay_c_numbers(built):
    """A read whose value only meets arithmetic, a comparison or a C result is not boxed."""
    work, _ = built
    source = (work / "numpyv_nomx_cy" / "_mx_classes.py").read_text(encoding="utf-8")
    for name in ("plain", "divide"):
        entry = re.search(rf"def _f_{name}\(self[^\n]*\n\s*return (_mx_dense_formula_\d+)", source)
        assert entry, name
        body = re.search(rf"def {entry.group(1)}\(.*?(?=\n@_mx_cy|\Z)", source, re.S).group(0)
        assert "f64_1(" in body and "_box(" not in body, name
    entry = re.search(r"def _f_round_read\(self[^\n]*\n\s*return (_mx_dense_formula_\d+)", source)
    assert "f64_box(" in re.search(rf"def {entry.group(1)}\(.*?(?=\n@_mx_cy|\Z)", source, re.S).group(0)
