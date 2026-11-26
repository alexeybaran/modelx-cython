"""Compiled table reads against live on the frames the table runtime must leave to pandas, or prepare again.

One small model, built once. Each case failed before its fix in ``runtime/_tables``:

- a frame Cell whose index pandas cannot order (an unordered categorical index, labels of mixed types) raised
  TypeError when it was computed, although nothing read it natively;
- a text key on a DatetimeIndex was absent (``in`` gave False, a read raised KeyError), a number inside an interval
  of an IntervalIndex likewise;
- a field of a row whose number columns pandas widens (uint8 beside float64) was read as the stored uint8;
- a frame edited in place and given to a new model was read from the copy the first model prepared: through a
  frame Cell that publishes its table, through one whose name another class also uses (it publishes none: the read
  finds the table by the frame), and through an item's Cell that returns a Reference's frame as it is.
"""
import json
import subprocess
import sys

import pytest

MODEL = '''\
import modelx as mx
import numpy as np
import pandas as pd


def build():
    m = mx.new_model("Contract")
    d = m.new_space("Data")
    d.pd, d.np = pd, np
    d.SRC = pd.DataFrame({'q': [0.5, 1.5, 2.5, 3.5]}, index=pd.Index([20, 21, 22, 23], name='age'))
    d.new_cells("flat", "def flat():\\n    return SRC\\n")
    d.new_cells("cat", "def cat():\\n    return pd.DataFrame({'q': [1.0, 2.0]}, index=pd.CategoricalIndex(['a', 'b']))\\n")
    d.new_cells("mix", "def mix():\\n    return pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=pd.Index([1, 'a', 2.5], dtype=object))\\n")
    d.new_cells("dt", "def dt():\\n    return pd.DataFrame({'q': [1.0, 2.0]}, index=pd.to_datetime(['2020-01-01', '2021-01-01']))\\n")
    d.new_cells("band", "def band():\\n    return pd.DataFrame({'q': [1.0, 2.0]}, index=pd.IntervalIndex.from_breaks([0, 10, 20]))\\n")
    d.new_cells("narrow", "def narrow():\\n    return pd.DataFrame({'q': [0.5, 1.5, 2.5], 'u': np.array([1, 2, 250], dtype=np.uint8)},"
                " index=pd.Index([10, 11, 12]))\\n")
    d.SRC2 = pd.DataFrame({'q': [0.5, 1.5, 2.5, 3.5]}, index=pd.Index([20, 21, 22, 23], name='age'))
    d.new_cells("shared", "def shared():\\n    return SRC2\\n")
    c = m.new_space("Calc")
    c.data = d
    c.new_cells("shared", "def shared(t):\\n    return data.shared().loc[20 + t, 'q']\\n")
    items = m.new_space("Items", formula="lambda i: None")
    items.data = d
    items.new_cells("tab", "def tab():\\n    return data.SRC2\\n")
    items.new_cells("iread", "def iread(t):\\n    return tab().loc[20 + t % 4, 'q']\\n")
    c.new_cells("ncat", "def ncat(t):\\n    return len(data.cat()) + t\\n")
    c.new_cells("nmix", "def nmix(t):\\n    return len(data.mix()) + t\\n")
    c.new_cells("cat_read", "def cat_read(t):\\n    return data.cat().loc[('a', 'b')[t % 2], 'q']\\n")
    c.new_cells("dt_in", "def dt_in(y):\\n    return (str(y) + '-01-01') in data.dt().index\\n")
    c.new_cells("dt_read", "def dt_read(y):\\n    return data.dt().loc[str(y) + '-01-01', 'q']\\n")
    c.new_cells("band_read", "def band_read(a):\\n    return data.band().loc[a, 'q']\\n")
    c.new_cells("row_field", "def row_field(t):\\n    row = data.narrow().loc[10 + t]\\n    return row['u'] + 250 + row['q']\\n")
    c.new_cells("plain", "def plain(t):\\n    return data.flat().loc[20 + t, 'q']\\n")
    return m


CASES = [("ncat", 0), ("nmix", 0), ("cat_read", 0), ("cat_read", 1), ("dt_in", 2020), ("dt_in", 2021), ("dt_in", 2019),
         ("dt_read", 2020), ("dt_read", 2021), ("band_read", 5), ("band_read", 15), ("row_field", 0), ("row_field", 2),
         ("plain", 0), ("plain", 1), ("plain", 2), ("plain", 3)]

if __name__ == "__main__":
    build().export("contract_nomx")
'''

SAMPLE = '''\
import warnings
warnings.simplefilter("ignore")
from contract_nomx import mx_model as m
from contract_model import CASES
for name, arg in CASES:
    getattr(m.Calc, name)(arg)
for t in range(4):
    m.Calc.shared(t)
for i in (1, 2):
    for t in range(16):
        m.Items[i].iread(t)
'''

CHECKS = '''\
import json
import warnings
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
warnings.simplefilter("ignore")
from contract_model import build, CASES

assert isinstance(import_module("contract_nomx_cy._mx_classes").__loader__, ExtensionFileLoader)
package = import_module("contract_nomx_cy")
compiled, live = package.mx_model, build()


def outcome(space, name, arg):
    try:
        value = getattr(space, name)(arg)
        return ["ok", type(value).__name__ in ("bool", "bool_"), float(value)]
    except Exception as error:
        return ["raise", type(error).__name__]


out = {f"{name}({arg})": [outcome(live.Calc, name, arg), outcome(compiled.Calc, name, arg)] for name, arg in CASES}

# a frame edited in place, then given to a new model: the new model reads it as it is now
frame = compiled.Data.SRC
before = compiled.Calc.plain(2)
frame.loc[23, "q"] = 99.0
fresh = type(compiled)()
fresh.Data.SRC = frame
out["new model reads the edited frame"] = [["ok", False, float(frame.loc[23, "q"])], outcome(fresh.Calc, "plain", 3)]
out["first model keeps its cached value"] = [["ok", False, 2.5], ["ok", False, float(before)]]

# the same through a frame Cell that publishes no table, and through an item's Cell returning the Reference's frame
other = compiled.Data.SRC2
reads = [compiled.Calc.shared(3), compiled.Items[1].iread(3)]
other.loc[23, "q"] = 77.0
fresh = type(compiled)()
fresh.Data.SRC2 = other
out["new model, unpublished frame Cell"] = [["ok", False, 77.0], outcome(fresh.Calc, "shared", 3)]
out["new model, item Cell returning the frame"] = [["ok", False, 77.0], outcome(fresh.Items[1], "iread", 3)]
out["first model, before the edit"] = [["ok", False, 3.5], ["ok", False, float(reads[0])]]
print(json.dumps(out))
'''


def _run(folder, *args):
    return subprocess.run([sys.executable, *args], cwd=folder, check=True, timeout=2400, capture_output=True, text=True)


@pytest.fixture(scope="module")
def verdicts(tmp_path_factory):
    work = tmp_path_factory.mktemp("contract")
    (work / "contract_model.py").write_text(MODEL, encoding="utf-8")
    (work / "sample.py").write_text(SAMPLE, encoding="utf-8")
    (work / "checks.py").write_text(CHECKS, encoding="utf-8")
    (work / "spec.py").write_text(repr({"parallel_workers": 1}), encoding="utf-8")
    _run(work, "contract_model.py")
    _run(work, "-m", "modelx_cython", "contract_nomx", "--sample", "sample.py", "--spec", "spec.py")
    return json.loads(_run(work, "checks.py").stdout)


def test_compiled_equals_live(verdicts):
    differing = {case: pair for case, pair in verdicts.items() if pair[0] != pair[1]}
    assert not differing, "live against compiled:\n" + "\n".join(f"{c}: {p[0]} | {p[1]}" for c, p in differing.items())
