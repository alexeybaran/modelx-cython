"""A row cell of a base Space overridden or read through another Reference in a derived Space: the public values
and rows equal the exported model's. A row cell another class of the module also defines is read as written."""

import json
import subprocess
import sys
import textwrap

import pytest

MODEL = """\
import modelx as mx
import pandas as pd
POINTS = pd.DataFrame({"sa": [100.0, 200.0, 300.0], "age": [30, 40, 50]}, index=pd.Index([1, 2, 3], name="policy_id"))
OTHER = POINTS.assign(sa=POINTS["sa"] * 10)
IDS = (1, 2, 3)


def build(override):
    model = mx.new_model("Derived")
    data = model.new_space("Data", refs={"POINTS": POINTS, "OTHER": OTHER})
    data.new_cells("points", "def points():\\n    return POINTS.copy()\\n")
    data.new_cells("other", "def other():\\n    return OTHER.copy()\\n")
    base = model.new_space("Base", refs={"data": data})
    base.new_cells("model_point", "lambda: data.points().loc[policy_id]")
    base.new_cells("sa", 'lambda: float(model_point()["sa"])')
    base.new_cells("pv", "lambda: sa() * 2.0")
    proj = model.new_space("Projection", bases=base, formula="lambda policy_id: None", refs={"ids": IDS})
    if override == "row":
        proj.model_point.formula = "lambda: data.other().loc[policy_id]"
    elif override == "reader":
        proj.sa.formula = 'lambda: float(model_point()["sa"]) + 1.0'
    elif override == "alias_line":
        proj.sa.formula = 'def sa():\\n    row = model_point(); extra = 1\\n    return float(row["sa"]) + extra\\n'
    elif override == "early_alias":
        proj.sa.formula = ('def sa():\\n    if policy_id == 4:\\n        return float(row["sa"])\\n'
                           '    row = model_point()\\n    return float(row["sa"])\\n')
    elif override in ("reference", "shared"):
        other = model.new_space("Other", refs={"POINTS": OTHER.iloc[::-1]})
        other.new_cells("points", "def points():\\n    return POINTS.copy()\\n")
        proj.data = other
        if override == "shared":
            proj.sa.formula = 'def sa():\\n    row = model_point()\\n    return float(row["sa"])\\n'
            model.new_space("Shared", formula="lambda policy_id: None", refs={"projection": proj, "ids": IDS})
    return model
"""

CHECKS = """\
import json
import sys
from importlib import import_module
m = import_module(sys.argv[1]).mx_model
p = m.Projection
print(sum(p[i].pv() for i in (1, 2, 3)))
print(json.dumps([(p[i].sa(), p[i].pv(), type(p[i].model_point()).__name__,
                   sorted((str(k), float(v)) for k, v in p[i].model_point().items()))
                  for i in (1, 2, 3)]))
"""


@pytest.mark.parametrize(("override", "total"), [
    ("row", 12000), ("reader", 1206), ("none", 1200), ("reference", 12000),
    ("alias_line", 1206), ("early_alias", 1200),
])
def test_derived_space_field_admission_preserves_public_rows(tmp_path, override, total):
    (tmp_path / "derived_model.py").write_text(MODEL, encoding="utf-8")
    name = f"d_{override}"
    (tmp_path / "build_model.py").write_text(textwrap.dedent(f"""\
        from derived_model import build
        m = build({override!r})
        m.export({name!r}, locked_spaces=["Data"])
        m.close()
        """), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        f"from {name} import mx_model\nfor k in (1, 2, 3):\n    mx_model.Projection[k].pv()\n"
        "    mx_model.Projection[k].model_point()\n", encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600,
                                    capture_output=True, text=True)
    run("build_model.py")
    run("-m", "modelx_cython", name, "--sample", "sample.py", "--no-spec")
    outs = {pkg: run("checks.py", pkg).stdout.splitlines() for pkg in (name + "_cy", name)}
    assert outs[name + "_cy"] == outs[name], outs
    assert all(float(value) == total for value in outs[name][:-1])
    assert all(row[2] == "Series" for row in json.loads(outs[name][-1]))
