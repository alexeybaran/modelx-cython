"""Match derived row admissions and the public values on both collect ownership routes."""

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
    proj = model.new_space("Projection", bases=base, formula="lambda policy_id: None", refs={"ids": IDS},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"pv": pv()}}')
    if override == "row":
        proj.model_point.formula = "lambda: data.other().loc[policy_id]"
    elif override == "reader":
        proj.sa.formula = 'lambda: float(model_point()["sa"]) + 1.0'
    elif override in ("reference", "shared"):
        other = model.new_space("Other", refs={"POINTS": OTHER.iloc[::-1]})
        other.new_cells("points", "def points():\\n    return POINTS.copy()\\n")
        proj.data = other
        if override == "shared":
            model.new_space("Shared", formula="lambda policy_id: None", refs={"projection": proj, "ids": IDS},
                            collect_over={"policy_id": IDS},
                            collect_outputs='lambda: {"sum": {"pv": projection[policy_id].pv()}}')
    return model
"""

CHECKS = """\
import json
import sys
from importlib import import_module
m = import_module(sys.argv[1]).mx_model
p = m.Projection
if hasattr(m, "Shared"):
    print(m.Shared.collect().sum.pv)
print(p.collect().sum.pv)
print(json.dumps([(p[i].sa(), p[i].pv(), type(p[i].model_point()).__name__,
                   sorted((str(k), float(v)) for k, v in p[i].model_point().items()))
                  for i in (1, 2, 3)]))
"""


@pytest.mark.parametrize(("override", "total"), [
    ("row", 12000), ("reader", 1206), ("none", 1200), ("reference", 12000), ("shared", 12000),
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
        f"from {name} import mx_model\nfor k in (1, 2, 3):\n    mx_model.Projection[k].collect_outputs()\n"
        "    mx_model.Projection[k].model_point()\nmx_model.Projection.collect()\n"
        "if hasattr(mx_model, 'Shared'):\n    mx_model.Shared.collect()\n", encoding="utf-8")
    (tmp_path / "spec.py").write_text(
        repr({"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}), encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600,
                                    capture_output=True, text=True)
    run("build_model.py")
    run("-m", "modelx_cython", name, "--sample", "sample.py", "--spec", "spec.py")
    outs = {pkg: run("checks.py", pkg).stdout.splitlines() for pkg in (name + "_cy", name)}
    assert outs[name + "_cy"] == outs[name], outs
    assert all(float(value) == total for value in outs[name][:-1])
    assert all(row[2] == "Series" for row in json.loads(outs[name][-1]))
    report = json.loads((tmp_path / f"{name}_cy/_mx_admission.json").read_text(encoding="utf-8"))
    rows = {k: v for k, v in report.items() if isinstance(v, dict) and k.endswith("._f_model_point")}
    assert len(rows) == 2
    reader, = (v for k, v in report.items() if k.endswith("._c_Projection._f_sa"))
    if override == "row":
        assert all(not v["row_cell_inlined"] and "another class" in v["row_cell_kept"]
                   for v in rows.values())
        assert reader["row_fields_inlined"] == 0
    elif override == "reference":
        assert reader["row_fields_inlined"] == 0
    else:
        assert reader["row_fields_inlined"] == 1
