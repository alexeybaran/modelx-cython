"""Python access to compiled table cells and table References (D101, D101a)."""

import subprocess
import sys
import textwrap


MODEL = """\
import modelx as mx
import pandas as pd
from modelx_helpers import pdn

FRAME = pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0], "rate": [0.01, 0.02, 0.03, 0.04]},
                     index=pd.Index([1, 2, 3, 4], name="policy_id"))
IDS = (1, 2, 3, 4)


def build():
    model = mx.new_model("Tables")
    data = model.new_space("Data", refs={"pd": pd, "pdn": pdn})
    data.raw = FRAME
    data.new_cells("points", "def points():\\n    return pdn.table(raw)\\n")
    data.new_cells("table", "def table():\\n    return pdn.table(FRAME2)\\n")
    data.FRAME2 = FRAME
    proj = model.new_space("Projection", formula="lambda policy_id: None", refs={"data": data, "ids": IDS},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"pv": pv()}}')
    proj.new_cells("sa", 'lambda: data.points().loc[policy_id, "sa"]')
    proj.new_cells("rate", 'lambda: data.table().loc[policy_id, "rate"]')
    proj.new_cells("pv", "lambda: sum(sa() * (1.0 + rate()) ** t for t in range(12))")
    return model
"""

CHECKS = """\
import sys, threading
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
import pandas as pd
from modelx_helpers import pdn
from tables_model import FRAME, build

package = import_module("tables_nomx_cy")
assert isinstance(import_module("tables_nomx_cy._mx_classes").__loader__, ExtensionFileLoader)
runtime = import_module("modelx_cython.runtime.table_cells")
Model = type(package.mx_model)


def total(m, workers):
    m._mx_collect_set_workers_v2(workers)
    return m.Projection.collect().sum.pv


def live(points=None, table=None, raw=None):
    m = build()
    try:
        if points is not None:
            m.Data.points[()] = pdn.table(points)
        if table is not None:
            m.Data.table = pdn.table(table)
        if raw is not None:
            m.Data.raw = raw
        return sum(m.Projection[i].pv() for i in (1, 2, 3, 4))
    finally:
        m.close()


base = live()
m = Model()
v = m.Data.points()
assert isinstance(v, pd.DataFrame) and v is m.Data.points() and v is m.Data.points[()]
assert v.equals(FRAME) and float(v.loc[2, "sa"]) == 200.0
for edit in (lambda t: t.loc.__setitem__((1, "sa"), 1.0), lambda t: t.__setitem__("sa", 0.0),
             lambda t: t.at.__setitem__((1, "sa"), 1.0), lambda t: t.fillna(0, inplace=True)):
    try:
        edit(v)
    except TypeError as error:
        assert "Data.points" in str(error) and "Data.points[()] = t" in str(error), error
    else:
        raise AssertionError("an in-place write did not raise")
assert abs(total(m, 1) - base) < 1e-9

doubled = FRAME.assign(sa=FRAME["sa"] * 2)
for workers in (1, 4):
    m = Model()
    before = m.Projection[1].pv()
    assert abs(total(m, workers) - base) < 1e-9
    m.Data.points[()] = doubled
    assert abs(m.Projection[1].pv() / before - 2.0) < 1e-12      # dependants recalculated
    assert abs(total(m, workers) - live(points=doubled)) < 1e-9
    m.Data.table = FRAME.assign(rate=FRAME["rate"] * 0)            # the space.cell spelling
    assert abs(total(m, workers) - live(points=doubled, table=FRAME.assign(rate=FRAME["rate"] * 0))) < 1e-9

m = Model()
assert abs(total(m, 4) - base) < 1e-9
m.Data.raw = doubled                       # a Reference holding a table, after a run
assert abs(total(m, 4) - live(raw=doubled)) < 1e-9
assert m.Data.raw.equals(doubled)

m = Model()
for bad, text in ((FRAME.drop(columns="sa"), "no column 'sa'"), (FRAME.assign(sa="x"), "column 'sa' holds"),
                  (FRAME.set_axis(pd.Index(list("abcd"), name="policy_id")), "key levels")):
    try:
        m.Data.points[()] = bad
    except ValueError as error:
        assert "Data.points" in str(error) and text in str(error), error
    else:
        raise AssertionError("an unfit table was accepted")
assert abs(total(m, 1) - base) < 1e-9

m = Model()
m._mx_collect_set_workers_v2(4)
real_end, seen = runtime.collect_end, []
def probing_end(model):
    def other():
        try:
            model.Data.points[()] = doubled
        except RuntimeError as error:
            seen.append(str(error))
    thread = threading.Thread(target=other)
    thread.start(); thread.join()
    real_end(model)
runtime.collect_end = probing_end
try:
    assert abs(m.Projection.collect().sum.pv - base) < 1e-9
finally:
    runtime.collect_end = real_end
assert seen and "collect" in seen[0], seen

m = Model()
m.Data.points[()] = doubled
m.Projection.clear_all()
assert abs(total(m, 1) - live(points=doubled)) < 1e-9       # inputs of Data kept, as live
m.Data.points.clear_all()
assert abs(total(m, 4) - base) < 1e-9                       # the cell's input dropped, as live
m.Data.points[()] = doubled
m.Data.clear_all()
assert abs(total(m, 1) - base) < 1e-9 and m.Data.points().equals(FRAME)
print("ok")
"""


def test_table_cells_get_assign_edit(tmp_path):
    (tmp_path / "tables_model.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / "build_model.py").write_text(textwrap.dedent("""\
        from tables_model import build
        m = build()
        m.export("tables_nomx", locked_spaces=["Data"])
        m.close()
        """), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from tables_nomx import mx_model\nfor k in (1, 2, 3, 4):\n    mx_model.Projection[k].collect_outputs()\n"
        "mx_model.Projection.collect()\n", encoding="utf-8")
    (tmp_path / "spec.py").write_text(
        repr({"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}), encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600)
    run("build_model.py")
    run("-m", "modelx_cython", "tables_nomx", "--sample", "sample.py", "--spec", "spec.py")
    generated = (tmp_path / "tables_nomx_cy/_mx_classes.py").read_text(encoding="utf-8")
    pxd = (tmp_path / "tables_nomx_cy/_mx_classes.pxd").read_text(encoding="utf-8")
    assert "def _mx_cell_points(self)" in generated and "def _mx_cell_table(self)" in generated
    assert "self._mx_ref_raw" in generated and "cdef object _mx_ref_raw" in pxd
    assert "cdef object _mx_ref_FRAME2" in pxd
    assert "cdef public object ids" in pxd               # a Reference that holds no table keeps its name
    run("checks.py")
