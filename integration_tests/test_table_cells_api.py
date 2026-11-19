"""Python access to compiled table cells and table References (D101a, D130).

D130: nothing is cleared automatically. A table cell returns the plain DataFrame it was built
from, editable in place; assigning a table (cell or Reference) or a scalar Reference changes no
computed value; ``space.clear_all()`` clears that Space (and its items) and the next values,
on the collect and the direct route, read the edited or new input. The exported pure-Python
model behaves the same (asserted on both)."""

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
    data.new_cells("points", "def points():\\n    return raw.copy()\\n")       # plain pandas (D121)
    data.new_cells("table", "def table():\\n    return FRAME2.copy()\\n")
    data.FRAME2 = FRAME
    proj = model.new_space("Projection", formula="lambda policy_id: None",
                           refs={"data": data, "ids": IDS, "factor": 1.0},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"pv": pv()}}')
    proj.new_cells("sa", 'lambda: data.points().loc[policy_id, "sa"]')
    proj.new_cells("rate", 'lambda: data.table().loc[policy_id, "rate"]')
    proj.new_cells("pv", "lambda: factor * sum(sa() * (1.0 + rate()) ** t for t in range(12))")
    return model
"""

CHECKS = """\
import sys, threading
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
import pandas as pd
from modelx_helpers import pdn
from tables_model import FRAME, build

name = sys.argv[1]
compiled = name.endswith("_cy")
package = import_module(name)
if compiled:
    assert isinstance(import_module(name + "._mx_classes").__loader__, ExtensionFileLoader)
    runtime = import_module("modelx_cython.runtime.table_cells")
Model = type(package.mx_model)


def total(m, workers):
    m.collect_workers = workers
    return m.Projection.collect().sum.pv


def direct(m):
    return sum(m.Projection[i].pv() for i in (1, 2, 3, 4))


def live(points=None, raw=None, factor=None):
    m = build()
    try:
        if raw is not None:
            m.Data.raw = raw
        if points is not None:
            m.Data.points[()] = points
        if factor is not None:
            m.Projection.factor = factor
        return sum(m.Projection[i].pv() for i in (1, 2, 3, 4))
    finally:
        m.close()


def close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


base = live()
edited = FRAME.copy()
edited.loc[2, "sa"] = 2000.0
doubled = FRAME.assign(sa=FRAME["sa"] * 2)

# (1) a table cell returns a plain, editable DataFrame, the same object every time
m = Model()
v = m.Data.points()
assert type(v) is pd.DataFrame and v is m.Data.points() and v.equals(FRAME), type(v)
assert close(total(m, 1), base) and close(direct(m), base)

# (2) in-place edit: nothing changes until the dependent Space is cleared; then both routes read it
m = Model()
item = m.Projection[2]
before = item.pv()
assert close(total(m, 1), base)
m.Data.points().loc[2, "sa"] = 2000.0
assert m.Projection[2].pv() == before and close(total(m, 1), base)      # no clear: old values
m.Projection.clear_all()
assert m.Projection[2] is not item                                        # items dropped by the clear
assert close(direct(m), live(points=edited))
for workers in (1, 4):
    assert close(total(m, workers), live(points=edited))
assert m.Data.points().loc[2, "sa"] == 2000.0                             # Data kept its edited frame

# (3) assigning a table cell (the compiled model's assign API; the exported model's cells are plain
# methods without one): no change until the clear
if compiled:
    m = Model()
    assert close(total(m, 1), base) and close(direct(m), base)
    m.Data.points[()] = doubled
    assert close(total(m, 1), base) and close(direct(m), base)
    m.Projection.clear_all()
    assert close(direct(m), live(points=doubled)) and close(total(m, 4), live(points=doubled))

# (4) assigning a Reference that holds a table: no change until the clears (Data recomputes points)
m = Model()
assert close(total(m, 1), base) and close(direct(m), base)
m.Data.raw = doubled
assert m.Data.raw.equals(doubled)
assert close(total(m, 1), base) and close(direct(m), base)
m.Data.clear_all()
m.Projection.clear_all()
assert close(direct(m), live(raw=doubled)) and close(total(m, 1), live(raw=doubled))

# (5) a scalar Reference: the same rule
m = Model()
assert close(total(m, 1), base) and close(direct(m), base)
m.Projection.factor = 3.0
assert close(total(m, 1), base) and close(direct(m), base)
m.Projection.clear_all()
assert close(direct(m), live(factor=3.0)) and close(total(m, 4), live(factor=3.0))

# (6) inputs: kept by a clear of another Space, dropped by the clear of their own Space, as live
if compiled:
    m = Model()
    m.Data.points[()] = doubled
    m.Projection.clear_all()
    assert close(total(m, 1), live(points=doubled))
    m.Data.clear_all()                          # drops it: the formula runs again
    m.Projection.clear_all()
    assert close(total(m, 4), base) and m.Data.points().equals(FRAME)
    # only Space.clear_all() and Model.clear_all() are provided: every other clear of live modelx raises
    for target, method in ((m.Data.points, "clear"), (m.Data.points, "clear_all"), (m.Data.points, "clear_at"),
                           (m.Projection, "clear_cells"), (m.Projection, "clear_items"), (m.Projection, "clear_at")):
        try:
            getattr(target, method)()
        except NotImplementedError as error:
            assert method + "() is not provided" in str(error) and "Space.clear_all()" in str(error) \
                and "Model.clear_all()" in str(error), error
        else:
            raise AssertionError(method + " ran")
else:
    for method in ("clear_cells", "clear_items", "clear_at"):
        assert not hasattr(m.Projection, method), method

# (7) one ItemSpace (del, as before) and the whole model
m = Model()
item = m.Projection[3]
del m.Projection[3]
assert m.Projection[3] is not item
m = Model()
assert close(total(m, 1), base)
m.Data.points().loc[2, "sa"] = 2000.0
m.Projection.factor = 3.0
m.clear_all()                                   # the whole model: Data reruns its formula (edit lost, as live)
assert close(direct(m), live(factor=3.0)) and close(total(m, 1), live(factor=3.0))

if compiled:
    m = Model()
    for bad, text in ((FRAME.drop(columns="sa"), "no column 'sa'"), (FRAME.assign(sa="x"), "column 'sa' holds"),
                      (FRAME.set_axis(pd.Index(list("abcd"), name="policy_id")), "key levels")):
        try:
            m.Data.points[()] = bad
        except ValueError as error:
            assert "Data.points" in str(error) and text in str(error), error
        else:
            raise AssertionError("an unfit table was accepted")
    assert close(total(m, 1), base)

    # (8) assignment and clear are refused while a collect runs
    m = Model()
    m.collect_workers = 4
    real_end, seen = runtime.collect_end, []
    def probing_end(model):
        def other():
            for attempt in (lambda: model.Data.points.__setitem__((), doubled),
                            lambda: model.Projection.clear_all()):
                try:
                    attempt()
                except RuntimeError as error:
                    seen.append(str(error))
        thread = threading.Thread(target=other)
        thread.start(); thread.join()
        real_end(model)
    runtime.collect_end = probing_end
    try:
        assert close(m.Projection.collect().sum.pv, base)
    finally:
        runtime.collect_end = real_end
    assert len(seen) == 2 and all("collect" in text for text in seen), seen
print("ok", name)
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
    run("checks.py", "tables_nomx_cy")
    run("checks.py", "tables_nomx")         # the exported pure-Python model behaves the same (D130)
