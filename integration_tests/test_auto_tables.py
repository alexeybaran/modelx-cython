"""Plain pandas input tables admitted on their traced value (auto-tables, D104), in compiled builds.

One model holds two plain pandas tables (``Data.f`` keyed (sex, age), ``Data.g`` keyed age) and the
44 operations of the assess-syntax2 experiment, each written twice: in a zero-argument cell of the
per-policy Space (class B unless lowered) and in a cell with an argument (class C unless lowered).
Awkward tables sit beside them: duplicate keys (not admitted: stays pandas), float keys, string
keys, NaN values, an enum key level, a MultiIndex with unused levels, a table built per policy, the
same frame returned by two cells, a Series, and tuple-valued cells (a float tuple used as a key). The checks compare every
cell of the compiled model with live modelx, at 1 and 4 workers, and read the build report.
"""
import json
import re
import subprocess
import sys
import textwrap

OPS = [
    # (name, expression on T = data.<table>(), table, class when written per step: A lowered, C pandas)
    ("loc_full_key", 'T.loc[("F", 31), "mort_rate"]', "f", "A"),
    ("at_key", 'T.at[31, "q"]', "g", "A"),
    ("in_index", "31 in T.index", "g", "A"),
    ("index_max", "T.index.max()", "g", "A"),
    ("prefix_then_key", 'T.loc["M"].loc[31, "mort_rate"]', "f", "A"),
    ("iter_index", "[k for k in T.index]", "g", "A"),
    ("len_index", "len(T.index)", "g", "A"),
    ("asof", "T.index.asof(31.5)", "g", "C"),
    ("prefix_with_column", 'T.loc["F", "mort_rate"]', "f", "C"),
    ("label_slice", "T.loc[31:32]", "g", "C"),
    ("label_slice_column", 'T.loc[31:32, "q"]', "g", "C"),
    ("list_of_keys", 'T.loc[[30, 32], "q"]', "g", "C"),
    ("mask", 'T[T["q"] > 0.015]', "g", "C"),
    ("mask_and", 'T[(T["lapse"] == 0.1) & (T["mort_rate"] <= 0.002)]', "f", "C"),
    ("mask_loc_column", 'T.loc[T["q"] > 0.015, "w"]', "g", "C"),
    ("iloc", "T.iloc[1]", "g", "C"),
    ("iloc_last_column", 'T.iloc[-1]["q"]', "g", "C"),
    ("values", "T.values", "g", "C"),
    ("column_to_numpy", 'T["q"].to_numpy()', "g", "C"),
    ("groupby_sum", 'T.groupby(level="sex")["mort_rate"].sum()', "f", "C"),
    ("merge", 'pd.merge(T.reset_index(), data.g_raw.reset_index(), on="age")', "g", "C"),
    ("iterrows", '[(k, r["q"]) for k, r in T.iterrows()]', "g", "C"),
    ("itertuples", "[r.q for r in T.itertuples()]", "g", "C"),
    ("len", "len(T)", "g", "C"),
    ("column_in", '"q" in T', "g", "C"),
    ("iter_columns", "list(T)", "g", "C"),
    ("column", 'T["q"]', "g", "C"),
    ("column_times_2", 'T["q"] * 2', "g", "C"),
    ("column_sum", 'T["q"].sum()', "g", "C"),
    ("iter_column", '[v for v in T["q"]]', "g", "C"),
    ("zip_index_column", 'list(zip(T.index, T["q"]))', "g", "C"),
    ("isna_key", 'pd.isna(T.loc[31, "q"])', "g", "A"),
    ("interp_columns", 'np.interp(2.5, T["w"], T["q"])', "g", "C"),
    ("interp_index", 'np.interp(31.5, T.index, T["q"])', "g", "C"),
    ("transpose", "T.T", "g", "C"),
    ("sort_values", 'T.sort_values("q", ascending=False)', "g", "C"),
    ("sort_index", "T.sort_index(ascending=False)", "f", "C"),
    ("shape", "T.shape", "g", "C"),
    ("columns", "list(T.columns)", "g", "C"),
    ("level_values", 'list(T.index.get_level_values("age"))', "f", "C"),
    ("row_then_column", 'T.loc[31]["q"]', "g", "A"),
    ("row_sum", "T.loc[31].sum()", "g", "A"),
    ("chained", 'T["q"][31]', "g", "A"),
    ("missing_key", 'T.loc[99, "q"]', "g", "A"),
    # beyond the experiment's 44: operators on the table itself
    ("table_times_2", "T * 2", "g", "C"),
    ("table_compare", "(T > 1.5).sum().sum()", "g", "C"),
]
assert len(OPS) == 46

MODEL = '''\
import enum
import modelx as mx
import numpy as np
import pandas as pd

from auto_enums import Sex

OPS = {ops!r}
IDS = (1, 2, 3)


F_RAW = pd.DataFrame({{"sex": ["F", "F", "F", "M", "M", "M"], "age": [30, 31, 32, 30, 31, 32],
                      "mort_rate": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
                      "lapse": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]}})
G_RAW = pd.DataFrame({{"age": [30, 31, 32, 33], "q": [0.01, 0.02, 0.03, 0.04], "w": [1.0, 2.0, 3.0, 4.0]}})

DATA = {{
    "f": "return f_raw.set_index(['sex', 'age'])",
    "g": "return g_raw.set_index('age')",
    "g2": "return g()",                                     # the same frame from two cells
    "dup": "return pd.DataFrame({{'x': [1.0, 2.0, 3.0]}}, index=pd.Index([1, 1, 2], name='k'))",
    "fkey": "return pd.DataFrame({{'x': [1.0, 2.0]}}, index=pd.Index([0.5, 1.5], name='k'))",
    "fkey2": ("return pd.DataFrame({{'x': [1.0, 2.0]}}, "
              "index=pd.MultiIndex.from_tuples([(0.5, 1.5), (1.5, 2.5)], names=['a', 'b']))"),
    "skey": "return pd.DataFrame({{'x': [1.0, 2.0], 'label': ['a', 'b']}}, index=pd.Index(['p', 'q'], name='k'))",
    "nans": "return pd.DataFrame({{'x': [1.0, np.nan], 's': ['a', None]}}, index=pd.Index([1, 2], name='k'))",
    "ekey": ("frame = f_raw.copy()\\n    frame['sex'] = [Sex[n] for n in frame['sex']]\\n"
             "    return frame.set_index(['sex', 'age'])"),
    "unused": "frame = f().loc[['M']]\\n    return frame",          # MultiIndex with an unused level
    "ser": "return g()['q']",
    "rowkeys": "return g_raw.copy()",                        # unnamed RangeIndex: row numbers as keys
}}

PROJ = {{
    "sa": "return 100.0 * policy_id",
    "dup_x": "return data.dup().loc[2, 'x']",
    "fkey_x": "return data.fkey().loc[1.5, 'x']",
    "skey_x": "return data.skey().loc['q', 'x'] + len(data.skey().loc['p', 'label'])",
    "nan_x": "return float(pd.isna(data.nans().loc[2, 'x'])) + float(pd.isna(data.nans().loc[2, 's']))",
    "ekey_x": "return data.ekey().loc[(Sex.M, 31), 'mort_rate']",
    "unused_x": "return data.unused().loc[('M', 32), 'lapse'] + len(data.unused().index.levels[0])",
    "per_policy": ("return pd.DataFrame({{'v': [sa() * k for k in range(3)]}}, "
                   "index=pd.Index(range(3), name='k'))"),
    "per_policy_x": "return per_policy().loc[2, 'v']",
    "g2_x": "return data.g2().loc[33, 'q']",
    "ser_x": "return data.ser()[32]",
    "rows_x": "return data.rowkeys().loc[1, 'q']",
    "vec": "return (sa(), sa() * 2.0, 0.5)",                # a tuple of floats stays a tuple
    "ivec": "return (1, 2, int(policy_id))",                # a tuple of ints stays a tuple
    "ivec_key": "return data.f().loc[('F', 30 + ivec()[1]), 'lapse']",
    "fkey_tuple": "return (0.5, 1.5 + 0.0 * sa())",          # a float key: a tuple, never a vector
    "fkey_tuple_x": "return data.fkey2().loc[fkey_tuple(), 'x'] + data.fkey2().loc[(1.5, 2.5), 'x']",
    "lvec": "return [sa(), sa() * 2.0, 0.5]",               # a list of floats: a vector
    "mp": "return data.g().loc[30 + int(policy_id)]",       # a row of an admitted table (D119)
    "mp_q": "return mp()['q'] * 2.0 + mp()['w']",           # compiled code reads the row natively
    "vec_sum": "return sum(vec()) + vec()[1] + sum(ivec()) + fkey_tuple_x() + ivec_key() + sum(lvec())",
    "scaled_x": "return scaled(2).loc[31, 'q'] + scaled(3).loc[32, 'w']",
    "report": ("return pd.DataFrame({{'sa': [sa() * t for t in range(3)]}}, "     # read by nobody natively
               "index=pd.Index(range(3), name='t'))"),
}}


def _op_body(expression, table):
    return (f"T = data.{{table}}()\\n    try:\\n        return {{expression}}\\n"
            "    except Exception as error:\\n        return type(error).__name__")


def build():
    m = mx.new_model("Auto")
    data = m.new_space("Data", refs={{"pd": pd, "np": np, "f_raw": F_RAW, "g_raw": G_RAW, "Sex": Sex}})
    for name, body in DATA.items():
        data.new_cells(name, f"def {{name}}():\\n    {{body}}\\n")
    proj = m.new_space("Projection", formula="lambda policy_id: None",
                       refs={{"data": data, "pd": pd, "np": np, "Sex": Sex, "ids": IDS}},
                       collect_over={{"policy_id": IDS}},
                       collect_outputs='lambda: {{"sum": {{"total": total()}}}}')
    for name, body in PROJ.items():
        proj.new_cells(name, f"def {{name}}():\\n    {{body}}\\n")
    proj.new_cells("scaled", "def scaled(k):\\n    return data.g() * float(k)\\n")   # a frame per argument
    for name, expression, table, _ in OPS:
        proj.new_cells("b_" + name, f"def b_{{name}}():\\n    {{_op_body(expression, table)}}\\n")
        proj.new_cells("c_" + name, f"def c_{{name}}(t):\\n    {{_op_body(expression, table)}}\\n")
    proj.new_cells("total", "def total():\\n    return vec_sum() + per_policy_x() + sum(c_len(t) for t in range(3))\\n")
    return m


if __name__ == "__main__":
    model = build()
    model.export("auto_nomx", locked_spaces=["Data"])
    model.close()
'''

SAMPLE = '''\
from auto_nomx import mx_model
from auto_model import PROJ, OPS
for k in (1, 2, 3):
    p = mx_model.Projection[k]
    for name in PROJ:
        getattr(p, name)()
    for name, *_ in OPS:
        getattr(p, "b_" + name)()
        for t in range(2):
            getattr(p, "c_" + name)(t)
    p.collect_outputs()
mx_model.Projection.collect()
'''

CHECKS = '''\
import sys
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
import numpy as np
import pandas as pd
from auto_model import OPS, PROJ, build

package = import_module("auto_nomx_cy")
assert isinstance(import_module("auto_nomx_cy._mx_classes").__loader__, ExtensionFileLoader)
Model = type(package.mx_model)


def same(a, b):
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return isinstance(b, (pd.DataFrame, pd.Series)) and a.equals(b)
    if isinstance(a, np.ndarray):
        # a cell whose value is a float64 array returns a memoryview of it from a compiled model, as any
        # array-valued cell does (not a table rule): compared by value
        b = np.asarray(b)
        return a.shape == b.shape and bool((a == b).all())
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a != a:
        return b != b
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return abs(a - b) <= 1e-7 + 1e-10 * abs(a)
    return type(a) is type(b) and a == b or a == b


live = build()
expected = {}
for k in (1, 2, 3):
    p = live.Projection[k]
    for name in PROJ:
        expected[k, name] = getattr(p, name)()
    for name, *_ in OPS:
        expected[k, "b_" + name] = getattr(p, "b_" + name)()
        expected[k, "c_" + name] = getattr(p, "c_" + name)(1)
live_total = sum(live.Projection[k].total() for k in (1, 2, 3))
for name, *_ in OPS:
    assert expected[1, "c_" + name] is not None
assert expected[1, "c_missing_key"] == "KeyError", expected[1, "c_missing_key"]

failures = []
for workers in (1, 4):
    m = Model()
    m.collect_workers = workers
    total = m.Projection.collect().sum.total
    if abs(total - live_total) > 1e-7 + 1e-10 * abs(live_total):
        failures.append(("collect", workers, total, live_total))
    for (k, name), want in expected.items():
        cell = getattr(m.Projection[k], name)
        got = cell(1) if name.startswith("c_") else cell()
        if not same(want, got):
            failures.append((workers, k, name, repr(want)[:80], repr(got)[:80]))
assert not failures, "\\n".join(map(str, failures))

m = Model()
frame = m.Data.g()
assert isinstance(frame, pd.DataFrame) and frame.equals(live.Data.g())
assert m.Data.g2() is m.Data.g()            # the same frame from two cells: one table
assert type(m.Projection[1].report()) is pd.DataFrame       # a report stays the model's own DataFrame
row = m.Projection[2].mp()                  # D130: a row cell gives Python the plain pandas row, as live
assert type(row) is pd.Series and row.equals(live.Projection[2].mp()) and row.name == live.Projection[2].mp().name
assert row is m.Projection[2].mp()
assert type(m.Projection[1].b_mask()) is pd.DataFrame
assert isinstance(m.Data.dup(), pd.DataFrame) and m.Data.dup().equals(live.Data.dup())
assert type(m.Data.g()) is pd.DataFrame     # D130: the plain frame, editable in place
before = m.Projection[1].c_column_sum(0)
m.Data.g()["q"] = m.Data.g()["q"] * 3       # an in-place edit: nothing changes until a clear
assert m.Projection[1].c_column_sum(0) == before
m.Projection.clear_all()
assert abs(m.Projection[1].c_column_sum(0) - 3 * before) < 1e-12
m.Data.g[()] = live.Data.g().assign(q=live.Data.g()["q"] * 2)
assert abs(m.Projection[1].c_column_sum(0) - 3 * before) < 1e-12      # assignment clears nothing
m.Projection.clear_all()
assert abs(m.Projection[1].c_column_sum(0) - 0.2) < 1e-12
try:
    m.Data.g[()] = pd.DataFrame({"q": [1.0, 2.0], "w": [1.0, 1.0]}, index=pd.Index([1, 1], name="age"))
except ValueError as error:
    assert "Data.g: duplicate key 1" in str(error) and "(C3)" in str(error), error
else:
    raise AssertionError("a duplicate-key frame was assigned")
print("ok")
'''


def _write(tmp_path):
    ops = [(n, e, t, c) for n, e, t, c in OPS]
    (tmp_path / "auto_model.py").write_text(MODEL.format(ops=ops), encoding="utf-8")
    (tmp_path / "auto_enums.py").write_text(
        "import enum\n\n\nclass Sex(enum.IntEnum):\n    F = 0\n    M = 1\n", encoding="utf-8")
    (tmp_path / "sample.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")


def _spec(tmp_path, **extra):
    spec = {"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}, **extra}
    path = tmp_path / ("spec_strict.py" if extra else "spec.py")
    path.write_text(repr(spec), encoding="utf-8")
    return path.name


def _run(tmp_path, *args, check=True):
    return subprocess.run([sys.executable, *args], cwd=tmp_path, check=check, timeout=1200,
                          capture_output=True, text=True)


def test_auto_tables_compiled_equal_live(tmp_path):
    _write(tmp_path)
    _run(tmp_path, "auto_model.py")
    build = _run(tmp_path, "-m", "modelx_cython", "auto_nomx", "--sample", "sample.py", "--spec", _spec(tmp_path))
    log = build.stdout + build.stderr
    summary = json.loads((tmp_path / "auto_nomx_cy/_mx_table_admission.json").read_text(encoding="utf-8"))
    admitted = set(summary["admitted"])
    assert {"Data.f", "Data.g", "Data.g2", "Data.fkey", "Data.skey", "Data.nans", "Data.ekey",
            "Data.unused", "Data.rowkeys"} <= admitted, admitted
    # D120: a DataFrame built per item (a Projection cell) is never converted; it stays pandas and its key
    # reads are pandas reads, warned (the per-policy table, the report, the operation cells)
    assert not any(name.startswith("Projection.") for name in admitted), admitted
    assert {"Projection.per_policy", "Projection.report", "Projection.b_mask",
            "Projection.b_transpose"} <= set(summary["per_item_frames"]), summary["per_item_frames"]
    assert re.search(r"Projection.per_policy_x: .* reads Projection.per_policy, a DataFrame built for every item", log)
    # a cell with arguments is never wrapped: no lowered read takes a receiver with arguments
    assert summary["not_admitted"] == {
        "Data.dup": ["duplicate key 1 (1 duplicated rows) (C3); keys must be unique"]}
    assert summary["same_frame_as"] == {"Data.g2": "Data.g"}
    assert "Data.rowkeys" in summary["row_numbers_as_keys"]
    assert not summary["built_more_than_once"], summary["built_more_than_once"]
    assert "Data.ser" in summary["series_cells"]
    assert "Data.dup: not admitted as a table: duplicate key 1" in log
    report = json.loads((tmp_path / "auto_nomx_cy/_mx_admission.json").read_text(encoding="utf-8"))

    def reads(cell):
        key = next(k for k in report if k.endswith("._c_Projection." + cell) or k.endswith("._f_" + cell))
        return report[key].get("table_admission", {}).get("reads", [])

    classes = {}
    for name, expression, table, per_step in OPS:
        b, c = reads("b_" + name), reads("c_" + name)
        assert b and c, name
        assert all(r["class"] in "AB" for r in b), (name, b)
        assert all(r["class"] in "AC" for r in c), (name, c)
        classes[name] = {r["class"] for r in c}
        if per_step == "A":
            assert classes[name] == {"A"}, (name, c)
        elif per_step == "C":
            assert "C" in classes[name], (name, c)
            assert f"Projection.c_{name}: " in log, name
    assert any(e["kind"] == "pandas-per-step" for k, v in report.items() if k.endswith("c_mask")
               for e in v["slow_paths"])
    # tuples stay tuples whatever their elements (a float tuple can be a key: fkey_tuple)
    tuples = [v for k, v in report.items() if k.endswith(("_f_vec", "_f_ivec", "_f_fkey_tuple"))]
    assert len(tuples) == 3 and not any(v.get("vectors", {}).get("shape") for v in tuples), tuples
    lists = [v for k, v in report.items() if k.endswith("_f_lvec")]
    assert lists and lists[0]["vectors"]["shape"] == "V", lists     # a list of floats is a vector
    assert "Data.fkey2" in admitted
    generated = (tmp_path / "auto_nomx_cy/_mx_classes.py").read_text(encoding="utf-8")
    assert "def _mx_cell_mp(self)" in generated and "row_series(self._mx_cell_mp(), 'Projection.mp')" in generated
    assert "self._mx_cell_mp()" in generated.split("def _f_mp_q")[1].split("def ")[0] or "_mx_cell_mp" in generated
    (tmp_path / "ops-classes.json").write_text(json.dumps(classes, default=sorted), encoding="utf-8")
    assert _run(tmp_path, "checks.py").stdout.strip().endswith("ok")


def test_auto_tables_strict_option(tmp_path):
    _write(tmp_path)
    _run(tmp_path, "auto_model.py")
    result = _run(tmp_path, "-m", "modelx_cython", "auto_nomx", "--sample", "sample.py", "--spec",
                  _spec(tmp_path, strict_tables=True), "--translate-only", check=False)
    assert result.returncode != 0
    text = result.stdout + result.stderr
    assert "strict_tables" in text and "Data.dup: not admitted" in text, text[-2000:]
