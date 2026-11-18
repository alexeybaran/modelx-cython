"""Tables in compiled models behave exactly as on live modelx (T-1 probe; design toolkit p2/tables/DESIGN.md).

One model applies each family of pandas use to tables held by zero-argument Data cells: a flat frame (int
index), a two-level frame, a Series, a non-unique two-level frame (as Annuity_JP_S's mort_anchor_table), and the
same frame through the table library's ``pdn.table`` (only while that library exists). Every Calc cell has an
argument ``t`` so its key depends on the call. Each cell is compared with live modelx for t = 0..3 STRICTLY:
same type and value, or the same exception. ``KNOWN`` lists, by cell, how the compiled model differs today;
each step of the table redesign removes entries, and the final state is an empty ``KNOWN`` except the documented
boundary types (``BOUNDARY``: a C-typed cell result returns ``float``/``int``, an array cell a memoryview, as
upstream modelx-cython does).

Separately, ``test_reassign_and_clear`` checks the input-change contract agreed with the owner: assigning a new
frame to a Data cell and calling clear_all (on the dependent Space and on the model) gives live's results on the
new frame; without clear_all, cached results stay, but no prepared copy of the reassigned cell survives.
Model.clear_all discards an assigned input on live modelx (the cell recomputes its formula); compiled the same.
"""
import json
import subprocess
import sys

import pytest

OPS = {  # name: (sources, expression on T = data.<source>() and the argument t)
    'loc_scalar': ('flat', "T.loc[20 + t, 'q']"),
    'loc_row': ('flat', "T.loc[20 + t]"),
    'loc_row_field': ('flat', "T.loc[20 + t]['q']"),
    'loc_missing': ('flat', "T.loc[99 + t, 'q']"),
    'loc_tuple': ('multi', "T.loc[(('A', 'B', 'C')[t % 3], 1 + t % 2), 'rate']"),
    'loc_prefix': ('multi', "T.loc[('A', 'B', 'C')[t % 3]]"),
    'loc_prefix_key': ('multi', "T.loc[('A', 'B', 'C')[t % 3]].loc[1 + t % 2, 'rate']"),
    'loc_prefix_col': ('multi', "T.loc[('A', 'B', 'C')[t % 3], 'rate']"),
    'loc_slice': ('flat', "T.loc[20:20 + t, 'q']"),
    'loc_list': ('flat', "T.loc[[20, 20 + t], 'q']"),
    'loc_mask': ('flat', "T.loc[T['q'] > 0.001 * t]"),
    'loc_callable': ('flat', "T.loc[lambda d: d.index > 20 + t]"),
    'at': ('flat', "T.at[20 + t, 'r']"),
    'at_tuple': ('multi', "T.at[(('A', 'B', 'C')[t % 3], 1 + t % 2), 'rate']"),
    'nonunique_key_col': ('dup', "T.loc[(('A', 'B')[t % 2], 1), 'rate'].iloc[0]"),
    'nonunique_rows': ('dup', "T.loc[(('A', 'B')[t % 2], 1)]"),
    'iloc': ('flat', "T.iloc[t, 0]"),
    'iloc_row': ('flat', "T.iloc[t]"),
    'iat': ('flat', "T.iat[t, 1]"),
    'col_attr': ('flat', "T.q"),
    'col_item': ('flat', "T['q']"),
    'col_keyed': ('flat', "T['q'][20 + t]"),
    'col_list': ('flat', "T[['q', 'r']]"),
    'index_max': ('flat multi', "T.index.max()"),
    'index_min': ('flat', "T.index.min()"),
    'index_max_part': ('multi', "T.index.max()[t % 2]"),
    'index_asof': ('flat', "T.index.asof(20.5 + t)"),
    'index_get_loc': ('flat', "T.index.get_loc(20 + t)"),
    'index_arith': ('flat', "T.index + t"),
    'index_cmp': ('flat', "T.index > 20 + t"),
    'index_list': ('flat', "list(T.index)"),
    'index_len': ('flat', "len(T.index) + t"),
    'index_in': ('flat multi', "(20 + t) in T.index"),
    'len': ('flat series', "len(T) + t"),
    'iter': ('flat', "list(T)"),
    'contains': ('flat', "'q' in T"),
    'arith': ('flat series', "T * (t + 1)"),
    'col_arith': ('flat', "T['q'] * (t + 1)"),
    'cmp': ('flat series', "T > 0.001 * t"),
    'sum': ('flat series', "T.sum()"),
    'col_mean': ('flat', "T['q'].mean() + t"),
    'max': ('flat series', "T.max()"),
    'values': ('flat series', "T.values"),
    'to_numpy': ('flat series', "T.to_numpy()"),
    'iterrows': ('flat', "[(k, r['q']) for k, r in T.iterrows()]"),
    'iterrows_sum': ('flat', "sum(r['q'] * (k - 19) for k, r in T.iterrows()) + t"),
    'itertuples': ('flat', "[tuple(x) for x in T.itertuples()]"),
    'groupby': ('multi', "T.groupby(level=0).sum()"),
    'merge': ('flat', "pd.DataFrame({'age': [20, 20 + t], 'w': [1.0, 2.0]}).merge(T, left_on='age', right_index=True)"),
    'join': ('flat', "pd.DataFrame({'age': [20, 20 + t]}).join(T, on='age')"),
    'apply': ('flat', "T.apply(sum)"),
    'transpose': ('flat', "T.T"),
    'pickle': ('flat', "pickle.loads(pickle.dumps(T))"),
    'isinstance': ('flat series', "isinstance(T, (pd.DataFrame, pd.Series))"),
    'type_name': ('flat multi series', "type(T).__name__"),
    'np_asarray': ('flat', "np.asarray(T)"),
    'np_exp': ('flat', "np.exp(T['q'])"),
    'pd_concat': ('flat', "pd.concat([T, T])"),
    'pd_frame': ('flat', "pd.DataFrame(T)"),
    'series_key': ('series', "T[20 + t]"),
    'series_loc': ('series', "T.loc[20 + t]"),
    'series_sum': ('series', "T.sum() + t"),
    'identity': ('flat', "T is data.flat()"),
    'mutate_loc': ('mut', "(T.loc.__setitem__((20, 'q'), 9.0 + t), T.loc[20, 'q'])[1]"),
}

DATA = {
    'flat': "pd.DataFrame({'q': [0.001, 0.002, 0.004, 0.008], 'r': [1.0, 2.0, 3.0, 4.0]},\n"
            "                        index=pd.Index([20, 21, 22, 23], name='age'))",
    'multi': "pd.DataFrame({'rate': [1.5, 1.6, 2.5, 2.6, 3.5, 3.6]},\n"
             "        index=pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2), ('C', 1), ('C', 2)],\n"
             "                                        names=['plan', 'year']))",
    'series': "pd.Series([0.001, 0.002, 0.004, 0.008], index=pd.Index([20, 21, 22, 23], name='age'), name='q')",
    'mut': "pd.DataFrame({'q': [0.001, 0.002, 0.004, 0.008], 'r': [1.0, 2.0, 3.0, 4.0]},\n"
           "                        index=pd.Index([20, 21, 22, 23], name='age'))",
    'dup': "pd.DataFrame({'rate': [1.5, 1.7, 2.5, 2.7]},\n"
           "        index=pd.MultiIndex.from_tuples([('A', 1), ('A', 1), ('B', 1), ('B', 1)], names=['tab', 'sex']))",
}

# Compiled differs from live today (84befe4), by cell: what the redesign removes step by step.
KNOWN = {
    'loc_row__flat': 'Record for Series',
    'loc_prefix__multi': 'NativeTable for DataFrame',
    'loc_callable__flat': 'KeyError on the callable',
    'merge__flat': 'TypeError: NativeTable is not a DataFrame',
    'join__flat': 'ValueError: Joining multiple DataFrames',
    'pickle__flat': 'NativeTable for DataFrame',
    'isinstance__flat': 'False for True',
    'type_name__flat': 'NativeTable for DataFrame',
    'type_name__multi': 'NativeTable for DataFrame',
    'np_asarray__flat': 'TypeError: Expected str',
    'pd_concat__flat': 'TypeError: cannot concatenate NativeTable',
    'pd_frame__flat': 'frame of column names',
    'mutate_loc__mut': "AttributeError: '_FrameTableLoc' object has no attribute '__setitem__'",
}

# Boundary types that equal upstream modelx-cython's (a C-typed cell result, an array-valued cell)
BOUNDARY = {'float for numpy.float64', 'int for numpy.int64', 'memoryview for ndarray', 'vector for list'}

MODEL = '''\
import modelx as mx
import numpy as np
import pandas as pd
import pickle
OPS = {ops!r}
DATA = {data!r}


def names():
    for op, (sources, _) in OPS.items():
        for source in sources.split():
            yield f"{{op}}__{{source}}"


def build():
    model = mx.new_model("Probe")
    data = model.new_space("Data")
    for name, expression in DATA.items():
        data.new_cells(name, f"def {{name}}():\\n    return {{expression}}\\n")
    data.pd = pd
    calc = model.new_space("Calc")
    calc.data, calc.pd, calc.np, calc.pickle = data, pd, np, pickle
    for op, (sources, expression) in OPS.items():
        for source in sources.split():
            calc.new_cells(f"{{op}}__{{source}}",
                           f"def {{op}}__{{source}}(t):\\n    T = data.{{source}}()\\n    return {{expression}}\\n")
    return model


if __name__ == "__main__":
    build().export("probe_nomx", locked_spaces=["Data"])
'''

SAMPLE = '''\
from probe_nomx import mx_model
from probe_model import names
for name in names():
    for t in range(4):
        try:
            getattr(mx_model.Calc, name)(t)
        except Exception:
            pass
'''

CHECKS = '''\
import json
import sys
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
import numpy as np
import pandas as pd
from probe_model import build, names

assert isinstance(import_module("probe_nomx_cy._mx_classes").__loader__, ExtensionFileLoader)
compiled = import_module("probe_nomx_cy").mx_model
live = build()


def outcome(space, name, t):
    try:
        return "ok", getattr(space, name)(t)
    except Exception as error:
        return "raise", error


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, (pd.DataFrame, pd.Series, pd.Index)):
        return a.equals(b) and (not isinstance(a, pd.DataFrame) or list(a.columns) == list(b.columns))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a != a:
        return b != b
    return bool(a == b)


def boundary(a, b):
    """The documented boundary types: equal value, the type a C-typed or array cell gives Python."""
    kinds = {(np.float64, float): "float for numpy.float64", (np.int64, int): "int for numpy.int64"}
    kind = kinds.get((type(a), type(b)))
    if kind and a == b:
        return kind
    if isinstance(a, np.ndarray) and type(b).__name__ == "_memoryviewslice" and np.array_equal(a, np.asarray(b)):
        return "memoryview for ndarray"
    if isinstance(a, list) and type(b).__name__ in ("I64Vec", "F64Vec") and list(b) == a:
        return "vector for list"
    if isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
        parts = [boundary(x, y) if not same(x, y) else "" for x, y in zip(a, b)]
        if all(p is not None for p in parts):
            found = {p for p in parts if p}
            return found.pop() if len(found) == 1 else ("" if not found else None)
    return None


result = {}
for name in names():
    verdict = "same"
    for t in range(4):
        (lk, lv), (ck, cv) = outcome(live.Calc, name, t), outcome(compiled.Calc, name, t)
        if lk == "raise" and ck == "raise":
            original = lv.__cause__ or lv
            if type(cv) is type(original) or type(cv).__name__ + ":" in str(lv):
                continue
            verdict = f"other error: live {type(lv).__name__}, compiled {type(cv).__name__}: {str(cv)[:120]}"
        elif ck == "raise":
            verdict = f"raises {type(cv).__name__}: {str(cv)[:120]}"
        elif lk == "raise":
            verdict = f"returns {type(cv).__name__} where live raises {type(lv).__name__}"
        elif not same(lv, cv):
            kind = boundary(lv, cv)
            verdict = f"boundary: {kind}" if kind else f"differs: live {type(lv).__name__} {repr(lv)[:60]}, compiled {type(cv).__name__} {repr(cv)[:60]}"
        if verdict != "same":
            break
    result[name] = verdict
json.dump(result, sys.stdout, indent=1)
'''


CONTRACT = '''\
import json
import sys
from importlib import import_module
import pandas as pd
from probe_model import build

compiled = import_module("probe_nomx_cy").mx_model
live = build()
checks = {}


def both(name, t):
    return getattr(live.Calc, name)(t), getattr(compiled.Calc, name)(t)


def equal(pair):
    a, b = pair
    return float(a) == float(b)


old = both("loc_scalar__flat", 1)
checks["before: equal to live"] = equal(old)
frame2 = live.Data.flat().assign(q=live.Data.flat()["q"] * 2)
live.Data.flat[()] = frame2
compiled.Data.flat[()] = frame2.copy()
# assignment without clear_all: the owner's rule is no automatic clearing on input change, so a result
# cached before stays; a result not cached before reads the NEW frame (no prepared copy survives)
checks["reassign, no clear: Data cell returns the assigned frame"] = compiled.Data.flat().equals(frame2)
checks["reassign, no clear: cached result stays"] = float(compiled.Calc.loc_scalar__flat(1)) == float(old[1])
checks["reassign, no clear: uncached result reads the new frame"] = equal(both("loc_scalar__flat", 2))
compiled.Calc.clear_all()
checks["reassign + Space.clear_all: equal to live on the new frame"] = equal(both("loc_scalar__flat", 1))
# Model.clear_all discards assigned inputs on live modelx (the cell recomputes its formula): compiled the same
live.clear_all()
compiled.clear_all()
checks["reassign + Model.clear_all: assignment discarded, equal to live"] = equal(both("loc_scalar__flat", 1)) and \\
    compiled.Data.flat().equals(live.Data.flat())
frame3 = live.Data.flat().assign(q=live.Data.flat()["q"] * 5)
live.Data.flat[()] = frame3
compiled.Data.flat[()] = frame3.copy()
compiled.Calc.clear_all()
live.Calc.clear_all()
checks["second reassign + Space.clear_all: equal to live"] = equal(both("loc_scalar__flat", 3))
# an in-place edit of the bound frame followed by Space.clear_all: as live, the edit is used
live.Data.flat().loc[21, "q"] = 0.5
compiled.Data.flat().loc[21, "q"] = 0.5
compiled.Calc.clear_all()
live.Calc.clear_all()
checks["in-place edit + Space.clear_all: equal to live"] = equal(both("loc_scalar__flat", 1))
json.dump(checks, sys.stdout, indent=1)
'''

# Contract checks that fail today (84befe4), removed step by step
KNOWN_CONTRACT = {'reassign, no clear: uncached result reads the new frame'}


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, *args], cwd=tmp_path, check=True, timeout=2400,
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    work = tmp_path_factory.mktemp("probe")
    (work / "probe_model.py").write_text(MODEL.format(ops=OPS, data=DATA), encoding="utf-8")
    (work / "sample.py").write_text(SAMPLE, encoding="utf-8")
    (work / "checks.py").write_text(CHECKS, encoding="utf-8")
    (work / "contract.py").write_text(CONTRACT, encoding="utf-8")
    (work / "spec.py").write_text(repr({"parallel_workers": 1}), encoding="utf-8")
    _run(work, "probe_model.py")
    build = _run(work, "-m", "modelx_cython", "probe_nomx", "--sample", "sample.py", "--spec", "spec.py")
    verdicts = json.loads(_run(work, "checks.py").stdout)
    return work, build.stdout + build.stderr, verdicts


def test_probe_compiled_equals_live(probe):
    _, _, verdicts = probe
    differing = {name: v for name, v in verdicts.items()
                 if v != "same" and not (v.startswith("boundary: ") and v[len("boundary: "):] in BOUNDARY)}
    unexpected = {name: v for name, v in differing.items() if name not in KNOWN}
    fixed = sorted(name for name in KNOWN if name not in differing)
    assert not unexpected, "compiled differs from live where it did not before:\n" + "\n".join(
        f"{n}: {v}" for n, v in sorted(unexpected.items()))
    assert not fixed, f"now equal to live, remove from KNOWN: {fixed}"


def test_reassign_and_clear(probe):
    work, _, _ = probe
    checks = json.loads(_run(work, "contract.py").stdout)
    failing = {name for name, ok in checks.items() if not ok}
    assert failing == KNOWN_CONTRACT, f"failing: {sorted(failing)}; expected failing: {sorted(KNOWN_CONTRACT)}"
