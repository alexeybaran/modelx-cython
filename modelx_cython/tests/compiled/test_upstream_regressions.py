"""Behaviours a compiled model keeps from live modelx, where upstream's compiled model did not.

One model, built once: a compiled package imports and runs without modelx_cython; a Cell called
outside its declared or sampled range raises an error naming the Cell, the argument and the spec
key to declare; a variable a loop that never ran leaves unbound raises as in Python; a float or
other object arriving where an integer was traced raises or keeps its value, never truncates.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx
import pytest

CELLS = {
    "f": "lambda t: t * 2",
    "g": "lambda t, u: t + u",
    "k": "lambda t: f(t + 5)",
    "kk": "lambda t: g(t, t + 5)",
    "h": "lambda t: str(t)",
    "half": "lambda t: f(t / 2)",
    "rx": "lambda t: t + x",
    "rr": "lambda t: t if t < 10 else rx(t) / 2",
    "nc": "lambda a: a * 2.0",
    "z": "def z(n):\n    for i in range(n):\n        pass\n    return i\n",
    "w": "def w(n):\n    for i in range(n):\n        y = i * 1.5\n    return y\n",
    "zi": "def zi(n):\n    i = -1\n    for i in range(n):\n        pass\n    return i\n",
    # a lambda's and a generator's own parameter named as the formula's typed one
    "lam": "def lam(t):\n    scale = lambda t: t * 2.5\n    return sum(scale(t) for t in range(t)) + t\n",
    # a formula written on one line (upstream compiles it)
    "one": "def one(t): return x * t\n",
}
SAMPLE = """\
from exported import mx_model
S = mx_model.S
for t in range(10):
    S.f(t); S.g(t, t); S.k(t); S.kk(t); S.h(t); S.half(2 * t); S.rx(t); S.rr(t)
S.nc(1.5); S.nc(2.5)
S.z(3); S.w(3); S.zi(3); S.lam(4); S.one(2)
mx_model.P[3].v()
"""


def _outcome(call):
    try:
        return ["value", repr(call())]
    except Exception as error:      # noqa: BLE001 - the type and message are the subject
        return [type(error).__name__, str(error)]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    work = tmp_path_factory.mktemp("upstream-regressions")
    model = mx.new_model("Regressions")
    try:
        space = model.new_space("S", refs={"x": 3})
        for name, formula in CELLS.items():
            space.new_cells(name, formula)
        items = model.new_space("P", formula=lambda i: None)
        items.new_cells("v", "lambda: i * 2")
        live = {"z": _outcome(lambda: space.z(0)), "w": _outcome(lambda: space.w(0)),
                "zi": _outcome(lambda: space.zi(0)), "f": _outcome(lambda: space.f(2.5)),
                "lam": _outcome(lambda: space.lam(7)), "one": _outcome(lambda: space.one(5))}
        model.export(work / "exported")
    finally:
        model.close()
    (work / "sample.py").write_text(SAMPLE)
    # the sample's integers are small, so 'auto' would build unchecked: the overflow checks below need the
    # checked mode (spec key integer_checks)
    (work / "spec.py").write_text(repr({"integer_checks": "checked", "spaces": {"S": {"cells_param_size": {"t": 20}}}}))
    run = subprocess.run([sys.executable, "-B", "-m", "modelx_cython", "exported", "--sample", "sample.py",
                          "--spec", "spec.py"], cwd=work, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout + run.stderr
    return work, live


def _probe(work, body):
    """Run ``body`` in a child that cannot import modelx_cython; it prints one JSON line."""
    script = textwrap.dedent('''\
        import importlib.abc, json, sys
        class Blocked(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "modelx_cython" or name.startswith("modelx_cython."):
                    raise ImportError("modelx_cython is not installed here")
        sys.meta_path.insert(0, Blocked())
        def outcome(call):
            try:
                return ["value", repr(call())]
            except Exception as error:
                return [type(error).__name__, str(error)]
        from exported_cy import mx_model
        S = mx_model.S
    ''') + textwrap.dedent(body)
    run = subprocess.run([sys.executable, "-B", "-c", script], cwd=work, capture_output=True, text=True,
                         timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_compiled_model_runs_without_modelx_cython(built):
    work, _ = built
    out = _probe(work, '''\
        values = [S.f(3), S.k(2), S.nc(1.5), mx_model.P[2].v()]
        print(json.dumps([values, sorted(m for m in sys.modules if m.startswith("modelx_cython"))]))
    ''')
    assert out == [[6, 14, 3.0, 4], []]
    for source in (work / "exported_cy").glob("*.py"):
        assert "modelx_cython" not in source.read_text(encoding="utf-8"), source.name


def test_outside_the_range_names_the_cell_the_argument_and_the_spec_key(built):
    work, _ = built
    out = _probe(work, '''\
        print(json.dumps({label: outcome(call) for label, call in [
            ("S.f t", lambda: S.f(25)), ("S.f t", lambda: S.f(-1)), ("S.g u", lambda: S.g(1, 25)),
            ("S.f t ", lambda: S.k(18)), ("S.g u ", lambda: S.kk(18)), ("S.h t", lambda: S.h(40))]}))
    ''')
    for label, (kind, text) in out.items():
        cell, argument = label.split()
        assert kind == "IndexError", (label, kind, text)
        assert text.startswith(f"{cell}:") and f"{argument} in [" in text and "'cells_param_size'" in text, text


def test_a_variable_a_loop_left_unbound_raises_as_live(built):
    work, live = built
    out = _probe(work, '''\
        print(json.dumps({"z": outcome(lambda: S.z(0)), "w": outcome(lambda: S.w(0)),
                          "zi": outcome(lambda: S.zi(0)), "z3": outcome(lambda: S.z(3))}))
    ''')
    for name in ("z", "w"):
        assert out[name][0] == "UnboundLocalError", out[name]
        assert "UnboundLocalError" in live[name][1], live[name]
    assert out["zi"] == live["zi"] == ["value", "-1"]
    assert out["z3"] == ["value", "2"]


def test_a_float_where_an_integer_was_traced_never_truncates(built):
    work, live = built
    out = _probe(work, '''\
        import numpy as np
        results = {
            "argument": outcome(lambda: S.f(2.5)),
            "numpy float": outcome(lambda: S.f(np.float64(3))),
            "numpy int": outcome(lambda: S.f(np.int64(3))),
            "from a cell": outcome(lambda: S.half(3)),
            "item key": outcome(lambda: mx_model.P[2.5].v()),
            "result": outcome(lambda: S.rr(12)),
        }
        def assign():
            S.x = 2.5
        results["reference"] = outcome(assign)
        results["reference kept"] = outcome(lambda: (S.x, S.rx(1)))
        print(json.dumps(results))
    ''')
    assert live["f"] == ["value", "5.0"]
    for label, cell in (("argument", "S.f"), ("numpy float", "S.f"), ("from a cell", "S.f"),
                        ("item key", "P.__call__"), ("result", "S.rr"), ("reference", "S.x")):
        assert out[label][0] == "TypeError" and out[label][1].startswith(cell), (label, out[label])
    assert out["numpy int"] == ["value", "6"]
    assert out["reference kept"] == ["value", "(3, 4)"]


def test_a_nested_scope_shadowing_a_typed_parameter_builds_and_equals_live(built):
    work, live = built
    out = _probe(work, '''\
        print(json.dumps(outcome(lambda: S.lam(7))))
    ''')
    assert out == live["lam"] == ["value", "59.5"]


def test_a_formula_on_one_line_builds_and_its_arithmetic_is_checked(built):
    work, live = built
    out = _probe(work, '''\
        def big():
            fresh = type(mx_model)().S          # a new model: the first call's value stays cached
            fresh.x = 2 ** 62
            return fresh.one(5)
        print(json.dumps([outcome(lambda: S.one(5)), outcome(big)]))
    ''')
    assert out[0] == live["one"] == ["value", "15"]
    assert out[1][0] == "OverflowError" and out[1][1].startswith("S.one"), out[1]
