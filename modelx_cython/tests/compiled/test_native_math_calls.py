"""Calls of the math module through a Reference compile to the C library behind an identity check.

A Reference that held the standard math module in the sample has its log, exp, sqrt, erf and isinf
calls on C numbers lowered (native_math.py). Values, domain errors, overflow, infinities and NaN equal
live; a Reference rebound to another module after compilation calls that module's member; a user
module held by a Reference named ``math`` is never lowered.
"""
import json
import math
import os
import subprocess
import sys
import textwrap

import pytest

CELLS = {
    "lg": "lambda x: math.log(x)", "ex": "lambda x: math.exp(x)", "sq": "lambda x: math.sqrt(x)",
    "ef": "lambda x: math.erf(x)", "ii": "lambda x: math.isinf(x)",
    "big": "lambda: math.log(huge)",          # an int beyond 64 bits: math's own exact method
    "bs": "def bs(x):\n    d = math.log(x / 2.0) + 0.5 * math.sqrt(x)\n"
          "    return 0.5 * (1.0 + math.erf(float(d) / math.sqrt(2.0))) * math.exp(-x)\n",
}
MODULES = {"usermath.py": "def log(x):\n    return 42.0\n", "oddmath.py": "def log(x):\n    return 'odd'\n"}
PROBE = textwrap.dedent('''\
    import importlib, json, sys
    inf, nan = float("inf"), float("nan")
    def outcome(call):
        try:
            value = call()
            return ["value", "nan" if value != value else repr(value)]
        except Exception as error:
            return [type(error).__name__, str(error)]
    m = importlib.import_module(sys.argv[1]).mx_model
    out = []
    for name in ("lg", "ex", "sq", "ef", "ii"):
        for x in (2.0, 0.0, -1.0, -0.0, inf, -inf, nan, 710.0, -800.0, 5e-324, 3):
            out.append([name, repr(x), outcome(lambda: getattr(type(m)().S, name)(x))])
    out.append(["bs", outcome(lambda: type(m)().S.bs(1.5))])
    out.append(["big", outcome(lambda: type(m)().S.big())])
    out.append(["user module named math", outcome(lambda: type(m)().U.lg(2.0))])
    import usermath, oddmath
    import math
    back = type(m)()
    back.S.math = usermath
    back.S.math = math                 # rebound to another module and back: the C path again
    out.append(["back to math", outcome(lambda: back.S.lg(2.0))])
    for module in (usermath, oddmath):
        rebound = type(m)()
        rebound.S.math = module
        out.append([module.__name__, outcome(lambda: rebound.S.lg(2.0))])
    print(json.dumps(out))
''')


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    work = tmp_path_factory.mktemp("native-math")
    for name, source in MODULES.items():
        (work / name).write_text(source, encoding="utf-8")
    script = textwrap.dedent(f'''\
        import math, modelx as mx, usermath
        m = mx.new_model("NativeMath")
        s = m.new_space("S", refs={{"math": math, "huge": 10 ** 400}})
        for name, formula in {CELLS!r}.items():
            s.new_cells(name, formula)
        m.new_space("U", refs={{"math": usermath}}).new_cells("lg", "lambda x: math.log(x)")
        m.export("exported")
        m.close()
    ''')
    env = dict(os.environ, PYTHONPATH=str(work))
    subprocess.run([sys.executable, "-c", script], cwd=work, env=env, check=True)
    (work / "sample.py").write_text("from exported import mx_model as m\nfor x in (0.5, 1.5, 2.0):\n"
                                    "    m.S.lg(x); m.S.ex(x); m.S.sq(x); m.S.ef(x); m.S.ii(x); m.S.bs(x); m.U.lg(x)\nm.S.big()\n")
    (work / "spec.py").write_text("{}")
    run = subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                          "--spec", "spec.py"], cwd=work, env=env, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    results = {}
    for package in ("exported", "exported_cy"):
        probe = subprocess.run([sys.executable, "-c", PROBE, package], cwd=work, env=env,
                               capture_output=True, text=True, timeout=300)
        assert probe.returncode == 0, probe.stdout + probe.stderr
        results[package] = json.loads(probe.stdout.strip().splitlines()[-1])
    return work, results


def test_calls_are_lowered_only_for_the_standard_math_module(built):
    work, _ = built
    source = (work / "exported_cy" / "_mx_classes.py").read_text(encoding="utf-8")
    for name in ("log", "exp", "sqrt", "erf", "isinf"):
        # the Reference passes as a borrowed pointer (no reference count of the shared module per call)
        assert f"_mx_sys._mx_math_{name}(_mx_cy.cast(_mx_cy.p_void, self.math), self, 'math', x, 'S." in source, name
    assert "_mx_sys._mx_math_log(_mx_cy.cast(_mx_cy.p_void, self.math), self, 'math', self.huge, 'S.big')" in source
    assert "return self.math.log(x)" in source       # U: a user module held by a Reference named math


def test_values_and_errors_equal_live(built):
    _, results = built
    live, compiled = results["exported"], results["exported_cy"]
    assert compiled[:-1] == live[:-1]
    try:
        math.log(0.0)
    except ValueError as error:
        wording = str(error)     # the interpreter's own text, which changed between Python versions
    assert [case for case in compiled if case[0] == "lg" and case[1] == "0.0"] == [
        ["lg", "0.0", ["ValueError", wording]]]
    assert ["ex", "710.0", ["OverflowError", "math range error"]] in compiled
    assert ["big", ["value", repr(math.log(10 ** 400))]] in compiled
    assert compiled[-4:-1] == [["user module named math", ["value", "42.0"]], ["back to math", ["value", "0.6931471805599453"]],
                               ["usermath", ["value", "42.0"]]]


def test_a_rebound_module_returning_another_type_raises_naming_the_cell(built):
    _, results = built
    assert results["exported"][-1] == ["oddmath", ["value", "'odd'"]]
    kind, text = results["exported_cy"][-1][1]
    assert kind == "TypeError" and text.startswith("S.lg: log() of the Reference returned str"), text
