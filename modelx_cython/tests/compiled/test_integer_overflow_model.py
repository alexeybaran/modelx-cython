"""The integer overflow model: C integer arithmetic only where it cannot overflow.

Each Cell computes an integer whose exact (Python) value leaves 64 bits on the way, or depends on
Python's integer semantics. The sample keeps every value small, as a real book does, so the Cells
are typed as C integers; the run then calls them with the values below. Every result equals live
modelx, or the call raises an error naming the Cell, and it raises only where live's own result
does not fit the Cell's C integer type: nothing wraps, whatever the sample reached.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx
import pytest

CELLS = {
    "h": "def h(t):\n    return t\n",
    "c": "def c(t):\n    return h(t) * h(t) * h(t)\n",
    "neg": "def neg(t):\n    x = t * BIG - BIG - BIG\n    return -x\n",
    "pw": "def pw(t):\n    return h(t) ** 20\n",
    "sh": "def sh(t):\n    return h(t) << 60\n",
    "rs": "def rs(t):\n    return (h(t) - 10) >> 1\n",
    "fd": "def fd(t):\n    return (h(t) - 10) // 4\n",
    "md": "def md(t):\n    return (h(t) - 10) % 4\n",
    "fm": "def fm(t):\n    x = t * BIG - BIG - BIG\n    return x // (h(t) - 1)\n",
    "il": "def il(t):\n    x = h(t)\n    return x * x * x * x // (x * x)\n",
    "ie": "def ie(t):\n    return h(t) * h(t) * h(t) * h(t) // (h(t) * h(t))\n",
    "ls": "def ls(t):\n    s = 0\n    for i in range(4):\n        s += c(t)\n    return s\n",
    "sm": "def sm(t):\n    return sum(c(k) for k in range(t, t + 4))\n",
    "di": "def di(t):\n    return int(h(t) * 1e18)\n",
    "mx": "def mx(t):\n    return max(c(t), c(t) + c(t))\n",
    "pl": "def pl(t):\n    x = h(t)\n    return x ** 20\n",
    "pf": "def pf(t):\n    return float(h(t) ** 20)\n",
    "pv": "def pv(t):\n    return h(t) ** h(t)\n",
    "pc": "def pc(t):\n    return h(t) ** 20 > 10\n",
    "dl": "def dl(t):\n    x = int(h(t) * 1e18)\n    return x // 1000\n",
    "lf": "def lf(t):\n    return h(t) + 4611686018427387904 * 4\n",
    "lg": "def lg(t):\n    return h(t) + (4611686018427387904 * 4) // 8\n",
    # LLONG_MIN // -1 and LLONG_MIN % -1 trap in C
    "ff": "def ff(t):\n    x = t * BIG - BIG - BIG\n    return float(x // (h(t) - 1))\n",
    "mf": "def mf(t):\n    x = t * BIG - BIG - BIG\n    return float(x % (h(t) - 1))\n",
    "fn": "def fn(t):\n    x = t * BIG - BIG - BIG\n    return x // -1\n",
    "fa": "def fa(t):\n    x = t * BIG - BIG - BIG\n    x //= h(t) - 1\n    return x\n",
    "ma": "def ma(t):\n    x = t * BIG - BIG - BIG\n    x %= h(t) - 1\n    return x\n",
    "rm": "def rm(t):\n    return (t * BIG - BIG - BIG) % (h(t) - 1)\n",
    # the two reproductions of the sample-chosen unchecked mode: a local grown in a loop, and a
    # value of another Cell multiplied past 64 bits and divided back
    "gr": "def gr(t):\n    x = 1\n    for i in range(t):\n        x = x * 1000\n    return x % 7\n",
    "bs": "def bs():\n    return B\n",
    "kk": "def kk():\n    k = bs()\n    return (k * k * k) // (k * k)\n",
    # a counter and a parameter of a checked domain stay C integers; their sums are exact
    "ct": "def ct(t):\n    total = 0\n    for i in range(t):\n        total = total + (i + t - 1) % 5\n    return total\n",
}
RUNS = {"neg": 0, "pw": 9, "sh": 9, "rs": 3, "fd": 3, "md": 3, "fm": 0, "il": 100000, "ie": 100000,
        "ls": 2097151, "sm": 2097148, "di": 100, "mx": 2097151, "pl": 9, "pf": 9, "pv": 16, "pc": 9,
        "dl": 100, "lf": 1, "lg": 1, "ff": 0, "mf": 0, "fn": 0, "fa": 0, "ma": 0, "rm": 0, "gr": 7,
        "kk": None, "ct": 1000}
INT64 = range(-2 ** 63, 2 ** 63)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    work = tmp_path_factory.mktemp("overflow-model")
    model = mx.new_model("OverflowModel")
    try:
        space = model.new_space("Sp", refs={"BIG": 2 ** 62, "B": 3_000_000})
        for name, formula in CELLS.items():
            space.new_cells(name, formula)
        live = {name: _call(getattr(space, name), t) for name, t in RUNS.items()}
        model.export(work / "exported")
    finally:
        model.close()
    sample = "".join(f"Sp.{n}(2); Sp.{n}(3)\n" for n in CELLS if n not in ("bs", "kk"))
    (work / "sample.py").write_text("from exported import mx_model\nSp = mx_model.Sp\n" + sample
                                    + "Sp.h(0)\nSp.kk()\n")
    (work / "spec.py").write_text(repr({"spaces": {"Sp": {"cells_param_size": {"t": 2097156}}}}))
    run = subprocess.run([sys.executable, "-B", "-m", "modelx_cython", "exported", "--sample", "sample.py",
                          "--spec", "spec.py"], cwd=work, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout + run.stderr
    probe = subprocess.run([sys.executable, "-B", "-c", textwrap.dedent(f'''
        import json
        from exported_cy import mx_model
        out = {{}}
        for name, t in {RUNS!r}.items():
            cell = getattr(mx_model.Sp, name)
            try:
                value = cell() if t is None else cell(t)
                out[name] = ["value", repr(value)]
            except (OverflowError, TypeError) as error:
                out[name] = [type(error).__name__, str(error)]
        print(json.dumps(out))
    ''')], cwd=work, capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    return json.loads(probe.stdout.strip().splitlines()[-1]), live, work


def _call(cell, t):
    value = cell() if t is None else cell(t)
    return value, repr(value)


def test_every_result_is_exact_or_raises_naming_the_cell(built):
    outcome, live, _ = built
    raised = set()
    for name in RUNS:
        value, text = live[name]
        if outcome[name][0] != "value":
            # only where live's result has no C integer: the result's conversion raises, naming the Cell
            assert type(value) is int and value not in INT64, (name, outcome[name], text)
            assert outcome[name][0] == "OverflowError" and outcome[name][1].startswith(f"Sp.{name}"), (
                name, outcome[name])
            raised.add(name)
        else:
            assert outcome[name] == ["value", text], (name, outcome[name], text)
    # the Cells typed C integers whose live result leaves 64 bits
    assert raised == {"neg", "pw", "sh", "fm", "ls", "sm", "di", "mx", "pl", "pv", "fn", "fa"}


def test_the_reproductions_of_the_unchecked_mode(built):
    outcome, live, _ = built
    assert live["gr"][0] == 6 and outcome["gr"] == ["value", "6"]
    assert live["kk"][0] == 3_000_000 and outcome["kk"] == ["value", "3000000"]


def test_no_integer_mode_and_no_report(built):
    _, _, work = built
    source = (work / "exported_cy" / "_mx_classes.py").read_text(encoding="utf-8")
    assert "overflowcheck" not in source and "infer_types" not in source
    assert not (work / "exported_cy" / "_mx_integer_checks.json").exists()
