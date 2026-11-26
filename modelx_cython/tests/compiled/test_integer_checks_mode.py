"""Spec key ``integer_checks``: ``"auto"`` (default) resolves after the sample run.

The sample's largest integer decides: below 2**31 in magnitude the model builds ``"unchecked"`` (C
integer ``+ - *`` and unary minus are plain C operators), at or above it ``"checked"`` (Cython's
overflowcheck on the formula, an OverflowError naming the Cell). ``"checked"`` and ``"unchecked"`` force a mode.
Both modes keep the boundaries: an argument from Python and a write of an integer Reference must be
exact 64-bit integers, results are converted exactly, and ``//``, ``%``, shifts and integer powers keep
their helpers. An unchecked model wraps, as on upstream, when an intermediate result exceeds 64 bits
on inputs larger than the sample's: ``test_unchecked_model_wraps_past_the_sample`` shows it.
"""
import json
import re
import shutil
import subprocess
import sys
import textwrap

import modelx as mx
import pytest

CELLS = {
    "h": "def h(t):\n    return t\n",
    "c": "def c(t):\n    return h(t) * K * K\n",
    "fd": "def fd(t):\n    return (h(t) - 10) // (h(t) + 1)\n",
    "md": "def md(t):\n    return (h(t) - 10) % (h(t) + 1)\n",
    "sh": "def sh(t):\n    return h(t) << (h(t) * 7)\n",
    "pw": "def pw(t):\n    return h(t) ** 20\n",
    "di": "def di(t):\n    return int(h(t) * 1.0 * K * K)\n",
}
# the sample's integers: K, 3 << 21 (sh(3)) and smaller ones, so the largest is 3 << 21 with K = 3
# and K itself with K = 2**31
SAMPLE = ("from exported import mx_model\nSp = mx_model.Sp\n"
          "Sp.c(0); Sp.di(0); Sp.pw(0); Sp.pw(1)\n"
          "for t in (2, 3):\n    Sp.h(t); Sp.fd(t); Sp.md(t); Sp.sh(t)\n")
WIDE = 4_000_000_000        # K * K = 1.6e19: beyond 64 bits

PROBE = textwrap.dedent(f'''
    import json
    from exported_cy import mx_model
    out = {{}}

    def outcome(name, call):
        try:
            out[name] = ["value", repr(call())]
        except (OverflowError, TypeError) as error:
            out[name] = [type(error).__name__, str(error)]

    built = mx_model.Sp
    outcome("c(1)", lambda: built.c(1))
    outcome("fd(3)", lambda: built.fd(3))
    outcome("md(3)", lambda: built.md(3))
    outcome("sh(9)", lambda: built.sh(9))
    outcome("pw(9)", lambda: built.pw(9))
    outcome("c(1.5)", lambda: built.c(1.5))
    outcome("c(2**64)", lambda: built.c(2 ** 64))
    wide = type(mx_model)().Sp
    outcome("K=0.5", lambda: setattr(type(mx_model)().Sp, "K", 0.5))
    wide.K = {WIDE}
    outcome("wide c(1)", lambda: wide.c(1))
    outcome("wide di(1)", lambda: wide.di(1))
    print(json.dumps(out))
''')
HELPERS = ("_mx_signed_floordiv", "_mx_signed_remainder", "_mx_signed_left_shift", "_mx_signed_pow",
           "_mx_exact_int64", "_mx_integer_result")


def _wrap(value):
    return (value + (1 << 63)) % (1 << 64) - (1 << 63)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """The same Cells exported twice: with K = 3 (every traced integer small) and K = 2**31."""
    work = tmp_path_factory.mktemp("integer-checks")
    live = {}
    for name, k in (("small", 3), ("large", 2 ** 31)):
        model = mx.new_model("IntegerChecks")
        try:
            space = model.new_space("Sp", refs={"K": k})
            for cell, formula in CELLS.items():
                space.new_cells(cell, formula)
            live[name] = {"c(1)": repr(space.c(1)), "fd(3)": repr(space.fd(3)), "md(3)": repr(space.md(3))}
            space.K = WIDE
            live[name]["wide c(1)"] = repr(space.c(1))
            space.K = k
            model.export(work / name)
        finally:
            model.close()
    return work, live


def _build(exported, tmp_path, which, spec, translate_only=False):
    work, _ = exported
    shutil.copytree(work / which, tmp_path / "exported")
    (tmp_path / "sample.py").write_text(SAMPLE)
    (tmp_path / "spec.py").write_text(repr({"spaces": {"Sp": {"cells_param_size": {"t": 16}}}, **spec}))
    run = subprocess.run([sys.executable, "-B", "-m", "modelx_cython", "exported", "--sample", "sample.py",
                          "--spec", "spec.py", *(["--translate-only"] if translate_only else [])],
                         cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout + run.stderr
    package = tmp_path / "exported_cy"
    report = json.loads((package / "_mx_integer_checks.json").read_text())
    return report, (package / "_mx_classes.py").read_text(encoding="utf-8")


def _probe(tmp_path):
    probe = subprocess.run([sys.executable, "-B", "-c", PROBE], cwd=tmp_path, capture_output=True, text=True,
                           timeout=300)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    return json.loads(probe.stdout.strip().splitlines()[-1])


def _assert_checked(source, checked):
    """Checked: Cython's overflowcheck on the formulas, an OverflowError naming the Cell; unchecked:
    neither. The boundaries and the helpers of // % << ** in both modes."""
    assert ("@_mx_cy.overflowcheck(True)" in source) is checked
    assert ("_mx_sys._mx_named_overflow(" in source) is checked
    assert "_mx_flag" not in source and "overflowcheck" not in source.splitlines()[0]
    for helper in HELPERS:
        assert f"_mx_sys.{helper}(" in source, helper


def _assert_boundaries(outcome, live):
    """What both modes share: exact boundaries, exact conversions and Python's // % << **."""
    assert outcome["c(1)"] == ["value", live["c(1)"]]
    assert outcome["fd(3)"] == ["value", live["fd(3)"]] == ["value", "-2"]
    assert outcome["md(3)"] == ["value", live["md(3)"]] == ["value", "1"]
    for name, cell in (("sh(9)", "Sp.sh"), ("pw(9)", "Sp.pw"), ("wide di(1)", "Sp.di"), ("c(2**64)", "Sp.c")):
        kind, text = outcome[name]
        assert kind == "OverflowError" and text.startswith(cell), (name, outcome[name])
    for name, cell in (("c(1.5)", "Sp.c"), ("K=0.5", "Sp.K")):
        kind, text = outcome[name]
        assert kind == "TypeError" and cell in text, (name, outcome[name])


def test_sample_at_or_above_the_limit_builds_checked(exported, tmp_path):
    """K = 2**31 is traced: the build is checked, and an overflow raises naming the Cell."""
    report, source = _build(exported, tmp_path, "large", {})
    assert report["requested"] == "auto" and report["mode"] == "checked", report
    assert report["largest_traced_integer"] == 2 ** 31 and report["seen_at"] == "Sp.K", report
    assert report["unchecked_below"] == 2 ** 31 and "limit" not in report, report
    _assert_checked(source, True)
    outcome = _probe(tmp_path)
    _assert_boundaries(outcome, exported[1]["large"])
    kind, text = outcome["wide c(1)"]
    assert kind == "OverflowError" and text.startswith("Sp.c"), outcome["wide c(1)"]


def test_sample_below_the_limit_builds_unchecked(exported, tmp_path):
    """Every traced integer is below 2**31: the build is unchecked; the boundaries stay."""
    report, source = _build(exported, tmp_path, "small", {})
    assert report["requested"] == "auto" and report["mode"] == "unchecked", report
    assert report["largest_traced_integer"] == 3 << 21 and report["overflowcheck"] == [], report
    assert "wrap" in report["limit"] and '"integer_checks": "checked"' in report["limit"], report
    _assert_checked(source, False)
    _assert_boundaries(_probe(tmp_path), exported[1]["small"])


def test_unchecked_model_wraps_past_the_sample(exported, tmp_path):
    """The documented limit of the unchecked mode: the sample traced only small integers, so
    ``h(t) * K * K`` is a plain C product; with K = 4e9 set after the build its exact value,
    1.6e19, leaves 64 bits and the compiled Cell returns it wrapped (two's complement), as an
    upstream build does, where live modelx returns the exact int. A checked build raises instead
    (``test_sample_at_or_above_the_limit_builds_checked``; ``"integer_checks": "checked"``)."""
    _build(exported, tmp_path, "small", {})
    outcome, live = _probe(tmp_path), exported[1]["small"]
    assert live["wide c(1)"] == repr(WIDE * WIDE)
    assert outcome["wide c(1)"] == ["value", repr(_wrap(WIDE * WIDE))], outcome["wide c(1)"]
    assert outcome["wide c(1)"] != ["value", live["wide c(1)"]]


@pytest.mark.parametrize("which, forced", [("small", "checked"), ("large", "unchecked")])
def test_forced_modes_ignore_the_sample(exported, tmp_path, which, forced):
    report, source = _build(exported, tmp_path, which, {"integer_checks": forced}, translate_only=True)
    assert report["requested"] == report["mode"] == forced, report
    assert report["largest_traced_integer"] == (3 << 21 if which == "small" else 2 ** 31), report
    assert ("limit" in report) is (forced == "unchecked"), report
    _assert_checked(source, forced == "checked")
