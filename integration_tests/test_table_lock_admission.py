"""Report when export locking prevents a compiled cached table read."""

import json
import subprocess
import sys

import modelx as mx
from modelx_helpers.immutable_inputs import Record, Table
import pytest


@pytest.mark.parametrize("locked", [False, True])
def test_cross_space_table_cache_lock_report(tmp_path, locked):
    model = mx.new_model("Market")
    try:
        portfolio = model.new_space("Portfolio")
        rates = Table([1], [(0.05,)], ["rate"])
        source = portfolio.new_space(
            "MarketInputs", refs={"rates": rates, "groups": Record({1: rates})}
        )
        source.new_cells("curve", """\
def curve() -> "modelx_helpers.immutable_inputs.Table":
    return rates
""")
        source.new_cells("raw_curve", "lambda: rates")
        source.new_cells("grouped_curve", "lambda: groups")
        projection = portfolio.new_space("Projection", refs={"market": source})
        projection.new_cells("rate", 'lambda t: market.curve().lookup_float(t, "rate")')
        projection.new_cells("raw_rate", 'lambda t: market.raw_curve().lookup_float(t, "rate")')
        projection.new_cells("table_key", "lambda t: t")
        projection.new_cells("bounded_rate", """\
def bounded_rate(t):
    table = market.curve()
    key = table_key(t)
    low = int(min(table.index))
    high = table.index.max_int()
    if key < low:
        return 0.0
    return table.lookup_float(min(key, high), "rate")
""")
        projection.new_cells("grouped_rate", """\
def grouped_rate(t):
    schedule = table_key(t)
    if schedule not in market.grouped_curve():
        return 0.0
    key = table_key(t)
    table = market.grouped_curve().slice(schedule)
    if key not in table.index:
        return 0.0
    return min(table.lookup_float(key, "rate"), bounded_rate(t))
""")
        model.export(
            tmp_path / "exported", locked_spaces=["Portfolio.MarketInputs"] if locked else []
        )
    finally:
        model.close()

    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\n"
        "projection = mx_model.Portfolio.Projection\n"
        "assert projection.rate(1) == projection.raw_rate(1) == 0.05\n"
        "assert projection.bounded_rate(1) == projection.grouped_rate(1) == 0.05\n",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-m", "modelx_cython", "exported", "--no-spec", "--translate-only"],
        cwd=tmp_path, check=True,
    )
    package = tmp_path / "exported_cy"
    report = json.loads((package / "_mx_admission.json").read_text(encoding="utf-8"))
    prefix = "exported_cy._m_Portfolio._mx_classes._c_Projection._f_"
    for consumer, kind, source, operation in (
        ("rate", "cached", "curve", None),
        ("bounded_rate", "bounded", "curve", "read_bounded_table_float"),
        ("grouped_rate", "grouped", "grouped_curve", "group_slice_context_value"),
    ):
        details = report[prefix + consumer]
        reason = (
            f"native {kind} table read from Portfolio.MarketInputs.{source} requires "
            "Portfolio.MarketInputs in export locked_spaces"
        )
        assert (reason in details["reasons"]) is not locked
        if operation is not None:
            assert any(name.endswith("." + operation)
                       for name in details["native_numeric_helpers"]) is locked
    assert all("locked_spaces" not in reason for reason in report[prefix + "raw_rate"]["reasons"])
    translated = (package / "_m_Portfolio" / "_mx_classes.py").read_text(encoding="utf-8")
    assert ("_mx_load_flag(_mx_cy.address(self.market._has_curve))" in translated) is locked
