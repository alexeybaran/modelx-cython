"""Verify private Collect reuse with rebound object inputs and native output domains."""

import subprocess
import sys
import textwrap

import modelx as mx
import pytest


def test_factory_private_refs(tmp_path):
    model = mx.new_model("PrivateFactory")
    try:
        data = model.new_space("Data", refs={"weights": ((3.0,), (5.0,), (7.0,), (11.0,))})
        data.new_cells("row", "lambda i: weights[i]")
        run = model.new_space(
            "Runs", refs={"data": data, "row": (0.0,), "points": (0, 1, 2, 3)},
            formula="""def factory(case, point):
    if case < 0:
        raise ValueError('negative case')
    return {'refs': {'row': data.row(point)}}
""",
        )
        run.new_cells("value", "lambda t: row[0] + t")
        run.new_cells("total", "lambda: value(case)")
        run.new_cells("steps", "lambda case: range(case)")
        run.set_collect(over={"point": run.points},
                        outputs='lambda: {"sum": {"total": total(), "series": {t: value(t) for t in steps(case)}}}')
        model.export(tmp_path / "exported")
    finally:
        model.close()
    source = (tmp_path / "exported" / "_mx_sys.py").read_text()
    if "_mx_collect_packed_sums_v1 = True" not in source:
        pytest.skip("requires exporter with packed Collect sums")
    options = {"factory_collect_reuse": True, "factory_collect_private_refs": True,
               "factory_collect_subtotals": True, "serial_collect": "reduce"}
    (tmp_path / "spec.py").write_text(repr({"spaces": {"Runs": options}, "cells_param_size": {"t": 8, "case": 8}}))
    (tmp_path / "sample.py").write_text("from exported import mx_model as m\nm.Runs.collect(2)\n")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported"], cwd=tmp_path, check=True)
    probe = textwrap.dedent("""\
        from exported_cy import mx_model as m
        from exported_cy._mx_collect_semantics import NativeCollectedSeries
        options = m.Runs._mx_collect_options_v2()
        assert options[1] and options[2]
        direct = m.Runs[2, 0]
        assert direct.total() == 5.
        for workers, case in ((1, 2), (4, 3)):
            m.collect_workers = workers
            result = m.Runs.collect(case)
            assert result.sum.total == 26. + 4 * case
            assert type(result.sum.series) is NativeCollectedSeries
            assert list(result.sum.series.values()) == [26. + 4 * t for t in range(case)]
        try:
            m.Runs.collect(-1)
        except ValueError as error:
            assert str(error) == 'negative case'
        else:
            raise AssertionError('factory validation bypassed')
        assert m.Runs[2, 0] is direct and direct.total() == 5.
        """)
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True)
