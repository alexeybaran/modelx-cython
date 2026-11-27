"""Direct and Collect calls share a model while keeping separate ItemSpace caches.

The sample calls only Collect. The compiled model must also support a Python call to a
Cell outside the Collect outputs and a model Cell that calls both routes on one Space.
"""

import subprocess
import sys
import textwrap

import modelx as mx
import pytest
from modelx.export import exporter


pytestmark = pytest.mark.skipif(
    not hasattr(mx.core.space.UserSpace, "set_collect"), reason="modelx without collect"
)
EXPORTS_AUTHORED_INPUTS = hasattr(exporter, "ItemInputSnapshot")


@pytest.fixture(scope="module")
def compiled_model(tmp_path_factory):
    folder = tmp_path_factory.mktemp("collect_direct_isolation")
    model = mx.new_model("CollectDirect")
    try:
        data = model.new_space("Data", refs={"events": []})
        data.new_cells("rates", textwrap.dedent('''\
            def rates():
                events.append(("rates",))
                return tuple(k * 0.25 for k in range(1, 5))
            '''))
        projection = model.new_space(
            "Projection", formula="lambda key: None",
            refs={"data": data, "fail_key": -1, "gate": None},
        )
        projection.new_cells("keys", "lambda: (0, 1, 2, 3)")
        projection.new_cells("cashflow", textwrap.dedent('''\
            def cashflow(t):
                data.events.append(("cashflow", key, t))
                if t == 3 and gate is not None:
                    gate.wait(3)
                failure = ("failed", key)
                if key == fail_key and failure not in data.events:
                    data.events.append(failure)
                    raise LookupError(key)
                if t == 0:
                    return 100.0 + key
                return cashflow(t - 1) + t * data.rates()[key]
            '''))
        projection.new_cells("details", "lambda: (key, cashflow(3))")
        projection.new_cells("pv", "lambda: details()[1]")
        projection.new_cells("extra", "lambda t: (key, t, data.rates()[key] * t)")
        projection.set_collect(
            over={"key": projection.keys},
            outputs='lambda: {"individual": {"pv": pv()}, "sum": {"pv": pv()}}',
        )
        portfolio = model.new_space("Portfolio", refs={"projection": projection})
        portfolio.new_cells("mixed", textwrap.dedent('''\
            def mixed(key):
                before = projection[key].details()
                total = projection.collect()
                after = projection[key].details()
                extra = projection[key].extra(11)
                return before, total.individual[key].pv, after, extra
            '''))
        tree = model.new_space("Tree", formula="lambda key: None")
        leaf = tree.new_space("Leaf", refs={"gate": None})
        leaf.new_cells("point", textwrap.dedent('''\
            def point():
                if gate is not None:
                    gate.wait(3)
                return {"key": key}
            '''))
        tree.new_cells("keys", "lambda: (0, 1, 2, 3)")
        tree.set_collect(
            over={"key": tree.keys},
            outputs='lambda: {"individual": {"point": Leaf.point()}}',
        )
        if EXPORTS_AUTHORED_INPUTS:
            inputs = model.new_space(
                "Inputs", formula="lambda key, scen=1: None", refs={"domain": (0, 1)}
            )
            inputs.new_cells("keys", "lambda: domain")
            inputs.new_cells("value", "lambda t: key + scen + t")
            inputs.set_collect(
                over={"key": inputs.keys},
                outputs='lambda: {"individual": {"value": value(0)}}',
            )
            inputs[2, 7].value[0] = 1 << 70
            inputs[2, 7].value[1 << 80] = "authored"
        model.export(folder / "exported", locked_spaces=["Data", "Projection", "Portfolio", "Tree"])
    finally:
        model.close()

    sample = "from exported import mx_model\nmx_model.Projection.collect()\nmx_model.Tree.collect()\n"
    if EXPORTS_AUTHORED_INPUTS:
        sample += "mx_model.Inputs.collect(scen=7)\n"
    (folder / "sample.py").write_text(sample, encoding="utf-8")
    build = subprocess.run(
        [sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py", "--no-spec"],
        cwd=folder, capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    return folder


PROBE = textwrap.dedent('''\
    import gc
    import sys
    import threading
    from importlib.machinery import ExtensionFileLoader

    from exported import mx_model as exported
    from exported_cy import _mx_classes, mx_model as model

    assert isinstance(_mx_classes.__loader__, ExtensionFileLoader)
    projection = model.Projection
    assert type(projection)._mx_collect_hook_version == 1
    expected = exported.Projection.collect()
    key = 2

    def cashflow_calls(key):
        return [event[2] for event in model.Data.events if event[:2] == ("cashflow", key)]

    def check_result(result):
        assert result.sum.pv == expected.sum.pv
        assert {key: result.individual[key].pv for key in range(4)} == {
            key: expected.individual[key].pv for key in range(4)
        }

    def direct_first():
        item = projection[key]
        retained = item.details()
        before = cashflow_calls(key)
        assert len(before) == 4
        rates = model.Data.rates()

        result = projection.collect()
        check_result(result)
        assert cashflow_calls(key) == before + before
        assert projection[key] is item
        assert item._mx_lock is model._mx_lock
        assert item.details() is retained
        assert retained == (key, expected.individual[key].pv)
        assert projection.collect() is result
        assert cashflow_calls(key) == before + before
        assert model.Data.rates() is rates
        assert model.Data.events.count(("rates",)) == 1

    def collect_first():
        check_result(projection.collect())
        before = cashflow_calls(key)
        assert len(before) == 4

        item = projection[key]
        # extra is outside Collect and was never called by the compilation sample.
        assert item.extra(11) == exported.Projection[key].extra(11) == (key, 11, 8.25)
        assert cashflow_calls(key) == before
        retained = item.details()
        assert retained == (key, expected.individual[key].pv)
        assert cashflow_calls(key) == before + before
        assert projection[key] is item and item.details() is retained
        assert cashflow_calls(key) == before + before

    def mixed_cell():
        before, pv, after, extra = model.Portfolio.mixed(key)
        assert (before, pv, after, extra) == exported.Portfolio.mixed(key)
        assert before is after is projection[key].details()
        assert extra == projection[key].extra(11) == (key, 11, 8.25)
        assert len(cashflow_calls(key)) == 8

    def failed_collect():
        # Select the one-time observer failure before either route evaluates a Cell.
        projection.fail_key = 0
        item = projection[key]
        retained = item.details()

        def live_items():
            return sum(type(obj) is type(projection) for obj in gc.get_objects())

        gc.collect()
        gc.disable()
        try:
            before = live_items()
            try:
                projection.collect()
            except LookupError as error:
                assert error.args == (0,)
            else:
                raise AssertionError("collect did not propagate the first error")
            remaining = live_items() - before
        finally:
            gc.enable()

        assert projection[key] is item and item.details() is retained
        failed_calls = cashflow_calls(0)
        assert len(failed_calls) == 1
        result = projection.collect()
        check_result(result)
        assert cashflow_calls(0) == failed_calls + [3, 2, 1, 0]
        assert projection.collect() is result
        assert cashflow_calls(0) == failed_calls + [3, 2, 1, 0]
        assert model.Data.events.count(("failed", 0)) == 1
        assert projection[key] is item and item.details() is retained
        assert remaining == 0, remaining

    def enable_workers():
        # Barrier.wait releases the GIL. This exercises lock ownership even on ordinary Python.
        sys._is_gil_enabled = lambda: False
        model._mx_collect_workers = 4

    def parallel_collect():
        enable_workers()
        projection.gate = threading.Barrier(4)
        check_result(projection.collect())
        assert model.Data.events.count(("rates",)) == 1

    def parallel_generic():
        enable_workers()
        model.Tree.Leaf.gate = threading.Barrier(4)
        got = model.Tree.collect().individual
        assert {key: dict(record.point) for key, record in got.items()} == {
            key: {"key": key} for key in range(4)
        }

    def locked_call():
        enable_workers()
        assert model.Portfolio._mx_lock is model._mx_lock
        mixed_cell()

    def authored_inputs():
        for space in (exported.Inputs, model.Inputs):
            space.domain = (2,)
            direct = space[2, 7]
            assert direct.value(1 << 80) == "authored"
            assert direct.value(0) == space.collect(scen=7).individual[2].value == 1 << 70
            assert space[2, 7] is direct
            assert space[2, 1].value(0) == space.collect(scen=1).individual[2].value == 3

    {
        "direct_first": direct_first,
        "collect_first": collect_first,
        "mixed_cell": mixed_cell,
        "failed_collect": failed_collect,
        "parallel_collect": parallel_collect,
        "parallel_generic": parallel_generic,
        "locked_call": locked_call,
        "authored_inputs": authored_inputs,
    }[sys.argv[1]]()
    ''')


@pytest.mark.parametrize("case", (
    "direct_first", "collect_first", "mixed_cell", "failed_collect",
    "parallel_collect", "parallel_generic", "locked_call",
    pytest.param("authored_inputs", marks=pytest.mark.skipif(
        not EXPORTS_AUTHORED_INPUTS, reason="modelx exporter without authored inputs",
    )),
))
def test_collect_and_direct_calls_do_not_interfere(compiled_model, case):
    result = subprocess.run(
        [sys.executable, "-c", PROBE, case], cwd=compiled_model,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
