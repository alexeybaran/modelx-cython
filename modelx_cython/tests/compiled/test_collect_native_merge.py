"""Compare native and Python Collect results, errors and key order with the export."""
import importlib
import json
import math
import subprocess
import sys
import textwrap

import modelx as mx


def test_native_and_python_outputs_equal_the_export(tmp_path, monkeypatch):
    (tmp_path / "collect_arithmetic.py").write_text(textwrap.dedent("""\
        class LeftFold:
            def __init__(self, value):
                self.value = value
            def __add__(self, right):
                return LeftFold(f"({self.value}+{right.value})")
            def __repr__(self):
                return self.value

        def keyword_echo(**kwargs):
            return kwargs["self"]
        """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    arithmetic = importlib.import_module("collect_arithmetic")
    model = mx.new_model("NativeMerge")
    try:
        keys = (3, 1, 4, 5, 9, 2)
        months = (0, 1, 2, 5, 3)            # not contiguous, not sorted: positions in D, not values
        empty = ()
        common = {"keys": keys, "months": months, "empty": empty, "contiguous": (0, 1, 2),
                  "LeftFold": arithmetic.LeftFold,
                  "keyword_echo": arithmetic.keyword_echo}
        outputs = {
            "Mixed": lambda: {
                "individual": {"state": {t: cf(t) for t in months}, "metadata": metadata()},
                "sum": {"pv": pv(), "cf": {t: cf(t) for t in months},
                        "native_cf": {t: cf(t) for t in contiguous}, "none": {t: cf(t) for t in empty}},
            },
            "MixedPython": lambda: {
                "individual": {"keys": [key]},
                "sum": {"pv": pv(), "native_cf": {t: cf(t) for t in contiguous},
                        "metadata": {"fold": LeftFold(str(key))}},
            },
            "RootLocal": lambda: {"individual": {"state": {root: cf(root) for root in months}}, "sum": {"pv": pv()}},
            "RootUnused": lambda: {"individual": {"state": [cf(0) for root in months]}, "sum": {"pv": pv()}},
            "Keyword": lambda: {"individual": {"state": keyword_echo(self=3)}, "sum": {"pv": pv()}},
            "Duplicate": lambda: {"sum": {"pv": pv(), "pv": 1}},
            "CellDomain": lambda: {"sum": {"cf": {t: cf(t) for t in domain()}}},
            "MissingKey": lambda: {"sum": {"pv": pv(), "cf": {t: cf(t) for t in range(size())}}},
        }
        for name, collect_outputs in outputs.items():
            space = model.new_space(name, formula="lambda key: None", refs=dict(common),
                                    collect_over={"key": "keys"}, collect_outputs=collect_outputs)
            space.new_cells("cf", "lambda t: key / 3.0 + t * 0.1 + (1e16 if key == 4 and t == 0 else 0.0)")
            space.new_cells("pv", "lambda: sum(cf(t) for t in months) / 7.0")
            space.new_cells("domain", "lambda: months")
            space.new_cells("size", "lambda: 2 if key == 5 else 3")
            if name == "Mixed":
                space.new_cells("metadata", "lambda: {'label': LeftFold(str(key))}")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(textwrap.dedent("""\
        from exported import mx_model as m
        m.Mixed.collect(); m.CellDomain.collect()
        m.MixedPython.collect(); m.RootLocal.collect(); m.RootUnused.collect(); m.Keyword.collect(); m.Duplicate.collect()
        for key in m.MissingKey.keys:
            m.MissingKey[key].pv(); m.MissingKey[key].cf(0); m.MissingKey[key].size()
        """), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                    "--no-spec"], cwd=tmp_path, check=True)
    probe = textwrap.dedent("""\
        import json, pickle, sys
        from collections.abc import Mapping
        from collect_arithmetic import LeftFold
        package = sys.argv[1]
        Model = type(__import__(package).mx_model)
        def plain(value):
            if hasattr(value, '_fields'):
                return {k: plain(v) for k, v in value._fields.items()}
            if isinstance(value, Mapping):
                return [[repr(k), plain(v)] for k, v in value.items()]
            return repr(value)
        out = {}
        for workers in (1, 4):
            m = Model()
            m._mx_collect_workers = workers
            result = m.Mixed.collect()
            python_result = m.MixedPython.collect()
            if package == 'exported_cy':
                from exported_cy._mx_collect import CollectedSeries, _NativeFields
                assert type(result.sum._CollectResult__storage) is _NativeFields
                series = result.sum.native_cf
                assert type(series) is CollectedSeries
                assert series(1) == series[1] and list(series) == [0, 1, 2]
                assert dict(pickle.loads(pickle.dumps(series))) == dict(series)
                for key, item in result.individual.items():
                    assert type(item.metadata) is CollectedSeries
                    assert type(item.metadata['label']) is LeftFold
                    assert item.metadata['label'].value == str(key)

                assert type(python_result.sum.metadata) is CollectedSeries
                assert type(python_result.sum.metadata['fold']) is LeftFold
                assert type(python_result.sum.native_cf) is CollectedSeries
                assert type(python_result.sum._CollectResult__storage) is dict
            assert [item.keys for item in python_result.individual.values()] == [[3], [1], [4], [5], [9], [2]]
            out[f'mixed{workers}'] = plain(result)
            out[f'python{workers}'] = plain(python_result)
            out[f'root{workers}'] = plain(m.RootLocal.collect())
            out[f'unused{workers}'] = plain(m.RootUnused.collect())
            out[f'keyword{workers}'] = plain(m.Keyword.collect())
            out[f'duplicate{workers}'] = plain(m.Duplicate.collect())
            out[f'cell{workers}'] = plain(m.CellDomain.collect())
            try:
                m.MissingKey.collect()
                out[f'missing{workers}'] = 'no error'
            except Exception as error:
                out[f'missing{workers}'] = [type(error).__name__, str(error)]
        print(json.dumps(out))
        """)
    results = {}
    for package in ("exported", "exported_cy"):
        done = subprocess.run([sys.executable, "-c", probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        results[package] = json.loads(done.stdout)
    live, compiled = results["exported"], results["exported_cy"]
    for key in ("mixed1", "cell1", "missing1", "python1", "python4", "root1", "root4",
                "unused1", "unused4", "keyword1", "keyword4", "duplicate1", "duplicate4"):
        assert compiled[key] == live[key], key
    assert compiled["python4"]["sum"]["metadata"] == [["'fold'", "(((((3+1)+4)+5)+9)+2)"]]
    assert compiled["duplicate4"]["sum"]["pv"] == "6"
    # 4 workers: the individual section in domain order and with the serial values
    assert compiled["mixed4"]["individual"] == live["mixed1"]["individual"]
    assert compiled["missing4"] == live["missing4"] == live["missing1"]
    assert compiled["missing1"][0] == "ValueError" and "different structure" in compiled["missing1"][1]
    mixed = compiled["mixed1"]
    assert [entry[0] for entry in mixed["sum"]["cf"]] == ["0", "1", "2", "5", "3"]
    assert mixed["sum"]["none"] == []
    parallel_sum = compiled["mixed4"]["sum"]
    assert parallel_sum["none"] == []
    assert math.isclose(float(parallel_sum["pv"]), float(mixed["sum"]["pv"]), rel_tol=1e-12)
    for field in ("cf", "native_cf"):
        assert [key for key, _ in parallel_sum[field]] == [key for key, _ in mixed["sum"][field]]
        for (_, got), (_, want) in zip(parallel_sum[field], mixed["sum"][field]):
            assert math.isclose(float(got), float(want), rel_tol=1e-12)


def test_native_collect_result_supports_native_and_python_consumers(tmp_path):
    model = mx.new_model("CollectConsumers")
    try:
        keys = (1, 2, 3, 4)
        periods = (0, 1, 2)
        projection = model.new_space(
            "Projection", formula="lambda key: None", refs={"keys": keys, "periods": periods},
            collect_over={"key": "keys"},
            collect_outputs=lambda: {"sum": {"cash": {t: cash(t) for t in periods}}})
        projection.new_cells("cash", "lambda t: key * 0.5 + t")
        consumers = model.new_space("Consumers", refs={"projection": projection, "numbers": math})
        consumers.new_cells(
            "numeric", "lambda t: numbers.sqrt(float(projection.collect().sum.cash(t)))")
        consumers.new_cells(
            "report", 'lambda: {t: f"{value:.2f}" for t, value in projection.collect().sum.cash.items()}')
        model.export(tmp_path / "exported")
    finally:
        model.close()

    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\n"
        "m.Consumers.report()\n"
        "for t in m.Projection.periods:\n    m.Consumers.numeric(t)\n",
        encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                    "--no-spec"], cwd=tmp_path, check=True)

    admission = json.loads((tmp_path / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))
    numeric = admission["exported_cy._mx_classes._c_Consumers._f_numeric"]
    report = admission["exported_cy._mx_classes._c_Consumers._f_report"]
    assert numeric["return_type"] == "double"
    assert any(helper.endswith(".m_sqrt") for helper in numeric["native_helpers"])
    assert report["return_type"] == "object"

    probe = textwrap.dedent("""\
        import math
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled
        from exported_cy._mx_collect import CollectedSeries, _NativeFields

        for workers in (1, 4):
            live, native = type(exported)(), type(compiled)()
            native._mx_collect_workers = workers
            collected = native.Projection.collect()
            series = collected.sum.cash
            assert type(collected.sum._CollectResult__storage) is _NativeFields
            assert type(series) is CollectedSeries
            assert dict(series) == dict(live.Projection.collect().sum.cash) == {0: 5., 1: 9., 2: 13.}
            for t in (0, 1, 2):
                expected = math.sqrt(series[t])
                assert native.Consumers.numeric(t) == live.Consumers.numeric(t) == expected
            report = native.Consumers.report()
            assert type(report) is dict
            assert report == live.Consumers.report() == {0: "5.00", 1: "9.00", 2: "13.00"}
            assert native.Projection.collect() is collected
            assert collected.sum.cash is series
        """)
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True)


def test_collect_with_other_parameters_is_native_per_argument_set(tmp_path):
    """Valuation(a, key, b).collect(a, b): the other parameters reach the compiled evaluator (protocol 2
    extension); each call has its own accumulators (allocated and freed in that call), and the collect
    cache is keyed by the arguments, so two argument sets never share a total. One item and many argument
    sets is the per-call cost case."""
    model = mx.new_model("CollectArguments")
    try:
        book = tuple(range(1, 9))
        for name, keys in (("Book", book), ("One", (7,)), ("Reduce", book)):
            months = tuple(range(12))
            space = model.new_space(name, formula="lambda a, key, b=-1: None",
                                    refs={"keys": keys, "months": months}, collect_over={"key": "keys"},
                                    collect_outputs=lambda: {"sum": {"pv": pv(), "cf": {t: cf(t) for t in months}}})
            space.new_cells("cf", "lambda t: key * 0.37 + a / 7.0 + b * 0.011 + t / 3.0")
            space.new_cells("pv", "lambda: sum(cf(t) for t in months) / (1.0 + a / 100.0)")
            if name == "Reduce":
                space.new_cells("need", "lambda: a + key + b")
        space = model.new_space(
            "ArgumentDomain", formula="lambda key, periods: None",
            refs={"keys": book}, collect_over={"key": "keys"},
            collect_outputs=lambda: {"sum": {"pv": pv(), "cf": {t: cf(t) for t in periods}}})
        space.new_cells("pv", "lambda: key * 0.25")
        space.new_cells("cf", "lambda t: pv() + t")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    spec = {"spaces": {name: {"serial_collect": "reuse", "parallel_collect_reuse": True} for name in ("Book", "One")}}
    spec["spaces"]["Reduce"] = {"serial_collect": "reduce",
                                "cells_param_size": {"t": 12}, "param_bound_checks": {"t": "need"}}
    spec["spaces"]["ArgumentDomain"] = {"serial_collect": "reuse", "parallel_collect_reuse": True}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\n"
        "for a in (0, 12, 24):\n"
        "    for b in (-1, 0, 6):\n"
        "        m.Book.collect(a, b); m.One.collect(a, b); m.Reduce.collect(a, b)\n"
        "        for k in m.Reduce.keys:\n"
        "            m.Reduce[a, k, b].need()\n"
        "m.ArgumentDomain.collect((2, 0, 1))\n",
        encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                    "--spec", "spec.py"], cwd=tmp_path, check=True)
    probe = textwrap.dedent("""\
        import json, sys
        Model = type(__import__(sys.argv[1]).mx_model)
        def plain(r):
            return [repr(r.sum.pv), [[k, repr(v)] for k, v in r.sum.cf.items()]]
        out = {}
        for workers in (1, 4):
            m = Model()
            m.collect_workers = workers
            out[f'book{workers}'] = [plain(m.Book.collect(a, b)) for a in (0, 12, 24) for b in (-1, 0, 6)]
            out[f'one{workers}'] = [plain(m.One.collect(a, b)) for a in range(0, 300, 12) for b in range(-1, 7)]
            out[f'argument_domain{workers}'] = plain(m.ArgumentDomain.collect((1, 2)))
            if workers == 1:
                out['reduce'] = [plain(m.Reduce.collect(a, b)) for a in (0, 12, 24) for b in (-1, 0, 6)]
        out['native'] = [hasattr(getattr(m, n), '_mx_collect_options_v2') and getattr(m, n)._mx_collect_options_v2()[1] is not None
                         for n in ('Book', 'One', 'Reduce', 'ArgumentDomain')]
        if out['native'][2]:
            try:
                Model().Reduce.collect(25, 6)
            except IndexError as error:
                out['reduce_bound'] = str(error)
        print(json.dumps(out))
        """)
    results = {}
    for package in ("exported", "exported_cy"):
        done = subprocess.run([sys.executable, "-c", probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        results[package] = json.loads(done.stdout)
    live, compiled = results["exported"], results["exported_cy"]
    assert compiled["native"] == [True, True, True, True]
    assert compiled["book1"] == live["book1"] and compiled["one1"] == live["one1"] == compiled["one4"]
    assert compiled["reduce"] == live["reduce"]
    assert "policy 8 needs t for need() = 39" in compiled["reduce_bound"]
    for workers in (1, 4):
        assert compiled[f"argument_domain{workers}"] == live[f"argument_domain{workers}"]
    report = json.loads((tmp_path / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))
    leaves = report["exported_cy._mx_classes._c_ArgumentDomain._f_collect_outputs"]["collect_outputs"]
    assert {leaf["output"]: leaf["native"] for leaf in leaves} == {"pv": True, "cf": False}
    assert len({json.dumps(r) for r in compiled["book1"]}) == 9            # nine argument sets, nine totals
    # 4 workers: chunk totals merge in the parallel order, so the last bits may differ from the serial sum
    for got, want in zip(compiled["book4"], live["book1"]):
        assert math.isclose(float(got[0]), float(want[0]), rel_tol=1e-12)
        assert [k for k, _ in got[1]] == [k for k, _ in want[1]]
        for (_, x), (_, y) in zip(got[1], want[1]):
            assert abs(float(x) - float(y)) <= 1e-12 * abs(float(y))
