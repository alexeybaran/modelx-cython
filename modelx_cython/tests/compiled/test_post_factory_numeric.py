"""A numeric collect hook must observe the References from the ordinary factory."""

import json
import subprocess
import sys
from textwrap import dedent

import modelx as mx


DOMAIN = (7, 2, 9, 4, 11, 5, 16, 6, 12, 3, 8, 10)
MONTHS = (5, 1, 3)
FACTORY = """def factory(case, key):
    value = Hook.bind(case, key)
    months = Hook.months(case, key)
    return {'refs': {'chosen': value, 'months': months}}
"""


def test_post_factory_numeric_preserves_policy_order_and_errors(tmp_path, monkeypatch):
    hook_path = tmp_path / "post_factory_fixture_hook.py"
    hook_path.write_text(dedent("""
        events = []
        early = []
        monthly = []
        terms = {7: float(2**53), 2: 1.0, 9: -float(2**53)}
        calendar = (5, 1, 3)

        def bind(case, key):
            events.append((case, key))
            if case == -1 and key in (2, 11):
                raise ValueError(f"invalid key {key}")
            return (terms.get(key, float(key))) + case * 4.0

        def months(case, key):
            if key == 2:
                if case == 3:
                    return list(calendar)
                if case == 4:
                    return calendar[::-1]
            return calendar
        """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import post_factory_fixture_hook as hook

    model = mx.new_model("PostFactoryNumeric")
    try:
        runs = model.new_space(
            "Runs", formula=FACTORY,
            refs={"Hook": hook, "chosen": 0.0, "months": MONTHS, "keys": DOMAIN},
            collect_over={"key": "keys"},
            collect_outputs=lambda: {
                'sum': {'total': total(), 'by_month': {m: amount(m) for m in months}}
            },
        )
        runs.new_cells("amount", """def amount(month):
    Hook.monthly.append((case, key, month))
    if case == 5 and key == 2 and month == 1:
        raise RuntimeError('second month failed')
    return chosen + 0.5 * month
""")
        runs.new_cells("total", """def total():
    Hook.early.append((case, key))
    return chosen
""")
        model.export(tmp_path / "exported")
    finally:
        model.close()

    (tmp_path / "spec.py").write_text(repr({
        "spaces": {"Runs": {"serial_collect": "reduce", "cells_param_size": {"month": 6}}},
    }), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\nm.Runs.collect(0)\n", encoding="utf-8",
    )
    build = subprocess.run(
        [sys.executable, "-B", "-m", "modelx_cython", "exported",
         "--sample", "sample.py", "--spec", "spec.py"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr

    probe = dedent("""
        from functools import reduce
        from importlib import import_module
        import json
        from operator import add
        import sys

        package, workers = sys.argv[1], int(sys.argv[2])
        model = import_module(package).mx_model
        hook = import_module("post_factory_fixture_hook")
        domain = (7, 2, 9, 4, 11, 5, 16, 6, 12, 3, 8, 10)
        months = (5, 1, 3)
        model.collect_workers = workers
        packed = []
        plain = []
        merged = []
        if package == "exported_cy":
            assert hasattr(model.Runs, "_mx_collect_numeric_output_v1")
            assert hasattr(model.Runs, "_mx_collect_add_values_v1")
            semantics = import_module("exported_cy._mx_collect_semantics")
            detach = semantics.detached_numeric_sum
            def observe(outputs, live_types):
                values = outputs["by_month"]
                if type(values) is semantics.PackedFloatSeries:
                    packed.append(tuple(values.keys))
                    add_values = values.add_values
                    assert add_values == type(model.Runs)._mx_collect_add_values_v1
                    def merge(total, row):
                        merged.append(len(row))
                        return add_values(total, row)
                    values.add_values = merge
                else:
                    assert type(values) is dict
                    plain.append(tuple(values))
                return detach(outputs, live_types)
            semantics.detached_numeric_sum = observe
        result = model.Runs.collect(0)
        assert sorted(hook.events) == sorted((0, key) for key in domain)
        assert tuple(result.sum.by_month) == months
        if package == "exported_cy":
            assert packed == [months] * len(domain)
            assert merged == [len(months)] * (len(domain) - 1)
            assert not plain
        def expected(case, month):
            return reduce(add, ((hook.terms.get(key, float(key)) + case*4.0) + 0.5*month
                                for key in domain))
        assert result.sum.total == expected(0, 0)
        assert list(result.sum.by_month.items()) == [(m, expected(0, m)) for m in months]
        collected = list(hook.events)
        assert model.Runs.collect(0) is result
        assert hook.events == collected
        try:
            model.Runs.collect(-1)
        except ValueError as error:
            assert str(error) == "invalid key 2"
        else:
            raise AssertionError("factory validation did not run")
        packed.clear()
        plain.clear()
        merged.clear()
        transition = model.Runs.collect(3)
        assert transition.sum.total == expected(3, 0)
        assert list(transition.sum.by_month.items()) == [(m, expected(3, m)) for m in months]
        if package == "exported_cy":
            assert packed == [months] * (len(domain) - 1)
            assert plain == [months] and not merged
        try:
            model.Runs.collect(4)
        except ValueError as error:
            assert "structure changed" in str(error)
        else:
            raise AssertionError("reversed keys were accepted")
        try:
            model.Runs.collect(5)
        except RuntimeError as error:
            assert str(error) == "second month failed"
        else:
            raise AssertionError("a Cell failure was ignored")
        assert hook.events.count((5, 2)) == hook.early.count((5, 2)) == 1
        assert [m for case, key, m in hook.monthly if (case, key) == (5, 2)] == [5, 1]
        hook.events.clear()
        result = model.Runs.collect(2)
        assert sorted(hook.events) == sorted((2, key) for key in domain)
        assert result.sum.total == expected(2, 0)
        assert list(result.sum.by_month.items()) == [(m, expected(2, m)) for m in months]
        print(json.dumps({"total": result.sum.total, "months": list(result.sum.by_month.items())}))
    """)

    results = []
    for workers in (1, 4):
        for package in ("exported", "exported_cy"):
            run = subprocess.run(
                [sys.executable, "-B", "-c", probe, package, str(workers)],
                cwd=tmp_path, capture_output=True, text=True,
            )
            assert run.returncode == 0, run.stdout + run.stderr
            results.append(json.loads(run.stdout))
    assert results.count(results[0]) == len(results)
