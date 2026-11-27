"""Collect must observe the References and errors from the ordinary factory."""

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
        # All sums are exact even when workers finish in a different order.
        terms = {7: float(2**40), 2: 1.0, 9: -float(2**40)}
        calendar = (5, 1, 3)

        def bind(case, key):
            events.append((case, key))
            if case == -1 and key == 2:
                raise ValueError(f"invalid key {key}")
            return (terms.get(key, float(key))) + case * 4.0

        def months(case, key):
            if key == 2:
                if case == 3:
                    return list(calendar)
                if case == 4:
                    return calendar[::-1]
                if case == 6:
                    return calendar[:-1]
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
        model.export(tmp_path / "exported", locked_spaces=[runs])
    finally:
        model.close()

    (tmp_path / "spec.py").write_text(repr({
        "spaces": {"Runs": {"cells_param_size": {"month": 6}}},
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
        import pytest

        package, workers = sys.argv[1], int(sys.argv[2])
        model = import_module(package).mx_model
        hook = import_module("post_factory_fixture_hook")
        domain = (7, 2, 9, 4, 11, 5, 16, 6, 12, 3, 8, 10)
        months = (5, 1, 3)
        if package == "exported_cy":
            model._mx_collect_workers = workers

        def expected(case, month):
            return reduce(add, ((hook.terms.get(key, float(key)) + case*4.0) + 0.5*month
                                for key in domain))

        def check(case):
            result = model.Runs.collect(case)
            assert sorted(key for seen, key in hook.events if seen == case) == sorted(domain)
            assert result.sum.total == expected(case, 0)
            series = result.sum.by_month
            assert dict(series) == {m: expected(case, m) for m in months}
            assert all(series(m) == series[m] for m in months)
            return result

        initial = check(0)
        before = list(hook.events)
        assert model.Runs.collect(0) is initial
        assert hook.events == before
        with pytest.raises(ValueError, match="^invalid key 2$"):
            model.Runs.collect(-1)
        check(3)  # a factory can return a different iterable with the same month keys
        check(4)  # reversing the keys on one item must preserve values at each key
        with pytest.raises(ValueError, match="different structure"):
            model.Runs.collect(6)
        with pytest.raises(RuntimeError, match="^second month failed$"):
            model.Runs.collect(5)
        assert hook.events.count((5, 2)) == hook.early.count((5, 2)) == 1
        assert [m for case, key, m in hook.monthly if (case, key) == (5, 2)] == [5, 1]
        result = check(2)
        assert model.Runs.collect(0) is initial
        print(json.dumps({"total": result.sum.total, "months": dict(result.sum.by_month)}))
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
