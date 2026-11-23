"""Exercise native locked calls and per-item numeric Collect domains."""
import subprocess
import sys
import textwrap

import modelx as mx
import numpy as np


def test_locked_native_calls_and_dependent_collect_domain(tmp_path):
    model = mx.new_model("LockedDomains")
    try:
        market = model.new_space("Market", refs={"np": np})
        market.new_cells("level", "lambda: 10.0")
        market.new_cells("price", """def price(t):
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)):
        raise TypeError("expected integer")
    return level() + t
""")
        market.new_cells("rate", "def rate(t):\n    return price(t) / price(t + 1)\n")
        book = model.new_space(
            "Book", formula="lambda close, key: None", refs={"keys": (1, 2, 3, 4), "market": market},
            collect_over={"key": (1, 2, 3, 4)},
            collect_outputs='lambda: {"sum": {"pv": pv(), "cash": {t: cash(t) for t in periods(close)}}}')
        book.new_cells("periods", "lambda month: range(4 - month)")
        book.new_cells("cash", "lambda t: market.price(t) + key")
        book.new_cells("pv", "lambda: key * 2.0")
        model.export(tmp_path / "exported", locked_spaces=["Market"])
    finally:
        model.close()
    spec = {"spaces": {"Market": {"cells_param_size": {"t": 10}},
                       "Book": {"cells_param_size": {"t": 5, "month": 5}}}}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\n"
        "m.Market.rate(1); m.Market.price(2)\n"
        "m.Book.collect(0); m.Book.collect(1); m.Book.collect(4)\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                    "--spec", "spec.py"], cwd=tmp_path, check=True)
    source = (tmp_path / "exported_cy/_mx_classes.py").read_text(encoding="utf-8")
    assert "_mx_held_formula__c_Market_rate" in source
    assert "_mx_held_cache__c_Market_price" in source
    assert "domain = self.periods(self.close)" in source
    probe = textwrap.dedent("""\
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from exported import mx_model as live
        from exported_cy import mx_model as native

        fresh = type(native)()
        fresh.Market.np = SimpleNamespace()
        try:
            fresh.Market.price(3)
        except AttributeError as error:
            assert "integer" in str(error)
        else:
            raise AssertionError("the eager tuple member lookup must still happen")

        with ThreadPoolExecutor(max_workers=4) as pool:
            rates = list(pool.map(native.Market.rate, (1, 2, 3, 4) * 10))
        assert rates == [live.Market.rate(t) for t in (1, 2, 3, 4) * 10]
        for workers in (1, 4):
            model = type(native)()
            model.collect_workers = workers
            for close in (0, 1, 4):
                expected = live.Book.collect(close).sum
                result = model.Book.collect(close).sum
                assert result.pv == expected.pv
                assert dict(result.cash) == dict(expected.cash)
        """)
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True)
