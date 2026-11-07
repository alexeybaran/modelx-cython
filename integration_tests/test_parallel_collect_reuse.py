import subprocess
import sys
import textwrap

import modelx as mx


def test_parallel_collect_reuse_keeps_order_isolation_and_retry(tmp_path):
    model = mx.new_model("PrivateCollect")
    try:
        values = (1e16, 1.0, -1e16, 1.0, 2.0, 3.0, 4.0, -0.0)
        keys = tuple(range(len(values)))
        for name in ("Reused", "Ordinary"):
            space = model.new_space(
                name, formula="lambda key: None", refs={"keys": keys, "values": values, "adjust": float},
                collect_over={"key": keys}, collect_outputs='lambda: {"sum": {"value": value()}}',
            )
            space.new_cells("part", "lambda t: values[key] + t")
            space.new_cells("value", "lambda: adjust(part(0))")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    spec = {"parallel_workers": 4, "spaces": {
        "Reused": {"serial_collect": "reuse", "parallel_collect_reuse": True},
    }}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Reused.collect(); mx_model.Ordinary.collect()\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--spec", "spec.py"], cwd=tmp_path, check=True)
    generated = (tmp_path / "exported_cy/_mx_classes.py").read_text(encoding="utf-8")
    assert "reuse_in_parallel = True" in generated or "reuse_in_parallel=True" in generated
    probe = textwrap.dedent("""\
        from concurrent import futures
        import sys
        from threading import Barrier, Lock, get_ident

        gil = getattr(sys, "_is_gil_enabled", lambda: True)
        before = gil()
        from exported_cy import mx_model as m
        assert gil() == before

        pool = futures.ThreadPoolExecutor
        starts = Barrier(4, timeout=15)
        lock = Lock()
        threads = set()
        class Observer(pool):
            def submit(self, fn, /, *args, **kwargs):
                assert args[-1] is not None
                def run():
                    with lock:
                        threads.add(get_ident())
                    starts.wait()
                    return fn(*args, **kwargs)
                return super().submit(run)

        ordinary = m.Ordinary.collect().sum.value
        futures.ThreadPoolExecutor = Observer
        failure = LookupError("retry")
        def broken(x):
            if x == -1e16:
                raise failure
            return x
        m.Reused.adjust = broken
        try:
            m.Reused.collect()
        except LookupError as error:
            assert error is failure
        else:
            raise AssertionError("failure swallowed")
        m.Reused.adjust = float
        result = m.Reused.collect()
        assert result.sum.value == ordinary
        assert m.Reused.collect() is result
        assert len(threads) >= 4
        try:
            del m.Reused[3]
        except KeyError:
            pass
        else:
            raise AssertionError("collect left a public ItemSpace")
        assert m.Reused[3].value() == 1.0
        assert gil() == before
    """)
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True, timeout=90)
