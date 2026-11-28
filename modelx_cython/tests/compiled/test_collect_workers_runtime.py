"""The number of Collect workers is a runtime setting (``model._mx_collect_workers``): a model compiled
with the default (one worker) and set to 4 afterwards gives the same values and, on a free-threaded
build, evaluates its items on several threads at once."""
import subprocess
import sys
import textwrap

import modelx as mx
import pytest


@pytest.mark.skipif(getattr(sys, "_is_gil_enabled", lambda: True)(), reason="parallel collect needs a free-threaded build")
def test_compiled_with_one_worker_runs_four_after_compilation(tmp_path):
    model = mx.new_model("WorkersAfterBuild")
    try:
        keys = tuple(range(8))
        space = model.new_space(
            "Policies", formula="lambda key: None", refs={"keys": keys, "probe": float})
        space.set_collect(over={"key": "keys"}, outputs=lambda: {"sum": {"value": value()}})
        space.new_cells("part", "lambda t: probe(key) * 2.0 + t")
        space.new_cells("value", "lambda: part(0) + part(1)")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text("from exported import mx_model\nmx_model.Policies.collect()\n",
                                        encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py", "--no-spec"],
                   cwd=tmp_path, check=True)
    probe = textwrap.dedent("""\
        import threading
        from exported_cy import mx_model as m

        serial_threads, lock = set(), threading.Lock()
        def serial_probe(key):
            with lock:
                serial_threads.add(threading.get_ident())
            return float(key)

        m.Policies.probe = serial_probe
        serial = m.Policies.collect().sum.value
        assert serial == sum(4.0 * k + 1.0 for k in range(8)) == 120.0
        assert serial_threads == {threading.get_ident()}

        threads = set()
        together = threading.Barrier(2, timeout=20)                 # two items evaluated at the same time
        def probe(key):
            with lock:
                threads.add(threading.get_ident())
            if key in (0, 4):                                       # these two items must overlap
                together.wait()
            return float(key)
        m = type(m)()
        m.Policies.probe = probe
        m._mx_collect_workers = 4
        assert m.Policies.collect().sum.value == serial
        assert len(threads) >= 2 and not together.broken

        m = type(m)()
        m._mx_collect_workers = 4
        m._mx_collect_workers = 1
        m.Policies.probe = serial_probe
        serial_threads.clear()
        assert m.Policies.collect().sum.value == serial
        assert serial_threads == {threading.get_ident()}
        print("ok")
        """)
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stdout + result.stderr
