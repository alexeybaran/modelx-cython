"""Public v1 Collect keeps serial sum semantics and dispatches workers only with the GIL off."""
import os
import subprocess
import sys
import textwrap

import modelx as mx


def test_collect_keeps_serial_sum_and_uses_workers_without_gil(tmp_path):
    model = mx.new_model("DomainDefault")
    try:
        cancellation = (1e16, 1.0, -1e16, 1.0, 2.0, 3.0, 4.0, -0.0)
        exact = (16.0, 1.0, -16.0, 1.0, 2.0, 3.0, 4.0, -0.0)
        keys = tuple(range(len(cancellation)))
        for name, values in (("Cancellation", cancellation), ("Exact", exact)):
            space = model.new_space(
                name, formula="lambda key: None", refs={"keys": keys, "values": values},
                collect_over={"key": "keys"},
                collect_outputs=lambda: {"sum": {"value": value()}, "individual": {"value": value()}},
            )
            space.new_cells("value", "lambda: values[key] + 0.0")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Cancellation.collect()\nmx_model.Exact.collect()\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--no-spec"], cwd=tmp_path, check=True)
    probe = textwrap.dedent("""\
        import sys
        import threading
        import pytest

        gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)
        before = gil_enabled()
        assert before == bool(int(sys.argv[1]))
        from exported import mx_model as exported
        from exported_cy import mx_model as m
        from exported_cy._mx_collect import _NativeFields
        assert gil_enabled() == before

        def result_values(result):
            return result.sum.value, {key: item.value for key, item in result.individual.items()}

        def collect(space, workers):
            threads = []
            ready = threading.Barrier(workers, timeout=10) if workers else None

            class CollectThread(threading.Thread):
                def run(self):
                    threads.append(self)
                    if ready is not None:
                        ready.wait()
                    super().run()

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(threading, "Thread", CollectThread)
                result = space.collect()
            assert len({thread.ident for thread in threads}) == workers
            assert all(not thread.is_alive() for thread in threads)
            assert type(result._CollectResult__storage) is _NativeFields
            return result_values(result)

        serial = collect(m.Cancellation, 0)
        assert serial == result_values(exported.Cancellation.collect())
        assert serial[0] == 10.0

        m = type(m)()
        m._mx_collect_workers = 4
        if before:
            assert collect(m.Cancellation, 0) == serial
        # The four-worker inputs sum exactly under any grouping.
        assert collect(m.Exact, 0 if before else 4) == result_values(exported.Exact.collect())
        assert gil_enabled() == before
    """)
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    subprocess.run([sys.executable, "-c", probe, str(int(gil_enabled))], cwd=tmp_path, check=True, timeout=120)
    if not gil_enabled:
        subprocess.run([sys.executable, "-c", probe, "1"], cwd=tmp_path, check=True, timeout=120,
                       env={**os.environ, "PYTHON_GIL": "1"})
