import subprocess
import sys
import textwrap

import modelx as mx


def test_parallel_collect_reuse_merges_completed_chunks_and_retries(tmp_path):
    model = mx.new_model("PrivateCollect")
    try:
        # These values add exactly in any worker grouping.
        values = (16.0, 1.0, -16.0, 1.0, 2.0, 3.0, 4.0, -0.0)
        keys = tuple(range(len(values)))
        space = model.new_space(
            "Projection", formula="lambda key: None", refs={"keys": keys, "values": values, "adjust": float},
            collect_over={"key": "keys"},
            collect_outputs=lambda: {"sum": {"value": value()}, "individual": {"value": value()}},
        )
        space.new_cells("part", "lambda t: values[key] + t")
        space.new_cells("value", "lambda: adjust(part(0))")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Projection.collect()\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--no-spec"], cwd=tmp_path, check=True)
    assert (tmp_path / 'exported/_mx_sys.py').read_bytes() == (tmp_path / 'exported_cy/_mx_sys.py').read_bytes()
    probe = textwrap.dedent("""\
        import sys
        import threading
        import pytest

        gil = getattr(sys, "_is_gil_enabled", lambda: True)
        before = gil()
        from exported import mx_model as live
        from exported_cy import mx_model as native
        assert gil() == before

        def payload(result):
            return result.sum.value, [(key, item.value) for key, item in result.individual.items()]

        expected = payload(live.Projection.collect())
        assert expected == (11.0, list(enumerate(live.Projection.values)))
        serial = type(native)()
        serial._mx_collect_workers = 1
        assert payload(serial.Projection.collect()) == expected

        native._mx_collect_workers = 4
        projection = native.Projection
        held = projection[0]
        assert held.value() == 16.0
        threads = []
        expected_workers = 0 if before else 4
        starts = threading.Barrier(expected_workers, timeout=10) if expected_workers else None

        class Observer(threading.Thread):
            def run(self):
                threads.append(self)
                if starts is not None:
                    starts.wait()
                super().run()

        def collect():
            threads.clear()
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(threading, "Thread", Observer)
                try:
                    return projection.collect()
                finally:
                    assert len({thread.ident for thread in threads}) == expected_workers
                    assert all(not thread.is_alive() for thread in threads)

        failure = LookupError("retry")
        def broken(x):
            if x == -16.0:
                raise failure
            return x

        projection.adjust = broken
        for _ in range(2):
            with pytest.raises(LookupError) as error:
                collect()
            assert error.value is failure
        with pytest.raises(KeyError):
            del projection[3]

        seen = []
        def remember(x):
            seen.append(x)
            return x

        projection.adjust = remember
        result = collect()
        assert payload(result) == expected
        assert sorted(seen) == sorted(projection.values)
        assert projection.collect() is result
        assert len(seen) == len(projection.values)
        with pytest.raises(KeyError):
            del projection[3]
        assert projection[0] is held and held.value() == 16.0
        assert projection[3].value() == 1.0
        assert gil() == before
    """)
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True, timeout=90)


def test_obsolete_collect_export_requests_reexport(tmp_path):
    model = mx.new_model("Obsolete")
    try:
        keys = (0, 1)
        projection = model.new_space(
            "Projection", formula="lambda key: None", refs={"keys": keys},
            collect_over={"key": "keys"}, collect_outputs=lambda: {"sum": {"value": value()}},
        )
        projection.new_cells("value", "lambda: float(key)")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    runtime = tmp_path / "exported/_mx_sys.py"
    source = runtime.read_text(encoding="utf-8")
    marker = "_mx_collect_adapter_version = 2"
    assert marker in source
    runtime.write_text(source.replace(marker, "_mx_collect_adapter_version = 1"), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Projection.collect()\n", encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
         "--no-spec", "--translate-only"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "without export protocol 2" in result.stderr and "Re-export the model" in result.stderr
