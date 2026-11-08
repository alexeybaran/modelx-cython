import subprocess
import sys
import textwrap
from pathlib import Path

import modelx as mx
import libcst as cst
import pytest


@pytest.mark.parametrize('legacy_export', [False, True])
def test_parallel_collect_reuse_merges_completed_chunks_and_retries(tmp_path, legacy_export):
    model = mx.new_model("PrivateCollect")
    try:
        values = (1e16, 1.0, -1e16, 1.0, 2.0, 3.0, 4.0, -0.0)
        keys = tuple(range(len(values)))
        for name in ("Reused", "Ordered", "Ordinary"):
            space = model.new_space(
                name, formula="lambda key: None", refs={"keys": keys, "values": values, "adjust": float},
                collect_over={"key": keys}, collect_outputs='lambda: {"sum": {"value": value()}}',
            )
            space.new_cells("part", "lambda t: values[key] + t")
            space.new_cells("value", "lambda: adjust(part(0))")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    if legacy_export:
        # Rebuild the known unversioned export with its original generic
        # functions, retaining this model's own exported data and classes.
        runtime = tmp_path/'exported/_mx_sys.py'
        syntax = cst.parse_module(runtime.read_text(encoding='utf-8'))
        old = cst.parse_module((Path(__file__).parents[1]/
            'modelx_cython/tests/samples/collect_v0.txt').read_text(encoding='utf-8'))
        names = {'collect_from_space', '_mx_collect_parallel', '_mx_collect_chunk',
                 '_mx_collect_native_coords'}
        body = []
        for node in syntax.body:
            if isinstance(node, cst.FunctionDef) and (node.name.value in names
                    or node.name.value.startswith('_mx_collect_')):
                continue
            if (isinstance(node, cst.SimpleStatementLine) and len(node.body) == 1
                    and isinstance(node.body[0], cst.Assign)
                    and isinstance(node.body[0].targets[0].target, cst.Name)
                    and node.body[0].targets[0].target.value in
                        {'_mx_collect_adapter_version', '_mx_parallel_workers'}):
                continue
            body.append(node)
        runtime.write_text(syntax.with_changes(body=(*body, *old.body)).code, encoding='utf-8')
    spec = {"parallel_workers": 4, "spaces": {
        "Reused": {"serial_collect": "reuse", "parallel_collect_reuse": True},
        "Ordered": {"serial_collect": "reuse", "parallel_collect_reuse": True,
                    "parallel_collect_order": "domain"},
    }}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Reused.collect(); mx_model.Ordered.collect(); mx_model.Ordinary.collect()\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--spec", "spec.py"], cwd=tmp_path, check=True)
    generated = (tmp_path / "exported_cy/_mx_classes.py").read_text(encoding="utf-8")
    assert "reuse_in_parallel" in generated
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

        def completion_order(tasks):
            for index in (2, 0, 1, 3):
                yield tasks[index]

        assert m.Ordinary.collect().sum.value == 9.0
        futures.ThreadPoolExecutor = Observer
        futures.as_completed = completion_order
        assert m.Ordered.collect().sum.value == 9.0
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
        first_failure = ValueError("first domain chunk")
        later_failure = ValueError("later domain chunk")
        def multiple_failures(x):
            if x == 1e16:
                raise first_failure
            if x == -1e16:
                raise later_failure
            return x
        def reversed_failures(tasks):
            for index in (2, 1, 0, 3):
                yield tasks[index]
        futures.as_completed = reversed_failures
        m.Reused.adjust = multiple_failures
        try:
            m.Reused.collect()
        except ValueError as error:
            assert error is first_failure
        else:
            raise AssertionError("first domain failure lost")
        futures.as_completed = completion_order
        m.Reused.adjust = float
        result = m.Reused.collect()
        assert result.sum.value == 8.0
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
