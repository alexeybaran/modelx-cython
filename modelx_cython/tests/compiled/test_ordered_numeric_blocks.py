"""Differential test of the factory path on a non-insurance model."""
import json
import math
import subprocess
import sys
import textwrap

import modelx as mx


def test_factory_alias_and_rejection(tmp_path):
    model = mx.new_model("FactoryReduction")
    try:
        source = """def factory(case, key):
    if case < 0:
        raise ValueError('invalid case')
    if case == 13 and key in (9, 25):
        raise ValueError(str(key))
    return {'refs': {'point': key}}
"""
        keys = tuple(range(35))
        s = model.new_space("Runs", formula=source,
                            refs={"point": 0, "periods": (3, 1, 2), "keys": keys, "offset": 0.0},
                            collect_over={"key": keys},
                            collect_outputs='lambda: {"sum": {"total": total(), "zero": zero(), "cancel": cancel(), '
                                            '"by_period": {i: charge(i) for i in periods}}}')
        s.new_cells("charge", "lambda i: point * 0.25 + case + i + offset")
        s.new_cells("zero", "lambda: -0.0")
        s.new_cells("cancel", "lambda: 1e16 if point % 3 == 0 else (1.0 if point % 3 == 1 else -1e16)")
        s.new_cells("total", "lambda: charge(1) + charge(3)")
        other = model.new_space("Other", formula=source, refs={"point": 0, "keys": keys, "math": math},
                                collect_over={"key": keys},
                                collect_outputs='lambda: {"sum": {"v": value()}}')
        other.new_cells("value", "lambda: float(math.floor(point + case))")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    options = {"serial_collect": "reduce", "factory_collect_reuse": True}
    (tmp_path / "spec.py").write_text(repr({"spaces": {"Runs": options, "Other": options}}))
    (tmp_path / "sample.py").write_text("from exported import mx_model as m\nm.Runs.collect(2)\nm.Other.collect(2)\n")
    subprocess.run([sys.executable, "-B", "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--spec", "spec.py"], cwd=tmp_path, check=True)
    generated = (tmp_path / "exported_cy" / "_mx_classes.py").read_text()
    if "def _mx_fill_ordered_rows" in generated:
        import ast
        classes = {c.name: c for c in ast.parse(generated).body if isinstance(c, ast.ClassDef)}
        def evaluator(name):
            return next(f for f in classes[name].body if getattr(f, 'name', None) == '_mx_evaluate_serial_collect')
        assert "_mx_reuse_private" in ast.dump(evaluator("_c_Runs"))
        assert "_mx_reuse_private" not in ast.dump(evaluator("_c_Other"))
    probe = textwrap.dedent("""\
        import json, math, sys, importlib
        m = __import__(sys.argv[1]).mx_model
        runtime = importlib.import_module(sys.argv[1] + '._mx_sys')
        supports_rows = getattr(runtime, '_mx_collect_ordered_rows_v1', False)
        blocks = []
        if sys.argv[1] == 'exported_cy' and supports_rows:
            native = runtime._OrderedNumericRows.block
            def witness(instance, coords):
                blocks.append(len(coords))
                return native(instance, coords)
            runtime._OrderedNumericRows.block = witness
        direct = m.Runs[2, 9]
        direct_value = direct.total()
        out = []
        for w in (1, 4):
            m.collect_workers = w
            try:
                m.Runs.collect(-1)
            except ValueError as exc:
                out.append(str(exc))
            else:
                raise AssertionError("factory rejection disappeared")
            try:
                m.Runs.collect(13)
            except ValueError as exc:
                assert str(exc) == "9"
            else:
                raise AssertionError("later coordinate failure disappeared")
            report = m.Runs.collect(w)
            assert list(report.sum.by_period.items()) == [
                (i, sum(key * .25 + w + i for key in range(35))) for i in (3, 1, 2)]
            assert math.copysign(1., report.sum.zero) == -1.
            folded = 1e16
            for key in range(1, 35):
                folded += (1e16, 1., -1e16)[key % 3]
            assert report.sum.cancel == folded
            out.append([report.sum.total, list(report.sum.by_period.items())])
        if sys.argv[1] == "exported_cy":
            assert m.Runs._mx_collect_options_v2()[1] is not None
            assert hasattr(m.Runs, "_mx_prepare_ordered_rows_v1") == supports_rows
        if sys.argv[1] == "exported_cy" and supports_rows:
            if not getattr(sys, '_is_gil_enabled', lambda: True)():
                assert 8 in blocks and 3 in blocks
                blocks.clear()
                sys.setprofile(lambda *info: None)
                try:
                    assert m.Runs.collect(6).sum.total > 0.
                finally:
                    sys.setprofile(None)
                assert not blocks
            m.Runs.periods = [3, 1]
            assert m.Runs._mx_prepare_ordered_rows_v1(tuple(range(35)), (4,), 8, 4) is None
            blocks.clear()
            assert list(m.Runs.collect(5).sum.by_period) == [3, 1] and not blocks
        assert m.Runs[2, 9] is direct and direct.total() == direct_value
        m.Runs.offset = 2.0
        assert m.Runs.collect(7).sum.total == sum(2 * key * .25 + 22 for key in range(35))
        assert m.Runs[2, 9] is direct and direct.total() == direct_value
        assert m.Other.collect(1).sum.v == sum(range(1, 36))
        print(json.dumps(out))
    """)
    def invoke(package):
        result = subprocess.run([sys.executable, "-B", "-c", probe, package], cwd=tmp_path,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert invoke("exported") == invoke("exported_cy")
