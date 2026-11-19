"""collect_outputs with several outputs: a ``sum`` output written ``{k: cell(k) for k in D}`` (D a
Reference holding integers) or ``cell()`` is summed natively; any other output, and the ``individual``
section, is merged as the export merges it. Results, errors and the order of keys equal the exported
model's, and the build report says which outputs are native and why the others are not."""
import json
import subprocess
import sys
import textwrap

import modelx as mx


def test_native_and_python_outputs_equal_the_export(tmp_path):
    model = mx.new_model("NativeMerge")
    try:
        keys = (3, 1, 4, 5, 9, 2)
        months = (0, 1, 2, 5, 3)            # not contiguous, not sorted: positions in D, not values
        empty = ()
        common = {"keys": keys, "months": months, "empty": empty}
        outputs = {
            "Mixed": ('lambda: {"individual": {"state": {t: cf(t) for t in months}}, '
                      '"sum": {"pv": pv(), "cf": {t: cf(t) for t in months}, "count": 1, '
                      '"none": {t: cf(t) for t in empty}}}'),
            "CellDomain": 'lambda: {"sum": {"cf": {t: cf(t) for t in domain()}}}',
            "MissingKey": 'lambda: {"sum": {"pv": pv(), "cf": {t: cf(t) for t in range(size())}}}',
        }
        for name, collect_outputs in outputs.items():
            space = model.new_space(name, formula="lambda key: None", refs=dict(common),
                                    collect_over={"key": keys}, collect_outputs=collect_outputs)
            space.new_cells("cf", "lambda t: key / 3.0 + t * 0.1 + (1e16 if key == 4 and t == 0 else 0.0)")
            space.new_cells("pv", "lambda: sum(cf(t) for t in months) / 7.0")
            space.new_cells("domain", "lambda: months")
            space.new_cells("size", "lambda: 2 if key == 5 else 3")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    spec = {"spaces": {name: {"serial_collect": "reuse"} for name in outputs}}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(textwrap.dedent("""\
        from exported import mx_model as m
        m.Mixed.collect(); m.CellDomain.collect()
        for key in m.MissingKey.keys:
            m.MissingKey[key].pv(); m.MissingKey[key].cf(0); m.MissingKey[key].size()
        """), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                    "--spec", "spec.py"], cwd=tmp_path, check=True)
    probe = textwrap.dedent("""\
        import json, sys
        package = sys.argv[1]
        m = __import__(package).mx_model
        def plain(value):
            if hasattr(value, '_fields'):
                return {k: plain(v) for k, v in value._fields.items()}
            if isinstance(value, dict):
                return [[repr(k), plain(v)] for k, v in value.items()]
            return repr(value)
        out = {}
        for workers in (1, 4):
            m.collect_workers = workers
            m.clear_all()
            out[f'mixed{workers}'] = plain(m.Mixed.collect())
            out[f'cell{workers}'] = plain(m.CellDomain.collect())
            try:
                m.MissingKey.collect()
                out[f'missing{workers}'] = 'no error'
            except Exception as error:
                out[f'missing{workers}'] = [type(error).__name__, str(error)]
        out['native'] = {name: getattr(m, name)._mx_collect_options_v2()[1] is not None
                         if hasattr(getattr(m, name), '_mx_collect_options_v2') else None
                         for name in ('Mixed', 'CellDomain', 'MissingKey')}
        print(json.dumps(out))
        """)
    results = {}
    for package in ("exported", "exported_cy"):
        done = subprocess.run([sys.executable, "-c", probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        results[package] = json.loads(done.stdout)
    live, compiled = results["exported"], results["exported_cy"]
    assert compiled["native"] == {"Mixed": True, "CellDomain": False, "MissingKey": True}
    for key in ("mixed1", "cell1", "missing1"):
        assert compiled[key] == live[key], key
    # 4 workers: the individual section in domain order and with the serial values
    assert compiled["mixed4"]["individual"] == live["mixed1"]["individual"]
    assert compiled["missing4"] == live["missing4"] == live["missing1"]
    assert compiled["missing1"][0] == "ValueError" and "structure changed" in compiled["missing1"][1]
    mixed = compiled["mixed1"]
    assert [entry[0] for entry in mixed["sum"]["cf"]] == ["0", "1", "2", "5", "3"]
    assert mixed["sum"]["none"] == [] and mixed["sum"]["count"] == repr(6)
    report = json.loads((tmp_path / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))
    def leaves(space):
        entry = report[f"exported_cy._mx_classes._c_{space}._f_collect_outputs"]
        return {(leaf["mode"], leaf["output"]): (leaf["native"], leaf.get("reason", ""))
                for leaf in entry["collect_outputs"]}
    mixed_leaves = leaves("Mixed")
    assert {key: native for key, (native, _) in mixed_leaves.items()} == {
        ("individual", ""): False, ("sum", "pv"): True, ("sum", "cf"): True, ("sum", "count"): False,
        ("sum", "none"): True}
    assert "Reference" in leaves("CellDomain")[("sum", "cf")][1]
    assert leaves("MissingKey")[("sum", "pv")][0] and not leaves("MissingKey")[("sum", "cf")][0]
