"""A compiled model without table cells imports and runs without modelx_cython installed, as
upstream's compiled models do: the generated modules import the table-cell runtime only when a
table cell reads it."""
import subprocess
import sys

import modelx as mx


def test_tableless_model_imports_without_the_compiler(tmp_path):
    model = mx.new_model("NoCompiler")
    try:
        space = model.new_space("Sp")
        space.new_cells("f", "def f(t):\n    return round(t * 1.25, 1) + max(t, 2)\n")
        live = repr([space.f(t) for t in range(4)])
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nfor t in range(4): mx_model.Sp.f(t)\n", encoding="utf-8")
    build = subprocess.run([sys.executable, "-B", "-m", "modelx_cython", "exported", "--sample", "sample.py",
                            "--no-spec"], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert build.returncode == 0, build.stdout + build.stderr
    source = (tmp_path / "exported_cy" / "_mx_classes.py").read_text(encoding="utf-8")
    assert "modelx_cython" not in source
    run = subprocess.run([sys.executable, "-B", "-c",
                          "import sys\n"
                          "sys.modules['modelx_cython'] = None\n"      # any import of it raises
                          "from exported_cy import mx_model\n"
                          "print(repr([mx_model.Sp.f(t) for t in range(4)]))"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip().splitlines()[-1] == live
