"""An export without collect protocol 2 (as a released modelx writes it) compiles and runs equal to
live when the build uses neither collect nor parallel workers, as on upstream."""
import subprocess
import sys

import modelx as mx


def test_plain_model_of_a_protocol_one_export_compiles(tmp_path):
    model = mx.new_model("PlainExport")
    try:
        space = model.new_space("Sp")
        space.new_cells("f", "def f(t):\n    return t * 2 + 0.5\n")
        space.new_cells("g", "def g(t):\n    return f(t) if t == 0 else g(t - 1) + f(t)\n")
        live = repr([space.g(t) for t in range(4)])
        model.export(tmp_path / "exported")
    finally:
        model.close()
    runtime = tmp_path / "exported/_mx_sys.py"
    source = runtime.read_text(encoding="utf-8")
    marker = "_mx_collect_adapter_version = 2"
    assert marker in source
    runtime.write_text(source.replace(marker, "_mx_collect_adapter_version = 1"), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Sp.g(3)\n", encoding="utf-8")
    build = subprocess.run([sys.executable, "-B", "-m", "modelx_cython", "exported", "--sample", "sample.py",
                            "--no-spec"], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([sys.executable, "-B", "-c",
                          "from exported_cy import mx_model\n"
                          "print(repr([mx_model.Sp.g(t) for t in range(4)]))"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip().splitlines()[-1] == live
