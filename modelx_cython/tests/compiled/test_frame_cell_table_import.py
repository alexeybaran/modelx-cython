"""A frame cell read by row is bound through the table runtime: the compiled module imports it.

The prepared table of a Data frame cell is published with ``_mx_table_cells.bind_frame``, a call added by
table_cells.finish, the last pass. The module must import the table runtime for that call; the model builds
and its values equal live.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd

PROBE = textwrap.dedent('''\
    import json
    from exported_cy import mx_model as m
    print(json.dumps([[m.Projection[i].sa(), m.Projection[i].rate(t)] for i in (1, 2, 3, 4) for t in range(3)]))
''')


def test_a_frame_cell_read_by_row_imports_the_table_runtime(tmp_path):
    frame = pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0], "rate": [0.01, 0.02, 0.03, 0.04]},
                         index=pd.Index([1, 2, 3, 4], name="policy_id"))
    model = mx.new_model("FrameImport")
    try:
        data = model.new_space("Data", refs={"pd": pd})
        data.FRAME = frame
        data.new_cells("points", "def points():\n    return FRAME.copy()\n")
        proj = model.new_space("Projection", formula="lambda policy_id: None", refs={"data": data})
        proj.new_cells("sa", 'lambda: data.points().loc[policy_id, "sa"]')
        proj.new_cells("rate", 'lambda t: data.points().loc[policy_id, "rate"] * (1 + t % 2)')
        live = [[float(proj[i].sa()), float(proj[i].rate(t))] for i in (1, 2, 3, 4) for t in range(3)]
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\n"
        "for i in (1, 2, 3, 4):\n    m.Projection[i].sa()\n    for t in range(3):\n        m.Projection[i].rate(t)\n",
        encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py", "--no-spec"],
                   cwd=tmp_path, check=True, capture_output=True)
    source = (tmp_path / "exported_cy" / "_mx_classes.py").read_text(encoding="utf-8")
    assert "_mx_table_cells.bind_frame(" in source, "the model does not exercise the frame-cell bind"
    out = json.loads(subprocess.run([sys.executable, "-c", PROBE], cwd=tmp_path, check=True,
                                    capture_output=True, text=True).stdout)
    assert out == live
