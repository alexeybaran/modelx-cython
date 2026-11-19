"""Free-threaded stress of the published prepared tables (T-4.2, T-5c), marked ``stress``: left out of the default run.

What is supported, and what this checks:
- **Clears and table assignments from another thread during four-worker collects.** The runtime makes them one
  step with its check (``runtime/table_cells.py``: a collect count and a "clearing" mark under the runtime lock).
  Each one either raises (a collect is running) or runs wholly between two collects. A collect that would start
  while one runs raises. Every collect that runs returns the intact result. Items read the published table of a
  Data frame cell without a lock, and the frame cell is cleared and reassigned, MX_CY_STRESS_RUNS rounds (200).
- **One read of the published field** (``_mx_prepared_<name>``): the fast path reads the field once and takes the
  locked path when it is None. Checked on the generated source.

Undefined by rule (``doc/source/freethreading.md``, as on main): a clear or an assignment from a user's own thread
while another of the user's threads evaluates the model. That interleaving is not exercised here.
"""
import os
import subprocess
import sys
import textwrap

import pytest

RUNS = int(os.environ.get('MX_CY_STRESS_RUNS', '200'))

MODEL = """\
import modelx as mx
import pandas as pd

FRAME = pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0], "rate": [0.01, 0.02, 0.03, 0.04]},
                     index=pd.Index([1, 2, 3, 4], name="policy_id"))
IDS = (1, 2, 3, 4)


def build():
    model = mx.new_model("Stress")
    data = model.new_space("Data", refs={"pd": pd})
    data.FRAME = FRAME
    data.new_cells("points", "def points():\\n    return FRAME.copy()\\n")
    proj = model.new_space("Projection", formula="lambda policy_id: None", refs={"data": data, "ids": IDS},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"pv": pv()}}')
    proj.new_cells("sa", 'lambda: data.points().loc[policy_id, "sa"]')
    proj.new_cells("rate", 'lambda t: data.points().loc[policy_id, "rate"] * (1 + t % 2)')
    proj.new_cells("pv", "lambda: sum(sa() * (1.0 + rate(t)) ** t for t in range(24))")
    return model
"""

CHECKS = """\
import math, sys, threading, time
from importlib import import_module
from stress_model import FRAME

m = import_module("stress_nomx_cy").mx_model
m._mx_collect_set_workers_v2(1)
base = m.Projection.collect().sum.pv
m._mx_collect_set_workers_v2(4)
stop, other = threading.Event(), []
counts = {"cleared": 0, "assigned": 0, "raised": 0}

def disturber():
    k = 0
    while not stop.is_set():
        k += 1
        try:
            if k % 3 == 0:
                m.Data.points = FRAME.copy()        # a new frame of the same values: a new published table
                counts["assigned"] += 1
            else:
                (m.Data if k % 3 == 1 else m.Projection).clear_all()
                counts["cleared"] += 1
        except RuntimeError as error:
            if "while a collect of this model is running" in str(error):
                counts["raised"] += 1
            else:
                other.append(str(error))

thread = threading.Thread(target=disturber)
thread.start()
results, refused = [], 0
deadline = time.monotonic() + 300
try:      # RUNS collects at least, and until the other thread has cleared and assigned often enough
    while len(results) < int(sys.argv[1]) or counts["cleared"] < 20 or counts["assigned"] < 10:
        assert time.monotonic() < deadline, (counts, refused, len(results))
        try:
            results.append(m.Projection.collect().sum.pv)
        except RuntimeError as error:
            assert "of this model is running" in str(error), error
            refused += 1
finally:
    stop.set()
    thread.join()
bad = [r for r in results if not math.isclose(r, base, rel_tol=1e-12)]
assert results and not bad, (base, bad[:3])
assert not other, other[:3]
assert counts["raised"], counts          # some clears met a running collect
print("ok", len(results), "collects,", refused, "refused;", counts)
"""


@pytest.mark.stress
def test_clears_and_assignments_during_four_worker_collects(tmp_path):
    (tmp_path / 'stress_model.py').write_text(MODEL, encoding='utf-8')
    (tmp_path / 'build_model.py').write_text(textwrap.dedent("""\
        from stress_model import build
        m = build()
        m.export("stress_nomx", locked_spaces=["Data"])
        m.close()
        """), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        "from stress_nomx import mx_model\nfor k in (1, 2, 3, 4):\n    mx_model.Projection[k].collect_outputs()\n"
        "mx_model.Projection.collect()\n", encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'parallel_workers': 1}), encoding='utf-8')
    (tmp_path / 'checks.py').write_text(CHECKS, encoding='utf-8')
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=2400,
                                    capture_output=True, text=True)
    run('build_model.py')
    run('-m', 'modelx_cython', 'stress_nomx', '--sample', 'sample.py', '--spec', 'spec.py')
    generated = (tmp_path / 'stress_nomx_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    # the reads of the items go through the published table of Data.points, read once per call of the accessor
    assert 'self.data._mx_prepared_points()' in generated
    accessor = generated[generated.index('def _mx_prepared_points(self)'):][:700]
    assert 'table = self._mx_ptable_points' in accessor and 'if table is not None:' in accessor, accessor
    out = run('checks.py', str(RUNS))
    assert out.stdout.startswith('ok'), out.stdout + out.stderr
