"""Free-threaded race test of the per-item table holds (T-4.2, T-5c), marked ``stress``: left out of the default run.

Four threads read item cells (new items, first binds of the Data tables and of the published prepared tables) while
the main thread reassigns Data.flat and clears Items, for MX_CY_STRESS_RUNS rounds (200 by default). Every value
read must be one of the two frames' live values. The one error allowed is a KeyError on the item number of an item
a clear dropped while it was being read (Space clears are not synchronised with running reads, on main the same:
17 of 200 rounds there); it is counted. Any other error, or another value, fails.
"""
import json
import os

import pytest

import test_table_probe as probe

RUNS = int(os.environ.get('MX_CY_STRESS_RUNS', '200'))

READERS = '''\
import json
import sys
import threading
from importlib import import_module
from probe_model import build

compiled = import_module("probe_nomx_cy").mx_model
live = build()
frames = [live.Data.flat(), live.Data.flat().assign(q=live.Data.flat()["q"] * 3)]
expected = []
for frame in frames:
    live.Data.flat[()] = frame
    live.Items.clear_all()
    expected.append({t: float(live.Items[1].v__flat(t)) for t in range(4)})
failing, reads, refused, dropped = [], 0, 0, 0
for run in range(int(sys.argv[1])):
    stop = threading.Event()
    bad = []

    def read(worker):
        global reads, dropped
        i = 100 + worker
        while not stop.is_set():
            for t in range(4):
                try:
                    value = float(compiled.Items[i].v__flat(t))
                except KeyError as error:
                    if error.args != (i,):
                        bad.append(f"KeyError: {error}")
                        return
                    dropped += 1        # the item was dropped by a clear: the base's behaviour
                    break
                except Exception as error:      # noqa: BLE001 - recorded
                    bad.append(f"{type(error).__name__}: {error}")
                    return
                reads += 1
                if value not in (expected[0][t], expected[1][t]):
                    bad.append(f"value {value} at t={t}")
            i += 4

    threads = [threading.Thread(target=read, args=(w,)) for w in range(4)]
    for thread in threads:
        thread.start()
    for k in range(20):
        try:
            compiled.Data.flat[()] = frames[k % 2].copy()
            compiled.Items.clear_all()
        except RuntimeError:        # a clear refused while a read runs: not this test
            refused += 1
    stop.set()
    for thread in threads:
        thread.join()
    if bad:
        failing.append((run, bad[:3]))
json.dump({"failing": failing, "reads": reads, "refused": refused, "dropped": dropped}, sys.stdout)
'''


@pytest.mark.stress
def test_readers_while_reassigned(tmp_path):
    (tmp_path / 'probe_model.py').write_text(
        probe.MODEL.format(ops=probe.OPS, stmts=probe.STMTS, data=probe.DATA, t5_items=probe.T5_ITEMS),
        encoding='utf-8')
    (tmp_path / 'sample.py').write_text(probe.SAMPLE, encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'parallel_workers': 1}), encoding='utf-8')
    (tmp_path / 'readers.py').write_text(READERS, encoding='utf-8')
    probe._run(tmp_path, 'probe_model.py')
    probe._run(tmp_path, '-m', 'modelx_cython', 'probe_nomx', '--sample', 'sample.py', '--spec', 'spec.py')
    result = json.loads(probe._run(tmp_path, 'readers.py', str(RUNS)).stdout)
    assert result['reads'] > 0 and not result['failing'], result
