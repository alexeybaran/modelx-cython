"""Two threads that evaluate Cells of the same item get the values live gives.

A ``return round(x, n)`` of a double Cell used to keep a one-entry memo per item (input, digits,
result) that two threads could tear: one thread read the other's result for its own input. Here the
rounded amount repeats over ``t`` (one value for even ``t``, another for odd), one thread takes the even
``t`` and one the odd: with the memo, some calls returned the other thread's value. The memo is gone;
each call rounds its own input."""
import subprocess
import sys

import modelx as mx

N = 40000       # t per item, all in the dense cache window
KEYS = (1, 2, 3)


def test_two_threads_on_one_item_round_their_own_inputs(tmp_path):
    model = mx.new_model('RoundThreads')
    try:
        proj = model.new_space('Projection', formula='lambda key: None')
        proj.new_cells('rounded', '''\
def rounded(t):
    amount = key * 0.37 + (t % 2) * 0.011
    adjustment = 0.125
    total = amount + adjustment
    return round(total, min(t, 3))
''')
        live = [proj[k].rounded(t) for k in KEYS for t in range(N)]
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'Projection': {'cells_param_size': {'t': N}}}}),
                                      encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'for k in (1, 2):\n    [m.Projection[k].rounded(t) for t in range(10)]\n', encoding='utf-8')
    build = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                            '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert build.returncode == 0, build.stdout + build.stderr
    (tmp_path / 'expected.txt').write_text(repr(live), encoding='utf-8')
    run = subprocess.run([sys.executable, '-B', '-c', f'''
import threading
from exported_cy import mx_model as m
expected = eval(open('expected.txt').read())
got = {{}}
def work(item, k, parity):
    for t in range(parity, {N}, 2):
        got[k, t] = item.rounded(t)
for k in {KEYS!r}:
    item = m.Projection[k]
    threads = [threading.Thread(target=work, args=(item, k, p)) for p in (0, 1)]
    [th.start() for th in threads]; [th.join() for th in threads]
values = [got[k, t] for k in {KEYS!r} for t in range({N})]
wrong = sum(a != b for a, b in zip(values, expected))
print(wrong)
'''], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip().splitlines()[-1] == '0'
