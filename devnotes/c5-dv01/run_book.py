"""Run the DV01 book on the compiled package: a thread pool, one task per policy (the direct route).

    python run_book.py PACKAGE --workers N [--policies N] [--closes N] [--save F] [--check F]

Each task makes Outer[i], computes dv01 and the reserve at every close (three Valuation items each,
down, base and up),
and deletes the policy's Outer, Valuation and Actual items. Prints one JSON line: wall and CPU
seconds of the book (model import and first touch of the shared Spaces excluded), peak RSS MiB,
minor page faults, and a checksum.
"""
import argparse
import importlib
import json
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.setrecursionlimit(10_000)
ap = argparse.ArgumentParser()
ap.add_argument('package')
ap.add_argument('--workers', type=int, default=1)
ap.add_argument('--policies', type=int, default=1050)
ap.add_argument('--closes', type=int, default=3)
ap.add_argument('--save')
ap.add_argument('--check')
# Experiments only (values then differ from the book's): keep every item alive; give each worker its own
# Market items (closes 3k .. 3k+2 for worker k) so that no Market item is shared between threads.
ap.add_argument('--keep', action='store_true')
ap.add_argument('--private-market', action='store_true')
args = ap.parse_args()
m = importlib.import_module(args.package).mx_model
closes = range(args.closes)
# The shared Spaces are filled once before the clock starts, as a close's run would find them.
m.Data.model_point_table()
for c in range(12 if args.private_market else args.closes):
    for s in (0, 1, 2):
        m.Market[c, s].disc(0)


import threading
lanes = {}


def run(i):
    base = 0
    if args.private_market:
        base = 3 * lanes.setdefault(threading.get_ident(), len(lanes))
    o = m.Outer[i]
    out = [o.dv01(base + c) for c in closes] + [o.reserve(base + c) for c in closes]
    del o
    if not args.keep:
        del m.Outer[i]
        for c in closes:
            for s in (0, 1, 2):
                del m.Valuation[i, base + c, s]
            del m.Actual[i, base + c]
    return out


ru0, t0, c0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter(), time.process_time()
with ThreadPoolExecutor(args.workers) as pool:
    results = list(pool.map(run, range(args.policies)))
wall, cpu = time.perf_counter() - t0, time.process_time() - c0
ru = resource.getrusage(resource.RUSAGE_SELF)
print(json.dumps({'workers': args.workers, 'policies': args.policies, 'wall_s': round(wall, 3),
                  'cpu_s': round(cpu, 3), 'user_s': round(ru.ru_utime - ru0.ru_utime, 3),
                  'sys_s': round(ru.ru_stime - ru0.ru_stime, 3), 'maxrss_mib': round(ru.ru_maxrss / 1024, 1),
                  'minflt': ru.ru_minflt - ru0.ru_minflt, 'sum': sum(map(sum, results))}))
if args.save:
    json.dump(results, open(args.save, 'w'))
if args.check:
    ref = json.load(open(args.check))
    bad = [(i, c) for i, row in enumerate(ref[:len(results)]) for c, v in enumerate(row)
           if not abs(results[i][c] - v) <= max(1e-7, 1e-10 * abs(v))]
    print(json.dumps({'checked': min(len(ref), len(results)), 'mismatches': len(bad), 'first': bad[:5]}))
    sys.exit(1 if bad else 0)
