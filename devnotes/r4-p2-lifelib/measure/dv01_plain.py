"""dv01_plain.py BUILD BOOK: the report from the compiled Outer Cells, every item kept; seconds, CPU, peak MB, totals."""
import resource, sys, time
sys.path.insert(0, '/home/user/ext/lifelib-dv01')
sys.setrecursionlimit(20_000)
from lifelib.libraries.uslib.products.fixed_deferred_annuity.dv01_sec import compiled
model = compiled.load(sys.argv[1], sys.argv[2])
start, cpu = time.perf_counter(), time.process_time()
dv01, cs01 = model.Outer.annual_risk_report(1, 2)
print(round(time.perf_counter() - start, 2), round(time.process_time() - cpu, 2),
      round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
      repr(float(dv01.to_numpy().sum())), repr(float(cs01.to_numpy().sum())), len(model.Valuation.point_indices()))
