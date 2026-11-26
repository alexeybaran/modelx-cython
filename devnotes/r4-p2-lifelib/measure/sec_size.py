"""sec_size.py MODEL: per-item object size and time on the shipped book -> a book size whose kept items fit memory."""
import importlib, sys, time
M = sys.argv[1]
m = importlib.import_module(f'{M}_nomx_cy').mx_model
P = m.Projection; m.Data.model_point_table(); keys = P.book()
s = time.perf_counter()
for r in range(5):
    for k in keys:
        P[k].collect_outputs(); del P[k]
per = (time.perf_counter() - s) / 5 / len(keys)
size = type(P).__basicsize__
n = int(min(20000, 4.0e9 / (size * 1.2 + 64e3), 15.0 / per))
print(M, size, round(per * 1e6), n)
