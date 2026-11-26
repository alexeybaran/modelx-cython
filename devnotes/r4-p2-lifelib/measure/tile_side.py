"""tile_side.py SIDE.csv ROWS N DST: a per-point side table for a book tiled from ROWS points to N."""
import sys, pandas as pd
src, rows, n, dst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
df = pd.read_csv(src, dtype=str, keep_default_na=False)
parts = []
for c in range(-(-n // rows)):
    part = df.copy()
    part['point_id'] = [str(int(p) + c * rows) for p in part['point_id']]
    parts.append(part[[int(p) <= n for p in part['point_id']]])
pd.concat(parts, ignore_index=True).to_csv(dst, index=False)
