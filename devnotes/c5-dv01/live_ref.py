"""Reference DV01s and reserves from live modelx for the first N policies, as run_book.py computes them:
python live_ref.py MODEL_DIR N OUT.json"""
import json
import sys

import modelx as mx

m = mx.read_model(sys.argv[1])
rows = []
for i in range(int(sys.argv[2])):
    o = m.Outer[i]
    rows.append([o.dv01(c) for c in range(3)] + [o.reserve(c) for c in range(3)])
    del m.Outer[i]
    for c in range(3):
        for s in (0, 1, 2):
            del m.Valuation[i, c, s]
        del m.Actual[i, c]
json.dump(rows, open(sys.argv[3], 'w'))
