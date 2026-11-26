"""prep_sec.py Term_US_SEC|MYGA_US_SEC N OUT [live]: copy the product, tile its SEC book to N points,
declare collect over idx with a book-total cell, export."""
import shutil, sys, pathlib
import pandas as pd
import modelx as mx

LIB = pathlib.Path('/home/user/ext/lifelib/lifelib/libraries/uslib/products')
name, n, out = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
if out.exists():
    shutil.rmtree(out)
shutil.copytree(next(LIB.glob(f'*/{name}')).parent, out / 'src')
table = out / 'src' / 'model_point_table_sec.csv'
df = pd.read_csv(table, dtype=str, keep_default_na=False)
tiled = df.iloc[[i % len(df) for i in range(n)]].copy()
tiled['point_id'] = [str(i + 1) for i in range(n)]
if 'policy_id' in tiled:
    tiled['policy_id'] = [f'{p}-{i // len(df)}' for i, p in enumerate(tiled['policy_id'])]
tiled.to_csv(table, index=False)
m = mx.read_model(out / 'src' / name)
P = m.Projection
P.new_cells('net_cf_total', '''def net_cf_total():
    total = 0.0
    for t in range(proj_len()):
        total = total + net_cf(t)
    return total
''')
P.new_cells('book', 'lambda: tuple(range(len(data.model_point_table())))')
P.set_collect(over={'idx': P.book}, outputs='lambda: {"sum": {"net_cf": net_cf_total()}}')
B = m.new_space('Book')
B.proj = P
B.new_cells('total', 'lambda: proj.collect().sum.net_cf')
if len(sys.argv) > 4:
    print('LIVE', repr(B.total()))
m.export(out / 'src' / f'{name}_nomx', locked_spaces=['Data'])
