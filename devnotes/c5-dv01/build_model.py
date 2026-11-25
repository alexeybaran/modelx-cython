"""Build C5's reconstruction of the SEC DV01 example on lifelib's MYGA_US_SEC.

Not D0's model text (unreachable from C5): same Space shape, Data, ESG, Market (shared, locked),
Actual, Valuation (Projection's formulas, a dynamic Reference ``act``), Outer.

    python build_model.py PRODUCT_DIR OUT_DIR [--book N] [--no-export]

Writes OUT_DIR/DV01_SEC (the modelx model), OUT_DIR/DV01_SEC_nomx (its export, Data, ESG and
Market locked), the book CSV beside them, and copies the product's other CSVs.
"""
import shutil
import sys
from pathlib import Path

import modelx as mx
import pandas as pd

product, out = Path(sys.argv[1]), Path(sys.argv[2])
book = int(sys.argv[sys.argv.index('--book') + 1]) if '--book' in sys.argv else 1050
out.mkdir(parents=True, exist_ok=True)
for csv in product.glob('*.csv'):
    shutil.copy(csv, out / csv.name)
# The book: the SEC table's in-force and new points, repeated, numbered 1..N.
table = pd.read_csv(product / 'model_point_table_sec.csv')
rows = pd.concat([table] * (book // len(table) + 1), ignore_index=True).iloc[:book].copy()
rows['point_id'] = range(1, book + 1)
rows['policy_id'] = [f'DV01-{i:06d}' for i in rows['point_id']]
rows.to_csv(out / 'model_point_table_book.csv', index=False)

m = mx.read_model(product / 'MYGA_US_SEC', name='DV01_SEC')
m.Data.model_point_file = 'model_point_table_book.csv'

esg = m.new_space('ESG')
esg.np = __import__('numpy')
esg.horizon = 1322
esg.new_cells('zero_rates', '''def zero_rates(close):
    """Annual zero rates by month for a close: a Nelson-Siegel curve whose level moves by close."""
    tau = np.maximum(np.arange(horizon, dtype=float) / 12.0, 1.0 / 12.0)
    x = tau / 2.5
    slope = (1.0 - np.exp(-x)) / x
    return 0.035 + 0.0025 * close - 0.012 * slope + 0.01 * (slope - np.exp(-x))
''')
esg.new_cells('spread', 'lambda close: 0.0060 - 0.0005 * close')

market = m.new_space('Market', formula='lambda close, shock: None',
                     refs={'esg': esg, 'data': m.Data, 'bp': 0.0001, 'np': esg.np})
market.new_cells('zero_vec', 'lambda: esg.zero_rates(close) + (shock - 1) * bp')
market.new_cells('disc_vec', '''def disc_vec():
    t = np.arange(len(zero_vec()), dtype=float) / 12.0
    return np.exp(-zero_vec() * t)
''')
# The scenario's level at month 0 plus the ESG curve's move from month 0, shifted by the shock: [s, t].
market.new_cells('curve_move', 'lambda: esg.zero_rates(close) - esg.zero_rates(close)[0]')
market.new_cells('market_rate_arr',
                 'lambda: data.market_rate_array()[:, :1] + 0.5 * curve_move() + (shock - 1) * bp + 0.0005 * close')
market.new_cells('ref_yield_arr',
                 'lambda: data.mva_ref_yield_array()[:, :1] + curve_move() + (shock - 1) * bp + esg.spread(close) - 0.006')
market.new_cells('disc', 'lambda t: float(disc_vec()[t])')
market.new_cells('market_rate', 'lambda t, s: float(market_rate_arr()[s, t])')
market.new_cells('ref_yield', 'lambda t, s: float(ref_yield_arr()[s, t])')

actual = m.new_space('Actual', formula='lambda idx, close: None', refs={'data': m.Data})
actual.new_cells('pols_if', 'lambda: float(data.policy_count()[idx]) * (1.0 - 0.01 * close)')
actual.new_cells('av_scale', 'lambda: 1.0 + 0.002 * close')

val = m.new_space('Valuation', bases=m.Projection,
                  formula="lambda idx, close, shock: {'refs': {'act': Actual[idx, close]}}")
val.Actual = actual
val.Market = market
val.new_cells('mkt', 'lambda: Market[close, shock]')
val.cells['scenario_rate'].set_formula('''def scenario_rate(t, name):
    s = scenario_number()
    if t < data.scenario_first_mths()[s]:
        raise ValueError(f"point_id {point_id()}: the rate scenario sets no level at t = {t}")
    if name == MVA_REF_YIELD:
        return mkt().ref_yield(t, s)
    elif name == MARKET_RATE:
        return mkt().market_rate(t, s)
    raise ValueError("invalid name")
''')
val.cells['pols_if_init'].set_formula('''def pols_if_init():
    if duration_mth_init() < 0:
        return 0.0
    return act.pols_if()
''')
val.new_cells('pv_net_cf', '''def pv_net_cf():
    total = 0.0
    for t in range(proj_len()):
        total += net_cf(t) * mkt().disc(t)
    return total
''')

outer = m.new_space('Outer', formula='lambda idx: None', refs={'Valuation': val, 'bp': 0.0001})
outer.new_cells('pv', 'lambda close, shock: Valuation[idx, close, shock].pv_net_cf()')
outer.new_cells('dv01', 'lambda close: (pv(close, 0) - pv(close, 2)) / 2.0')
outer.new_cells('reserve', 'lambda close: pv(close, 1)')

m.write(out / 'DV01_SEC')
if '--no-export' not in sys.argv:
    m.export(out / 'DV01_SEC_nomx', locked_spaces=['Data', 'ESG', 'Market'])
m.close()
