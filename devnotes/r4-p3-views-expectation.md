# Sub-table views and row walks on the fast line: what I expect, before measuring

Branch `r4/p3-views`, on `r4/p3-tables` (e29d599). I wrote this before the first timing of any build. The port
source is `r4/integration` at 856b984 only. I used it as a reference for the generated code: I ran its translator
on VA_US_S, IP_UK_S and Term_US_S, and I read the code it gives. Nothing is ported from a later commit.

## Scope, from the text of the 57 regional models (upstream lifelib main 20d4dae)

A text scan of every formula lists the forms below. I count only forms in formulas that run per step, or once per
item in an item Space.

| form | example | models (per step unless marked) |
|---|---|---|
| view local `v = T.loc[prefix]`, then `v.loc[k, "c"]` | `grid.loc[years, "surr_charge_rate"]` | ADE, Annuity, CI_UK, Cancer_KR, Euro, FIA, IP, LTC_KR, MYGA, PER, Pension, RILA, TD, VA, VUL, WOL, WP |
| `float(v.loc[k, name])`, the column chosen at run time | `float(sub.loc[max(months), name])` | VA, MYGA, FIA, RILA, CI_UK, Cancer_KR |
| extremes of a view: `v.index.max()` / `.min()`, `min(v.index)` | `int(grid.index.max())` | TD, Euro, PER, WOL, WP, VA, VUL, FIA |
| comprehensions over `v.index`, `T.index` or `T.loc[p].index` (list, set, generator, dict; under `sorted`, `max`, `sum`, ...) | `[m for m in sub.index if m <= ...]` | VA, MYGA, FIA, RILA, IP, ADE, CI_UK, LTC_KR, Pension, Annuity, Dep, Medical_JP, LTC_JP (item), RILA |
| `list` / `tuple` / `sorted` of an index | `list(tbl.index)` | Cancer_KR; VA, VUL, Medical_KR, WholeLife_KR (item) |
| `for k in X.index:` | `for year_from in tab.index:` | Sofort (item), Child |
| `for k, row in T.iterrows():`, fields `row["c"]` | `shock_lapse_rate` | Term_US (item), FIA, Immediate_KR |

None of the 57 formulas has a cell that returns a sub-frame which another formula reads as a view ("prefix cell":
`return T.loc[prefix]` with fewer parts than levels, then `.index` or `.loc` of its value). Every zero-argument
`return T.loc[...]` is a row cell, read by field, and slice 1 covers it. I expect to leave prefix cells out.

## What comes, in three commits (non-test lines)

| commit | pieces | expected lines |
|---|---|---|
| view locals | prefix codes and the view's stored rows in C locals, held per item when every prefix part is fixed per item; `v.loc[k, "c"]` of float64, int64 and object columns; `float(v.loc[k, name])`; `v.index.max()` / `.min()` and `max(v.index)` / `min(v.index)` of one remaining int level; the assignment is evaluated on pandas whenever the view is not native, so a missing prefix raises where live raises it | 350 |
| index walks | comprehensions and `for` loops over `X.index` (view, whole frame, inline prefix, a local holding the index), one name per remaining level or a tuple of names; `list` / `tuple` / `sorted` / `set` / `len` of an index; labels as C integers on int levels and as the stored label objects elsewhere; pandas positions mapped to stored rows on a whole table; a prefix walked natively only when its stored rows are in pandas order (decided per table) | 350 |
| row walks | `for k, row in T.iterrows():` over a one-level frame, `row["c"]` read as the stored field of the kind pandas gives it, every load of `row` inside the loop | 120 |

That is about 820 lines in all.

## Speed I expect (per policy: a new item, the Python sum of `net_cf(t)`, the item deleted)

The figures in brackets are slice 1's own report.

- **VA_US_S** (223 ms). Its four per-step pandas cells are about 3,600 pandas evaluations per policy, each a prefix
  `.loc`, a walk of the sub-index and a keyed read. They should all go native. Line B's 5.6 ms also includes the
  native reductions (`sum(... for i in sub_ids())`), which are a later slice. I expect 6 to 15 ms.
- **IP_UK_S** (85 ms). `inception_rate` should go native. The `zip` sums over tuples (vectors) stay as they are,
  and so do the cached tuple cells. I expect 20 to 40 ms, still several times line B's 6.4 ms.
- **Term_US_S** (0.39 ms). Its `iterrows()` walk runs once per policy. I expect at most a few percent faster, and
  not slower.
- **KLV_DE_S** (0.42 ms) and **UC_FR_S** (27.0 ms). Neither has a form of this slice on a per-step path. KLV's
  `check_net_cf_resid` row and its `result_cf_annual` walk run on a frame the item computes itself, in formulas
  without parameters. I expect them unchanged within noise.
- All five equal live on every policy, at rel 1e-10 / abs 1e-7.

## Build cost I expect

- **Translate time.** VA_US_S and IP_UK_S should take 5% to 15% longer than on slice 1, because the lowered
  formulas are larger. The other three should be within noise.
- **C build time.** The runtime grows by a few inline functions, and VA and IP get larger formulas. I expect +2% to
  +8%.
- **Term_US_SEC and MYGA_US_SEC.** Generated files byte-identical to slice 1, with translate time unchanged: their
  table reads are in formulas run once per model.
