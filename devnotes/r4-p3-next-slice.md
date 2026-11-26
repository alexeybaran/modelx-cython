# Choosing the next slice: what still runs on pandas or as Python objects in the eleven slowest models

This is analysis only; no code changed. I translated and compiled the eleven models of the census's tail on this
branch (09482aa): BU_DE_S, VA_KR_S, Cancer_JP_S, Dep_FR_S, UC_FR_S, FXWholeLife_JP_S, CI_KR_S, LTC_KR_S, IP_UK_S,
IUL_US_S and Child_KR_S. The text is upstream lifelib main 20d4dae, with the default build.

## Method

- **Sample.** Every policy of the shipped book: each public zero-argument cell, then `net_cf(t)` for every t.
- **Specs.** FXWholeLife_JP_S and CI_KR_S raised IndexError beyond their sampled windows without a spec. They were
  built with `cells_param_size` only: `t`, and `y` or `s`; CI_KR_S with negative lower bounds.
- **What was counted.** callgrind instructions while running one pass over the book: per policy a new item, then
  `sum(net_cf(t))`, then the item deleted. Collection is on only inside `net_cf`.
- **Attribution to cells.** The cost of the interpreter, dispatch and builtin functions is folded into the calling
  function (`--fn-skip`). Each cell gets its formula's own instructions plus every non-cell callee: pandas, the
  interpreter, the runtime. Per model the cells add up to 100-110% of the total. FXWholeLife_JP_S adds up to 200%,
  because `claims` is counted with `lapse_rate_base` inside it.
- **Evaluations per policy.** Calls of the cell's formula, from the same profile.

The totals per policy are close to the census's: VA_KR_S 225M (census 207.5M), Cancer_JP_S 1,342M (1,474M), UC_FR_S
126M (118M), LTC_KR_S 56M (59M), IUL_US_S 101M (107M), Child_KR_S 171M (196M). Lower here: BU_DE_S 436M (634M),
Dep_FR_S 588M (873M), IP_UK_S 95M (123M), FXWholeLife_JP_S 461M (422M). CI_KR_S is 370M (155M); its build has the
spec and the sample problem below.

## Where the remaining cost is (cells over 2% of a model's instructions)

| model | cell (evaluations / policy) | share | form | group |
|---|---|---|---|---|
| BU_DE_S | `dis_cohorts_first` (352), `dis_cohorts` (338), `dis_exits_first` (351), `dis_exits` (337) | 88.5% | cells returning `(list, list)` of cohort values; loops over them; `object` result, dict cache | b |
| VA_KR_S | `lapse_rate` (896), `risk_prem_rate` (17) | 70.1% | `rows = tbl[tbl["dur_from"] <= x]`, then `rows.iloc[-1]["c"]` (a mask, then the last row) | d |
| VA_KR_S | every per-step cell | (225M to 190M) | typed `object`, "formula not observed by the sample" (see g) | g |
| Cancer_JP_S | `hosp_stay_days` (843) | 98.1% | `tbl[(tbl["basis"] == b) & (tbl["age_from"] <= age(t))].sort_values(...)["days"].iloc[-1]` | d |
| Dep_FR_S | `dep_cohorts` (486), `instalments` (486) | 65.8% | a cell returning four lists; Python loops over them | b |
| Dep_FR_S | `claims` (1,942) | 28.3% | `claims(t, kind)` over string kinds; a dict built per call; a loop over the cohort lists | c, f, b |
| UC_FR_S | `uc_return_mth` (138) | 98.1% | `tbl = T.loc[[k]]`, then `zip(tbl["from_month"], tbl["to_month"], tbl["r"])` | d (with the zip form) |
| FXWholeLife_JP_S | `lapse_rate_base` (710) | 97.5% | `T.xs(shape(), level="shape")`, then `.index.max()` and `.loc[y, "c"]` | d |
| CI_KR_S | `pols_if_ci_at` (49,824), `resid_db_pp` (49,824), `lapse_rate_base`, `ci_surv`, `claims`, `pol_val_pp`, ... | ~100% | every top cell `object` with a dict cache: not observed by the sample | g |
| LTC_KR_S | `care_surv(s, t)` (recursive) | 60.4% | a two-parameter recursion on a hashed numeric cache | f |
| LTC_KR_S | `claims` (3,955), `share_ge_at(grade, x)` | 19.2% | string `kind` / `grade` parameters | c |
| LTC_KR_S | `av_ratio_at`, `share_ge_at` | 9.6% | `sorted(...)` lists built per call, Python search loops | b |
| IP_UK_S | `sick_cohorts` (279) | 37.2% | a tuple cell (one value per claim duration) | b |
| IP_UK_S | `pols_sick_surv` (279), `pols_recovery` (202) | 46.2% | `sum(a * b for a, b in zip(v, surv) if a != 0.0)` over tuples | a with b |
| IP_UK_S | `inception_rate` (278) | 5.6% | native view since slice 2; left: `max(a for a in ages if a <= x)` over the walked list | a |
| IUL_US_S | `av_pp_at(t, timing)` (4,560), `fa_pp_at` (5,472), `claims` (2,736) | 34.8% | string `timing` / `kind` parameters, dict caches | c |
| IUL_US_S | `seg_bal_pp(t, m)` (9,962), `seg_ded_cum_pp`, `seg_wd_cum_pp`, `seg_ded_pp`, `seg_share` | 40.2% | two-parameter cells on hashed numeric caches | f |
| Child_KR_S | `claims` (12,756), `inc_rate_mth` (9,549), `inc_rate` (11,669), `benefit_pp` (7,441), `frac_open` (4,248), `cover_open` (4,241), `inc_rate_at` (964) | ~90% | every one takes a string `cause` / `kind` / `cover`: dict caches keyed by text | c |
| Child_KR_S | `refund_build` (1,063), `inc_rate_at` | 18% (overlaps the row above) | a list of pairs built per call (`[(float(k), float(v)) for k, v in tab.items()]`), then searched in a Python loop | b |

## By group

| group | models (of 11), with their share | share of each model's remaining cost | line B 856b984 | lines to port |
|---|---|---|---|---|
| (a) generator sums / reductions, incl. the zip form | 2: IP (46% with b, 6% alone); UC's zip is part of its (d) site | IP 52% | lowers `sum` / `min` / `max` over generators (B2, `reductions.py`). The zip over tuples needs vectors too. UC's `.loc[[k]]` + zip stays pandas there | ~1,250 (B2: `reductions.py` 942, `_float_sum_state.pxd` 233, `_native_streams.pxd` 28, hooks) |
| (b) vectors: tuple or list cells, loops over them | 5: BU 88%, Dep 66% (+ part of `claims`), IP 37% (83% with a), Child ~18%, LTC ~10% | as listed | lowers BU's and IP's (60 and 67 vector calls on this text, census). It does not lower Dep's, Child's or LTC's: 0 vector calls on this text | ~3,140 (B3: `vectors.py` 2,005, `_native_vectors.*` 570, hooks); needs B2 |
| (c) text labels as parameters (`claims(t, "DIAG")`, `timing`, `cause`) | 9 over 1.5%; 5 over 5%: Child ~90%, IUL 35%, Dep 28%, LTC 19%, CI 7%; also VA_KR, IP, BU, Cancer_JP 2% | as listed | no. B1 lowers enum members, and StrEnums only on the lean text. String literals of the original text stay dict-keyed there too | not portable: new work, a rule for a parameter whose every caller passes a literal from a closed set (B1's `enums.py` 339 + ~155 hooks as a base) |
| (d) mask, list-keyed or partial-key reads | 4: Cancer_JP 98%, UC 98%, FX 98%, VA_KR 70% | as listed | no: masks, `sort_values`, `.loc[[k]]` and `xs` all stay pandas there | new work on the prepared table. The pieces: a step lookup "last row whose column <= x" (VA_KR, Cancer_JP, with an equality filter), `xs(k, level=L)` as a view (L the first level: about 40 lines on slice 2's views), and `.loc[[k]]` with a zip of its columns (the zip entry points of 856b984 are a start) |
| (e) held pandas objects | 0 over 2% among these cells | - | - | - |
| (f) dict caches and plain Python structures | 3: LTC 60% (`care_surv(s, t)`), IUL 40% (`seg_*(t, m)`), Dep (the per-call dict in `claims`) | as listed | no: line A removed line B's dense-by-use sizing | try a `cells_param_size` tuple spec first (a dense 2-D cache) before any compiler work |
| (g) other: cells never typed | 2: CI_KR ~100%, VA_KR 16% | as listed | not a port | sample and tracer (below) |

## The two models whose cells are never typed

In VA_KR_S and CI_KR_S the build report says "formula not observed by the sample" for most per-step cells, which then
compile as `object` with dict caches.

- **The cause.** The cells are reached only through deep recursion (`pols_if(t - 1)`, cohort chains). Under the
  tracer that recursion exceeds Python's limit. Tracing adds frames.
- **What happens then.** The RecursionError unwinds the calls, so they are not recorded.
- **VA_KR_S, ascending sample.** With a sample that runs `net_cf(t)` in ascending t before anything else, VA_KR_S
  types every per-step cell. It drops from 225M to 190M per policy, and `lapse_rate` (the mask read) is then 75% of
  it.
- **CI_KR_S, ascending sample.** The build still stops with RecursionError inside the tracer (`tracer.py:747`).
- **The fix.** A sample convention, or a raised recursion limit while the sample is traced. Neither needs porting.

## Calls from Python into cells with table reads (BasicTerm_SE)

I built BasicTerm_SE with line A (90da50b) and with this branch, with the same sample (`Projection[1..10].pv_net_cf()`).
I counted instructions with callgrind: 500,000 calls against none, the difference divided by the calls.

| measure | line A 90da50b | this branch |
|---|---|---|
| a cached call of a per-step cell from Python, `p.net_cf(5)` | 1,115 | 1,106 |
| a cached call of a zero-argument cell, `p.premium_pp()` | 966 | 947 |
| a new item, `pv_net_cf()`, the item deleted, per policy | 784,613 | 861,380 (+9.8%) |

- **Cached calls cost the same.**
- **Fresh evaluation costs more, and the cause is slice 1.** Slice 1 and this branch generate identical code for
  BasicTerm_SE.
  - `model_point()` is a Series the item computes. Under slice 1's rule it is a receiver in any formula with
    parameters.
  - So `duration_mth(t)` and `pols_new_biz(t)` bind it, and prepare a table from it per item, to read one field with
    a literal key in their `t == 0` branch.
  - Those reads run once per policy. The preparation adds about 84,000 instructions to `duration_mth` per policy: the
    whole difference.
  - Both formulas also leave the overflow flag for the `overflowcheck` fallback (5 such formulas on line A, 7 here).
- **A fix for the table layer, not done here.** Do not prepare a frame or Series the item computes for reads whose
  key parts are all fixed per item. Hold the read value per item, or leave it to pandas. A row read once per item does
  not pay for a preparation.

## Next slice by counted need

1. **(d) mask, list and `xs` reads.** Four models, each at 70-98% of its remaining cost: Cancer_JP_S, UC_FR_S,
   FXWholeLife_JP_S, VA_KR_S. Line B has none of it, so it is new work, but small pieces on the prepared table and on
   slice 2's views. I expect the largest gain per line: 3x to 40x on those four.
2. **(a) + (b) reductions and vectors.** Port B2 then B3 from 856b984, about 4,400 lines. This is for BU_DE_S and
   IP_UK_S (88% and 83%). Dep_FR_S's cohort lists are not lowered by line B on this text either.
3. **(c) text labels.** New rule. It is the largest single cost in Child_KR_S (~90%) and IUL_US_S (35%), and it is
   in five models over 5%.
4. **(f) and (g).** Try a spec and the sample order first (LTC_KR_S, IUL_US_S, CI_KR_S, VA_KR_S); no compiler work
   until those are measured.
5. **The receiver rule from the BasicTerm_SE check**, in whichever slice touches the table layer next.
