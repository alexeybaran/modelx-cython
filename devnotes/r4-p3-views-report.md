# Views of a prefix and row walks on the fast line: report

Branch `r4/p3-views` = `r4/p3-tables` (e29d599) + six commits. The port source is `r4/integration` at 856b984. I read
its code and ran its translator on the models to see what it lowers. Nothing came from a later commit. I wrote the
expectation before the first measurement: `r4-p3-views-expectation.md`.

Stack: CPython 3.13.18, Cython 3.3.2, pandas 3.0.5, numpy 2.5.3, modelx 0.33.0 (upstream), lifelib main 20d4dae
(upstream text, shipped books). The machine has 4 vCPUs.

## What came (lines against e29d599)

| commit | pieces | non-test | tests |
|---|---|---|---|
| views (33130c3) | `v = T.loc[prefix]` at the top of a formula: prefix codes and the view's stored rows in C locals, held per item when every part is fixed per item; `v.loc[k, "c"]` / `v.at[k, "c"]` of float64, int64 and object columns; `float(v.loc[k, name])`; `v.index.max()` / `.min()`, `max(v.index)` / `min(v.index)` | +212 / -10 | `test_forms_views` |
| index walks (b014f51) | comprehensions (list, set, dict, and a generator consumed at once by a builtin) and `for` loops over `X.index`: a view local, a whole frame, an inline prefix, a local holding an index; `list` / `tuple` / `set` of an index; runtime `pos_row`, `walk_within`, `label_obj`, `whole_row`, signature kind -7 | +305 / -15 | `test_forms_walks` |
| column walks (60ab13c) | a float64 or int64 column of a view or prefix (`v["c"]`, `T.loc[p]["c"]`, `T.loc[p, "c"]`, or a local of one): values and `.items()` walked, `.tolist()`, `list()`, a walk of its index, `C.iloc[i]`, `C.loc[k]`; runtime `position` | +143 / -16 | `test_forms_column_walks` |
| row walks (b354d63) | `for k, row in T.iterrows():` over a one-level frame, `row["c"]` as a row local's field | +72 / -3 | `test_forms_iterrows` |
| collect half (4d258c3) | test only | 0 | `test_views_collect` |
| total | | +716 / -28 against e29d599 | 5 files, 369 lines |

The expectation was about 820 lines. The column walks were not in it. I added them after line B's translator showed
that it lowers them in four models (EC_FR_S, Medical_KR_S, Medical_JP_S, Child_KR_S).

## How it was measured

Every model is exported from lifelib main with `mx.read_model(...).export(...)`, and the product's CSVs sit beside
the export. Each tree builds with the default build: `--no-spec` and one generic sample. The sample visits every
policy of the shipped book and calls each public zero-argument cell of its item. The same export and sample are used
for all three trees:

- upstream d22426f;
- slice 1 (e29d599);
- this branch (4d258c3).

The SEC models use their own `sample_sec.py` and `spec_sec.py`. Because the sample differs from slice 1's, the
slice-1 column below is re-measured, not copied from its report.

- **Per policy.** Each policy is a new item, then the Python sum of `net_cf(t)` over `proj_len()`, then `del
  Projection[pid]`. A round is one fresh process that walks the book several times. The time of a round is the mean
  per policy after the first pass, which is a warm-up. The figure is the median of 5 interleaved rounds (11 rounds
  for KLV_DE_S and UC_FR_S on slice 1 and this branch, whose spread is wide).
- **Build time.** The median of 3 interleaved builds, two at a time, as slice 1 measured.
- **Values.** On every policy, `net_cf(t)` at every t and every value of each zero-argument `result*` frame are
  compared with live modelx (rel 1e-10 / abs 1e-7).

## Values

All five models equal live on every policy, with no difference: KLV_DE_S 14 policies, Term_US_S 2, UC_FR_S 11,
VA_US_S 9, IP_UK_S 7. Every `net_cf` sum is bit-identical between slice 1 and this branch.

Term_US_SEC and MYGA_US_SEC give byte-identical generated files on slice 1 and on this branch: the sha256 of every
.py/.pxd/.pyx/.json, 9 and 11 files.

The full suite passes after each commit. On this branch's head: 582 passed, 15 skipped.

- **Skipped for want of a modelx with `set_collect`:**
  - `integration_tests/test_views_collect.py::test_views_and_walks_through_collect` (new);
  - `integration_tests/test_table_reads_collect.py::test_table_reads_through_collect`;
  - the four tests of `modelx_cython/tests/compiled/test_collect.py`.
- **Skipped for other reasons:** seven 'main' samples, one test that needs match/case with Cython 3.2, and one that
  needs exported dynamic References.
- **The collect test's model, run on the direct route.** This route has no collect. On all 30 policies the model
  equals live bit for bit, and its generated code has every pattern the test asserts.
- **Environment.** This fresh container lacked the `lifelib` and `psutil` that the lifelib sample tests import. They
  fail the same way on slice 1 without them. Installing lifelib main and psutil made them pass. Nothing in the
  repository was changed for this.

## Speed per policy (ms)

| model | upstream d22426f | line A (slice 1 report) | slice 1 e29d599 | this branch | line B 856b984 (given) |
|---|---|---|---|---|---|
| KLV_DE_S | raises IndexError | 111.5 | 0.427 | 0.428 | 0.47 |
| Term_US_S | 107.6 | 108.4 | 0.388 | **0.298** | 0.30 |
| UC_FR_S | 57.4 | 61.3 | 26.9 | 26.4 | 27.7 |
| VA_US_S | raises IndexError | 356 | 210 | **6.65** | 5.6 |
| IP_UK_S | 135.0 | 169 | 79.3 | **7.00** | 6.4 |

- **Upstream d22426f.** Its compiled KLV_DE_S and VA_US_S raise IndexError ("array index out of range") in this
  work, from a dense cache window beyond the range the sample reached (`res_guar_pp`, `depleted_flag`). Slice 1 and
  this branch do not.
- **VA_US_S** is 32 times faster than slice 1 and 19% behind line B. Its four per-step pandas cells are now native.
  Line B also has the native reductions (`sum(... for i in sub_ids())`, a later slice).
- **IP_UK_S** is 11 times faster than slice 1. With views and walks alone it is 9% behind line B, which also has
  vectors and reductions. `inception_rate` was almost all of IP's time.
  - What is left is the Python-level generator sums: `max(a for a in ages if a <= x)` over the walked list,
    `sum(a * b for a, b in zip(v, surv))` over tuples.
  - Native, those are the vector and reduction slices.
- **Term_US_S** is 23% faster. Its once-per-policy `iterrows()` walk is native, and it is level with line B.
- **KLV_DE_S and UC_FR_S** are unchanged within noise, as expected.
  - UC's remaining per-step pandas read is `T.loc[[k]]` plus `zip` (a later slice). On 856b984 it stays pandas too.

## Build time (seconds, translate / C build, median of 3)

| model | upstream d22426f | slice 1 | this branch | translate vs slice 1 | C build vs slice 1 |
|---|---|---|---|---|---|
| KLV_DE_S | 16.2 / 44.0 | 28.2 / 69.9 | 28.6 / 72.0 | +1% | +3% |
| Term_US_S | 4.0 / 20.7 | 8.9 / 40.2 | 9.5 / 38.3 | +7% | -5% |
| UC_FR_S | 8.6 / 37.9 | 20.2 / 60.1 | 19.7 / 58.9 | -2% | -2% |
| VA_US_S | 30.1 / 78.4 | 48.0 / 116.5 | 48.9 / 119.7 | +2% | +3% |
| IP_UK_S | 10.2 / 28.3 | 15.3 / 47.0 | 15.5 / 50.2 | +1% | +7% |

No model is near the 30% limit over slice 1. Against upstream, translate time is still the slice-1 cost: +52% (IP)
to +137% (Term). Translate times of the SEC models are unchanged (median of 3, slice 1 / this branch):

- Term_US_SEC: 12.7 / 12.4 s.
- MYGA_US_SEC: 37.8 / 37.7 s.

## Build-cost idea from slice 1 (a note only, not done)

Slice 1 proposed emitting one reference to the held table per formula, instead of a cast of the item field
`_mx_cy.cast(<helper>.PreparedRef, self._mx_tp_<n>)` in every read.

- **What the cast costs.** In VA_US_S's generated `_mx_classes.py` the cast appears 164 times on this branch (118
  on slice 1), 10.3 kB of 628 kB (1.6%).
- **Where most of the growth is.** It is in the pandas fallback branches, which every lowered read, walk and view
  carries whole. One per-formula local would save little.
- **The larger lever.** A fallback that calls the formula's own unlowered copy (one function per formula, not one
  pandas expression per read) would shrink both the generated code and the input of every later pass.
- **Not done.** It changes how exceptions are placed, and it needs its own proof that results are live's. That is
  not this slice.

## What still reads pandas, per model, after this slice

The list comes from the generated code: pandas operations outside the pandas branch of a lowered read.

- **KLV_DE_S.** Per step: nothing. Per item: `model_point` (the row cell itself), `result_cf_annual` (a walk of a
  frame the item computes, in a formula without parameters).
- **Term_US_S.** Per step: nothing. Per item: `model_point`, `result_cf_annual`, and `shock_lapse_rate`'s
  `iloc[-1]` after its loop, which runs only when no bucket matches. Its `iterrows()` walk is native.
- **UC_FR_S.** Per step: `uc_return_mth` (`T.loc[[k]]`, then `zip` over its columns). Per item: `model_point`.
- **VA_US_S.** Per step: nothing; on slice 1 it was `inv_return_mth`, `scenario_rate`, `gawa_pct_at_age` and
  `surr_charge_rate`. Per item: `model_point`. `sub_ids` (`list(T.loc[p].index)`) is now native.
- **IP_UK_S.** Per step: nothing; on slice 1 it was `inception_rate`. Per item: `model_point`.

## Rules kept

- **No read recorder.** Every decision comes from the text and the traced schema. Each one is guarded at run time.
  If a guard fails, the pandas expression runs, or for a walk the whole pandas comprehension or loop runs.
  - **The table.** It must be of the kinds the read was compiled for (its slot). Walks of a prefix also need the
    stored rows of each prefix to be in pandas order, decided once per table (signature kind -7, `walk_within`).
  - **The view.** It must exist (start >= 0).
  - **Key parts.** A part not known to be one label must be one label at run time (`is_label`).
  - **Columns.** They must be of the traced kind.
- **Exceptions are live's.**
  - **A missing prefix.** When a view is not native, its assignment is evaluated on pandas where it stood. A missing
    prefix therefore raises KeyError there, before any later statement (tested: a ZeroDivisionError after the view
    is not raised first). When a use of the local is left on pandas, the assignment stays as written.
  - **A missing key** in a view read raises KeyError.
  - **`C.iloc[i]` out of bounds** raises IndexError with pandas' message.
- **Live's types.**
  - **Walks** give the Python scalars pandas iteration gives. That is a C integer for an int level, boxed to `int`,
    and the stored label object otherwise.
  - **`iterrows()` rows.** pandas builds them from `DataFrame.values`, so a row with an object column has Python
    numbers. Those fields are never boxed as numpy scalars, also under `type()` / `isinstance()`. On a row of number
    columns they are numpy's, as for `T.loc[k]`.
  - **`C.iloc[i]` / `C.loc[k]`** are numpy's where the value meets numpy's rules.
- **Borrowed references.** A generator over an index is walked natively only when it is the argument of a builtin
  that consumes it at once (`sorted`, `max`, `sum`, ...). The table is read through a borrowed reference, so a
  generator kept for later stays pandas.
- **Integer guard.** Every new runtime function that returns a C number is registered: `level_code_at`,
  `whole_row`, `position`. `label_obj` returns an object. `test_provider_returns` passes.
- **Line A's and slice 1's tests are untouched and pass.**
  - Slice 1's `test_forms_extremes` caught a doubled lowering during this slice. `max(y for s, y in T.index)` is
    slice 1's level extreme, and its generator was also being walked. The fix is in the walk commit.

## Known differences (loud, documented, not fixed)

- A KeyError from a view read lists the parts found of the whole key, for example `('M', None)`. pandas shows the
  view's own key. The type is pandas'.
- When a view is not native, each pandas fallback evaluates `T.loc[prefix]` again. The values are the same, because
  the parts are cached Cells or the formula's own values. The cost is one pandas `.loc` per use, and only on the
  pandas path.

## Left out, by the census

I checked these against a text scan of all 57 regional models (upstream text) and against what line B's 856b984
translator lowers on 27 of them, the ones with sub-table or walk sites. The 27 include VA_US_S, IP_UK_S and Term_US_S,
each run with the generic sample.

| left out | models using it | lines saved (about) |
|---|---|---|
| prefix cells: a cell returning `T.loc[prefix]` whose value another formula reads as a view (`self.P()`, `self.P(a)`, `P.index.asof(x)`) | 0: every `return T.loc[...]` cell is a row cell read by field; line B lowers none on this text | 150 |
| `v.index[i]`, `searchsorted`, `asof` through a view (`search`, `asof_at`) | 0 (no `asof` or `searchsorted` in the text; `.index[` only with a mask) | 120 |
| lexicographic extremes of several remaining levels of a view | 0 | 25 |
| `len(X.index)` | 0 | 10 |
| a view of a view (`v.loc[more]` as a view) | 0 | 30 |
| a whole-frame column walked (`T["c"]`, `.items()`) and object columns walked | 0 (the four models walk float columns of a prefix) | 30 |
| `iterrows()` over a frame of several levels | 0 | 10 |
| a range held packed in one 64-bit word (`view_range`) | none needed: start and stop are held as two fields, with no platform-width dependence | 15 |

Not in this slice:

- **`zip` entry points.** UC_FR_S `uc_return_mth` (`T.loc[[k]]` then `zip` of its columns) and Obseques_FR_S
  (`zip(tbl.index, tbl["surr_value"])`) need what 856b984 has and 8c03194 removed:
  - `zip_prepared_prefix(prepared, frame, accessor, key, signature)`, with `_zip_axes` kept at preparation;
  - `_zip_slot` in `_decide`;
  - the held string selector of the zip walk.

  Line B numbers its zip signature kind -7. Here -7 is the walk order, so the zip slice needs another number.
- **Vectors and reductions.** IP_UK_S and VA_US_S need `sum(... zip(v, rec))` over tuples and generator
  reductions over walked lists.
- **Labels.** Not in this slice.

No deletion in this slice blocks those slices. Nothing of slice 1's runtime was removed. `label_ll` now reads its
code through `level_code_at`, which gives the same values.
