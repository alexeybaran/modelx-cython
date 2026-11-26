# R4 P3: why the original lifelib text is slower than the fork text (UC_FR_S, Dep_FR_S, Child_KR_S)

Asked: the instruction gap between upstream lifelib main's original text with a default build and the fork text
04d69db with its spec, explained cell by cell at line B's commit 856b984.

## Method

- **Compiler:** commit 856b984 (a worktree), on development modelx `r4/modelx-integration` df3ced9, pandas 2.3.3,
  CPython 3.13. Translate only. Nothing was compiled and no instructions were counted; the gap figures below are
  the coordinator's.
- **Texts:**
  - **original:** lifelib main 20d4dae.
  - **fork:** 04d69db. Its enum references are pickled under `lifelib.libraries...enums`, so the export only
    restores them with that lifelib tree importable. Without it, they silently become `None`.
  - **lean:** `r4/l1-lean-b` c98ff5c9.
- **Sample:** the fork's shipped `sample.py`, which is the same file for all three models. It runs every policy of
  the book, calls `collect_outputs` where the text defines it, and then calls every public zero-argument cell. It was
  used for every build. A second original-text build used `run.py` (one policy) as its sample.
- **What counts as native:** a per-step cell is a cell with parameters. It is native when its arguments and result
  are C numbers, its cache is not a dictionary, and the read audit records no per-step pandas read in it.
- **Weight of each cause:** calls and evaluations (cache misses) per policy, counted on the exported Python model
  for the first policy of each book. UC's first policy has 12 steps, Dep's 480 and Child's 1,200.

## Per-step cells: native / all, with cells reading pandas per step in brackets

| model | fork + spec | fork, no spec | original + fork spec | original, no spec | original, no spec, run.py sample | lean + spec |
|---|---|---|---|---|---|---|
| UC_FR_S | 78/82 [0] | 78/82 [0] | 74/82 [1] | 74/82 [1] | 61/82 [1] | 74/82 [1] |
| Dep_FR_S | 80/93 [0] | 80/93 [0] | 76/93 [0] | 76/93 [0] | 67/93 [0] | 76/93 [0] |
| Child_KR_S | 68/74 [0] | 68/74 [0] | 54/72 [5] | 54/72 [5] | 53/72 [5] | 54/72 [5] |

At 856b984 the lean text compiles exactly as the original does. Its StrEnum labels are not lowered there; line B's
head lowers them by use, see the note's section 7.

## (a) The spec, and the sample

- **The spec holds sizes only.** Each spec has exactly one key, `cells_param_size`:
  - UC: `t` 1226.
  - Dep: `t` 842, `y` 73, `z` 843, `n` 72, `y_e` 73.
  - Child: `t` 1334, `x` 121.

  There are no locked Spaces, no types and no other keys.
- **The spec changes no cell.** With a book-wide sample, it changes only the dense windows (75, 75 and 54 cells),
  and not one cell's kind or C type, on either text.
- **What a user must know to write it:** the largest value each parameter takes over the whole run. The spec files
  derive these from the product limits in comments. Without them, a call beyond the sampled range raises
  IndexError. That is a correctness matter, not speed.
- **The sample matters more than the spec.** With `run.py` as the sample instead of the book, more per-step cells
  were never traced and stay Python objects with dictionary caches: 13 in UC (e.g. `euro_interest_pp`,
  `perf_factor`), 9 in Dep (e.g. `pols_aggravation`, `pols_carence_cum`, `pols_dead_cum`) and 1 in Child
  (`cv_grade_ratio`).
- **What a user must know about the sample:** it has to reach every per-step cell, including report-only ones. A
  generic "every policy, every zero-argument cell" script does that, and the fork's script is that script.

## (b) Text rewrites in the fork

The fork changes 34, 34 and 42 functions of the original. The lean text changes 18, 13 and 23, and adds none. Each
row gives counts, one example, and whether the lean text has the rewrite.

| rewrite | UC | Dep | Child | example | lean text |
|---|---|---|---|---|---|
| 1. labels passed as Cell arguments become IntEnum members | 3 families; 4 cells become native | 5 families; 4 cells become native | 7 families; 10 cells become native | `av_pp_at(t, "BEF_WD")` becomes `av_pp_at(t, AvTiming.BEF_WD)`. On the original text, `av_pp_at(t, timing: str)` is a dictionary-cached object call on every step | yes, as StrEnum (value = label text); lowered only by line B's head |
| 2. model-point text fields become IntEnum, and tables are keyed by them | 6 families | 5 | 6 | `status()` returns `Status[...]`; `status() in (Status.partial, Status.total)` | no |
| 3. cached per-policy cells added | 2 (`uc_scenario_segments`, `mort_schedule`) | 3 (`mort_schedule`, `prevalence_schedule`, `mort_top_age`) | 8 (`av_build_*`/`av_taper_*` curve, keys, values; `inc_pivot_ages/rates`) | Child: the fork caches `av_build_keys()` and `av_build_values()` as two numeric lists; the original rebuilds `[(float(k), float(v)) for k, v in data.av_table().loc["build"]["value"].items()]` on every call of `refund_build(d)` (1,201 calls per policy) | no; it also drops IP_UK_S `inception_schedule`, as you noted |
| 4. a cell re-keyed to a smaller domain | - | `rente_pay_pp(t, z)` becomes `rente_pay_pp(y, y_e)`: 9,680 evaluations per policy fall to 820 | - | the amount depends on `t` and `z` only through two policy years | no |
| 5. `index.asof` over a re-indexed table | `uc_return_mth`: the table is re-indexed by (scenario_id, from_month) and read with `tbl.index.asof(t)` | - | - | the original reads `tbl = T.loc[[scenario]]` and then `for lo, hi, r in zip(tbl[...], ...)` on every step | no (and line B's head no longer lowers asof) |
| 6. a column-keyed read instead of a Series held by a cell | - | - | `mort_series(s)` returns the frame and is read `.loc[x, "mort_rate"]` | the original's `mort_series(s)` returns `T.loc[s]["mort_rate"]`, read per step by `.loc[x]` | no |
| 7. others | | `claims(t, kind=None)` becomes a required `kind` with `Kind.ALL`; `"share_" + kind` becomes branches | `inc_pivots` list of tuples becomes two lists | | `claims` only, as a StrEnum |

**Per-policy cached cells (row 3) are not needed for the fork's speed when the compiler already holds the key per
item.** The original's `data.mort_table().loc[(sex(), x), "mort_rate"]` is already a native read at 856b984, because
`sex()` is a per-item key part. Row 3 matters only where the original builds a Python list per call, as Child does.

## (c) Forms in the original text that no layer lowers at 856b984 (per step)

- **A sub-table chosen by a list key and walked with `zip`:** UC `uc_return_mth`, read audit "a key part selects
  several rows". This is UC's only per-step pandas read.
- **A Series held by a cell and read with `.loc[x]`:** Child `mort_rate_at_age`, 1,436 calls per policy, read audit
  "no rule recognises this form".
- **A label sub-table, then a column, then `.items()`, built on every call:** Child `refund_build` and
  `refund_taper`, 1,201 calls each, read audit "the row key is not complete".
- **Strings as Cell arguments, which make a dictionary cache with object keys:**
  - UC: 132 calls per policy over 12 steps (`av_pp_at`, `pols_if_at`, `claims`).
  - Dep: 9,120 calls (`carence_months` 1,440, `prev_param` 2,400, `severity_share` 1,920, `claims` 3,360).
  - Child: about 153,000 calls (`inc_rate_mth` 29,891, `claims` 27,623, `frac_open` 17,941, ...).
- **A list of tuples built from a Series and searched in a Python loop on every step:** Child `inc_pivots` /
  `inc_rate_at`, 13,145 calls.

## Attribution (by reading the counts; not measured)

- **Child_KR_S (181.3M against 12.7M):** about 153,000 dictionary-cached calls with string keys per policy, plus
  about 3,800 per-step pandas reads and list builds (rows 1, 3 and 6, and the forms in (c)). At about 1,000
  instructions per Python-level call this is the right order for the whole gap.
- **UC_FR_S (117.9M against 0.37M):** the per-step `.loc[[...]]` plus zip on pandas in `uc_return_mth` (rows 3 and
  5), then the string-keyed `av_pp_at` and `pols_if_at` (row 1). Most of the gap is the pandas read.
- **Dep_FR_S (238.3M against 22.0M):** with a book-wide sample, the counted causes are 9,120 string-keyed calls and
  8,860 extra evaluations of `rente_pay_pp`. That is roughly 20-30M, about a tenth of the gap.
  - The likely remainder is the default build's sample. If it was `run.py`, nine more per-step cells (cumulative
    `pols_*` recursions among them) are untyped Python objects.
  - Cheapest check: build the original text twice, with the book sample and with `run.py`, and count instructions
    per policy as you did.

## Smallest conventions plus spec that should bring the original text within 2x of the fork (estimated)

| # | convention or spec content | what it removes | could a compiler do it automatically? |
|---|---|---|---|
| 1 | a sample that reaches every per-step cell (every policy, every zero-argument cell) | Python object cells that were never traced (Dep 9, UC 13 with run.py) | yes: the fork's `sample.py` is generic, and the compiler could ship and run it by default |
| 2 | labels passed as Cell arguments are enum members, or plain string literals from a closed set | every string-keyed dictionary cache (UC 3, Dep 4, Child 10 cells) | yes, for string literals: line B's enum rule by use, applied to a parameter whose every caller passes a literal from a closed set (the text plus the sample); the codes would be internal and the strings stay at the Python boundary. Not built |
| 3 | no per-step pandas object held by a cell: read `T.loc[key, "col"]`, or cache a per-policy sub-table cell | UC `uc_return_mth`, Child `mort_rate_at_age`, `refund_*` | partly: hoisting a per-item-invariant `T.loc[k]` / `T.loc[[k]]` into a held sub-table is B6's prefix machinery; lowering `zip`/`.items()` walks over it and a Series held by a cell are new forms |
| 4 | interpolation tables as two per-policy numeric lists, not a list of tuples rebuilt per call | Child `inc_pivots`, `refund_*` loops on objects | hoisting the invariant list into a per-item cache: yes (Data reads are cached, hence pure); splitting a list of pairs into two vectors: new work |
| 5 | spec `cells_param_size`: the largest value of each parameter over the run | nothing for speed with a book-wide sample; IndexError beyond the sampled range | no: it needs the product's limits, though a book-wide sample lowers the need |

The Dep re-keying (row 4 of (b)) and the asof form are not in the smallest set. The re-keying is a mathematical
property of the formula, which no compiler can supply. Asof is replaced by convention 3.

Expected outcome if conventions 1-4 hold (not measured): Child and UC within 2x of the fork; Dep within 2x only if
the remainder above is the sample. The cheapest confirmation is to build Child_KR_S with convention 2 applied by hand
(17 label edits) and count instructions per policy.
