# Table reads on the fast line: report

Branch `r4/p3-tables` = `r4/p2-collect-3` (90da50b) + eight commits (the second round, below: ordered lookups and a
runtime without memoryviews). The port is from `r4/integration` 8c03194, and it was
also compared with 856b984. Expectation written before the first measurement: `r4-p3-tables-expectation.md`.
Stack: CPython 3.13.18, Cython 3.3.2, pandas 3.0.5, numpy 2.5.3, modelx 0.33.0 (upstream), lifelib main 20d4dae
(upstream text). The machine has 4 vCPUs.

## What came (non-test lines, against 90da50b)

| commit | pieces | non-test | tests |
|---|---|---|---|
| table base | `_tables.pyx/.pxd` (preparation, a slot per read signature, level codes, keyed reads, numpy boxing, the name map of run-time columns); `frame_reads.py` (receivers by traced type, binding per item, holds of key codes, L1 reads, object columns, `float(T.loc[k, name])`); `key_facts.py`; frame schema (tracer), resolver, hold kinds, transformer and cli hooks; spec keys `prepared_tables`, `strict_lowering`; `_TABLE_RETURNS` registry | +2,108 / -10 | 11 files, 12 tests |
| held rows | `T.loc[k]["c"]`, row locals, row cells' fields (no inliner), run-time field under `float()`, rows held per item | +385 / -13 | 4 files, 10 tests |
| extremes and membership | `T.index.max()/min()`, `max(T.index)`, the extreme of a prefix's int level, `get_level_values(L).max()`, `max(a for a, s in T.index)`, pandas' extreme label of any other index; `k in T.index` with a complete key | +399 / -5 | 2 files, 2 tests |
| collect half | test only | 0 | 1 file, 1 test |
| total | | +2,874 / -10 | 18 files, 25 tests, 1,207 lines |

This is 2,874 lines, against about 2,800 expected and 3,600 allowed. The table-cell API of line B and the recorder are not
ported (see below).

## Interfaces resolved

- **Header install.** `_tables` is a .pyx/.pxd pair, installed by `native_sources.install` as the numeric caches are. No
  header-only install is needed in this slice.
- **Run-time import.** No compiled module imports `modelx_cython.runtime`. The runtime is the package's own copy
  (`_mx_helper_<hash>`). It loads its read signatures from the package's `_mx_table_signatures.py`. Line A's
  no-modelx_cython-at-run-time test is unchanged and passes. `test_table_cells_api` runs the compiled table model with
  `modelx_cython` blocked and asserts that no `modelx_cython` module is loaded.
- **Integer guard.** Every C-number function of `_tables.pxd` is listed in `_TABLE_RETURNS`. `_PROVIDER_RETURNS` is
  unchanged, because line A's `test_table_matches_the_pxd` pins it. `test_provider_returns` parses every native header and
  fails when a number-returning function is unlisted or listed with the wrong kind. The numeric caches' unsigned words are
  listed in `_WORD_RETURNS`.
- **Collect reset.** The hold fields are `object` and `Py_ssize_t`, and a field is unbound at zero. Line A's reused receiver
  resets objects to None and C values to zero, so a reset item binds again.

## Models (upstream text, shipped book)

KLV_DE_S has 14 policies and Term_US_S has 2. On every policy, every value of `result_cf`, `result_cf_annual`,
`result_surplus` and `result_pols`, and the `net_cf` total, equal live at rel 1e-10 / abs 1e-7. The generated files of
Term_US_SEC and MYGA_US_SEC are byte-identical to line A's (sha256 of every .py/.pxd/.pyx/.json). The full suite passes:
577 passed, 13 skipped. Line A's tests are untouched. The uslib ULSG_US_S build test, a table model, passes.

## Speed

Each policy is a new item, then the work, then the item is deleted. The figures are medians of 5 interleaved rounds, each
in a fresh process; the book is tiled ×20 for KLV and ×100 for Term. The Ir figures are instructions per policy from
callgrind: (I(4 passes) − I(1 pass)) / (3 × points), with the work being the Python sum of `net_cf(t)`.

| model | measure | line A 90da50b | this branch | line B 8c03194 | line B 856b984 |
|---|---|---|---|---|---|
| KLV_DE_S | `result_cf()` | 78.4 ms | 0.728 ms | 0.699 ms | 0.683 ms |
| KLV_DE_S | sum of `net_cf(t)` | 94.1 ms | 395 us | 368 us | 353 us |
| KLV_DE_S | Ir per policy | 402.8 M | 4.41 M | 4.38 M | 4.33 M |
| Term_US_S | `result_cf()` | 101 ms | 0.612 ms | 0.652 ms | 0.610 ms |
| Term_US_S | sum of `net_cf(t)` | 109 ms | 306 us | 293 us | 298 us |
| Term_US_S | Ir per policy | 524.2 M | 4.24 M | 4.08 M | 3.96 M |

KLV_DE_S runs 91 times fewer instructions than on line A. It is within 1% of 8c03194 and within 2% of 856b984 in Ir. Its
cash flows are all forms of this slice. Term_US_S keeps its `iterrows()` walk of the shock lapse table on pandas (a walk,
not in this slice). That walk runs once per policy, and Term_US_S is 4% behind 8c03194. On the upstream text, 856b984 is
not faster than 8c03194 for these two models. The Python call of `net_cf(t)` is in every column: about 360 calls per KLV
policy.

Translation seconds, median of 3, `--translate-only --no-spec`:

| model | upstream d22426f | line A | this branch |
|---|---|---|---|
| KLV_DE_S | 11.3 | 22.2 | 25.3 (+14%) |
| Term_US_S | 4.0 | 8.0 | 8.8 (+10%) |

The lowering itself takes 1.3 s of an 86 s profiled KLV translation. The rest of the increase is in the passes after it,
the integer guard and entry lowering, which run over the larger formulas. A pre-check that skips formulas without a
table call saved 0.2 s and was not kept. The expectation of 10% is met on Term_US_S and missed by 4 points on KLV_DE_S.

## Rules kept

- **No recorder.** Where line B used the sample's notes, two run-time tests decide instead. A read lowers only on a key
  part known from the text to be one label (a C number or a text literal); any other part gets the run-time test
  `is_label`. A list, a mask, a slice, an index, a table or an unhashable value runs the pandas expression. A held code is
  -2 for "not one label", and membership runs pandas then. Tests: a label or a mask by one expression, both sampled; a mask
  never sampled; a list, a slice and a mask held in a name; `T.index[mask] in T.index` raises TypeError as live.
- **Lowered only per step or per item.** A formula without parameters of a Space without items runs once per model, so its
  reads stay on pandas. This also keeps the SEC models byte-identical.
- **Receivers by text.** In an item class, a frame the item computes itself is a receiver only in a formula with
  parameters. A Data frame is a receiver in any formula.
- **numpy rules.** A value that meets numpy's own rules is boxed as the numpy scalar. Extremes read as C integers are
  lowered only where no such use follows. `T.index.max() ** -1` stays on pandas.

## Known differences (loud, documented, not fixed)

- An unsampled mask that reaches a read in a cell typed as a number raises TypeError, where live returns a Series.
- A frame edited in place after a compiled read prepared it is read as prepared, in that model and in any model that
  reads the same object. Line B started a new generation of prepared tables on each new model. That was dropped because no
  census model needs it: each model instance reads its own frames.

## Left out, by the census

These were checked against the generated code and the reports of line B 8c03194 for all 57 regional models on the
upstream text.

| left out | models using it | lines saved (about) |
|---|---|---|
| tuple keys held in a local and checked at run time | 0 | 40 |
| bool-literal keys of a pandas indexer (`__getitem__`) | 0 | 6 |
| run-time column on frames of one kind (`named_slot`) | 0 | 15 |
| L4 `T["c"][k]` | 0 | 25 |
| part of a multi-level extreme `T.index.max()[k]` | 0 | 30 |
| lexicographic extremes of a prefix of several levels (order kind -3) | 0 | 25 |
| membership with a partial key (`view_start` test) | 0 | 55 |
| `T.loc[a, b]` read as a row prefix (kind -5) | 0 | 15 |
| check mode `verify_inputs` | off by default | 40 |
| table-cell API: renamed C entries, `TableCell` properties, prepared-at-bind publishing, renew on a new model | not needed for speed | 300 |
| the row-cell inliner | its result equals the inliner-off result on all 57 models | 450 |
| the read recorder and every report file | the recorder was replaced by run-time tests; the reports are a later layer | 400 |

Also not in this slice:

- Extremes of a view local (`rows = T.loc[p]`, then `rows.index.max()`): TD_FR_S and FIA_US_S, one site each. They need
  the view-local part of the sub-table layer.
- Walks such as `iterrows`: Term_US_S, once per policy.

## Tests from line B

| file | layer | change |
|---|---|---|
| `forms_probe.py` | helper | `native_reads` reads the generated code instead of `_mx_table_reads.json` |
| the five `test_forms_*` | B4 | assertions on `native_reads` |
| `test_held_table_handles.py`, `test_constant_key_reads.py` | B4 | the handle is bound by `bind_ref(self.market.curve())` |
| `test_table_cells_api.py` | B4 | the rename assertions dropped; runs with modelx_cython blocked |
| `test_diag_run8_row_fields*.py`, `test_table_numpy_values.py` | B5 | renamed `test_row_fields*`; the inliner's report keys replaced by the reads in the generated code |
| `test_enum_by_type_b4.py` | B4 | not brought: it needs B1 (IntEnum References fixed in the compiled model) |
| `test_diag_run8_row_cells.py` | B5 | not brought: it tests the row-cell inliner's analysis |

New tests:

- `test_forms_several_rows_key`
- `test_forms_row_fields`
- `test_forms_extremes`
- `test_forms_membership`
- `test_provider_returns`
- `test_table_reads_collect`

## Not verified here

`test_table_reads_collect.py` is the collect half. It runs collect at one and four workers with one reused item per worker,
which resets every hold. It is skipped here: modelx 0.33.0 has no `set_collect`, and attaching the modelx fork that has it
was refused in this session. Its model was run on the direct route and equals live bit for bit. Every pattern the test
asserts is in its generated code.

Python 3.10 was not available here. 3.11 to 3.13 are installed, and the suite ran on 3.13. No construct newer than 3.8 is
used.


## Second round: ordered lookups, five models, compile time

Two commits were added after the first report.

- **Ordered lookups** (+140 non-test lines, `test_forms_asof`). `R.index.asof(x)` and `R.loc[prefix].index.asof(x)` read
  an int level stored in pandas' order. Under `int()` the result is a C integer; any other use gets pandas' numpy
  integer. pandas' own asof runs when x is NaN, x is not one number at run time, x is below the first label, the prefix
  has no rows, or the order is not increasing.
  - Only whole frames and inline prefixes are covered. A view local (`sub = T.loc[p]`, then `sub.index.asof(x)`) and the
    prefix-cell `lo = P.index.asof(x)` need the sub-table layer and were not ported.
  - `searchsorted` was not ported.
  - The upstream lifelib text has no `asof` at all. UC_FR_S's `asof` is in the fork text, which could not be attached
    here (add_repo for alexeybaran/lifelib was refused). So no model in this report exercises it.
- **Runtime without memoryviews** (+28 / -25). The header's column kinds and slots were typed memoryviews. That made every
  model module compile Cython's memoryview support (Term_US_S `_mx_classes.c`: 0 memoryview references on line A, 1,010
  with the slice). They are now raw pointers taken from each array's address. The column-name hashes are stored at the
  width of Py_hash_t on every platform. No read changed.
- **Not ported, the zip/vector entry points** that 856b984 had and 8c03194 removed: `zip_prepared_prefix`, `_zip_axes`,
  and the held string selector of the zip walk. The vector layer needs them for IP_UK_S's `sum(... zip(v, rec))`.

### Per policy (upstream text, shipped book, medians of 5 interleaved rounds)

Each policy is a new item, then `result_cf()` or the Python sum of `net_cf(t)`, then the item is deleted. Every build
equals live on every policy.

| model | line A: result_cf / net_cf | this branch | line B 856b984 |
|---|---|---|---|
| KLV_DE_S | 88.5 ms / 111.5 ms | 0.99 ms / 0.42 ms | 0.77 ms / 0.47 ms |
| Term_US_S | 94.9 / 108.4 | 0.78 / 0.39 | 0.88 / 0.30 |
| UC_FR_S | 57.4 / 61.3 | 27.3 / 27.0 | 30.3 / 27.7 |
| VA_US_S | 338 / 356 | 223 / 223 | 5.4 / 5.6 |
| IP_UK_S | 158 / 169 | 76.0 / 85.2 | 7.0 / 6.4 |

What still runs on pandas, per step (cell: form):

- KLV_DE_S: nothing.
- Term_US_S: nothing. Per item, `shock_lapse_rate` (`iterrows()` walk plus `iloc[-1]`) runs once per policy.
- UC_FR_S: `uc_return_mth` (`T.loc[[list]]`, a several-rows selection). It stays on pandas on 856b984 too, so UC_FR_S is
  27 ms on both.
- VA_US_S:
  - `gawa_pct_at_age`, `inv_return_mth`, `scenario_rate`: `sub = T.loc[prefix]`, a comprehension over `sub.index`, then
    `sub.loc[max(...), c]` (forms L3 and L8);
  - `surr_charge_rate` (`grid = T.loc[p]`, then `grid.loc[years, c]`, L3).
- IP_UK_S: `inception_rate` (`sub = T.loc[...]`, `sorted(... for a in sub.index)`, `sub.loc[lo, c]`: L3 and L8); the
  vector reductions are B3.

The next layer by measured need is the view locals and index walks of the sub-table layer (VA_US_S 40x, IP_UK_S 12x
behind 856b984), then vectors for IP_UK_S.

### Compile time (median of 3; seconds translate / C build; two builds ran at a time throughout)

| model | upstream d22426f | line A | table base | held rows | extremes | ordered lookups | no memoryviews |
|---|---|---|---|---|---|---|---|
| KLV_DE_S | 11.8 / 44.0 | 22.1 / 52.8 | 23.7 / 76.3 | 25.3 / 77.1 | 25.0 / 78.2 | 25.5 / 79.4 | 24.9 / 70.3 |
| Term_US_S | 3.7 / 20.1 | 8.1 / 24.7 | 8.3 / 41.9 | 8.7 / 45.7 | 8.6 / 45.3 | 9.1 / 45.4 | 8.8 / 37.2 |
| UC_FR_S | 7.7 / 35.1 | 15.4 / 43.3 | 17.3 / 64.9 | 18.6 / 64.0 | 18.8 / 66.5 | 19.1 / 67.6 | 18.9 / 60.2 |
| VA_US_S | 25.6 / 72.3 | 39.9 / 91.9 | 42.4 / 114.3 | 44.1 / 121.0 | 43.3 / 118.8 | 42.9 / 124.7 | 45.6 / 110.3 |
| IP_UK_S | 9.4 / 27.0 | 12.5 / 33.7 | 12.8 / 52.1 | 13.3 / 52.6 | 13.1 / 52.6 | 13.4 / 54.5 | 14.0 / 47.0 |

Translate time over line A is +9% to +23% (UC_FR_S the most), under the 30% limit. C build time over line A is
+12.5 to +18.4 s per model (+20% VA_US_S to +51% Term_US_S). The C build is the larger cost:

- the copied runtime is compiled once per model;
- the model's own classes grow (Term_US_S `_mx_classes.c` 76k to 101k lines before the memoryview change).

Term_US_SEC and MYGA_US_SEC translate times are unchanged (Term 12.7 s / 13.1 s, MYGA 38.6 s / 38.1 s, line A / branch,
median of 3), and their generated files are byte-identical.

The three largest additions to translate time, from cProfile of KLV_DE_S, line A against the branch (profiled seconds):

1. The integer guard: +7.4 s. It runs over larger formulas: each lowered read adds a conditional, four to six runtime
   calls and casts.
2. The transformer: +5.1 s. Of this, 2.2 s is the lowering itself and the rest is the later passes on the larger
   formulas.
3. Entry lowering (+2.2 s) and building the shared scope facts (+1.8 s) on the larger trees.

On whether these can use line A's shared facts:

- The lowering builds its own bindings-and-loads visitor (`key_facts._Formula`, about 0.2 s). That could be built on
  line A's shared `CstFacts` instead, but the gain is small.
- The larger cost is the size of what is emitted. One reference to the held table per formula, instead of a cast of the
  item field in every read, would shrink every later pass's input.

### Fork text and the table-forms merge

The fork text (alexeybaran/lifelib 04d69db) could not be attached here, so KLV_DE_S was not measured on it. From reading
the diff 856b984..19c3147, the candidates that slow or un-lower KLV are:

1. **Text-level key parts.** A key part on a text level that is not known to be one label is now tested at run time,
   and its read stays a Python object instead of a C double. In 856b984 those parts had no test and the read was typed.
   On the lean (fork) text, text levels whose keys are member-derived values or locals would take this path. On the
   original text KLV has none per step.
2. **Frames an item computes itself.** These are now prepared per item in every formula with parameters, decided by the
   text; 856b984 required at least 8 evaluations per item in the sample. For example `check_net_cf_resid(t)` reads
   `self.result_cf().loc[t]`, so a collect route that runs the checks prepares `result_cf` per policy (0.3 to 2M
   instructions).

Neither form is left out of this branch: both are line B 8c03194 behaviour, kept for correctness (1) and for not depending
on call counts (2).

### Tests skipped here for want of a modelx with collect

`integration_tests/test_table_reads_collect.py::test_table_reads_through_collect`, together with line A's own
`modelx_cython/tests/compiled/test_collect.py`. Both skip on `not hasattr(UserSpace, "set_collect")`.
