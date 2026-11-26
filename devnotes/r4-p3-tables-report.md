# Table reads on the fast line: report

Branch `r4/p3-tables` = `r4/p2-collect-3` (90da50b) + five commits. The port is from `r4/integration` 8c03194, and it was
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
