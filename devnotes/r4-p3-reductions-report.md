# Reductions on the fast line: report

Branch `r4/p3-reductions` = `r4/p3-views` (09482aa + the analysis note 18c6b03) + the commits below. Port source:
`r4/integration` at 856b984 (`reductions.py`, `_float_sum_state.pxd`, `runtime/_native_streams.pxd` and their tests).
Expectation written before the first measurement: `r4-p3-reductions-expectation.md`.

Stack: CPython 3.13.18, Cython 3.3.2, pandas 3.0.5, numpy 2.5.3, modelx 0.33.0 (upstream), lifelib main 20d4dae.
4 vCPUs. Absolute timings on this machine are 10-20% above those of the earlier reports. Compare columns within
this report only.

## What came (lines against 18c6b03)

| commit | pieces | non-test | tests |
|---|---|---|---|
| general reductions (fdf5a99) | `reductions.py` transplanted. Sites: `sum`, `min`, `max`, `any`, `all` of one iterable as the whole value of `return`, `x =` or `x op=`, and nested sites lifted where Python evaluates them unconditionally. Sources: a range of C bounds, a literal tuple unrolled, any other iterable walked as an object, a direct iterable. `PySum` and `F64Sum` in `_reductions.pxd`, a header of inline code installed without a .pyx. The integer guard knows the header's C-number functions (`_REDUCTION_RETURNS`). The unbound-local check leaves alone a local filled through its address. Admission keys for formulas with a site. | +1,267 / -4 | 8 files of 856b984, `test_reduction_returns` |
| table walks (e5c48a6) | `R((native) if test else (pandas))`, the generator the index walks emit, becomes two lowered branches under an `if`. Local kinds are read from the formula as written. Handlers and `:=` that the table reads insert no longer stop the lowering. C-declared locals of the reads are known | +52 / -6 | `test_forms_reductions` |
| tuples, lists, zip (19ee271) | sources evaluated once in order, then read as C doubles (every source an exact tuple or list of exact floats) or C integers (one source of ints within 64 bits), else the loop as written. Only when the element and conditions cannot run code; int-with-float comparisons are excluded | +224 / -7 | `test_reduction_sequences` |
| sum-state test (0e77a10) | test only | 0 | `test_exact_sum_state` (856b984) |
| total | | +1,543 / -17 | 12 files, 828 lines |

The tuple/list/zip path is not in 856b984. Line B lowers `zip` over its typed vectors, which are the next slice.
Here the same sources are Python tuples, read through a run-time check.

## Values

All five models equal live on every policy (rel 1e-10 / abs 1e-7): IP_UK_S 7, VA_US_S 9, KLV_DE_S 14, UC_FR_S 11,
Term_US_S 2. Every `net_cf` sum is bit-identical to 09482aa's.

Term_US_SEC and MYGA_US_SEC give byte-identical generated files: sha256 of every .py/.pxd/.pyx/.json, 9 and 11
files. Their formulas have no reduction of one iterable.

Suite: 594 passed, 15 skipped at the head. The intermediate commits: 591 and 592 passed, 15 skipped. The skips are
the same as in the views report, including the collect tests for want of `set_collect`.

**One earlier test changed.** `test_flagged_integer_arithmetic` asserted that `S.reduced`
(`sum(float(t * t * t) for t in range(top()))`) goes to the overflowcheck fallback. Its own comment named the
reason: "a nested scope: reductions are one too". The reduction is now a loop, and its `t * t * t` is flagged like
any other C integer arithmetic. The assertion now expects `['S.element']` only. Values and raises are still checked
by the test.

## Speed per policy (ms, median of 5 interleaved fresh processes, same sample and export)

| model | base 09482aa | this branch | change | sites native |
|---|---|---|---|---|
| IP_UK_S | 8.07 | **5.77** | -29% | 10 in 9 cells |
| VA_US_S | 6.52 | 5.96 | -9% | 9 |
| KLV_DE_S | 0.461 | 0.463 | noise | 15 |
| UC_FR_S | 31.5 | 29.0 | noise (spread 25-34) | 7 |
| Term_US_S | 0.395 | 0.373 | noise | 0 |

IP_UK_S, as expected:
- The zip sums of `pols_sick_surv`, `pols_recovery` and `pols_death_sick` now run over C doubles.
- `inception_rate`'s `max`/`min` over the walked list run over C integers.
- **Flagged for later measurement:** `pols_sick`'s `sum(sick_cohorts(t))`, a direct iterable, is line B's object
  loop as transplanted. It may be slower than the built-in `sum` over a tuple. The sequence path could cover it
  with a few lines; I left it as the source has it.

UC_FR_S is not taken any further. Its cost is `T.loc[[k]]`, then `zip` of the selected frame's columns. That is a
list-keyed pandas read, not a reduction. The zip entry points of 856b984 (`zip_prepared_prefix`, `_zip_axes`,
`_zip_slot`) were not ported. They need signature kind -7 renumbered (-7 is the walk order here).

## Build time (seconds, median of 3 translates; C build one run each)

| model | translate base / branch | C build base / branch |
|---|---|---|
| IP_UK_S | 19.4 / 20.6 (+6%) | 54.3 / 54.4 |
| KLV_DE_S | 34.1 / 37.7 (+11%) | 80.9 / 80.7 |
| UC_FR_S | 24.1 / 24.1 | 68.2 / 68.0 |
| VA_US_S | 57.6 / 58.9 (+2%) | 139.1 / 139.8 |
| Term_US_S | 11.0 / 10.7 | 42.9 / 43.2 |
| Term_US_SEC | 15.8, 16.6 / 16.1, 16.0 (two runs) | - |
| MYGA_US_SEC | 43.2, 47.6 / 43.2, 48.2 (two runs) | - |

No model is near +30%.

## Not brought, and why

- **Specialised lanes of 633c9be.** The double-sum and `all` lanes, with their builtin identity guard and a cold
  fallback per site. 952ce76 (before 856b984) folded their sites into the general lowering with the same `F64Sum`
  state.
- **Typed n-argument min/max of 633c9be** (`ordered_arguments.typed_extrema`/`object_extrema`). They need line B's
  math provider registry and `expression_kinds`. Not a generator or zip consumer.
- **The rounding memo.** Not a reduction.
- **633c9be's `test_builtin_reduction_lowering` and `test_reduction_lift_table_key`.** They need the helper package
  (`modelx_helpers`).
- **`test_reduction_in_vector_subscript`.** Vectors.
- **`sum(iterable, start)`.** Not lowered at 856b984 either.
- **The per-item borrowed integer path over lists.** Line B's is tuple-only and is kept as it is. VA_US_S's
  `sum(sa_pp_at(t, i, timing) for i in sub_ids())` walks its list as objects.

## For the next slices and the rebase

- **Vectors.** Where a source is a vector cell, the reduction reads the cell's Python tuple or list through the run-time
  check. The vector slice can feed typed buffers to the same emission point: `emit_level` kind `'seq'` takes
  `(variant, temps)`, and the reads are `seq_f64_at` / `seq_i64_at`.
- **Integers after the rebase** (no overflow flag, `integer_checks` from the sample). Re-wire:
  - `_REDUCTION_RETURNS` and `native_return_kind` in `integer_boundaries.py`, and their test;
  - the `_unbound_locals` addressed-local exception;
  - the changed `test_flagged_integer_arithmetic` assertion.

  Integer sums are already `PySum`'s exact loop, as in the source. The element's C integer arithmetic becomes plain
  or `overflowcheck` C.
- **pr/fast-sec math.** `native_math.lower` runs at the end of `_rewrite_powers`, after this pass. A math call in an
  element is an object at reduction time (`sum_add_object`, object min/max). The rebase should give `kind()` a branch
  for `self.<math ref>.<name>(x)`, so that the double state is kept.
