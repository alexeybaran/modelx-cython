# R4 S2: trial cut of layer 1

Branch `r4/s2-layer1-trial`, cut by deletion from `r4/s1-zeroing` (3bc7be5) and diffed against upstream
`d22426f`. Runs on upstream modelx main (fec516d, 0.33.0) and upstream lifelib main (20d4dae), Python 3.14t.

## Expectation (as set in the brief, unchanged)
It builds; all twelve SEC models match live modelx (rel 1e-10, abs 1e-7); instructions per policy of
Term_US_SEC and MYGA_US_SEC equal the full tree's within 1%.

## Results
The expectation holds. All twelve SEC models build on upstream modelx and lifelib, and 12 of 12 match live
modelx: 0 mismatches in 11.8M compared values and raises. Instructions per policy, mean of 3 runs, trial /
full 3bc7be5 on the same machine: Term_US_SEC 7.420M / 7.427M = 0.999; MYGA_US_SEC 8.574M / 8.579M = 0.999
(run-to-run spread under 0.5%).

## How it was measured (gate tool not used)
The gate toolkit could not be read: attaching `modelx-benchmark-toolkit` was refused by a permission check,
so `RUN.md` and the gate were not used. Development modelx 636877b is not reachable either. Instead:
- Build: `mx.read_model` + `export` (upstream modelx), `mx2cy` with the product's `sample_sec.py`, `spec_sec.py`.
- Values: on live modelx, every no-argument cell, then every `t` cell for `t = 0 .. proj_len()`, then every
  cached value at every argument (lifelib's `full_sweep` + `_cached_values`); the same calls on the compiled
  package in a child. rel 1e-10 / abs 1e-7; kinds must match; a raise must meet the same raise (modelx's
  FormulaError unwrapped to its inner error line, as lifelib's `_error_text` does).
- Instructions per policy: callgrind collecting only inside the compiled `_mx_new_root__c_Projection`,
  `_c_Projection._f_result_cf` and `_mx_release_item__c_Projection`; a pass builds every point, calls
  `result_cf()`, deletes it; ipp = (I(4 passes) - I(1 pass)) / (3 x points), so first-pass reads cancel.
  Whole-process counts were 1-2% noisy run to run (free-threaded interpreter), this window is not.
- Full tree: 3bc7be5 refuses an upstream export (`collect_adapter.require_current`, protocol 2); for the
  baseline that one check was disabled in the installed copy only. No collect is generated from an upstream
  export, so this is its non-collect path; its table and vector providers were active in both builds.

## Lines (non-test, .py/.pyx/.pxd under modelx_cython)
Trial 12,937 lines against upstream 6,300 (+6,637); the full tree is about 32,400. Grown upstream modules: transformer 1,584 (1,096), builder 1,154 (987), tracer 1,123 (943), powers 815
(453), parser 672 (658), cli 624 (617), usage 616 (588), _mx_sys.pxd 548 (78), config 373 (175),
typedefs 145 (130); unchanged: monkeytype_tracing, consts, __init__, __main__.
New modules: integer_boundaries 869, entry_pipeline 520, inline_entries 361, admission 245, build 220,
locked_entries 202, item_locals 200, space_results 195, checked_arithmetic 187, _numeric_cache.pyx 177,
_wide_numeric_cache.pyx 169, cell_entries 150, numeric_cache 146, syntax 132, automatic 126,
held_arrays 125, ordered_arguments 117, scopes 110, reparse_check 105, value_storage 82, holds 77,
_numeric_cache.pxd 59, item_roots 48, _wide_numeric_cache.pxd 34, native_sources 31 (new here), exceptions 20;
doc/ not cut (table_reads.md, spec.md and two contract pages describe removed features).

## Kept outside the layer-1 list ("used": emitted in at least one SEC build)
1. admission.py 245 (from 558): `cli` calls `describe` for `_mx_admission.json`; `transformer` imports
   `over_limit_warning`/`use_warning`; 22 kept test files read the report. Move the two warnings (~70) into
   `automatic`; keep the report (~175) in layer 1 as the tests' oracle, or rewrite those tests.
2. space_results.py 195: `ModuleTransformer._rewrite_powers` calls `space_results.lower`, `leave_Module`
   `pin_function`. Used by no SEC model. Split out to a later layer (~195 + ~15 in transformer).
3. item_locals.py 200: `_rewrite_powers` calls `type_item_locals`, read by `inline_entries`. Used by no SEC
   model. Later layer (~200 + ~10).
4. checked_arithmetic.py 187: `_rewrite_powers` (`CheckedArithmetic`), and `integer_boundaries` imports its
   signed-local proof. Used (MYGA, SPIA divisibility). Name it in layer 1 under integer checks.
5. ordered_arguments.py 117 (from 235): `_rewrite_powers` calls `preserve_argument_order`; keeps Python power
   semantics in object results. Used (all 12). Layer 1 with the power rewrite. The provider min/max half
   (typed extrema, expression_kinds 264) was removed.
6. locked_entries.py 202: `cli` calls `lower_locked_entries`; locked Spaces are exported by upstream modelx.
   Layer 1 (formula entries).
7. build.py 220, reparse_check.py 105, exceptions.py 20, syntax 132, scopes 110: `cli`/`transformer`
   infrastructure (setup with fp-contract off, Cython error hints by cell, emitted text re-parse,
   conditional error values, shared CST helpers). Layer 1; build's error hints (~60) could wait.
8. native_sources.py 31 (new): the numeric caches are a .pyx/.pxd pair copied into the model package, which
   `helpers.install` did in the full tree; this is the provider installer without manifests and guards.
9. Small edits the real cut must repeat: collect fields of `BaseModel`, the tracer's collect filter,
   `collect_shared_item` (5 places), enum `readonly_refs`, provider binders (builder, transformer,
   inline_entries, usage), rounded-index helpers in `_mx_sys.pxd`/`space_results` (they need `_rounding.pxd`).

## Dependencies on modelx behaviour not in upstream modelx
None found for layer-1 code. All twelve SEC models export, compile and match on upstream modelx main.
Every export name the cut reads (`_mx_itemspaces`, `_mx_roots`, `_mx_spaces`, `_mx_walk`, `_mx_cells`,
`_mx_lock`, `_mx_copy_refs`, `_mx_copy_params`, `_mx_assign_params`, `_mx_assign_refs`, `_mx_is_cells_set`)
is written by upstream's exporter. What tied the full tree to development modelx, all removed:
`collect_adapter.require_current` (refuses an export without protocol 2), `parser` reading
`_mx_collect_cache`, `pyproject` `modelx>=0.33.1.dev0` (now `modelx`). Kept tests still call development
modelx API (`new_space(..., collect_over=...)`): see Tests.

## Configuration keys left
Model level (all optional, with defaults): `automatic_optimizations`, `dense_bounds_proofs`,
`dense_miss_binding`, `dense_one_axis`, `propagate_exact_receiver`, `native_cache_hits`, `infer_types`,
`integer_checks`. Space level: `spaces`, `cells`, `cells_param_size` (`cells_params` deprecated),
`max_dense_entries`. Cell level: `return_type`, `dense_window`, `native_numeric_cache`,
`numeric_cache_capacity`, `native_divisibility`, `memoize_power`, `max_dense_entries`.
Removed: `parallel_workers`, `native_math`, `typed_table_reads`, `held_table_handles`, `strict_tables`,
`prepared_tables`, `verify_inputs`, `strict_lowering`, `serial_collect`, `parallel_collect_*`,
`param_bound_checks`, `native_vector_ref_reads`, and `MX_CY_STRICT_TABLES` / `MX_CY_STRICT_LOWERING`.
A user must set only `cells_param_size` where a run calls a cell beyond the sampled range (every SEC
`spec_sec.py` does); a dense cache raises IndexError outside its declared or sampled domain.

## Tests (on the cut, upstream modelx, Python 3.14t)
- Layer-1 unit tests (`modelx_cython/tests`, not `compiled/`): 437 passed, 8 skipped, 0 failed. 25 files and
  helper_contracts/ removed (excluded features); collect, provider-extrema and rounding-key cases cut from 6 kept files.
- Compiled and integration tests: 129; 118 passed, 11 failed. All 11 build models with
  `new_space(collect_over=...)`, development modelx API (compiled_contract x3, input_scenarios,
  locked_numeric_domains, over_limit_hash_store x2, raised_sample_calls, shared_dense_publish,
  uncached_entries, value_storage). Their subjects are layer 1 checked on the direct and the collect route:
  the real cut must split each, keeping the direct route. 47 compiled and 27 integration test files of
  excluded features were removed, plus one needing the development helper package.
- Upstream's untouched suite (d22426f tests) on the cut: 248 passed, 7 skipped, 3 failed. All 3 are contract
  changes the full tree made on purpose and its own copies of these tests already assert: a dict or object
  cache outside the declared or sampled range raises IndexError (`test_various_types` x2), and the power memo
  spells `_mx_memo_pow` (`test_uslib`). The updated copies pass on the cut.
