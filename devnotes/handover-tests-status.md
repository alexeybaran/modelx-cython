# Complete test set of the earlier compiler tree: status

Each run collects every test file under `modelx_cython/tests` and `integration_tests`. That includes the opt-in
markers `stress` and `fuzz_full` (`-m "stress or fuzz_full or not (stress or fuzz_full)"`). The runs use 4 xdist
workers. Raw output was kept per run outside the repository.

## Environment

- CPython 3.13.18 (GIL build), Cython 3.3.2, pandas 3.0.5, numpy 2.5.3, libcst 1.9.0, lifelib 20d4dae.
- modelx is installed per run in its own venv (`--system-site-packages`, `pip install --no-deps` of a checkout of
  alexeybaran/modelx): 636877b for the reference run, pr/collect 1aa21f4 for the others.
- `modelx_helpers` comes from alexeybaran/modelx-benchmark-toolkit,
  `parallel/20261113/round3/handoff/reproduction/modelx_helpers` at commit 3a6a1c19. It is put on the path by a
  `.pth` file. modelx 636877b does not contain it.

## Runs

| run | compiler tree | modelx | modelx_helpers | failed | passed | skipped | errors | time |
|---|---|---|---|---|---|---|---|---|
| R1 (reference) | 0b77986 (633c9be + files added from the proposal line) | 636877b | yes | 15 | 1737 | 9 | 6 | 57 min |
| R2 | c342db9 | 1aa21f4 | yes | 67 | 1642 | 9 | 9 | 43 min |
| R3a | r4/p3-inverse-adds 0a4f2ab | 1aa21f4 | yes | 64 | 1649 | 9 | 3 | 43 min |
| R3b | r4/p3-inverse-int 5e67b27 | 1aa21f4 | yes | 67 | 1596 | 9 | 3 | 37 min |
| R2 without helpers | c342db9 | 1aa21f4 | no | stopped at 97% | | | | |

Runs without `modelx_helpers` were not completed. The one for c342db9 was stopped at 97%: at that point about 800
tests had failed or errored, against about 780 passing. R3a and R3b were not run without it.

Skips in every run: 7 sample "main" targets, 1 test that needs a free-threaded Python, and 1 that needs match/case
with Cython 3.2. The `stress` tests ran on a GIL build.

## R1, the reference

**Failing test files that are not part of 633c9be.** They were added from the proposal line, and they fail on the
633c9be compiler: 13 failures and 6 errors.
- test_collect (3)
- test_value_storage (2)
- test_native_math_calls (2)
- test_upstream_regressions (6 errors)
- test_numeric_cache_results::test_table_matches_the_pxd
- test_dense_by_range
- test_dynamic_ref_items
- test_shared_item_reads
- test_unguarded_outer_operations
- test_flagged_integer_arithmetic

**633c9be's own tests: 2 failures, both opt-in `fuzz_full`, compiled not equal to live.**
- `test_fuzz_full_compiled_equals_live` (seed 1293137996): v45 and v1121 raise "must be real number, not Series";
  v1397 raises KeyError; v915 gives TypeError "unhashable type: 'list'" where live gives FormulaError.
- `test_fuzz_full_vector_keys_compiled_equals_live` (seed 2140845533): the same kinds.

## R2 against R1: 58 failures and 3 errors new

### Tests written against modelx 636877b's collect declaration

These declare `new_space(..., collect_over={"policy_id": ...}, collect_outputs=...)`, a dict of key domains. modelx
1aa21f4 refuses it: "TypeError: the collect domain must be a Cells of the Space, or the name of a Cells or a
Reference of the Space". The rows below are 56 tests plus 3 setup errors.

**Testing something other than collect.** These must be re-pointed at the new collect declaration or run without
collect:

| file | tests | what it tests |
|---|---|---|
| integration_tests/test_constant_key_reads.py | 1 | tables |
| integration_tests/test_diag_mt_shared_refs.py | 2 | tables, enum members |
| integration_tests/test_held_record_reads.py | 1 | held reads |
| integration_tests/test_held_table_handles.py | 1 | held reads |
| integration_tests/test_diag_run8_row_fields.py | 2 | tables |
| integration_tests/test_diag_run8_row_fields_derived.py | 7 | tables |
| integration_tests/test_diag_run8_extreme_part_forms.py | 1 | tables |
| integration_tests/test_diag_run8_extreme_part_tables.py | 1 | tables |
| integration_tests/test_diag_run8_scalar_index_subscript.py | 3 (errors) | tables |
| integration_tests/test_table_cells_api.py | 1 | tables |
| integration_tests/test_table_mixed_reads.py | 2 | tables |
| integration_tests/test_table_stress.py | 1 | tables |
| test_enum_results.py | 1 | enums |
| test_input_scenarios.py | 1 | scenarios |
| test_param_bounds.py | 3 | parameter bounds, enum domain |
| test_factory_private_refs.py | 3 | references |
| test_compiled_contract.py | 3 | export contract |
| test_diag_mt2_math_borrowed.py | 1 | math |
| test_locked_numeric_domains.py | 1 | caches |
| test_native_object_provider.py | 2 | providers |
| test_over_limit_hash_store.py | 2 | caches |
| test_post_factory_numeric.py | 1 | numeric results |
| test_private_lookup_roots.py | 1 | lookups |
| test_raised_sample_calls.py | 1 | sample |
| test_shared_dense_publish.py | 1 | caches |
| test_uncached_entries.py | 1 | entries |
| test_tracer_export_contract.py | 1 | tracer |

**Testing collect itself:**

| file | tests |
|---|---|
| test_collect_direct_isolation.py | 1 |
| test_collect_fallback_roots.py | 1 |
| test_collect_name_collision.py | 1 |
| test_collect_native_merge.py | 3 |
| test_collect_shared_items.py | 1 |
| test_parallel_collect_domain_default.py | 1 |
| test_parallel_collect_reuse.py | 2 |

### Other failures

- **The old collect side, which was removed:**
  `test_tracer::test_code_filter_distinguishes_collection_adapter_and_user_cell`.
- **Known Cython defect:** `integration_tests/test_table_fuzz.py::test_fuzz_vector_keys_compiled_equals_live`.
  - The test uses a fixed seed. Its check process dies with SIGSEGV, while it passes on R1.
  - The shape at fault is fuzz formula v47: `return T['q'] > 0.02 not in T.loc[('A', 'B')[t % 2]].index`.
  - Cython does not check the error from the truth test of the first link of a chained comparison when that link
    returns an object. The result is a use-after-free that a later garbage collection trips over.
  - It is not modelx-dependent; the generated sources are the same with every modelx commit tried. See
    `cython-chained-comparison-error-check.md`.

## R3a (0a4f2ab) against R2

**Passing in R3a, failing in R2:**
- test_upstream_regressions (6 errors)
- test_value_storage (2)
- test_over_limit_hash_store (2)
- test_dense_by_range
- test_dynamic_ref_items
- test_shared_item_reads
- test_fuzz_full_compiled_equals_live (random seed)

**New in R3a:**
- **Math path:**
  - test_native_module_math::test_module_math_bound_at_build_time: `('erf', set())`.
  - test_native_scalar_math::test_registered_scalar_math[math]: `('erf', [])`.
  - Both expect `erf` on the helper registry, which now goes through the native math path.
- **Integers:**
  - test_native_integer_boundaries::test_native_integer_boundary[reference]: TypeError "requires an integer".
  - test_upstream_regressions::test_a_variable_a_loop_left_unbound_raises_as_live: compiled gives a value where
    live raises.
- **Opt-in, random seed:** test_fuzz_full_vector_keys_compiled_equals_live: v1473 raises "must be real number, not
  Series".

## R3b (5e67b27) against R2

It has 1596 passing, against 1642: the branch removes integer-check tests.

**Passing or removed in R3b:**
- test_flagged_integer_arithmetic
- test_unguarded_outer_operations
- test_numeric_cache_results::test_table_matches_the_pxd
- test_value_storage (2)
- test_dynamic_ref_items
- test_upstream_regressions (6 setup errors)

**New in R3b:**
- **Environment:** 5 test_upstream_regressions tests fail with ImportError "modelx_cython is not installed here".
  The compiler tree was on PYTHONPATH, not pip-installed, and the test removes PYTHONPATH by design.
- **Opt-in, random seed:** test_fuzz_full_vector_keys_compiled_equals_live: v1084, where live gives FormulaError
  and compiled gives KeyError: 0.

In every 1aa21f4 run, the collect-declaration failures, the tracer test and the Cython-defect test are unchanged.
