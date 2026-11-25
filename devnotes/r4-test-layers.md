# R4: tests and modules by layer

The tree is `r4/integration` (merge 09cbd68 plus this sort). Each layer can later be removed by deleting its
whole files. Layers: **P1** is the fast compiler for the unchanged SEC models (entries, numeric caches and
allocation, holds, powers, integer handling, ranges, item release, the math and rounding providers, n-argument
min/max). **collect** stands on P1 alone. The syntax bridge stands on P1 and comes in this order:
**B1** labels (IntEnum members as integers), **B2** reductions, **B3** vectors, **B4** table base with scalar
reads (L1, L4, and the build report of the reads, `_mx_table_reads.json`), **B5** table rows (L2), **B6**
sub-tables with index extremes, membership and index walks (L3, L5, L7, L8), **B7** the read recorder and the
read audit (`read_notes.py`, `_mx_read_audit.json`, the report and warnings of reads left to pandas). Ordered
lookups (asof, searchsorted: L6) and the zip spelling are no longer lowered (`review/r4-table-forms-b`); until that
branch, B7 meant "ordered lookups and index walks", and the first versions of this record used that meaning.
**join** means the file needs collect and a bridge layer together. **upstream** means the file is untouched
since d22426f.

## Rules used

- A file's layer is the lowest layer whose tree holds every mechanism its assertions are about. A bridge form
  that only appears as a value compared with live modelx does not count: without the layer it runs as Python,
  and the comparison still holds. A pandas read, an IntEnum member or a `sum` over a generator are examples.
- When a test needs several bridge layers, it goes in the highest one (bridge layers are cumulative).
- A test whose model has `collect_over` (development modelx) needs the collect layer. If its subject is P1 and
  collect is only how the test reaches that subject, it gets a direct-route twin in the P1 file. The original
  test moves unchanged to a `*_collect.py` file. If its subject is a bridge layer, it goes to **join** (no twin
  was made; candidates are listed below).
- A mixed file is split by moving whole test functions, unchanged, into `<file>_<layer>.py` (`_b1` .. `_b7`,
  `_collect`, `_join`). A moved test imports the helpers it shares from the file of a lower layer in the same
  directory, which still exists in that layer's tree. No helper is copied.
- No test was deleted and no assertion was edited. The only changes in the original files are the moved
  functions and the imports they left unused, removed with them. The twins are new tests.

## Totals per layer (test files)

| layer | files | tests | lines |
|---|---|---|---|
| upstream | 3 | 51 | 471 |
| P1 | 88 | 559 | 9136 |
| collect | 30 | 74 | 2787 |
| B1 | 13 | 104 | 1146 |
| B2 | 16 | 21 | 1438 |
| B3 | 22 | 93 | 1778 |
| B4 | 10 | 9 | 582 |
| B5 | 4 | 21 | 556 |
| B6 | 14 | 63 | 2592 |
| B7 | 5 | 10 | 367 |
| join | 13 | 22 | 1554 |
| total | 218 | 1027 | 22407 |

The tests counts are collected cases (parametrized cases counted one by one). Before the sort the suite
collected 950 cases (942 passed, 8 skipped). After the sort it collected 962: the same 950 under their
original names, plus 12 cases from 11 twins (`test_param_bounds_inherit` has two parameters). All 962 passed,
with the same 8 skips. The counts above are the current tree, after the merges listed below.

## Merged since the sort

| branch | head | tests deleted | tests edited | tests added (layer) |
|---|---|---|---|---|
| `review/r4-reductions-a` | 945787c | `compiled/test_builtin_iterable_provider.py`: the `_native_iterables` provider it tested is removed with the specialised lanes | `compiled/test_native_reductions.py` and `integration_tests/test_diag_mt_shared_refs.py`: the removed builtin guard (`*_accepts`) and the specialised-site count are replaced by stronger checks (no generator, guard or cold fallback is left; the float state `f64_add` is used). `test_exact_sum_state.py`: every bit is now compared instead of a tolerance. `test_audit_contracts_b2.py` and `test_generator_builtin_lowering.py`: the constructor no longer takes a fallback factory, and they assert `general_count == 0` | `compiled/reduction_cases.py` (helper), `test_reduction_boolean_list`, `test_reduction_direct_literals`, `test_reduction_integer_attribute`, `test_reduction_keyword_arguments` (B2); `test_reduction_in_vector_subscript` (B3: a reduction inside a typed vector read) |
| `review/r4-reductions-b` (5b78f14, test only) | 5b78f14 | | | `test_a_provider_with_only_a_pxd_is_installed_as_a_header` in `test_audit_contracts.py` (P1); the rest of that branch was not needed (same edits as 945787c and 4a5f979) |
| `review/r4-table-base-a` | e17f329 | `test_prepared_tables.py::test_asof`: it tested only `PreparedTable.asof`, removed with the ordered-lookup arrays no compiled read used | `test_prepared_tables.py`: the lookups removed from the runtime (`key_codes`, `row`, `value`, `contains`, `column`) are made in the test's own probe through the methods compiled code uses; every comparison with pandas is kept | in `test_prepared_tables.py`: `test_index_pandas_looks_up_by_its_own_rules`, `test_bool_and_huge_keys`, `test_frame_bound_again_is_prepared_again`, `test_row_of_a_narrow_number_column_is_pandas`, `test_object_columns_hold_what_pandas_returns`; `integration_tests/test_table_runtime_contract.py` (B6: cell reads, a row field and index membership, no collect) |
| `review/r4-table-base-b` (the three commits after the nine of -a) | 35eb09d | | `integration_tests/test_table_runtime_contract.py`: cases added (a frame Cell that publishes no table, an item Cell returning a Reference's frame); nothing removed | `integration_tests/test_table_numpy_values.py` (B5: cell reads and a row field kept as numpy scalars, no collect) |
| `r4/w1-vectors` (up to and including the marker) | 8725799 | `compiled/test_vector_ref_reads.py`: `vector_ref_reads.py` and its opt-in spec key `native_vector_ref_reads` are removed (a spec that sets the key still builds, on the general path) | `integration_tests/test_diag_kr_forms.py`: the generated-text check that `[...] + list(range(...))` is built without `ivec_of` is dropped with that one-model spelling; the value comparisons with live stay. `test_vectors.py`: the vector sum now uses `_float_sum_state`'s `f64_add`/`f64_result` | `compiled/vector_live.py` (helper) and nine `compiled/test_*vector*` files (B3) |
| `review/r4-rest-b` | ea67e1d | `helper_contracts/test_native_rounding_provider.py` (the round memo provider, deleted), `test_provider_manifest.py` and `compiled/test_native_object_provider.py` with its twin and collect half (object providers, deleted). `test_removed_private_lookup_spec_key` was in the deleted file but checks a collect spec key that `serial_collect.py` still refuses: it moved unchanged to `compiled/test_private_lookup_roots_collect.py` | `compiled/test_private_lookup_roots*.py`: the memo probe is dropped, the value checks on both routes stay (applied to the collect file and to its direct-route twin). `compiled/test_native_builtin_math.py`: the guarded round in place of the memo operation. `compiled/test_native_typed_extrema.py`, `compiled/test_native_builtins_w18.py` (and its `_b2` file), `test_ordered_arguments.py`: min/max in object-result Cells is the built-in call again, so they assert the built-in in the report instead of the C comparison; value checks kept | `compiled/test_round_threads.py`, `compiled/test_export_without_collect_protocol.py`, `compiled/test_import_without_compiler.py`, `compiled/test_math_in_nested_scopes.py` (all P1) |
| `review/r4-table-forms-b` | 3732197 | `integration_tests/test_prepared_zip_prefix.py`: the zip spelling it tested is removed | `integration_tests/test_table_probe.py`: an asof or searchsorted inside the key of a lowered read is now the formula's own pandas call, so the check that native branches call no pandas skips those two calls only. `integration_tests/test_diag_kr_forms.py`: docstring (asof is no longer an integer key) | `integration_tests/forms_probe.py` (helper) and `test_forms_alias_edit`, `test_forms_hold_branch`, `test_forms_list_key`, `test_forms_own_frame`, `test_forms_scalar_type` (B4), `test_forms_view_keyerror` (B6); a case in `test_read_notes.py` (B7) |
| `r4/w1-vectors` (after the marker: tuples only) | d89c23e | none | vector tests converted to tuple cells (15 compiled files, `test_vectors.py`, `test_diag_kr_forms.py`); `test_local_vectors_without_vector_cells.py`: D94 removed, it now asserts that no formula is lowered and no `F64Vec` appears, and the checks of lowered local vectors go with D94; `test_collect_direct_isolation.py`: `is` became `==` (Python gets a new equal tuple per call); `test_leading_vector_sum.py`: the fallback text is "vector used in Python code" (as the branch's notes `devnotes/w1-vectors-tuple-experiment.md` state) | `compiled/test_tuple_vectors.py` (B3) |
| `review/r4-table-base-b` 2961ac2 (runtime only) | 2961ac2 | none | none | none: the zip and ordered-lookup runtime left unreachable by the table forms is removed from `_tables.pyx/.pxd` |

The review documents of these branches (`devnotes/REVIEW-BRIEF.md`, `devnotes/review-reductions-a.md`,
`devnotes/review-table-base-a.md`) were left out. The edits of the two reduction cases in `test_audit_contracts.py` were applied to `test_audit_contracts_b2.py`,
where the sort had moved them. `runtime/_native_streams.pxd` is now listed as B2; the first version of this
document wrongly listed it as P1, and it is used only by `reductions.py`. `test_prepared_tables.py` moved from B7 to B6:
after the table-base branch its probe uses no ordered lookup, only `prefix_range` (B6). After the table-forms branch the table
layers were relabelled (B6 takes the index walks, B7 is the recorder and audit; see the top of this record).

## Direct-route twins (new tests, P1) and their collect halves

| P1 file (twin) | collect file (original test, unchanged) | what the twin drops |
|---|---|---|
| `compiled/test_diag_mt2_math_borrowed.py` | `compiled/test_diag_mt2_math_borrowed_collect.py` | collect at 1 and 4 workers; the twin compares 60 direct items |
| `compiled/test_over_limit_hash_store.py` | `compiled/test_over_limit_hash_store_collect.py` (with `test_removed_spec_key_reuse_dense_windows_is_refused`) | collect totals; report, warning and item values kept |
| `compiled/test_private_lookup_roots.py` | `compiled/test_private_lookup_roots_collect.py` | collect total (the memo probe left with the memo in rest-b) |
| `compiled/test_uncached_entries.py` | `compiled/test_uncached_entries_collect.py` | collect totals, replaced by the sum of direct `value()` calls |
| `compiled/test_shared_dense_publish.py` | `compiled/test_shared_dense_publish_collect.py` | four collect workers; the fence code is checked the same way |
| `compiled/test_locked_numeric_domains.py` | `compiled/test_locked_numeric_domains_collect.py` | collect domain `periods(close)`; locked Market calls from four threads kept |
| `compiled/test_raised_sample_calls.py` | `compiled/test_raised_sample_calls_collect.py` | `collect_over` and the collect spec only (the probe was already direct) |
| (`compiled/test_native_object_provider.py`, deleted with object providers) | (`compiled/test_native_object_provider_collect.py`, deleted; `test_removed_private_lookup_spec_key` moved to `test_private_lookup_roots_collect.py`) | `Pricing.collect()`, replaced by the sum of direct items |
| `compiled/test_param_bounds.py` | `compiled/test_param_bounds_collect.py` | `param_bound_checks` at collect entry |
| `compiled/test_input_scenarios.py` | `compiled/test_input_scenarios_collect.py` | `collect().sum` in `check_results` |
| `test_tracer_export_contract.py` | `test_tracer_export_contract_collect.py` | the collect adapter method |

`compiled/test_value_storage.py` needed no twin. Its plain-model test already covers the direct route. The
collect-mode test moved to `compiled/test_value_storage_collect.py`, and the `collect` branch of the shared
helper `_check` stays in the P1 file.

## Files split (tests moved unchanged)

| from | to (layer) | tests moved |
|---|---|---|
| `test_audit_contracts.py` | `_b1` (B1) | the five enum-literal cases |
| | `_b2` (B2) | `test_reduction_does_not_type_a_nested_receiver_from_outer_cells`, `test_typed_reduction_preserves_receiver_binding_and_frame_observation` |
| | `_collect` | `test_collection_admission_reuses_the_existing_syntax_without_ast` |
| `test_automatic_cache_policy.py` | `_collect` | `test_automatic_collect_keeps_custom_lifecycle_on_normal_path` |
| `test_parser.py` | `_collect` | `test_exported_collection_and_dynamic_ref_storage` |
| `test_tracer.py` | `_collect` | `test_code_filter_distinguishes_collection_adapter_and_user_cell` |
| `test_read_audit.py` (B6) | `_b7` (B7) | `test_renamed_comprehension_target_matches_sample_notes`, `test_audit_counts_only_pandas_operations_at_an_items_loop_source` |
| `compiled/test_audit_runtime.py` | `_b2` (B2) | `test_scope_overrides_aliases_and_arithmetic` (asserts `native_reductions`) |
| `compiled/test_enum_by_type.py` (B1) | `_b4` (B4) | `test_table_key_levels_converted_through_a_fixed_reference` (with its `DATA_CHECK`) |
| `compiled/test_native_builtins_w18.py` | `_b2` (B2) | `test_nested_reductions_lifted` |
| `compiled/test_value_storage.py` | `_collect` | `test_unwritten_values_on_collect_and_direct_routes` |
| `compiled/test_param_bounds_collect.py` | `compiled/test_param_bounds_join.py` (join) | `test_enum_argument_domain_is_the_enum` (B1 enum domain on the collect build) |

## Direct-route twins of the join files (new tests) and their join halves

The table and label tests whose models used `collect_over` only to build the sample, or to read the result
through collect as well as directly, each got a twin with the same model without `collect_over`. The twin is in
the file's former name, in its bridge layer. The original moved unchanged to `<file>_join.py`. A twin keeps every
assertion of the generated code and the reports, and compares the direct items with live. It drops only the
collect totals and the collect spec keys.

| twin (layer) | join half |
|---|---|
| `integration_tests/test_held_table_handles.py` (B4) | `test_held_table_handles_join.py` |
| `integration_tests/test_table_cells_api.py` (B4) | `test_table_cells_api_join.py` |
| `integration_tests/test_constant_key_reads.py` (B4) | `test_constant_key_reads_join.py` |
| `integration_tests/test_diag_run8_row_fields.py`, `test_diag_run8_row_fields_derived.py` (B5) | their `_join` files |
| `integration_tests/test_diag_mt_shared_refs.py` (B6), `test_diag_mt_shared_refs_b1.py` (B1, the IntEnum case) | `test_diag_mt_shared_refs_join.py` |
| `integration_tests/test_diag_run8_extreme_part_forms.py`, `test_diag_run8_extreme_part_tables.py`, `test_diag_run8_scalar_index_subscript.py`, `test_table_mixed_reads.py` (B6) | their `_join` files |
| `compiled/test_enum_results.py`, `compiled/test_param_bounds_b1.py` (B1) | `test_enum_results_join.py`, `test_param_bounds_join.py` |

The `shared` case of `test_diag_run8_row_fields_derived` is only in the join half. Its subject is a Shared ItemSpace
that reaches Projection's items through collect. Without collect, the row field of that shape is not inlined
(the values still equal live).

`integration_tests/test_table_stress.py` has no twin. Its subject is four collect workers reading a published
table while the first readers bind it on a new model, so collect is the subject.

## Observations for the carve

- `test_prepared_tables.py` is B6. Its probe calls `prefix_range`, so its B4 cases can move only once
  `_tables.pyx` is split.
- `test_table_safety_net.py` and `test_table_fuzz.py` import `test_table_probe` (B6), so they are B6 as well.
- `test_read_audit.py` and `test_read_audit_b7.py` are both B7 now (the split was made under the earlier meaning
  of B6 and B7); they can be joined again.
- The runtime's ordered-lookup and zip helpers that the table forms left without a caller were removed
  with 2961ac2.

- `integration_tests/conftest.py` and `modelx_cython/tests/conftest.py` set strict table lowering and the
  fuzz/stress markers. They are listed as P1 because the setting is harmless without tables.

## Test files

| file | layer | tests | lines |
|---|---|---|---|
| `modelx_cython/tests/test_annuallife.py` | upstream | 3 | 68 |
| `modelx_cython/tests/test_review_regressions.py` | upstream | 2 | 105 |
| `modelx_cython/tests/test_usage.py` | upstream | 46 | 298 |
| `integration_tests/conftest.py` | P1 | 0 | 28 |
| `integration_tests/test_none_argument.py` | P1 | 1 | 54 |
| `modelx_cython/tests/compiled/test_audit_runtime.py` | P1 | 2 | 86 |
| `modelx_cython/tests/compiled/test_authored_miss_budget.py` | P1 | 12 | 110 |
| `modelx_cython/tests/compiled/test_automatic_policy.py` | P1 | 1 | 59 |
| `modelx_cython/tests/compiled/test_checked_native_range.py` | P1 | 1 | 96 |
| `modelx_cython/tests/compiled/test_d82_unguarded_outer_operation.py` | P1 | 3 | 155 |
| `modelx_cython/tests/compiled/test_declared_lower_bound.py` | P1 | 4 | 86 |
| `modelx_cython/tests/compiled/test_dense_by_use.py` | P1 | 2 | 71 |
| `modelx_cython/tests/compiled/test_dense_formula_defaults.py` | P1 | 1 | 20 |
| `modelx_cython/tests/compiled/test_dense_negative_bound.py` | P1 | 6 | 49 |
| `modelx_cython/tests/compiled/test_diag_mt2_math_borrowed.py` | P1 | 1 | 76 |
| `modelx_cython/tests/compiled/test_diag_scale_local_math.py` | P1 | 1 | 131 |
| `modelx_cython/tests/compiled/test_export_without_collect_protocol.py` | P1 | 1 | 34 |
| `modelx_cython/tests/compiled/test_formula_entry_domains.py` | P1 | 1 | 95 |
| `modelx_cython/tests/compiled/test_guarded_integer_slow_paths.py` | P1 | 2 | 71 |
| `modelx_cython/tests/compiled/test_held_array_reads.py` | P1 | 2 | 92 |
| `modelx_cython/tests/compiled/test_import_without_compiler.py` | P1 | 1 | 33 |
| `modelx_cython/tests/compiled/test_inferred_cache_domain.py` | P1 | 2 | 86 |
| `modelx_cython/tests/compiled/test_input_scenarios.py` | P1 | 1 | 112 |
| `modelx_cython/tests/compiled/test_integer_checks_without_inference.py` | P1 | 2 | 52 |
| `modelx_cython/tests/compiled/test_integer_overflow_model.py` | P1 | 3 | 155 |
| `modelx_cython/tests/compiled/test_integer_power_checked.py` | P1 | 4 | 129 |
| `modelx_cython/tests/compiled/test_item_locals_compiled.py` | P1 | 2 | 157 |
| `modelx_cython/tests/compiled/test_local_fractional_power.py` | P1 | 1 | 63 |
| `modelx_cython/tests/compiled/test_locked_numeric_domains.py` | P1 | 1 | 70 |
| `modelx_cython/tests/compiled/test_math_in_nested_scopes.py` | P1 | 1 | 35 |
| `modelx_cython/tests/compiled/test_multiline_expressions.py` | P1 | 1 | 44 |
| `modelx_cython/tests/compiled/test_native_builtin_math.py` | P1 | 1 | 112 |
| `modelx_cython/tests/compiled/test_native_builtins_w18.py` | P1 | 3 | 213 |
| `modelx_cython/tests/compiled/test_native_cache_hits.py` | P1 | 1 | 84 |
| `modelx_cython/tests/compiled/test_native_integer_boundaries.py` | P1 | 30 | 577 |
| `modelx_cython/tests/compiled/test_native_module_math.py` | P1 | 1 | 135 |
| `modelx_cython/tests/compiled/test_native_typed_extrema.py` | P1 | 1 | 66 |
| `modelx_cython/tests/compiled/test_numeric_cache.py` | P1 | 1 | 61 |
| `modelx_cython/tests/compiled/test_object_power_contract.py` | P1 | 1 | 80 |
| `modelx_cython/tests/compiled/test_ordered_arguments_contract.py` | P1 | 1 | 88 |
| `modelx_cython/tests/compiled/test_original_contract_regressions.py` | P1 | 2 | 54 |
| `modelx_cython/tests/compiled/test_over_limit_hash_store.py` | P1 | 1 | 93 |
| `modelx_cython/tests/compiled/test_param_bounds.py` | P1 | 2 | 88 |
| `modelx_cython/tests/compiled/test_power_sinks.py` | P1 | 1 | 87 |
| `modelx_cython/tests/compiled/test_private_lookup_roots.py` | P1 | 1 | 64 |
| `modelx_cython/tests/compiled/test_provider_build_cache.py` | P1 | 1 | 37 |
| `modelx_cython/tests/compiled/test_public_cache_bounds_proofs.py` | P1 | 6 | 63 |
| `modelx_cython/tests/compiled/test_public_one_axis.py` | P1 | 10 | 45 |
| `modelx_cython/tests/compiled/test_raised_sample_calls.py` | P1 | 1 | 109 |
| `modelx_cython/tests/compiled/test_round_threads.py` | P1 | 1 | 58 |
| `modelx_cython/tests/compiled/test_scalar_union_contract.py` | P1 | 1 | 55 |
| `modelx_cython/tests/compiled/test_serial_native_transfers.py` | P1 | 2 | 162 |
| `modelx_cython/tests/compiled/test_shared_dense_publish.py` | P1 | 1 | 46 |
| `modelx_cython/tests/compiled/test_shared_space_first_read.py` | P1 | 1 | 66 |
| `modelx_cython/tests/compiled/test_shifted_cache.py` | P1 | 3 | 83 |
| `modelx_cython/tests/compiled/test_space_item_results.py` | P1 | 2 | 197 |
| `modelx_cython/tests/compiled/test_typed_dispatch_locals.py` | P1 | 1 | 63 |
| `modelx_cython/tests/compiled/test_uncached_entries.py` | P1 | 1 | 151 |
| `modelx_cython/tests/compiled/test_value_storage.py` | P1 | 2 | 305 |
| `modelx_cython/tests/compiled/test_wide_numeric_cache.py` | P1 | 1 | 76 |
| `modelx_cython/tests/conftest.py` | P1 | 0 | 4 |
| `modelx_cython/tests/helper_contracts/__init__.py` | P1 | 0 | 0 |
| `modelx_cython/tests/test_audit_contracts.py` | P1 | 5 | 87 |
| `modelx_cython/tests/test_automatic_cache_policy.py` | P1 | 4 | 34 |
| `modelx_cython/tests/test_build_diagnostics.py` | P1 | 2 | 68 |
| `modelx_cython/tests/test_builder.py` | P1 | 46 | 301 |
| `modelx_cython/tests/test_cache_hits.py` | P1 | 25 | 135 |
| `modelx_cython/tests/test_cli.py` | P1 | 3 | 62 |
| `modelx_cython/tests/test_config.py` | P1 | 14 | 65 |
| `modelx_cython/tests/test_dict_lookup.py` | P1 | 2 | 88 |
| `modelx_cython/tests/test_expression_context.py` | P1 | 3 | 18 |
| `modelx_cython/tests/test_fp_contract.py` | P1 | 3 | 42 |
| `modelx_cython/tests/test_helper_members.py` | P1 | 4 | 53 |
| `modelx_cython/tests/test_infer_types_default.py` | P1 | 4 | 19 |
| `modelx_cython/tests/test_item_locals.py` | P1 | 6 | 89 |
| `modelx_cython/tests/test_literal_syntax.py` | P1 | 39 | 35 |
| `modelx_cython/tests/test_numeric_cache_reservation.py` | P1 | 1 | 80 |
| `modelx_cython/tests/test_object_return_annotation.py` | P1 | 7 | 38 |
| `modelx_cython/tests/test_ordered_arguments.py` | P1 | 11 | 143 |
| `modelx_cython/tests/test_param_bounds_spec.py` | P1 | 3 | 35 |
| `modelx_cython/tests/test_parser.py` | P1 | 25 | 147 |
| `modelx_cython/tests/test_powers.py` | P1 | 111 | 540 |
| `modelx_cython/tests/test_range_increment.py` | P1 | 11 | 18 |
| `modelx_cython/tests/test_reparse_check.py` | P1 | 9 | 75 |
| `modelx_cython/tests/test_samples.py` | P1 | 30 | 654 |
| `modelx_cython/tests/test_scalar_result_variants.py` | P1 | 12 | 50 |
| `modelx_cython/tests/test_sibling_accessor_counts.py` | P1 | 1 | 39 |
| `modelx_cython/tests/test_single_frontend.py` | P1 | 1 | 15 |
| `modelx_cython/tests/test_tracer.py` | P1 | 34 | 475 |
| `modelx_cython/tests/test_tracer_export_contract.py` | P1 | 1 | 37 |
| `modelx_cython/tests/test_uslib.py` | P1 | 1 | 123 |
| `modelx_cython/tests/compiled/test_collect_direct_isolation.py` | collect | 1 | 151 |
| `modelx_cython/tests/compiled/test_collect_fallback_roots.py` | collect | 1 | 108 |
| `modelx_cython/tests/compiled/test_collect_name_collision.py` | collect | 1 | 55 |
| `modelx_cython/tests/compiled/test_collect_native_merge.py` | collect | 3 | 333 |
| `modelx_cython/tests/compiled/test_collect_shared_items.py` | collect | 1 | 97 |
| `modelx_cython/tests/compiled/test_collect_workers_runtime.py` | collect | 1 | 61 |
| `modelx_cython/tests/compiled/test_compiled_contract.py` | collect | 3 | 118 |
| `modelx_cython/tests/compiled/test_diag_mt2_math_borrowed_collect.py` | collect | 1 | 96 |
| `modelx_cython/tests/compiled/test_factory_private_refs.py` | collect | 3 | 101 |
| `modelx_cython/tests/compiled/test_input_scenarios_collect.py` | collect | 1 | 113 |
| `modelx_cython/tests/compiled/test_locked_numeric_domains_collect.py` | collect | 1 | 71 |
| `modelx_cython/tests/compiled/test_ordered_numeric_blocks.py` | collect | 1 | 129 |
| `modelx_cython/tests/compiled/test_over_limit_hash_store_collect.py` | collect | 2 | 122 |
| `modelx_cython/tests/compiled/test_parallel_collect_domain_default.py` | collect | 1 | 68 |
| `modelx_cython/tests/compiled/test_parallel_collect_reuse.py` | collect | 2 | 158 |
| `modelx_cython/tests/compiled/test_param_bounds_collect.py` | collect | 2 | 114 |
| `modelx_cython/tests/compiled/test_post_factory_numeric.py` | collect | 1 | 180 |
| `modelx_cython/tests/compiled/test_private_lookup_roots_collect.py` | collect | 2 | 88 |
| `modelx_cython/tests/compiled/test_raised_sample_calls_collect.py` | collect | 1 | 112 |
| `modelx_cython/tests/compiled/test_shared_dense_publish_collect.py` | collect | 1 | 57 |
| `modelx_cython/tests/compiled/test_uncached_entries_collect.py` | collect | 1 | 158 |
| `modelx_cython/tests/compiled/test_value_storage_collect.py` | collect | 1 | 7 |
| `modelx_cython/tests/test_audit_contracts_collect.py` | collect | 1 | 13 |
| `modelx_cython/tests/test_automatic_cache_policy_collect.py` | collect | 3 | 22 |
| `modelx_cython/tests/test_collect_adapter.py` | collect | 12 | 84 |
| `modelx_cython/tests/test_collect_factory_alias.py` | collect | 5 | 16 |
| `modelx_cython/tests/test_factory_reuse.py` | collect | 18 | 77 |
| `modelx_cython/tests/test_parser_collect.py` | collect | 1 | 22 |
| `modelx_cython/tests/test_tracer_collect.py` | collect | 1 | 19 |
| `modelx_cython/tests/test_tracer_export_contract_collect.py` | collect | 1 | 37 |
| `integration_tests/test_diag_mt_shared_refs_b1.py` | B1 | 1 | 77 |
| `modelx_cython/tests/compiled/test_d98_enum_identity.py` | B1 | 1 | 77 |
| `modelx_cython/tests/compiled/test_diag_scale_enum_conditionals.py` | B1 | 1 | 111 |
| `modelx_cython/tests/compiled/test_enum_by_type.py` | B1 | 1 | 61 |
| `modelx_cython/tests/compiled/test_enum_containers.py` | B1 | 1 | 67 |
| `modelx_cython/tests/compiled/test_enum_object_uses.py` | B1 | 5 | 128 |
| `modelx_cython/tests/compiled/test_enum_reference_compare.py` | B1 | 1 | 65 |
| `modelx_cython/tests/compiled/test_enum_results.py` | B1 | 1 | 77 |
| `modelx_cython/tests/compiled/test_native_enum_membership.py` | B1 | 1 | 67 |
| `modelx_cython/tests/compiled/test_native_enums.py` | B1 | 2 | 88 |
| `modelx_cython/tests/compiled/test_param_bounds_b1.py` | B1 | 1 | 29 |
| `modelx_cython/tests/test_audit_contracts_b1.py` | B1 | 5 | 56 |
| `modelx_cython/tests/test_enum_literals.py` | B1 | 83 | 243 |
| `modelx_cython/tests/compiled/reduction_cases.py` | B2 | 0 | 68 |
| `modelx_cython/tests/compiled/test_audit_runtime_b2.py` | B2 | 1 | 177 |
| `modelx_cython/tests/compiled/test_diag_scale_int_items.py` | B2 | 2 | 143 |
| `modelx_cython/tests/compiled/test_generator_boolean_ranges.py` | B2 | 1 | 252 |
| `modelx_cython/tests/compiled/test_generic_composition.py` | B2 | 1 | 84 |
| `modelx_cython/tests/compiled/test_native_builtins_w18_b2.py` | B2 | 1 | 62 |
| `modelx_cython/tests/compiled/test_native_generator_reductions.py` | B2 | 1 | 125 |
| `modelx_cython/tests/compiled/test_native_reduction_cases.py` | B2 | 1 | 117 |
| `modelx_cython/tests/compiled/test_native_reductions.py` | B2 | 1 | 109 |
| `modelx_cython/tests/compiled/test_reduction_boolean_list.py` | B2 | 1 | 23 |
| `modelx_cython/tests/compiled/test_reduction_direct_literals.py` | B2 | 1 | 18 |
| `modelx_cython/tests/compiled/test_reduction_integer_attribute.py` | B2 | 1 | 15 |
| `modelx_cython/tests/compiled/test_reduction_keyword_arguments.py` | B2 | 1 | 16 |
| `modelx_cython/tests/test_audit_contracts_b2.py` | B2 | 4 | 42 |
| `modelx_cython/tests/test_exact_sum_state.py` | B2 | 3 | 167 |
| `modelx_cython/tests/test_generator_builtin_lowering.py` | B2 | 1 | 20 |
| `modelx_cython/tests/compiled/test_float_vector_of_integers.py` | B3 | 1 | 17 |
| `modelx_cython/tests/compiled/test_float_vector_writes.py` | B3 | 1 | 34 |
| `modelx_cython/tests/compiled/test_integer_vector_cells.py` | B3 | 1 | 94 |
| `modelx_cython/tests/compiled/test_integer_vector_updates.py` | B3 | 1 | 21 |
| `modelx_cython/tests/compiled/test_integer_vector_writes.py` | B3 | 1 | 36 |
| `modelx_cython/tests/compiled/test_leading_vector_sum.py` | B3 | 1 | 115 |
| `modelx_cython/tests/compiled/test_local_vectors_without_vector_cells.py` | B3 | 1 | 108 |
| `modelx_cython/tests/compiled/test_power_memo_with_vectors.py` | B3 | 1 | 73 |
| `modelx_cython/tests/compiled/test_reduction_in_vector_subscript.py` | B3 | 1 | 17 |
| `modelx_cython/tests/compiled/test_tuple_vectors.py` | B3 | 1 | 99 |
| `modelx_cython/tests/compiled/test_vector_cells.py` | B3 | 2 | 247 |
| `modelx_cython/tests/compiled/test_vector_conditionals.py` | B3 | 1 | 22 |
| `modelx_cython/tests/compiled/test_vector_displays.py` | B3 | 1 | 111 |
| `modelx_cython/tests/compiled/test_vector_extrema_and_cache.py` | B3 | 1 | 147 |
| `modelx_cython/tests/compiled/test_vector_list_mutators.py` | B3 | 3 | 59 |
| `modelx_cython/tests/compiled/test_vector_loop_renames.py` | B3 | 1 | 25 |
| `modelx_cython/tests/compiled/test_vector_placeholders.py` | B3 | 1 | 22 |
| `modelx_cython/tests/compiled/test_vector_python_exits.py` | B3 | 1 | 104 |
| `modelx_cython/tests/compiled/test_vector_reference_reads.py` | B3 | 1 | 50 |
| `modelx_cython/tests/compiled/vector_live.py` | B3 | 0 | 70 |
| `modelx_cython/tests/test_vector_extrema_typing.py` | B3 | 24 | 121 |
| `modelx_cython/tests/test_vectors.py` | B3 | 47 | 186 |
| `integration_tests/forms_probe.py` | B4 | 0 | 122 |
| `integration_tests/test_constant_key_reads.py` | B4 | 1 | 102 |
| `integration_tests/test_forms_alias_edit.py` | B4 | 1 | 14 |
| `integration_tests/test_forms_hold_branch.py` | B4 | 1 | 16 |
| `integration_tests/test_forms_list_key.py` | B4 | 1 | 11 |
| `integration_tests/test_forms_own_frame.py` | B4 | 1 | 17 |
| `integration_tests/test_forms_scalar_type.py` | B4 | 1 | 16 |
| `integration_tests/test_held_table_handles.py` | B4 | 1 | 91 |
| `integration_tests/test_table_cells_api.py` | B4 | 1 | 139 |
| `modelx_cython/tests/compiled/test_enum_by_type_b4.py` | B4 | 1 | 54 |
| `integration_tests/test_diag_run8_row_fields.py` | B5 | 2 | 138 |
| `integration_tests/test_diag_run8_row_fields_derived.py` | B5 | 6 | 99 |
| `integration_tests/test_table_numpy_values.py` | B5 | 2 | 128 |
| `modelx_cython/tests/test_diag_run8_row_cells.py` | B5 | 11 | 191 |
| `integration_tests/_diag_run8_c.py` | B6 | 0 | 26 |
| `integration_tests/test_diag_kr_forms.py` | B6 | 1 | 262 |
| `integration_tests/test_diag_mt_shared_refs.py` | B6 | 1 | 92 |
| `integration_tests/test_diag_run8_extreme_part_forms.py` | B6 | 1 | 101 |
| `integration_tests/test_diag_run8_extreme_part_tables.py` | B6 | 1 | 93 |
| `integration_tests/test_diag_run8_scalar_index_subscript.py` | B6 | 3 | 101 |
| `integration_tests/test_forms_view_keyerror.py` | B6 | 1 | 16 |
| `integration_tests/test_prepared_table_integer_keys.py` | B6 | 1 | 79 |
| `integration_tests/test_table_fuzz.py` | B6 | 5 | 290 |
| `integration_tests/test_table_mixed_reads.py` | B6 | 1 | 234 |
| `integration_tests/test_table_probe.py` | B6 | 5 | 695 |
| `integration_tests/test_table_runtime_contract.py` | B6 | 1 | 143 |
| `integration_tests/test_table_safety_net.py` | B6 | 2 | 70 |
| `modelx_cython/tests/test_prepared_tables.py` | B6 | 40 | 390 |
| `integration_tests/table_matrix/test_pandas_read_report.py` | B7 | 2 | 113 |
| `integration_tests/test_table_read_warnings.py` | B7 | 1 | 68 |
| `modelx_cython/tests/test_read_audit.py` | B7 | 2 | 66 |
| `modelx_cython/tests/test_read_audit_b7.py` | B7 | 2 | 47 |
| `modelx_cython/tests/test_read_notes.py` | B7 | 3 | 73 |
| `integration_tests/test_constant_key_reads_join.py` | join | 1 | 115 |
| `integration_tests/test_diag_mt_shared_refs_join.py` | join | 2 | 194 |
| `integration_tests/test_diag_run8_extreme_part_forms_join.py` | join | 1 | 102 |
| `integration_tests/test_diag_run8_extreme_part_tables_join.py` | join | 1 | 94 |
| `integration_tests/test_diag_run8_row_fields_derived_join.py` | join | 7 | 104 |
| `integration_tests/test_diag_run8_row_fields_join.py` | join | 2 | 148 |
| `integration_tests/test_diag_run8_scalar_index_subscript_join.py` | join | 3 | 101 |
| `integration_tests/test_held_table_handles_join.py` | join | 1 | 106 |
| `integration_tests/test_table_cells_api_join.py` | join | 1 | 144 |
| `integration_tests/test_table_mixed_reads_join.py` | join | 1 | 251 |
| `integration_tests/test_table_stress.py` | join | 0 | 87 |
| `modelx_cython/tests/compiled/test_enum_results_join.py` | join | 1 | 79 |
| `modelx_cython/tests/compiled/test_param_bounds_join.py` | join | 1 | 29 |

## Source modules (`modelx_cython`, no tests)

A shared module is listed with the split I would make and its estimated size per layer. The estimate is per
function, by the forms and mechanisms the function handles; its P1 share is the remainder. `transformer.py`
is the shared wiring, and only the integrator resolves conflicts in it.

| module | layer | lines | split |
|---|---|---|---|
| `modelx_cython/_mx_sys.pxd` | shared: P1 ≈573, B1 ≈15 | 588 | `_mx_check_enum_schema` (B1) |
| `modelx_cython/admission.py` | shared: P1 ≈262, collect ≈15, B1 ≈3, B2 ≈15, B3 ≈30, B4 ≈200 | 525 | report keys: reductions (B2), vector cache/refusals (B3), `table_read_sites` and read warnings (B4, with form names of B5-B7), collect bound checks (collect) |
| `modelx_cython/builder.py` | shared: P1 ≈1138, collect ≈4, B1 ≈10, B3 ≈35 | 1187 | vector shape/cache of a Cell (B3); enum Reference info (B1) |
| `modelx_cython/cli.py` | shared: P1 ≈627, collect ≈16, B1 ≈3, B4 ≈30, B7 ≈40 | 716 | `enums.literals` call (B1); table setup in `run_sample`/`main_handler` (B4); read-note recording and the read audit file (B7); collect runtime check (collect) |
| `modelx_cython/config.py` | shared: P1 ≈360, collect ≈35, B4 ≈32 | 427 | collect spec keys (collect); table policy keys (B4) |
| `modelx_cython/frame_reads.py` | shared: B4 ≈1189, B5 ≈307, B6 ≈975, B7 ≈188 | 2659 | by recogniser: L1/L4, the shared machinery and the numpy-scalar boxing of reads (B4); L2 row fields (B5); L3 prefixes, L5 extremes, L7 membership, L8 index walks, comprehensions, iterrows (B6); the read audit (`audit`, `audit_subtables`, `without_fallbacks`) (B7) |
| `modelx_cython/holds.py` | shared: P1 ≈90, B4 ≈13 | 103 | `hold_table` (B4); the registry is P1 |
| `modelx_cython/parser.py` | shared: P1 ≈660, collect ≈16 | 676 | `collect_cache_classes` (collect) |
| `modelx_cython/runtime/_tables.pxd` | shared: B4 ≈231, B5 ≈15, B6 ≈106 | 352 | declarations of the above |
| `modelx_cython/runtime/_tables.pyx` | shared: B4 ≈530, B5 ≈40, B6 ≈131 | 701 | preparation, binding and keyed reads (B4); `hold_row` (B5); `prefix_range`, views, extremes and the view walks `view_start`, `view_stop`, `position` (B6) |
| `modelx_cython/table_cells.py` | shared: B3 ≈250, B4 ≈284 | 534 | `vector_cells` and the vector exits of `RenameMembers` (B3); table cells, frame binding and publication (B4) |
| `modelx_cython/tracer.py` | shared: P1 ≈1010, collect ≈15, B1 ≈75, B3 ≈100, B4 ≈60 | 1260 | IntEnum value info and enum-argument widening (B1); vector shapes and their merge (B3); frame schema and read notes hooks (B4); collect-adapter filter (collect) |
| `modelx_cython/transformer.py` | shared: P1 ≈1538, collect ≈80, B1 ≈10, B2 ≈22, B3 ≈50, B4 ≈97, B5 ≈70, B6 ≈63 | 1930 | shared wiring; table hooks `_table_reads*`, `_lower_frames`, `_pandas_reads` (B4), `_row_cells`, `_table_reads_with_row_fields` (B5), `_prefix_cells`, `_param_prefix_cells` (B6); reduction and vector passes in `_rewrite_powers`/`update_method`; collect hooks in `__init__`, `leave_Module`, `leave_ClassDef`; enum `prepare`/storage |
| `modelx_cython/usage.py` | shared: P1 ≈630, B4 ≈25 | 655 | pandas receiver resolution (B4); the memoryview usage classifier is P1 |
| `modelx_cython/__init__.py` | P1 | 35 | |
| `modelx_cython/__main__.py` | P1 | 24 | |
| `modelx_cython/_call_layout.pxd` | P1 | 12 | |
| `modelx_cython/_numeric_cache.pxd` | P1 | 59 | |
| `modelx_cython/_numeric_cache.pyx` | P1 | 177 | |
| `modelx_cython/_rounding.pxd` | P1 | 61 | |
| `modelx_cython/_wide_numeric_cache.pxd` | P1 | 34 | |
| `modelx_cython/_wide_numeric_cache.pyx` | P1 | 169 | |
| `modelx_cython/automatic.py` | P1 | 126 | |
| `modelx_cython/build.py` | P1 | 222 | |
| `modelx_cython/cell_entries.py` | P1 | 150 | |
| `modelx_cython/checked_arithmetic.py` | P1 | 187 | |
| `modelx_cython/consts.py` | P1 | 121 | |
| `modelx_cython/entry_pipeline.py` | P1 | 520 | |
| `modelx_cython/exceptions.py` | P1 | 20 | |
| `modelx_cython/held_arrays.py` | P1 | 126 | |
| `modelx_cython/helper_members.py` | P1 | 99 | |
| `modelx_cython/helpers.py` | P1 | 385 | |
| `modelx_cython/inline_entries.py` | P1 | 372 | |
| `modelx_cython/integer_boundaries.py` | P1 | 869 | |
| `modelx_cython/item_locals.py` | P1 | 200 | |
| `modelx_cython/item_roots.py` | P1 | 49 | |
| `modelx_cython/locked_entries.py` | P1 | 202 | |
| `modelx_cython/monkeytype_tracing.py` | P1 | 396 | |
| `modelx_cython/numeric_cache.py` | P1 | 169 | |
| `modelx_cython/ordered_arguments.py` | P1 | 145 | |
| `modelx_cython/powers.py` | P1 | 815 | |
| `modelx_cython/provider_build_cache.py` | P1 | 144 | |
| `modelx_cython/reparse_check.py` | P1 | 105 | |
| `modelx_cython/rounding.py` | P1 | 35 | |
| `modelx_cython/runtime/__init__.py` | P1 | 1 | |
| `modelx_cython/runtime/_native_math.pxd` | P1 | 35 | |
| `modelx_cython/runtime/_native_math.pyx` | P1 | 180 | |
| `modelx_cython/scopes.py` | P1 | 110 | |
| `modelx_cython/space_results.py` | P1 | 197 | |
| `modelx_cython/syntax.py` | P1 | 132 | |
| `modelx_cython/typedefs.py` | P1 | 145 | |
| `modelx_cython/value_storage.py` | P1 | 82 | |
| `modelx_cython/collect_adapter.py` | collect | 88 | |
| `modelx_cython/factory_individuals.py` | collect | 141 | |
| `modelx_cython/factory_reuse.py` | collect | 271 | |
| `modelx_cython/runtime/factory_reuse.py` | collect | 22 | |
| `modelx_cython/serial_collect.py` | collect | 1137 | |
| `modelx_cython/enums.py` | B1 | 339 | |
| `modelx_cython/_float_sum_state.pxd` | B2 | 233 | |
| `modelx_cython/reductions.py` | B2 | 942 | |
| `modelx_cython/runtime/_native_streams.pxd` | B2 | 28 | |
| `modelx_cython/runtime/_native_vectors.pxd` | B3 | 252 | |
| `modelx_cython/runtime/_native_vectors.pyx` | B3 | 363 | |
| `modelx_cython/vectors.py` | B3 | 2066 | |
| `modelx_cython/key_facts.py` | B4 | 402 | |
| `modelx_cython/runtime/table_cells.py` | B4 | 63 | |
| `modelx_cython/row_cells.py` | B5 | 410 | |
| `modelx_cython/read_notes.py` | B7 | 477 | |

| layer | lines (shared modules by the estimated split) |
|---|---|
| P1 | 13798 |
| collect | 1840 |
| B1 | 455 |
| B2 | 1240 |
| B3 | 3146 |
| B4 | 3156 |
| B5 | 842 |
| B6 | 1275 |
| B7 | 705 |
| unused | 0 |
| total | 26457 |
