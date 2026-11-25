# R4 C1: helper table lane and helper providers removed

Base `r4/s1-zeroing` 3bc7be5. Map: `r3/p2-remove-legacy-helpers` (24c2bf7, 3ea62f9) and
`r3/helper-free-qualification` (c73aeb5), re-applied by hand.

## Expectation (written before any measurement)
Generated C of every model module is unchanged: instructions equal within 0.2%, values equal live.
Only the provider copies `_native_math.pyx` and `_native_streams.pyx` change (a guard comparison
that was always redundant), so their t0 hashes may differ; a model-module hash change is a defect.

## Lines removed (3bc7be5..head without this report, `git diff --numstat`)
| area | removed | added |
|---|---|---|
| non-test | 3,742 | 67 |
| test | 4,054 | 11 |
| docs | 440 | 48 |
| total | 8,236 | 126 |

Non-test by topic: helper table lane 3,573 (`_native_inputs` pyx/pxd/manifest 3,020,
`table_admission` 149, compiler lane and residue 404); providers 170 (helper-package fallbacks
of the math and stream providers 34, registered helper providers 136).

## Configuration keys removed
- Spec key `held_table_handles` (read by nothing); manifest field `handle` (typed handles); the
  registration attributes `__modelx_cython_{helper,scalar_result,call_layout,result_type}__`.
  Comments of `typed_table_reads`, `strict_tables`, `prepared_tables` updated (no behaviour change).

## Tests deleted (only exercised removed code; all import the helper package except where noted)
- integration: table_matrix/{cases_w1_2, cases_w1_3, done_w1_2, done_w1_3, squares, w1_2_driver.pyx,
  w1_2_support, test_grid_index_d86, test_table_handles_w1_2, test_table_matrix}, test_bound_table_round2_keys,
  test_diag_jp_forms, test_held_record_reads, test_native_table_cell_result, test_native_table_index_semantics,
  test_reduction_lift_table_key, test_table_forms_w1_9.
- unit/compiled: test_builtin_reduction_lowering, test_native_scalar_math (module math stays covered
  by test_native_module_math), test_public_object_return_spec, helper_contracts/{test_grouped_report_metadata,
  test_raw_scalar_preparation, test_table_operations}, test_complete_key_helper_modes, test_ordered_index_modes,
  test_pandas_preparation_modes, test_table_admission, test_table_keys_provider, test_tracer_table_schema,
  test_provider_handle_manifest (typed handle field), test_provider_contract (registration).
- single tests: test_audit_runtime::test_prepared_helper_resources_do_not_break_model_pickling,
  test_audit_contracts::test_helper_capability_shape_is_validated_at_registration,
  test_provider_manifest::test_function_provider_does_not_need_a_manifest, test_helper_members:
  aliases_share_capabilities, invalid_member_capability, shared_capabilities_reach_namespaces,
  scalar_result_is_physical_abi_metadata, scalar_result_rejects_undeclared_physical_types.

## Tests edited
- test_audit_runtime::test_scope_overrides_aliases_and_arithmetic: plain `float`/`sum` instead of helper
  callables (as c73aeb5); assertions unchanged.
- test_enum_by_type: legacy-marker case removed (marker came from the helper package).
- test_table_cells_api: unused helper import and Reference removed.
- test_helper_members::test_foreign_receiver_boundaries (renamed): direct-Reference helper assertion
  removed (that resolution is gone); other assertions kept.
- test_native_builtin_math: vacuous assertion that `_native_inputs` is not installed removed;
  test_generator_builtin_lowering and test_enum_by_type: docstrings.

## Gate (t0, t2): NOT RUN
Adding the benchmark toolkit repository was denied by a permission check; per the rules no other
route was tried. t0 and t2 were not run on 3bc7be5 nor on head: the expectation is unverified.
Without the pinned modelx 636877b (not public) no model can be translated here (export protocol 2).

## Compiler tests (same machine: Python 3.13, upstream modelx fec516d, lifelib 0.17.4)
`pytest -n 4 modelx_cython/tests integration_tests`
| | passed | failed | errors | skipped |
|---|---|---|---|---|
| 3bc7be5 | 685 | 194 | 65 | 10 |
| head | 650 | 179 | 44 | 9 |

No test present on both sides changed outcome; the difference is 74 deleted cases (base: 36 passed,
16 failed, 21 errors, 1 skipped) and 2 renamed ones. Most failures are environmental (export
protocol 2, `collect_over`, helper package missing): compiled tests, edited ones included, do not run here.

## Not removed, and why
- Not taken from the reference: in `key_facts` it dropped an `is_int` branch and reworded a reason
  that prepared-frame lowering still reaches through its `_int_index_read` hook; both kept, only the
  helper schema code was removed.
- `PANDAS_RECEIVER` report text still says "not a helper Table" (reference changed it): kept so
  reports and test_pandas_read_report stay unchanged.
- Comment naming `_native_inputs` inside the verbatim C block of `_native_vectors.pxd`: it is copied
  into generated C, so it is left to keep that C byte-identical.
- Math provider operations taking a callable (`erf`, `exp`, ...): still reached without any
  registration by a module Reference member that is a math builtin under another name.
- Object providers (`__modelx_cython_type__` manifests): generic extension, not part of the lane.
