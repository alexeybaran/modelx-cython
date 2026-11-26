# Helper-object trim: status

Branch r4/p3-inverse-trim-helpers (from 09f66ec). Commits: e579f6b, 6ce2072, 8980f46, a03f2d9 (deletions), 75831e7 (pandas test ports).

Removed (pushed): runtime/_native_inputs.{pyx,pxd,provider.json}, runtime/table_admission.py, the helper
Record/row/frame branches of table_cells.py and runtime/table_cells.py, the helper schema and read-report
pieces of key_facts, tracer, admission, transformer, usage and builder.
Removed in a03f2d9: helper numeric_math identities in
_native_math.pyx/.pxd, the helper sum marker in _native_streams.pyx, frame_row_result, table_schemas,
_TABLE_TYPES, the enum-marker comment.
Kept: shared integer-key inference (vector alias import in key_facts untouched), frame_reads, runtime/_tables,
generic provider registry, inline_entries' TableHandle check (generic manifest handle type).

Deleted test files, ported to pandas fixtures (75831e7): test_reduction_lift_table_key,
test_held_record_reads, test_table_cells_api, test_table_forms_w1_9 (untyped_asof: exported np.int64 vs
compiled float, same at 09f66ec).
Not ported (helper API or handle identity): test_bound_table_round2_keys, test_native_table_cell_result,
test_native_table_index_semantics, test_builtin_reduction_lowering, test_provider_contract,
helper_contracts/* (3), test_complete_key_helper_modes, test_ordered_index_modes,
test_pandas_preparation_modes, test_table_admission, test_table_keys_provider, test_tracer_table_schema,
table_matrix (10 files; the driver uses TableHandle/NativeTable; a pandas port is a rewrite, > 2 h).

Suites (venv modelx 0.33.0, CPython 3.13 GIL, Linux): 09f66ec 777 failed / 23 errors / 857 passed
(740 missing modelx_helpers); 8980f46 56 failed / 6 errors / 843 passed, no new failure.
BU_DE_S: not checked (the measuring session withdrew).
