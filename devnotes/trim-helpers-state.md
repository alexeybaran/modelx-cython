# State note: removing the helper-object paths (experiment)

Brief: branch r4/p3-inverse-trim-helpers from r4/p3-inverse-trunk 09f66ec. Remove, by deletion only, every
compiler line that exists only for modelx_helpers.immutable_inputs Table / Record objects (runtime/_native_inputs.*,
runtime/table_admission.py, the helper parts of table_cells.py and runtime/table_cells.py, TABLE_TYPE lines in
admission, key_facts, tracer). Keep anything pandas reads also use; list those. One commit per file group, tree
importable after each. Delete tests that test helper objects only; list them.

Checks: unit + compiled suites against 09f66ec's own failing set (modelx: say which). Send each commit id to session
session_019ikg7LnpLr1h29pNqMmASM (fork-text model builds: no model >1% above baseline, values equal live; a deletion
that costs speed is reverted and listed).

Limits: new branch push only; if refused, stop and tell the coordinator. No force push, no PR. Plain commit messages,
no internal labels. Report to the coordinator at 60 minutes, at most 200 words: non-test lines removed per commit,
shared pieces kept, doubts.

Ruled out / found:
- Pushed: e579f6b (runtime of helper objects), 6ce2072 (table-cell members), 8980f46 (schema/read reports).
- Fourth commit (math/streams helper identities, frame_row_result, table_schemas, _TABLE_TYPES, enum marker
  comment; two tests keep their standard-library case) is staged in the worktree; the commit was refused by
  the permission check, so it is not committed or sent for measurement.
- The vector alias import in key_facts (pandas integer-key inference) is untouched.
- Kept, not provably helper-only: inline_entries' TableHandle check (generic provider manifest handle type).
- Tests ported to pandas (uncommitted): test_reduction_lift_table_key (passes), test_held_record_reads
  (object column keeps None; verified in a top-level variant: the nested collected Space hits a relative
  `from ._mx_collect import` in collect.py, separate from this work), test_table_cells_api (24c2bf7's port;
  verified with a Reference-named domain, as this venv's modelx 0.33.0 refuses a tuple domain),
  test_table_forms_w1_9 (helper report reasons removed; untyped_asof: exported np.int64 vs compiled float).
- table_matrix: the compiled side drives TableHandle/NativeTable through a Cython driver; a pandas port is a
  rewrite over the prepared-table API (more than two hours).
- Baseline 09f66ec: 777 failed, 23 errors, 857 passed; 740 are `No module named 'modelx_helpers'`, 36 the
  collect-domain TypeError of modelx 0.33.0.
- untyped_asof int/float difference reproduces at 09f66ec (trunk behaviour, not from the deletions).
- Suite at 8980f46 (venv modelx 0.33.0, CPython 3.13 GIL, Linux): 56 failed, 6 errors, 843 passed; no failure
  that 09f66ec did not have; the 30 base failures gone are all in deleted helper tests.
- The measuring session withdrew (sources compared on the runner machines instead).
