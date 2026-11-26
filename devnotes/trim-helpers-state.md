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

Ruled out / found: (fill in as the work goes)
