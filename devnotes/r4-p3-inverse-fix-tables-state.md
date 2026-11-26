# State: table fixes on the inverse bridge

Branch `r4/p3-inverse-fix-tables`, cut from `r4/p3-inverse` c342db9: the proposal line with 633c9be's compiler tree,
and the proposal's collect.

Task (from the coordinator):

- **What to re-apply.** The table-area F rows of `devnotes/r4-p3-ledger.md`, on review/r4-p3-reconcile d992517:
  rows 21, 22, 23, 24, 27, 43, 49, 50, 51, 52, 53 and 54, plus view row 48.
- **Also bring, if the defect exists here.** The text-key and bounds repairs of r4/p3-reads. Show the failing case
  first.
- **How.** Each fix is its own commit with its tests. Try a cherry-pick from the old line first.
- **Tests to bring if absent.** test_prepared_tables, test_table_runtime_contract, test_forms_view_keyerror.
- **Limits.** Touch only table/read files and tests. No trimming, no new mechanisms. Do not bring the BasicTerm
  structural rule (f551058) unless a model needs it, and say which model.
- **After each commit.** Run the suites. Run the fork-text ratchet on VA_US_S and UC_FR_S if they can be built here.
  Otherwise send the commit ids to session_019ikg7LnpLr1h29pNqMmASM.
- **When to stop.** Stop and report on any move above 1% or any value mismatch. Stop if the push is refused.
- **Report.** At 90 minutes (started 2026-11-26 14:13 UTC; due 15:43), at most 150 words.

## Progress

- **Cherry-picked from the old line, with -x, in their original order:** e1b0642, c2c8c74, e37f3e8, c69b0f9,
  dc43e8f, 23c4819, 288fa5a, a0230c9, 9fa932f, 07a3b8a, cff460d, 6154c3e, c6bc317, 34c84ca and 6551a27.
- **Conflicts:**
  - 23c4819: `test_table_runtime_contract.py` is absent here, so it was taken whole from 23c4819. In
    `table_cells.py` its `_reset_lines` was dropped, because it imports the serial collect this tree removed.
  - c6bc317: one comment line beside `row_float`.
- **Not applied:** 856b984. That merge repair assumes the result binder was removed. This tree still has
  `native_result_binder`, so the old line is right here.
- **Not applicable:** the hold-handler repair of r4/p3-reads. This tree's integer guard has no handler check.
- **To do:** the text-key and bounds repair of r4/p3-reads applies to `runtime/_tables.*`. The case is being measured
  on Term_UK_S, upstream text, before and after.
- **Baseline (c342db9).** Full suite under modelx 0.33.0. 20 test modules fail to import (`modelx_helpers` absent).
