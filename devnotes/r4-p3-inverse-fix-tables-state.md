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
- **Report.** At 90 minutes (started 2026-11-26 about 12:00 UTC), at most 150 words.

## Progress

(updated as commits land)
