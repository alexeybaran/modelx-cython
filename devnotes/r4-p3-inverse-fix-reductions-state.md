# State note: reductions and vector fixes on the inverse bridge

Branch `r4/p3-inverse-fix-reductions` from `r4/p3-inverse` c342db9. Task: re-apply the reductions-area and vector-area
fixes of `devnotes/r4-p3-ledger.md` (branch review/r4-p3-reconcile d992517) as separate commits with their tests,
cherry-picked from the old line where possible. Only reductions/vector files and tests. No trimming, no new mechanisms.

Order:
- reductions: bcddcca, bd11687, bef4b7a, c658d20, 5f6898e, 11c2a96 (a58f302 fixes 1616462's loop, which the base
  does not have);
- vectors: 030de71, 1eef6ff, c51c847, 95e73b6, a7bcf56, 7b5c27c, e79b132, 131cd37, e82916d;
- then the test test_reduction_in_vector_subscript.

After each commit, run the full suite. Then run the fork-text ratchet on IP_UK_S and Term_UK_S, or send the commit ids
to session_019ikg7LnpLr1h29pNqMmASM. On any move above 1% or a value mismatch, stop and report.

Progress (commit -> new id, suite):
