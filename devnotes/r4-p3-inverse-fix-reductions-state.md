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

Done, head 612f2e1 (accepted by the coordinator; the ratchet decides on the figure with item allocation):
- Cherry-picked with -x, in order:
  - 3c35b67 (d5758f6, the live/compiled test helper);
  - reductions: 5a48113 bcddcca, 8cfe1d4 bd11687, 03fb477 bef4b7a, 7da1c27 c658d20, 691689e 5f6898e,
    75b656c 11c2a96;
  - vectors: 1877b4c 030de71, b289ea3 1eef6ff, 989c1ea c51c847, fe2d9a1 95e73b6, 3239209 a7bcf56, 084035b 7b5c27c,
    03083e6 131cd37, b04507f e82916d.
- d6b29f9 (e79b132) was cherry-picked and then reverted in 612f2e1. It is a speed change, not a fix. It broke
  test_vector_ref_reads at a frame_reads.py:1120 build crash against the tree's vector_ref_reads. Row 36 is not
  applied.
- Not applied: a58f302, which fixes 1616462's loop; the base does not have that loop.
- test_reduction_in_vector_subscript came with bd11687.
- Suite: base c342db9 has 782 failed, 819 passed. Head 612f2e1 has 782 failed, 835 passed. The failing sets are
  identical. Those failures are environment: modelx_helpers missing, no collect-enabled modelx, the libcst version.
  The per-commit suites were stopped after the base and b04507f, at the coordinator's request.
- Ratchet on 612f2e1 against c342db9, with alloc (without alloc in brackets); all equal live:
  - Term_UK_S -0.08% (+1.06%);
  - Dep_FR_S +0.03% (-0.08%);
  - IP_UK_S -0.02% (-0.04%).
- Commit messages keep the old line's text, as the cherry-picks carry it.
