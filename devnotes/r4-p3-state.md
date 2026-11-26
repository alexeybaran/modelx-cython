# State note (for me after context compression)

**Duties:**
- **Ratchet for session_019ikg7MRVyucYLYquPeKtBb (priority).** On a commit id from it, run
  `/home/user/deps/e3/ratchet.sh <commit>`. Send both it and the coordinator `@parent` the one line; do not
  investigate.
- **Task E-series (from the coordinator):**
  - M1: SEC instructions per policy (Term, MYGA, RILA _US_SEC, upstream lifelib) on c342db9 and on
    r4/p2-collect-7 58ff45d, modelx pr/collect.
  - M2: the 58ff45d test files run on c342db9, pass/fail per file.
  - E: whole-file swaps from 58ff45d on r4/p3-inverse-swaps (from c342db9), one file or natural pair per commit,
    ascending diff size. Keep a swap if both fork figures stay within 1% and SEC is not worse; otherwise revert it
    and record one line. No hand edits.
  - Report at most 300 words at 120 min.

**Limits:** no pr/* pushes, no force push, new r4/* branches only, no vendor or internal labels in commit messages,
no modelx edits.

**Baselines (c342db9, fork text 04d69db, modelx pr/collect 1aa21f4):** Term_UK_S 3,690,576; Dep_FR_S 24,291,555.
An independent gate gives Term 0.994 and Dep 1.000 of the handover reference.

**Method:**
- `e3/build.py` exports with the venv's modelx and builds with the compiler on PYTHONPATH, using the generic sample
  `e2/sample.py`.
- `e3/measure3.sh`: callgrind toggled on the compiled `net_cf` entry and `_mx_new_root`, over the whole book,
  PYTHONHASHSEED=0, ÷ policies.
- `e3/cmp.py`: values against `e3/live/*.json`.
- Venvs: `/home/user/deps/venv-collect` (pr/collect), `venv-handover` (636877b).
- Text trees: lifelib-04d (fork), lifelib-upstream (20d4dae).
