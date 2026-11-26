# State: the integration trunk on the inverse bridge

Branch `r4/p3-inverse-trunk` starts at `r4/p3-inverse-int` 5e67b27: the 4c1bf9a additions, with the integer overflow
checks removed. It is built by real merges, one merge commit each, with no rebase of pushed history.

**Merge order:**
1. r4/p3-inverse-fix-reductions 612f2e1.
2. r4/p3-inverse-fix-tables 9e65ea2.

**Held until released by the coordinator:** r4/p3-inverse-fix-labels (69612fd), and the commits of
r4/p3-inverse-adds after 4c1bf9a.

**Conflicts.** Resolve each by keeping both intents. List every one resolved by hand, and every site where a fix uses
an integer helper the removal took away.

**After each merge:**
- unit and compiled suites, compared with the failing set of the first parent;
- the models (VA_US_S, UC_FR_S, Term_UK_S, IP_UK_S, KLV_DE_S), against c342db9 and equal to live — upstream text
  here, since the fork text cannot be built in this container;
- the integration metric: `git diff --numstat 58ff45d HEAD -- modelx_cython ':!modelx_cython/tests'`, summed
  deletions (221 at 5e67b27).

Send each head to the ratchet session and to the coordinator. Report after step 2 (the coordinator's step 3), at
most 200 words.

## Progress

- 5e67b27: metric 221.
