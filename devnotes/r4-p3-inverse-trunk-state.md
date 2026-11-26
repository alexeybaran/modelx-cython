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
- **f67859b, merge of fix-reductions 612f2e1.** No conflicts, no removed helper used. Metric 221.
- **e4cdfc9, merge of fix-tables 9e65ea2.** Metric 221.
  - Hand conflict in table_cells.py: the trunk's rewrite_class and RenameMembers no longer take integer Reference
    paths.
  - The table fixes' `_renew_on_new_model` and `_drop_unused_import` are kept.
- **09f66ec, merge of fix-labels 00f501c** (released as step 4; kept as its own merge so it can be backed out).
  Metric 221.
  - Hand conflict in table_cells.finish: the label fix's foreign-IntEnum check, with no integer paths, and the
    dropped import kept.
  - Removed helper: `_foreign_members` unwrapped `_mx_integer_result`. The unwrapping is dropped, since nothing
    wraps arguments now.
- **Running:** suites of 5e67b27 and f67859b; models for f67859b and e4cdfc9. Next: suites of e4cdfc9 and 09f66ec,
  models for 09f66ec.
