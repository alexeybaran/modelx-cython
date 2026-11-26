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
- **Figures vs c342db9** (upstream text, all equal to live):
  - f67859b: Term +0.49%, UC −0.08%, VA +0.09%, IP +0.10%, KLV −0.14%.
  - e4cdfc9: Term +0.55%, UC +0.31%, VA +0.13%, IP +1.02%, KLV +0.19%.
  - The IP step is already in fix-tables alone (9e65ea2: +1.05%).
- **7af873a: repair for the IP_UK_S step.**
  - Cause: `scopes.has_scope_effects` counted the holds' unnamed `except Exception:` and the `_mx_okey_` walrus as
    scope effects, so every formula with a table hold lost the reduction, rounding and row passes.
  - Fix: an unnamed handler and a walrus of an `_mx_` name no longer count.
  - IP_UK_S −0.10%.
- **Six regional models (fork text).** The step was in the fix-tables merge. Closed at d342f58, all within 3% of the
  reference.
  - Revert of 75cc21a (838d0a0; an allowed sample-count rule was restored). That reverts c9db767 too.
  - 21f0a0e: a raising hold is remembered as tried.
  - d342f58: a vector the compiler fills is no frame edit (Child_KR_S, LTC_KR_S, BU_DE_S).
- **Coordinator's order after that.**
  - (1) 53dcea2: text guards from the recorder experiment, plus f988be7 (numpy type of a label-checked read).
  - (2) 3b48dad: merge of fix-other 28aad6f.
  - (3) ac1ad83: merge of adds up to dbc7132. Hand conflicts: table_cells.finish uses `without_unused_runtime`;
    test_space_item_results keeps both assertions.
- **Suites per step:** no new failure at steps 1 and 2. Step 3 has one new failure,
  test_native_module_math::test_module_math_bound_at_build_time. It fails on the adds branch alone: the report no
  longer lists m_erf in native_helpers. Reported; stopped there.
