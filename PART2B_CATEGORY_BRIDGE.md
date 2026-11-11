# Part 2B — Python-helper/native category bridge

## Status

Implemented on top of Phase 2r1p2a. Version: `0.1.0+phase2r1p2b`.

The Python-key Cell ABI introduced in Part 1 and integrated in Part 2A remains the
execution mechanism. Part 2B adds the missing generic category representation
boundary between a detached Python-key helper and a native recursive Cell.

## Runtime boundary

Inside a Python-key helper, categories remain ordinary Python values (`str`/`None`).
At a call into a native recursive Cell, the existing `_CoreCacheCategoryBoundaryRewriter`
now emits the proven category encoder for categorical arguments. A categorical native
result is decoded when it re-enters Python semantics.

This does not add a second execution graph and does not permit live modelx calls.
The logical Cell call remains compiler-owned `r_<uid>()` on both sides of the boundary.
Direct N0/N1 -> Python-key-helper calls remain fail-closed.

## Category evidence changes

The runtime encoder can only be emitted when the category plan proves the complete
finite domain needed at the selected callsite. Two generic evidence gaps surfaced in
p21 and were fixed without product-specific rules:

1. A category-valued Cell result passed directly to another Cell parameter now creates
   a demand for every proven label of that result at that exact callsite.
2. Parameter-label acceptance is path-sensitive for finite category guards. A label
   handled by an early terminal branch is not rejected because an unreachable later
   statement uses the parameter in a table index or other opaque expression.

Selected-source category-result proof also now:

- allows a finite string/None guard to establish the category of an otherwise opaque
  local on the constrained branch; and
- explores unrelated numeric/boolean branches conservatively instead of abandoning the
  category proof.

These changes preserve fail-closed behavior: labels are still admitted only from proven
finite source/provider domains and every newly demanded label must be accepted by the
callee's reachable source semantics.

## Real fixture validation

All validation uses the frozen Phase 2r1 fixtures and references.

- **p30 LTC_JP_S**: recovered. 8/8 policies; max absolute error `0.0` live and after
  `m.close()`. Python-key helpers: `inc_rate_mth`, `prev_grade`, `inc_rate`.
- **p21 Rente_FR_S**: recovered. 11/11 policies; max absolute error `0.0` live and after
  `m.close()`. Python-key helpers: `tariff_lives`, `annuity_factor`. The source-valid
  mortality label `"mix"` is retained in the native category ABI and recursive
  `(t, table_sex)` memoization is exercised.
- **p38 LTC_KR_S**: recovered. 9/9 policies; max absolute error `0.0` live and after
  `m.close()`. Fourteen Python-key helpers are selected, covering mixed numeric/object
  helper keys and category crossings.
- **p52 CI_UK_S regression**: still passes. 7/7 policies; max absolute error
  `3.637978807091713e-12` live and after model close.

Source-tree regression suite: **18/18 tests pass**.

For validation only, generated C was compiled directly with Clang `-O0` because GCC
compilation in this container was anomalously slow. This is not a compiler requirement
or production implementation change.

## Aggregate coverage

The full 71-root census has **not** been rerun in this milestone. The last published
complete-census aggregate therefore remains **53/71 from Phase 2r1**. The isolated
fixture results prove p21/p30/p38/p52 behavior but should not be converted into a new
published census count until the complete harness is rerun.

## Remaining work

The next logical milestone is census/performance consolidation rather than another ABI
feature: run all 71 roots, record preparation/compile/bind/calculation timings, inspect
any newly exposed failures, and then decide which Python-key helpers are worth promoting
to native key ABIs for performance.
