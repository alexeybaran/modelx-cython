# A frame the item computes itself: prepared only for computed keys

**Rule.** A frame or Series an item computes itself is prepared per item only for a formula with parameters that
reads it by a computed key or walks it. Reads by literal keys, or by a choice between literals, stay on pandas, and
so do fields of a row cell over such a frame. Frames of a Space without items are prepared once per model, as
before. The rule is decided from the text alone, not from call counts.

**Why.** Slice 1 prepared an item's own `model_point()` Series per item for two literal-key reads that run once per
policy (BasicTerm_SE `duration_mth`, `pols_new_biz`). That preparation was about 84k instructions per policy. Only
three of the 57 regional models change, and all in the same direction. Part 2 of the follow-up task (mask,
list-key and `xs` reads) was not built: the coordinator withdrew it.

| measure | line A 90da50b | r4/p3-views | this branch |
|---|---|---|---|
| BasicTerm_SE, a fresh policy (instructions inside `pv_net_cf`) | 887.6k | 974.6k | 891.2k (generated code identical to line A) |
| BasicTerm_S, a fresh policy | 19.61M | - | 19.13M (generated code identical to line A) |

Under this rule:

- **Regional models whose generated code changes (3 of 57):** Rente_FR_S, PA_UK_S and SPIA_US_S. In each,
  `death_mth(life)` reads `model_point()["a" if life == 1 else "b"]` and no longer prepares the item's row. The
  change is one fewer per-item preparation, so it can only be faster.
- **Unchanged:** every other model, including KLV_DE_S, Term_US_S, UC_FR_S, VA_US_S and IP_UK_S. Their generated
  code is identical, so their times are the slice-2 figures. The generated files of Term_US_SEC and MYGA_US_SEC are
  byte-identical.
- **Suite:** 583 passed, 15 skipped.
- **Test change:** `integration_tests/test_forms_own_frame.py` now states the rule. Its first case keeps slice 1's
  assertions, because a computed key is still native. The second case adds literal-key, choice, computed-key and
  local reads.

A first version that never prepared an item's own frames changed 15 models. It made four US models slower:
`coi_rate_guar(t)` reads `coi_rate_scale()[y]` with a per-step key. So that version was not kept.

## Two repairs on forms already ported

Both are their own commits. Generated code is otherwise unchanged, and the suite passes (583 passed, 15 skipped).
The six models measured below equal live on every policy.

- **Exact `str` and `int` key parts take a fast path.**
  - In `code_obj`, an exact `str` key is one dict lookup. An exact `int` key is one C conversion with an overflow
    flag, instead of two Python comparisons.
  - The numpy types other values are tested against are held in module fields.
  - `is_label` answers for an exact `str`, `int` or `float` without a call.
- **The hold's handler no longer forces the overflowcheck fallback.** It is now named `_mx_unbound`, and the integer
  guard passes it. That removes one fallback in KLV_DE_S (17 to 16). The other fallbacks there come from operands
  of unknown C type, not from the holds.

Instructions per policy, from callgrind collecting only inside `net_cf`, one pass over the book, generic sample
(this method's totals are higher than the census's):

| model | before (r4/p3-views) | after | change |
|---|---|---|---|
| Term_UK_S | 7.02M | 6.59M | -6.1% |
| KLV_DE_S | 5.25M | 5.22M | -0.5% |
| VA_US_S | 42.32M | 42.44M | +0.3% (noise) |
