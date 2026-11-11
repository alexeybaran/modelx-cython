# Part 2A — detached Python-key helper Cells

## Scope

Part 2 was deliberately split again to keep the integration step bounded.

- **Part 2A (this package):** form a compiler-owned detached helper closure for
  P Cells that need ordinary Python argument semantics, emit real
  `pycalc_<uid>()` bodies, and route helper calls through the Part-1 Python-key
  `r_<uid>()` memo ABI. Existing native/category boundary behavior is left
  unchanged and conservative.
- **Part 2B (next):** support category conversion at Python-key-helper ↔ native
  ABI crossings, complete the remaining failure fixtures, wire any required
  ranked-driver support, and run the full census/performance suite.

The split is architectural, not product-specific. Part 2A is intended to remain
in the final design.

## What Part 2A implements

### Separate physical helper closure

`_mixed_recursive_uids()` now tracks two distinct recursive closures:

1. the existing native-key recursive closure; and
2. a Python-key helper closure for detached P Cells whose parameters cannot use
   the native recursive key ABI.

Both retain the same logical Cell identity: callers use `r_<uid>()`. The
physical memo implementation differs.

A native N0/N1 formula is not allowed to call a Python-key helper directly.
That edge fails closed. A Python-key helper is reachable only from a detached P
body that is already executing with the GIL held.

### Detached helper candidate discovery

A new fixed-point helper-candidate pass admits non-bridgeable scalar P formulas
when they can be reconstructed safely from:

- other Python-key helpers;
- existing native recursive Cells that require no new category conversion;
- materializable detached references/providers; and
- source-preserved Python expressions.

Live modelx Cell/Space execution remains forbidden after binding/model close.

### Generated helper bodies

Each selected helper is emitted as a real Cython function of the form:

```text
pycalc_<uid>(policy_i, object arg0, object arg1, ...)
```

Its Cell wrapper is the Part-1 Python-key `r_<uid>()`, which provides:

- Python dict key semantics;
- recursive memoization;
- same-key cycle protection;
- per-policy epoch invalidation;
- diagnostics and explicit clearing.

Helper-to-helper calls are rewritten to the Python-key `r_<uid>()` wrapper.
Helper-to-native calls continue through the existing native `r_<uid>()` path.

### Cache lifecycle

Python-key helper caches share the native policy epoch. They are lazily cleared
when the epoch changes, explicitly cleared during input rebinding/reset, and
cleared safely on uint32 epoch wrap. This keeps memo contents policy-local
without creating a dict per policy.

## Deliberate Part-2A boundary

A helper candidate is rejected if calling an existing native recursive Cell
would require a new category encode/decode operation. This is the main Part-2B
feature.

For example:

```text
Python-key helper: inc_rate(t, grade="Ann")
        |
        +--> native Cell: grade_share(grade)
```

`grade_share` has a native category ABI. Part 2A does not yet encode the Python
string `grade` to that native category code. The helper therefore remains
fail-closed rather than falling back to a live modelx call or silently changing
semantics.

The same restriction applies to native categorical return values that would
need decoding when entering Python semantics.

## Validation

### Source-tree tests

**16/16 tests pass.** This includes:

- all prior Part-1 tests;
- helper-closure formation;
- rejection of direct N1 → Python-key-helper edges;
- rejection/deferment of unimplemented category-boundary crossings; and
- a compiled end-to-end Cython generator test where a detached P formula enters
  a recursive Python-key helper, the helper calls itself, and it also calls a
  native recursive Cell.

### Real frozen failure fixture: p52 / CI_UK_S

Part 2A recovers p52 without any product-specific rule.

Selected Python-key helper:

```text
pivot_interp(x, sex, smoker, column)
```

Observed result for the 7 frozen policies:

- compile succeeds;
- maximum absolute error versus the frozen reference:
  **3.637978807091713e-12**;
- after closing the live model and resetting/rerunning, maximum absolute error
  remains **3.637978807091713e-12**;
- Python-key memo diagnostics for `pivot_interp`:
  **390 misses, 4,290 hits, 100 final entries** on the observed final epoch.

This directly proves that the new helper path can preserve Python string/object
arguments and detached provider data while maintaining the no-live-modelx
invariant.

### Remaining frozen failures

- **p30 / LTC_JP_S:** remains blocked because `inc_rate(t, grade)` needs to call
  native `grade_share(grade)` across a category argument boundary. This is an
  intended Part-2B case.
- **p21 / Rente_FR_S:** remains blocked because the Python-key
  `tariff_lives(t, table_sex)` helper ultimately calls a native mortality Cell
  whose `table_sex` parameter is categorical. This is also an intended Part-2B
  case.
- **p38 / LTC_KR_S:** full fixture validation was not completed in this
  milestone. Its previously identified helper graph contains mixed Python
  numeric/object keys and category-like arguments and is retained as the broad
  Part-2B stress case.

The full 71-root census was **not rerun** in Part 2A. Do not publish a new
census total from this package. One previously failing frozen product is proven
recovered in isolation; the historical full-census baseline remains the last
verified aggregate result until Part 2B runs the census again.

## Main implementation points

The principal changes are in:

- `modelx_graph/mixed_recursive_native.py`
  - `_PythonKeyMemoCallRewriter`
  - `_native_detached_recursive_catalog`
  - `_python_key_helper_candidate`
  - `_python_key_helper_candidates`
  - extended `_core_cache_cython_p_candidate`
  - split closure construction in `_mixed_recursive_uids`
  - `MixedRecursiveProgram.python_key_*` fields
  - `_emit_python_key_formula`
  - `_integrate_python_key_memos`
- `modelx_graph/source_preserving_native.py`
  - Python-semantics recognition for Python-key helper UIDs.
- `tests/test_python_key_helper_part2a.py`
- `tests/test_python_key_helper_codegen_part2a.py`

## Explicit limitations retained for Part 2B

1. No new category encode/decode at helper/native call boundaries.
2. Python-key helpers are scalar-result helpers only in this implementation.
3. The ranked driver explicitly fails closed if a Python-key helper reaches it;
   it has not yet been wired to the helper runtime.
4. No product-specific workaround is permitted.
5. The live-modelx safety gate remains mandatory.

## Recommended Part 2B implementation order

1. Extend helper/native call analysis to carry proven category-domain metadata
   for both arguments and results.
2. At generated helper → native `r_<uid>()` call sites, encode Python category
   values to native category codes using the existing compiler-owned domain.
3. Decode native categorical results back to their Python value only when they
   cross into a Python-semantic helper body.
4. Keep native → Python-key-helper direct calls forbidden; the transition must
   remain behind a detached P/GIL boundary.
5. Revalidate in increasing complexity: p30 → p21 → p52 regression → p38.
6. Wire ranked-driver helper support only if a real selected closure requires
   it; preserve fail-closed behavior otherwise.
7. Run all source tests, regression products, then the full 71-root census and
   performance breakdown.
