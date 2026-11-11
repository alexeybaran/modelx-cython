# Detached Python helper work split — Part 1 implemented

## Why the work is split

The remaining p21/p30/p38/p52 failures all need a compiler-owned Cell memo ABI
that can accept ordinary Python keys (strings, mixed int/float objects, tuples)
while keeping recursive calls detached from live modelx objects.

Implementing runtime storage, planner admission, formula rewriting, category
boundaries and four-product validation in one change made the failure surface too
large. The work is now split at a stable architectural boundary.

## Part 1 — implemented here

Part 1 adds the **runtime primitive only**:

- new `modelx_graph.python_key_memo_native` module;
- compiler-owned `r_<uid>()` wrappers backed by a Python `dict`;
- exact Python key semantics:
  - zero args -> Cell-local scalar sentinel (`None` in the Cell-local dict),
  - one arg -> argument object directly,
  - multiple args -> ordinary Python tuple;
- Python-object values;
- recursive calls with different keys;
- same-key re-entry / cycle detection via an active-key set;
- lazy per-Cell cache invalidation when the shared `_epoch` changes;
- explicit clear hook for binding/reset and future native epoch-wrap handling;
- per-Cell hits, misses, entries, active-key count and epoch diagnostics;
- explicit selection API (`selected_uids`) so this runtime cannot accidentally
  change execution classification before Part 2.

The existing mixed planner and `_mixed_recursive_uids()` are intentionally
unchanged. No current census product is routed through the new runtime yet.

## Part 1 validation

`tests/test_python_key_memo_native.py` compiles a real Cython probe and checks:

1. recursive Fibonacci-style Cell calls are memoized correctly;
2. a repeated call in the same epoch is a hit;
3. `65` and `65.0` share one cache entry, matching Python dict/modelx semantics;
4. multi-string keys such as `(sex, smoker, column)` work;
5. same-key recursive re-entry is detected rather than recursing forever;
6. advancing `_epoch` lazily drops the prior policy cache;
7. explicit clearing resets all Python-key memo storage.

Full source-tree test result after the change: **12 passed**.

## Part 2 — next implementation

Part 2 should wire the new primitive into detached-P planning and codegen:

1. Build an explicit detached Python helper closure before external-reference
   rewriting.
2. Classify owner-local Cell edges as either:
   - existing native `r_<uid>()`, or
   - Python-key `r_<uid>()` from Part 1.
3. Keep genuine external/provider references on the current prebinding path.
4. Emit `pycalc_<uid>()` bodies for Python-key Cells and rewrite their Cell
   dependencies to the appropriate `r_<uid>()` implementation.
5. Apply category encode/decode only when crossing between Python semantics and
   the native ABI.
6. Reject any direct N0/N1 -> Python-key edge; Python-key Cells may be entered
   only from a GIL-held detached-P body.
7. Hook `_clear_python_key_memos()` into model reset and the rare native epoch
   wrap path.
8. Validate in order: p30 -> p21 -> p52 -> p38, then run the full 71-root census.

The target remains one logical recursive Cell ABI. `python_key_memo_native`
changes only the physical memo representation for Cells whose keys require
Python semantics; it does not reintroduce live modelx execution.
