# Fast compiler for unchanged models (fast SEC)

Branch `r4/p1-fast-sec`, against upstream modelx-cython `d22426f`. It compiles models exported by upstream
modelx with no change to the model and runs them several times faster than upstream's compiled package,
with the same values: the three SEC models built and swept here (Term_US_SEC, MYGA_US_SEC, SPIA_US_SEC) equal
live modelx on every point, and all twelve SEC models translate on unmodified upstream modelx (only the SEC
DV01 example's dynamic Space reference needs a modelx change, outside this pull request). Stack: CPython 3.10 to 3.14 (free-threaded 3.13t and 3.14t too), Cython 3.3.1,
modelx and lifelib from upstream.

## What the compiler adds, by mechanism (non-test lines, 13,029 against upstream's 6,300)

| mechanism | modules (lines) |
|---|---|
| typed entries: a dense cache per Cell over its declared or sampled domain, checked once at entry; private formula entries on an exact receiver; locked Spaces | entry_pipeline 522, inline_entries 348, cell_entries 150, locked_entries 202, item_roots 48; transformer +494 (with the fast item construction, also under dynamic References), builder +167 |
| native storage: numeric key caches, unwritten-value allocation, held arrays and holds | _numeric_cache .pyx/.pxd 236, _wide_numeric_cache .pyx/.pxd 203, numeric_cache 146, native_sources 31, value_storage 82, held_arrays 125, holds 77 |
| integer guards: C `+ - *` and negation set an overflow flag tested once per formula; checked conversions at arguments, results, ItemSpace keys and References | integer_boundaries 1,244; _mx_sys.pxd +548 (with the other runtime helpers) |
| powers: real powers on C `pow` with a one-entry memo; argument order kept where Python's semantics need it | powers +362, ordered_arguments 117 |
| held items of a shared Space (the SEC DV01 example reads one) | space_results 195 |
| cache sizing from the declared or sampled range | automatic 107 |
| build infrastructure: setup (fp-contract off), Cython errors named by Cell, re-parse check, build report, shared CST helpers (definite assignment) | build 220, reparse_check 105, exceptions 20, admission 228, syntax 132, scopes 260 |
| grown upstream modules | tracer +120, config +177, usage +28, parser +14, typedefs +15, cli +5 |

## Measurements

Instructions per policy (callgrind; one pass = build each point, `result_cf()`, delete it; ipp = (I(4 passes) -
I(1 pass)) / (3 x points), inclusive cost of the three entries), same machine, Python 3.14t:

| model | trial cut (start) | this branch | change |
|---|---|---|---|
| Term_US_SEC | 7.601M | 7.286M | -4.1% |
| MYGA_US_SEC | 8.650M | 8.416M | -2.7% |
| SPIA_US_SEC | 8.943M | 8.846M | -1.1% |

The trial cut ran about 4.0x (Term) and 4.4x (MYGA) fewer instructions than upstream's compiled package.
Steps: the flag design +0.5 / +0.7 / +2.1%; removing the divisibility test, item locals and dead paths
-0.1 / +0.1 / 0.0%; range-based cache sizing +0.3 / +0.04 / +0.1%; reading a fitting int at the checked
64-bit boundary without exception handling -4.8 / -3.6 / -3.2%. Values: lifelib's own C4 check (every point
swept, every cached value and raise compared, the compiled run in a child) at rel 1e-12 finds no difference
on any of the three.

An alternative integer policy (C arithmetic only where operand ranges prove it fits, Python ints elsewhere)
was built and measured: +3.7% Term, +15.8% MYGA and SPIA. The flag design replaces it.

## Integers

- Checked by default (`integer_checks: "checked"`): each formula's C integer `+ - *` and unary minus go
  through helpers that return the wrapped result and set a flag (GCC/Clang `__builtin_*_overflow`; MSVC
  unsigned arithmetic, `_mul128` on x64, `__mulh` on ARM64). The formula tests the flag once before it
  returns, and before a loop or call that receives a flagged value, and raises OverflowError naming the Cell.
  A formula with an operation whose C type cannot be told keeps Cython's overflowcheck for itself alone
  (`overflowcheck_fallback` in `_mx_integer_checks.json`). `"unchecked"` is the opt-out: plain C, as upstream.
  In the twelve SEC models the fallback covers mostly `Data` table builders (run once per model) and 0 to 12
  `Projection` cells per model (Term 1, MYGA 0, SPIA 4; VA 3, RILA 9, IUL 12); no SEC model has an integer power
  or shift, whose helpers raise per operation.
- The mode never depends on the sample: the sample-magnitude switch and the `auto` mode are gone.
- `//`, `%`, shifts and integer powers of C integers keep Python's results through exact helpers.
- Arguments, results, ItemSpace keys and References typed as C integers cross a checked conversion: a
  float or an int beyond 64 bits raises naming the Cell or Reference; nothing is truncated.
- A local a read can reach unassigned (a zero-trip loop, an untaken branch) stays a Python object, so it
  raises UnboundLocalError as live does (a C local has no unbound state).

## Differences from upstream behaviour

1. Integer arithmetic beyond 64 bits raises OverflowError naming the Cell (upstream wraps silently).
2. A Cell called outside its declared or sampled range raises IndexError naming the Cell, each argument's
   range and the spec key `cells_param_size`; upstream's dense caches raised a bare `IndexError("array index
   out of range")`, and its dictionary caches accepted any argument. The range check of dictionary caches is
   part of the entry machinery (kept as it is by the owner's decision).
3. A float, NaN or an int beyond 64 bits passed where an integer was traced (argument, ItemSpace key, Reference
   write) raises TypeError or OverflowError naming it; upstream truncated a float written to a Reference.
4. A variable left unbound by a loop that never ran raises UnboundLocalError; upstream returned 0.
5. An integer power of C integers goes through a checked helper that raises beyond 64 bits (upstream:
   Cython's C power, which wraps); a real power uses C `pow` with a one-entry memo per Cell (`_mx_memo_pow`).
6. The generated package imports nothing from modelx_cython at run time (tested with it blocked).

## Tests

- Upstream's own tests at `d22426f`, untouched, on this branch: 247 passed, 7 skipped, 4 failed. The failures are the
  deliberate differences 2 (`test_various_types` x2: a dictionary cache outside the sampled range raises)
  and 5 (`test_fractional_power`, `test_mx2cy_with_uslib`: the source text of integer and memoized
  powers); the repository's copies assert the new forms and pass.
- The repository's suite on upstream modelx (Python 3.14t, local): 560 passed, 9 skipped; the skip added here,
  `test_dynamic_ref_items`, needs an export that applies dynamic References (the DV01 example's modelx change).
- New here: `test_upstream_regressions.py` (runs with modelx_cython blocked; range errors name the Cell,
  argument and spec key; unbound locals; no truncation of floats; a lambda and a generator shadowing a
  typed parameter), `test_flagged_integer_arithmetic.py`, `test_integer_overflow_model.py`,
  `test_dense_by_range.py`, and the range check at the extremes of the signed range.
- The eleven compiled tests that built their model with development modelx's `collect_over` run as
  direct-route twins on upstream modelx: six from `r4/integration`; `test_compiled_contract` written here
  (same model without collect, `parallel_workers` and ItemSpace formula References; collect totals become
  sums of direct items); the collect route of `test_value_storage` and the removed-key test of
  `serial_collect` belong to the collect layer.
- Changed assertions, each for a listed reason: a float written to an integer Reference raises (it asserted
  the truncation); `(t - 1) % n` and `(t + 1) % step` at the signed range's ends raise naming the Cell (the
  divisibility test is removed); two formulas using vector or reduction syntax take the overflowcheck
  fallback here (those are bridge layers); storage follows the range, not the sample's call counts.
- CI (GitHub Actions, unit, compiled, allocation and SEC lanes; Term, MYGA and SPIA compared with live):
  (run 2cf4691 in progress; filled in when it completes).

## What a user declares

Only `cells_param_size`, where a run calls a Cell with arguments beyond those the sample reached (every SEC
model's `spec_sec.py` does). Everything else is chosen from the sample's types and the declared or sampled
ranges; `integer_checks: "unchecked"` is an opt-out, not a requirement.

## Removed from the trial cut

The typed item locals, the native divisibility test (+0.7% on MYGA, -0.1% on SPIA without it), the
`auto` integer mode with its sample-magnitude switch, and the two cache-sizing rules that read the sample's
call counts (and the tracer's per-receiver counting).
