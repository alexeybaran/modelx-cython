# Review: reductions (reviewer a)

Tree b326334. Stack: free-threaded CPython 3.14.6, Cython 3.3.1, modelx 636877b; CPython 3.13.18 for one check.
`R` = `modelx_cython/reductions.py`, `P` = `modelx_cython/_float_sum_state.pxd`. Line numbers are of b326334.
Everything marked *measured* or *compiled* was run here; the rest says "read only". No lifelib model was built.

Evidence used below:
- **Adversarial model** (56 one-line cells, live against compiled, type + repr + exception): 4 defects, section 4.
- **Micro model** (10 cells, 360,000 calls each from Python, loops of 0-120 items, best of 5-7 runs, ms).
  Untouched cells move by up to 8% between runs: that is the noise.

| build | s1 `return sum(f(i) for range)` | s2 two cells | s3 `f(t) * sum(..) + k` | s4 int sum | s5 `if` | m1 max | m2 min | a2 any | e1 tuple of pairs |
|---|---|---|---|---|---|---|---|---|---|
| as is | 344-352 | 371-385 | 414-446 | 338-347 | 335-356 | 312-331 | 273-296 | 273-282 | 272-288 |
| reductions off | 1525 | 1590 | 1590 | 1398 | 1070 | 1636 | 575 | 345 | 442 |
| specialised lanes off | 392-404 | 398-407 | 407-428 | 333-363 | 356-372 | 319-330 | 270-308 | 276-303 | 272-293 |
| **(a)**: lanes off + 8-line two-double state | 322-331 | 353-357 | 414-426 | 338-346 | 338-355 | 313-332 | 284-299 | 274-287 | 270-330 |
| lanes, hoist, unroll, all/any, tuple-int off | 330 | 357 | **1604** | 360 | 349 | 345 | 303 | **368** | **396** |

- **Source count** (ast over the 46 `*_S` + 12 `*_US_S` model folders of lifelib 04d69db, original text, not the lean
  text): 541 one-argument sites: `all` 297, `sum` 218, `min` 15, `max` 8, `any` 3. Generator argument 486, direct
  iterable 55. First iterable: `range` 366, tuple/list literal 52, `zip` 13, other 55. Whole `return`/assignment 452,
  nested in a larger expression 89 (of these 9 behind `and`/`or`/conditional expression, 6 inside another
  comprehension). `return all(...)` is 254 sites and nearly all are check cells evaluated once per run
  (`all(abs(resid(t)) <= tol for t in range(proj_len()))`).

## 1. Mechanisms

| # | Mechanism | Where | Lines | Lost without it |
|---|---|---|---|---|
| M1 | General lowering: `sum/min/max/all/any` of a generator, list comprehension or iterable as a loop in the cell, typed `range` variable, renamed variables | R:137-329, 385-408, 433-454, 488-523 | ~300 | the area's speed: 3.3x-5x on s1-s5, m1 (table); the typed loop variable is what lets cell reads inline |
| M2 | `PySum`: built-in `sum` of 3.14, branch for branch (int, float compensated, complex, object) | P:36-219 | 184 | equality with live: Cython's own inlined `sum(genexpr)` adds objects uncompensated |
| M3 | Kinds of elements and of range bounds | R:937-1018 | 82 | typed adds and typed min/max state; object adds otherwise |
| M4 | Literal tuple unrolled (up to 32 instances), constants substituted | R:96, 249-264, 331-383, 435-454 | ~80 | e1 1.2x slower (measured); 52 source sites |
| M5 | Nested site lifted to its own statement, evaluation order kept (W1.8) | R:526-757 | 232 | s3 3.8x slower (measured); 89 source sites |
| M6 | Specialised double `return sum(genexpr)`: `F64Sum`, run-time identity guard on `sum`, cold fallback function, capture of the preceding assignment, enum tuple as range | R:790-804, 810-881, 918-935, 1020-1052, 1136-1280; P:10-33; `runtime/_native_streams.pyx`:36-42; `transformer.py`:626-631, 713-715 | ~310 | s1 13-17%, s2 5-7% against M1 alone; nothing against (a) (measured) |
| M7 | Specialised `return all(...)` over `range(cell())` of `abs(cell(i)) < float` or a bool cell | R:1054-1123; `_native_iterables.pyx/.pxd` | 97 | nothing measurable (a1 1-3 ms in every build) |
| M8 | Exact tuple of ints read borrowed, items passed as C integers (diag-scale P3) | R:410-432, 456-486; `runtime/_native_streams.pxd`:9-29 | 75 | thread scaling only (shared reference counts); single-thread value not measured |
| M9 | Reports: untyped and object entries, formula lines | R:16-33, 274-282, 896-912 | ~50 | build report rows; `formula_lines`/`untyped_entry` are also imported by `enums.py`, `helpers.py`, `ordered_arguments.py` |
| M10 | PEP 479 conversion, empty-input message, safe range step | R:69-108, 310-317 | ~45 | exception equality with live |

## 2. Complexity

| Cut | Lines | Expected cost | Risk | Verification (at most ten models) |
|---|---|---|---|---|
| C1. Delete M6 and M7; in `emit_general` (R:284-294) use `F64Sum`/`f64_add`/`f64_result` and no `try/finally` when every element kind is `double` and the result is read as a double (8 lines, *compiled and measured*: row (a)) | -405 (R -355, `_native_iterables.*` -27, streams `f64_accepts` -7, transformer cold-sum wiring -16) | none measured: row (a) equals or beats "as is"; values identical (first add from 0.0 is exact; empty gives 0.0 at the double boundary in both) | low: with the lanes off 16 of 17 tests of the 11 reduction test modules pass; the one failure is the counter `native_reductions == 1` (`tests/compiled/test_native_reductions.py:49`) | ten models with the most `return sum(` in double cells: values equal, `native_reductions` 0, `reductions_native` raised by the same count, generated C smaller by one `_mx_cold_sum_N` per site, time within noise |
| C2. Dead wiring: `exact_receiver_guards` is never filled (R:776; `transformer.py`:686, 706; read in `inline_entries.py`:369, 375); `refs` unused in `has_reduction_contract` (R:41); `getattr(self, 'tuple_int_loops', 0)` | -8 | none | none | unit tests |
| C3. Three copies of the Neumaier step: `f64_add` (P:20-27), `_cs_add` (P:67-74), `runtime/_native_vectors.pxd`:94-114 ("to be unified") | -30 | none | low | vector and reduction tests |
| C4. M8 (tuple-int loop) | -75 | not measured; single-thread expectation zero, thread scaling unknown | medium: needs the thread benchmark | the models that iterate enum tuples, 1 and N threads |
| C5. `all`/`any` lowering (R:242-246, 304-307, 437, 450-451, 496-498, 517-523): Cython already inlines `any/all(genexpr)` in a cfunc without a closure (the "off" build compiles them) | -35 | a2 1.28x on the micro (the loop variable becomes an object); 300 of 541 source sites but almost all are once-per-run checks | low | t2 of five models with check cells; expected under 1% |
| C6. Conditional lifts (`a and R`, `a or R`, conditional expression: R:557-587, 608-616) | -45 | 9 of 541 source sites fall back to the interpreter's form | low | the models holding those 9 sites |
| C7. M4 unrolling | -80 | e1 1.2x on the micro; 52 source sites, some hot (claims by kind) | medium | models summing over `Kind` tuples |
| C8. Complex mode of `PySum` (P:56, 63, 103-104, 116-119, 139-140, 153-154, 186-197) | -25 | complex sums take the object loop: equal values on 3.12/3.13, uncompensated on 3.14 | no model sums complex numbers | `test_exact_sum_state.py` |
| C9. Direct-iterable lowering (`sum(res)`, `all(res)`: R:195-198, 267-270): the built-in already loops in C | -8 | none expected (not measured); 55 source sites | low | values only |

Duplicates of Cython/CPython: C5 (Cython inlines `any`/`all`), C9 (CPython loops in C), M6's guard (Cython itself
never guards a built-in name; M1 does not either, so the two lanes disagree on whether a replaced `builtins.sum`
is honoured). Serving no model by reading: complex mode; `sum(iterable, start)` is already declined (R:221-222).
Translation time: `lower_general` builds a `MetadataWrapper` per `CstFacts(...)` call and repeats it for the same
iterator up to three times (R:204, 215-219); `formula_lines` builds one per formula. Not measured.

## 3. Generic

| Rule | Fitted to | Case where it does not fire or changes behaviour |
|---|---|---|
| M7 `single_range` (R:1068-1082) | the lifelib check-cell spelling: zero-argument cell as the bound, one-parameter cell under `abs`, comparator a float literal or `self.x`, `<`/`<=`/`>`/`>=`, bint return | `all(abs(r(t)) <= tol for t in range(n()))` with `tol` a local, or `range(1, n())`, or `== 0.0`: not this lane (M1 takes it, equally fast). With a derived Space the lane runs the interpreter's generator (`_mx_exact_receiver` guard, R:1109) while M1 stays native |
| M6 `enum_range` (R:848-851) | enum tuples with consecutive values, single owner | `(Kind.A, Kind.C)` declines; M4 handles both |
| M6 `captures` (R:790-804) | an assignment on the line directly before the `return sum(...)` | one blank statement between them: not captured; `formula_kinds` (R:764-769) already covers it for M1 |
| `bound_kind` `max(0, t - w())` (R:1012-1017) | Term_JP_S (comment at R:996) | generic enough: any `min`/`max` of C integers |
| M4 unrolls `cst.Tuple` only (R:339) | tuple spelling | `for k in ["DEATH", "LAPSE", "MATURITY"]` (present in lifelib source) iterates a Python list of objects: same value, slower. Speed depends on the bracket |
| `scope_safe` (R:784-788, 890-894) | formulas without lambda/try/import/walrus/del | one `try` or `sorted(key=lambda ...)` anywhere in a formula leaves every reduction in it to Cython (reported as untyped) |
| `builtin_bounds` (`transformer.py`:689-691; R:152) | models with no Reference or cell named `range`, `min`, `max`, `abs` | a model with a cell named `max` loses all lowering, including `sum` that needs none of the four |
| statement forms (R:155-163, 676-680) | `return`, `x = `, `x op= `, expression, `if` test | `elif`/`while` test, `x += a * sum(...)`, `d[k] = sum(...)`: not lowered, reported |

Behaviour change for an unmodified upstream user:
1. **Keyword named as the loop variable raises** (defect R1 below): `sum(pv(t=t) for t in range(n))` works upstream.
2. Values of `sum(genexpr)` change in the last bits against upstream's compiled result (Cython adds uncompensated);
   they now equal live on 3.12+. Intended, but a visible change between two compiled builds.
3. No configuration key turns the area off (`config.py` has none), so a user cannot fall back if a lowering is wrong.
4. Every model with one reduction builds one more provider copy (`_mx_helper_<hash>` from `_native_streams.pyx`).
5. On 3.10/3.11 `f64_accepts` is always false (`_native_streams.pyx`:42), so every M6 site runs the cold
   fallback, a Python generator, while M1 sites run native: the slower lane is chosen by Python version (read only).

## 4. Reliable

| # | Class | Minimal reproduction | Live | Compiled | Fix |
|---|---|---|---|---|---|
| R1 | **raises where live works; upstream regression** (*compiled*) | `fl = lambda t: t * 0.5`; `c = lambda t: sum(fl(t=t) for t in range(t))`; `c(1)` | `0.0` | `TypeError: fl() got an unexpected keyword argument '_mx_reduce_0_v_t'`; with a literal tuple (`for t in (1, 2)`) the translator stops with `ParserSyntaxError` on `self.fl(2=2)` | the keyword `Name` has an expression context, so `_Substitute.leave_Name` (R:118-124) renames it. Add to `_Substitute`: `def leave_Arg(self, before, after): return after if after.keyword is before.keyword else after.with_changes(keyword=before.keyword)` (3 lines; the same in `Rename`, R:1224-1228, if M6 stays). *Compiled: equal live.* Source count: 0 such sites in the 58 folders, which is why it was not seen |
| R2 | **raises where live works** on every call (*compiled*) | `vec = lambda t: [0.1 * i for i in range(t)]` (a typed vector cell); `c = lambda t: vec(t + 1)[min(i for i in range(t, t + 2))]` | `0.1` at `t = 1` | `TypeError: Cannot convert list to ..._mx_helper_....F64Vec` | the vector pass has already written `helper.vec_get(self.vec(t + 1), <site>)`; `hoist` (R:549-555) moves the callee and the typed argument into object temporaries: `_mx_hoist = (helper.vec_get)`, `_mx_hoist_ = (self.vec(t + 1))`. `_compile_time` (R:714-718) only knows `_mx_cy`. At R:542 also decline when a preceding node is the `func` of a call and its root name starts with `_mx_` (not `_mx_hoist`): 8 lines. *Compiled: equal live*, also for `vec(t + 1)[t] * sum(...)` and `divmod(sum(..), max(..))[1]`, which still lift |
| R3 | translator stops with a traceback (*compiled*) | `sum(x.real for x in (1, 2, t))` | `3 + t` | `ParserSyntaxError` at `(2.real)` | an integer literal substituted before `.attr`: drop `cst.Integer` from `_ATOMIC` (R:95), so it is parenthesised. 1 line. *Compiled: equal live* |
| R4 | **silent difference on 3.12 and 3.13** (*compiled on 3.13.18 with the test probe*) | `sum(v for v in (1e16, 1, 1, -1e16))`, or any float sum holding an int item | `0.0` (`[0.1, 1e16, 1, -1e16]`: `0.1`) | `2.0` (`1.1`) | 3.12/3.13 add an int item to the float total without compensation; `PySum` always compensates (P:152, 184), as 3.14 does. Add `_cs_add_int` (compensated only when `PY_VERSION_HEX >= 0x030E0000`, else `hi[0] += x`) and use it at those two lines: 7 lines. *Compiled on 3.13: 0 bit differences in 200,000 random mixed lists; on 3.14 `test_exact_sum_state.py` passes.* Needs cancellation to exceed 1e-10; `test_exact_sum_state.py`:74-84 scales its tolerance by `sum(abs(items))`, which hides it. Left over: on Windows 3.12/3.13 an int beyond 32 bits leaves the float loop in live and not here (read only) |
| R5 | platform: speed by Python version | any M6 site on 3.10/3.11 | native in M1 | cold fallback | goes with C1 (read only) |
| R6 | exception type, theoretical (read only, not reproduced) | `__lt__`, `__add__` or the first `__iter__` raising `StopIteration` | `StopIteration` | `RuntimeError`: comparison, add and `iter()` run inside the `try` of R:311-313 | none proposed |
| R7 | raises where live works, loudly | `sum(1 for i in range(big(t) * 4, big(t) * 4 + 3))`, `big` near 2**62 | `3` | `OverflowError: ... (integer arithmetic beyond 64 bits)` | allowed by the rules; no change |

Checked and found equal to live (*compiled*): evaluation order and first exception of lifted statements, `and`/`or`
and conditional-expression lifts, loop variable shadowing a parameter or a local, a variable read by its own first
iterable, dependent inner iterables with unrolling, `StopIteration` in a generator against a list comprehension,
empty `min`, NaN in typed `max`, int overflow into big ints, mixed int/float (3.14), bool sums, negative zero,
negative and variable range steps, `all` over a list comprehension evaluating every element, `zip` targets,
augmented assignment in a loop, `if` test lifts, strings, dict values, nested reductions.
Sample, thread timing, call frequency: nothing in this area reads them; element kinds come from the traced cell
types, which only select the add function and the boundary type (an empty or all-int sum in a double cell is
`0.0`, live `0`: the existing boundary rule, not this area). M8's borrowed read is safe: exact tuple, held by a local.

## 5. Table split
Not a table area.

## 6. Bottom line

Area today: about 1,620 non-test lines (R 1,280; P 219; `_native_iterables.*` 27; streams 45; wiring about 50).

| Version | Lines | What it is | Speed |
|---|---|---|---|
| **(a) keeps the measured speed** | **about 1,210** (-410) | C1 + C2: one lowering (M1-M5, M8-M10) with the two-double state for all-double sums; no guard, no cold fallback, no `_native_iterables` | micro: equal to today within noise, s1/s2 5-7% faster (row (a)); the 70 models not measured here |
| **(b) within 25-30%** | **about 950** (-670) | (a) + C3 + C4 + C5 + C6 + C7 + C8 + C9: `sum`/`min`/`max` over ranges and iterables with typed state, unconditional lifts kept | micro: `any` 1.28x, tuple-of-pairs 1.2x, the rest equal; per model expected well inside 25% because `all` sites are once-per-run checks, but C4 (threads) and C7 (hot `Kind` tuples) must be measured on models first |

Not recommended for (b): removing M5 (232 lines) costs 3.8x on each of 89 source sites (16%).

Ranking for use: R1 and R2 are blocking (a run raises where live and upstream work; fixes are 3 and 8 lines,
compiled). R4 is a silent difference on two supported Python versions (7 lines, compiled). R3 is one line.
C1 and C2 cost nothing measurable and take under half a day. C4-C9 belong in the proposal text with the micro
figures above; none was measured on a lifelib model.
