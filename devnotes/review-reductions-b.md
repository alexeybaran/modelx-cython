# Review: reductions (reviewer b)

Branch `review/r4-reductions-b` at b326334. Stack: CPython 3.14.6t (free-threaded), Cython 3.3.1, modelx 636877b;
3.13.18 (ordinary build, same Cython and modelx) for the version findings. No lifelib model was built. Every number
below comes from a small model of 2-8 Cells built with `python -m modelx_cython exported --sample sample.py --no-spec`,
live and compiled run in separate processes. Variants were made by monkeypatching `reductions` before the CLI runs
(no source change), so each pair differs only in the mechanism named. Timings: 1,000 cache-missing calls, each
reducing 1,200 cached Cell values, best of 3-5 interleaved runs. Tests run: the 10 reduction test files (26 passed on 3.14t).

## 1. Mechanisms
| # | Mechanism | Lines | Measured (per site, this machine) | Lost without it |
|---|---|---|---|---|
| M1 | General lowering W1.4: `sum/min/max/all/any` over a generator, list comprehension or direct iterable as a C loop; typed range variables; `return`/`x =`/`x op=` | `reductions.py:137-524` (388), `kind`/`bound_kind` `:937-1018` (82) | sum 60 -> 4.3 ms (14x; 27x with M4 or the merge in C1), min 66 -> 4.0 ms (16x), nested `for .. for j in (1,2,3)` 190 -> 9-20 ms, all 5.4 -> 1.3 ms (4x over Cython's own inlined `all`) | Builtin call over a Python generator (sum, min, max); Cython's inlined `all`/`any` with a boxed loop variable |
| M2 | Exact built-in sum state `PySum` (int, float, complex, generic loops with CPython 3.14's transitions) | `_float_sum_state.pxd:33-219` (187) | needed by M1's `sum` | an exact `sum` needs the builtin |
| M3 | Literal-tuple unrolling (<= 32 instances), constants and enum members substituted, temporaries in order | `:331-383`, `:435-453` (72) | not isolated | 44 lean sites `sum(c(t, k) for k in (Kind.A, ...))` iterate a tuple object |
| M4 | Specialised lane: `_specialized` F64 sum with guard `f64_accepts(sum)` and a cold `cfunc` fallback, `captures`, `double_expression`, `enum_range`, `callable_name`; `single_range` `all` with `_native_iterables` | `:36`, `:783-804`, `:810-881`, `:918-935`, `:1020-1123`, `:1136-1280` (355); `_native_iterables.*` (27); `_native_streams.pyx:3-13`; `transformer.py:626-632,705,713-715,1009,1076` | sum 2.0-2.3 ms vs M1 4.3 ms; `all` 1.3 vs 1.2-1.7 (equal); `all(abs(..) < 1e9 ..)` 2.7-2.9 vs M1 1.8-2.1 (M1 faster) | nothing once C1 is applied (measured: 2.0-3.5 vs 2.0-3.9 ms) |
| M5 | Nested sites W1.8: lift an unconditionally evaluated reduction out of an expression, earlier operands to temporaries in order; `and`/`or`/`if-else` as statements | `:527-692`, helpers `:694-757` (230) | `2.0 * sum(..) + 1.0`: 6.7 vs 62-77 ms (10x) | ~80 of ~550 lean sites (36 `all`, ~33 `sum`) stay builtin calls |
| M6 | P3 exact-int tuple loop: items of an exact tuple read borrowed as `long long` when the variable only feeds `long long` Cell parameters | `:410-432`, `:456-486` (55); `_native_streams.pxd:9-29` (21) | 8.3-8.8 vs 16-18 ms single thread (2x); free-threaded refcount contention not measured | ~10 lean sites over id tuples |
| M7 | Reports: untyped and object entries, `formula_lines` (also used by enums, helpers, ordered_arguments) | `:16-33`, `:274-282`, `:896-912` (~45) | diagnostics only | build report detail |

Lean text of the 58 `*_S` directories (scan of `__init__.py`, not translated): ~550 one-iterable sites: `all` 280 (236 are
`return all(abs(check_x(t)) <= tol * s for t in range(proj_len()))`), `sum` ~230, `min`/`max` ~20; 1,291 n-argument `min`/`max` (not this area).

## 2. Complexity
| Cut | Lines | Cost | Risk | Verify (<= 10 models) |
|---|---|---|---|---|
| C1 Delete M4 and give M1 the F64 state when every item is a double and the result is a double (`exact_float` and `kinds == {'double'}`: `F64Sum`/`f64_add`/`f64_result`, no `finally`) | -412 +8 (reductions 355, `_native_iterables` 27, streams 15, transformer 14, `inline_entries.py:369,375` dead `exact_reduction_guards` 2) | none measured: merged 2.0-3.5 ms vs base 2.0-3.9 ms (sum); 8.8-9.2 vs 11.8-19.8 (nested tuple: faster, base used `PySum` there); `all` equal or faster | items all double: `PySum` and `F64Sum` give the same bits (first add 0 + x vs 0.0 + x is equal for every double incl. -0.0, inf, nan) | built and run as a monkeypatch only, values equal live; on code: the 6 lean models with most `return sum(...)` (Basis_DE, BU_DE, Index_DE, Medical_JP, Riester_DE, Sofort_DE) t0 and values |
| C2 Drop the streams provider extension: models cimport the header-only `_float_sum_state` directly; move `tuple_int64`/`exact_tuple` (21 lines) there | -17 pyx and its provider wiring | none expected (inline code) | low | build time: the provider extension alone takes 2.7 s to build here (less if `provider_build_cache` hits); not measured on a model |
| C3 Unify `runtime/_native_vectors.pxd:91-120` (private F64 copy) with `F64Sum` (add `nogil`) | -25 | none | low | vectors area: vec_sum tests |
| C4 Small dead code: `has_reduction_contract` unused `refs`; `emit_body(loop=)`; `exact_receiver_guards` (always empty) | -5 | none | none | unit tests |
| C5 (redesign, not built) Replace M5 by lowering every site into its own `cfunc` taking the free names (as `cold_fallback` already does): evaluation order is then the call's own, so `first_site`, `effect_free`, `_evaluation_order`, `_replaced`, the and/or and if-else cases and the "conditional site" report go | ~ -170 | unmeasured: one C call per reduction | free locals need kinds (have: `formula_kinds`) | t0 on the 6 models above |

Duplicates: Cython inlines `all`/`any` of a generator (`Cython/Compiler/Optimize.py:1913-1980`) but with a boxed range
variable (M1 is 4x faster); Cython's `sum` inlining is disabled (`__handle_simple_function_sum`, :2036) and `min`/`max`
of one iterable is left to Python (:2101), so M1/M2 duplicate nothing there. M2 mirrors CPython's `builtin_sum_impl` by
design. Upstream d22426f has no reduction lowering.

## 3. Generic
| Case | Source | Effect |
|---|---|---|
| G1 A direct literal tuple: `lambda: max((n, 2))`, also `sum((a, b, c))` | direct-iterable path sets `facts = None` (`:196`) but `classify` (`:339`) still unrolls the tuple and `substituted` (`:230`) then leaves `_mx_reduce_0_item` unbound | build stops: "undeclared name not builtin: _mx_reduce_0_item", "Cython could not compile cell 'Projection.tmax'". Upstream builds it (Cython inlines `max` of a tuple). Regression for an upstream user. Not in the lean text |
| G2 `all([..])`/`any([..])` over a list comprehension | `action` `:523` tests the truth of every item after the result is decided | see R2: raises where live works |
| G3 Coverage depends on spelling (M5): `x += 1 + sum(..)`, `elif`/`while` tests, one-line suites, a reduction after `and`/`or` or in a later chained comparison are not lowered; `x = x + 1 + sum(..)` is | `:669-680`, `:904-908` | that site 10x slower; values equal |
| G4 M4 fits spellings: `single_range` only `all(abs(c(t)) <op> <float or attribute> for t in range(n()))` with a one-parameter `long long` Cell and a `bint` result; `enum_range` only contiguous member values; `captures` only the line just before | `:1054-1123`, `:810-852`, `:792-804` | performance only, and M1 covers all of it (C1) |
| G5 M6 needs every read of the variable to be a positional `long long` Cell argument: `if k != 0` or `w[k]` switches it off | `:456-486` | 2x on that site |

## 4. Reliable
| # | Reproduction | Expected | Actual | Fix |
|---|---|---|---|---|
| R1 **blocking, silent** (3.12, 3.13) | Cells `ov(i)`: `ovals[i]` with `ovals = (1e16, 1, -1e16, None)` (object-typed); `lambda: sum(ov(i) for i in range(3))`; `cv(i)` over `(1e16, 1j, 1.0, -1e16)` summed over `range(4)` | live 3.13: `0.0` and `1j` | compiled 3.13: `1.0` and `(1+1j)` | `PySum` copies 3.14 for every version. CPython 3.12/3.13 add an int item in the float loop as `f_result += (double)value` after `PyLong_AsLongAndOverflow` (C `long`: 32-bit on Windows, else generic loop), uncompensated, and have no complex loop (3.14 source: `PyLong_AsDouble` + `cs_add`, complex loop). Fix in `_float_sum_state.pxd` (+14 -4, below): compiled 3.13 then gives `0.0` and `1j`; `test_exact_sum_state` passes on 3.13 and 3.14t |
| R2 **raise where live works** | `arr(i)`: `(0, numpy.array([1, 2]))[i]`; `lambda: all([arr(i) for i in range(2)])` (and `any` with `(1, array)`) | `False` (`True`) | `ValueError: The truth value of an array ... is ambiguous` | `reductions.py:523`: `if {p}_result and not (...)` for `all`, `if not {p}_result and (...)` for `any` (2 lines). Built: compiled `False`, generated `if _mx_reduce_0_result and not (self.arr(..))` |
| R3 **build stops, upstream regression** | G1 | builds, `1200` | Cython error | `reductions.py:339`: add `self.general_state['facts'] is not None and` (1 line; a direct tuple then iterates `iter((..))`). Built: `1200` |
| R4 test blind spot | `test_exact_sum_state.py:73-74,108` scales the tolerance by `sum(abs(items))`, so `[1e16, 1, 1, -1e16]` passes with 2.0 against 0.0 | rel 1e-10 / abs 1e-7 of the result | R1 not caught | compare against the result (cancellation cases exact); run `test_native_reduction_cases` on 3.12 or 3.13 in CI |
| R5 cross-area (tracer) | `ov` above without the `None`: traced `float, int, float` and typed `double` | live 3.13 `0.0` | `1.0`: the int item reaches the sum as 1.0 | an int traced in a float-typed Cell changes `sum` on 3.12/3.13 only; harmless on 3.14. Owner: return typing |
| R6 cross-area (config, `integer_checks: auto`) | `lambda: sum(float(t * t * t) for t in range(cn))`, sample `cn = 10`, run `cn = 2_200_000` | `5.856e24` | `3.959e24`, silently; same with M1 off (`infer_types=True, overflowcheck=False` types `t`) | mode chosen by the traced sample; not this area |

Checked and found equal to live: `PyLong_AsDouble` overflow (`sum([1.5, 2**1100])` raises the same `OverflowError`:
Cython checks `except? -1`); NaN, -0.0 and first-extreme rules of typed `min`/`max`; empty `min` message (taken from the
building interpreter, which is the running one); `StopIteration` in a generator body; P3 borrowed reads (the tuple is held
by a local; no shared state, no dependence on thread timing or call frequency). 3.10/3.11 on Windows (CPython's `long`
int accumulator) give the same values as the compiled `long long` one: there is no compensation before 3.12.

R1 fix (built and run on 3.13; not built on Windows or 3.12):
```
_sum_generic:    elif PY_VERSION_HEX >= 0x030E0000 and source <= SUM_FLOAT and PyComplex_CheckExact(result):
sum_add_ll:      elif state.mode == SUM_FLOAT and PY_VERSION_HEX < 0x030E0000:
                     if LONG_MIN <= v <= LONG_MAX: state.hi += <double>v
                     else: _sum_generic(state, v)
sum_add_object:  if PyLong_Check(item) and PY_VERSION_HEX < 0x030E0000:      # float loop
                     b = PyLong_AsLongAndOverflow(item, &overflow)
                     if overflow == 0: state.hi += <double>b; return 0
                 elif PyLong_Check(item): (3.14 branch as now)
cimports:        PyLong_AsLongAndOverflow; LONG_MAX, LONG_MIN
```

## 5. Table split
Not a table area. For the reductions commit: C1 and R1-R3 first, so the commit adds one lowering, not two.

## 6. Bottom line
| Version | Keeps | Non-test lines (now ~1,610: reductions 1,280, sum state 219, `_native_iterables` 27, streams provider 46, transformer wiring ~40) |
|---|---|---|
| (a) measured speed | M1 + M2 + M3 + M5 + M6 + M7 with C1-C4 and R1-R3 | ~1,190 (-420; C3 saves 25 more in vectors). Every measured site equal or faster |
| (b) within 25-30% | (a) without M6 and M7, M5 replaced by C5 | ~920. M6 is 2x on ~10 sites and M7 is reports (keep `untyped_entry`/`formula_lines`, used by three other modules), so the model-level loss should be small, but that is an estimate: no lifelib model was built. C5 is unbuilt |

Order of work: R1, R2, R3 now (blocking, under half a day, built); C1 now (deletion with no measurable cost; built as a
patch of the generated text only, so the real edit still needs its t0 run); C2-C4 now; C5 into the proposal text.
