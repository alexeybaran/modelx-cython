# Review of area `rest` (reviewer b)

Branch `review/r4-rest-b` at b326334; upstream d22426f. Stack: CPython 3.14.6t, Cython 3.3.1, modelx 636877b
(and PyPI modelx 0.33.0 for B2), lifelib 04d69db. Built one small model at a time; no lifelib model built.
"Measured" below means measured in this review; anything else says "not measured" or "not compiled".

## 0. Blocking findings (fix now)

| # | Defect | Where | Repro (this review) | Fix |
|---|---|---|---|---|
| B1 | Default `integer_checks: 'auto'` picks unchecked C arithmetic when the sample's integers are small; the branch also types formula locals as C integers. Result: silent wrong integers where upstream is exact or raises | `config.py:82,106-118`, `cli.py:311-318`, `transformer.py:1105-1113`, `integer_boundaries.py:774` | model below, default spec | default `'checked'`; keep `'auto'`/`'unchecked'` opt-in (cost not measured) |
| B2 | Every build needs export protocol 2, even with no collect, so an export from released modelx (PyPI 0.33.0) is refused; upstream builds it | `cli.py:399-400` → `collect_adapter.py:62-69`; `pyproject.toml:20` pins `modelx>=0.33.1.dev0` | `ValueError: This export was written by a modelx without export protocol 2` | raise only when the export uses collect or `parallel_workers > 1`; the rest of `main_handler` already handles `runtime.hook` false (`cli.py:402,414,421`) |
| B3 | Round memo (`LookupContext`) gives silently wrong values when two threads evaluate the same item: its writes are not atomic and memo hits return the other thread's result | `runtime/_native_rounding.pxd:15-29`, `item_roots.py:25-37`, `rounding.py:44-178` | 2 threads, one context: 95-2,398 wrong of 2,000,000 per thread | delete the memo (C1 below): it saves 2.7 ns per hit and costs 4.1 ns per miss |

B1 model: `h(t)=t`; `il(t): x = h(t); return x*x*x*x // (x*x)`; `c(t)=h(t)**3` spelled `h(t)*h(t)*h(t)`;
`ls(t): s = 0; for i in range(4): s += c(t); return s // 4`; `cube(t): x = h(t); return x*x*x`.
The sample calls each cell with t=2,3; the spec sets only `cells_param_size`. The report `_mx_integer_checks.json`
says `mode: unchecked`.

| call | live | upstream d22426f | branch default | branch `infer_types: False` | branch `checked` |
|---|---|---|---|---|---|
| il(100000) | 10000000000 | 10000000000 | **776627963** | **776627963** | OverflowError (named) |
| ls(1400000) | 2744000000000000000 | 2744000000000000000 | **-1867686018427387904** | equal | OverflowError (named) |
| cube(3000000) | 27000000000000000000 | OverflowError | **8553255926290448384** | **8553255926290448384** | OverflowError (named) |

`il` and `cube` are wrong even with `infer_types: False`: the branch types `x = h(t)` as a C integer
(`powers.formula_local_kinds`). Upstream keeps it a Python int. The existing test
`tests/compiled/test_integer_overflow_model.py` (UNCHECKED_PLAIN) shows the same wrong values. It does not
fail only because its `BIG = 2**62` Reference makes `auto` choose checked. In `checked` mode `il` and `ls` raise
where upstream was exact. The rules allow that, but it is a behaviour change to state in the proposal.

B2: I measured a build with the check turned into a no-op (monkeypatched, not committed). PyPI modelx 0.33.0
then exported, translated, compiled and ran `f(3) == 6`. So the narrow fix is enough for this model. I have not
compiled the proposed condition itself.

## 1. Mechanisms

| Mechanism | Lines | What it does | Lost without it |
|---|---|---|---|
| Module-math binding | `helpers.py:46-84,128-146`, `helper_members.py:1-98`, `runtime/_native_math.pyx:44-218` | `self.<math>.exp/log/erf/sqrt/isinf/floor/ceil(x)` and `math.pi`… call C `m_*`. The Reference is passed as a borrowed pointer, and any other namespace takes the Python path | about one Python attribute read and call per math call; per-call cost not measured |
| `math.isclose` | `helpers.py:86-126`, `_native_math.pyx:279-305` | C `isclose` when the tolerances are literals, typed References or parameters | the same; 61 `math.isclose` sites in lifelib |
| Formula-local `import math` | `helpers.py:148-162,209-243,329-373` | lowers `import math` inside a formula and drops the import if nothing else reads it | native math in BasicTerm_S/SC/SE, CashValue_SE, Child_KR_S (they use this spelling) |
| Cell used as a value | `helpers.py:290-304` | `cast(object, self).cell` so a Cell passed as a value stays a bound Python callable | correctness: Cython would coerce a C method pointer |
| Aliased math callable | `_native_math.pyx:1-41`, `.pxd:1-9`, `helpers.py:172-189` | a module member that is `math.exp` under another name | nothing in the 70 models (no such alias in lifelib) |
| Guarded `round` | `rounding.py:15-41`, `_rounding.pxd:1-61`, `transformer.py:672-676,1043-1044` | `round(x)` / `round(x, n)` take a C path when `round` is the builtin at run time | C round instead of a builtin call; 416 `round(` sites in lifelib |
| Round memo | `rounding.py:44-178`, `runtime/_native_rounding.*` (60), `transformer.py:213,325-326,653-679,1087-1088,1231-1247`, `item_roots.py:25-37` | caches the last `round(x, n)` input per item | nothing measurable (benchmark below); also B3 |
| Typed min/max | `ordered_arguments.py:101-132,186-213`, `_native_math.pxd:47-94` | `min`/`max` of doubles or C ints become C comparisons; other operand mixes are compared as objects with no tuple | part of the measured 1.00x-1.02x |
| `int()` identity | `ordered_arguments.py:73-94` | `int(x)` of a proven C int becomes `x`, which keeps derived locals typed | part of 1.00x-1.02x |
| Object power boxing | `ordered_arguments.py:41-71` | unproved `**` evaluated with Python objects | correctness (no complex promotion) |
| Expression kinds | `expression_kinds.py:1-264` | kinds of locals (fixpoint) and builtins, used to decide typed min/max | only those lowerings |
| Divisibility | `checked_arithmetic.py:41-68,137-187`, `integer_boundaries.py:462-500,567-600`, `_mx_sys.pxd:283-298` | `a % b == 0` with both operands signed C-integer names becomes `_mx_divisible` | part of 1.00x-1.02x; never fires on a literal divisor (`:141`) |
| Object providers | `provider_manifest.py` (166), `helpers.py:382-423`, `usage.py:628-667`, `builder.py:566-621`, `transformer.py:212-216,482,1520,1615-1618`, `table_cells.py:39,128`, `tracer.py:236-237` | third-party types registered with `__modelx_cython_type__` and a `.provider.json` manifest | nothing in the 70 models (no registration in lifelib or modelx) |
| Provider build cache | `provider_build_cache.py` (144), `build.py:183,210` | when `MODELX_CYTHON_PROVIDER_CACHE` is set, reuses compiled providers keyed by inputs and toolchain | compile time: `_tables` 21.7 s, `_native_vectors` 4.8 s, `_native_math` 3.8 s per build (measured) |
| `_native_streams` | 46 | `f64_accepts` guard (3.12+) and the float-sum state cimports | belongs to reductions; `provider_version()` `.pyx:16-17` is never called |
| Policy keys | `config.py:35-126,272-344,410-427` | `CompilerPolicy`: 15 keys, two of them read from environment variables; scoped `cells_param_size` with `(lower, stop)` | configuration |
| Reports | `admission.py` (540), `cli.py:464-523` | `_mx_admission.json` (about 50 fields per cell), integer-check, read-audit, table-signature and notes files | diagnostics only; about 50 test files read them |
| Spec reader | `syntax.py:54-104` | replaces `ast.literal_eval`: agrees with it on 22 forms tested, errors included | nothing (it is a copy of the standard library) |

Differential check of the providers, measured: 400,000 random and edge inputs compiled against the
CPython builtins. `native_round_digits`, `_mx_round_one`, `_mx_round_digits`, `m_exp/log/erf/sqrt` (bitwise),
`m_log` of `long long`, `m_isclose` and the four mixed min/max variants (value and class) were all identical.
`m_floor_ll`/`m_ceil_ll` raise OverflowError beyond ±2**63, where live returns an int. That is documented and allowed.

## 2. Complexity: cuts

| Cut | Non-test lines | Expected cost | Risk | Verify (≤10 models) |
|---|---|---|---|---|
| C1 Round memo (`RoundedInputs`, `LookupContext`, contexts wiring) | ~235 (+ tests `test_native_rounding_provider`, `test_private_lookup_roots`, parts of 3 others) | none: hit 2.72 ns vs 5.49 ns without memo, miss 10.17 vs 6.07 ns (measured micro-benchmark) | low; it also fixes B3 | the BasicTerm_S/SC, Savings, CashValue_SE models: `round` sites unchanged in `_mx_admission.json`, t2 |
| C2 Object providers | ~300 (+ 355 test lines: `test_provider_manifest`, `test_native_object_provider`) | none for the 70 models | none for them; a separate opt-in feature for others | grep: no `__modelx_cython_type__` in lifelib/modelx; generated C of 10 models identical |
| C3 Divisibility | ~120 (checked_arithmetic + integer_boundaries branches + `_mx_sys.pxd`) | ≤2% total together with C5 (brief); Cython with `cdivision=False` already gives Python `%` | medium: `integer_boundaries` shares `CheckedArithmetic`; not compiled | t2 on the models whose admission lists `_mx_divisible` |
| C4 Aliased math callable | ~70 | none (no alias in lifelib) | low | `native_helpers` in admission never lists non-`m_` `_native_math.*` |
| C5 Typed min/max, `int()` identity, `expression_kinds` | ~375 (`ordered_arguments.py:73-213` ~140, `expression_kinds.py` 264, minus the kept power boxing) | ≤2% (brief: 1.00x-1.02x for these and C3) | low | t2 on 10 models |
| C6 Dead code | ~8: `_native_streams.pyx:16-17`; `checked_arithmetic.py:111-113` (`parameter_intervals`, `local_intervals` unused) | none | none | unit tests |
| R1 (redesign, write up only) | ~100 | none | medium | two local-kind inferers: `expression_kinds.local_kinds` (fixpoint) and `powers._assigned_local_kinds` (`powers.py:564-680`); merge into one |
| R2 (write up only) | — | — | — | `native_math` (`config.py:78`) controls only `round` (`transformer.py:666`), not math; rename it or document it |

Keep `provider_build_cache`. It is not duplicated: the cache refuses to run under ccache (`:67`), and it saves
about 33 s of provider compilation per model build (sum of the measured times above). That matters for the
compile-time guard.

## 3. Generic: fitted rules and upstream changes

| Rule | Where | Case that does not fire |
|---|---|---|
| Round memo needs the builtin `min` in scope (unrelated to `round`) and at least 4 statements | `rounding.py:67,70` | `def f(t): return round(g(t) * 1.1, 2)` |
| Memo only for a `round(x, n)` that is the whole return value or one branch of it | `rounding.py:129-133` | `return round(x, 2) * 100` |
| Divisibility only for name % name | `checked_arithmetic.py:141` | `t % 12 == 0` (the common spelling) |
| A native call taken as signed only with literal arguments within ±2**25 | `checked_arithmetic.py:122` | `self.f(t, 40000000)` |
| Mixed int/float min/max only for two operands | `ordered_arguments.py:122` | `min(a_float, b_int, c_float)` stays an object comparison |
| `isclose` tolerances must be literals, typed References or parameters | `helpers.py:109-117` | `rel_tol=self.tol()` |
| floor/ceil only on the `math` module itself | `helper_members.py:30-31` | a module Reference that re-exports `floor` |

Changes for an existing upstream user: B1 and B2. `infer_types=True` by default (`config.py:79`) together
with typed locals: checked mode raises where upstream was exact (table in B1). Six extra report files in every
package. `strict_tables` and `strict_lowering` follow the environment variables `MX_CY_STRICT_*` (`config.py:81,85`),
so the same spec can give a different build. When a replaced Reference is the namespace, `isclose` gets float
tolerances where an int was written (`helpers.py:118`): a rare edge case.

## 4. Reliable

| Case | Repro | Expected / actual | Fix |
|---|---|---|---|
| B1 integer mode depends on the sample | §0 | exact or raise / silent wrap | §0 |
| B3 round memo under threads | `nr.LookupContext()` with private lookup enabled; 2 threads × 2e6 `repeated_round_float(ctx, round, v_i, 2)` with v=1.234567 and 9.876543 | 0 wrong / 95-2,398 wrong per thread | C1. The same "items are partitioned" assumption also backs `numeric_cache.exclusively_owned` (another area): check that its cache writes are idempotent under a race, not torn |
| `m_floor_ll` beyond 2**63 | `math.floor(1e300)` in a `long long` cell | int / OverflowError | allowed (raises) |
| Python 3.10: `math.erf` is CPython's own series before 3.11; `m_erf` uses libm | not run (no 3.10 here) | differences of a few ulp, below rel 1e-10 | none needed; mention |
| `_mx_canonical_round` captured at import | patch `builtins.round` after import | the patched `round` / the patched `round` (the function object is compared on each call) | none |

## 5. Table split

Not a table area. The only shared table wiring in `transformer.py` (`_table_reads_with_row_fields:850-893`,
`_lower_frames:895-923`) restores state by clearing `vars(cells)` after a failed attempt. That is fragile but
it belongs to the table reviewers.

## 6. Bottom line

(a) Keep the measured speed (1.00x-1.02x): make the B1-B3 fixes and cuts C1, C2, C4 and C6. That removes about
**610 non-test lines** (plus about 600 test lines). Keep math binding, guarded round, typed min/max, `int()`
identity, divisibility, the build cache and the reports.

(b) Speed within 25-30%: the whole measured value of this area is at most 2%, so C3 and C5 can go as well:
about **1,100 non-test lines** removed. The remaining area is math binding, guarded `round`, object power
boxing, the Cell-as-value cast, policy keys, the build cache and the reports. The per-call value of the math
binding was not measured; measure it before cutting it.
