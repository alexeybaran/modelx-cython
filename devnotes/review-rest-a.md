# Review of area `rest`, reviewer a

Tree: `review/r4-rest-a` at b326334. Stack used: free-threaded CPython 3.14.6, Cython 3.3.1, modelx 636877b,
x86-64 Linux, GCC. Upstream d22426f built in a second environment for comparison. No lifelib model was built;
lifelib 04d69db was only searched as text. All experiments are small scratch models (one at a time).
"Measured" below means measured here on those models; everything else is marked "not compiled" or "not measured".

Blocking defects (details in section 4): **R1** silent wrong integers in the default build (regression against
upstream), **R2** a compiled model without tables no longer imports without `modelx_cython` installed
(regression), **R3** a valid formula no longer builds (regression, build stops).

## 1. What the code does

Own modules of the area: 2,128 non-test lines (`helpers` 423, `provider_manifest` 166, `provider_build_cache` 144,
`helper_members` 98, `_native_math.*` 312, `rounding` 178, `_rounding.pxd` 61, `_native_rounding.*` 60,
`expression_kinds` 264, `ordered_arguments` 235, `checked_arithmetic` 187). Shared: `config.py` 427 (+254 on
upstream), `admission.py` 540 (new), `build.py` 222 (new), `transformer.py` +992, `builder.py` +305, `cli.py` +160.

| # | Mechanism | Lines | Lost without it (measured, ns per loop iteration) |
|---|---|---|---|
| M1 | `self.math.f(x)` and formula-level `import math` bound to libm behind a pointer check of the module | `helpers.py:36-379`, `helper_members.py`, `_native_math.pyx:43-218`, `.pxd:11-45` | exp+log+sqrt 145 -> 15 (9.6x); floor 50 -> 4.5 (11x) |
| M2 | Older math entry taking the callable (`erf(function, x)`) for a module member under another name | `helpers.py:168-191`, `_native_math.pyx:6-41`, `.pxd:1-9` | nothing on `math` itself |
| M3 | Guarded `round(x)` / `round(x, n)`; falls back to the built-in | `rounding.py:14-41`, `_rounding.pxd`, `transformer.py:672-676,1042-1044` | `round(x, 2)` 166 -> 33 (5x); 24M fuzz cases equal to `round` |
| M4 | One-entry memo of the last `round(x, n)` on worker-private receivers | `rounding.py:44-177`, `_native_rounding.*`, `transformer.py:212-218,325-326,677-679,1231-1245`, `item_roots.py:28-34` | not measured here; active only inside collect |
| M5 | Typed `min`/`max` in object-result scopes, operand kinds from a second classifier | `ordered_arguments.py:96-132,186-213`, `expression_kinds.py`, `_native_math.pxd:47-94` | see section 2, C2: no gain over Cython for one kind, a loss for mixed kinds |
| M6 | `int(x)` of a proven C integer is `x` | `ordered_arguments.py:73-94` | 20.9 -> 1.9 (keeps the locals derived from it in C) |
| M7 | Unproved `**` and untyped `min`/`max` kept on objects (result class as live) | `ordered_arguments.py:41-71,183-184` | class of the result (correctness, not speed) |
| M8 | `x % k == 0` on signed C integers | `checked_arithmetic.py:41-70,137-187`, `_mx_sys.pxd:282-298` | nothing measurable: a loop Cython compiles alone runs 3.3-3.5 |
| M9 | Object providers (`__modelx_cython_type__` + `<name>.provider.json`, typed views, import guard) | `provider_manifest.py`, `helpers.py:393-422`, `builder.py:590-621`, `usage.py:628-665`, `tracer.py:236-237`, `transformer.py:214-216,482,612-615` | nothing: no manifest ships, no lifelib file uses it |
| M10 | Provider install into the model package; opt-in build cache; `-ffp-contract=off` | `helpers.py:382-392`, `provider_build_cache.py`, `build.py:104-128` | cache: 30.8 s -> 26.6 s on a rebuild of the same model |
| M11 | `CompilerPolicy`: 15 top-level keys, 2 environment variables, 5 per-cell keys | `config.py:36-118` | - |
| M12 | Reports: six files per build | `admission.py`, `cli.py:463-525` | diagnostics only |
| M13 | Cython diagnostics mapped to cell names, recursion budget, array-type patch | `build.py:20-102,159-200` | diagnostics; pickling of large classes |

## 2. Complexity

| # | Cut | Non-test lines | Expected cost | Risk | Verify (at most ten models) |
|---|---|---|---|---|---|
| C1 | Remove object providers (M9) | about 285 (+355 test, 62 doc) | none: lifelib has 0 uses of `__modelx_cython_type__`; only a test fixture defines a manifest | low; propose separately if wanted | generated C of ten models byte-identical |
| C2a | Same-kind `min`/`max`: emit the plain call, drop `min_f64/max_f64/min_i64/max_i64` | 12 (`_native_math.pxd:50-60`) and the math-provider dependency of min/max | none: measured 28.8 with the plain calls against 30.6-32.6 with the helpers (same loop) | low: Cython lowers same-type operands to the same comparison | the three typed-extrema tests; one model with object-result cells |
| C2b | Remove typed min/max and `expression_kinds.py` entirely (everything to the object form) | about 360 | measured 31 -> 81 on a loop of three min/max in an object-result cell; the brief's ablation says <= 2% on models | medium | models with list-returning cells: run time within noise |
| C3 | Remove the round memo (M4); M3 still handles those calls | about 230 (+ tests) | **not measured**; the memo only hits when the same `(x, n)` repeats in consecutive calls of one cell inside collect | medium until measured | the five models whose text has `round(`: collect run time |
| C4 | Remove M2; bind only when the Reference is the `math` module | about 70 | none on the 70 models (they call `math.<name>`); for another module the `m_` path is already a Python call every time (`_native_math.pyx:76-124`, no fast path) | low | generated C unchanged |
| C5 | Remove M8 and its spec key `native_divisibility` | about 90 | not measured on models; brief: <= 2% together with M5, M6 | low; **not compiled** (`integer_boundaries.py:6-8` imports four names of the module, keep those) | ten models, values and time |
| C6 | Remove the build cache | 144 (+37 test) | 4-5 s of 31 s per rebuild; the key holds the extension name (`provider_build_cache.py:113`), so nothing is shared between differently named models | low | - |
| C7 | Dead report keys and attributes | about 15 | none | none | - |
| C8 | One report file instead of six; no recorder and no table reports for a model without frames | not counted | less translation time (brief: recorder is 1.5x) | low | - |

Detail for C7: `nogil_qualified` is the constant False (`admission.py:87`); `table_reads_typed` and
`table_reads_keys` are only ever set to `0` / `[]` (`transformer.py:926,969`); `record_reads_untyped`, `math_object`
are never set (`admission.py:92,318,343`); `parameter_intervals`, `local_intervals`, `signed_atom` are never read
(`checked_arithmetic.py:111-114`, `integer_boundaries.py:312`); `getattr(self.module.spec.policy, 'x', False)` on a
dataclass that always has the field (`transformer.py:329,758,905,929`).

Duplicates: M5 same-kind duplicates Cython's own min/max lowering (C2a). `expression_kinds.local_kinds` is a
second fixpoint beside `powers.formula_local_kinds`, which `transformer.py:695,729` already runs twice per formula;
one classifier should serve powers, reductions and extrema (redesign, not compiled). `argument_kind`
(`helper_members.py:57-98`) is a third, smaller classifier with its own scope bug (R3).
An idea not compiled: let Cython classify min/max operands with a fused inline helper, which would remove
`expression_kinds.py` without the cost of C2b.

Mixed operands: `min(int, float)` goes to `min_i64_f64`, which returns an object. In a float expression that
makes the whole expression object arithmetic: measured 28.8 with only that helper kept against 5.8 with Cython's
promotion (upstream behaviour, 15.8 for the whole upstream loop). It is correct (the selected operand keeps its
class) but about 5x slower than upstream on such a line. Proposal text: keep the object form only where the result
escapes (returned, stored, passed on); not built.

Configuration (M11): the keys an upstream user needs are `integer_checks`, `prepared_tables`, `strict_tables`,
`parallel_workers`, `automatic_optimizations`. `dense_bounds_proofs`, `dense_miss_binding`, `dense_one_axis`,
`propagate_exact_receiver`, `native_cache_hits` (five values), `typed_table_reads`, `native_math`, `infer_types`,
`strict_lowering` are ablation switches; each is a configuration nobody runs in production. Keep them out of the
documented surface or drop them with the measurements they served. `typed_table_reads`, `strict_tables` and
`prepared_tables` appear in no test by name.

## 3. Generic

| Rule | Fires only when | Case that does not fire |
|---|---|---|
| M4 `rounding.py:67-70,125-134` | the formula has at least 4 statements, `min` is not shadowed, the last statement is `return round(x, n)` or `return a if c else round(x, n)` | a three-line formula; `v = round(x, n); return v`. The accepted shape is lifelib's `return r if dp < 0 else round(r, dp)` (9 sites) |
| M1 `helper_members.py:22-39` | the Reference is the `math` module, or `import math` is a top-level statement of the formula | `refs={'exp': math.exp}`, `from math import exp`, `np.exp(x)` on a scalar stay Python calls |
| M5 `ordered_arguments.py:157-158` | the cell returns an object, or the call is in a comprehension or lambda | a double cell: left to Cython (as upstream) |
| M8 `checked_arithmetic.py:42-60,120-127` | `x % k == 0` or `!= 0`, operands are typed parameters or locals assigned from `self.cell(<parameters or literals>)` | `0 == x % k`, `not x % k`, a loop variable as dividend |

Changes for an unmodified upstream user (same model, same sample, no spec):
- R1, R2, R3 below.
- `min`/`max` with mixed kinds in an object-result cell is slower than upstream (above), values equal.
- The generated `setup.py` imports `modelx_cython.build` and `provider_build_cache`; upstream's needs only Cython
  and setuptools, so `--translate-only` output can no longer be compiled on a machine without the compiler.
- Six report files are written into the model package, and the read recorder runs for every model
  (`cli.py:308-309`), also when it has no frame.
- `-ffp-contract=off` and `# cython: infer_types=True` are new defaults (`build.py:104-128`, `config.py:48`).
- The spec reader changed from `ast.literal_eval` to `syntax.read_literal`; 16 literal forms tried, same results.

## 4. Reliable

### R1 (blocking: silent, regression, and depends on the traced sample)
`infer_types=True` (default, `config.py:48`, `transformer.py:1106-1109`) types a local assigned from an integer
literal as a C integer; `integer_checks: auto` (`config.py:56,113-118`) compiles without overflow checks when
no traced argument or result reaches 2**31. Intermediate values are not traced.
```
def powloop(n):                def fact_big(n):
    x = 1                          r = 1
    for i in range(n):             for i in range(1, n + 1):
        x = x * 1000                   r *= i
    return x % 7                   return r > 10 ** 20
```
Sample and run: `powloop(0..11)`, `fact_big(0..29)` (every run argument is in the sample).

| Build | `powloop(7)` | `fact_big(22)` | differences in 42 cases |
|---|---|---|---|
| live modelx | 6 | True | - |
| upstream d22426f | 6 | True | 0 |
| this branch, no spec | **3** | **False** | 12, all silent |
| this branch, `infer_types: False` | 6 | True | 0 |
| this branch, `integer_checks: checked` | OverflowError | OverflowError | 0 silent |

Adding one sampled call whose result is 10**20 to the same model turns the mode to `checked` and the same calls
raise: the outcome depends on the sample. Fix, either of two, both verified only through the spec keys above:
(a) `transformer.py:1106`: add `infer_types=True` only when `checked` is true, so the unchecked mode keeps
Cython's default inference, as upstream (one line; run-time cost not measured); (b) make `checked` the default
(`config.py:56`; cost not measured; it raises where live works, which the rules allow). Not fixed by either:
a product of typed parameters (`n*n*n*n*n*n*n*n` with n = 1000) wraps silently here and on upstream alike.

### R2 (blocking: regression, raises at import)
`transformer.py:1064` adds `from modelx_cython.runtime import table_cells as _mx_table_cells` to every generated
module (`table_cells.py:29`). For a model without tables the name is not used again, but the compiled model then
needs `modelx_cython` at run time. Reproduction: build the two cells above, block `modelx_cython` on the import
path, `from exported_cy import mx_model`. Upstream build: imports and runs. This branch: `ModuleNotFoundError:
modelx_cython`. Fix: emit the import only when the module's final text uses `_mx_table_cells.` (decide in
`table_cells.finish`, the last pass). Verified by deleting the line from the generated module of a model with
math, rounding and min/max, `--compile-only`, and running with `modelx_cython` blocked: values correct.
The generator change itself is not compiled. Models with table cells keep the dependency; say so in the proposal.

### R3 (regression: build stops on a valid model)
`helper_members.py:67-70` takes the C type of a name from the formula's parameter of that name without looking
at the scope; `helpers.py:75-78` then picks `m_floor_ll`/`m_ceil_ll`.
```
def lam(x):                          # x traced as float
    f = lambda x: math.ceil(x)
    return [f(v) for v in items] + [x]
```
Expected: builds (upstream does) and equals live. Actual: `no suitable method found (candidates: 2)` /
`Invalid use of fused types` at `m_ceil_ll(..., x)`; Cython cannot compile cell `S.lam`. Fix compiled and run
(15 lines in `helpers.py`): `own_names(arg)` in `_ScopedHelperCalls` returns False when any name in the argument
has a binding outside the formula's own scope (ScopeProvider, as `expression_kinds._own` does); `module_math`
appends `_ll` only when it holds. With it the lambda gets `m_ceil(...)`, the formula level keeps `m_floor_ll`,
52 cases equal live. A comprehension variable of the same name did not reproduce (the parameter is typed object).

### Lower
| # | Finding | Evidence | Fix |
|---|---|---|---|
| R4 | Providers are compiled without `-ffp-contract=off` (`build.py:122-123`), and `_native_rounding` contains `native_round_digits`, whose tie test needs the product and the `fma` kept apart (`_rounding.pxd:31-36`) | x86-64 GCC: 24M cases equal `round`. A compiler with FMA contraction across statements (GCC on aarch64) was not tested | apply the flag to providers too (not compiled; changes the cache key) |
| R5 | `MX_CY_STRICT_TABLES`, `MX_CY_STRICT_LOWERING` change whether a build stops (`config.py:81,85`) | read | spec keys only |
| R6 | `math.floor`/`ceil` of a typed double beyond 2**63 raises OverflowError where live returns an int (`_native_math.pyx:127-158`) | read; raises, never silent | document |
| R7 | `LookupContext._mx_enable_private_lookup` is an ordinary Python method (`_native_rounding.pyx:23-24`); enabled on a receiver two threads share, the unsynchronised memo could return a result for another input | read; the compiler enables it only on item roots (`item_roots.py:28-34`) | goes away with C3 |

Checked and equal to live (rel 1e-10, class included): 2,796 cases over `round` with one and two arguments
(ties, negative digits, large values), `math.exp/log/sqrt/erf/isinf/floor/ceil/isclose` and constants through a
Reference and through `import math [as n]`, min/max of one kind, mixed kinds, nan, -0.0 and integers beyond 2**53,
`int()` identity, divisibility with negative and zero divisors (same exception class). Native rounding fuzz:
24,000,000 comparisons of `_mx_round_digits`/`_mx_round_one` with `round`, digits -22..22, 0 differences.
`math.erf` equals libm `erf` on CPython 3.10 (2M values). Unit tests of the area: 58 passed.

## 5. Table split
Not a table area. One remark from the wiring: R2 and C8 argue that the base table commit should add the runtime
import, the recorder and the table reports only for a model that has a frame; then a model without tables is
byte-identical before and after the table commits.

## 6. Bottom line
Own modules today: 2,128 non-test lines.

(a) Keeps the measured speed: about **1,600** lines (C1 285, C4 70, C2a 12, C7 11, C6 144; C6 costs 4-5 s of
compile time per rebuild, no run time). Every one of these is under half a day.

(b) Within 25-30%: about **930** lines: additionally C3 (230), C2b (360), C5 (90). What stays is M1 (math),
M3 (guarded round), M6 (`int()` identity) and M7 (result class of `**` and min/max). The brief's ablation puts
M5 + M6 + M8 together at <= 2%; C3 is the one cut without a number and must be timed on the five models that
call `round` before it is taken.

Order of work: R1, R2, R3 now; C1, C4, C2a, C7 now; C3 after one measurement; C2b, C5, C8 and the
escaping-result rule for mixed min/max as proposal text.
