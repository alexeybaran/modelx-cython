# Faster compiled models

Compiled scalar insurance models spend much of their time looking up Cell values and crossing Python call boundaries. This change emits typed Cell entries and chooses native storage from the Cell's argument types, ranges and observed usage. Exact receivers take the native path. The existing public interfaces and supported fallback paths remain available.

## What the compiler does

| Mechanism | Effect on generated code |
| --- | --- |
| Shared formula analysis | Translation passes reuse syntax, scope and definite-assignment facts per formula. The build checks the emitted source, maps Cython errors back to the affected Cells and disables floating-point contraction. |
| Typed Cell calls | Generated calls on exact receivers use a typed entry. The entry checks its applicable domain before reading the cache or evaluating the formula. Calls outside the native proof use the appropriate public or Python route. |
| Native caches | Eligible bounded integer coordinates index a dense array, including ranges that start below or above zero. Dense value storage can skip zero-fill because separate publication flags prevent reads before assignment. Eligible wider or sparse keys can use open-addressed native numeric hash storage. |
| Shared items and arrays | Generated consumers can borrow eligible held items and arrays through a shared Space. A compiled owner holds their lifetime. Explicitly locked Spaces synchronize cache publication. Typed ItemSpace construction binds parameters and supported dynamic References. |
| Real powers | The compiler can prove real types through local assignments, accumulation in loops, conditional expressions, typed array indexing and typed calls. Eligible private Cells retain a one-entry memo for the last operands and result. Newly proved scalar expressions can call the checked native real-power helper. Argument evaluation order is preserved. |
| Standard `math` calls and rounding | A native call checks the module's identity through a borrowed pointer. Native `log` takes positive numbers, `sqrt` takes nonnegative numbers, and `exp` takes cases with a finite result. The native `erf` and `isinf` accept supported real arguments. Rebound References and other cases take the corresponding Python operation. Guarded rounding shares this scalar support. |

## Behavior model authors can rely on

- An out-of-range call to a bounded integer cache raises `IndexError` naming the Cell and its accepted range. Use `cells_param_size` to declare a wider domain, for example `{"year": (1900, 2100)}` for an exclusive upper bound of 2100.
- The compiler uses declared bounds and sampled argument ranges to size and guard storage. The stated numeric validation tolerance is relative `1e-10` and absolute `1e-7`. A call that raises in both live and compiled execution can raise different exception types. The tracer records argument ranges and calls that raise.
- A formula that reads a local before assigning it raises `UnboundLocalError`. Writing a fractional number into an integer Reference from Python raises `TypeError`. Ordinary signed-64-bit arithmetic follows the upstream compiler's contract.
- The generated package runs standalone. The wheel and source distribution include the compiler's native sources. The Python floor is 3.10. Standard Python uses Cython 3.2.4 or later. The `free-threaded` extra selects Cython 3.3.2 or later for a free-threaded build.

The existing tests remain in the series. The sample fixtures now assert the bounded-cache `IndexError` for coordinates beyond the sample. The fractional-power test accepts the typed exception signature and checks both `int_pow(0) == 1` and `int_pow(1) == 0.5`. The locked-Space and uslib power tests use the syntax tree to identify classes and native call operands. Added compiled tests cover shared publication, cache fallback, unbound locals, integer Reference writes and import independence.

The released-model check builds with `mx2cy` and compares all shipped points against live modelx, including cached values and raised calls. Run `python .github/scripts/sec_compare.py <work-dir> <model> <product>` with the model's name and lifelib product directory. The SEC CI examples are Term, MYGA and SPIA.

## Evidence

All 12 released SEC products build on both compilers and match live modelx with lifelib's documented native build settings. At one worker, Fast SEC runs **3.04–7.51× as fast** and uses **56–77% fewer instructions per policy** across the twelve pairs. Each pair uses the Direct route and the same ordered book. At one worker the Direct driver collects garbage every 512 projections on each compiler. At four workers a rolling queue keeps 64-projection tasks available while the caller runs generation-zero garbage collection after roughly 512 completed projections. The driver uses the same queue and cleanup on both compilers. Times are medians of five fresh-process rounds on WSL2 with free-threaded CPython 3.14.8, the GIL off and Cython 3.3.2.

| Model | Projections | Upstream 1w µs/proj. | Fast SEC 1w µs/proj. | Upstream 4w µs/proj. | Fast SEC 4w µs/proj. | Instruction reduction (upstream / Fast SEC) | 1w elapsed speedup | 4w elapsed speedup | Fast SEC 1w/4w |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BasicTerm_SC | 100,000 | 89.9 | 33.9 | 38.6 | 19.8 | 2.59× | 2.65× | 1.95× | 1.71× |
| DIA | 12,000 | 1,198.9 | 394.2 | 437.0 | 245.3 | 2.28× | 3.04× | 1.78× | 1.61× |
| FIA | 10,900 | 1,662.2 | 288.2 | 505.0 | 174.7 | 3.34× | 5.77× | 2.89× | 1.65× |
| IUL | 3,500 | 5,302.7 | 945.8 | 1,417.7 | 412.8 | 3.28× | 5.61× | 3.43× | 2.29× |
| MYGA | 34,300 | 1,737.3 | 274.6 | 520.5 | 130.1 | 4.00× | 6.33× | 4.00× | 2.11× |
| RILA | 27,000 | 2,184.4 | 374.8 | 991.9 | 170.9 | 4.32× | 5.83× | 5.80× | 2.19× |
| SPIA | 13,200 | 992.4 | 226.1 | 275.3 | 86.9 | 2.37× | 4.39× | 3.17× | 2.60× |
| Term | 31,900 | 574.4 | 103.4 | 160.3 | 40.3 | 3.80× | 5.55× | 3.98× | 2.56× |
| ULSG | 9,000 | 1,934.0 | 333.3 | 560.9 | 167.0 | 3.40× | 5.80× | 3.36× | 2.00× |
| UL | 9,500 | 1,712.5 | 271.7 | 535.8 | 150.3 | 3.81× | 6.30× | 3.56× | 1.81× |
| VA | 8,000 | 3,210.1 | 427.4 | 1,031.7 | 276.1 | 4.03× | 7.51× | 3.74× | 1.55× |
| VUL | 5,300 | 4,840.6 | 776.0 | 1,500.7 | 340.3 | 3.00× | 6.24× | 4.41× | 2.28× |
| WholeLife | 13,900 | 1,102.6 | 173.1 | 321.6 | 99.0 | 3.61× | 6.37× | 3.25× | 1.75× |
| MYGA DV01 | 42,000 (1,050 × 40) | — | 349.1 | — | 84.1 | — | — | — | 4.15× |

Lifelib's build sets `infer_types=True, cpow=True` on the generated Cython class module and uses `-ffp-contract=off` with GCC. Applying these settings to both arms restored the DIA and VUL upstream builds. Upstream instruction counts also fell for all ten products that had compiled under the generic recipe, by 2.3–31.8%. The largest changes were SPIA, IUL and VA. [Build procedure and per-product comparison](SEC-BUILD-BENCHMARKS.md)

Each released SEC product runs one projection per policy. BasicTerm_SC runs `pv_net_cf()` on ten passes through the same 10,000-point book. Both compilers evaluate the same 100,000 projections. [Full one-worker and four-worker times](SEC-BUILD-BENCHMARKS.md)

The separate MYGA DV01 report runs one Actual and 39 Valuation projections per policy across one ESG path, with one base and 12 shocked valuations at each of three closes. The 1,050-policy book therefore runs **42,000 projections**. The full report took **349.1 µs per projection** at one worker (14.664 s total) and **84.1 µs** at four (3.530 s total). All 72 results matched live modelx bit for bit. The example requires dynamic Reference export and has no matched upstream result.

The detailed benchmark report includes elapsed ranges, peak RSS, translation time and native build time. All 12 exact review CI jobs passed across Windows, Linux and macOS on standard Python 3.12 and free-threaded Python 3.14. [Released SEC report](SEC-BUILD-BENCHMARKS.md) · [SEC DV01 results](REFRESH.md#full-sec-dv01-and-cs01-report) · [Exact CI](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105)

### Fast SEC with whole-book scheduling

This is the same scheduling protocol used for the compiled Direct timing in the Collect comparison. Fast SEC queues the whole book and performs one explicit collection at the end. The upstream comparison above uses periodic cleanup to keep the released compiler inside the WSL memory limit.

| Model | Projections | Fast SEC 1w µs/projection | Fast SEC 4w µs/projection | Scaling |
| --- | --- | --- | --- | --- |
| BasicTerm_SC | 100,000 | 25.6 [25.4–27.6] | 7.9 [7.7–8.2] | 3.22× |
| DIA | 12,000 | 307.6 [305.2–309.2] | 203.3 [197.7–210.1] | 1.51× |
| FIA | 10,900 | 219.2 [216.4–223.6] | 89.9 [86.0–92.1] | 2.44× |
| IUL | 3,500 | 971.3 [932.1–1,022.8] | 327.8 [320.1–330.8] | 2.96× |
| MYGA | 34,300 | 207.1 [206.4–217.6] | 69.7 [67.2–82.2] | 2.97× |
| RILA | 27,000 | 283.8 [275.7–284.3] | 114.0 [113.3–131.4] | 2.49× |
| SPIA | 13,200 | 215.2 [211.5–222.6] | 60.9 [59.1–63.0] | 3.54× |
| Term | 31,900 | 90.8 [89.2–92.4] | 25.8 [24.7–26.3] | 3.52× |
| ULSG | 9,000 | 329.7 [314.4–338.2] | 100.5 [95.2–112.6] | 3.28× |
| UL | 9,500 | 311.6 [282.6–318.9] | 91.4 [86.2–92.9] | 3.41× |
| VA | 8,000 | 377.4 [338.2–382.9] | 182.4 [176.4–194.6] | 2.07× |
| VUL | 5,300 | 660.8 [612.2–731.4] | 247.1 [229.6–256.0] | 2.67× |
| WholeLife | 13,900 | 183.4 [179.5–190.9] | 50.5 [48.5–54.6] | 3.63× |

## Review order

The eight commits are based on compiler main `d22426f`. Counts are additions and deletions from the final branch diff. The compiler can be reviewed with modelx 0.33.0. The optional SEC DV01 example also needs the separate dynamic Space Reference export support.

| Commit | What to review |
| --- | --- |
| `a7389d2` | Native storage and typed-entry primitives |
| `7584758` | Translation and dispatch through those entries |
| `15c2acd` | Cache boundaries and supported fallback tests |
| `6620cb6` | Compiled contract and released-product CI |
| `87e654e` | Standard `math` Reference calls and fixtures |
| `d5bef6a` | Guarded native helpers and lifetimes of shared items |
| `331b417` | Scalar inference and checked real powers |
| `eb7fcc4` | Installation requirements and tested toolchains |

The branch adds 6,922 lines and removes 288 in production compiler code. Tests add 6,114 and remove 44. Documentation, CI and packaging add 358 and remove 170. [Commit diff](https://github.com/alexeybaran/modelx-cython/compare/d22426fbac8637f84510902c2fa33b6466e5acb4...eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df)

The existing `use_libm` proposal intersects the native math emitter. The other open proposals around parameter types, compiler directives, and reserved-name validation intersect translation. This branch is based on current main. [Native math proposal](https://github.com/fumitoh/modelx-cython/pull/60) · [Parameter types](https://github.com/fumitoh/modelx-cython/pull/59) · [Directives](https://github.com/fumitoh/modelx-cython/pull/58) · [Name validation](https://github.com/fumitoh/modelx-cython/pull/57)

## Earlier qualification

The broad released-model census used the earlier Fast SEC compiler `503b07c` against upstream `d22426f` with identical modelx and lifelib text. Of the 95 driven models, 73 passed with both compilers, nine passed only with Fast SEC, and 13 had a shared failure class. The report lists each failure. It excludes unmatched book sizes or hosts from ratios as applicable. Median candidate-to-upstream ratios were 0.266× for SEC instructions, 0.990× for Regional instructions, 1.380× for translation time and 1.237× for native build time. `LTC_JP_S` used 4.76% more instructions with a native hash cache for `Projection.care_surv`, with about 22% lower estimated incremental memory per policy. The owner accepted this trade-off. [Census, timings and cache attribution](https://github.com/alexeybaran/modelx-cython/blob/c39337225c8d4097bb93b4cd540c547e569ecaa4/devnotes/PR-DESCRIPTION.md#broad-census-of-released-lifelib-models)

The original description also records separate power ablations, the borrowed math-pointer experiment, a Python cached-call and Reference-write microbenchmark, the detailed test changes, and the earlier CI matrix with unassigned macOS jobs. Those measurements retain their original pins. The current description uses the eight-commit branch and exact package CI above. [Original Fast SEC description](https://github.com/alexeybaran/modelx-cython/blob/c39337225c8d4097bb93b4cd540c547e569ecaa4/devnotes/PR-DESCRIPTION.md)
