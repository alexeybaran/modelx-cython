# Faster compiled models

Compiled scalar insurance models spend much of their time looking up Cell values and crossing Python call boundaries. This change emits typed Cell entries and chooses native storage from the Cell's argument types, ranges and observed usage. Exact receivers take the native path. The existing public interfaces and supported fallback paths remain available.

## What the compiler does

| Mechanism | Effect on generated code |
| --- | --- |
| Shared formula analysis | Translation passes reuse syntax, scope and definite-assignment facts. The build maps Cython errors back to the affected Cells and disables floating-point contraction. |
| Typed Cell calls | Generated calls on exact receivers use a typed entry. The entry checks its applicable domain before reading the cache or evaluating the formula. Calls outside the native proof use the appropriate public or Python route. |
| Native caches | Eligible bounded integer coordinates index a dense array, including ranges that start below or above zero. Publication flags distinguish calculated values from allocated slots. Eligible wider or sparse keys can use native numeric hash storage. |
| Shared items and arrays | Generated consumers can read held results through a shared Space. Explicitly locked Spaces synchronize cache publication and preserve the shared read contract. |
| Real powers | The compiler proves scalar kinds through assignments, supported expressions and typed calls. Eligible private Cells retain their one-entry power memo. Newly proved scalar expressions can call the checked native real-power helper. The generated code preserves argument evaluation order. |
| Standard `math` calls and rounding | A native call through a `math` Reference checks the module's identity using a borrowed pointer. Rebound References and unsupported cases take the corresponding Python operation. Guarded rounding and other supported scalar operations share the same compiler support. |

## Behavior model authors can rely on

- An out-of-range call to a bounded integer cache raises `IndexError` naming the Cell and its accepted range. Use `cells_param_size` to declare a wider domain, for example `{"year": (1900, 2100)}` for an exclusive upper bound of 2100.
- A formula that reads a local before assigning it raises `UnboundLocalError`. Writing a fractional number into an integer Reference from Python raises `TypeError`. Ordinary signed-64-bit arithmetic follows the upstream compiler's contract.
- The generated package runs standalone. The wheel and source distribution include the compiler's native sources. The Python floor is 3.10. Standard Python uses Cython 3.2.4 or later. The `free-threaded` extra selects Cython 3.3.2 or later for a free-threaded build.

The generated-source tests now inspect the relevant calls and operands through the syntax tree. Boundary tests assert the accepted values and errors for ranges, real powers, unbound locals and integer References. Compiled tests also exercise shared publication, cache fallback and import independence.

## Evidence

All 12 released SEC products build on both compilers and match live modelx with lifelib's documented native build settings. At one worker, Fast SEC runs **3.04–7.51× as fast** and uses **56–77% fewer instructions per policy** across the twelve pairs. Each pair uses the Direct route and the same ordered book. For the SEC products and BasicTerm_SC, both compilers and worker counts use the same cleanup schedule, with garbage collection after each batch of 512 completed projections. Times are medians of five fresh-process rounds on WSL2 with free-threaded CPython 3.14.8, the GIL off and Cython 3.3.2.

| SEC product | Projections | Upstream 1w µs/projection | Fast SEC 1w µs/projection | Instruction reduction (upstream / Fast SEC) | 1w elapsed speedup | Upstream 1w/4w | Fast SEC 1w/4w | Scaling ratio (Fast SEC / upstream) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BasicTerm_SC | 100,000 | 89.9 | 33.9 | 2.59× | 2.65× | 2.07× | 1.66× | 0.80× |
| DIA | 12,000 | 1,198.9 | 394.2 | 2.28× | 3.04× | 2.66× | 1.60× | 0.60× |
| FIA | 10,900 | 1,662.2 | 288.2 | 3.34× | 5.77× | 3.42× | 1.75× | 0.51× |
| IUL | 3,500 | 5,302.7 | 945.8 | 3.28× | 5.61× | 3.46× | 2.32× | 0.67× |
| MYGA | 34,300 | 1,737.3 | 274.6 | 4.00× | 6.33× | 3.41× | 2.24× | 0.66× |
| RILA | 27,000 | 2,184.4 | 374.8 | 4.32× | 5.83× | 2.34× | 2.24× | 0.96× |
| SPIA | 13,200 | 992.4 | 226.1 | 2.37× | 4.39× | 3.45× | 2.37× | 0.69× |
| Term | 31,900 | 574.4 | 103.4 | 3.80× | 5.55× | 3.04× | 2.34× | 0.77× |
| ULSG | 9,000 | 1,934.0 | 333.3 | 3.40× | 5.80× | 3.50× | 2.18× | 0.62× |
| UL | 9,500 | 1,712.5 | 271.7 | 3.81× | 6.30× | 3.18× | 1.93× | 0.61× |
| VA | 8,000 | 3,210.1 | 427.4 | 4.03× | 7.51× | 3.46× | 1.78× | 0.51× |
| VUL | 5,300 | 4,840.6 | 776.0 | 3.00× | 6.24× | 3.72× | 2.54× | 0.68× |
| WholeLife | 13,900 | 1,102.6 | 173.1 | 3.61× | 6.37× | 2.99× | 1.94× | 0.65× |
| MYGA DV01 | 42,000 (1,050 × 40) | — | 349.1 | — | — | — | 4.15× | — |

Lifelib's build sets `infer_types=True, cpow=True` on the generated Cython class module and uses `-ffp-contract=off` with GCC. Applying these settings to both arms restored the DIA and VUL upstream builds. Upstream instruction counts also fell for all ten products that had compiled under the generic recipe, by 2.3–31.8%. The largest changes were SPIA, IUL and VA. [Build procedure and per-product comparison](SEC-BUILD-BENCHMARKS.md)

Each released SEC product runs one projection per policy. BasicTerm_SC runs `pv_net_cf()` on ten passes through the same 10,000-point book. Both compilers evaluate the same 100,000 projections. The scaling ratio is Fast SEC's 1w/4w divided by upstream's 1w/4w. [Full one-worker and four-worker times](SEC-BUILD-BENCHMARKS.md)

The separate MYGA DV01 report runs one Actual and 39 Valuation projections per policy across one ESG path, with one base and 12 shocked valuations at each of three closes. The 1,050-policy book therefore runs **42,000 projections**. The full report took **349.1 µs per projection** at one worker (14.664 s total) and **84.1 µs** at four (3.530 s total). All 72 results matched live modelx bit for bit. The example requires dynamic Reference export and has no matched upstream result.

The detailed benchmark report includes elapsed ranges, peak RSS, translation time and native build time. All 12 exact review CI jobs passed across Windows, Linux and macOS on standard Python 3.12 and free-threaded Python 3.14. [Released SEC report](SEC-BUILD-BENCHMARKS.md) · [SEC DV01 results](REFRESH.md#full-sec-dv01-and-cs01-report) · [Exact CI](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105)

The earlier census remains available with its original pins and limitations. It documented an LTC Japan allocation and instruction-count trade involving `Projection.care_surv`. The owner accepted the earlier result. The current release comparison covers supported SEC workloads. [Earlier full census](https://github.com/alexeybaran/modelx-cython/blob/c39337225c8d4097bb93b4cd540c547e569ecaa4/devnotes/PR-DESCRIPTION.md)

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
