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

All 12 released SEC products built and matched live modelx. In the 10 equal-book pairs that also built upstream, Fast SEC used 69–77% fewer instructions per policy and ran in one quarter to about one sixth of the elapsed time. Each row uses the Direct route with one worker. Elapsed figures are medians of five alternating fresh-process rounds on WSL2, with free-threaded CPython 3.14.8, the GIL off and Cython 3.3.2.

| SEC product | Policies | Upstream µs/policy | Fast SEC µs/policy | Fast SEC / upstream instructions |
| --- | ---: | ---: | ---: | ---: |
| DIA | 12,000 | Build failed | 443.3 | — |
| FIA | 10,900 | 1,702.7 | 333.7 | 0.306× |
| IUL | 3,500 | 5,768.1 | 1,031.8 | 0.270× |
| MYGA | 34,300 | 1,948.3 | 296.6 | 0.246× |
| RILA | 27,000 | 1,932.7 | 397.0 | 0.246× |
| SPIA | 13,200 | 1,235.9 | 287.3 | 0.299× |
| Term | 31,900 | 622.8 | 155.8 | 0.275× |
| ULSG | 9,000 | 1,972.9 | 367.2 | 0.300× |
| UL | 9,500 | 1,917.6 | 311.2 | 0.267× |
| VA | 8,000 | 2,996.2 | 499.0 | 0.228× |
| VUL | 5,300 | Build failed | 780.9 | — |
| WholeLife | 13,900 | 1,114.5 | 185.8 | 0.284× |

The upstream DIA and VUL builds fail in Cython with `complex types are unordered`. The separate SEC DV01 example requires dynamic Reference export and has no upstream performance ratio. The detailed results include elapsed ranges, peak RSS, translation time and native build time. All 12 exact review CI jobs passed across Windows, Linux and macOS on standard Python 3.12 and free-threaded Python 3.14. [Benchmark report](REFRESH.md) · [Exact CI](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105)

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
