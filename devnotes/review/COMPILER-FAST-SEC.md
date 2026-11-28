# Faster compiled scalar models

This change reduces the cost of evaluating scalar Cells in compiled models. Generated code can call other Cells through typed entry points and read cached values from native storage. Public Cell calls and Python fallback remain available.

On all 12 released lifelib SEC products, the compiled results matched live modelx. On the same books, the new compiler used **56–77% fewer instructions per projection** and ran **3.04–7.51× as fast at one worker** and **1.78–5.80× as fast at four workers**.

## Compiler changes

- **Typed calls and storage.** Generated Cell calls check the receiver type and argument bounds before taking the native path. Bounded integer coordinates can index a dense value array, including ranges that begin below or above zero. Separate flags mark written entries, so the value array can skip zero-fill. Wider or sparse keys can use a native hash table.
- **Shared results.** Generated code can borrow eligible items and arrays while their owners keep them alive. Explicitly locked Spaces synchronize cache publication. Typed ItemSpace construction binds parameters and supported dynamic References.
- **Scalar operations.** The compiler infers real types through local assignments, loops and typed expressions. It lowers eligible powers, standard `math` calls and rounding while retaining the one-entry power cache where applicable. Unsupported arguments and rebound `math` References use the Python operation.
- **Translation.** Compiler passes share formula syntax and scope analysis. Build errors identify the affected Cell. The native build disables floating-point contraction.

For model authors, a call outside a bounded cache raises an `IndexError` that names the Cell and accepted range. Use `cells_param_size` to widen it, for example `{"year": (1900, 2100)}` with an exclusive upper bound. Reading an unassigned local raises `UnboundLocalError`, and assigning a fractional number to an integer Reference raises `TypeError`. Ordinary signed-64-bit arithmetic follows the upstream compiler's behavior.

## Evidence

Both compilers use lifelib's documented native build settings, the same ordered books, and the same Direct driver at each worker count. At four workers a rolling queue keeps work available during periodic cleanup, which bounds memory on the upstream builds. Timings are medians of five fresh-process rounds on WSL2 with free-threaded CPython 3.14.8, Cython 3.3.2 and the GIL off. Scaling is the Fast SEC one-worker time divided by its four-worker time. [Build settings, timings, ranges and peak RSS](SEC-BUILD-BENCHMARKS.md#matched-upstream-comparison)

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

Each SEC product runs one projection per policy. BasicTerm_SC runs `pv_net_cf()` ten times through the same 10,000-point book. MYGA DV01 is a full report with 1,050 policies and 40 projections per policy, including Actual and Valuation. It uses the dynamic Space Reference exporter. All 72 report values matched live modelx bit for bit. [DV01 method and results](REFRESH.md#full-sec-dv01-and-cs01-report)

A whole-book Fast SEC run with one final cleanup scaled **2.07–3.63×** across the SEC products other than DIA. DIA scaled **1.51×**, with a substantial share of its first run spent preparing shared premium tables. [Whole-book results by model](SEC-BUILD-BENCHMARKS.md#fast-sec-whole-book-comparison) · [DIA preparation analysis](SEC-BUILD-BENCHMARKS.md#whole-book-direct-and-collect-scaling)

## Validation and compatibility

All 12 exact review jobs passed on Windows, Linux and macOS with standard Python 3.12 and free-threaded Python 3.14. The released SEC checks build Term, MYGA and SPIA with `mx2cy` and compare all shipped model points against live modelx, including cached values and raised calls. The numeric benchmark tolerance is relative `1e-10` and absolute `1e-7`. Tests also cover cache bounds, powers, shared cache publication and standalone imports. [Exact CI](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105) · [Install and reproduction steps](README.md#repeat-the-installed-example)

The Python floor is 3.10. Standard builds require Cython 3.2.4 or later, and the `free-threaded` extra selects Cython 3.3.2 or later. Wheels and source distributions include the native sources. Generated packages run independently of the compiler.

## Review order

The eight commits are based on compiler main `d22426f`. The compiler can be reviewed with modelx 0.33.0.

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

The `use_libm` proposal touches the same math emitter. Parameter types, Cython directives, and reserved-name validation also touch translation. [Merge review notes](README.md#open-upstream-compiler-changes)
