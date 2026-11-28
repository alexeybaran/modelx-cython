# Reduce memory for portfolio calculations with Collect

An ordinary Direct loop keeps policy ItemSpaces and cached values available for inspection. A portfolio calculation often needs only a total and a small set of results per policy. `Collect` evaluates policies in temporary ItemSpaces, keeps the requested outputs, and releases the rest of the calculation state. The same Space still supports Direct calls, including calls to Cells outside the Collect outputs. Requesting more individual values increases the size of the retained report.

## Peak RAM on full MYGA SEC projections

Both routes sum monthly `net_cf` over the same ordered book. Direct keeps all policy state through measurement. The driver deletes no policies in either route. These are peak RSS medians and ranges from five alternating fresh-process pairs at one worker, in MiB.

| Mode | Policies | Direct MiB | Collect MiB | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 14 | 751.66 [751.34–751.88] | 123.23 [122.91–123.62] | 83.6% |
| Exported Python | 14 | 140.30 [140.27–140.69] | 102.09 [101.85–102.13] | 27.2% |
| Compiled | 14 | 116.57 [116.45–116.62] | 104.14 [104.05–104.32] | 10.7% |
| Exported Python | 1,050 | 3,291.62 [3,291.30–3,291.71] | 102.76 [102.65–102.88] | 96.9% |
| Compiled | 1,050 | 1,208.00 [1,207.98–1,208.02] | 104.29 [104.21–104.84] | 91.4% |

The 14-policy book is the common comparison across all three execution modes. The harness checks the retained Direct item count in live modelx and exported Python, and checks item identity on repeated Direct lookup in compiled code. The 1,050-policy book shows the cost of retaining a larger set of policy caches in standalone Python and compiled code. [Earlier memory procedure and receipts](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/r3/handover-review/parallel/20261113/round3/review/r4/collect-qualification/PR2-RELEASE.md#peak-ram-on-full-myga-sec-projections)

## Python API

`set_collect()` binds one Space parameter to a zero-argument Cell or a named Reference that supplies a nonempty sequence of distinct coordinates. The `outputs` argument is a zero-argument Python function. Passing a string raises `TypeError`. `collect()` is available on the declaring Space and takes the remaining Space parameters. Reading it on an individual ItemSpace raises `AttributeError`.

```python
import modelx as mx

model = mx.new_model('Portfolio')
projection = model.new_space('Projection', formula=lambda policy_id, scenario: None)
projection.policy_ids = (1, 2, 3)

@mx.defcells(projection)
def cashflow(month):
    return 1.0 * policy_id * scenario * (month + 1)

def outputs():
    return {
        'individual': {'cashflow': {m: cashflow(m) for m in range(4)}},
        'sum': {'pv': cashflow(0) + cashflow(1) + cashflow(2) + cashflow(3)},
    }

projection.set_collect(over={'policy_id': 'policy_ids'}, outputs=outputs)
result = projection.collect(3)
assert result.sum.pv == 180.0
assert result.individual[2].cashflow(1) == result.individual[2].cashflow[1] == 12.0
assert projection[2, 3].cashflow(1) == 12.0
```

### Live dependency tracing

Live modelx records each Collect invocation as a cached dependency node. Consumers depend on that node. Reads of shared Cells and References inside the temporary policies attach to the Collect node. Changes to the coordinate domain, shared inputs or authored policy inputs invalidate the Collect result and downstream Cells. In the example, assigning `projection.policy_ids = (1, 2)` changes the next `projection.collect(3).sum.pv` to `90.0`. Temporary policy Cells can cache values during one evaluation. Their internal calls leave no persistent per-policy dependency graph. Public Direct items keep their usual caches and traces. Collect declarations survive saving and reopening the model.

### Python export and compilation

The standalone Python export implements the same API. For eligible calls it can use multiple workers on free-threaded Python with the GIL off. Choose the worker count with `_mx_collect_workers`. Compiled packages also accept `collect_workers` in the spec. A compiled package defaults to one worker. The generated class preserves the export's Collect property and implements its versioned hook.

Parallel calls require safe shared reads. The compiled runtime reports a shared Space Reference without a lock, or an unsupported direct read of the model or parent, with `ValueError`. Export the shared Space in `locked_spaces` or request one worker. Checks also cover dynamic bindings. Approved calls execute synchronously when the GIL is on or when an active caller lock or an unresolved child binding prevents safe parallel execution. A parallel evaluation starts workers for the call and joins them before returning.

Compiled Collect keeps eligible numeric fields and subtotals in native storage. Compiled Cells consume them directly. Python views are created when needed, and Python-valued fields can coexist with numeric fields. Workers combine subtotals independently, so parallel sums can differ in floating-point rounding. A model formula can mix Direct and Collect on the same Space. Python callers can also call a Direct Cell outside the Collect outputs, including when the compiler sample used only Collect for that Space.

An eligible worker retains a private receiver between coordinates and resets its fields, numeric caches and written flags before reuse. A dynamic Space factory can take a fresh ItemSpace path. Cleanup covers temporary Collect-owned policy state. Public Direct items and intentionally retained shared results have their own lifetime. Individual records are addressable by the collected coordinate.

The optional SEC DV01 example calls `Actual.collect(scen_id)` before Valuation to gather opening records for the requested path. Its Direct route reads Actual's public policy ItemSpaces. Both routes use the same model-owned CSV interface and valuation formulas.

## Live modelx and exported Python timing

Each comparison projects the same ordered MYGA SEC book and checks the same cash flows. The timed Direct driver deletes the policy after calculating it and performs periodic cleanup. The retained-state RAM comparison above keeps all Direct policy items available.

| Mode | Policies | Workers | Direct ms/policy | Collect ms/policy |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 28 | 1 | 541.925 | 151.141 |
| Exported Python | 840 | 1 | 20.659 | 16.802 |
| Exported Python | 840 | 4 | 7.720 | 4.710 |

## Compiled Direct and Collect timing

Both compiled routes evaluate the same ordered book and return the same cash-flow output. Direct deletes each policy item inside the timed interval and queues the whole book in 64-projection tasks at four workers, with one explicit cleanup at the end. Collect uses a single public call per pass. All timings are medians of five alternating fresh-process rounds with the GIL off. Four workers run on separate physical cores.

| Model | Projections | Direct 1w µs/proj. | Collect 1w µs/proj. | Direct 4w µs/proj. | Collect 4w µs/proj. | Direct 1w/4w | Collect 1w/4w | 4w speedup (Direct / Collect) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BasicTerm_SC | 100,000 | 26.1 | 25.4 | 7.9 | 7.2 | 3.30× | 3.51× | 1.10× |
| DIA | 12,000 | 308.9 | 302.5 | 202.4 | 201.3 | 1.53× | 1.50× | 1.01× |
| FIA | 10,900 | 219.3 | 215.4 | 87.8 | 72.1 | 2.50× | 2.99× | 1.22× |
| IUL | 3,500 | 949.5 | 942.2 | 353.1 | 254.0 | 2.69× | 3.71× | 1.39× |
| MYGA | 34,300 | 203.1 | 220.0 | 68.6 | 68.3 | 2.96× | 3.22× | 1.00× |
| RILA | 27,000 | 307.4 | 301.7 | 130.1 | 88.9 | 2.36× | 3.39× | 1.46× |
| SPIA | 13,200 | 211.7 | 219.7 | 62.2 | 60.2 | 3.40× | 3.65× | 1.03× |
| Term | 31,900 | 92.9 | 94.5 | 25.7 | 24.9 | 3.62× | 3.79× | 1.03× |
| ULSG | 9,000 | 311.8 | 308.9 | 100.4 | 85.4 | 3.10× | 3.62× | 1.18× |
| UL | 9,500 | 305.8 | 298.0 | 93.4 | 89.3 | 3.28× | 3.34× | 1.05× |
| VA | 8,000 | 367.0 | 367.2 | 185.0 | 104.6 | 1.98× | 3.51× | 1.77× |
| VUL | 5,300 | 717.2 | 582.3 | 292.2 | 169.5 | 2.45× | 3.43× | 1.72× |
| WholeLife | 13,900 | 190.7 | 192.9 | 56.4 | 52.7 | 3.38× | 3.66× | 1.07× |
| MYGA DV01 | 42,000 (1,050 × 40) | 332.5 | 273.2 | 83.1 | 73.4 | 4.00× | 3.72× | 1.13× |

The largest four-worker gains were VA at 1.77×, VUL at 1.72× and RILA at 1.46×. The four-worker medians for MYGA and DIA were within 1% across routes. DIA still spends much of its first run preparing the shared premium schedule. [Elapsed ranges and preparation analysis](SEC-BUILD-BENCHMARKS.md#whole-book-direct-and-collect-scaling)

Each standard SEC model runs one projection per policy. BasicTerm_SC recalculates the same 10,000-point book ten times, with policy state released and the Collect result cleared between passes. [All matched elapsed ranges](SEC-BUILD-BENCHMARKS.md#compiled-collect-comparison)

### Full SEC DV01 and CS01 report

The MYGA DV01 row uses one ESG path and three closes. Each of the 1,050 policies runs one Actual and 39 Valuation projections. Full-report medians were 13.964 s and 3.489 s for Direct, and 11.473 s and 3.085 s for Collect, at one and four workers respectively. All 72 DV01 and CS01 values matched live within the agreed parallel-sum tolerance. [DV01 method and receipts](REFRESH.md#full-sec-dv01-and-cs01-report)

The separate Collect-enabled builds for MYGA, RILA and DIA took 8–13% longer to compile than their Direct builds. [Separate build-time evidence](REFRESH.md)

## Installation and review

This compiler change depends on the modelx Collect API and exported hook, version 1. The compiler requires `modelx>=0.33.1.dev0`. For review before a compatible release is published, install both checkouts in one pip command. Use the `free-threaded` compiler extra on a free-threaded interpreter. A normal public installation needs a release containing that API.

The compiler commits separate the native evaluator and result storage (`6e7c883`), compiler wiring and Direct access (`2370a90`), tests (`8b728f8`), and installation requirements (`0cb1cb7`). The compiler delta is +1,672/−19 production lines, +2,719/−0 test lines and +53/−3 documentation and packaging lines. The [modelx companion](MODELX-COLLECT.md) defines the API and the export protocol used here.

All six exact Collect CI jobs passed across Linux, Windows and macOS on Python 3.12 and free-threaded Python 3.14. Test coverage includes function-only declarations, serialization, exported and compiled isolation, mixed native and Python outputs and consumers, both orders of Direct and Collect calls, and Direct access after a Collect-only sample. Reuse tests check numeric cache reset, discard of state from earlier coordinates, bounded live object counts, and a diagnostic for an unsupported reset declaration. [Exact package verification](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105) · [Package checks and reproduction](https://github.com/alexeybaran/modelx-cython/tree/review/fast-sec-collect-evidence/devnotes/review)

## Earlier qualification

The original description includes the worker-slot reuse and reset experiments, retained and deleting Direct controls, an eight-worker DV01 comparison, the borrowed-read controls and the original Regional pilot. Its DV01 memory measurements used Valuation Collect with retained Direct Actual items. The current example collects opening records from Actual and measures the full report on the current build. The original ordered Python accumulation design and its bitwise agreement figures describe those earlier builds. The current compiled path uses native subtotals and the parallel-sum tolerance explained above. [Original compiled Collect description](https://github.com/alexeybaran/modelx-cython/blob/be40323abe7e5c5f8d12a7f1dbe7b2d89d77f5d9/devnotes/PR-DESCRIPTION-collect.md) · [Earlier API and validation](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/r3/handover-review/parallel/20261113/round3/review/r4/collect-qualification/PR2-RELEASE.md)
