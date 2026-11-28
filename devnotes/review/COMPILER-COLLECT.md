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

The 14-policy book is the common comparison across all three execution modes. The 1,050-policy book shows the effect of retaining a larger set of policy caches in standalone Python and compiled code.

## Python API

`set_collect()` binds one Space parameter to a zero-argument Cell or a named Reference that supplies the coordinates. The `outputs` argument is a zero-argument Python function. `collect()` takes the remaining Space parameters.

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

Live modelx records each Collect invocation as a cached dependency node. Consumers depend on that node. Changes to the coordinate domain, shared inputs or authored policy inputs invalidate the Collect result and downstream Cells. Temporary policy Cells can cache values during one evaluation. Their internal calls leave no persistent per-policy dependency graph. Public Direct items keep their usual caches and traces. Collect declarations survive saving and reopening the model.

### Python export and compilation

The standalone Python export implements the same API. For eligible calls it can use multiple workers on free-threaded Python with the GIL off. Choose the worker count with `_mx_collect_workers`. Compiled packages also accept `collect_workers` in the spec. Existing lock declarations govern shared input reads.

Compiled Collect keeps eligible numeric fields and subtotals in native storage. Compiled Cells consume them directly. Python views are created when needed, and Python-valued fields can coexist with numeric fields. Workers combine subtotals independently, so parallel sums can differ in floating-point rounding. A model formula can mix Direct and Collect on the same Space. Python callers can also call a Direct Cell outside the Collect outputs, including when the compiler sample used only Collect for that Space.

The optional SEC DV01 example calls `Actual.collect(scen_id)` before Valuation to gather opening records for the requested path. Its Direct route reads Actual's public policy ItemSpaces. Both routes use the same model-owned CSV interface and valuation formulas.

## Supporting timing on full MYGA SEC projections

The elapsed-time runs use the same book on both routes within each mode. The Direct driver deletes each policy item inside the timed interval and runs generation-zero GC every 64 policies at one worker. Times are medians from five runs.

| Mode | Policies | Workers | Direct ms/policy | Collect ms/policy |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 28 | 1 | 541.925 | 151.141 |
| Exported Python | 840 | 1 | 20.659 | 16.802 |
| Exported Python | 840 | 4 | 7.720 | 4.710 |
| Compiled | 34,300 | 1 | 0.290 | 0.246 |
| Compiled | 34,300 | 4 | 0.070 | 0.065 |

## Full SEC DV01 and CS01 report

The report uses 1,050 policies, one ESG path and three closes. All 72 DV01 and CS01 values matched the live reference within the agreed parallel-sum tolerance. Wall times are medians of five fresh processes per route and worker count.

| Route | 1 worker | 4 workers | Scaling | Peak RSS at 4 workers |
| --- | ---: | ---: | ---: | ---: |
| Direct | 13.964 s | 3.489 s | 4.00× | 349.7 MiB |
| Collect | 11.473 s | 3.085 s | 3.72× | 180.9 MiB |

RILA and MYGA compiled Collect scaled 3.71× and 3.76×. DIA Collect ran faster than DIA Direct at each worker count and scaled 1.83×. For these three products, compiling the Collect build took 8–13% longer than compiling Direct. [Matched performance evidence](REFRESH.md)

## Installation and review

This compiler change depends on the modelx Collect API and exported hook, version 1. The compiler requires `modelx>=0.33.1.dev0`. For review before a compatible release is published, install both checkouts in one pip command. Use the `free-threaded` compiler extra on a free-threaded interpreter. A normal public installation needs a release containing that API.

The compiler commits separate the native evaluator and result storage (`6e7c883`), compiler wiring and Direct access (`2370a90`), tests (`8b728f8`), and installation requirements (`0cb1cb7`). The compiler delta is +1,672/−19 production lines, +2,719/−0 test lines and +53/−3 documentation and packaging lines. The [modelx companion](MODELX-COLLECT.md) defines the API and the export protocol used here.

All six exact Collect CI jobs passed across Linux, Windows and macOS on Python 3.12 and free-threaded Python 3.14. Test coverage includes function-only declarations, serialization, exported and compiled isolation, mixed native and Python outputs and consumers, both orders of Direct and Collect calls, and Direct access after a Collect-only sample. [Exact package verification](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105) · [Package checks and reproduction](https://github.com/alexeybaran/modelx-cython/tree/review/fast-sec-collect-evidence/devnotes/review)
