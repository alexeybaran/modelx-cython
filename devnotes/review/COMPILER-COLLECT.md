# Compile Collect with temporary policy state and native numeric results

Portfolio calculations often need a total and a small set of results per policy. `Collect` releases each policy's temporary calculation state after gathering those results. The same Space continues to support Direct calls and their retained ItemSpace caches, including calls to Cells omitted from the Collect outputs.

## Example

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

Live modelx records each Collect invocation as one dependency node. Reads of shared data and authored inputs invalidate that node and its dependents when the data changes. The temporary policy calculations leave no retained per-policy dependency graph. A modelx Python export can use eligible parallel Collect on free-threaded Python with the GIL off. Its worker setting is `_mx_collect_workers` on the loaded model. Compiled packages also accept `collect_workers` in the spec.

Compiled Collect keeps eligible numeric fields and subtotals in native storage. Compiled readers can consume those results directly, with Python views created when needed. Python-valued outputs and native fields can coexist. Parallel subtotal combination can change floating-point rounding. Shared inputs continue to follow the exported lock declarations.

## Memory and runtime evidence

The MYGA SEC comparison runs the same cash-flow projection in live modelx, exported Python and compiled code. The memory runs leave Direct policy state in the model throughout the measurement. Collect disposes of per-policy calculation state. These are peak RSS medians and ranges from five alternating fresh-process pairs on the review heads, in MiB.

| Mode | Policies | Direct MiB | Collect MiB | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 14 | 751.66 [751.34–751.88] | 123.23 [122.91–123.62] | 83.6% |
| Exported Python | 14 | 140.30 [140.27–140.69] | 102.09 [101.85–102.13] | 27.2% |
| Compiled | 14 | 116.57 [116.45–116.62] | 104.14 [104.05–104.32] | 10.7% |
| Exported Python | 1,050 | 3,291.62 [3,291.30–3,291.71] | 102.76 [102.65–102.88] | 96.9% |
| Compiled | 1,050 | 1,208.00 [1,207.98–1,208.02] | 104.29 [104.21–104.84] | 91.4% |

The elapsed-time runs use the same book on both routes within each mode. The Direct driver deletes each policy item inside the timed interval and runs generation-zero GC every 64 policies at one worker. Times are medians from five runs.

| Mode | Policies | Workers | Direct ms/policy | Collect ms/policy |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 28 | 1 | 541.925 | 151.141 |
| Exported Python | 840 | 1 | 20.659 | 16.802 |
| Exported Python | 840 | 4 | 7.720 | 4.710 |
| Compiled | 34,300 | 1 | 0.290 | 0.246 |
| Compiled | 34,300 | 4 | 0.070 | 0.065 |

On the full 1,050-policy SEC DV01 report, compiled Collect took 11.473 s at one worker and 3.085 s at four, or 3.72× scaling. All 72 DV01 and CS01 values matched the live reference within the agreed tolerance for parallel sums. RILA and MYGA Collect scaled 3.71× and 3.76×. DIA Collect was faster than DIA Direct at each worker count and scaled 1.83×. The detailed timings report the additional compile cost of Collect-enabled builds. [Matched performance evidence](REFRESH.md)

## Installation and review

This compiler change depends on the modelx Collect API and exported hook, version 1. The compiler requires `modelx>=0.33.1.dev0`. For review before a compatible release is published, install both checkouts in one pip command. Use the `free-threaded` compiler extra on a free-threaded interpreter. A normal public installation needs a release containing that API.

The compiler commits separate the native evaluator, the export hook and Direct compatibility, the tests, and the package requirement. The companion modelx change defines the Python API and export protocol. The DV01 example is a separate lifelib consumer. [Exact paired package verification](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105) · [Package checks and reproduction](https://github.com/alexeybaran/modelx-cython/tree/review/fast-sec-collect-evidence/devnotes/review)
