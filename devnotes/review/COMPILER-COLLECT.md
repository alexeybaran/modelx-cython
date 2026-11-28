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

## Measured memory savings

Historical one-worker MYGA SEC full projections used the same book and cash-flow total on both routes. Direct retained all policies during measurement, and the driver performed no policy deletion. These are median peak RSS values from five fresh-process pairs.

| Mode | Policies | Direct MiB | Collect MiB | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 14 | 754.3 | 119.0 | 84.2% |
| Exported Python | 14 | 134.8 | 96.8 | 28.1% |
| Compiled | 14 | 110.9 | 98.3 | 11.3% |
| Exported Python | 1,050 | 3,286.1 | 97.5 | 97.0% |
| Compiled | 1,050 | 1,205.7 | 99.3 | 91.8% |

A separate timing comparison deleted Direct policies inside the stopwatch. The one-worker Direct driver also ran generation-zero GC every 64 policies. Timing compares equal books within each row and reports milliseconds per policy. Peak memory above is from the retained-state runs only.

| Mode | Policies | Workers | Direct ms/policy | Collect ms/policy |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 28 | 1 | 407.675 | 129.591 |
| Exported Python | 168 | 1 | 16.729 | 12.757 |
| Compiled | 39,056 | 1 | 0.278 | 0.226 |
| Compiled | 39,056 | 4 | 0.071 | 0.069 |

These timings are supporting evidence from WSL2 with an eight-core Ryzen 7 5800H and free-threaded CPython 3.14.8. The compiled four-worker figures show near parity. Shared serial data preparation can limit cold scaling. The later SEC DV01 control used 1,050 policies, one path and three closes. It ran the full Collect report in 11.774 s at one worker and 3.185 s at four (3.70×) on its measured pin. The last shared scalar change was verified separately with paired native instruction counts and 72 matching report values. It does not replace that scaling receipt.

## Installation and review

This compiler change depends on the modelx Collect API and exported hook, version 1. The compiler requires `modelx>=0.33.1.dev0`. For review before a compatible release is published, install both checkouts in one pip command. Use the `free-threaded` compiler extra on a free-threaded interpreter. A normal public installation needs a release containing that API.

The compiler commits separate the native evaluator, the export hook and Direct compatibility, the tests, and the package requirement. The companion modelx change defines the Python API and export protocol. The DV01 example is a separate lifelib consumer. [Exact paired package verification](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105) · [Package checks and reproduction](https://github.com/alexeybaran/modelx-cython/tree/review/fast-sec-collect-evidence/devnotes/review)
