# Reduce portfolio memory with `Space.collect()`

This PR implements the compiler hook for modelx's `Space.collect()`. Collect evaluates policies in temporary ItemSpaces, keeps the declared outputs, and releases the other calculated values. Direct calls on the same Space keep their own ItemSpaces and caches for inspection.

## Peak RAM on full MYGA SEC projections

Both routes sum monthly `net_cf` over the same ordered book. Direct keeps all policy state through measurement. The driver deletes no policies in either route. These are peak RSS medians and ranges from five alternating fresh-process pairs at one worker, in MiB.

| Mode | Policies | Direct MiB | Collect MiB | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 14 | 751.66 [751.34–751.88] | 123.23 [122.91–123.62] | 83.6% |
| Exported Python | 14 | 140.30 [140.27–140.69] | 102.09 [101.85–102.13] | 27.2% |
| Compiled | 14 | 116.57 [116.45–116.62] | 104.14 [104.05–104.32] | 10.7% |
| Exported Python | 1,050 | 3,291.62 [3,291.30–3,291.71] | 102.76 [102.65–102.88] | **96.9%** |
| Compiled | 1,050 | 1,208.00 [1,207.98–1,208.02] | 104.29 [104.21–104.84] | **91.4%** |

The 14-policy book is common to all three modes. The harness verifies that Direct items remain accessible. Collect's peak memory depends on how much detail the report asks it to retain. [Memory measurement details](REFRESH.md#myga-sec-memory-with-no-policy-deletion)

## Python API

`set_collect()` binds one Space parameter to a zero-argument Cell or named Reference that supplies distinct coordinates. A zero-argument Python function declares individual values, sums, or both. Call `collect()` on the declaring Space with the remaining parameters.

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

Live modelx records one dependency node per Collect call. Reads of shared Cells and References attach to that node, so input changes invalidate the result and its consumers. In the example, setting `projection.policy_ids = (1, 2)` makes the next `projection.collect(3).sum.pv` return `90.0`. Temporary policy caches last for the evaluation. Collect declarations survive a save and reopen.

### Python export and compilation

- **Workers.** Exported Python and compiled models can run eligible calls in parallel on free-threaded Python with the GIL off. Set `_mx_collect_workers` on the model or `collect_workers` in the compiler spec. Compiled Collect defaults to one worker. Parallel access to shared Spaces requires `locked_spaces`. The compiler checks those bindings and can use serial execution when the caller already holds a shared lock.
- **Native results.** Numeric outputs and subtotals can stay in native storage. Compiled Cells can read them directly. Python views are created on demand, and a result can contain both native and Python-valued fields. Parallel workers combine subtotals independently, which can change floating-point rounding.
- **Item reuse.** An eligible worker resets and reuses a private ItemSpace. Other calls use fresh items. Compiled models still support Direct access to Cells outside the Collect outputs, including when the compiler sample exercised only Collect on that Space.

## Timing

Each comparison uses the same ordered book and checks the same outputs across routes. The timing drivers delete Direct ItemSpaces after evaluating them.

### Live modelx and exported Python timing

These runs use MYGA SEC and periodic Direct cleanup.

| Mode | Policies | Workers | Direct ms/policy | Collect ms/policy |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 28 | 1 | 541.925 | 151.141 |
| Exported Python | 840 | 1 | 20.659 | 16.802 |
| Exported Python | 840 | 4 | 7.720 | 4.710 |

### Compiled Direct and Collect timing

At four workers, the compiled Direct driver queues 64-projection tasks with one cleanup at the end. Times are medians of five alternating fresh-process rounds with the GIL off. Four workers run on separate physical cores.

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

The largest four-worker gains were VA, VUL and RILA. DIA's first run spends much of its time preparing the shared premium schedule. [Timing ranges and preparation analysis](SEC-BUILD-BENCHMARKS.md#whole-book-direct-and-collect-scaling)

Each standard SEC model runs one projection per policy. BasicTerm_SC recalculates the same 10,000-point book ten times, with policy state released and the Collect result cleared between passes. [All matched elapsed ranges](SEC-BUILD-BENCHMARKS.md#compiled-collect-comparison)

### Full SEC DV01 and CS01 report

The full DV01 and CS01 report uses the same CSV inputs and valuation formulas in both routes. Collect gathers opening records with `Actual.collect(scen_id)`, and Direct reads the public Actual items. The benchmark runs one ESG path and three closes, for one Actual and 39 Valuation projections per policy. All 72 report values matched live within the parallel-sum tolerance. [DV01 method and elapsed totals](REFRESH.md#full-sec-dv01-and-cs01-report)

The separate Collect-enabled builds for MYGA, RILA and DIA took 8–13% longer to compile than their Direct builds. [Separate build-time evidence](REFRESH.md)

## Installation and review

Review the modelx companion first. This compiler requires its Collect API and version 1 of the export hook, with `modelx>=0.33.1.dev0`. Install the two pinned checkouts together for review. Use the `free-threaded` compiler extra on a free-threaded interpreter. [modelx companion](MODELX-COLLECT.md) · [Install and reproduction steps](README.md#repeat-the-installed-example)

The four commits cover native evaluation and storage (`6e7c883`), compiler wiring and Direct access (`2370a90`), tests (`8b728f8`), and packaging (`0cb1cb7`).

All six exact Collect CI jobs passed on Linux, Windows and macOS with standard Python 3.12 and free-threaded Python 3.14. Tests cover declaration and serialization, parallel input checks, cache reset and release, mixed native and Python outputs and consumers, and Direct and Collect calls in both orders. [Exact CI](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105)
