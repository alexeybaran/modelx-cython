# Thin collect: report

Branch `r4/p2-collect` = `r4/p1-fast-sec` (edf84e7) + two commits. Built against modelx `r4/p2-modelx-collect`
(348f6cf): the compiled class defines `_mx_collect_hook` (version 1, `doc/source/reference/collect_hook.rst`).

## What it is (non-test lines: +203 / -2)

| piece | where | lines |
|---|---|---|
| declaration: `cdef public dict _mx_collect_cache` of a collect Space | parser +4, transformer +2 | 6 |
| hook generated from the exported `collect` and `_mx_collect_item` | `collect.py` `hook` | 45 |
| new item built as `__call__` builds a root (`item_roots`), released after its outputs | `collect.py` `hook` | 9 |
| releasing evaluator: outputs checked by the export's `prepare_output`, added by its `Accumulator` in domain order | `collect.py` `evaluator` | 12 |
| worker threads: shared index, outputs added in domain order as they complete, first failing coordinate raised | `collect.py` `evaluator` | 38 |
| refusal (ValueError) of more than one worker over an unlocked Space or a formula reading `_model`/`_parent`/`_space` | `collect.py` | 14 |
| spec key `collect_workers` (default 1), `_mx_collect_workers` on the model object; wiring | config +10, transformer +10 | 20 |
| module docstrings | `collect.py` | 59 |

Values are the export's own, added in domain order: equal to live modelx bit for bit at one and four workers
(the old implementation's four-worker sums differ from serial in the last bits, rel 5e-14 on KLV). Expected
150 lines (`r4-p2-collect-expectation.md`); 144 without docstrings.

Tests: `tests/compiled/test_collect.py` (one build: values equal the export and live at 1 and 4 workers, other
Space parameters, existing items used and kept, created items released, first failing coordinate at any worker
count, the unlocked-Space refusal; the spec key). The full upstream suite passes unchanged.

## Equal to live (this branch's compiler, Python 3.14t, book total `Book.total() = Projection.collect().sum.net_cf`)

| model | book | live | compiled, 1 worker | compiled, 4 workers |
|---|---|---|---|---|
| KLV_DE_S | 14 policies | -148382.62038214627 | same | same |
| VA_US_S | 9 policies | -20208.677575601363 | same | same |

## Like for like against `r4/integration` collect

On this branch's compiler a KLV_DE_S policy costs about 50 ms (its table reads stay on pandas until the syntax
bridge), on `r4/integration` 45 us, so the collect machinery is compared on the same item code: the thin collect
(this branch's `collect.py` and its wiring, nothing else) applied to `r4/integration` in a local scratch tree,
against `r4/integration` + modelx `r4/modelx-integration` with its fastest collect (`serial_collect: reuse`,
`parallel_collect_reuse: true`). Same spec otherwise (`parallel_workers: 4`, the products' `cells_param_size`).
Books tiled from the shipped model points: KLV_DE_S 60,000, VA_US_S 20,000. Collect output
`{"sum": {"net_cf": net_cf_total()}}`, `net_cf_total` a loop over `net_cf(t)`. Wall time of the book total
after the inputs are read, peak RSS of the process; fresh process per run; 5 rounds interleaved; medians.
4 vCPU, 3.14.6t, Cython 3.3.1. Direct: the same items through `Projection[k]` and `del`, a hand-written loop
(4 workers: four interleaved slices, partial sums added at the end).

| model | form | 1 worker s | 1 worker MB | 4 workers s | 4 workers MB | scaling |
|---|---|---|---|---|---|---|
| KLV_DE_S | old collect | 2.521 | 128.2 | 0.784 | 131.6 | 3.22x |
| KLV_DE_S | thin, as committed | 2.759 (+9.4%) | 127.6 | 1.081 (+37.9%) | 158.3 (+20%) | 2.55x |
| KLV_DE_S | thin without the root fast path | 2.978 | 127.7 | 1.107 | 148.6 | 2.69x |
| KLV_DE_S | direct | 2.589 | 126.8 | 0.999 | 152.8 | 2.59x |
| VA_US_S | old collect | 7.880 | 114.1 | 2.312 | 124.2 | 3.41x |
| VA_US_S | thin, as committed | 7.698 (-2.3%) | 113.6 | 2.655 (+14.8%) | 178.7 (+44%) | 2.90x |
| VA_US_S | thin without the root fast path | 8.271 | 113.6 | 2.861 | 188.8 | 2.89x |
| VA_US_S | direct | 8.224 | 114.1 | 2.758 | 173.0 | 2.98x |

Round-to-round spread on this machine is 5-15% at four workers; the medians are stable in sign.

The thin form does not meet the guard: KLV at one worker (+9.4%), both models at four workers (+38% / +15%) and
peak memory at four workers (+20% / +44%). Scaling stays above 2.5x. Every form that allocates an item per policy
(thin, direct) shows the same four-worker excess: a VA item is a 1.6 MB object, and the free-threaded allocator
keeps freed item pages per thread (with `MIMALLOC_PURGE_DELAY=0` the RSS falls to 147 MB and the time doubles).
No machinery was added back.

## What each piece of the old implementation would recover (for the owner's decision)

| piece | lines | recovered |
|---|---|---|
| root fast path for new items (reuses `item_roots`; kept) | 9 | measured: KLV 7.4% (1 w) / 2.3% (4 w), VA 6.9% / 7.2% |
| reused receiver per worker, reset between policies (old `serial_collect` reuse and parallel reuse: reset and restart lines, hold resets, isolation checks) | about 250 in the old code | not built in thin form; upper bound = the rest of the gap: KLV 8.6% (1 w) / 27% (4 w), VA 0% / 13%, and the four-worker memory (one item per worker, no allocation per policy) |
| native C sums of scalar outputs (old `Leaf`, packed sums) | about 300 in the old code | measured: `prepare_output` + `Accumulator.add` cost 1.35 us per item, at most 3% (KLV), 0.3% (VA) |
| contiguous chunks with per-chunk partial sums instead of a shared index and in-order adds | about 60 in old `_mx_sys` | estimated under 1% (two uncontended lock rounds per item); changes four-worker sums in the last bits |

Out, as asked: borrowed references and a persistent thread pool. Thread start costs about 0.2 ms per collect call
here, under 0.1% of these books; a pool pays only for many small collects. Borrowed references are part of the
general `r4/integration` compiler and present in both like-for-like builds, so their worth is not separable here.
