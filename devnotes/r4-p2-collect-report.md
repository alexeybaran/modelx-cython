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

## Second round: rebase, reused receiver, SEC models

Branch `r4/p2-collect-2` = the three commits above rebased on `r4/p1-fast-sec` 26129ab, plus one commit: collect
reuses one item per worker thread (non-test +114 / -11; about 74 lines without docstrings).

How the reset is proven complete: it is generated in the class's .pxd from the class's own declaration block and
the declarations of its bases in the shipped `_mx_sys.pxd`, the list that defines the object's layout. An object or
memoryview attribute becomes None, as `tp_new` leaves it; a C value, array or struct becomes zero bytes, as the
allocator leaves it; the dense value arrays stay unwritten behind their flags, as the S1 allocator already leaves a
new object's. A declaration of any other kind stops the build, naming the attribute and its Cell. The item is then
built again on the same memory, as `__call__` builds a root (`__init__`, References, roots). An item is kept only
when nothing else holds it (count 1 after its self-references are released), and never for a Space with dynamic
References. Tests: consecutive policies differ in every cache kind, one Cell is reached by one policy only and one by
every other policy (a reset that skips the flag arrays fails it); peak RSS of 3,000 items minus 20 items under 30 MB
at one and four workers (keeping the items fails it: +57 MB).

Like for like, medians of 5 interleaved rounds (s / peak MB), same set-up as above:

| model | workers | old collect | thin | thin + receiver | direct |
|---|---|---|---|---|---|
| KLV_DE_S | 1 | 2.507 / 128.8 | 2.789 / 128.2 | 2.678 / 128.5 | 2.830 / 127.4 |
| KLV_DE_S | 4 | 0.830 / 132.8 | 1.047 / 159.2 | 0.944 / 132.6 | 0.931 / 168.5 |
| VA_US_S | 1 | 7.767 / 114.6 | 8.180 / 114.3 | 8.216 / 114.6 | 7.899 / 114.8 |
| VA_US_S | 4 | 2.107 / 125.1 | 2.603 / 183.2 | 2.344 / 125.2 | 2.629 / 181.5 |

The receiver brings four-worker peak memory to the old one's and halves the four-worker time gap (KLV +14%, VA
+11%); at one worker it stays 6-7% behind on both. What remains is the item's `__init__`, run again for every
policy, which the old reuse skips.

The old path that kept items: `serial_collect: "reduce"` (fresh receivers, without `parallel_collect_reuse`)
peaked at 3,846 MB at one worker on KLV_DE_S (20,000 policies) and 131 MB at four; `reuse` with parallel reuse,
measured above, keeps one receiver.

### SEC models on this branch's own compiler

Term_US_SEC and MYGA_US_SEC take a collect declaration as they are (`lambda idx: None`, domain
`book = tuple(range(len(data.model_point_table())))`, outputs `{"sum": {"net_cf": net_cf_total()}}`, Book.total
the book-total cell), in a measurement script (`model_point_table_sec.csv` tiled; nothing in lifelib changed);
nothing in the models prevents it. Books sized so that the direct route keeping every item fits memory. Medians of
3 rounds (s / peak MB); collect values equal live bit for bit at both worker counts (Term 4668178.785059491,
MYGA -162913609.4115729); the four-worker direct sums differ from serial in the last bits.

| model | points | workers | direct, items kept | thin collect | thin + receiver |
|---|---|---|---|---|---|
| Term_US_SEC | 12,000 | 1 | 2.433 / 3684 | 0.897 / 96.0 | 0.935 / 95.7 |
| Term_US_SEC | 12,000 | 4 | 0.860 / 3686 | 0.316 / 113.6 | 0.298 / 102.3 |
| MYGA_US_SEC | 3,000 | 1 | 2.017 / 3274 | 0.604 / 98.0 | 0.624 / 97.7 |
| MYGA_US_SEC | 3,000 | 4 | 0.685 / 3276 | 0.288 / 125.8 | 0.223 / 105.2 |

A kept thread pool: not built (another measurement found no gain on DV01 and KLV).
