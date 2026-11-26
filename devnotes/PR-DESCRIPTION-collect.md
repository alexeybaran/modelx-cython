# Compiled collect

modelx declares collect on a Space (`Space.set_collect(over={param: domain}, outputs=...)`): `collect()` evaluates
the Space at every coordinate of a domain, reads each ItemSpace's `collect_outputs`, adds the `sum` outputs and keeps
the `individual` ones. An exported model calls one hook of the compiled Space class, `_mx_collect_hook` (version 1),
with the checked domain. This change makes the compiled class define that hook, so that a book is projected and
totalled in compiled code, on several threads on free-threaded Python, without the model's author writing a driver.

## Depends on

The modelx change that exports collect with the hook call must be merged first: https://github.com/alexeybaran/modelx,
branch `pr/collect` (1aa21f4), seven commits on upstream main: two for dynamic References, five for collect. Without it
a model exports no collect and the compiled class gains nothing. CI here installs upstream modelx, so it skips
`tests/compiled/test_collect.py` as a whole (its four tests: values, workers key, reused items, reset rule); the
borrowed-read tests run. The collect tests run in CI only on a branch that installs the modelx change; run
45831454650 on c802f9b is pending.

This change goes on top of the fast SEC change (`pr/fast-sec`, including its native math-module calls).

## Commits

In order, on `pr/fast-sec` (f86f38d):

| # | commit | subject | non-test lines | test lines |
|---|---|---|---|---|
| 1 | 757eb98 | Compile a Space declared with collect: the cache slot and the collect hook | +249 / -4 | +149 |
| 2 | 71a87ed | Reuse one item per worker thread in collect | +114 / -11 | +82 |
| 3 | bc216a4 | Borrow the held item of an exclusively owned item at a read | +17 / -5 | +146 / -3 |
| 4 | beda0ee | Refuse an unlocked Space Reference or a parent read at several workers on every build | +14 / -12 | |
| 5 | c802f9b | Count the live items and objects after collect instead of resident memory | | +21 / -10 |

Total (f86f38d..c802f9b): 10 files, +759 / -12; net non-test +371 / -9, test +388 / -3.

Commit 1 adds `collect.py`, its wiring in `config.py` (spec key `collect_workers`), `parser.py` and
`transformer.py`, and its documentation (`spec.md`: `collect_workers`; `freethreading.md`: the run-time setting
`_mx_collect_workers` and the refusal over unlocked Spaces). It keeps the export's `collect` property, which a
decorated method of the export is, as a property and not a Cell, so an ItemSpace has no `collect`, as in live modelx;
its test checks this; commit 2 changes `collect.py` and `transformer.py`; commit 3 changes
`space_results.py` and, as its message states, also changes the direct route. Commit 4 moves the refusals of
`collect.py` ahead of its one-thread path, so that an unlocked Space Reference or a parent read at several workers
raises on every interpreter build, with or without the GIL. Commit 5 replaces the resident-memory bound of the
reused-items test, whose unit differs by platform and whose `resource` module Windows lacks, by a direct count of the
live items and of the objects the garbage collector tracks after collect of 20 and of 3,000 coordinates.

**Declaration and evaluator.** The compiled class declares the collect cache and defines the hook from the exported
`collect` and `_mx_collect_item` methods. For each coordinate the hook builds the item as `__call__` builds a new
root, reads `collect_outputs()`, checks and detaches it with the export's own `prepare_output`, and releases the item
when collect created it; an existing ItemSpace is used and kept. The export's `Accumulator` adds the outputs in domain
order. Workers: the spec key `collect_workers` (default 1) or `_mx_collect_workers` on the compiled model object.
With more than one, items run on that many threads from a shared index; outputs are added in domain order as they
complete. With the GIL, several workers compute serially on one thread, with the same values and errors.

**Reused receiver.** A new item that nothing else holds is kept by its worker thread and returned to the state of a
new object before the next coordinate, then built again on the same memory. The reset is generated in the class's
`.pxd` from the class's own attribute declarations and those of its bases (the object's layout): an object or
memoryview attribute becomes None, as `tp_new` leaves it; a C value, array or struct becomes zero bytes, as the
allocator leaves it; dense value arrays stay unwritten behind their flags, as the allocator already leaves them. A
declaration of any other kind stops the build, naming the attribute and its Cell. Spaces with dynamic References
keep fresh items.

**Borrowed read.** The pinned read of a held item returns the pointer without a new reference, so an item of a shared
Space held by every worker's items is read without writing its reference count. It changes every compiled read of a
held item: the direct route (items built by `Projection[k]`) as well as collect.

## Guarantees

- Values are the export's own, added in domain order: equal to live modelx bit for bit at any number of workers.
- An error raises the exception of the first failing coordinate in domain order, at any number of workers.
- An ItemSpace has no `collect`, as in live modelx: reading it raises `AttributeError` and `hasattr` is False, in
  live modelx, the export and the compiled model.
- Items collect creates are released: the live items and tracked objects after collect do not grow with the number
  of items, at one worker or four.
- A reused item holds nothing from the previous coordinate (tested on every cache kind, on a Cell reached by one
  coordinate only and on one reached by every other coordinate; a reset that skips the flag arrays fails the test).
- Python 3.10 and later, with and without the GIL; the full upstream suite passes unchanged.

## Limits

- More than one worker requires every Space the items read to be locked. A Reference of the collect Space to a Space
  without a lock, or a formula of it reading `_model`, `_parent` or `_space`, raises `ValueError` naming it before
  any thread starts, on every interpreter build. A model satisfies it by exporting those Spaces in `locked_spaces`; one worker always runs.
  Locked Cells read their computed values without the lock; a miss takes the model lock, so concurrent first fills
  of a locked Space run one at a time.
- Collect releases only the items of the collect Space. ItemSpaces of other Spaces that its items create stay in
  those Spaces until the model deletes them.
- Sums are added in domain order on one thread; collect does not add in C or merge partial sums.
- Threads are started per call and joined before it returns.

## Considered and declined

| design | why declined |
|---|---|
| allow an unlocked Space reached only through its items, keyed by the collected parameter | safe within one call, but two collects started by user threads at once can share an item; not provable without serialising them, and locking the Space costs nothing measurable on the models below |
| skip re-running `__init__` on a reused item (reset only the caches) | measured: 0-3% on both models, within noise |
| native C sums of scalar outputs | the Python path costs 1.35 us per item: at most 3% of the fastest model, 0.3% of the others |
| contiguous chunks with partial sums | under 1%, and four-worker sums would differ from one worker's in the last bits |
| a thread pool kept per model | no measurable change on the DV01 example and KLV_DE_S |

## Evidence

All measurements: free-threaded CPython 3.14.6, Cython 3.3.1, 4 vCPU, fresh process per run, runs interleaved.

### The twelve `*_US_SEC` lifelib models (medians of 3, wall s / peak MB)

Collect declared in a measurement script, `Data` locked, books tiled to the size at which the direct route keeping
every item fits 4 GB. On the shipped book collect equals live modelx bit for bit on all twelve, at 1 and 4 workers,
with and without the borrowed read.

| model | points | (a) direct kept 1w | (a) direct kept 4w | (c) collect 1w | (c) collect 4w | scaling direct / collect | collect without borrowed read 1w / 4w (scaling) |
|---|---|---|---|---|---|---|---|
| DIA_US_SEC | 2229 | 1.67/1747 | 0.79/1750 | 0.87/100 | 0.51/108 | 2.13 / 1.73 | 0.87 / 0.53 (1.65x) |
| FIA_US_SEC | 1156 | 1.05/1802 | 0.40/1804 | 0.28/99 | 0.12/108 | 2.64 / 2.30 | 0.29 / 0.13 (2.33x) |
| IUL_US_SEC | 792 | 1.45/1946 | 0.54/1949 | 0.70/101 | 0.27/114 | 2.71 / 2.65 | 0.74 / 0.26 (2.88x) |
| MYGA_US_SEC | 1552 | 1.18/1740 | 0.40/1742 | 0.33/97 | 0.14/104 | 2.95 / 2.27 | 0.34 / 0.14 (2.44x) |
| RILA_US_SEC | 863 | 1.34/1396 | 2.13/1398 | 0.72/101 | 2.05/111 | 0.63 / 0.35 | 0.73 / 2.01 (0.37x) |
| SPIA_US_SEC | 3299 | 1.52/1824 | 0.46/1826 | 0.65/96 | 0.23/103 | 3.32 / 2.81 | 0.77 / 0.23 (3.35x) |
| Term_US_SEC | 8117 | 1.70/2522 | 0.62/2523 | 0.65/94 | 0.23/100 | 2.76 / 2.79 | 0.61 / 0.22 (2.75x) |
| ULSG_US_SEC | 1449 | 1.20/1815 | 0.40/1818 | 0.42/95 | 0.17/104 | 3.00 / 2.47 | 0.50 / 0.18 (2.75x) |
| UL_US_SEC | 2046 | 1.37/2083 | 0.47/2085 | 0.48/95 | 0.19/103 | 2.90 / 2.53 | 0.45 / 0.19 (2.43x) |
| VA_US_SEC | 954 | 1.39/2020 | 0.95/2022 | 0.49/104 | 0.16/117 | 1.46 / 3.01 | 0.43 / 0.17 (2.57x) |
| VUL_US_SEC | 817 | 2.33/2322 | 2.49/2339 | 1.53/101 | 0.49/116 | 0.94 / 3.10 | 1.58 / 0.50 (3.14x) |
| WholeLife_US_SEC | 3178 | 1.56/2363 | 0.59/2364 | 0.54/95 | 0.22/102 | 2.62 / 2.39 | 0.53 / 0.24 (2.25x) |

(b) a per-policy loop that deletes each item, on three models (1 worker, 4 workers, scaling): MYGA_US_SEC 0.34 / 98
and 0.15 / 122 (2.24x), Term_US_SEC 0.57 / 96 and 0.24 / 148 (2.38x), VA_US_SEC 0.46 / 105 and 0.21 / 163 (2.20x);
at one worker it equals collect.

Direct without the borrowed read at four workers: MYGA 0.40 s, Term 0.60 s, VA 0.97 s. Collect uses about 100 MB
where the direct route keeping its items uses 1.4-2.5 GB; a loop deleting each item uses 100-160 MB. Four-worker runs take 0.1-0.5 s, so their scaling includes
thread start. RILA_US_SEC is slower on four workers than on one on both routes, for a reason outside collect.

### The DV01 example on MYGA_US_SEC (medians of 5, the report Cells of 3; wall s / CPU s / peak MB)

| route | policies | 1 worker | 4 workers | per policy, 1 worker | scaling |
|---|---|---|---|---|---|
| (a) the report Cells, every item kept | 140 | 3.79 / 3.8 / 5631 | - | 27 ms, 40.2 MB | - |
| (b) pooled driver, deletes each policy's items | 1,050 | 9.97 / 9.9 / 252 | 3.80 / 13.4 / 352 | 9.5 ms, 0.24 MB | 2.62x |
| (c) collect | 1,050 | 9.81 / 9.7 / 1120 | 3.75 / 12.5 / 1099 | 9.3 ms, 1.07 MB | 2.62x |
| (c) collect without the borrowed read | 1,050 | 10.31 / 10.2 / 1120 | 7.79 / 28.6 / 1099 | | 1.32x |

On the same 140-policy book (medians of 3): (b) 1.48 s / 177 MB and 0.88 s / 237 MB, (c) 1.46 s / 325 MB and
0.98 s / 299 MB. (a) runs on one thread (the Cells sum policy by policy) and keeps all 40,950 Valuation
ItemSpaces with the book's `Actual` ItemSpaces, so its book is sized to this 16 GB machine. (c) sits between (a) and
(b): it releases each Valuation ItemSpace once its outputs are added, but the `Actual` ItemSpaces belong to another
Space, every close reads them, and they stay alive across the 39 collects; (b) deletes each policy's as soon as that
policy is done. Totals equal in every run (1,050: DV01 129960.61614055187; 140: 17328.0821516274); on 70 policies
every route equals live modelx bit for bit.

The declaration adds five lines to the model, and the report through collect is 21 lines instead of a per-policy
task, a thread pool and its own sums.

### Against the previous collect implementation, on the same item code (medians of 5, wall s / peak MB)

The previous implementation (a reused receiver per worker, native sums, contiguous chunks with partial sums) and this
one were built by the same compiler on lifelib's KLV_DE_S (60,000 policies) and VA_US_S (20,000 policies), collect
output the book's net cash flow. Direct: the same items through `Projection[k]` and `del`, four interleaved slices.

| model | workers | previous | this, without the reused receiver | this | direct |
|---|---|---|---|---|---|
| KLV_DE_S | 1 | 2.507 / 128.8 | 2.789 / 128.2 | 2.678 / 128.5 | 2.830 / 127.4 |
| KLV_DE_S | 4 | 0.830 / 132.8 | 1.047 / 159.2 | 0.944 / 132.6 | 0.931 / 168.5 |
| VA_US_S | 1 | 7.767 / 114.6 | 8.180 / 114.3 | 8.216 / 114.6 | 7.899 / 114.8 |
| VA_US_S | 4 | 2.107 / 125.1 | 2.603 / 183.2 | 2.344 / 125.2 | 2.629 / 181.5 |

The previous implementation is 6-7% faster at one worker and 11-14% at four, by running its per-item path natively;
this one has the previous one's peak memory, a fraction of its code, and four-worker sums equal to one worker's bit
for bit (the previous one's differ in the last bits).
