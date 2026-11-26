# Collect re-measured on 58ff45d (r4/p2-collect-7)

Compiler 58ff45d, modelx pr/collect 1aa21f4, lifelib pr/sec-dv01-collect 6a456523 (Data and Actual locked), the
same 1,050-policy book as before (the seven new-business points repeated), free-threaded CPython 3.14.6, 4 vCPU.
Fresh process per run; the runner's own compiled child; CPU and peak RSS of that child.

## DV01, 1,050 policies: 5 alternating rounds, median (range)

| route | workers | wall s | CPU s | peak MB | scaling |
|---|---|---|---|---|---|
| direct (pooled driver) | 1 | 16.27 (14.97-18.05) | 17.3 | 253 | |
| direct (pooled driver) | 4 | 5.09 (4.41-5.23) | 19.8 | 353 | 3.20x |
| collect | 1 | 12.85 (11.82-13.37) | 13.9 | 1121 | |
| collect | 4 | 4.89 (4.74-5.16) | 18.0 | 1100 | 2.63x |

DV01 129960.61614055187 and CS01 157856.97087213397 in every run, as before (bit for bit).

The machine is slower today than at the earlier measurement: the earlier build (`r4/p2-collect-3`, same book), re-run
now, takes 5.16 s for collect at four workers where it took 3.75 s. Interleaved on the same machine, 3 rounds, median:

| route | workers | earlier build | 58ff45d | change |
|---|---|---|---|---|
| direct | 1 | 15.31 | 14.34 | -6% |
| direct | 4 | 5.05 | 4.48 | -11% |
| collect | 1 | 14.41 | 12.27 | -15% |
| collect | 4 | 5.16 | 4.86 | -6% |

Peak memory is unchanged: collect 1.10-1.12 GB, direct 0.25 / 0.34-0.35 GB. Nothing is worse than before.

## RILA_US_SEC and DIA_US_SEC collect, 3 rounds interleaved with the earlier build, median

| model | points | build | 1 worker s | 4 workers s (range) | scaling |
|---|---|---|---|---|---|
| RILA_US_SEC | 863 | earlier | 1.081 | 1.313 (1.302-1.362) | 0.82x |
| RILA_US_SEC | 863 | 58ff45d | 0.441 | 0.528 (0.494-0.535) | 0.84x |
| DIA_US_SEC | 2229 | earlier | 1.542 | 0.712 (0.680-0.766) | 2.17x |
| DIA_US_SEC | 2229 | 58ff45d | 1.159 | 0.706 (0.597-0.883) | 1.64x |

Values equal between builds (RILA -38020248.73820954, DIA -652661961.9109885). RILA runs 2.5x faster on one worker
with native math, but four workers are still slower than one; the direct route behaves the same way, so the cause is
outside collect and is not explained here. DIA's four-worker time is unchanged, so its scaling falls as its one-worker
time falls.

## Why RILA_US_SEC does not scale, and DIA_US_SEC stops at 1.6x

Same builds, books and machine as above (RILA 863 points, DIA 2229); wall s, fresh process per run, interleaved.

**RILA: one shared reference count, written by every native math call.** The native-math lowering compiles
`self.math.exp(x)` to `_mx_sys._mx_math_exp(self.math, x, path)`, the helper taking the module as `object`. Cython
takes a new reference to `self.math` around every call. On free-threaded Python the `math` module belongs to the main
thread, so each worker's increment and decrement is an atomic write to the same word of the module object, shared by
every item and every thread. RILA's option formula calls exp, log, sqrt and erf about 6,800 times per policy. The model
text decides how often this happens; the compiler output is what writes the count. modelx is not involved.

The evidence that singles it out:

| observation | 1 worker | 4 workers |
|---|---|---|
| instructions (callgrind) | 4.62 G | 4.75 G (+3%) |
| user CPU s for the whole process | 1.37-1.53 | 2.68-2.82 |
| `_Py_DecRefShared` calls (callgrind) | negligible | 5.88 M = 6,800 per policy, all from inlined `Py_DECREF` of `self->math` in `_mx_dense_formula_86` |
| four separate 1-worker processes at once | 0.38-0.50 each | (they scale) |

So the work is the same, CPU per instruction doubles, and the stall happens only between threads of one process. A
copy of the model per thread does not help (1w 0.493 / 4w 0.488), because the module is shared all the same. Ruled out:
book size (8,630 points: 1w 3.78 / 4w 5.04), locks (lock functions 148 K instructions in total, futex 171 calls),
OpenBLAS threads (OPENBLAS_NUM_THREADS=1: 0.465 / 0.529), allocator trimming and mmap (glibc tunables: 1w faster,
4w unchanged), mimalloc purging, the GC, page faults (fewer at 4w), memory bandwidth (separate processes scale).

Confirmation: the same generated code rebuilt with the helpers taking a borrowed pointer (`void* module`, compared
with the standard module; `(<object>module).exp(x)` only on the fallback; call sites
`_mx_math_exp(_mx_cy.cast(_mx_cy.p_void, self.math), x, path)`), three interleaved runs each:

| build | 1 worker | 4 workers | scaling |
|---|---|---|---|
| 58ff45d | 0.405-0.414 | 0.527-0.535 | 0.77x |
| borrowed module pointer | 0.389-0.540 | 0.131-0.154 | 2.8x |

The values are identical (-38020248.73820954). The fix is in the compiler: about 12 lines, in `_mx_sys.pxd` (five
helpers) and one argument in `native_math.py`. It is not committed. It needs the suite run, and a decision that a
Reference rebound during a computation is out of scope, as it already is for the borrowed read.

**DIA: not the same mechanism, and not singled out.** DIA makes no native math calls, and `_Py_DecRefShared` is
0.1% of its instructions at four workers. Its profile is dominated by object arithmetic in the formula behind
`_mx_dense_formula_73` (`a += (1.0 / m) * _mx_cy.cast(object, v) ** _mx_cy.cast(object, (s - tk) / 12.0) *
max(cert, lf)`). Each policy makes 8,500 boxed floats and 1,200 `PyNumber_Power` calls, and the dealloc path
(`_Py_MergeZeroLocalRefcount`) takes 12% of instructions. The casts come from the compiler, which keeps Python's
power semantics where the sample could not exclude a negative base. These are allocations local to each thread.
Whether they limit scaling (through allocator or cache pressure) is not shown. The next experiment would be the same
loop with C `pow` behind a sign check.

## Collect's fixed cost per policy against the earlier tree's own collect

Index_DE_S and UC_FR_S at lifelib 04d69db. Collect is declared over `point_id` with `{"sum": {"net_cf": net_cf_total()}}`,
`Data` is locked, and the build uses the shipped book. The run is at one worker on the book tiled to 20 and 1,020
policies. Instructions per policy are the callgrind difference divided by 1,000; runs vary by about ±5k, mostly
OpenBLAS thread spin. Proposal (A): compiler c342db9 with modelx 1aa21f4. Earlier tree (B): 0b77986 with modelx
636877b and spec `serial_collect: reuse`. Values are equal between A and B on both books.

| model | A | B | A - B |
|---|---|---|---|
| Index_DE_S | 556,703 | 514,066 | +42.6k |
| UC_FR_S | 565,939 | 519,068 | +46.9k |

The item's own computation is the same (Index 437.9k against 438.6k, UC 360.5k against 360.0k). The difference is
the per-coordinate overhead, measured as the inclusive cost of the collect loop minus the item's outputs Cell:

| per coordinate | Index A | Index B | UC A | UC B |
|---|---|---|---|---|
| export's `prepare_output` + `Accumulator.add`, interpreted (B sums in C) | 17.9k | - | 17.9k | - |
| reset of the reused item: memset of flag arrays | 4.7k | 4.0k | 37.8k | 17.6k |
| `_mx_copy_refs`: every Reference re-assigned (B captures them at entry) | 7.3k | - | 8.1k | - |
| `_mx_itemspaces.get`, `_mx_roots.append`, `threading.local` get/set, `__init__`, release | 8.6k | 0.7k | 9.2k | 13.7k |
| total overhead | 38.5k | 4.7k | 73.0k | 31.3k |

UC's reset clears the `short[1226]` flags of every Cell, 211 KB per item. B clears only the Cells that
`collect_outputs` reaches (`serial_collect._reset_lines(only=...)`). A Cython-compiled `_mx_collect.py` lowers the
evaluator row from 17.9k to 13.2k (Index), because the rest is dict, set and tuple building.

Changes inside the proposal, not made:
1. A per-worker slot passed to the item function instead of `threading.local`: about 10 lines, -1 to 3k, clearly
   right.
2. Keep References across reuse: leave them out of the reset and drop `_mx_copy_refs` on a reused item. About 8
   lines, -7 to 8k. The contract becomes "References as at collect entry", as in B.
3. Reset only the Cells reachable from `collect_outputs`, with a full reset when the reach is not static. About 15
   lines, -20k on UC. It needs the compiler's call graph.
4. A compiled fast path for `sum` outputs that are exact floats: added in domain order as `total = total + value`,
   so bit-identical to the Accumulator. About 30 lines, -15 to 18k. This is the native sum declined earlier at 1.35 us
   per item.
