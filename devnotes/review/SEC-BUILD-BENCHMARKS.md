# Released SEC benchmarks with lifelib's documented native build settings

This run compares upstream modelx-cython `d22426fbac8637f84510902c2fa33b6466e5acb4` with Fast SEC `eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df`. Both use modelx `fec516d7e9b5813e7d40819a72731bd35fc2972f` and lifelib `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4` for all twelve released SEC products.

The earlier generic harness omitted the documented SEC module directives. Cython rejected real-valued model expressions in DIA and VUL because it inferred complex powers. The recheck uses the lifelib build settings for both compilers. [DIA recipe](https://github.com/lifelib-dev/lifelib/blob/20d4daef3281d538b3fbda6d0fb951fcceb3a6f4/lifelib/libraries/uslib/products/deferred_income_annuity/model-sec.md#L1241) · [VUL recipe](https://github.com/lifelib-dev/lifelib/blob/20d4daef3281d538b3fbda6d0fb951fcceb3a6f4/lifelib/libraries/uslib/products/variable_ul/model-sec.md#L977)

## Procedure

1. Export the released product with its shipped spec and sample. The benchmark adds one Cell that totals monthly net cash flows, and a sample wrapper that runs the shipped sample and calls that Cell for the shipped points.
2. After translation, put `# cython: infer_types=True, cpow=True` at the start of the generated `_mx_classes.py` for each arm. Retain Python's original C flags and append `-ffp-contract=off`. Confirm that GCC compiles the classes with `-O3` and the contraction flag.
3. Build the native packages on the same WSL2 Ryzen 7 5800H host, using free-threaded CPython 3.14.8, Cython 3.3.2 and GCC 13.3.0. DIA and VUL were translated again. The other ten reuse the exact earlier translations, then rebuild their native packages with the documented settings. Their translation times describe that earlier translation stage.
4. Use the same ordered book at both worker counts and check the GIL is off in every process. Delete each Direct policy ItemSpace after evaluating it at both worker counts. At one worker the Direct driver collects garbage every 512 projections on each compiler. At four workers a rolling queue keeps 64-projection tasks available while the caller runs generation-zero garbage collection after roughly 512 completed projections. The driver uses the same queue and cleanup on both compilers.
5. Run five Direct timing rounds in fresh processes per compiler and worker count. Alternate the compiler order within each model. Pin one-worker jobs to separate physical cores, with up to four independent jobs at a time. Run four-worker measurements one process at a time on four physical cores. Use two separate one-worker instruction rounds per compiler with the same cleanup schedule. Validate each policy value and the aggregate against live modelx in every run. Wall time and peak RSS columns show the median and minimum–maximum range.

For `BasicTerm_SC`, use lifelib's shipped basiclife sample and spec. The sample also evaluates the first and last model points to cover the whole book. The workload evaluates `pv_net_cf()` on the shipped 10,000-point book, repeating complete passes to get a useful timing interval. Each projection is recalculated after releasing the prior state. [Lifelib BasicTerm_SC documentation](https://github.com/lifelib-dev/lifelib/blob/20d4daef3281d538b3fbda6d0fb951fcceb3a6f4/doc/source/libraries/basiclife/BasicTerm_SC.rst)

The historical generic comparison uses identical books and the earlier one-worker cleanup schedule on both sides. Instruction counts provide the more repeatable measure of the setting change across those two sessions. [Earlier generic run and separate SEC DV01 report](REFRESH.md)

## Matched upstream comparison

All 26 builds succeeded. The new four-worker receipts and the reused one-worker timings and instruction counts passed live-value checks. The one-worker cleanup and ordered books are the same in both scheduling protocols. The largest measured RSS among the rolling four-worker trials was 5,023.6 MiB. Each scaling factor divides the one-worker median by its four-worker median. The scaling ratio divides Fast SEC’s factor by upstream’s. MYGA DV01 uses its dedicated full-report driver and its own pinned model as described in [the Fast SEC description](COMPILER-FAST-SEC.md#evidence).

| Model | Projections | Upstream 1w µs/proj. | Fast SEC 1w µs/proj. | Upstream 4w µs/proj. | Fast SEC 4w µs/proj. | Instruction reduction (upstream / Fast SEC) | 1w elapsed speedup | 4w elapsed speedup | Upstream 1w/4w | Fast SEC 1w/4w | Scaling ratio (Fast SEC / upstream) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BasicTerm_SC | 100,000 | 89.9 | 33.9 | 38.6 | 19.8 | 2.59× | 2.65× | 1.95× | 2.33× | 1.71× | 0.73× |
| DIA | 12,000 | 1,198.9 | 394.2 | 437.0 | 245.3 | 2.28× | 3.04× | 1.78× | 2.74× | 1.61× | 0.59× |
| FIA | 10,900 | 1,662.2 | 288.2 | 505.0 | 174.7 | 3.34× | 5.77× | 2.89× | 3.29× | 1.65× | 0.50× |
| IUL | 3,500 | 5,302.7 | 945.8 | 1,417.7 | 412.8 | 3.28× | 5.61× | 3.43× | 3.74× | 2.29× | 0.61× |
| MYGA | 34,300 | 1,737.3 | 274.6 | 520.5 | 130.1 | 4.00× | 6.33× | 4.00× | 3.34× | 2.11× | 0.63× |
| RILA | 27,000 | 2,184.4 | 374.8 | 991.9 | 170.9 | 4.32× | 5.83× | 5.80× | 2.20× | 2.19× | 1.00× |
| SPIA | 13,200 | 992.4 | 226.1 | 275.3 | 86.9 | 2.37× | 4.39× | 3.17× | 3.61× | 2.60× | 0.72× |
| Term | 31,900 | 574.4 | 103.4 | 160.3 | 40.3 | 3.80× | 5.55× | 3.98× | 3.58× | 2.56× | 0.72× |
| ULSG | 9,000 | 1,934.0 | 333.3 | 560.9 | 167.0 | 3.40× | 5.80× | 3.36× | 3.45× | 2.00× | 0.58× |
| UL | 9,500 | 1,712.5 | 271.7 | 535.8 | 150.3 | 3.81× | 6.30× | 3.56× | 3.20× | 1.81× | 0.57× |
| VA | 8,000 | 3,210.1 | 427.4 | 1,031.7 | 276.1 | 4.03× | 7.51× | 3.74× | 3.11× | 1.55× | 0.50× |
| VUL | 5,300 | 4,840.6 | 776.0 | 1,500.7 | 340.3 | 3.00× | 6.24× | 4.41× | 3.23× | 2.28× | 0.71× |
| WholeLife | 13,900 | 1,102.6 | 173.1 | 321.6 | 99.0 | 3.61× | 6.37× | 3.25× | 3.43× | 1.75× | 0.51× |
| MYGA DV01 | 42,000 (1,050 × 40) | — | 349.1 | — | 84.1 | — | — | — | — | 4.15× | — |

## Detailed results by compiler

| Model | Compiler | 1w µs/projection | 4w µs/projection | Instructions/projection | Peak MiB at 1w | Translation + C build seconds |
| --- | --- | --- | --- | --- | --- | --- |
| BasicTerm_SC | Upstream | 89.9 [89.3–91.2] | 38.6 [37.1–40.0] | 724,295 | 137.7 [137.6–137.7] | 5.6 + 20.2 |
| BasicTerm_SC | Fast SEC | 33.9 [33.6–34.1] | 19.8 [18.4–20.9] | 279,236 | 112.4 [112.3–112.6] | 8.3 + 23.4 |
| DIA_US_SEC | Upstream | 1,198.9 [1,188.7–1,485.9] | 437.0 [417.6–478.1] | 6,253,734 | 967.2 [967.2–967.3] | 37.3 + 117.6 |
| DIA_US_SEC | Fast SEC | 394.2 [386.6–412.0] | 245.3 [242.3–267.9] | 2,747,358 | 122.1 [122.0–122.4] | 63.8 + 148.1 |
| FIA_US_SEC | Upstream | 1,662.2 [1,569.0–2,200.2] | 505.0 [499.2–621.5] | 7,838,924 | 1,768.5 [1,768.5–1,768.8] | 55.3 + 138.3 |
| FIA_US_SEC | Fast SEC | 288.2 [277.3–299.3] | 174.7 [171.7–178.4] | 2,347,003 | 109.1 [109.0–109.3] | 81.3 + 169.1 |
| IUL_US_SEC | Upstream | 5,302.7 [4,962.6–6,669.4] | 1,417.7 [1,378.1–1,519.3] | 28,326,344 | 2,545.0 [2,544.9–2,545.2] | 58.3 + 131.3 |
| IUL_US_SEC | Fast SEC | 945.8 [907.9–1,023.8] | 412.8 [400.1–421.9] | 8,646,047 | 107.4 [107.3–107.5] | 81.8 + 157.6 |
| MYGA_US_SEC | Upstream | 1,737.3 [1,682.7–1,785.0] | 520.5 [512.6–530.4] | 7,632,400 | 1,352.0 [1,351.7–1,352.1] | 31.5 + 105.5 |
| MYGA_US_SEC | Fast SEC | 274.6 [270.5–291.0] | 130.1 [128.3–132.7] | 1,907,245 | 126.1 [125.9–127.4] | 56.9 + 130.6 |
| RILA_US_SEC | Upstream | 2,184.4 [1,861.1–2,550.7] | 991.9 [908.6–1,054.7] | 10,528,153 | 3,240.5 [3,240.4–3,240.6] | 45.0 + 148.0 |
| RILA_US_SEC | Fast SEC | 374.8 [333.2–395.8] | 170.9 [168.6–174.1] | 2,437,989 | 119.3 [119.1–120.1] | 83.2 + 177.9 |
| SPIA_US_SEC | Upstream | 992.4 [970.1–998.3] | 275.3 [272.8–280.0] | 5,445,860 | 669.2 [669.1–669.3] | 36.6 + 73.7 |
| SPIA_US_SEC | Fast SEC | 226.1 [223.9–243.9] | 86.9 [86.2–89.0] | 2,296,397 | 106.1 [105.8–106.2] | 46.8 + 93.2 |
| Term_US_SEC | Upstream | 574.4 [573.2–609.5] | 160.3 [157.2–164.7] | 3,615,600 | 313.3 [313.1–313.3] | 11.3 + 43.0 |
| Term_US_SEC | Fast SEC | 103.4 [100.0–106.7] | 40.3 [39.1–43.1] | 950,274 | 104.1 [104.1–104.4] | 23.2 + 47.8 |
| ULSG_US_SEC | Upstream | 1,934.0 [1,886.6–2,434.3] | 560.9 [548.3–593.2] | 8,958,046 | 1,422.7 [1,422.7–1,422.8] | 29.4 + 93.9 |
| ULSG_US_SEC | Fast SEC | 333.3 [307.3–368.9] | 167.0 [165.3–167.8] | 2,633,860 | 103.1 [103.0–103.2] | 49.4 + 114.9 |
| UL_US_SEC | Upstream | 1,712.5 [1,708.0–1,778.1] | 535.8 [531.3–542.0] | 9,554,512 | 1,027.3 [1,027.3–1,027.4] | 26.6 + 82.2 |
| UL_US_SEC | Fast SEC | 271.7 [256.4–277.3] | 150.3 [149.0–154.6] | 2,507,849 | 102.2 [102.1–102.4] | 42.2 + 83.3 |
| VA_US_SEC | Upstream | 3,210.1 [3,029.0–3,564.6] | 1,031.7 [850.3–1,080.9] | 12,255,825 | 2,803.2 [2,803.1–2,803.3] | 86.4 + 184.9 |
| VA_US_SEC | Fast SEC | 427.4 [416.6–439.4] | 276.1 [268.2–299.3] | 3,042,807 | 113.6 [113.5–113.7] | 111.4 + 205.6 |
| VUL_US_SEC | Upstream | 4,840.6 [3,901.3–5,638.3] | 1,500.7 [1,338.8–2,071.0] | 14,054,401 | 3,919.5 [3,919.3–3,919.8] | 41.3 + 109.1 |
| VUL_US_SEC | Fast SEC | 776.0 [705.5–831.3] | 340.3 [332.1–372.5] | 4,680,194 | 107.4 [107.3–107.6] | 64.9 + 139.0 |
| WholeLife_US_SEC | Upstream | 1,102.6 [1,082.9–1,121.3] | 321.6 [318.0–325.6] | 6,319,282 | 691.3 [691.3–691.4] | 26.5 + 80.1 |
| WholeLife_US_SEC | Fast SEC | 173.1 [169.7–186.5] | 99.0 [93.4–107.2] | 1,749,570 | 103.8 [103.7–103.9] | 39.5 + 80.6 |

## Effect of the 512-projection batches on worker scaling

The timing driver divides each batch of up to 512 policy projections into tasks of up to 64. It waits for every task in the batch to finish before running generation-zero garbage collection and submitting the next batch. The 34,300-policy MYGA book has 67 batches. The common cleanup schedule bounds memory for the upstream builds. Earlier upstream four-worker runs without this cleanup reached about 15 GiB in the 16 GiB WSL guest.

The diagnostic below reused the Fast SEC binaries and ordered books from the paired comparison. It held task size fixed and varied the placement of the waits and explicit garbage collection. Automatic garbage collection was disabled during each diagnostic interval. The whole-book driver queued all tasks and collected once at the end. Each median comes from three fresh processes with alternate run order, on the same WSL2 machine, with one or four distinct physical cores pinned, the GIL verified off, and every value checked against live modelx.

| Model | Projections | Batched with cleanup, 1w s | Batched with cleanup, 4w s | Batched with end-only cleanup, 4w s | Whole book, 1w s | Whole book, 4w s | Batched scaling | Whole-book scaling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DIA | 12,000 | 4.286 | 3.618 | 3.266 | 4.085 | 2.923 | 1.18× | 1.40× |
| FIA | 10,900 | 2.803 | 2.307 | 2.242 | 2.708 | 1.537 | 1.21× | 1.76× |
| MYGA | 34,300 | 8.092 | 5.657 | 5.050 | 7.738 | 2.856 | 1.43× | 2.71× |
| RILA | 27,000 | 8.595 | 5.899 | 5.540 | 8.385 | 4.544 | 1.46× | 1.85× |

Removing the waits between batches improved four-worker time on all four models with the same end-only cleanup. For MYGA the waits accounted for 2.194 s of the 2.801 s recovered. Median process CPU time fell from 15.311 s to 11.076 s at that step. DIA still scaled only 1.40×, and FIA and RILA were below 2.5× in this matched diagnostic. The earlier driver collected explicitly every 64 projections at one worker and imposed no explicit cleanup at four workers. Its scaling figures therefore included the effect of different cleanup schedules. The matched upstream table above now uses the common rolling queue. The whole-book test was run on Fast SEC only.

## Whole-book Direct and Collect scaling

This focused comparison used the same ordered CSV books and cash-flow output as the main tables. It alternated all six configurations in five fresh processes per configuration, on WSL2 with a Ryzen 7 5800H and free-threaded CPython 3.14.8. The GIL stayed off. One-worker runs used Linux CPU 0 and four-worker runs used Linux CPUs 0, 2, 4 and 6 on four separate physical cores. The policy table was loaded before timing to confirm the book size. Each Direct output and each Collect total matched live modelx.

Direct used the whole-book queue, deleted each public policy item and explicitly collected garbage once at the end at both worker counts. Collect used one public call with its ordinary garbage-collection behavior and temporary policy items. PR1 used the same binaries as the previous section. PR2 used compiler `0cb1cb7f9dfeac4bff45df4ff0b96780f2cd725f` and modelx `f83a55a0ca3b7f15896184234f2b8fd06a1da215`. Both used released lifelib `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4`. The PR1 build had the documented explicit Cython directives and disabled floating-point contraction. PR2 Direct and Collect used the same default native build settings as each other. Three MYGA rounds that overlapped the FIA build were discarded and replaced.

| Model | Projections | Route | 1w wall s [min–max] | 4w wall s [min–max] | Scaling | Process CPU s, 1w → 4w | 4w peak MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| MYGA | 34,300 | PR1 Direct | 7.105 [7.078–7.464] | 2.391 [2.304–2.818] | 2.97× | 7.104 → 9.222 | 171.4 |
| MYGA | 34,300 | PR2 Direct | 6.967 [6.939–7.269] | 2.352 [2.189–2.580] | 2.96× | 6.972 → 9.044 | 167.3 |
| MYGA | 34,300 | PR2 Collect | 7.548 [7.271–7.965] | 2.344 [2.174–2.416] | 3.22× | 7.560 → 9.048 | 141.6 |
| RILA | 27,000 | PR1 Direct | 7.663 [7.443–7.676] | 3.079 [3.058–3.548] | 2.49× | 7.670 → 12.070 | 197.8 |
| RILA | 27,000 | PR2 Direct | 8.301 [7.930–8.339] | 3.513 [3.248–3.964] | 2.36× | 8.286 → 13.800 | 190.2 |
| RILA | 27,000 | PR2 Collect | 8.145 [7.793–8.164] | 2.400 [2.346–2.661] | 3.39× | 8.139 → 9.285 | 140.2 |
| FIA | 10,900 | PR1 Direct | 2.390 [2.359–2.437] | 0.980 [0.937–1.003] | 2.44× | 2.389 → 3.558 | 159.4 |
| FIA | 10,900 | PR2 Direct | 2.391 [2.374–2.425] | 0.957 [0.928–0.960] | 2.50× | 2.388 → 3.434 | 161.3 |
| FIA | 10,900 | PR2 Collect | 2.348 [2.338–2.354] | 0.786 [0.749–0.822] | 2.99× | 2.344 → 2.758 | 125.3 |
| DIA | 12,000 | PR1 Direct | 3.691 [3.663–3.711] | 2.440 [2.373–2.521] | 1.51× | 3.689 → 4.327 | 163.8 |
| DIA | 12,000 | PR2 Direct | 3.707 [3.690–3.752] | 2.429 [2.341–2.582] | 1.53× | 3.703 → 4.384 | 161.9 |
| DIA | 12,000 | PR2 Collect | 3.630 [3.572–3.685] | 2.415 [2.391–2.467] | 1.50× | 3.642 → 4.215 | 143.9 |

DIA spends about 1.53–1.55 seconds preparing shared premium data on the caller or one worker. A separate three-round control timed `Data.premium_array()`, `Data.premium_rows()` and `Data.premium_last()` before the projection stopwatch. They each group the same schedule by policy. The prepared projection took 2.167 / 0.753 seconds in PR2 Direct and 2.070 / 0.715 seconds in PR2 Collect, giving 2.88× and 2.89× scaling. Table preparation represents about 63% of the cold four-worker Collect time.

The time in RILA and FIA moves in a different way. In PR2 Direct, RILA averaged 3.93 effective cores at 4w and FIA 3.59, calculated as process CPU divided by elapsed time. Their process CPU rose 67% and 44% from one worker to four. For Collect, the increases were 14% and 18%. The four-worker observer saw all four workers with roughly even CPU time in both routes. Direct creates each public policy item and removes it from the shared ItemSpace cache. The native Collect path reuses a private item per worker when safe and accumulates cash flows natively. At 4w, RILA's median minor faults fell from 26,544 in Direct to 6,028 in Collect, and FIA fell from 16,373 to 4,329.

Separate hardware-counter runs measured one fresh process per configuration. The largest increase in retired instructions from 1w to 4w was 0.77% across these eight pairs. The following cycle changes measure the added execution cost per retired instruction.

| Model | PR2 Direct CPI change, 1w → 4w | PR2 Collect CPI change, 1w → 4w |
|---|---:|---:|
| MYGA | +13% | +14% |
| RILA | +52% | +19% |
| FIA | +42% | +11% |
| DIA | +10% | +16% |

The RILA control split the original book into four consecutive index ranges and pinned each fresh one-worker Direct process to its own core. Running the same processes concurrently raised total CPU from about 8.03 s to 9.54–9.64 s over two alternating rounds, with essentially identical retired instructions. In the single-process PR2 Direct diagnostic with four threads, CPU rose from 7.90 s to 12.63 s. Concurrent execution alone imposes a measurable cost, with an additional penalty in the shared-process Direct path. The exact shared object or allocation site remains unresolved. RILA's generated `norm_cdf` passed the math module as a borrowed pointer on all three builds.

For MYGA the four-worker medians were effectively equal. Its larger Collect scaling ratio comes from the higher measured one-worker median. In the later counter pair, one-worker results were 7.08 s for Collect and 7.11 s for Direct. An additional three-round check of one-worker Collect measured 7.05 s with the default garbage collection and 7.46 s with automatic collection disabled. The one-worker difference varied across measurement sessions.

## Fast SEC whole-book comparison

These are Fast SEC-only timings on the same books, using the full task queue and one final cleanup. The first table reports the memory-bounded upstream pair.

| Model | Projections | Fast SEC 1w µs/projection | Fast SEC 4w µs/projection | Scaling |
| --- | --- | --- | --- | --- |
| BasicTerm_SC | 100,000 | 25.6 [25.4–27.6] | 7.9 [7.7–8.2] | 3.22× |
| DIA | 12,000 | 307.6 [305.2–309.2] | 203.3 [197.7–210.1] | 1.51× |
| FIA | 10,900 | 219.2 [216.4–223.6] | 89.9 [86.0–92.1] | 2.44× |
| IUL | 3,500 | 971.3 [932.1–1,022.8] | 327.8 [320.1–330.8] | 2.96× |
| MYGA | 34,300 | 207.1 [206.4–217.6] | 69.7 [67.2–82.2] | 2.97× |
| RILA | 27,000 | 283.8 [275.7–284.3] | 114.0 [113.3–131.4] | 2.49× |
| SPIA | 13,200 | 215.2 [211.5–222.6] | 60.9 [59.1–63.0] | 3.54× |
| Term | 31,900 | 90.8 [89.2–92.4] | 25.8 [24.7–26.3] | 3.52× |
| ULSG | 9,000 | 329.7 [314.4–338.2] | 100.5 [95.2–112.6] | 3.28× |
| UL | 9,500 | 311.6 [282.6–318.9] | 91.4 [86.2–92.9] | 3.41× |
| VA | 8,000 | 377.4 [338.2–382.9] | 182.4 [176.4–194.6] | 2.07× |
| VUL | 5,300 | 660.8 [612.2–731.4] | 247.1 [229.6–256.0] | 2.67× |
| WholeLife | 13,900 | 183.4 [179.5–190.9] | 50.5 [48.5–54.6] | 3.63× |

## Compiled Collect comparison

Both compiled routes evaluate the same ordered book and return the same cash-flow output. Direct deletes each policy item inside the timed interval and queues the whole book in 64-projection tasks at four workers, with one explicit cleanup at the end. Collect uses a single public call per pass. All timings are medians of five alternating fresh-process rounds with the GIL off. Four workers run on separate physical cores.

Compiler `0cb1cb7f9dfeac4bff45df4ff0b96780f2cd725f` and modelx `f83a55a0ca3b7f15896184234f2b8fd06a1da215` use the released lifelib text for the ordinary SEC models and BasicTerm_SC. The four products in the focused comparison above reuse those exact timing receipts. Each other comparison uses Direct and Collect on the same Collect-enabled build. The full MYGA DV01 result reuses its dedicated model benchmark with unreduced workload, and appears in the [Collect description](COMPILER-COLLECT.md#compiled-direct-and-collect-timing).

| Model | Route | Projections | 1w µs/projection | 4w µs/projection | Scaling |
| --- | --- | --- | --- | --- | --- |
| BasicTerm_SC | Direct | 100,000 | 26.1 [24.7–28.6] | 7.9 [7.6–8.3] | 3.30× |
| BasicTerm_SC | Collect | 100,000 | 25.4 [24.8–27.6] | 7.2 [7.1–7.3] | 3.51× |
| DIA | Direct | 12,000 | 308.9 [307.5–312.7] | 202.4 [195.1–215.1] | 1.53× |
| DIA | Collect | 12,000 | 302.5 [297.7–307.1] | 201.3 [199.2–205.6] | 1.50× |
| FIA | Direct | 10,900 | 219.3 [217.8–222.5] | 87.8 [85.2–88.1] | 2.50× |
| FIA | Collect | 10,900 | 215.4 [214.5–216.0] | 72.1 [68.7–75.4] | 2.99× |
| IUL | Direct | 3,500 | 949.5 [911.8–976.5] | 353.1 [337.8–372.7] | 2.69× |
| IUL | Collect | 3,500 | 942.2 [923.6–947.0] | 254.0 [241.7–270.1] | 3.71× |
| MYGA | Direct | 34,300 | 203.1 [202.3–211.9] | 68.6 [63.8–75.2] | 2.96× |
| MYGA | Collect | 34,300 | 220.0 [212.0–232.2] | 68.3 [63.4–70.4] | 3.22× |
| RILA | Direct | 27,000 | 307.4 [293.7–308.8] | 130.1 [120.3–146.8] | 2.36× |
| RILA | Collect | 27,000 | 301.7 [288.6–302.4] | 88.9 [86.9–98.6] | 3.39× |
| SPIA | Direct | 13,200 | 211.7 [206.8–224.6] | 62.2 [58.1–63.7] | 3.40× |
| SPIA | Collect | 13,200 | 219.7 [205.6–235.2] | 60.2 [59.4–62.2] | 3.65× |
| Term | Direct | 31,900 | 92.9 [86.9–95.6] | 25.7 [25.3–27.1] | 3.62× |
| Term | Collect | 31,900 | 94.5 [91.5–98.0] | 24.9 [23.7–26.3] | 3.79× |
| ULSG | Direct | 9,000 | 311.8 [303.5–324.6] | 100.4 [93.9–101.5] | 3.10× |
| ULSG | Collect | 9,000 | 308.9 [294.3–325.2] | 85.4 [84.6–90.6] | 3.62× |
| UL | Direct | 9,500 | 305.8 [300.7–317.3] | 93.4 [89.0–99.5] | 3.28× |
| UL | Collect | 9,500 | 298.0 [287.6–310.9] | 89.3 [85.2–91.6] | 3.34× |
| VA | Direct | 8,000 | 367.0 [325.2–377.1] | 185.0 [175.8–223.2] | 1.98× |
| VA | Collect | 8,000 | 367.2 [349.7–395.5] | 104.6 [100.3–109.5] | 3.51× |
| VUL | Direct | 5,300 | 717.2 [674.4–722.5] | 292.2 [264.8–328.9] | 2.45× |
| VUL | Collect | 5,300 | 582.3 [530.1–596.8] | 169.5 [156.5–204.3] | 3.43× |
| WholeLife | Direct | 13,900 | 190.7 [187.4–198.0] | 56.4 [54.0–58.4] | 3.38× |
| WholeLife | Collect | 13,900 | 192.9 [178.9–197.8] | 52.7 [51.9–54.3] | 3.66× |

## Impact on the earlier generic measurements

The table below isolates the build-setting correction by using the same earlier one-worker cleanup schedule before and after the change. Upstream retired instructions dropped on every available earlier pair. SPIA changed most at 31.8%. The Fast SEC instruction differences were at most 0.1% except RILA, which fell 5.1%. WholeLife's elapsed change across those sessions was much larger than its instruction change. The fresh matched pair above is the basis for the current WholeLife speed ratio.

| Model | Compiler | Elapsed change vs generic | Instruction change vs generic |
| --- | --- | --- | --- |
| DIA_US_SEC | Upstream | — | — |
| DIA_US_SEC | Fast SEC | +7.6% | +0.0% |
| FIA_US_SEC | Upstream | -3.4% | -2.7% |
| FIA_US_SEC | Fast SEC | +5.4% | -0.0% |
| IUL_US_SEC | Upstream | -12.2% | -12.5% |
| IUL_US_SEC | Fast SEC | -5.5% | -0.0% |
| MYGA_US_SEC | Upstream | -6.0% | -7.7% |
| MYGA_US_SEC | Fast SEC | +0.3% | +0.0% |
| RILA_US_SEC | Upstream | +3.7% | -4.2% |
| RILA_US_SEC | Fast SEC | +5.1% | -5.1% |
| SPIA_US_SEC | Upstream | -16.0% | -31.8% |
| SPIA_US_SEC | Fast SEC | +2.1% | +0.0% |
| Term_US_SEC | Upstream | -4.5% | -7.0% |
| Term_US_SEC | Fast SEC | +0.6% | -0.0% |
| ULSG_US_SEC | Upstream | -6.5% | -2.3% |
| ULSG_US_SEC | Fast SEC | -16.5% | +0.0% |
| UL_US_SEC | Upstream | -1.2% | -3.0% |
| UL_US_SEC | Fast SEC | +3.0% | +0.1% |
| VA_US_SEC | Upstream | -9.9% | -11.9% |
| VA_US_SEC | Fast SEC | +3.4% | -0.0% |
| VUL_US_SEC | Upstream | — | — |
| VUL_US_SEC | Fast SEC | +3.6% | +0.0% |
| WholeLife_US_SEC | Upstream | +14.2% | -4.0% |
| WholeLife_US_SEC | Fast SEC | +43.4% | +0.0% |
