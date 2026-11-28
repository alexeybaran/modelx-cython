# Released SEC benchmarks with lifelib's documented native build settings

This run compares upstream modelx-cython `d22426fbac8637f84510902c2fa33b6466e5acb4` with Fast SEC `eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df`. Both use modelx `fec516d7e9b5813e7d40819a72731bd35fc2972f` and lifelib `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4` for all twelve released SEC products.

The earlier generic harness omitted the documented SEC module directives. Cython rejected real-valued model expressions in DIA and VUL because it inferred complex powers. The recheck uses the lifelib build settings for both compilers. [DIA recipe](https://github.com/lifelib-dev/lifelib/blob/20d4daef3281d538b3fbda6d0fb951fcceb3a6f4/lifelib/libraries/uslib/products/deferred_income_annuity/model-sec.md#L1241) · [VUL recipe](https://github.com/lifelib-dev/lifelib/blob/20d4daef3281d538b3fbda6d0fb951fcceb3a6f4/lifelib/libraries/uslib/products/variable_ul/model-sec.md#L977)

## Procedure

1. Export the released product with its shipped spec and sample. The benchmark adds one Cell that totals monthly net cash flows, and a sample wrapper that runs the shipped sample and calls that Cell for the shipped points.
2. After translation, put `# cython: infer_types=True, cpow=True` at the start of the generated `_mx_classes.py` for each arm. Retain Python's original C flags and append `-ffp-contract=off`. Confirm that GCC compiles the classes with `-O3` and the contraction flag.
3. Build the native packages on the same WSL2 Ryzen 7 5800H host, using free-threaded CPython 3.14.8, Cython 3.3.2 and GCC 13.3.0. DIA and VUL were translated again. The other ten reuse the exact earlier translations, then rebuild their native packages with the documented settings. Their translation times describe that earlier translation stage.
4. Use the same ordered book at both worker counts and check the GIL is off in every process. Delete each Direct policy ItemSpace after evaluating it at both worker counts. Pause after each batch of 512 projections to run generation-zero garbage collection with all work in that batch complete. This uses the same cleanup schedule on both compilers and avoids the memory buildup seen in the earlier four-worker trials.
5. Run five Direct timing rounds in fresh processes per compiler and worker count. Alternate the compiler order within each model. Pin one-worker jobs to separate physical cores, with up to four independent jobs at a time. Run four-worker measurements one process at a time on four physical cores. Use two separate one-worker instruction rounds per compiler with the same cleanup schedule. Validate each policy value and the aggregate against live modelx in every run. Wall time and peak RSS columns show the median and minimum–maximum range.

For `BasicTerm_SC`, use lifelib's shipped basiclife sample and spec. The sample also evaluates the first and last model points to cover the whole book. The workload evaluates `pv_net_cf()` on the shipped 10,000-point book, repeating complete passes to get a useful timing interval. Each projection is recalculated after releasing the prior state. [Lifelib BasicTerm_SC documentation](https://github.com/lifelib-dev/lifelib/blob/20d4daef3281d538b3fbda6d0fb951fcceb3a6f4/doc/source/libraries/basiclife/BasicTerm_SC.rst)

The historical generic comparison uses identical books and the earlier one-worker cleanup schedule on both sides. Instruction counts provide the more repeatable measure of the setting change across those two sessions. [Earlier generic run and separate SEC DV01 report](REFRESH.md)

## Matched upstream comparison

All 26 optimized builds succeeded. All 260 timing runs and 52 instruction runs passed live-value checks. For each compiler, 1w/4w is its one-worker median divided by its four-worker median. The scaling ratio divides Fast SEC’s factor by upstream’s factor, so values above 1 indicate greater four-worker scaling on Fast SEC.

| SEC product | Projections | Upstream 1w µs/projection | Fast SEC 1w µs/projection | Instruction reduction (upstream / Fast SEC) | 1w elapsed speedup | Upstream 1w/4w | Fast SEC 1w/4w | Scaling ratio (Fast SEC / upstream) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BasicTerm_SC | 100,000 | 89.9 | 33.9 | 2.59× | 2.65× | 2.07× | 1.66× | 0.80× |
| DIA | 12,000 | 1,198.9 | 394.2 | 2.28× | 3.04× | 2.66× | 1.60× | 0.60× |
| FIA | 10,900 | 1,662.2 | 288.2 | 3.34× | 5.77× | 3.42× | 1.75× | 0.51× |
| IUL | 3,500 | 5,302.7 | 945.8 | 3.28× | 5.61× | 3.46× | 2.32× | 0.67× |
| MYGA | 34,300 | 1,737.3 | 274.6 | 4.00× | 6.33× | 3.41× | 2.24× | 0.66× |
| RILA | 27,000 | 2,184.4 | 374.8 | 4.32× | 5.83× | 2.34× | 2.24× | 0.96× |
| SPIA | 13,200 | 992.4 | 226.1 | 2.37× | 4.39× | 3.45× | 2.37× | 0.69× |
| Term | 31,900 | 574.4 | 103.4 | 3.80× | 5.55× | 3.04× | 2.34× | 0.77× |
| ULSG | 9,000 | 1,934.0 | 333.3 | 3.40× | 5.80× | 3.50× | 2.18× | 0.62× |
| UL | 9,500 | 1,712.5 | 271.7 | 3.81× | 6.30× | 3.18× | 1.93× | 0.61× |
| VA | 8,000 | 3,210.1 | 427.4 | 4.03× | 7.51× | 3.46× | 1.78× | 0.51× |
| VUL | 5,300 | 4,840.6 | 776.0 | 3.00× | 6.24× | 3.72× | 2.54× | 0.68× |
| WholeLife | 13,900 | 1,102.6 | 173.1 | 3.61× | 6.37× | 2.99× | 1.94× | 0.65× |

## Detailed results by compiler

| Model | Compiler | 1w µs/projection | 4w µs/projection | Instructions/projection | Peak MiB at 1w | Translation + C build seconds |
| --- | --- | --- | --- | --- | --- | --- |
| BasicTerm_SC | Upstream | 89.9 [89.3–91.2] | 43.5 [42.4–44.0] | 724,295 | 137.7 [137.6–137.7] | 5.6 + 20.2 |
| BasicTerm_SC | Fast SEC | 33.9 [33.6–34.1] | 20.5 [20.2–20.8] | 279,236 | 112.4 [112.3–112.6] | 8.3 + 23.4 |
| DIA_US_SEC | Upstream | 1,198.9 [1,188.7–1,485.9] | 451.5 [445.6–462.8] | 6,253,734 | 967.2 [967.2–967.3] | 37.3 + 117.6 |
| DIA_US_SEC | Fast SEC | 394.2 [386.6–412.0] | 246.0 [230.1–308.7] | 2,747,358 | 122.1 [122.0–122.4] | 63.8 + 148.1 |
| FIA_US_SEC | Upstream | 1,662.2 [1,569.0–2,200.2] | 486.6 [468.6–490.2] | 7,838,924 | 1,768.5 [1,768.5–1,768.8] | 55.3 + 138.3 |
| FIA_US_SEC | Fast SEC | 288.2 [277.3–299.3] | 164.2 [159.9–172.2] | 2,347,003 | 109.1 [109.0–109.3] | 81.3 + 169.1 |
| IUL_US_SEC | Upstream | 5,302.7 [4,962.6–6,669.4] | 1,532.6 [1,433.8–1,705.1] | 28,326,344 | 2,545.0 [2,544.9–2,545.2] | 58.3 + 131.3 |
| IUL_US_SEC | Fast SEC | 945.8 [907.9–1,023.8] | 407.9 [390.1–447.9] | 8,646,047 | 107.4 [107.3–107.5] | 81.8 + 157.6 |
| MYGA_US_SEC | Upstream | 1,737.3 [1,682.7–1,785.0] | 508.8 [501.2–512.7] | 7,632,400 | 1,352.0 [1,351.7–1,352.1] | 31.5 + 105.5 |
| MYGA_US_SEC | Fast SEC | 274.6 [270.5–291.0] | 122.8 [116.7–130.1] | 1,907,245 | 126.1 [125.9–127.4] | 56.9 + 130.6 |
| RILA_US_SEC | Upstream | 2,184.4 [1,861.1–2,550.7] | 933.8 [914.1–1,044.4] | 10,528,153 | 3,240.5 [3,240.4–3,240.6] | 45.0 + 148.0 |
| RILA_US_SEC | Fast SEC | 374.8 [333.2–395.8] | 167.5 [165.4–187.5] | 2,437,989 | 119.3 [119.1–120.1] | 83.2 + 177.9 |
| SPIA_US_SEC | Upstream | 992.4 [970.1–998.3] | 287.8 [281.5–299.8] | 5,445,860 | 669.2 [669.1–669.3] | 36.6 + 73.7 |
| SPIA_US_SEC | Fast SEC | 226.1 [223.9–243.9] | 95.3 [89.2–102.3] | 2,296,397 | 106.1 [105.8–106.2] | 46.8 + 93.2 |
| Term_US_SEC | Upstream | 574.4 [573.2–609.5] | 188.7 [185.9–194.8] | 3,615,600 | 313.3 [313.1–313.3] | 11.3 + 43.0 |
| Term_US_SEC | Fast SEC | 103.4 [100.0–106.7] | 44.2 [41.5–46.4] | 950,274 | 104.1 [104.1–104.4] | 23.2 + 47.8 |
| ULSG_US_SEC | Upstream | 1,934.0 [1,886.6–2,434.3] | 553.4 [537.6–573.8] | 8,958,046 | 1,422.7 [1,422.7–1,422.8] | 29.4 + 93.9 |
| ULSG_US_SEC | Fast SEC | 333.3 [307.3–368.9] | 153.2 [136.3–162.6] | 2,633,860 | 103.1 [103.0–103.2] | 49.4 + 114.9 |
| UL_US_SEC | Upstream | 1,712.5 [1,708.0–1,778.1] | 538.7 [530.9–542.4] | 9,554,512 | 1,027.3 [1,027.3–1,027.4] | 26.6 + 82.2 |
| UL_US_SEC | Fast SEC | 271.7 [256.4–277.3] | 141.1 [138.0–146.8] | 2,507,849 | 102.2 [102.1–102.4] | 42.2 + 83.3 |
| VA_US_SEC | Upstream | 3,210.1 [3,029.0–3,564.6] | 928.4 [847.9–973.8] | 12,255,825 | 2,803.2 [2,803.1–2,803.3] | 86.4 + 184.9 |
| VA_US_SEC | Fast SEC | 427.4 [416.6–439.4] | 240.6 [238.4–266.8] | 3,042,807 | 113.6 [113.5–113.7] | 111.4 + 205.6 |
| VUL_US_SEC | Upstream | 4,840.6 [3,901.3–5,638.3] | 1,300.1 [1,177.9–1,460.8] | 14,054,401 | 3,919.5 [3,919.3–3,919.8] | 41.3 + 109.1 |
| VUL_US_SEC | Fast SEC | 776.0 [705.5–831.3] | 305.8 [300.5–325.2] | 4,680,194 | 107.4 [107.3–107.6] | 64.9 + 139.0 |
| WholeLife_US_SEC | Upstream | 1,102.6 [1,082.9–1,121.3] | 368.6 [357.2–374.5] | 6,319,282 | 691.3 [691.3–691.4] | 26.5 + 80.1 |
| WholeLife_US_SEC | Fast SEC | 173.1 [169.7–186.5] | 89.3 [87.4–102.3] | 1,749,570 | 103.8 [103.7–103.9] | 39.5 + 80.6 |

## Effect of the 512-projection batches on worker scaling

The timing driver divides each batch of up to 512 policy projections into tasks of up to 64. It waits for every task in the batch to finish before running generation-zero garbage collection and submitting the next batch. The 34,300-policy MYGA book has 67 batches. The common cleanup schedule bounds memory for the upstream builds. Earlier upstream four-worker runs without this cleanup reached about 15 GiB in the 16 GiB WSL guest.

The diagnostic below reused the Fast SEC binaries and ordered books from the paired comparison. It held task size fixed and varied the placement of the waits and explicit garbage collection. Automatic garbage collection was disabled during each diagnostic interval. The whole-book driver queued all tasks and collected once at the end. Each median comes from three fresh processes with alternate run order, on the same WSL2 machine, with one or four distinct physical cores pinned, the GIL verified off, and every value checked against live modelx.

| Model | Projections | Batched with cleanup, 1w s | Batched with cleanup, 4w s | Batched with end-only cleanup, 4w s | Whole book, 1w s | Whole book, 4w s | Batched scaling | Whole-book scaling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DIA | 12,000 | 4.286 | 3.618 | 3.266 | 4.085 | 2.923 | 1.18× | 1.40× |
| FIA | 10,900 | 2.803 | 2.307 | 2.242 | 2.708 | 1.537 | 1.21× | 1.76× |
| MYGA | 34,300 | 8.092 | 5.657 | 5.050 | 7.738 | 2.856 | 1.43× | 2.71× |
| RILA | 27,000 | 8.595 | 5.899 | 5.540 | 8.385 | 4.544 | 1.46× | 1.85× |

Removing the waits between batches improved four-worker time on all four models with the same end-only cleanup. For MYGA the waits accounted for 2.194 s of the 2.801 s recovered. Median process CPU time fell from 15.311 s to 11.076 s at that step. DIA still scaled only 1.40×, and FIA and RILA were below 2.5× in this matched diagnostic. The earlier driver collected explicitly every 64 projections at one worker and imposed no explicit cleanup at four workers. Its scaling figures therefore included the effect of different cleanup schedules. The matched upstream table above uses common batching for both compilers. The whole-book test was run on Fast SEC only.

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
