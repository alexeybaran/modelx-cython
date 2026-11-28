# Released SEC benchmarks with lifelib's documented native build settings

This run compares upstream modelx-cython `d22426fbac8637f84510902c2fa33b6466e5acb4` with Fast SEC `eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df`. Both use modelx `fec516d7e9b5813e7d40819a72731bd35fc2972f` and lifelib `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4` for all twelve released SEC products.

The earlier generic harness omitted the documented SEC module directives. Cython rejected real-valued model expressions in DIA and VUL because it inferred complex powers. The recheck uses the lifelib build settings for both compilers. [DIA recipe](https://github.com/lifelib-dev/lifelib/blob/20d4daef3281d538b3fbda6d0fb951fcceb3a6f4/lifelib/libraries/uslib/products/deferred_income_annuity/model-sec.md#L1241) · [VUL recipe](https://github.com/lifelib-dev/lifelib/blob/20d4daef3281d538b3fbda6d0fb951fcceb3a6f4/lifelib/libraries/uslib/products/variable_ul/model-sec.md#L977)

## Procedure

1. Export the released product with its shipped spec and sample. The benchmark adds one Cell that totals monthly net cash flows, and a sample wrapper that runs the shipped sample and calls that Cell for the shipped points.
2. After translation, put `# cython: infer_types=True, cpow=True` at the start of the generated `_mx_classes.py` for each arm. Retain Python's original C flags and append `-ffp-contract=off`. Confirm that GCC compiles the classes with `-O3` and the contraction flag.
3. Build the native packages on the same WSL2 Ryzen 7 5800H host, using free-threaded CPython 3.14.8, Cython 3.3.2 and GCC 13.3.0. DIA and VUL were translated again. The other ten reuse the exact earlier translations, then rebuild their native packages with the documented settings. Their translation times describe that earlier translation stage.
4. Once the builds finish, run five alternating fresh-process Direct rounds per pair and two dedicated instruction rounds. Check the GIL is off, use one worker per assigned physical core, and use the same ordered book as the corresponding generic test. Validate each policy value and the aggregate against live modelx. Wall time and peak RSS columns show the median and minimum–maximum range.

The comparison with the generic harness uses identical books. Instruction counts provide the more repeatable measure of the setting change across the two sessions. Elapsed changes across the sessions include run-to-run variation. [Earlier generic run and separate SEC DV01 report](REFRESH.md)

## Matched upstream comparison

All 24 optimized builds succeeded. All 120 timing rounds and 48 instruction rounds passed live-value checks.

| SEC product | Policies | Upstream µs/policy | Fast SEC µs/policy | Fast SEC / upstream instructions | Elapsed speedup |
| --- | --- | --- | --- | --- | --- |
| DIA | 12,000 | 1,293.8 | 477.0 | 0.455× | 2.71× |
| FIA | 10,900 | 1,644.8 | 351.6 | 0.314× | 4.68× |
| IUL | 3,500 | 5,064.8 | 975.0 | 0.309× | 5.19× |
| MYGA | 34,300 | 1,830.8 | 297.4 | 0.267× | 6.16× |
| RILA | 27,000 | 2,004.7 | 417.2 | 0.244× | 4.80× |
| SPIA | 13,200 | 1,038.7 | 293.5 | 0.439× | 3.54× |
| Term | 31,900 | 594.6 | 156.7 | 0.296× | 3.80× |
| ULSG | 9,000 | 1,843.8 | 306.7 | 0.307× | 6.01× |
| UL | 9,500 | 1,894.8 | 320.4 | 0.275× | 5.91× |
| VA | 8,000 | 2,698.8 | 516.1 | 0.259× | 5.23× |
| VUL | 5,300 | 3,566.2 | 808.8 | 0.341× | 4.41× |
| WholeLife | 13,900 | 1,272.5 | 266.5 | 0.296× | 4.77× |

## Detailed results by compiler

| Model | Compiler | µs/policy | Instructions/policy | Peak MiB | Translation + C build seconds |
| --- | --- | --- | --- | --- | --- |
| DIA_US_SEC | Upstream | 1,293.8 [1,267.6–1,310.2] | 6,416,078 | 226.6 [224.6–226.9] | 37.3 + 117.6 |
| DIA_US_SEC | Fast SEC | 477.0 [453.6–492.6] | 2,919,602 | 122.2 [122.0–122.3] | 63.8 + 148.1 |
| FIA_US_SEC | Upstream | 1,644.8 [1,639.8–1,650.2] | 8,006,194 | 314.4 [313.9–314.5] | 55.3 + 138.3 |
| FIA_US_SEC | Fast SEC | 351.6 [337.2–358.1] | 2,514,970 | 109.0 [108.9–109.1] | 81.3 + 169.1 |
| IUL_US_SEC | Upstream | 5,064.8 [4,975.3–5,084.9] | 28,491,299 | 409.3 [409.2–409.6] | 58.3 + 131.3 |
| IUL_US_SEC | Fast SEC | 975.0 [960.7–1,079.0] | 8,808,902 | 107.4 [107.3–107.6] | 81.8 + 157.6 |
| MYGA_US_SEC | Upstream | 1,830.8 [1,751.2–1,883.7] | 7,805,650 | 275.6 [273.9–275.7] | 31.5 + 105.5 |
| MYGA_US_SEC | Fast SEC | 297.4 [297.0–346.3] | 2,080,612 | 127.7 [126.5–127.8] | 56.9 + 130.6 |
| RILA_US_SEC | Upstream | 2,004.7 [1,838.4–2,030.4] | 10,698,778 | 510.2 [510.1–510.4] | 45.0 + 148.0 |
| RILA_US_SEC | Fast SEC | 417.2 [376.3–433.3] | 2,609,086 | 120.2 [120.0–120.2] | 83.2 + 177.9 |
| SPIA_US_SEC | Upstream | 1,038.7 [1,001.4–1,052.0] | 5,611,752 | 175.3 [175.2–175.4] | 36.6 + 73.7 |
| SPIA_US_SEC | Fast SEC | 293.5 [282.8–301.2] | 2,462,501 | 105.7 [105.6–105.8] | 46.8 + 93.2 |
| Term_US_SEC | Upstream | 594.6 [580.3–605.7] | 3,786,286 | 130.1 [130.0–130.1] | 11.3 + 43.0 |
| Term_US_SEC | Fast SEC | 156.7 [153.0–179.1] | 1,120,721 | 103.6 [103.4–103.6] | 23.2 + 47.8 |
| ULSG_US_SEC | Upstream | 1,843.8 [1,735.8–1,875.5] | 9,122,016 | 266.7 [266.5–266.7] | 29.4 + 93.9 |
| ULSG_US_SEC | Fast SEC | 306.7 [301.8–332.8] | 2,798,205 | 103.2 [103.1–103.4] | 49.4 + 114.9 |
| UL_US_SEC | Upstream | 1,894.8 [1,877.9–1,903.3] | 9,722,471 | 216.6 [216.5–216.7] | 26.6 + 82.2 |
| UL_US_SEC | Fast SEC | 320.4 [302.1–330.7] | 2,675,619 | 102.0 [101.9–102.0] | 42.2 + 83.3 |
| VA_US_SEC | Upstream | 2,698.8 [2,556.3–2,985.8] | 12,423,597 | 448.1 [446.4–448.1] | 86.4 + 184.9 |
| VA_US_SEC | Fast SEC | 516.1 [468.1–536.3] | 3,211,602 | 113.2 [113.0–113.4] | 111.4 + 205.6 |
| VUL_US_SEC | Upstream | 3,566.2 [3,471.4–3,595.4] | 14,219,833 | 583.2 [583.0–583.2] | 41.3 + 109.1 |
| VUL_US_SEC | Fast SEC | 808.8 [778.3–838.1] | 4,843,580 | 106.8 [106.7–106.8] | 64.9 + 139.0 |
| WholeLife_US_SEC | Upstream | 1,272.5 [1,235.7–1,326.2] | 6,486,373 | 178.6 [178.6–178.7] | 26.5 + 80.1 |
| WholeLife_US_SEC | Fast SEC | 266.5 [251.7–268.4] | 1,917,162 | 104.6 [104.5–104.7] | 39.5 + 80.6 |

## Impact on the earlier generic measurements

Upstream retired instructions dropped on every available earlier pair. SPIA changed most at 31.8%. The Fast SEC instruction differences were at most 0.1% except RILA, which fell 5.1%. WholeLife's elapsed change across the two sessions was much larger than its instruction change. The matched pair above is the basis for the WholeLife speed ratio.

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
