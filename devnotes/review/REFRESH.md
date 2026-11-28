# Performance refresh on the review heads

The refresh runs on Ubuntu 24.04 under WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`, with eight physical Ryzen 7 5800H cores, free-threaded CPython 3.14.8, Cython 3.3.2, GCC 13.3.0, NumPy 2.5.5 and pandas 3.1.1. It verifies the GIL stays off. The [review branches](README.md) supply the exact source pins. [Earlier measurements](PERFORMANCE.md) retain their original toolchain and source pins.

| Workload | Matched comparison | Measurements |
| --- | --- | --- |
| All 12 released SEC products | Fast SEC and upstream compiler `d22426fbac8637f84510902c2fa33b6466e5acb4` with modelx `fec516d7e9b5813e7d40819a72731bd35fc2972f` and unchanged lifelib `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4` | Direct wall time per policy, process CPU, native instructions, peak RSS, translation and C build time. A failed upstream build gets an explicit row. |
| MYGA SEC | Direct and Collect in live modelx, exported Python and compiled code. Separate compiler Direct comparison between Fast SEC and Collect heads. | Equal-book time with Direct policy deletion. Separate retained-state peak RSS with no deletion. Exported and compiled four-worker timing. |
| RILA and DIA SEC | Compiled Direct and Collect on the Collect branch, with Fast SEC's one-worker Direct on the same input. | One and four workers, one-worker instructions, and translation and C-build time. |
| Full SEC DV01 and CS01 report | Fast SEC Direct and Collect-branch Direct and Collect on the corresponding rebased examples and the same ordered 1,050-policy book. | One and four workers. Wall time, process CPU, peak RSS and one-worker instructions. The timed Collect route calls a portfolio report Cell with the CSV input directory. |

Each matched elapsed comparison uses five alternating fresh-process rounds with no active builds. One-worker runs are pinned to distinct physical cores. Four-worker results use an isolated run on four physical cores. Instructions come from separate one-worker hardware-counter runs with event coverage checked. Each Direct policy value and Collect total is checked against live outputs. The full DV01 benchmark checks all 72 DV01 and CS01 values. Peak RSS uses the child's own high-water mark.

The SEC books are sized for several seconds on Fast SEC. The compiled MYGA comparison uses the same book at both worker counts. The higher-policy MYGA memory comparison covers exported and compiled modes. Live Direct retains a much larger per-policy dependency graph, so the live memory comparison uses the smaller common book.

The completed MYGA retained-state comparison is in [the Collect description](COMPILER-COLLECT.md). The timing drivers load the policy table before starting the MYGA stopwatch to verify the book size. The full DV01 report starts with its authored calculation and uses the CSV directory supplied to the model.

On the 34,300-policy MYGA book, Fast SEC Direct used 2,080,596 instructions per policy and the Collect compiler's Direct route used 2,080,600. Native Collect used 1,877,121, 9.8% below Direct. The upstream compiler's Direct control used 8,456,011. These are medians of two fresh one-worker instruction runs, each checked against live cash flows.

The refresh completed 280 checked timing and memory runs. Its released SEC Direct section used the generic compiler recipe. That recipe omitted the documented `infer_types=True, cpow=True` module directives, so the DIA and VUL failures below apply to that recipe. The corrected PR1 comparison uses lifelib's documented build and includes successful upstream runs for all twelve products. The initial build durations below are single observations from a queue with up to six simultaneous builds. [Corrected PR1 benchmark report](SEC-BUILD-BENCHMARKS.md)

For the three compiled Collect products, Direct instructions changed by at most 0.08% between the compiler branches. Collect used 7.5–9.8% fewer instructions. Collect compilation took 8.0–13.1% longer than Direct on the Collect compiler. The 4w scaling limit was DIA at 1.83×. DIA Collect's four-worker median was 2.340 s compared with 2.405 s for DIA Direct.

The full DV01 live totals were 129,960.61614055187 and 157,856.97087237984 for DV01 and CS01, respectively. All 72 values matched bit for bit in 25 of 30 timed runs. The five four-worker Collect runs used the agreed parallel-summation tolerance with relative tolerance `1e-10` and absolute tolerance `3e-6`. Their largest absolute difference was `1.3113e-6` and the largest relative difference from a nonzero value was `8.29e-8`. The complete report scaled 3.72× with Collect. The two DV01 builds took 355.2 + 331.0 s and 694.7 + 267.0 s for translation and native build, respectively. The second build contains the benchmark report Cell and the Collect-enabled example.

All wall and RSS entries are median [minimum–maximum] across five fresh processes. Retired instruction rows use two independent one-worker counter passes. Translation and native build durations are separate stages.

## Released SEC products with Direct under the earlier generic recipe

| Product | Book | Compiler | µs/policy | CPU seconds | Instructions/policy | Peak MiB | Translation + C build seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| DIA_US_SEC | 12,000 | Upstream | — | — | — | — | Cython build failed (`complex types are unordered`) |
| DIA_US_SEC | 12,000 | Fast SEC | 443.3 [425.0–468.7] | 5.318 [5.098–5.622] | 2,918,749 [2,917,857–2,919,641] | 122.1 [121.8–122.2] | 74.0 + 152.7 |
| FIA_US_SEC | 10,900 | Upstream | 1,702.7 [1,669.8–1,722.3] | 18.557 [18.104–18.770] | 8,225,660 [8,225,651–8,225,669] | 314.2 [313.8–314.4] | 55.3 + 160.0 |
| FIA_US_SEC | 10,900 | Fast SEC | 333.7 [319.2–346.2] | 3.633 [3.479–3.773] | 2,515,001 [2,515,001–2,515,002] | 108.6 [108.6–108.9] | 81.3 + 198.0 |
| IUL_US_SEC | 3,500 | Upstream | 5,768.1 [5,735.6–5,802.3] | 19.959 [19.832–20.177] | 32,573,896 [32,573,866–32,573,925] | 409.1 [409.0–409.3] | 58.3 + 143.5 |
| IUL_US_SEC | 3,500 | Fast SEC | 1,031.8 [984.4–1,054.1] | 3.611 [3.445–3.689] | 8,809,088 [8,808,907–8,809,269] | 107.2 [107.1–107.3] | 81.8 + 171.4 |
| MYGA_US_SEC | 34,300 | Upstream | 1,948.3 [1,866.2–2,054.0] | 66.799 [64.004–70.436] | 8,456,011 [8,455,991–8,456,031] | 273.9 [273.6–275.6] | 31.5 + 112.2 |
| MYGA_US_SEC | 34,300 | Fast SEC | 296.6 [287.6–330.0] | 10.172 [9.859–11.316] | 2,080,596 [2,080,596–2,080,596] | 127.5 [127.4–127.6] | 56.9 + 135.9 |
| RILA_US_SEC | 27,000 | Upstream | 1,932.7 [1,918.7–2,083.5] | 52.178 [51.798–56.250] | 11,171,512 [11,171,381–11,171,643] | 509.9 [509.8–510.0] | 45.0 + 148.3 |
| RILA_US_SEC | 27,000 | Fast SEC | 397.0 [388.6–465.1] | 10.719 [10.492–12.527] | 2,748,086 [2,748,086–2,748,086] | 119.6 [118.9–119.8] | 83.2 + 203.2 |
| SPIA_US_SEC | 13,200 | Upstream | 1,235.9 [1,185.3–1,274.5] | 16.314 [15.646–16.823] | 8,228,257 [8,228,257–8,228,258] | 175.3 [175.0–175.5] | 36.6 + 78.8 |
| SPIA_US_SEC | 13,200 | Fast SEC | 287.3 [270.8–292.9] | 3.793 [3.575–3.866] | 2,462,139 [2,462,132–2,462,145] | 105.6 [105.4–105.6] | 46.8 + 87.8 |
| Term_US_SEC | 31,900 | Upstream | 622.8 [619.3–658.3] | 19.824 [19.605–20.776] | 4,071,492 [4,071,487–4,071,497] | 130.2 [129.9–130.3] | 11.3 + 52.5 |
| Term_US_SEC | 31,900 | Fast SEC | 155.8 [138.8–161.4] | 4.960 [4.422–5.020] | 1,120,804 [1,120,711–1,120,897] | 103.6 [103.4–103.7] | 23.2 + 55.9 |
| ULSG_US_SEC | 9,000 | Upstream | 1,972.9 [1,867.0–2,010.5] | 17.754 [16.794–18.094] | 9,338,629 [9,338,485–9,338,772] | 266.8 [266.7–266.9] | 29.4 + 108.9 |
| ULSG_US_SEC | 9,000 | Fast SEC | 367.2 [311.4–385.6] | 3.305 [2.801–3.469] | 2,798,196 [2,798,193–2,798,198] | 102.8 [102.7–102.9] | 49.4 + 113.3 |
| UL_US_SEC | 9,500 | Upstream | 1,917.6 [1,912.6–1,969.7] | 18.210 [18.170–18.710] | 10,018,337 [10,018,336–10,018,337] | 215.0 [215.0–216.6] | 26.6 + 81.0 |
| UL_US_SEC | 9,500 | Fast SEC | 311.2 [308.6–356.7] | 2.954 [2.932–3.389] | 2,672,378 [2,672,373–2,672,383] | 101.9 [101.8–102.1] | 42.2 + 90.8 |
| VA_US_SEC | 8,000 | Upstream | 2,996.2 [2,911.0–3,078.2] | 23.968 [23.288–24.625] | 14,095,962 [14,095,932–14,095,992] | 446.4 [446.3–448.0] | 86.4 + 174.0 |
| VA_US_SEC | 8,000 | Fast SEC | 499.0 [489.0–503.4] | 3.992 [3.912–4.027] | 3,211,651 [3,211,635–3,211,667] | 113.2 [113.1–113.3] | 111.4 + 213.3 |
| VUL_US_SEC | 5,300 | Upstream | — | — | — | — | Cython build failed (`complex types are unordered`) |
| VUL_US_SEC | 5,300 | Fast SEC | 780.9 [767.3–826.1] | 4.137 [4.040–4.376] | 4,843,565 [4,843,565–4,843,565] | 106.9 [106.8–106.9] | 52.4 + 104.3 |
| WholeLife_US_SEC | 13,900 | Upstream | 1,114.5 [1,109.7–1,129.8] | 15.484 [15.425–15.704] | 6,754,439 [6,754,374–6,754,503] | 178.6 [178.4–178.7] | 26.5 + 73.8 |
| WholeLife_US_SEC | 13,900 | Fast SEC | 185.8 [185.6–187.5] | 2.583 [2.579–2.607] | 1,916,578 [1,916,571–1,916,586] | 104.5 [104.4–104.6] | 39.5 + 83.2 |

| Product | Policy book | Fast SEC / upstream elapsed | Fast SEC / upstream instructions |
| --- | ---: | ---: | ---: |
| DIA_US_SEC | 12,000 | — | — |
| FIA_US_SEC | 10,900 | 0.196× | 0.306× |
| IUL_US_SEC | 3,500 | 0.179× | 0.270× |
| MYGA_US_SEC | 34,300 | 0.152× | 0.246× |
| RILA_US_SEC | 27,000 | 0.205× | 0.246× |
| SPIA_US_SEC | 13,200 | 0.232× | 0.299× |
| Term_US_SEC | 31,900 | 0.250× | 0.275× |
| ULSG_US_SEC | 9,000 | 0.186× | 0.300× |
| UL_US_SEC | 9,500 | 0.162× | 0.267× |
| VA_US_SEC | 8,000 | 0.167× | 0.228× |
| VUL_US_SEC | 5,300 | — | — |
| WholeLife_US_SEC | 13,900 | 0.167× | 0.284× |

## MYGA SEC memory with no policy deletion

| Mode | Book | Direct peak MiB | Collect peak MiB | Peak change |
| --- | ---: | ---: | ---: | ---: |
| Live modelx | 14 | 751.66 [751.34–751.88] | 123.23 [122.91–123.62] | -83.6% |
| Python export | 14 | 140.30 [140.27–140.69] | 102.09 [101.85–102.13] | -27.2% |
| Compiled | 14 | 116.57 [116.45–116.62] | 104.14 [104.05–104.32] | -10.7% |
| Python export | 1,050 | 3,291.62 [3,291.30–3,291.71] | 102.76 [102.65–102.88] | -96.9% |
| Compiled | 1,050 | 1,208.00 [1,207.98–1,208.02] | 104.29 [104.21–104.84] | -91.4% |

## MYGA SEC elapsed time with Direct policy deletion

| Mode | Book | Workers | Direct ms/policy | Collect ms/policy | Collect vs Direct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Live modelx | 28 | 1 | 541.925 [537.920–551.897] | 151.141 [148.125–157.099] | -72.1% |
| Python export | 840 | 1 | 20.659 [20.164–20.937] | 16.802 [16.385–17.258] | -18.7% |
| Python export | 840 | 4 | 7.720 [7.627–7.844] | 4.710 [4.622–4.841] | -39.0% |
| Compiled | 34,300 | 1 | 0.290 [0.274–0.337] | 0.246 [0.227–0.268] | -15.2% |
| Compiled | 34,300 | 4 | 0.070 [0.063–0.081] | 0.065 [0.064–0.069] | -6.3% |

## Compiled Collect routes

| Product | Book | Route | 1w seconds | 4w seconds | Scaling | Instructions/policy 1w | Translation + C build seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| DIA_US_SEC | 12,000 | Fast SEC Direct | 5.320 [5.100–5.624] | — | — | 2,918,749 [2,917,857–2,919,641] | 74.0 + 152.7 |
| DIA_US_SEC | 12,000 | Direct | 5.414 [5.321–5.489] | 2.405 [2.364–2.466] | 2.252× | 2,920,977 [2,920,972–2,920,982] | 57.5 + 131.8 |
| DIA_US_SEC | 12,000 | Collect | 4.285 [4.188–4.525] | 2.340 [2.306–2.366] | 1.831× | 2,702,727 [2,702,711–2,702,742] | 59.4 + 150.5 |
| MYGA_US_SEC | 34,300 | Fast SEC Direct | 10.174 [9.864–11.319] | — | — | 2,080,596 [2,080,596–2,080,596] | 56.9 + 135.9 |
| MYGA_US_SEC | 34,300 | Direct | 9.935 [9.396–11.572] | 2.394 [2.170–2.765] | 4.151× | 2,080,600 [2,080,600–2,080,600] | 55.9 + 134.8 |
| MYGA_US_SEC | 34,300 | Collect | 8.423 [7.803–9.188] | 2.242 [2.197–2.363] | 3.757× | 1,877,121 [1,877,119–1,877,122] | 58.9 + 156.7 |
| RILA_US_SEC | 27,000 | Fast SEC Direct | 10.719 [10.492–12.558] | — | — | 2,748,086 [2,748,086–2,748,086] | 83.2 + 203.2 |
| RILA_US_SEC | 27,000 | Direct | 10.854 [10.499–12.077] | 3.305 [3.194–3.390] | 3.284× | 2,748,109 [2,748,109–2,748,109] | 66.6 + 156.3 |
| RILA_US_SEC | 27,000 | Collect | 9.056 [8.460–9.609] | 2.441 [2.264–2.626] | 3.710× | 2,542,636 [2,542,636–2,542,636] | 68.4 + 172.3 |

## Full SEC DV01 and CS01 report

Each round uses 1,050 policies and one ESG path. Each policy runs one Actual projection and 39 Valuation projections, with one base and 12 shocks at each of three closes. The wall time per policy-scenario divides the full report time by 42,000 projections. Each round checks all 72 report values.

| Branch | Route | Workers | Wall seconds | Wall µs/policy-scenario | CPU seconds | Peak MiB | Instructions/policy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast SEC | Direct | 1 | 14.664 [14.009–14.761] | 349.1 [333.5–351.4] | 14.458 [13.849–14.488] | 258.2 [258.1–258.2] | 82,738,337 [82,738,315–82,738,359] |
| Fast SEC | Direct | 4 | 3.530 [3.479–3.677] | 84.1 [82.8–87.5] | 12.647 [12.371–13.242] | 356.7 [329.8–365.4] | — |
| Collect | Direct | 1 | 13.964 [13.631–14.197] | 332.5 [324.5–338.0] | 13.515 [13.326–13.816] | 258.7 [258.6–258.8] | 82,218,660 [82,217,214–82,220,105] |
| Collect | Direct | 4 | 3.489 [3.407–3.709] | 83.1 [81.1–88.3] | 12.516 [12.180–13.302] | 349.7 [336.0–358.8] | — |
| Collect | Collect | 1 | 11.473 [11.321–12.080] | 273.2 [269.5–287.6] | 11.344 [11.162–11.827] | 171.6 [171.6–171.9] | 80,520,875 [80,520,534–80,521,216] |
| Collect | Collect | 4 | 3.085 [3.062–3.127] | 73.4 [72.9–74.5] | 10.685 [10.602–10.803] | 180.9 [180.7–182.3] | — |
