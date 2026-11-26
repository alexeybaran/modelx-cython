# Proposal 1 census evidence

The paired census covers 87 of 95 driven models across 11 of 12 matched shards. The candidate side of shard 5 now reports that all eight models matched live. The upstream side is missing, so none of those eight models enters the paired statistics. The complete catalog appears below, with missing pairs and undriven entries marked. [Candidate shard 5](https://github.com/alexeybaran/modelx-benchmark-toolkit/tree/c0faaee95e89f13dc44eb3d0419b6ea14a5eb9e6/parallel/20261113/round4/gate/results/census2-cand-5)

## Pins and coverage

| Role | Verified SHA |
|---|---|
| Proposal 1 head `pr/fast-sec-3` | `7957dc3beb9237c779872bb479c2c9978b0ed296` |
| Measured candidate compiler | `503b07c6f13d646ca4a15b020a18aea7c677a205` |
| Baseline compiler | `d22426fbac8637f84510902c2fa33b6466e5acb4` |
| modelx for both | `fec516d7e9b5813e7d40819a72731bd35fc2972f` |
| lifelib for both | `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4` |

The head advances the measured candidate by one commit adding `test_power_accumulator_shapes.py` and `test_python_call_no_entry.py`, 99 lines total. All 22 shard artifacts carry the expected pins, tool options, shard assignments, and matching CPU model and CPU flags within each pair. The historical artifacts report Linux, free-threaded CPython 3.14.6, GCC 13.3 and Valgrind 3.22. [Test-only commit comparison](https://github.com/alexeybaran/modelx-cython/compare/503b07c6f13d646ca4a15b020a18aea7c677a205...7957dc3beb9237c779872bb479c2c9978b0ed296)

| Published functional outcome | Models |
|---|---:|
| Both passed live comparisons | 66 |
| Candidate passed, baseline failed | 8 |
| Same failure class on both | 13 |
| Baseline passed, candidate failed | 0 |
| Awaiting a published matched pair | 8 |
| Catalog entries without a modelx driver | 10 |

## Instruction counts and build costs

Ratios divide the candidate measurement by the baseline measurement. Use instruction ratios only for rows with live comparisons passing, optimization enabled, and the same counted policy quantity for per-policy routes. Sweep results count instructions per swept value. Five passing sweeps lack a measured instruction pair. Translation and C build statistics use the census rows' `translate_s` and `build_s` fields, restricted to rows with both sides passing. Completed stage ratios still appear on candidate-only and matching-failure rows in the full table when both sides record a duration. Some of those durations time an unsuccessful stage.

| Metric | Comparable rows | Min | Median | Max |
|---|---:|---:|---:|---:|
| SEC instructions per policy | 10 | 0.2024× | 0.2655× | 0.4542× |
| Regional instructions per policy | 44 | 0.9201× | 0.9898× | 1.0476× |
| Legacy sweep instructions per value | 7 | 0.8072× | 0.9639× | 1.0022× |
| Translation seconds, both pass | 66 | 0.8098× | 1.3804× | 1.9078× |
| C build seconds, both pass | 66 | 1.0475× | 1.2367× | 1.4052× |
| Incremental memory kB per policy, both pass | 52 | 0.2125× | 0.8428× | 0.9939× |
| Peak RSS at equal book size, both pass | 26 | 0.2336× | 0.8608× | 0.9902× |

Peak RSS uses matched book sizes and positive measurements on both sides. Incremental memory uses the sizing estimates that are available on both sides. Estimates can exist even when the separate peak RSS measurement hits the memory cap.

**LTC_JP_S (shard 2).** The candidate uses 514,431,746.9 instructions per policy and the baseline uses 491,080,255.2 across the same 20-policy book, a 1.0476× ratio (+4.76%). Both match live exactly in the recorded comparison. Both hit the memory cap on the separate 1,000-policy timing run. This is the only published comparable instruction ratio more than 3% above baseline. The cause remains open. [Candidate evidence](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/4d62ea633d5028a066493cfdfda27d983ed076cb/parallel/20261113/round4/gate/results/census2-cand-2/c503b07c6f1-mufec516d-lu20d4dae-census-2-of-12.json), [baseline evidence](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/4d34d98e8fbb2107eda24c40638d47da0af8a0b1/parallel/20261113/round4/gate/results/census2-up-2/cud22426fbac-mufec516d-lu20d4dae-census-2-of-12.json)

**SEC range reconciled from the published shards.** The provisional handover listed 0.20–0.32×. Shard 7 records SPIA_US_SEC at 2,433,261.8 / 6,555,316.9 = 0.3712× (22 policies) and VUL_US_SEC at 9,890,923.3 / 21,777,437.9 = 0.4542× (27 policies). Both use `instructions_per_policy` on both sides, pass the live check, and use matching CPU signatures and counted policies. These ratios also agree with the rounded ratios in the original draft's separate 12-model SEC run. The final range from the 10 published census SEC pairs is 0.2024–0.4542×. ULSG_US_SEC and VA_US_SEC await shard 5. [Candidate shard 7](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/d7f52bf01803475572494b510170a59ef1cfe512/parallel/20261113/round4/gate/results/census2-cand-7/c503b07c6f1-mufec516d-lu20d4dae-census-7-of-12.json), [baseline shard 7](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/8e7c4ddd5f30000aa67f4708b247bc22af99cbe5/parallel/20261113/round4/gate/results/census2-up-7/cud22426fbac-mufec516d-lu20d4dae-census-7-of-12.json)

**Paired wall time is unpublished.** The census wall-time fields use separately timed runs and often different sized books. The operational log also records cloud speed drift and run-to-run spread. These measurements cannot substantiate an alternating matched wall-time claim. [Operations log](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/r3/handover-review/parallel/20261113/round3/review/r4/OPERATIONS.md), [original packaging draft](https://github.com/alexeybaran/modelx-cython/blob/ee2dff0a59176399aa9e0cb699b3dc40c0523305/devnotes/PR-DESCRIPTION.md)

## Published failure classes

| Affected models | Outcome and evidence |
|---|---|
| BU_DE_S, Dep_FR_S, Euro_FR_S, IncomeTerm_JP_S, VA_US_S | Candidate passes. Baseline compiled run raises `IndexError: array index out of range`. |
| Term_JP_S, CI_KR_S, Term_KR_S | Candidate passes. Baseline Cython build reports `complex types are unordered`. |
| simplelife/model, nestedlife/model, ifrs17sim/model, solvency2/model | Both translations raise `KeyError: 'ExcelWorkbook'`. |
| economic/BasicG2PP | Both translations raise `KeyError: 'ModuleIO'`. |
| TradLife_A, TradLife_A_EX1, TradLife_A_mx30, IntegratedLife, CashValue_ME_G2PP | The compiled comparisons report `FileNotFoundError` on both sides. The census does not establish whether harness placement causes these errors. |
| WholeLife_JP_S | Both match cash flows and differ on the `result_pols.pols_apl` report dtype, compiled `int64` versus live `float64` on policy 1. |
| assets/BasicBonds | Live model fails to import `QuantLib` on both sides. |
| fastlife/model | modelx export raises `RuntimeError: must not happen` on both sides. |

## Census catalog

`S` is the default regional build, `SEC` uses the shipped SEC sample and spec, and `sweep` evaluates the generic legacy calls. A dash marks unavailable or inapplicable data. Ins./unit counts are available only where live values pass on both sides and the runs yield comparable counts. Peak RSS ratios require equal book sizes. Shard links point to immutable result commits.

| Model | Shard | Driver | Candidate | Baseline | Ins./unit candidate | Ins./unit baseline | Ins. ratio | Translate ratio | C build ratio | kB/policy ratio | Peak RSS ratio |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| delib/Basis_DE_S | 1 | S | ok | ok | 515,856,634.1 | 520,791,568.1 | 0.9905 | 1.2471 | 1.2454 | 0.2125 | 0.2336 |
| delib/BU_DE_S | 2 | S | ok | run-raises | — | — | — | 1.4837 | 1.1537 | — | — |
| delib/FRV_DE_S | 3 | S | ok | ok | 407,588,833.7 | 411,296,759.9 | 0.9910 | 1.2619 | 1.3904 | 0.8260 | 0.8490 |
| delib/Index_DE_S | 4 | S | ok | ok | 347,717,879.6 | 349,173,512.1 | 0.9958 | 1.4357 | 1.1516 | 0.9164 | 0.9365 |
| delib/KLV_DE_S | 5 | S | Pending | Pending | — | — | — | — | — | — | — |
| delib/RV_DE_S | 6 | S | ok | ok | 325,475,694.2 | 326,224,451.1 | 0.9977 | 0.8098 | 1.2401 | 0.8996 | — |
| delib/Pflege_DE_S | 7 | S | ok | ok | 4,913,254,848.9 | 4,981,188,759.7 | 0.9864 | 0.9393 | 1.2479 | 0.6767 | 0.6803 |
| delib/Riester_DE_S | 8 | S | ok | ok | 62,988,864.1 | 66,971,324.0 | 0.9405 | 1.6116 | 1.2322 | 0.2968 | — |
| delib/RLV_DE_S | 9 | S | ok | ok | 92,820,354.0 | 100,084,716.2 | 0.9274 | 1.5098 | 1.2626 | 0.8616 | — |
| delib/Sofort_DE_S | 10 | S | ok | ok | 247,969,046.4 | 251,238,191.3 | 0.9870 | 1.2749 | 1.2331 | 0.8737 | — |
| frlib/ADE_FR_S | 11 | S | ok | ok | 328,068,405.4 | 329,885,930.8 | 0.9945 | 1.3990 | 1.2322 | 0.8437 | 0.8529 |
| frlib/Euro_FR_S | 12 | S | ok | run-raises | — | — | — | 1.4076 | 1.2579 | — | — |
| frlib/UC_FR_S | 1 | S | ok | ok | 200,614,614.4 | 201,790,505.9 | 0.9942 | 1.7281 | 1.2604 | 0.7058 | 0.7736 |
| frlib/Dep_FR_S | 2 | S | ok | run-raises | — | — | — | 1.2712 | 1.2667 | — | — |
| frlib/EC_FR_S | 3 | S | ok | ok | 232,919,296.1 | 234,262,199.9 | 0.9943 | 1.8117 | 1.4052 | 0.9450 | 0.9459 |
| frlib/Obseques_FR_S | 4 | S | ok | ok | 190,205,638.4 | 191,910,059.9 | 0.9911 | 1.2829 | 1.1792 | 0.9377 | — |
| frlib/PER_FR_S | 5 | S | Pending | Pending | — | — | — | — | — | — | — |
| frlib/Rente_FR_S | 6 | S | ok | ok | 74,375,885.0 | 77,611,032.9 | 0.9583 | 0.9891 | 1.0823 | 0.7913 | — |
| frlib/TD_FR_S | 7 | S | ok | ok | 180,203,276.9 | 181,644,227.8 | 0.9921 | 1.3950 | 1.3030 | 0.8778 | — |
| jplib/Cancer_JP_S | 8 | S | ok | ok | 1,724,534,878.5 | 1,732,894,263.4 | 0.9952 | 1.1199 | 1.1649 | 0.9068 | 0.9065 |
| jplib/Endowment_JP_S | 9 | S | ok | ok | 60,612,558.9 | 62,951,709.2 | 0.9628 | 1.7581 | 1.2979 | 0.9173 | — |
| jplib/FXWholeLife_JP_S | 10 | S | ok | ok | 802,095,993.8 | 806,054,795.4 | 0.9951 | 1.4364 | 1.2883 | 0.8327 | 0.8423 |
| jplib/IncomeTerm_JP_S | 11 | S | ok | run-raises | — | — | — | 1.3263 | 1.1936 | — | — |
| jplib/Annuity_JP_S | 12 | S | ok | ok | 161,665,060.7 | 163,810,612.4 | 0.9869 | 1.4572 | 1.2239 | 0.8359 | — |
| jplib/Medical_JP_S | 1 | S | ok | ok | 344,159,478.1 | 344,140,303.6 | 1.0001 | 1.3449 | 1.2392 | 0.8572 | 0.8609 |
| jplib/LTC_JP_S | 2 | S | ok | ok | 514,431,746.9 | 491,080,255.2 | 1.0476 | 1.1054 | 1.1571 | 0.7802 | — |
| jplib/Term_JP_S | 3 | S | ok | build-fails | — | — | — | 1.2761 | 7.9010 | — | — |
| jplib/WholeLife_JP_S | 4 | S | values-differ | values-differ | — | — | — | 1.4481 | 1.1708 | 0.6436 | — |
| krlib/Cancer_KR_S | 5 | S | Pending | Pending | — | — | — | — | — | — | — |
| krlib/Child_KR_S | 6 | S | ok | ok | 1,109,829,789.7 | 1,147,475,455.6 | 0.9672 | 1.0582 | 1.1344 | 0.9733 | — |
| krlib/CI_KR_S | 7 | S | ok | build-fails | — | — | — | 0.8562 | 10.2029 | — | — |
| krlib/Immediate_KR_S | 8 | S | ok | ok | 276,976,578.4 | 277,564,445.1 | 0.9979 | 1.3422 | 1.2953 | 0.7116 | 0.7639 |
| krlib/Medical_KR_S | 9 | S | ok | ok | 142,754,081.1 | 144,469,417.3 | 0.9881 | 1.9078 | 1.2759 | 0.9608 | — |
| krlib/LTC_KR_S | 10 | S | ok | ok | 194,921,949.1 | 211,855,849.9 | 0.9201 | 1.1770 | 1.2443 | 0.5756 | — |
| krlib/Pension_KR_S | 11 | S | ok | ok | 425,208,030.4 | 429,716,151.8 | 0.9895 | 1.3500 | 1.2748 | 0.7966 | 0.8095 |
| krlib/Term_KR_S | 12 | S | ok | build-fails | — | — | — | 1.3905 | 8.2005 | — | — |
| krlib/VA_KR_S | 1 | S | ok | ok | 223,246,603.1 | 225,873,119.4 | 0.9884 | 1.4424 | 1.2129 | 0.8910 | 0.8874 |
| krlib/WholeLife_KR_S | 2 | S | ok | ok | 208,269,961.1 | 223,583,201.8 | 0.9315 | 0.8456 | 1.3062 | — | — |
| uklib/CI_UK_S | 3 | S | ok | ok | 84,024,612.3 | 85,281,112.0 | 0.9853 | 1.6600 | 1.2474 | 0.9103 | — |
| uklib/IP_UK_S | 4 | S | ok | ok | 635,583,764.8 | 653,163,420.2 | 0.9731 | 1.1868 | 1.1264 | 0.9939 | 0.9902 |
| uklib/PA_UK_S | 5 | S | Pending | Pending | — | — | — | — | — | — | — |
| uklib/Term_UK_S | 6 | S | ok | ok | 230,299,488.9 | 232,601,548.2 | 0.9901 | 1.5576 | 1.2140 | 0.8957 | — |
| uklib/ULB_UK_S | 7 | S | ok | ok | 303,524,552.9 | 306,212,967.2 | 0.9912 | 1.5450 | 1.2921 | 0.8240 | 0.8419 |
| uklib/WOL_UK_S | 8 | S | ok | ok | 695,037,650.5 | 697,265,808.0 | 0.9968 | 1.1799 | 1.1771 | 0.9027 | 0.9092 |
| uklib/WP_UK_S | 9 | S | ok | ok | 600,223,024.0 | 600,019,329.8 | 1.0003 | 1.5529 | 1.2460 | 0.9412 | 0.9417 |
| uslib/DIA_US_S | 10 | S | ok | ok | 435,434,430.2 | 441,910,941.7 | 0.9853 | 1.3122 | 1.3928 | 0.7986 | 0.8063 |
| uslib/DIA_US_SEC | 11 | SEC | ok | ok | 2,359,829.5 | 7,284,446.2 | 0.3240 | 1.6404 | 1.2953 | 0.4543 | — |
| uslib/MYGA_US_S | 12 | S | ok | ok | 1,053,785,852.1 | 1,061,473,789.3 | 0.9928 | 1.3773 | 1.2916 | 0.9055 | 0.9053 |
| uslib/MYGA_US_SEC | 1 | SEC | ok | ok | 2,357,490.2 | 10,269,965.1 | 0.2296 | 1.7175 | 1.3160 | 0.4460 | — |
| uslib/FIA_US_S | 2 | S | ok | ok | 563,940,806.5 | 568,940,969.0 | 0.9912 | 1.4593 | 1.2787 | 0.8579 | 0.8608 |
| uslib/FIA_US_SEC | 3 | SEC | ok | ok | 2,906,650.1 | 11,126,259.4 | 0.2612 | 1.3214 | 1.2921 | 0.4607 | — |
| uslib/ULSG_US_S | 4 | S | ok | ok | 379,362,697.4 | 383,193,444.2 | 0.9900 | 1.6095 | 1.2331 | 0.9305 | 0.9310 |
| uslib/ULSG_US_SEC | 5 | SEC | Pending | Pending | — | — | — | — | — | — | — |
| uslib/SPIA_US_S | 6 | S | ok | ok | 254,306,719.2 | 259,038,077.7 | 0.9817 | 1.0936 | 1.1608 | 0.7994 | 0.8231 |
| uslib/SPIA_US_SEC | 7 | SEC | ok | ok | 2,433,261.8 | 6,555,316.9 | 0.3712 | 1.3128 | 1.2901 | 0.4839 | — |
| uslib/IUL_US_S | 8 | S | ok | ok | 579,960,327.6 | 612,069,568.4 | 0.9475 | 1.1688 | 1.0475 | — | — |
| uslib/IUL_US_SEC | 9 | SEC | ok | ok | 9,322,305.4 | 33,126,610.2 | 0.2814 | 1.4032 | 1.3938 | 0.4938 | — |
| uslib/RILA_US_S | 10 | S | ok | ok | 1,078,413,284.0 | 1,102,319,061.3 | 0.9783 | 1.3741 | 1.3425 | 0.8470 | 0.8443 |
| uslib/RILA_US_SEC | 11 | SEC | ok | ok | 3,479,604.1 | 17,187,973.5 | 0.2024 | 1.6510 | 1.2673 | 0.2530 | — |
| uslib/Term_US_S | 12 | S | ok | ok | 549,308,808.5 | 553,456,861.2 | 0.9925 | 1.6113 | 1.2905 | 0.8419 | 0.8981 |
| uslib/Term_US_SEC | 1 | SEC | ok | ok | 998,566.6 | 4,050,890.4 | 0.2465 | 1.7708 | 1.3289 | 0.7344 | — |
| uslib/UL_US_S | 2 | S | ok | ok | 649,173,767.6 | 656,567,225.2 | 0.9887 | 1.5331 | 1.0761 | 0.9234 | 0.9253 |
| uslib/UL_US_SEC | 3 | SEC | ok | ok | 2,656,070.7 | 11,290,102.9 | 0.2353 | 1.5041 | 1.1506 | 0.5367 | — |
| uslib/VA_US_S | 4 | S | ok | run-raises | — | — | — | 1.3128 | 1.3122 | — | — |
| uslib/VA_US_SEC | 5 | SEC | Pending | Pending | — | — | — | — | — | — | — |
| uslib/VUL_US_S | 6 | S | ok | ok | 1,780,881,357.0 | 1,803,089,464.2 | 0.9877 | 1.3406 | 1.1741 | 0.9421 | 0.9409 |
| uslib/VUL_US_SEC | 7 | SEC | ok | ok | 9,890,923.3 | 21,777,437.9 | 0.4542 | 1.4046 | 1.2170 | 0.3848 | — |
| uslib/WholeLife_US_S | 8 | S | ok | ok | 111,458,647.8 | 116,821,440.2 | 0.9541 | 1.2990 | 1.2437 | 0.9419 | — |
| uslib/WholeLife_US_SEC | 9 | SEC | ok | ok | 2,028,950.1 | 7,521,936.5 | 0.2697 | 1.5682 | 1.2742 | 0.6373 | — |
| annuallife/TradLife_A | 10 | sweep | values-differ | values-differ | — | — | — | 1.4711 | 1.0452 | — | — |
| annuallife/TradLife_A_EX1 | 11 | sweep | values-differ | values-differ | — | — | — | 1.4167 | 1.0046 | — | — |
| annuallife/TradLife_A_mx30 | 12 | sweep | values-differ | values-differ | — | — | — | 1.5503 | 1.0413 | — | — |
| appliedlife/IntegratedLife | 1 | sweep | values-differ | values-differ | — | — | — | 1.9239 | 1.1927 | — | — |
| assets/BasicBonds | 2 | sweep | live-fails | live-fails | — | — | — | — | — | — | — |
| basiclife/BasicTermASL_ME | 3 | sweep | ok | ok | — | — | — | 1.1537 | 1.1920 | — | — |
| basiclife/BasicTerm_M | 4 | sweep | ok | ok | — | — | — | 1.1909 | 1.1929 | — | — |
| basiclife/BasicTerm_ME | 5 | sweep | Pending | Pending | — | — | — | — | — | — | — |
| basiclife/BasicTerm_S | 6 | sweep | ok | ok | 8,161.4 | 8,235.3 | 0.9910 | 1.3836 | 1.1934 | — | — |
| basiclife/BasicTerm_SC | 7 | sweep | ok | ok | 839.1 | 870.5 | 0.9639 | 1.4464 | 1.1168 | — | — |
| basiclife/BasicTerm_SE | 8 | sweep | ok | ok | 11,313.2 | 11,557.1 | 0.9789 | 1.3671 | 1.1647 | — | — |
| cluster/BasicTerm_ME_for_Cluster | 9 | sweep | ok | ok | 407.1 | 477.5 | 0.8526 | 1.2286 | 1.1557 | — | — |
| economic/BasicG2PP | 10 | sweep | translate-fails | translate-fails | — | — | — | 1.0000 | — | — | — |
| economic/BasicHullWhite | 11 | sweep | ok | ok | 90.0 | 89.8 | 1.0022 | 1.4400 | 1.1364 | — | — |
| economic/CashValue_ME_G2PP | 12 | sweep | values-differ | values-differ | — | — | — | 1.2319 | 1.1704 | — | — |
| fastlife/model | 1 | sweep | export-fails | export-fails | — | — | — | — | — | — | — |
| ifrs17sim/model | 2 | sweep | translate-fails | translate-fails | — | — | — | 1.1000 | — | — | — |
| nestedlife/model | 3 | sweep | translate-fails | translate-fails | — | — | — | 1.0714 | — | — | — |
| savings/CashValue_ME | 4 | sweep | ok | ok | — | — | — | 0.9670 | 1.0536 | — | — |
| savings/CashValue_ME_EX1 | 5 | sweep | Pending | Pending | — | — | — | — | — | — | — |
| savings/CashValue_ME_EX2 | 6 | sweep | ok | ok | — | — | — | 0.8955 | 1.0842 | — | — |
| savings/CashValue_ME_EX4 | 7 | sweep | ok | ok | — | — | — | 1.1827 | 1.1148 | — | — |
| savings/CashValue_SE | 8 | sweep | ok | ok | 6,011.5 | 6,350.6 | 0.9466 | 1.5149 | 1.2342 | — | — |
| simplelife/model | 9 | sweep | translate-fails | translate-fails | — | — | — | 1.0000 | — | — | — |
| smithwilson/model | 10 | sweep | ok | ok | 3,068.7 | 3,801.6 | 0.8072 | 1.5714 | 1.0714 | — | — |
| solvency2/model | 11 | sweep | translate-fails | translate-fails | — | — | — | 1.1667 | — | — | — |
| economic_curves/EIOPA_smith_wilson_test | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |
| economic_curves/Metropolis_Hastings | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |
| economic_curves/NelsonSiegelSvensson | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |
| economic_curves/bisection_alpha | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |
| economic_curves/black_scholes | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |
| economic_curves/correlated_brownian_motion | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |
| economic_curves/smith_wilson | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |
| economic_curves/stationary_bootstrap | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |
| economic_curves/stationary_bootstrap_calibration | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |
| ifrs17a/ifrs17 | — | skip | Undriven | Undriven | — | — | — | — | — | — | — |

## Immutable shard sources

| Shard | Candidate | Baseline |
|---|---|---|
| 1 | [c114059410](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/c114059410aae04d2dcb256a16572dca30757ce7/parallel/20261113/round4/gate/results/census2-cand-1/c503b07c6f1-mufec516d-lu20d4dae-census-1-of-12.json) | [ededee9429](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/ededee942947e7167261300bb0926f55f2575498/parallel/20261113/round4/gate/results/census2-up-1/cud22426fbac-mufec516d-lu20d4dae-census-1-of-12.json) |
| 2 | [4d62ea633d](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/4d62ea633d5028a066493cfdfda27d983ed076cb/parallel/20261113/round4/gate/results/census2-cand-2/c503b07c6f1-mufec516d-lu20d4dae-census-2-of-12.json) | [4d34d98e8f](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/4d34d98e8fbb2107eda24c40638d47da0af8a0b1/parallel/20261113/round4/gate/results/census2-up-2/cud22426fbac-mufec516d-lu20d4dae-census-2-of-12.json) |
| 3 | [7a55bf5210](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/7a55bf5210507551c2d80a057e44a9d56afb7550/parallel/20261113/round4/gate/results/census2-cand-3/c503b07c6f1-mufec516d-lu20d4dae-census-3-of-12.json) | [c6db2e81bd](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/c6db2e81bd52b53d4193deae651ef432d511f3b0/parallel/20261113/round4/gate/results/census2-up-3/cud22426fbac-mufec516d-lu20d4dae-census-3-of-12.json) |
| 4 | [c3413fc0bf](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/c3413fc0bfd2a71a6d80e75ef92776cde33385e9/parallel/20261113/round4/gate/results/census2-cand-4/c503b07c6f1-mufec516d-lu20d4dae-census-4-of-12.json) | [358d772ea5](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/358d772ea525f179e47c9804811987140721ac79/parallel/20261113/round4/gate/results/census2-up-4/cud22426fbac-mufec516d-lu20d4dae-census-4-of-12.json) |
| 5 | Pending | Pending |
| 6 | [707780bd1b](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/707780bd1bd0de0848406adf50e822a856ccb6bc/parallel/20261113/round4/gate/results/census2-cand-6/c503b07c6f1-mufec516d-lu20d4dae-census-6-of-12.json) | [ae306f77f6](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/ae306f77f64772777a854e283c47da1849894ea6/parallel/20261113/round4/gate/results/census2-up-6/cud22426fbac-mufec516d-lu20d4dae-census-6-of-12.json) |
| 7 | [d7f52bf018](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/d7f52bf01803475572494b510170a59ef1cfe512/parallel/20261113/round4/gate/results/census2-cand-7/c503b07c6f1-mufec516d-lu20d4dae-census-7-of-12.json) | [8e7c4ddd5f](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/8e7c4ddd5f30000aa67f4708b247bc22af99cbe5/parallel/20261113/round4/gate/results/census2-up-7/cud22426fbac-mufec516d-lu20d4dae-census-7-of-12.json) |
| 8 | [8b42fcde77](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/8b42fcde77c85dc1b23a6164f56b4d1b28e5ea5a/parallel/20261113/round4/gate/results/census2-cand-8/c503b07c6f1-mufec516d-lu20d4dae-census-8-of-12.json) | [331d75264b](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/331d75264b61957525489377bedeb37f4b6033e2/parallel/20261113/round4/gate/results/census2-up-8/cud22426fbac-mufec516d-lu20d4dae-census-8-of-12.json) |
| 9 | [7de6e0fbab](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/7de6e0fbab03b0a474a665320e1ed8585bc018f4/parallel/20261113/round4/gate/results/census2-cand-9/c503b07c6f1-mufec516d-lu20d4dae-census-9-of-12.json) | [64d9cbd643](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/64d9cbd643fe423d85f49427885d7c86cd990d5e/parallel/20261113/round4/gate/results/census2-up-9/cud22426fbac-mufec516d-lu20d4dae-census-9-of-12.json) |
| 10 | [da9f13be52](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/da9f13be528db254cac74cb85db732bae4082798/parallel/20261113/round4/gate/results/census2-cand-10/c503b07c6f1-mufec516d-lu20d4dae-census-10-of-12.json) | [3b495b801c](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/3b495b801cef4351c1ba2de4f82573bc6e7e86d1/parallel/20261113/round4/gate/results/census2-up-10/cud22426fbac-mufec516d-lu20d4dae-census-10-of-12.json) |
| 11 | [70db1a00de](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/70db1a00ded976d7c65187725120d1eb78cd1429/parallel/20261113/round4/gate/results/census2-cand-11/c503b07c6f1-mufec516d-lu20d4dae-census-11-of-12.json) | [872bd2f530](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/872bd2f5306fb960e75b15a3676dcd628a90c8b1/parallel/20261113/round4/gate/results/census2-up-11/cud22426fbac-mufec516d-lu20d4dae-census-11-of-12.json) |
| 12 | [2393bca3af](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/2393bca3afad730c04a96d5eeab4f5322f58f377/parallel/20261113/round4/gate/results/census2-cand-12/c503b07c6f1-mufec516d-lu20d4dae-census-12-of-12.json) | [26744729ea](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/26744729ead36f004d68ddc4439ce77db8ba2b45/parallel/20261113/round4/gate/results/census2-up-12/cud22426fbac-mufec516d-lu20d4dae-census-12-of-12.json) |

The catalog defines 105 entries and the shard assignment follows the driven entries in catalog order, modulo 12. [Census catalog](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/c114059410aae04d2dcb256a16572dca30757ce7/parallel/20261113/round4/gate/census.txt), [handover](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/r3/handover-review/parallel/20261113/round3/review/r4/HANDOVER-TO-CODEX.md)
