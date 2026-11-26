# Fast SEC local gate pilot and paired timing

Candidate compiler `7957dc3beb9237c779872bb479c2c9978b0ed296` and baseline `d22426fbac8637f84510902c2fa33b6466e5acb4` used modelx `fec516d7e9b5813e7d40819a72731bd35fc2972f` and lifelib `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4`. The gate tool was `b3e06efc73e0e16477df20f434bf25b6cde36ebc`. The machine was an eight-core Ryzen 7 5800H under WSL2 with free-threaded CPython 3.14.10, Cython 3.3.1 and Valgrind 3.22.0. GIL was off. Timed calls passed the gate's book-total value check at relative 1e-9 and absolute 1e-4. The counted per-policy checks used relative 1e-10 and absolute 1e-7.

## Sizing pilot

Both arms used `GATE_POLICIES=20 GATE_TOTAL_MAX_S=5 gate.sh census jplib/LTC_JP_S,uslib/Term_US_SEC`, with no 1,000-policy minimum on the probe or estimate. LTC was sized to 23 policies on the candidate and 32 upstream. Their best first timing lengths were 3.90 and 4.71 seconds. The paired LTC call used 23 policies on both sides.

The Term probe overestimated the candidate's safe book. The selected 41,470-policy trial peaked at 12,508 MB and was rejected by the gate's 60% RAM limit. The upstream estimate chose 7,556 policies. Both Term builds were reused with `TOTAL_BOOK=8000`. The fixed books passed at about 2.5 GB candidate and 3.4 GB upstream. A fixed book overrides the time estimate and the upstream timing took more than five seconds.

## Equal-book timing

`gate.sh pairtime` ran one untimed warm-up per side and five alternating fresh-process repetitions per arm. Each case ran once more after exceeding the ratio spread limit. The table summarizes all ten repetitions. Term timings and its identical-build control were pinned to CPU 0. Times are seconds, median [minimum–maximum]. Ratios are candidate seconds divided by upstream seconds, or side B divided by side A for the control.

| Workload | Policies | Upstream or side A | Candidate or side B | Median paired ratio [range] | Best-time ratio | Gate result |
|---|---:|---:|---:|---:|---:|---|
| LTC JP, Direct book | 23 | 3.530 [3.459–3.584] | 3.656 [3.593–3.751] | 1.033 [1.019–1.058] | 1.039 | Resolved on retry |
| SEC Term, Direct book | 8,000 | 6.736 [6.609–8.320] | 2.451 [2.083–3.749] | 0.361 [0.303–0.560] | 0.315 | Unresolved |
| SEC Term, identical candidate builds | 8,000 | 2.435 [2.263–5.741] | 2.350 [2.283–2.939] | 0.967 [0.512–1.070] | 1.009 | Unresolved |

Term's paired ratio and its identical-build control were still outside the repeatability gate on retry. The timings are diagnostic and are excluded from the headline performance claim. The other eleven SEC products have no published same-book paired timings in this evidence set. The twelve-model instruction table is independent of wall-time variability.

[Gate results, including both sizing passes and timing controls](https://github.com/alexeybaran/modelx-benchmark-toolkit/tree/bc911037b36d2d293c831615847e5a118642ede6/parallel/20261113/round4/gate/results/p1-local-pilot-term-ltc)
