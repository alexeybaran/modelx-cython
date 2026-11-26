# LTC JP instruction and cache attribution

The published 20-policy census recorded 514,431,746.9 candidate and 491,080,255.2 upstream instructions per policy, an increase of 4.755%. Compiled values matched live on both sides. The independent WSL2 reproduction counted 515,521,551.75 and 490,812,058.15 instructions per policy, respectively, an increase of 5.03%.

## Pinned reproduction

- Candidate compiler `7957dc3beb9237c779872bb479c2c9978b0ed296` and upstream compiler `d22426fbac8637f84510902c2fa33b6466e5acb4`.
- Both compilers used upstream modelx `fec516d7e9b5813e7d40819a72731bd35fc2972f` and lifelib `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4`.
- AMD Ryzen 7 5800H, eight physical cores, WSL2, free-threaded CPython 3.14.10 with GIL disabled, Cython 3.3.1, Valgrind 3.22.0. Gate source `b3e06efc73e0e16477df20f434bf25b6cde36ebc`.
- Gate command `GATE_POLICIES=20 GATE_TOTAL_MAX_S=5 gate.sh census jplib/LTC_JP_S,uslib/Term_US_SEC`, run once per compiler with the three pins above. The counted Direct route constructs, evaluates, and deletes each policy's item with the tables already warm. Each arm checked the 20 policy results against live in the counted process. The local count agreed with the direction and approximate size of the independently published census result.

## What accounts for the increase

| Generated code and recorded cost | Upstream | Fast SEC |
|---|---:|---:|
| `Projection.care_surv(s, t)` storage in the generated `_mx_classes.pxd` | `double[948][948]` values and `bint[948][948]` flags | `NumericCache` |
| Cache access or insertion instructions per policy, exclusive | 3.35M in the dense Cell entry | 28.89M in `NumericCache.reserve` and `NumericCache.publish` |
| All instructions per policy | 490.81M | 515.52M |

The value array upstream reserves 7.19 MB per ItemSpace before its flags are counted. The candidate initializes a native hash cache with 4,096 slots and stores reached keys. In this generated model `care_surv` is the only `NumericCache` field. The hash lookup and insertion instructions account for approximately the net instruction increase. Other generated formulas and cache code also differ, so this profile is attribution evidence without a single-change ablation. The published census's probe estimated 22% lower incremental memory per policy with the candidate.

The frozen compiler keeps the sparse cache. The instruction increase and its memory trade-off are disclosed in the description.
