# Test and Benchmark Protocol

`PLAN.md` defines when these gates run. This file defines the stable measurement rules.

## Compiler promotion minimum

For every compiler change proposed for core:

1. untouched upstream modelx-cython suite;
2. focused tests for the mechanism;
3. original-model correctness/non-regression;
4. relevant adjusted/native-target correctness;
5. matched current-core vs experiment performance campaign;
6. generated-code/source inspection when the hypothesis concerns physical execution;
7. complexity/delta review.

## MYGA standard workload

Retain the historical seven-policy workload and the same complete calculation output union used by the H09 evidence. Primary performance is the full calculation required for all `net_cf(t)` values through each policy's actual projection. Public `result_cf()` is secondary. Keep the scalar full-projection diagnostic only as attribution evidence, never as a substitute for the primary workload.

Compile each candidate once for the required output union. Tracing, validation and timing are distinct stages.

## Timing rules

- Validate before timing.
- No overlapping compiler/test/validation work during accepted timing.
- Keep control and experiment on the same machine/toolchain/dependency set.
- Interleave trial order when possible.
- Keep raw trials.
- Fix thread-count environment variables.
- Record CPU/toolchain/power/environment.
- Separate calculation, numeric collection/report and lifecycle costs.
- Historical Linux values are reference only after migration; Windows decisions use contemporaneous Windows controls.

## Material benefit

There is no universal fixed percentage. Judge effect size against trial noise, workload importance and added complexity. A tiny statistically detectable improvement may still be immaterial if it adds substantial machinery. A large benefit on an adjusted model may justify keeping a feature branch without proving that all of its compiler prerequisites belong in core.
