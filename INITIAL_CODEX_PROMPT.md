# Initial prompt for Codex Desktop

Read `AGENTS.md`, `GUIDING_PRINCIPLES.md`, `TARGET_ARCHITECTURE.md`, `PLAN.md`, `DECISIONS.md`, and `CODEX_DESKTOP_SETUP.md` before doing anything else.

This is the first native-Windows verification of a modelx-cython performance project that previously ran on Linux. **Do not change compiler core or promote any branch in this task.** Treat all non-upstream implementations as experiments/references until their benefit is reproduced locally.

If no project Goal is active, set the durable Goal from `CODEX_GOAL.md`. Then use Plan mode first. Your plan must preserve the benchmark workload and distinguish environment/portability repairs from compiler optimization.

Then execute Stage 0 from `PLAN.md`:

1. Inspect Git branches and verify that `core` has compiler source identical to `upstream-pristine` (`git diff upstream-pristine..core -- modelx_cython` should be empty).
2. Run `codex doctor` and record native Windows/Python/MSVC/environment details.
3. Create a clean Python environment using the frozen dependency/version guidance under `project_refs/vendor/`. Prefer bundled modelx/lifelib source snapshots for baseline reproduction.
4. Run the untouched original modelx-cython test suite on `upstream-pristine` first. Fix only environment/Windows portability issues on a dedicated experiment/control branch; do not merge them into core automatically.
5. Verify/build and validate the supplied MYGA controls. Mandatory controls where source is provided are:
   - live modelx;
   - exported Python;
   - pristine modelx-cython;
   - exact minimum-helper experiment and the rebased `feature/helpers` branch;
   - H09 compatible reference;
   - previous optimized compiler;
   - fast native-frame prototype.
   H10/3A is a negative historical control. A retained negative-evidence summary includes its benchmark/test figures and exact source/archive identities, but the exact source is not in this package; rerun it only if the companion H10 source archive is supplied. Do not reconstruct H10 from the summary.
6. Validate every output before timing. For original and adjusted MYGA, reproduce the complete seven-policy calculation workload and secondary public report/output checks defined by the retained harness/evidence.
7. Rerun contemporaneous native-Windows MYGA benchmarks with raw trials. Compare controls within this Windows campaign; do not pool Linux and Windows absolute timings.
8. Specifically test the first promotion candidate: pristine upstream vs upstream + minimum helper support. Determine whether helpers still provide a material benefit on Windows and identify the smallest generic compiler prerequisite, if any, required by them.
9. Compare Windows results to retained Linux evidence and explain discrepancies by environment/toolchain/workload before attributing them to compiler source.
10. Produce `evidence/windows_baseline/BASELINE_REPORT.md` with:
    - environment/toolchain;
    - CI results;
    - correctness status for each control;
    - raw/summary benchmark table;
    - source identities;
    - blockers/portability fixes;
    - recommendation for the first promoted core change, or recommendation to promote nothing yet.

Do not implement the next optimization after this report. If you recommend promoting any structural compiler change, stop and ask me for approval first.
