# Modelx-Cython Development — Guiding Principles

**Status: canonical for the Desktop Codex migration, based on the user-approved upstream-first/evidence-gated operating model.**

These are the governing rules for this project. Read them before `TARGET_ARCHITECTURE.md` and `PLAN.md`.

## Goal

Evolve **modelx-cython itself** so that, on lifelib workloads where the previous optimized compiler achieved native execution, calculation performance approaches that previous compiler while keeping the implementation generic, maintainable, reviewable and inside a Cython-based modelx-cython architecture.

The final core should contain only changes whose value is demonstrated by evidence.

## Principles

1. **Upstream first.** Pristine modelx-cython is the accepted compiler baseline. Preserve its export structure, Space/Cell model, public compiled-package API, discovery/tracing machinery and semantics unless a change earns promotion.

2. **Experiments are not core.** Investigate each idea on an experiment or feature branch/worktree. A technically correct implementation is not accepted merely because it exists. Failed, neutral and slower experiments remain evidence, not ancestry.

3. **Promotion requires material benefit.** A compiler change may enter the evolving core only when it provides either (a) a reproducible material performance benefit on an important workload proportionate to its complexity, or (b) necessary generic correctness/coverage/functionality. Performance-neutral structural refactors do not qualify merely because they look cleaner. Promotion requires explicit user approval after evidence is presented.

4. **One mechanism at a time.** Baseline first, state the hypothesis, identify the donor/mechanism, run a controlled ablation, and attribute the result before combining optimizations. A material remaining gap triggers investigation before another subsystem is invented.

5. **Use donors, do not inherit them.** The prototype and previous compiler are performance/implementation donors and controls, not architectural bases. Transfer only individually justified mechanisms. Under approved D1, cache/runtime work does not automatically authorize a private execution frame, prepared-policy ABI, batch runtime, formula duplication, custom exception runtime or other donor architecture.

6. **Stay Cython based.** Let Cython compile ordinary formula semantics. Do not build a second formula interpreter or pandas-expression interpreter. Object-capable Cython is the normal compatibility path; Python objects, tuples and Python APIs do not by themselves require an interpreted Python formula body. An actual-Python body is a last-resort compatibility path only if required by demonstrated coverage.

7. **Specialize only proven domains.** Where types/domains are sufficiently established, hot paths may use native arguments, direct internal Cython calls, dense/scalar/hash storage, encoded finite categories or prepared generic lookups. Where they are not, preserve Python object/hash/equality semantics. Keep one logical Cell result/cache per Cell.

8. **Keep compiler logic generic.** No MYGA-, VA-, product-, Cell-name-, mortality-table-, scenario- or other business-specific compiler branches. Product/model adaptations and helper vocabulary live outside core compiler logic and must remain reviewable.

9. **Helpers are a parallel feature.** Develop helper syntax/runtime on a separate feature branch and use adjusted models as performance experiments. Helpers must be generic, small and readable, and models using them must work in live modelx, exported Python and compiled execution. Compiler support required by helpers must independently earn promotion into core.

10. **Readable formulas matter.** Prefer compiling readable Python formulas unchanged. Use a small teachable helper vocabulary or limited model adaptation only when it materially improves performance/coverage or avoids disproportionate compiler complexity. Do not expose cache layouts, memoryviews or backend-specific mechanics to actuarial authors.

11. **Correctness gates performance.** Validation precedes timing. Preserve supported Python/modelx semantics, arguments/defaults, object ownership, exceptions, cache behavior and public outputs. Run untouched upstream modelx-cython CI after every compiler change proposed for promotion; focused regressions supplement, never replace it.

12. **Benchmark the real calculation fairly.** Keep tracing, validation and timing distinct. Compile a model once for the required output union. Compare contemporaneous controls on the same machine/toolchain/workload. Separate calculation, numeric collection, public report and lifecycle costs. Never claim a speedup by changing the problem.

13. **Original and adjusted workloads have different jobs.** Original lifelib models protect upstream behavior and reveal general overhead. Adjusted models isolate compiler-friendly syntax/helpers and performance mechanisms. The immediate performance target is calculation performance on lifelib models where the previous compiler achieved native execution; broader census checks establish genericity later.

14. **Windows is the new execution authority, history remains evidence.** After migration, native Windows is the primary development environment. Reproduce CI, correctness and benchmarks there before promoting the first post-migration base. Linux/H09/H10 results are historical controls, not automatic acceptance on Windows. Platform fixes stay isolated until they independently satisfy the promotion gate.

15. **Keep the project auditable and small.** `TARGET_ARCHITECTURE.md` records durable boundaries, `PLAN.md` records current staged work, `DECISIONS.md` records approvals, and experiment evidence records what was tried and why it was accepted/rejected. Do not accrete handover packages after migration; use Git branches, reproducible tests/benchmarks and concise evidence instead.

## Definition of success

The project succeeds when the evolving modelx-cython core:

- stays structurally close to upstream modelx-cython and remains understandable/upstreamable;
- preserves the normal Space/Cell API and Cython-based execution model;
- retains object-capable Cython for general compatibility and uses native specialization only where justified;
- passes untouched upstream CI and project regressions on the accepted environment;
- preserves original lifelib correctness/performance while adjusted models validate against live modelx/exported Python;
- approaches the previous compiler's **calculation performance on lifelib workloads where that compiler achieved native execution**, or explicitly attributes any material residual before another mechanism is considered;
- and contains only structural/compiler changes with demonstrated material value.

When performance, simplicity, coverage and readability conflict, measure the trade-off and make it explicit. Do not keep a change because of sunk cost.
