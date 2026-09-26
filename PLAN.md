# Evidence-Gated Development Plan

## Overall objective

Starting from pristine modelx-cython, approach the previous optimized compiler's **calculation performance on lifelib models where it achieved native execution**, while keeping the evolving compiler Cython-based, generic, maintainable and close to upstream.

The project no longer advances by accepting each completed structural stage. It advances by running experiments and promoting only the minimal changes that prove material value.

## Stage 0 — native-Windows reconstruction and baseline

**No compiler-core changes are allowed in this stage.**

1. Establish a native Windows Python/C++ build environment.
2. Restore/verify these controls where source is supplied:
   - live modelx;
   - exported Python;
   - pristine modelx-cython (`upstream-pristine` / `core` compiler source);
   - `experiment/helpers-exact-linux-control` / the rebased `feature/helpers` candidate;
   - H09 compatible reference;
   - previous optimized compiler;
   - fast native-frame prototype;
   - H10/3A only if its exact archived source is supplied (negative control; not a blocker for development).
3. Run untouched upstream modelx-cython CI on pristine upstream and on the helper candidate.
4. Build and validate original and adjusted MYGA against live modelx/exported Python.
5. Rerun contemporaneous original/adjusted MYGA benchmarks on Windows. Preserve raw trials and compiler/toolchain identity.
6. Compare Windows results with retained Linux evidence without pooling the measurements.
7. Produce a baseline report and recommendation. **Do not promote the helper branch until the user approves based on this evidence.**

### Stage-0 exit gate

- Native Windows build path documented and reproducible.
- Pristine upstream untouched CI passes, or every platform-specific failure is precisely attributed.
- Mandatory benchmark/control matrix has valid correctness results before timing.
- Helper candidate benefit and complexity are explicitly evaluated on Windows.
- User approves the first evolving core state.

## Stage 1 — establish first promoted core

Expected experiment: `feature/helpers`, but the result is not predetermined.

If helper support demonstrates material benefit and all gates pass, identify the **smallest generic compiler prerequisite** that should be promoted. Helper/model code may remain on its parallel branch even when a small compiler prerequisite enters core.

Promotion report must include:

- hypothesis and mechanism;
- source delta from current core;
- untouched CI and focused regression results;
- original-model non-regression;
- adjusted-model correctness;
- paired benchmark distributions versus current core;
- generated-code difference where relevant;
- added complexity and rollback path;
- recommendation: promote / reject / continue investigation.

Stop for user approval before changing `core`.

## Stage 2+ — donor-mechanism experiments

Choose the next mechanism from profiling/generated-code comparison, not from the historical stage numbering. The likely investigations are below, but their order is evidence-driven.

### Candidate investigation A — native category / exact-key cache representation

Test only Cells where the exact map shows Python dictionary/tuple overhead but the donor uses a justified native representation. Preserve object semantics outside proven domains. Measure whether this explains a material part of the residual.

### Candidate investigation B — clean internal native Cell entry

Test whether compiled dependencies can call a private Cython entry directly while public Space/Cell API, overrides and formula ownership remain unchanged. Do not duplicate formula bodies. Measure call/error overhead independently.

### Candidate investigation C — input/reference preparation

Investigate mapped input-like Cells and repeated binding/reference access. Test a small generic preparation mechanism before considering any donor-style private execution frame.

### Candidate investigation D — call/error/cycle protocol

Only after profiling shows material residual overhead. Prefer Cython-native mechanisms and proven-safe specialization. Do not reintroduce H10/3A state machinery without a new ablation showing net value.

### D1 review boundary

Only if the above upstream-local mechanisms leave a specifically attributed material gap should the project ask whether to trial larger donor mechanisms such as a private execution frame or prepared-policy ABI. That requires a new explicit user decision.

## Standard promotion gate for every compiler experiment

1. **Before coding:** write the hypothesis, expected benefit, controls, workload, success criteria and rollback.
2. Branch/worktree from the current `core`; do not edit `core` directly.
3. Add focused tests for the mechanism without weakening untouched tests.
4. Run correctness/validation before timing.
5. Run contemporaneous current-core and experiment benchmarks on the same machine/toolchain/workload.
6. Run untouched upstream modelx-cython CI for any change proposed for promotion.
7. Check original lifelib non-regression and the relevant adjusted/native-target workload.
8. Inspect generated code when the hypothesis concerns nativeness/call/cache shape.
9. Record source delta, benefit, complexity, uncertainty and negative results.
10. If benefit is not material, reject the branch and preserve the evidence.
11. If it qualifies, present a promotion recommendation and **stop for user approval**.
12. After approval, integrate the smallest justified delta and rerun the exact gate on the resulting `core`.

## Standard benchmark set

The standard test workflow should retain at least:

- untouched upstream modelx-cython CI;
- original MYGA correctness + calculation performance;
- adjusted MYGA correctness + calculation performance;
- helper-specific three-environment tests when helper branch is involved;
- mixed/object regressions only for semantics actually retained by the evolving core;
- previous compiler and native prototype as performance references where runnable;
- BasicTerm_SC and other previously native lifelib models at planned genericity gates.

Do not silently change the workload, policy set, output union or timing boundary to improve a ratio.

## Later genericity gates

Do not run the full census after every edit. After one or more mechanisms have earned promotion on MYGA, rerun selected previously-native lifelib models (including BasicTerm_SC) to verify that gains generalize. Run a full census only at an explicit coverage milestone.

## Final acceptance

The project is complete only when the accepted `core`—not a collection of experiment branches—meets the definition of success in `GUIDING_PRINCIPLES.md`, with a small reviewable delta and complete evidence.
