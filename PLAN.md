# Evidence-Gated Development Plan

## Local audit closure — 2026-11-08

The cleanup candidate on `experiment/collect-15x` resolves lexical-scope/binding,
native representation, helper capability, generator admission, collection failure
ordering and serialization findings. Shared contracts replace inconsistent proof
logic. `EnumDefinitions` retains its existing fixed/rebuild-required schema:
construction rejects subclass schema shadowing, ordinary References and Cell
overrides retain their existing behavior, and formula constants need no runtime
enum mode. Unsuccessful guard/helper/proof experiments were removed.

Windows 3.12 and free-threaded 3.14 each pass 225 maintained tests (7 skips) and
128 candidate/helper regressions. Immutable untouched tests each yield 169 passes,
7 skips and the two disclosed generated-source assertion failures; maintained
structural assertions and their original runtime oracles pass. Full adjusted
MYGA/VA report, 2,500-policy, 2,499-transition and 1/4-worker route checks pass.

The preplanned eight serial paired rounds meet both <=2x September API gates:
MYGA 1.8843 median ratio (upper mean CI 1.9459), VA 1.9499 (upper mean CI 1.9849).
Four full confirmation rounds bring the campaign to 168 retained trials. VA route
paired medians are within 0.6% of parity; parallel mean uncertainty is disclosed.
Native class field declarations and public Cell signatures are preserved. Builds
use no native PGO or forced-inlining flags. See
`../september-gap/CLEANUP_AUDIT.md` and `doc/source/native_contracts.md`.

This qualifies the current adjusted MYGA/VA audit envelope, not all regional
models or a core promotion. The user assigned broader original census and
pandas/generator coverage to separate cloud work. Preserve source pins and merge
those changes only through isolated integration and the relevant Windows gates.

## Experiment update — native cache hits (2026-11-07)

**Superseding direction:** the user requires automatic selection with no
product-specific performance settings, and defers broader model validation
until those models are adjusted. The current experiment implements one common
policy for adjusted MYGA/VA: observed physical windows with exact spill and a
storage-cost check, signature-based numeric stores, observed primary sizing,
guarded math/memo reuse, and scalar/already-proven dense cache hits. Both build
specs contain only worker count and the same independent-policy/stable-Reference
reuse contract; neither has per-Cell tuning. Pickling regressions exposed by
automatic native storage are fixed using normal Cython serialization and
published numeric-cache state. See `../september-gap/AUTOMATIC_POLICY_PLAN.md`.
Eight final paired rounds meet the 2x September gate: MYGA 1.895x and VA 1.910x
median ratios, with upper mean confidence limits below 2x. All one/four-worker
direct/collect comparisons improve against the frozen control. Windows 3.12
and 3.14t each pass 108 candidate/helper regressions. Each untouched suite has
169 passes, seven skips and two failures at old generated-source assertions;
their unchanged threaded/financial runtime scripts pass separately. These
failures are disclosed and the untouched suite is not labelled all-green.
Final native attribution is recorded in the automatic report. The following
paragraphs describe the preceding manually selected diagnostic candidate.

On `experiment/collect-15x`, the one-worker September comparison is recorded in
`../september-gap/ATTRIBUTION.md`. Eight paired prototype rounds meet the 2x
collect/API gate: MYGA 1.895x and VA 1.963x median ratios. Integrated generated
programs and all native executable sections match the qualified prototypes.
The retained mechanisms are existing range/receiver proofs and an opt-in
`native_cache_hits` lowering. Scalar and dense modes are separate because
checked dense hits regress MYGA. The split-miss experiment is rejected.

Windows Python 3.12 and free-threaded 3.14 each pass 171 untouched tests and
99 candidate/helper tests; the final 25 admission tests also pass on both.
Original MYGA/BasicTerm_SC defaults retain generated/native code identity and
pass their output contracts at 1/4 workers. Full adjusted report, policy and
transition checks pass. Final integrated timing and repeated attribution are
tracked in the same evidence directory. This is an experimental candidate;
`core` remains unchanged and promotion still requires its separate approval.

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
