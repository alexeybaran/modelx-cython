# Codex instructions for this repository

Read, in order:
1. `GUIDING_PRINCIPLES.md`
2. `TARGET_ARCHITECTURE.md`
3. `PLAN.md`
4. `DECISIONS.md`

## Core rule

Never modify or merge into `core` merely to try an idea. Use an `experiment/*` or `feature/*` branch/worktree. A compiler change enters `core` only after evidence shows material performance benefit or necessary generic correctness/coverage/functionality **and the user explicitly approves promotion**.

## Before a compiler experiment

State:
- the hypothesis and donor/mechanism;
- current-core control;
- exact correctness/benchmark workload;
- expected material benefit;
- source area to change;
- rollback/rejection condition.

Use Plan mode for nontrivial experiments before editing code.

## Evidence before promotion

For a compiler change proposed for promotion:
- validation before timing;
- untouched upstream modelx-cython CI;
- focused regressions;
- original-model non-regression;
- relevant adjusted/native-target correctness;
- contemporaneous paired benchmark against current core;
- generated-code inspection when nativeness/call/cache shape is the hypothesis;
- source-delta and complexity review;
- preserve negative evidence.

If the benefit is unclear or immaterial, recommend rejection. Do not protect sunk cost.

## Boundaries

- Preserve modelx-cython Space/Cell/public compiled-package structure unless separately approved.
- Stay Cython based; do not add a second formula/pandas interpreter.
- Use object-capable Cython for general Python-object semantics.
- Do not add product/Cell-name-specific compiler branches.
- Treat helpers/model adaptations as a parallel feature track; compiler prerequisites must independently earn promotion.
- D1 applies: do not import private donor frames, prepared-policy ABIs, batch runtime, formula duplication or custom exception machinery without separate measured justification and user approval.
- Do not silently alter tests, policy sets, output unions, timing boundaries, principles, architecture or acceptance gates.

## Environment

Native Windows is the primary environment after migration. Historical Linux evidence is reference evidence only. Platform-specific fixes belong on experiment/control branches until they pass the same promotion gate.

## Project state

`core` begins with compiler source identical to pristine upstream plus project-governance documentation. `experiment/helpers-exact-linux-control` is the exact Linux control commit. `feature/helpers` is the rebased helper experiment branch. `reference/*` branches are research references, not production ancestry.

After migration, use Git + evidence files rather than creating new handover ZIPs.
