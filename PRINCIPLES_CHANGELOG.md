# Guiding-principles review at Desktop migration

The previous governing document had 67 numbered principles accumulated over many handover iterations. The new canonical `GUIDING_PRINCIPLES.md` reduces them to **15 durable principles plus the definition of success**.

This is a simplification, not a relaxation. Procedural detail was moved to `PLAN.md` and `TESTING.md` so the governing rules stay memorable.

Key changes:

1. **Explicit experiment/core separation.** A completed implementation is not presumed to be the next base.
2. **Material-benefit promotion gate.** Performance-neutral structural refactors no longer qualify by default. Necessary generic semantics/coverage can still qualify.
3. **Explicit user approval before structural promotion.** Codex may investigate and benchmark independently but must stop before integrating a proposed structural compiler change into `core`.
4. **Upstream-first reset.** Pristine modelx-cython becomes accepted compiler source; H09/H10 are references rather than ancestry.
5. **Helpers become a parallel feature track.** Adjusted models may use them for performance research, but compiler prerequisites must independently earn core promotion.
6. **Object-capable Cython remains the normal compatibility path.** Actual-Python bodies are exceptional, not a coequal performance architecture.
7. **D1 retained.** Donor cache/runtime mechanisms do not implicitly authorize larger donor architecture.
8. **Native Windows becomes the new execution authority** after migration; historical Linux results remain comparison evidence.
9. **End target clarified.** Approach previous-compiler calculation performance on lifelib workloads where that compiler achieved native execution, while balancing complexity and Cython/modelx-cython architectural boundaries.
10. **Handover rules removed from the durable core principles.** After migration, Git + evidence + Codex project files replace iterative ZIP handovers.

The previous detailed principles remain recoverable in earlier H09/H10 project artifacts, but they are superseded for ongoing Desktop Codex development by the concise canonical file in this repository.
