# Modelx-Cython performance project

This local repository is the Desktop Codex continuation of the modelx-cython performance work.

**Compiler-source authority:** pristine bundled modelx-cython (`a6e4b8c`) until an experiment is explicitly promoted.

Read `GUIDING_PRINCIPLES.md`, `TARGET_ARCHITECTURE.md`, `PLAN.md`, and `AGENTS.md` before development.

Key branches:

- `upstream-pristine` — immutable bundled upstream reference.
- `core` — evolving accepted project core; initially compiler-source-identical to upstream.
- `experiment/helpers-exact-linux-control` — exact minimal-helper Linux ablation commit (`c7fd713`), retained to match existing evidence.
- `feature/helpers` — helper experiment rebased onto project governance; parallel feature branch.
- `reference/h09-compatible` — prior H09/H08B compiler state for comparison only.
- `reference/h08-compatible-core` — prior compatible-core reference.

Large donor/model/evidence material is in ignored `project_refs/`; it is available to Codex but is not part of compiler Git history.

## Linux pre-migration evidence

The exact minimal-helper control completed the full local suite: 171 untouched upstream tests passed with 7 original skips, plus 2 helper tests passed. Native Windows must reproduce the relevant gates before promotion. H10/3A exact source is not bundled; only its negative evidence/archive identity is retained.
