# Phase 1 implementation: semantic IR and pass boundary

## Goal

Create a small, backend-neutral semantic compiler boundary before adding generic
`AsofLookup` and table IR capabilities. Phase 1 must not add product-specific
rules or change existing native/Python admission.

## Implemented

- Added `modelx_graph.semantic_ir`:
  - `SemanticFormulaIR`: immutable semantic formula snapshot;
  - deterministic AST fingerprinting;
  - ordered `SemanticPassRecord` evidence;
  - `FormulaNormalizationResult` separating semantic IR from existing physical lowering.
- Added `modelx_graph.semantic_passes`:
  - `AstSemanticPass`;
  - `SemanticPassPipeline` with deterministic ordering, duplicate-name rejection,
    structural result validation, and copy-on-pass isolation.
- Routed the fast scalar normalization path through two explicit semantic stages:
  1. source-shape normalization (finite vectors, bounded sequences, static sequence/category reductions);
  2. provider semantics (vector subscripts, finite mapping iteration, structured providers/tables).
- Category binding/encoding is recorded as an explicit semantic transition while
  retaining its existing implementation and semantics.
- Demand-driven authoritative evidence now starts from the semantic-normalized AST
  when available; raw source is provenance/fallback rather than a second semantic interpretation.
- `ModelxCythonFastPlan` now retains `semantic_formula_ir` and exposes deterministic
  semantic-pass diagnostics in its manifest.
- No MYGA/product names or product-specific branching were added.

## Validation

Focused pass-framework tests: **4 passed**.

Differential MYGA planning using the untouched standalone source versus Phase 1:

- output execution class: identical;
- execution-class plan: identical;
- selected closure: **52 N0 / 45 N1 / 3 P** in both;
- normalized operators: identical;
- parameter ranges: identical;
- fallback reasons: identical;
- Phase 1 semantic IR snapshots: **128** formulas.

This validates the intended Phase 1 contract: structural compiler separation with
no native-coverage or fallback change.

## Intentionally deferred

Phase 1 does **not** implement `AsofLookup`, table IR, column demand propagation,
aggregate fusion, or pandas materialization boundaries. Those should build on this
semantic boundary in subsequent phases.
