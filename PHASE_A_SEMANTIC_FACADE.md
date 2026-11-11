# Phase A — Semantic facade

## Goal

Introduce a single behavior-preserving semantic contract over the current fast
compiler without changing execution, memoization, code generation, or admission.
This is the migration foundation for the category, ranked-axis, and invocation
cleanup phases.

## New contracts

`modelx_graph.semantic_contract` defines:

- `ValueSpec`: logical dtype, category-domain identity, and logical axes.
- `AxisDomain`: semantic axis identity/provenance. Equal length alone never
  establishes identity.
- `CellSignature`: semantic parameter/result contract for one canonical Cell UID.
- `LoweringPlan`: current physical execution/ABI/memo facts, deliberately kept
  separate from `CellSignature`.
- `SemanticFacadePlan`: deterministic per-plan collection of the above.

`ModelxCythonFastPlan.build_semantic_facade()` derives these contracts from the
existing frontend rows, source-transparent Cell catalog, category plan,
ranked-output evidence, and N0/N1/P execution-class plan.

## Important semantic separation

A finite category is represented semantically as:

```text
ValueSpec(logical_dtype="category", category_domain_id="cat_...")
```

while its existing physical lowering can remain:

```text
LoweringPlan(parameter_abi_dtypes=("int64", ...))
```

Likewise a ranked value owns explicit `AxisDomain` objects. The selected traced
ranked output receives the authoritative point axis. Other ranked Cells are
assigned Cell-local `unresolved` axes even when their lengths happen to match;
Phase A intentionally refuses to turn shape equality into semantic identity.

## Fail-closed consistency checks

Fast-plan construction now derives the facade once before returning. It raises
`SemanticContractError` if any of these existing representations disagree:

- frontend Cell vs source-transparent Cell name/fullname;
- parameter names/dtypes/category domains;
- result dtype/category domain;
- logical rank/shape;
- ranked selected-output dtype/shape/axis keys;
- execution-class parameter/result ABI facts;
- category semantic value vs current int64 category ABI;
- category-domain IDs vs the current category plan.

The same facade is emitted under `semantic_facade` in the fast-plan manifest.
Code generation does not consume it yet.

## Explicit non-goals

Phase A does not:

- change N0/N1/P classification;
- change category inference or encoding;
- infer shared axes for equal-sized arrays;
- alter recursive ABI, Python-key helpers, memo policy, or GIL behavior;
- fix p34, p10, p45/p46, or p03.

Those later phases should migrate existing consumers to this semantic layer one
at a time and delete the corresponding legacy representations only after
behavioral parity is proven.
