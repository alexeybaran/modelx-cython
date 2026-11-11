# Phase 2 implementation: generic backward as-of lookup lowering

## Goal

Make ordered "latest key at or before query" table access a first-class semantic
operation, then lower it through the extension's existing native step-lookup
runtime. The implementation is generic: no MYGA, product, formula, table or column
names are present in the new compiler logic.

Phase 2 deliberately does not add Table IR or DataFrame materialization/fusion.
Those remain the next phase.

## Implemented

### 1. Backend-neutral `AsofLookupIR`

`modelx_graph.semantic_ir.AsofLookupIR` represents:

- source table expression;
- finite equality-prefix selector(s);
- ordered query;
- value-column selector;
- direction (`backward` in Phase 2);
- exact-match policy;
- missing-domain behavior.

The semantic node is independent of pandas storage and Cython code generation.

### 2. Generic source recognition

`modelx_graph.lookup_semantics.recognize_asof_lookups` recognizes the bounded
Python/pandas idiom:

```python
sub = table_expr.loc[prefix]
keys = [k for k in sub.index if k <= query]
return float(sub.loc[max(keys), value_selector])
```

and converts it to `AsofLookupIR`. The recognizer does not inspect or match any
product, formula, provider, table or column name.

### 3. Semantic-to-physical lookup handoff

The structured-provider normalization stage consumes `AsofLookupIR` and reuses the
existing prepared `NormalizedLookup1D` / native step-lookup machinery. Finite
prefix selectors and finite value-column selectors are expanded into a bounded
native dispatch over prepared numeric lookup leaves.

This keeps one physical lookup implementation rather than creating a second
MYGA-specific runtime.

### 4. Source-equivalent missing-domain guard

The source idiom uses `max(keys)` and therefore raises when no ordered key is less
than or equal to the query. Phase 2 preserves that contract.

If range evidence proves the query cannot precede the first anchor, no dynamic
check is needed. Otherwise semantic lowering inserts a guard before the native
step lookup. The existing guard mechanism records failure inside `nogil`; the
Python boundary raises after native execution returns.

The guard threshold is selected per finite prefix branch, so grouped tables may
have different first ordered keys.

### 5. Generic range-bound evidence

Selected-closure lower-bound propagation now understands coordinates introduced by
`range(...)`, including common local aliases and comprehensions, and propagates
those bounds through direct helper calls. This is generic supporting evidence for
ordered-domain proofs; Phase 2 does not require such a proof because the runtime
guard preserves semantics when proof is incomplete.

### 6. Late semantic guard normalization

Semantic lowering may introduce guards after source-level guard normalization.
`FormulaNormalizationResult` now carries its authoritative guard set, and the fast
backend normalizes semantic guards before physical classification/code generation.

## MYGA validation effect

On the unchanged full seven-policy MYGA census output:

- Phase 1 selected closure: **52 N0 / 45 N1 / 3 P**;
- Phase 2 selected closure: **99 N0 / 0 N1 / 2 P**;
- `scenario_rate`: **P -> N0**;
- 45 dependent cells: **N1 -> N0**;
- remaining Python-backed selected cells: `result_cf`, `census_output` only;
- live numerical max absolute error: **0**;
- detached/post-model-close max absolute error: **0**.

Same-session calculation timing:

- Phase 1: **105.605 ms per policy-scenario**;
- Phase 2: **1.06490 ms per policy-scenario**;
- calculation speedup: **99.17x**;
- reset + calculation speedup: **90.91x**.

The speedup is evidence for this MYGA workload, not a generic performance promise
for every ordered lookup.

## Intentionally deferred

Phase 2 does not make pandas/DataFrame results native. `result_cf` and
`census_output` remain Python-backed because the selected output constructs and
then reduces a pandas DataFrame. The next structural phase should add a small typed
Table IR, demand propagation and aggregate fusion rather than compiling general
pandas syntax.
