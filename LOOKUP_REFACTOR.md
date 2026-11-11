# Phase 2 lookup physicalizer refactor

## Goal

Remove duplicated StepLookup1D planning introduced by Phase 2 without changing
lookup semantics or the native runtime ABI.

## Structure after refactor

```text
legacy pandas step idiom ──┐
                           ├─> shared StepLookup1D physicalizer
AsofLookupIR ──────────────┘       ├─ prefix planner
                                   ├─ column planner
                                   ├─ prepared leaf builder
                                   └─ finite dispatch emitter
                                            │
                                            v
                                  existing __step_lookup_1d__
```

The semantic recognizers remain separate because they recognize different source
contracts.  Everything after semantic recognition is shared.

## Shared helpers

Within `GraphModelCompiler._normalize_external_tables` the refactor introduces:

- `_finite_step_integer_parameter`
- `_plan_step_lookup_prefix`
- `_step_lookup_row_choices`
- `_plan_step_lookup_columns`
- `_step_lookup_leaf_call`
- `_emit_step_lookup_dispatch`
- `_rewrite_step_lookup_result`

There is exactly one call to `registry.external_dataframe_step_lookup` in the
normalizer.  Neither `_rewrite_step_lookup_function` nor
`_rewrite_semantic_asof_lookup_function` allocates physical lookup leaves directly.

## Validation

Focused tests: 9/9 passed.  A structural regression test enforces one physical
StepLookup1D leaf builder and proves both frontend paths use the shared planner.

A full MYGA plan build on the current lifelib/modelx sources retains:

- selected closure: 99 N0 / 0 N1 / 2 P;
- `scenario_rate`: N0;
- `AsofLookupIR` branch count: 6;
- branch axis minima: 0/0/0;
- missing policy: native runtime guard;
- remaining P cells: `result_cf`, `census_output`.

The only intended normalized-operator metadata change is provenance now includes
`shared_step_lookup_physicalizer`.

The refactored compiler completed an optimized full seven-policy MYGA build and
15-trial run at **1.0517427 ms per policy-scenario**, with zero live error and zero
detached/post-model-close error.  The prior Phase 2 handoff recorded
**1.0648997 ms**, so the refactor retains Phase 2 performance; the small difference
should be treated as normal run/environment variation rather than a claimed speedup.
Preparation was 50.77 s (plan 28.50 s, native compile 21.30 s).
