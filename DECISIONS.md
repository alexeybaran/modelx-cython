# Approved project decisions

## D1 — donor-boundary decision

Transfer individually justified cache/runtime mechanisms into existing modelx-cython execution. Do not automatically import the prototype/previous compiler's private execution frame, prepared-policy ABI, batch runtime, formula duplication or custom exception machinery. Each additional donor mechanism requires separate measured justification and approval.

## Operating model

- Native Windows is the primary environment after migration.
- The accepted compiler starts from pristine upstream modelx-cython.
- `upstream + minimum helpers` is the first **promotion candidate**, not an accepted core.
- Helper development stays on a parallel feature/experiment branch and is used for adjusted-model performance testing.
- Only changes with demonstrated material performance benefit or necessary generic correctness/coverage/functionality may enter core.
- Performance-neutral structural refactors do not enter core merely because they appear cleaner.
- Structural compiler promotion requires explicit user approval after evidence is presented.

## Integration contracts — 2026-11-09

- The user permits independent exceptions to be raised in a different order.
  First-error order alone does not justify blanket numeric argument boxing.
- The user permits native float results in place of integers when the numbers
  do not change (for example 0.0 instead of 0). Restore upstream/c8 numeric
  widening rather than introducing a new mixed-scalar storage mechanism.
  Observed integers that cannot round-trip through double retain object storage.
  Explicit object result declarations still retain their scalar-class contract.
  This decision does not authorize product-specific tuning or core promotion.
- Historical H09/H10 code/results are controls and lessons, not mandatory ancestry.
