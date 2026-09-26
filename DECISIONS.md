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
- Historical H09/H10 code/results are controls and lessons, not mandatory ancestry.
