# Experiment: <name>

## Hypothesis

What specific mechanism should improve what measured cost, and why?

## Control and branch

- accepted core commit:
- experiment branch/worktree:
- donor/reference mechanism:

## Workload and correctness contract

- model/output/policies:
- tracing domain:
- validation domain:
- timing boundary:

## Expected benefit / rejection condition

State what would count as material benefit relative to complexity. State what result means the experiment should be rejected.

## Source delta

Files/lines/mechanism changed. Explicitly state anything intentionally not transferred from the donor.

## Results

- untouched CI:
- focused tests:
- validation:
- paired raw benchmark:
- generated-code observation:
- complexity/maintenance impact:

## Decision

`PROMOTE_CANDIDATE` / `REJECT` / `CONTINUE_INVESTIGATION`

Explain evidence and uncertainty. If promotion is recommended, stop for user approval before integrating into `core`.
