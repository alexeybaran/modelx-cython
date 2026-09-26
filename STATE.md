# Current Project State at Desktop Migration

## Accepted compiler core

`core` is the accepted project branch. At migration time its **`modelx_cython/` source is byte-identical to pristine bundled upstream** `a6e4b8c062b3b9cdfd3b97577274fc8d44432dce`. It adds project governance/docs/tooling only.

No H07/H08/H09/H10 compiler mechanism is accepted merely because it is later in history.

## First promotion candidate

`feature/helpers` contains the exact compiler source of the successful Linux `upstream + minimum helper support` control, rebased on the project-governance commit. Its compiler source is identical to the exact evidence commit `experiment/helpers-exact-linux-control` (`c7fd7137d89403583c570502da5b9305e84603c3`).

Status: **candidate, not promoted**. Native-Windows CI/correctness/performance reproduction is required, followed by user approval.

Historical same-host Linux result:

- upstream + minimum helpers: ~2.642 ms/policy adjusted-MYGA calculation;
- H09 broader compatible base + helpers: ~2.626 ms/policy;
- paired result showed no material H09 structural advantage in that campaign.

Use the raw control evidence under `project_refs/min_helpers/`; do not compare those absolute times to a different campaign.

## Reference branches

- `reference/h09-compatible` — H09/H08B compiler state; useful for ablation/reference only.
- `reference/h08-compatible-core` — prior compatible-core state.
- `experiment/helpers-exact-linux-control` — exact source tied to retained Linux helper-control evidence.

## Donors / external references

Ignored `project_refs/` carries the fast prototype and previous optimized compiler sources, H09 exact cache map/harness/evidence, models and frozen dependency sources.

H10/3A source is not bundled, but its negative performance/correctness evidence and exact archive hashes are retained under `project_refs/h10/`.

## Operating rule

Investigate first. Promote second. `core` changes only after the complete evidence gate and explicit user approval.
