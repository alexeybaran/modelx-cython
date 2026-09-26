# Project tools

Small workflow helpers only. They are project tooling, not compiler mechanisms.

- `windows_preflight.ps1` — verify native Windows/Codex/MSVC and pristine-core identity.
- `new_experiment.ps1` — create an isolated `experiment/*` worktree from `core`.
- `capture_environment.ps1` — write a Windows environment receipt before benchmarks.

The retained H09 benchmark/build harness lives under ignored `project_refs/h09/tools/` because it is reference infrastructure, not accepted compiler code.
