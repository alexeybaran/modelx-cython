# Codex Desktop Setup — Native Windows

This project is intentionally moving away from handover-package development. After the initial import, Git branches plus reproducible evidence are the project state.

## 1. Install the native Windows toolchain

Recommended starting environment for reproducing the retained Linux campaign as closely as practical:

- Windows 11 x64
- current Codex Desktop / Codex CLI
- Python 3.13 x64 (start with the same major/minor as retained Linux evidence)
- Visual Studio Build Tools 2022 with **Desktop development with C++**, MSVC x64/x86 build tools and a Windows SDK
- Git for Windows

OpenAI's current Codex Windows guidance includes `codex doctor`; run it before project work.

Create a short local path, for example:

```powershell
C:\src\modelx-cython-perf
```

Avoid OneDrive/network-synced folders and unusually long paths while validating Cython/MSVC builds.

## 2. Extract and open the project

Extract the migration package, then open the contained `modelx-cython` directory as the Codex project.

The repository already contains `AGENTS.md`; **do not replace it with `/init` output**. OpenAI documents `/init` as a way to generate an `AGENTS.md` scaffold, but this project already has one written for its evidence-gated workflow. If you want to inspect `/init`, do it in a disposable copy and compare rather than overwrite.

Run:

```powershell
git status
git branch --all
codex doctor
```

Expected initial branch: `core`.

Verify compiler-source identity:

```powershell
git diff upstream-pristine..core -- modelx_cython
```

It should be empty. `core` initially differs from upstream only by project governance/docs/tooling.

## 3. Create the Python environment

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Then install the frozen project dependencies using `project_refs\vendor\WINDOWS_REQUIREMENTS.md` as the version guide. Prefer the bundled `modelx`/`lifelib` source snapshots for baseline reproduction rather than silently switching to newer releases.

Before benchmarking, capture:

```powershell
python --version
python -m pip freeze
cl
where.exe cl
where.exe python
```

Store the output under a new Windows evidence run directory.

## 4. Use Goal for the durable outcome

OpenAI's current Codex guidance describes Goals as persistent objectives for work such as profiling and benchmark-driven optimization where the path depends on evidence.

Use the project Goal in `CODEX_GOAL.md` (or paste its `/goal` command). Keep **one durable project Goal**, rather than creating a new Goal for each optimization idea.

The Goal answers: **what must be true before this project is complete?**

## 5. Use Plan mode for each experiment

Use Plan mode before every nontrivial compiler experiment. A plan should say:

1. hypothesis / mechanism;
2. current-core control;
3. branch/worktree to use;
4. source files expected to change;
5. correctness tests;
6. benchmark workload and timing boundary;
7. expected material benefit;
8. rejection/rollback condition.

Plans answer **how should we test this idea?** Goals answer **what final outcome are we pursuing?**

Do not use Plan mode as permission to change `core`. Implementation occurs on an experiment branch. Promotion remains a separate evidence review and requires user approval.

## 6. Recommended branch/worktree pattern

Keep `core` clean and create a worktree per experiment:

```powershell
git switch core
git worktree add ..\wt-native-cache -b experiment/native-cache core
```

Run the experiment and evidence there. If rejected:

```powershell
git worktree remove ..\wt-native-cache
```

Keep the branch/report if the negative evidence is useful; delete throwaway build artifacts.

For helpers, use `feature/helpers` as a parallel track rather than merging it into core pre-emptively.

## 7. First Codex task: verify Windows before development

Use `INITIAL_CODEX_PROMPT.md`. The first task must not optimize the compiler. It must:

- verify native Windows build tools and dependencies;
- run pristine upstream CI;
- restore/build supplied controls;
- validate original/adjusted MYGA;
- rerun the benchmark matrix;
- compare Windows results with retained Linux evidence without pooling timings;
- recommend which exact change, if any, should be promoted first.

## 8. Performance discipline on Windows

For accepted timing:

- do not run builds/tests in parallel with benchmarks;
- keep numerical-library thread counts fixed;
- record CPU model, power mode and toolchain;
- use the same policy/output workload across compared branches;
- interleave control and experiment trials where practical;
- keep raw trial measurements, not only medians;
- validate outputs before timing;
- distinguish calculation from report/lifecycle costs.

Windows timings need not numerically match Linux. The important comparison is contemporaneous **within the same Windows campaign**.

## 9. Core promotion workflow

After an experiment, Codex should produce a short evidence report with:

- result: promote / reject / continue investigation;
- correctness/CI status;
- paired performance result and uncertainty/noise assessment;
- source delta/complexity;
- generated-code observation if relevant;
- remaining risks.

If promotion is recommended, Codex must stop and ask the user to approve. Only after approval should the smallest useful delta be cherry-picked/rebased onto `core` and the complete gate rerun there.

## 10. Side chats and long-running work

Keep the main Codex thread responsible for the active Goal/experiment. Use side chats for explanatory questions so the main engineering transcript remains focused. OpenAI's current guidance explicitly separates Plan (path) from Goal (outcome) and recommends side chats for questions that should not redirect ongoing work.

## Current OpenAI references

- Codex with ChatGPT / Windows / `codex doctor` / `/init`: https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan
- Goals: https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex
- Plan + long-horizon project files: https://developers.openai.com/blog/run-long-horizon-tasks-with-codex
- Plan vs Goal / side chats: https://developers.openai.com/blog/mastering-codex-remote-for-engineering
- `AGENTS.md` behavior: https://developers.openai.com/api/docs/guides/latest-model

## 11. Git remotes after Stage 0

Start with the packaged **local repository only**, as requested. Do not make GitHub connectivity part of the first Windows reproduction. After Stage 0 is stable, add your fork as `origin` and `fumitoh/modelx-cython` as `upstream`, fetch both, and verify that the bundled pristine baseline corresponds to the intended upstream source before preparing any eventual PR. Keep project-only governance/evidence commits separate from compiler changes so an upstream PR can be constructed from the minimal promoted compiler delta rather than the whole project-history branch.
