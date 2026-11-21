# Round 3 fork release verification

The workflow and `fork_ci/` are fork verification files. Keep them outside the three product series. The starting configuration is the historical fork workflow at `51f1455`. GitHub records ten completed successes in that run, including the unit and integration steps on Linux 3.9 and macOS 3.13, 3.14 and 3.14t. That run used older product commits and installed `modelx[helpers]`. Final helper-free evidence has a different test inventory. [Historical Actions run](https://github.com/alexeybaran/modelx-cython/actions/runs/44467159922)

## Required jobs

| Profile | Runners and versions | Evidence |
| --- | --- | --- |
| Release | Linux, macOS and Windows on CPython 3.13, 3.14 and 3.14t | Nine helper-free jobs with all three products installed from their exact commits |
| Cython floor | Linux and Windows on standard CPython 3.14 | Cython 3.2.4 exactly and all three products installed |
| Compatibility floor | Ubuntu 24.04 on CPython 3.9 | pandas 2.2.2 exactly with compiler and modelx installed. The exact lifelib commit is checked out and its incompatible `Requires-Python` recorded |
| Legacy helper compatibility | Linux on CPython 3.14t with the GIL disabled | The same three plain product installs plus the isolated `modelx_helpers` package from `ac6fc71fa15b0041c0d22eb281de21eefa05e3b4` |

These jobs cover the signed interpreter, runner and dependency lanes. The release jobs resolve compatible versions for the chosen interpreter and record the actual package versions. All jobs and inventory generation use pytest 8.4.2 because the default identifiers of some parametrizations vary between pytest releases. The pinned wheel declares Python 3.9 or later and produced byte-identical inventories on Windows standard and free-threaded CPython 3.14.8. The Windows and Linux Cython floor jobs use standard 3.14 so the floor can be built in both toolchains. The pandas 2.2.2 floor uses the supported Linux 3.9 wheel. The three Python versions between 3.9 and 3.13 are outside this matrix. [Delivery plan](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/redev/round-3/parallel/20261113/round3/DELIVERY-PLAN.md)

`fork_ci/exclusions.json` identifies cases that require the optional legacy `modelx_helpers` package. Module-level exclusions cover direct imports or a shared module fixture. Test-specific exclusions leave the independent cases in normal coverage. The Linux 3.9 lane also excludes lifelib-dependent cases because the final lifelib declares `Requires-Python >=3.10`. `fork_ci/source-coverage.json` records both complete default suites at the reviewed compiler commit and the seven default opt-in test IDs (six fuzz cases and one stress case). Each job reports its exact selected IDs, named exclusions, actual pytest deselections, individual skips and per-suite counts. A successful 3.9 job supports the compiler and modelx scope stated in the receipt.

The legacy compatibility job runs the exact legacy scope omitted from the helper-free lanes. It checks out the historical helper revision, copies only `modelx_helpers` into an isolated import directory, and verifies both the legacy import and the installed final modelx import. The collector requires the exact legacy IDs in `OPTIONAL-COVERAGE.tsv`, 87 unit and 731 integration cases, with zero skips. These include all 705 table matrix cases. Those cases use the legacy helper fixture in live, exported and compiled modes. The matrix declaration check runs in the helper-free lanes.

## Preflight and trigger

Use the dedicated branch `verify/round3-release` in `alexeybaran/modelx-cython`. Start the verifier branch from an isolated tree containing only the verifier files so it carries no other workflows. This verifier triggers when a push on that branch changes `fork_ci/pins.json` or the verifier workflow.

After approval, put the final 40-character compiler, modelx and lifelib SHAs into `fork_ci/pins.json` under keys `compiler`, `modelx` and `lifelib`. Verify remote reachability and exact SHA for each pinned checkout, including the legacy helper checkout. Regenerate the inventories from the final compiler checkout and review the resulting branch commit before pushing. The verifier rejects an inventory recorded at a different compiler SHA. Retain the verifier commit SHA separately from the product pins.

```sh
set -eu
python fork_ci/verify.py pins > reviewed-pins.txt
while IFS='=' read -r package sha; do
  repo="$package"
  if [ "$package" = compiler ]; then repo=modelx-cython; fi
  test "$(gh api "repos/alexeybaran/$repo/commits/$sha" --jq .sha)" = "$sha"
done < reviewed-pins.txt
legacy=ac6fc71fa15b0041c0d22eb281de21eefa05e3b4
test "$(gh api "repos/alexeybaran/modelx/commits/$legacy" --jq .sha)" = "$legacy"
git push https://github.com/alexeybaran/modelx-cython.git HEAD:refs/heads/verify/round3-release
```

The jobs use standard GitHub-hosted runners. For a retry, rerun the entire workflow so the collector can verify one attempt containing all 13 jobs and their attempt-specific artifacts. A skipped or cancelled job is a failed evidence gate.

In run `44889555465` attempt 1, the macOS job selected and installed the Darwin arm64 free-threaded CPython 3.14.10 archive. Its certificate installer then launched a standard Python with the job-wide `PYTHON_GIL=0` and failed during preinitialization. The workflow sets `PYTHON_GIL` only when it executes the selected interpreter. The install and test verification still checks the actual ABI and GIL state. [macOS 3.14t job](https://github.com/alexeybaran/modelx-cython/actions/runs/44889555465/job/136802181648)

All four Windows jobs reached both test suites. The fresh runners lacked the parent of the configured pytest temporary paths. The verifier now uses flat paths directly under `C:/`, keeping path lengths short for MSVC. The Windows logs also exposed the helper dependency of the 235 table matrix squares. Every square is declared complete in each of the three modes. The compatibility job requires all 705 results with zero skips. [Windows 3.13 job](https://github.com/alexeybaran/modelx-cython/actions/runs/44889555465/job/136802181819)

## Collect a completed attempt

Use the run ID and attempt number shown by GitHub. Keep the API responses and extracted artifacts beside the summary as the evidence for that attempt.

```sh
fork=alexeybaran/modelx-cython
gh api "repos/$fork/actions/runs/$run_id/attempts/$attempt" > run.json
gh api "repos/$fork/actions/runs/$run_id/attempts/$attempt/jobs?per_page=100" > jobs.json
gh run download "$run_id" -R "$fork" --pattern "junit-*-attempt-$attempt" --dir artifacts
python fork_ci/collect.py --run-json run.json --jobs-json jobs.json --artifacts artifacts \
  --run-id "$run_id" --attempt "$attempt" --verifier "$verifier_sha" \
  --compiler "$compiler_sha" --modelx "$modelx_sha" --lifelib "$lifelib_sha" --output summary.json
```

A passing summary requires the full matrix, successful named steps, matching source and attempt receipts, exact interpreter and GIL checks, dependency-floor evidence, and complete outcomes for both suites. The collector checks every selected test identity against the source inventory and JUnit cases and recomputes the outcome counts and file hashes. [Run attempts API](https://docs.github.com/en/rest/actions/workflow-runs#get-a-workflow-run-attempt), [artifact download command](https://cli.github.com/manual/gh_run_download)

## Local verifier checks

```sh
python -m unittest discover -s fork_ci -p 'test_verifier.py' -v
```

Run the inventory command with the exact installed products and isolated legacy helper import available. It records the full default collection, opt-in cases, fixture ports, remaining legacy tests and lifelib cases outside the scoped Python 3.9 lane.

```sh
python fork_ci/audit_optional_scope.py "$compiler_checkout" OPTIONAL-COVERAGE.tsv \
  --source-coverage fork_ci/source-coverage.json --historical-junit "$historical_linux_free_threaded_artifact"
```

The historical Linux 3.14t JUnit artifact has 1,018 unit results with eight skips and 758 integration results with two skips. Its Linux 3.9 artifact has the same totals, with ten unit skips and two integration skips. Both artifacts have zero failures and errors. `OPTIONAL-COVERAGE.tsv` records exact matching historical outcomes for the affected cases. A missing historical identity is shown separately from a prior pass. [Historical Actions run](https://github.com/alexeybaran/modelx-cython/actions/runs/44467159922)

Local Windows CPython 3.14.8t checks used modelx `175ee3f`, compiler source `7215c30` for the final test edits, and an isolated copy of the pinned helpers for the legacy runs. The initial legacy run passed all 87 unit cases and 22 of 26 integration cases. Four integration assertions still expected retired helper-only lowering. The table below records those expectations and their current equivalents. All six cases in those four modules passed after the edits, with no skips. The eight helper-free pandas checks also passed, exercising native integer keys, release value and exception behavior, native holds, fallbacks, KR formulas and MultiIndex extrema.

| Legacy test | Previous generated assertion | Verified generated path |
| --- | --- | --- |
| `test_forms_compile_natively_and_equal_the_interpreter` | Three-part helper-key proof in `table_reads_keys` | Tuple construction and Python helper member and row reads, alongside the existing native sum checks |
| `test_native_table_index_answers_as_the_helper` | `has_1` helper prefix dispatch | Live helper prefix and direct index reads, with the existing explicit NativeTable index check in its own case |
| `test_reduction_key_of_a_lowered_table_read_builds_and_matches` | An untyped reduction decline due to inserted Cython syntax | The lifted reduction result goes to the helper row lookup. The build, numerical and exception checks remain |
| `test_w1_9_forms_live_exported_compiled` | Legacy helper handle methods in generated code | Python helper reads in generated code and correct helper and pandas source reasons in the admission report |
