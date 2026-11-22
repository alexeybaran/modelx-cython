# Helper-free compiler verification

The verifier branch keeps workflow files outside the compiler product series.
Each hosted job installs the pinned compiler and modelx checkouts. Except for
the Python 3.9 floor, it also installs the pinned lifelib checkout. Jobs verify
source SHAs, interpreter, ABI, GIL state, dependency floors and JUnit identities.
The legacy helper package is absent from every lane.
Jobs with lifelib also verify same-named replacement CSVs in the Korean medical
and MYGA readers and cached list identity in UK Income Protection.

| Profile | Runners and interpreters | Coverage |
| --- | --- | --- |
| Release | Linux, macOS and Windows on CPython 3.13, 3.14 and 3.14t | Both default suites across the nine combinations |
| Cython floor | Linux and Windows on regular CPython 3.14 | Full suites with Cython 3.2.4 |
| Compatibility floor | Ubuntu 24.04 on CPython 3.9 | pandas 2.2.2 and compiler and modelx tests. Eight cases require lifelib's higher Python floor |
| Retained pandas | Linux on free-threaded CPython 3.14 with the GIL disabled | Explicit coverage for pandas rows, key inference, fallback, return identity, enums, arithmetic, math and builtin reductions |

The retained pandas lane runs a named subset of the default suites and requires
all selected cases to pass. The source inventory uses pytest 8.4.2. The reports
record selected cases, exclusions, opt-in cases and runtime dependencies.

## Exact pins and trigger

The workflow runs on `verify/r3-helper-free-qualification`. Push a reviewed
change to the manifest or workflow on this branch to start the 13 jobs. The
`fork_ci/pins.json` manifest pins the test subject and companions separately
from the verifier. Confirm `python fork_ci/verify.py pins` accepts the exact
compiler inventory before starting a new run. Keep retry evidence from a
single attempt.

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

## Local checks

```sh
python -m unittest discover -s fork_ci -p 'test_verifier.py' -v
python fork_ci/audit_optional_scope.py "$compiler_checkout" OPTIONAL-COVERAGE.tsv \
  --source-coverage fork_ci/source-coverage.json
```

Generate inventories with pytest 8.4.2 and the pinned product runtime sources
available. The selected source paths are relative to the compiler checkout.
