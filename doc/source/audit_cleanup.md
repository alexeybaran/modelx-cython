# Compiler audit repairs and generalization

Resolution of the [original compiler audit](../../../../analysis/compiler_audit_20261108/AUDIT.md).

The audit repairs and cleanup are implemented. Qualification preserves the
existing September performance gates, with the CI and parallel-execution limits
disclosed below. This is an experiment receipt, not an all-green release claim.

Final compiler source: `4bf6d702a927e9182980eac390ef1274ffc7c10d`.
The completed `803ea70` / `2b2cd7b` cohorts qualify the CST migration and serve
as controls for the subsequent measured scalar-math deletion. Final-source CI
and the four-model performance matrix were rerun; their results are separate
from those earlier controls.
Control: `299d8c10ca56d685eec2cf11b14b3a9c7d8846fa`.
Author reference: `0f3e5649669fea3d42ca204aef7808c10d66d417`.
The compiler remains on `experiment/audit-generalization-20261108`; no promotion
or change to core is part of this work. The exporter adapter is separately
committed at `b8b7472c3199874a71f823b0836aae83878b03ff` in the modelx experiment.
The original exporter branch remains at `f65730d`.

## Repairs

* Out-of-int64 generator literals and range arguments retain their ordinary
  generator path. This fixes both build failures and the silent sign change at
  `2**63`. Native ranges also prove the final increment: a finite one-element
  Python range must not become an overflowing, nonterminating C loop.
* Native lookup contexts belong to the Cell and provider together. Allocation
  avoids authored names, peer Cells and generated cache fields. Providers can
  coexist in one formula without incompatible context types.
* Multiline expressions retain their source CST rather than being detached and
  reparsed. This addresses the discovered regional formatting regressions.
* Resource loading retains the older importlib.resources API fallback. Versioned
  CI verifies all six provider files before running helper tests, and can run an
  immutable compiler snapshot. Actual full-runtime qualification here covers
  Windows Python 3.12 and free-threaded 3.14, not every advertised Python version.
* Registered sums can also assign one local result, while a replacement helper's
  arbitrary object result stays on the ordinary fallback.

## One source frontend

The original author compiler used LibCST for source analysis and `ast.literal_eval`
for the specification dictionary. The additional formula AST layer was removable;
it was not necessary for native execution performance.

Arithmetic, generator admission, scope facts, enum bindings, collection discovery,
cache layout recognition, direct-formula admission and range proofs now inspect
the existing LibCST nodes. Binding/read contexts use LibCST metadata. Proofs edit
the same call nodes they inspect, removing AST/CST position matching and queues.
There are no production imports of Python's `ast` module. Safe specification
decoding operates on literal CST nodes. LibCST internally uses `literal_eval` for
literal tokens; this is distinct from parsing formula bodies into a second tree.
Independent test and audit oracles still use Python ASTs.

Cython remains responsible for compilation. The compiler still needs source
structure to connect modelx Cells, cache domains, helper capabilities and public
override behavior to generated Cython. Those facts are unavailable to Cython's
ordinary Python optimizer. Reusing LibCST preserves those optimizations without
maintaining two source representations.

## Cleanup and retained mechanisms

One resolved immutable policy replaces repeated option interpretation. One name
allocator and shared primitive/scope contracts replace divergent utilities.
The entry pipeline uses shared in-memory syntax and facts, eliminating four disk
rewrite/reparse loops. Build generation and the probed Cython array-equality
compatibility repair are isolated from optimization policy.

The exporter owns collection protocol version 2. The compiler leaves its
runtime intact and supplies the evaluator and worker policy through the hook.
Older collection exports require re-exporting with current modelx. Explicit
domain-order merging is reproducible at a fixed worker count. Completion order
remains the default for reused chunks, and parallel grouping can change
floating-point roundoff.

Deletion screening retained all 96 trials and unchanged package hashes. Removing
terminal helper fusion made MYGA serial collection 1.577 times slower and four-worker
collection 3.339 times slower at the paired median. Removing bounds proofs made
serial MYGA 1.097 times slower and VA 1.088 times slower. Those deletions were
rejected; the useful mechanisms were ported to the shared frontend instead.
This screening does not replace final qualification.

The source is not dramatically smaller in lines: author 14 files / 5,637 lines,
control 34 / 9,648, final 37 / 9,687. The final net increase is 39 production
lines. The benefit is fewer duplicated representations and boundaries plus
necessary correctness, compatibility and adapter contracts. Further speculative
specializations are not justified by this work.

Independent deletion tests justify retaining primary reservations, cache hits
and one-axis specialization: their removal slows MYGA serial collection by
9.59%, 7.38% and 8.23% respectively at the paired median. VA also loses 5.43%
without cache hits and 5.68% without one-axis specialization. All variants
passed full financial validation before timing; all 120 screening trials remain.

Scalar math was the exception. Disabling it improves VA serial collection by
2.98% and is near parity on MYGA. Eight further pairs on original BasicTerm_SC
10k and both DV01 routes give medians within 0.7% of parity, upper mean bounds
below 1.4%, and unchanged validated values. The backend, its canonical-callable
bootstrap, tracer metadata and duplicated rounding helpers were removed.
Ordinary Cython calls retain custom numeric hooks, aliases, mutable module
References and exception behavior; the rich floating-point corpus is retained.
The old `native_math` boolean remains accepted for spec compatibility.

## Audit closure checklist

| Original finding or recommendation | Change and verification |
|---|---|
| Wide integer and multi-provider regressions | Shared representation/range checks, final-increment proof and provider-owned contexts; compiled composition, replacement, retry and boundary tests. |
| Python compatibility surface | Compatible resource loader and direct CST passes remove the new `files`/`ast.unparse` requirements. Provider guards already retain builtin float-sum semantics on older interpreters. Full runtime CI covers 3.12 and 3.14t; no claim of a real 3.8 full-suite run. |
| Wrong compiler/provider pairing | Versioned runner verifies six source hashes and an immutable compiler path; a mismatched-provider negative check is retained. |
| Repeated planning, entry reparsing and scope logic | Immutable policy, existing entry records, one CST frontend, metadata, shared names and in-memory entry pipeline. Public dispatch and checked fallbacks remain. |
| Helper-internal coupling | Provider owns data/lookup implementation and contexts. Compiler shares capability/binding contracts; the measured fusion deletion is rejected. No new helper vocabulary or table interpreter. |
| Reducer generalization | Shared iterator, type, binding, naming and fallback checks; return and single-assignment sums; ordinary Cython generators for declined cases. |
| Brittle exporter replacement and ordering | Exporter-owned v2 hook with an explicit domain-order option and tested failure order and lifecycle. |
| Weak-default attribution | Independent deletion experiments retain necessary fusion, proofs, reservations, hits and one-axis specialization. Scalar math passes four-model deletion qualification and is removed. |
| Independent compatibility corpus | Original behavioral oracles plus compositional wide values, providers, aliases, overrides, changed References, cancellation, exceptions and populated pickle tests. |
| Source-assertion mismatch | Immutable tests remain unchanged and their two failures are disclosed. Maintained structural checks retain their runtime assertions; generated text layout is not a public execution contract. |

## Architecture and remaining limits

Modelx export, discovery/tracing, Space/Cell layout, one logical cache per Cell,
public compiled-package API, virtual overrides and Cython execution remain.
There is no execution Frame, prepared-policy ABI, second formula/pandas
interpreter, duplicate actuarial formula implementation or custom exception
runtime. Generated MYGA/VA Python ASTs match the previously validated scalar-math
deletion variant. Against the frozen control, intentional changes include the
removed math wrappers and generic collection adapter; provider resources and
class declarations remain unchanged.

The four-worker original MYGA workload has intermittent access violations in
shared pandas indexing under free-threaded Python. The author compiler, frozen
control and the CST candidate reproduce it. Four retained repetitions each
give three passes and one crash in control and candidate; the author probe gives
two passes then the same crash. This lane is unqualified for timing, not a
performance pass. Original one-worker results and adjusted prepared-input lanes
have separate scopes. The compiler must not silently add a pandas interpreter,
copy shared inputs or serialize arbitrary formulas to disguise this boundary.

Untouched author tests retain their two known generated-source assertions:
`test_locked_spaces[locked_spaces]` and
`test_mx2cy_with_uslib[uslib_ulsg-ULSG_US_S]`. They are not relabelled as passes;
maintained structural assertions retain their behavioral oracles.

Original CLI modes, no-spec translation, explicit return-type specifications,
ordinary generators, object-valued and uncached Cells, defaults, keyword calls,
nested ItemSpaces, mutable unmarked References and subclass dispatch remain.
The repaired wide-value and multi-provider examples restore concrete paths that
the audited candidate had lost. Numeric specialization still requires a proven
representation; unfamiliar syntax retains the ordinary Cython path.

Opt-in contracts retain their limits: marked enum bindings are compiled schema;
receiver reuse requires confinement, independent policies and stable References.
`automatic_optimizations=False` disables automatic choices and is not an upstream
emulation switch. Public source/API compatibility does not establish native ABI
or cross-revision pickle compatibility; generated packages must be rebuilt.
Per-Cell cache caps do not establish a portfolio-wide memory bound. The compiler
is structurally generic, while accelerated coverage and performance remain
qualified by workload and semantics rather than by insurance region.

## Final-source CI

All three full suites were rerun against the immutable `4bf6d70` snapshot and
the manifest-matched provider. Counts overlap between suites and must not be
added into a unique test total.

| Runtime | Maintained compiler | Candidate/helper | Untouched author |
|---|---|---|---|
| Windows Python 3.12 | 298 passed, 7 skipped | 133 passed | 169 passed, 2 known source assertions failed, 7 skipped |
| Windows Python 3.14t | 298 passed, 7 skipped | 133 passed | 169 passed, 2 known source assertions failed, 7 skipped |

The exporter suites separately pass 70 tests on each runtime. The final wheel
contains all 37 production files with exact qualified-source hashes.

## Final-source performance

Timing uses free-threaded Windows Python 3.14 and Cython 3.2.4, one timing
process at a time. Each primary case has eight randomized paired rounds.
Ratios below are candidate/control paired medians; a ratio above one is slower.

| Workload | One worker | Four workers |
|---|---:|---:|
| Adjusted MYGA, collection | 0.9917 | 0.9940 |
| Adjusted MYGA, direct | 1.0100 | 1.1396, investigated below |
| Adjusted VA, collection | 0.9918 | 1.0038 |
| Adjusted VA, direct | 0.9905 | 0.9966 |
| Original MYGA, reports | 0.9991 | Unqualified shared-pandas lane |
| Original VA, reports | 0.9933 | 1.0014 |
| Original BasicTerm_SC, 10,000 policies | 1.0168 | 1.0116 |
| DV01, direct | 0.9914 | 1.0062 |
| DV01, collection | 0.9896 | 0.9712 |

Both unchanged September gates pass: MYGA median 1.8860 and upper 95% mean
bound 1.8968; VA median 1.8981 and upper bound 1.9087. Each must be at most two.
These compare the existing public API boundaries against the September
reference, separately from the frozen-299 comparisons in the table.

BasicTerm_SC evaluates the upstream mean public `pv_net_cf()` over policies
0 through 9,999 and traces only policies 1 through 10. Every policy value matches
the pristine author compiler exactly at one/four workers; the aggregate is
1448.9630534601538. Both DV01 routes validate all 105 policies and six groups.
Adjusted MYGA/VA validate reports, 2,500 singleton policies, 2,499 transitions
and one/four-worker direct/collection paths. All timed package hashes remain
unchanged and all qualified primary timing attempts succeed.

The MYGA four-worker direct primary result is retained as adverse evidence.
A subsequent 16-round comparison gives candidate/control median 1.0091,
mean 1.0196 and mean CI [1.0072, 1.0358]. Combining all 24 candidate/control
pairs gives median 1.0167, mean 1.0454 and CI [1.0228, 1.0702]. The earlier CST
builds with and without scalar math also show slower-tail samples (16-round
medians 1.0090 and 1.0143). Thus the 14% primary median does not reproduce,
but neither zero slowdown nor a scalar-math cause is established.

A further 16-round same-source rebuild control validates the unchanged 299
compiler and identical generated Python ASTs. Rebuilt299/frozen299 has median
1.0066 and mean 1.0311; candidate/frozen299 has median 1.0066 and mean 1.0333.
Candidate/rebuilt299 has median 1.0094, mean 1.0047 and mean CI [0.9704, 1.0377].
The unchanged compiler rebuild also reproduces slower-tail samples. This
supports a build/artifact contribution, without proving a unique cause.
All 40 candidate/frozen299 pairs together give median 1.0102, mean 1.0405 and
CI [1.0224, 1.0607]. No samples or earlier cohorts are removed. The final source
meets the existing gates; exact zero slowdown and identical tail latency are
not established. Confidence intervals describe these cohorts, not worst cases.

## Pre-deletion qualification controls

Windows 3.12 and 3.14t each pass 300 maintained tests (seven existing skips)
and 136 candidate/helper tests. The two untouched source assertions remain
as disclosed above, alongside 169 passes and seven skips per runtime.

Eight paired MYGA/VA rounds pass the unchanged September gates: MYGA median
1.8740, upper 95% mean bound 1.8812; VA median 1.9190, upper bound 1.9839.
All eight current/frozen route medians are within 1.1% of parity. Parallel
confidence intervals are wider; these are not worst-case slowdown bounds.

BasicTerm_SC uses the original mean public `pv_net_cf()` over all 10,000 policies,
tracing only policies 1 through 10. Every full-population value matches the
pristine author compiler exactly at one/four workers; the mean is
1448.9630534601538. Eight paired rounds give current/control medians 1.0047 at
one worker and 0.9957 at four. The earlier 18-policy report timings were the
wrong workload and are explicitly superseded, with their receipts retained.

DV01 completed eight paired rounds per route/worker case without failed
attempts; paired medians range 0.9805 to 0.9977 of the frozen control.
These results belong to `803ea70`; the final `4bf6d70` results are reported above.
The regional census was stopped at the user's direction; its partial receipts
do not constitute complete final-source regional qualification.
No all-regional native-performance or unrestricted parallel-safety claim is made.

## Reproduction

Evidence and scripts are under `analysis/compiler_cleanup_20261108` in the
workspace. `QUALIFIED_SOURCE.json`, `CONTROL.json` and per-run provenance pin
source, provider, exported inputs and generated packages. Source snapshots are
immutable. The selected original-model packages were built afresh against the
unchanged, durable 299 controls; vanished temp packages from earlier runs are
not used. The stopped broader census remains partial evidence only.
Canonical Git contents and the working checkout match the qualified archive
after normalizing Windows CRLF/LF endings. The wheel is checked byte-for-byte
against the actual immutable archive used for CI and compilation.

Use `devtools/run_windows_ci.ps1 -Suite audit -CompilerSource <snapshot>` with
the appropriate Python and standard-runtime dependency overlay. `campaign.py`
rebuilds and validates the unchanged financial matrix; `dv01.py` retains distinct
direct and collection fixtures. `final_gate.py`, `other_timings.py` and
`basicterm_10k.py` run only
after all compilation/validation jobs finish. Eight randomized paired rounds,
all outliers, the unchanged September API gate, package hashes before/after and
any failed/unqualified lane remain in the receipts. There is no native PGO or
forced-inlining build experiment.

The final evidence index is `analysis/compiler_cleanup_20261108/CLOSURE.json`.
Its checked inputs include:

| Receipt under that directory | Scope |
|---|---|
| `validation.json` | Exact full-suite counts, XML hashes, source/provider identities |
| `source-review.json` | Production inventory, generated-code comparison, unchanged prior branches |
| `packaging-4bf6d70/RESULT.json` | Wheel hash and all production members |
| `final-gate-4bf6d70/SUMMARY.json` | Adjusted MYGA/VA routes and unchanged September gates |
| `original-myga-va-dv01-4bf6d70/SUMMARY.json` | Original MYGA/VA and distinct DV01 routes |
| `basicterm-10k-4bf6d70/VALIDATED.json` and `SUMMARY.json` | Full 10,000-policy oracle, subset trace and timing |
| `myga-direct-confirm-4bf6d70/SUMMARY.json` | 16-round direct-route follow-up and original combined 24 pairs |
| `myga-rebuild-control-4bf6d70/SUMMARY.json` | Same-source rebuild, full validation and 16 further rounds |

Each timing cohort retains raw trials and package provenance before/after.
`WORKLOAD_CORRECTION.json` records why the earlier BasicTerm report cohort is
superseded. The failed initial final-source orchestration (`native-nomath.exit=1`)
is retained: its BasicTerm harness lacked the standard-runtime psutil overlay.
`basicterm-final-requal.exit=0` records the corrected, complete validation;
MYGA/VA and DV01 had already completed successfully in the initial orchestration.
`final_closure.py` verifies this index, source/provider identities, all cohort
package hashes and the final 40-pair direct-route estimate.
