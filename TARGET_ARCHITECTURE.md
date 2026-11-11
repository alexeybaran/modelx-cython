# Target Architecture

**Status: rewritten for the upstream-first Desktop Codex direction. It supersedes the architecture document from the earlier H09/H10 handovers.**

This document records durable architectural boundaries, not a predetermined implementation roadmap. The exact mechanism should be chosen by experiments and promotion gates in `PLAN.md`.

## 1. Structural base

The product remains **modelx-cython**:

```text
modelx model
   -> modelx export
   -> existing modelx-cython discovery / tracing / scoped transformation
   -> generated Cython Space/Cell package
   -> normal compiled-package API
```

Do not replace this with a new primary compiler, separate formula language or proprietary ordinary-use runtime.

## 2. Execution layers

A compiled Cell may use one of two normal physical paths beneath the same public Space/Cell API:

```text
                         modelx-cython Cell
                                |
                +---------------+---------------+
                |                               |
       proven native domain             object-capable Cython
                |                               |
   native args / direct calls /        Python objects / tuples /
   native scalar-dense-hash store      dicts / Python APIs as needed
```

Both remain Cython-generated execution. A Python object argument or result does **not** imply an interpreted Python formula.

An actual-Python formula body, if future census evidence proves one is required, is an exceptional compatibility fallback only. It must reuse the compiled receiver/downstream Cells and must not create a second model/cache/interpreter.

## 3. Native specialization

Native specialization is allowed only when its domain is justified by signatures, representative tracing, explicit generic specifications or other reliable evidence.

Potential specializations include:

- native scalar/dense Cell storage;
- native exact-key/hash storage;
- finite category encoding behind Python-visible labels;
- a direct internal Cython Cell entry that avoids repeated Python-visible dispatch while preserving public API/override semantics;
- prepared generic lookup resources behind a common author-facing helper interface.

Unknown/general object keys retain Python hash/equality/container semantics.

## 4. Cache/runtime boundary — approved D1

Cache/storage mechanisms may be transferred beneath existing modelx-cython Cells when evidence supports them. This does **not** automatically authorize other donor mechanisms.

The following are separate experiments requiring their own measured justification and approval:

- private execution frames replacing normal Space/Cell structure;
- prepared-policy ABIs that bypass normal model inputs;
- portfolio/batch runtimes as the ordinary API;
- formula-body duplication or a second formula language;
- custom C++ exception/unwinding machinery;
- broad compiler-side pandas interpretation.

D1 does not prohibit a small native structure owned by an existing compiled Space, nor a private internal Cython entry, if that mechanism independently proves material value while preserving semantics/API.

## 5. Helpers and adjusted models

Helpers are an **authoring/performance feature track**, not the compiler architecture itself.

```text
core                 feature/helpers
upstream-based       generic helper syntax/runtime
compiler             adjusted lifelib models
   |                         |
   +---------- performance experiments ----------+
```

Helper syntax must remain generic and readable in modelx, exported Python and compiled execution. Backend-native preparation may sit behind the same interface. The helper branch is not merged wholesale merely because an adjusted model is fast; each compiler-core prerequisite must satisfy the normal promotion gate.

## 6. Branch model

- `core` — accepted evolving compiler. Initially compiler-source-identical to pristine upstream.
- `upstream-pristine` — immutable bundled upstream reference.
- `experiment/*` — disposable/reviewable hypotheses. Never assumed to be future ancestry.
- `feature/helpers` — parallel helper track used for adapted-model performance testing.
- `reference/*` — historical H08/H09 or other research states, not production ancestry.

Promotion flow:

```text
core -> experiment -> correctness + CI + benchmark + complexity review
                               |
                         evidence report
                               |
                        user approval required
                               |
                    cherry-pick/rebase minimal delta
                               v
                              core
```

## 7. Performance target

The main end-state target is not "make MYGA fast at any cost." It is:

> Approach the previous compiler's calculation performance on lifelib models where that compiler achieved native execution, while keeping the delta generic, Cython-based, materially justified and as small/reviewable as practical.

MYGA is the immediate high-resolution optimization workload. BasicTerm_SC and additional previously native lifelib workloads are important cross-checks that a mechanism is generic rather than MYGA-specific. Broader census work follows at explicit gates rather than after every experiment.
