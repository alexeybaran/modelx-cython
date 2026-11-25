# Brief for independent reviewers of one compiler area

You review alone. Other reviewers get the same brief for the same or other areas; do not fetch or read any
branch named `review/...` other than your own. Be adversarial. Read the code, not the names.

## Context
modelx-cython compiles an exported modelx model (Python) to Cython. Upstream is commit `d22426f` (6,300 non-test
lines; fetch https://github.com/fumitoh/modelx-cython). This branch is the development tree with the old helper
lane removed: about 22,000 non-test lines added on upstream. It makes 70 lifelib models (57 regional `*_S`, 12
`*_US_SEC`, one DV01 example) run natively and equal to live modelx. The work will be proposed upstream as
separate pieces: a fast compiler for the SEC models; collect; and a "syntax bridge" from the SEC spelling toward
the original models, one commit per group of syntax enablers (labels as enum members; reductions; vectors; tables
in four commits). Your area is one of those pieces. The owner's targets: run time, memory, syntax convenience.
Guards: compilation time, compiler complexity, how generic the compiler is for models other than these 70,
reliability.

The stack to run tests: free-threaded CPython 3.14 (`uv python install 3.14t`), Cython 3.3.1, development modelx
`git clone https://github.com/alexeybaran/modelx` at commit 636877b, lifelib `git clone
https://github.com/alexeybaran/lifelib` at 04d69db (plain git clone; do not use an add-repository tool). You may run
unit tests and small experiments in a scratch folder; build at most one small model at a time; do not build
lifelib models.

## Areas (non-test modules; lines on this branch; measured value on the earlier stack, one mechanism off at a time)
| Area | Modules | Value measured | Use in the 57 lean models |
|---|---|---|---|
| reductions | `reductions.py` 1,280, `_float_sum_state.pxd` 219, `_native_iterables.*`, their wiring | 1.13x-4.22x, median 1.44x | `sum`/`min`/`max` over generators and ranges |
| vectors | `vectors.py` 1,950, `vector_ref_reads.py` 184, `runtime/_native_vectors.*` 551, `held_arrays.py` | 21x-32x on models with list cells; identical code elsewhere | cells returning lists |
| table-base | what every table read needs: receiver, keys, holds and slots in `frame_reads.py` (2,876 in all), `key_facts.py` 419, the read recorder `read_notes.py` 476, schema tracing in `tracer.py`, `runtime/_tables.*` 1,264, `table_cells.py` 613, audit and reports | with the forms: 15x-375x | every model; translation is about 1.5x slower with the recorder |
| table-forms | the per-form code in `frame_reads.py`, `row_cells.py` 410: scalar read with a literal column (358 sites), row field (1,358), sub-table (40), index extreme (107), membership (15), ordered lookup (0 in lean text, 39 before), index walk (83), column chosen by an expression (28), a two-statement spelling used by one model | as above | counts are read sites in the lean text of the 57 models |
| rest | providers (`helpers.py`, `provider_manifest.py`, `provider_build_cache.py`, `helper_members.py`, `runtime/_native_math.*`, `_native_streams.*`), rounding (`rounding.py`, `_rounding.pxd`, `runtime/_native_rounding.*`), `expression_kinds.py`, configuration keys in `config.py`, reports in `admission.py`, shared wiring in `transformer.py`, `builder.py`, `cli.py` | typed min/max, `int()` identity, divisibility: 1.00x-1.02x | all |

## What to deliver for your area (fixed grid, so two reports can be compared line by line)
1. What the code does, in a table of mechanisms with lines, and what would be lost without each.
2. Complexity: what can be removed or made shorter; what duplicates Cython, CPython, pandas or upstream; what
   serves no model. For each: lines saved, expected cost, risk, and how to verify on at most ten models.
3. Generic: what is fitted to these lifelib models or to one spelling; which rules would not fire on another
   author's model; what changes behaviour for an existing upstream user. Show the case.
4. Reliable: where compiled can differ from live silently, raise where live works, or depend on the traced
   sample, the platform, thread timing or call frequency. Minimal reproduction, expected, actual, fix with lines.
5. For table areas: whether the proposed split into four commits (base with scalar reads; rows; sub-tables with
   extremes and membership; ordered lookups and index walks as a tail that can be declined) is the right one, and
   what the smallest first commit is.
6. Bottom line: the smallest version of your area that keeps (a) the measured speed, (b) speed within 25-30%.

How findings will be used, so rank accordingly: a silent difference from live, or a regression for unmodified
upstream use, is blocking and is fixed now; a deletion that costs nothing measurable and takes under half a day
is done now; a redesign is written into the proposal text with your evidence and not built now. A suggestion you
have not compiled or measured must say so: earlier reviews proposed one fix that did not compile and one that
cost 7%.

Rules: results equal live (rel 1e-10, abs 1e-7) or the build stops or the run raises; nothing may depend on what
the sample reached, on spelling, call frequency, operating system or compiler; Python 3.10+, ordinary and
free-threaded builds. No assistant, model or vendor names and no personal data in what you write.

Write `devnotes/review-<area>-<letter>.md` (at most 200 lines, tables, file:line references), commit it on your
own branch `review/r4-<area>-<letter>` (given in your first message), push only that branch, never force-push,
open no pull request. Then send the coordinator a message (send_message to "@parent", at most 250 words): your two
bottom-line numbers, the five largest cuts with lines, and any blocking defect. If a permission check denies an
action, stop that action and say so in that message.
