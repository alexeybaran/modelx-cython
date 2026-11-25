# Tuple experiment (r4 w1-vectors)

Marker: the commits after this one are the experiment; the commits before it are the fixes and deletions.

Rule tried: a tuple of floats or integers BUILT AS A SEQUENCE (`tuple(<generator or list>)`, `(x,) * n`,
or those concatenated with other tuples, `(x,) + tuple(...)`) returned by a cell is a vector cell,
immutable for readers by construction; Python code gets an equal tuple. A literal display `(a, b)`,
a tuple used as a cell argument or a dictionary key, and a small record stay Python tuples.

Real shapes: lifelib r4/l1-lean-b, IP_UK_S, `claim_rate_vectors` and `sick_cohorts`, repeated small in
`modelx_cython/tests/compiled/test_tuple_vectors.py`.

Pass conditions: (a) those shapes compile as vectors and equal live; (b) records, argument tuples and key
tuples are untouched (generated code unchanged for them); (c) the experiment removes more lines than it
adds, counting the copy and ownership code a tuple makes unnecessary. Results: below, at the end.

## Results

Commits: 6617f5f (vectors.py, provider `vec_cat`), b549367 (table_cells.py public method), 6b77f2b
(tracer.py: turns it on), 25cb53f (test). Stack: free-threaded CPython 3.14.6, Cython 3.3.1, modelx 636877b.

| Condition | Result | Evidence |
|---|---|---|
| (a) the IP_UK_S shapes compile as vectors and equal live | **yes** | `test_tuple_vectors.py`: `claim_rate_vectors` is `('T', ('V','V','V'))`, `sick_cohorts` `'V'` (`(0.0,) * n`, `tuple(<generator>)`, `(x,) + tuple(<generator>)` as C loops and `vec_cat`), `counts` `'I'`; every reader (`v[z - 1]`, `sum(v)`, `sum(a * b for a, b in zip(v, surv) if a != 0.0)`) is lowered; values equal live (rel 1e-10, abs 1e-7) and Python gets tuples |
| (b) records, argument tuples and key tuples untouched | **yes** | generated code of `record`, `small` (a display of floats), `key_of`, `lookup` (a key), `scaled`, `with_arg` (arguments) is identical before and after; only the vector cells and their readers change |
| (c) removes more lines than it adds | **no** | adds 276 lines and removes 20 outside tests (vectors.py +240/-16, provider +23, table_cells.py +4/-1, tracer.py +9/-3); it removes nothing, because list vectors stay |

On (c): the copy and ownership code a tuple makes unnecessary is list-only, about 310 lines. In vectors.py that is
the D94 local lists (54), the R1 escape analysis (about 45), ownership and `VectorWriteError` (about 25), append,
element write and element update lowering (about 60), and the write checks `item` (8) with the strict-locals
rules (about 15). In the provider it is grow, set, slow set and append, and `vec_item`/`ivec_item` (about 110).
Tuple cells need none of it, but list cells do, and the lean models have list vector cells and list locals
built by appends and element writes. The experiment could only pass (c) if lists stopped being vectors, which
would mean rewriting those models' vector cells as tuples. That was not measured.

Micro timing, best of 5 processes, one compiled driver cell running the whole projection
(`max_dur` 360, `proj_len` 240; `pols_sick_surv`, `pols_death_sick`, `pols_sick` per t, `pols_sick_dur` per t and z):

| Form | live | compiled before the experiment | compiled with it |
|---|---|---|---|
| tuple (IP_UK_S as written) | 0.056 s | 0.0175 s (Python tuples) | **0.0018 s** |
| list (`[0.0] * n` filled, the same readers) | 0.056 s | 0.0017 s | 0.0017 s |

The tuple form compiles to the list form's speed: 9.6x faster than without the experiment. The list form's
generated code is unchanged apart from temporary numbering.

## Tuples only (owner's decision after the experiment)

A tuple of numbers built as a sequence is the vector form. A list is an ordinary Python object again: it is
correct but no longer accelerated.

- A cell returning a list compiles to an ordinary object cell with the dictionary cache. Its value is the
  cached list itself, as live's is, so it equals live (`test_vector_list_mutators`, `test_vector_cells`
  `listed`).
- Building a local list and returning `tuple(out)` stays a typed buffer. The returned buffer is the list's
  own, with no copy. `(x,) + tuple(...)`, `(x,) * n` and `tuple(<generator>)` are built natively.
- A local list is a typed buffer only where every use is native or an eager read. Eager reads are an eager
  builtin, a comprehension's source, `in` or a slice, and they get a copy (`tolist()`). Any other Python use
  (a method, a callee, a second name, a container) makes the local a Python list in a second lowering round.
  This one rule replaces the R1 escape analysis, the `VectorWriteError` build errors and the list copies at
  Python exits. A write to a cell's value (a tuple) raises `TypeError` as live does.
- Python code gets an equal new tuple (`totuple()`), never the cache. A tuple cannot change, so the copy
  cannot be told apart from live's tuple, apart from identity.
- D94 is gone: a module without vector cells is not lowered.

### Note for the model text (lifelib `r4/l1-lean-b`)

In the regional `_S` models (the `_SEC` variants have the same cells), 12 cells already return tuples built
as sequences and compile as vectors. These include IP_UK_S `claim_rate_vectors` and `sick_cohorts`, BU_DE_S
`recov_rates` and the other rate vectors, Medical_JP_S `los_days` and `los_probs`, VA_KR_S `fund_ids` and
WP_UK_S `guarantee_years`.

17 cells return lists and are now plain Python objects. The hot ones are the cohort vectors:
- BU_DE_S `dis_cohorts`, `runoff_cohorts` and their `_first` variants
- ADE_FR_S `itt_cohorts`, `itt_cohorts_raw` and `itt_rate_vectors`
- Dep_FR_S `dep_cohorts`

The others are small id or anchor lists:
- Obseques_FR_S `surr_scale_anchors`
- Annuity_JP_S `mort_anchor_ages`
- Child_KR_S `inc_pivots`
- CI_KR_S `ci_cohort_ids`
- Medical_KR_S `sev_points` and `shape_buckets`
- DIA_US_S `qlac_flags`
- VA_US_S `sub_ids`
- VUL_US_S `subaccount_ids`

To keep the hot ones fast, keep building the lists exactly as now and return tuples:
`return tuple(part), tuple(tot), tuple(totr), tuple(val)` in Dep_FR_S, and `return tuple(out)` elsewhere.
The local lists stay typed buffers and the returned tuples are those buffers, with no copy. Readers need no
change, since `v[z - 1]`, `len`, `sum`, `zip` and loops read tuples as they read lists. Two things would make
a reader stop working: changing a returned value in place (live raises on a tuple) and comparing it with a
list (a tuple is never equal to a list).

### Results

Lines outside tests:
- The tuples-only commits change +239 / −390, net −151, against my estimate of about 310 list-only lines.
- They removed the R1 escape analysis, `VectorWriteError`, list copies at Python exits, D94, the list `'X'`
  returns and the duplicate fallback notes.
- They added the untype round, `_copy_reads`, `totuple`, and `tuple(v)` pass-through and copy.
- Against the marker 8725799 the tree is still **+105** (+412 / −307). The experiment's tuple forms
  (sequence check, `(x,) + tuple(...)`, `vec_cat`) cost more than the list code they replace.

Cuts that would reach the target, none made (unmeasured, or in other files):
- diag-kr min/max of a generator over vectors (`expand_extreme`, about 45 lines). `reductions.py` would read
  a tuple copy; affects Annuity_JP_S and Cancer_KR_S.
- The D93 leading sum (about 35 lines; VUL_US_S, VA_KR_S).
- The powers arbitration (`scalar_only_power` with `leave_BinaryOperation`, about 56 lines), moved into
  `powers.py` through its `operand_kind` hook.

Micro timing, best of 7 processes, the projection driver of the experiment:

| Form | live | compiled |
|---|---|---|
| tuple, `(x,) + tuple(...)` (IP_UK_S as written) | 0.052 s | 0.0019 s (0.0018 s at the experiment head) |
| list built, `return tuple(out)` (the advised rewrite) | 0.050 s | **0.0016 s** |
| list returned (now a Python object) | 0.052 s | 0.0156 s (0.0017 s before tuples only) |

The concatenation form is about 10% slower than a list built in place, because it builds both parts and then
copies them (`vec_cat`). Writing the parts into one allocation would close the gap, for about 20 lines.

Python code no longer gets the cached object of a vector cell. It gets an equal new tuple per call, where
live returns the cached tuple itself, so `cell() is cell()` is now False
(`test_collect_direct_isolation` compares by value).

Tests: 700 unit tests pass and 207 compiled tests plus `test_diag_kr_forms` pass. The 6 lifelib-package tests
were not run.
