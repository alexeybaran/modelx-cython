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
