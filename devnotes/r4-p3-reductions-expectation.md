# Reductions on the fast line: expectation (written before the first measurement)

Branch `r4/p3-reductions` = `r4/p3-views` (09482aa, plus the analysis note 18c6b03) + the reductions slice.
Port source: `r4/integration` at 856b984 only (`reductions.py`, `_float_sum_state.pxd`, `runtime/_native_streams.pxd`
and their tests). Nothing from a later commit.

## What I expect to bring

| form group | pieces | non-test lines (about) |
|---|---|---|
| general reductions | `sum` / `min` / `max` / `any` / `all` of one iterable, as the whole value of `return`, `x =` or `x op=`; generator expressions and list comprehensions with any number of `for` / `if` clauses; `range` with C integer bounds as a C loop; a literal tuple unrolled (at most 32 instances); any other iterable as a loop over the object; a direct iterable (`sum(v)`). `sum` keeps the running CPython's own algorithm, branch for branch (`PySum`: the int loop, the float loop with its compensation from 3.12, the generic loop), or a two-double state when every item is a double and the result is read as one; `min`/`max` keep the first extreme and raise the built-in's ValueError on empty input; `all`/`any` stop where the generator stops; StopIteration in the body becomes the generator's RuntimeError. The header-only install of the state header. | 600 |
| nested sites | a reduction inside a larger expression, lifted to a statement where Python evaluates it unconditionally, with everything evaluated before it kept in order in temporaries (line B's hoist) | 230 |
| table walks | the generator over a table index that slice 2 rewrites to `(native) if test else (pandas)`: each branch its own lowered reduction under an `if`. Positions are C loops; the labels and reads are C values from the prepared table | 40 |
| tuples and `zip` | a source that is an exact tuple (or list) of numbers at run time, and `zip` of such sources: one C pass checks every item is an exact float (or an int that fits 64 bits), then the items are read as C values; otherwise the loop as written over the objects. Line B's borrowed-integer path for items passed only as C-integer Cell arguments | 150 |

About 1,000 non-test lines. The C-number functions of the new header are registered with the integer guard in a
registry of their own; `test_provider_returns` stays as it is and passes, and a new test checks the new header the
same way. Integer sums go through the exact int loop of `PySum` (no wrap: past 64 bits the sum continues on Python
integers as live does), and the integer arithmetic inside an element is the flagged arithmetic of the integer guard,
which runs over the lowered code as over any other.

Not brought: vectors as cell values (the next slice), line B's math helper registry (a reduction whose element calls
a math member keeps that call as this checkout compiles it), the `sum(iterable, start)` form (line B does not lower
it either), report entries of the admission file beyond the counts the tests read.

## Values

Every model equal to live on every policy (rel 1e-10 / abs 1e-7), every `net_cf` sum bit-identical to this slice's
base (09482aa), because the order of additions and the algorithm are the interpreter's. Term_US_SEC and MYGA_US_SEC:
their formulas have no built-in reduction of one iterable, so their generated files stay byte-identical and their
translate time unchanged.

## Speed per policy (base from `r4-p3-views-report.md`, re-measured on the same sample)

| model | base 09482aa | expected | why |
|---|---|---|---|
| IP_UK_S | 7.00 ms | 4 to 5 ms | `pols_sick_surv`, `pols_recovery`, `pols_death_sick` (zip of tuples) native over C doubles; `inception_rate`'s `max`/`min` over the walked list native; `sick_cohorts` itself (a tuple built per step) stays: the vector slice |
| UC_FR_S | 26.4 ms | unchanged (25-27) | its cost is `T.loc[[k]]` then `zip` of its columns: a list-keyed read, not a reduction. This slice does not touch it |
| VA_US_S | 6.65 ms | 5.5 to 6.3 ms | `sum(sa_pp_at(t, i, timing) for i in sub_ids())` with `i` a C integer argument; `any(... for u in range(...))` |
| KLV_DE_S | 0.428 ms | unchanged or a few % better | per-step `sum(claims(t, k) for k in (...))` unrolled; the range sums run once per item |
| Term_US_S | 0.298 ms | unchanged | no reduction per step |

## Build time

Translate time +3% to +10% (one more pass, only over formulas with a site; the integer guard runs over larger
formulas). C build time +0% to +5%: no new compiled module, the state header is inline code.
