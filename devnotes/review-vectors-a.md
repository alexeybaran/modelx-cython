# Review of the `vectors` area (reviewer a)

Scope: `vectors.py` 1,950, `vector_ref_reads.py` 184, `runtime/_native_vectors.pxd` 252 / `.pyx` 299,
`held_arrays.py` 126 (2,811 lines), plus the wiring in `tracer.py`, `builder.py`, `transformer.py`,
`table_cells.py`, `admission.py`, `serial_collect.py`.

**How this was checked.** Stack of the brief: free-threaded CPython 3.14.6, Cython 3.3.1, modelx at 636877b. No
lifelib model was built. Evidence levels used below: "built" = one small model exported, compiled and compared
with live in separate processes; "lowering" = `vectors.lower` run on a formula with the mock classes of
`tests/test_vectors.py`; "read" = by reading only. Timings are from one synthetic model (five cells, lists of
20,000 floats, 60 arguments, best of 3-5 processes; the machine is noisy, differences under about 15% are not
resolved). Trial fixes for R1, R2, R4, R5 are the second commit of this branch (38 lines in `vectors.py`, 6 in the
provider; line numbers in this report are those of the first commit): with them the reproduction model equals
live for R1, R4, R5 and raises `TypeError` for R2, `tests/test_vectors.py` and `test_vector_extrema_typing.py` pass (51), and the nine compiled vector test
files pass (11 tests, 281 s). They are evidence, not a proposal to merge as is.

Measured on the synthetic model, compiled against live: native chain of three vector cells 0.013 s against
0.240 s (18x); a loop with a derived index 35x; a cell of 200,000 cache hits 10.7x; a vector cell called from
Python 60 times 2.1x only, because each call copies the vector into a new list (`tolist`, M12).

## 1. What the code does

| # | Mechanism | Lines | Lost without it |
|---|---|---|---|
| M1 | Traced shape `V`/`I`/`E`/`('T', kinds)` of a returned list; cell return type becomes `F64Vec`/`I64Vec` | `tracer.py:111-144,466-472`, `builder.py:388-400,570-572`, `vectors.py:77-94` | everything: a list cell stays `object` in a dict cache |
| M2 | Provider: two buffer classes, new/full/of/slice/get/set/append, compensated sum, exact int sum | `.pxd:47-56,91-252`, `.pyx:18-237` | the typed storage; elements are boxed floats again |
| M3 | Dense cache `VecCache` and its budget plan (1-3 integer parameters, sampled window) | `vectors.py:97-222`, `.pxd:69-88`, `.pyx:243-299`, `transformer.py:358,1162,1478,1730`, `serial_collect.py:488` | dict cache for vector cells with arguments (hash and lock per hit) |
| M4 | Facts of one formula and the two typers: vector kind of locals (`kind`, `vector_kind`), scalar kind of expressions (`scalar`) | `vectors.py:385-529,557-750,753-899` | nothing can be lowered; only cells' own returns are converted |
| M5 | Element forms: `v[i]`, `v[i] = x`, `v[i] op= x`, `append`, `len`, `sum(v)`, `sum(v[a:b])`, `v[a:b]`, `[x] * n`, `[]`, tuples of vectors | `vectors.py:918-936,1056-1209,1285-1301,1927-1950` | element reads and writes go through Python objects |
| M6 | Loops: `for x in v`, `for a, b in zip(v, w)`; statement-level list comprehension and `sum(generator)` as C loops | `vectors.py:1303-1371,1401-1447,1891-1924` | per-element boxing in the hot loops (the main part of 21x-32x, estimate) |
| M7 | Loop versioning of `for i in range(a, b)`: one bounds test before the loop, direct buffer access inside | `vectors.py:1449-1556`, `.pxd:184-186` | about 20% on a loop-heavy chain (measured, C3) |
| M8 | C `long long` for range loop targets and "derived index" locals; bounds test repeated at the use site | `vectors.py:1559-1678,1832,1853-1854` | loop variables are Python ints in modules with vector cells |
| M9 | Powers with vector operands (`_mx_pow`) and the arbitration with `powers.py` (D91) | `vectors.py:938-1054` | `x ** y` on elements goes through Python `pow` |
| M10 | Displays `[a, b]` (D87), concatenation `[..] + list(range())` (diag-kr), `min`/`max` of a generator (diag-kr), sum leading a `+ - * /` chain (D93) | `vectors.py:340-346,423-452,1122-1136,1211-1283,1373-1398,1872-1888` | these four spellings build a Python list first (`vec_of`) or stay generators |
| M11 | Local lists typed in modules *without* vector cells (D94) | `vectors.py:229-290,553,786-796,1810,1821-1824` | no typed local lists outside vector-cell modules |
| M12 | A vector that reaches Python code becomes a new list (D154), decided twice: inside a formula and at cell calls | `vectors.py:1690-1793`, `table_cells.py:147-158,257-272,415-503` | Python would see `F64Vec`, which has no sequence protocol (`.pyx:18-35`) |
| M13 | Scalar tail reading a small integer vector of another Space (`native_vector_ref_reads`, off by default) | `vector_ref_reads.py:1-184`, `.pxd:58-66`, `transformer.py:743-745` | nothing in a default build |
| M14 | Typed-memoryview cell read by subscript held per item | `held_arrays.py:1-126` | the shared acquisition count per read (thread scaling of one named model) |

## 2. Complexity

| # | Cut | Lines saved | Expected cost | Risk | Verify (at most ten models) |
|---|---|---|---|---|---|
| C1 | Remove M13: opt-in key that no model, example or integration test in this tree sets (only `doc/source/spec.md:292` and its own test) | 184 + 9 (`.pxd:58-66`) + 3 = **196**, and 117 test lines | none in a default build | none | grep the 70 specs for the key; if absent, delete; else time that model |
| C2 | Remove `VecCache` (M3); vector cells with arguments use the existing dict cache | 126 + 57 + 20 + about 25 wiring = **228** | **measured: the hit-dominated cell is 3.0x slower** (0.108 s against 0.036 s); cells that loop over elements unchanged | not a free cut; only if hit-heavy cells are absent | count reads of vector cells with arguments per run in the list-cell models |
| C3 | Remove loop versioning (M7) | 108 + 4 = **112** | **measured: +20%** on the native chain (0.0155 s against 0.0128 s, two runs of five); removes the code growth of R9 | low | the 21x-32x models, run time and generated C size |
| C4 | Remove concatenation (M10, diag-kr): one spelling, falls back to a correct `vec_of(list)` | 7 + 19 + 3 + 56 = **85** | one list per call of the cells that use it | low | the diag-kr model |
| C5 | Powers: delete the second rewrite (M9) and give `powers.py` the element kinds through its existing `operand_kind` hook (`transformer.py:735`) | about **80** | none expected; removes D91 arbitration (`vectors.py:938-979`) | medium: not compiled | models where the report has `powers > 0` |
| C6 | One "native use" predicate instead of two (M12): `_ObjectUses._native` (`vectors.py:1735-1766`) and `RenameMembers._vector_native` (`table_cells.py:442-487`) | about **35** | none; they already differ (R4 exists because only one handles `IfExp`) | low | unit tests of both files |
| C7 | `min`/`max` of a generator (M10) duplicates `reductions.py:109,295-316` with a different typer | 26 + 7 + 8 = **41** | reductions must accept vector operands instead | medium: not compiled | models with `extreme_calls` in the report |
| C8 | Leading sum in a chain (D93): one spelling | 8 + 17 + 6 = **31** | that sum stays a generator | low | the diag-us2 model |
| C9 | (withdrawn) `_CheckedFloatReads` looked like a repeat of inline `vec_get`; **measured: without it the derived-index loop is 1.6x slower** (0.0020 s against 0.0012 s) | 0 | keep | - | - |
| C10 | `VecSum` is a "private copy" of `_float_sum_state` (`.pxd:9-11,91-115`) | **25** | none | low: a `cimport` across providers | unit tests of sums |
| C11 | Dead or broken: `VecCache.__reduce__` and `_vec_cache_from_items` (`.pyx:290-299`; pickling a vector raises `TypeError`, run), `vec_of` branch for `I64Vec` (`.pyx:93-94`; `list(I64Vec)` raises, run), `_ivec_sum_slow` generator that can only overflow (`.pyx:220-222`), `_Unchecked.mixed` never set (`vectors.py:1503,1471`), unused `node` of `in_lowered_scope`, stale `.pxd:6-7` ("len, index, slice, iteration, ==, buffer, pickle": none exists) | about **20** | none | none | provider unit build |
| C12 | `F64Vec`/`I64Vec` are the same code twice (`.pyx:18-135` and `139-237`) | about **70** with one templated include | none | medium: not compiled | provider unit build |
| C13 | Remove D94 (M11) | 62 + about 12 = **74** | local lists in modules without vector cells stay lists; "identical code elsewhere" becomes true; removes the upstream regression of R1 | low | models whose report has `vector_locals` but no vector cell |

`held_arrays.py` (M14) is not about vectors: it depends on `holds.py` and is best reviewed with table-base.

## 3. Generic

| What | Case | Effect |
|---|---|---|
| No switch | `config.py` has no key for vectors; the only opt-out is a return-type spec per cell (`vectors.py:87`); D94 has none | an upstream user cannot turn the area off |
| Public value changes | `m.S.cf() is m.S.cf()` was `True` (cached list), now `False` (built): `table_cells.py:257-272` returns `tolist()` per call | identity, in-place edits by the user, and O(n) per Python call: a vector cell called from Python is only 2.1x faster than live (measured) |
| Build stops where live works | `x = self.cf(t); x[0] = 0.0` or `.append`: `VectorWriteError` (`vectors.py:1063,1152`) | deliberate, but new for an upstream model |
| Cross-Space reads are slower than live | `projection.vec(n)[i]` in a loop from another Space: `kind()` knows only `self.<cell>` (`vectors.py:571`), and `_vector_native` returns `False` for a subscript of a `V` call (`table_cells.py:459-487`), so each read calls the public method, which builds a list of n floats | **built: 4,000 reads take 0.46 s compiled, 0.001 s live** |
| Fitted spellings | concatenation of comprehensions and `list(range())`; sum leading a chain; `val = [v * g for v in val]` induction (`vectors.py:830-844`); `NAME = range(a, b)` then `NAME.start/.stop` (`706-714`); `ivec_get` in `key_facts.py:357-360`; limits 64 / 2^20 / 128 of M13 | each fires for one diag model's text |
| Rules that do not fire elsewhere | a comprehension with `if`, two `for`s, or not the whole value of a statement (`724-733`, `403,441`); `enumerate(v)`, `v[::2]`, `reversed(v)`, `sorted(v)`; a `sum` whose element is not proven float (`1916`) | reported as fallbacks; each costs a `tolist()` copy per use, so a formula that falls back inside a loop is O(n^2) |
| Loop targets typed in every formula | any formula of a module with one vector cell gets `long long` range targets (`1807,1832`) | not "identical code elsewhere" for such modules |

## 4. Reliable

Blocking by the brief's ranking: R1, R2, R3 (silent, all three built), and the cross-Space read of section 3
(slower than live). R1 is also a regression for unmodified upstream use.

| # | Reproduction | Live | Compiled | Evidence | Fix |
|---|---|---|---|---|---|
| R1 | `out = []`; `out.append(rate(i))` in a loop; `out.sort()`; `return out` (also `.extend`, `.reverse`, `.pop`, `.insert`, `.clear`, `.remove`, and `fill(out)`) | sorted list | **unsorted**: lowered to `out.tolist().sort()`, the copy is sorted and dropped. Fires in a module with no vector cell too (D94): `xs.sort(); return xs[0] + xs[1] * 10.0` gives 10.0, live 0.0 | built: three cells differ from live | trial fix, built: `_Facts.visit_Call` (`vectors.py:517-521`) records a local that is the receiver of a method other than one-argument `append`, or an argument of a call that is not a reading builtin; `vector_kind` keeps it a list (10 lines). Equal to live after. For a cell result (`self.vec(t).sort()`) raise `VectorWriteError` in `_ObjectUses._leave` (4 lines, not run) |
| R2 | Integer cell `[data[z] for z in range(t)]`, or `acc = [0] * t; acc[i] = data[i]`, with `data` a dict Reference; sampled with ints, then the Reference is replaced and one value is `2.5` | list holding `2.5`; sum 31.5 | **`2`**; sum **31**, silently: `r.data[i] = (obj)`, `ivec_set`, `ivec_append` coerce an object to `long long` and Cython truncates a float (`vectors.py:793-794` checks only when `strict_locals`) | built | trial fix, built: provider `ivec_item` (exact int or `TypeError`) wrapped by the lowering wherever the value is not proven `INT` (`1067`, `1106`, `1176`, `1190`, `1352`; 20 lines). After: `TypeError: an integer vector holds integers, not float` |
| R3 | `v = vec(t); return v[2] // 0.1` with `v[2] == 1.0` (any `//` with two proven floats: elements, loop targets) | `9.0` | **`10.0`**: both operands are C doubles and Cython emits `floor(a / b)`, not CPython's remainder-based floor division. The same holds for scalar cells typed `double`, so it belongs to the shared typing too; vectors extends it to list elements | built | not fixed. A provider function with CPython's `float_floor_div` and a rewrite of `//` when `scalar()` is `FLOAT` (about 20 lines, not compiled); `%` agrees (compiled on 3.13) |
| R4 | `return vec(t) if t > 3 else vec(3)` in a float vector cell; `c = a if t > 3 else b` with vector locals | works | `TypeError: Cannot convert list to F64Vec` at run time: `kind()` types the `IfExp` (`608-611`), `destination` passes it (`1202`), then `_ObjectUses` appends `.tolist()` to both branches because `_native` has no `IfExp` case (`1735-1766`; `table_cells.py:476` has one) | built | trial fix, built: the `IfExp` case in `_native` (2 lines); equal to live after. Or C6 |
| R5 | `out = [None] * n`, fill in a loop, `return out` in a float vector cell | works | build stops: `vec_full(t, None)`, Cython "Cannot assign None to double" | lowering; Cython on 3.13 | trial fix, built: `kind()` at `586-593` gives `None` for a `None` or string element, the list stays and `vec_of` converts at the return (3 lines); equal to live after |
| R6 | (withdrawn) Arguments outside the sampled window raise `IndexError` at the cell's entry (`transformer.py:114-126,1668`; built), so `VecCache` never sees them: the "not cached" branch (`.pyx:263-264`) and the key `-1` (`vectors.py:200`) are unreachable; add them to C11 (about 6 lines) | - | - | built | - |
| R7 | `v[1.9]`, `sum(v[0.5:2])`, `range(2.5)` bounds of a lowered comprehension | `TypeError` | index truncated, a value is returned | compiled on 3.13: `vec_get(v, 1.9)` gives element 1 | same checked conversion as R2 for indices not proven `INT` (not run) |
| R8 | `for x in vec(t): pass` then `return x` with `t == 0` | `UnboundLocalError` | an uninitialised C double (`1.9e-311`) | built | initialise, or keep the target a Python object when it is read after the loop (not run) |
| R9 | Three nested `range` loops reading `v[i] * v[j] * v[k]` | - | eight copies of the innermost body (2^depth), each inner guard re-tested per outer iteration | lowering run | version only the innermost loop (3 lines) or C3 |
| R10 | Memory: one `VecCache` slot array per cell *per Space instance*, up to 16 MB per cell and 64 MB per class (`vectors.py:126-127`); "the vectors themselves are counted nowhere" (`102`) | - | the budget is per class, not per item count | read | count instances, or C2 |

Not found: no dependence on thread timing (publication is one compare-and-swap, `.pyx:265-269`; a losing thread
drops an equal value), on call frequency, or on the operating system. The compensated sum matches CPython's
for 3.12 and later and is plain before (`.pxd:104-109`), as CPython is. An unassigned vector local raises
`UnboundLocalError` (compiled on 3.13), not a crash.

## 5. Table commits

Not a table area. For the vectors commit itself the smallest first commit is M1, M2, M4, M5, M6 and M12 with the
dict cache; M3, M7, M9, M10 and M11 can each follow as a separate commit that can be declined.

## 6. Bottom line

Line counts are counted in the files. Speed is measured only on the synthetic model above, not on lifelib.

| Version | Content | Lines of 2,811 |
|---|---|---|
| (a) keeps the measured speed | all mechanisms, minus C1, C4, C6, C10, C11 (about 365 lines), plus the fixes R1-R5 (about 55 lines) | about **2,500** |
| (b) speed within 25-30% | (a) minus C3 (measured +20% on a loop chain), C5, C7, C8, C12, C13 (about 410) | about **2,100** |

`VecCache` (C2, 228 lines) stays in both: without it a hit-dominated cell is 3x slower. It can leave (b), giving
about 1,870, only if the list-cell models read vector cells with arguments rarely; that count was not taken.
Order of work: R1, R2, R3 and the cross-Space read now; R4, R5, C1 and C11 now (each under half a day, no
measurable cost); the rest into the proposal text.
