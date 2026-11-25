# Review of the `vectors` area (reviewer a)

Scope: `vectors.py` 1,950, `vector_ref_reads.py` 184, `runtime/_native_vectors.pxd` 252 / `.pyx` 299,
`held_arrays.py` 126 (2,811 lines), plus the wiring in `tracer.py`, `builder.py`, `transformer.py`,
`table_cells.py`, `admission.py`, `serial_collect.py`.

**How this was checked.** Installing the stack of the brief (free-threaded 3.14, development modelx, lifelib) was
denied by a permission check, so I stopped that action: no model was built and no run time was measured. What
I did run: (1) `vectors.lower` on small formulas with the mock classes of `tests/test_vectors.py`; (2) the
provider `_native_vectors` compiled with Cython 3.3.1 on CPython 3.13 (ordinary build), with hand-written
modules that repeat the lowered text statement by statement. Each finding says which of the two it rests on.
"Read" means by reading only. All line counts of cuts are counted in the files; all speed statements are
estimates, none is measured.

## 1. What the code does

| # | Mechanism | Lines | Lost without it |
|---|---|---|---|
| M1 | Traced shape `V`/`I`/`E`/`('T', kinds)` of a returned list; cell return type becomes `F64Vec`/`I64Vec` | `tracer.py:111-144,466-472`, `builder.py:388-400,570-572`, `vectors.py:77-94` | everything: a list cell stays `object` in a dict cache |
| M2 | Provider: two buffer classes, new/full/of/slice/get/set/append, compensated sum, exact int sum | `.pxd:47-56,91-252`, `.pyx:18-237` | the typed storage; elements are boxed floats again |
| M3 | Dense cache `VecCache` and its budget plan (1-3 integer parameters, sampled window) | `vectors.py:97-222`, `.pxd:69-88`, `.pyx:243-299`, `transformer.py:358,1162,1478,1730`, `serial_collect.py:488` | dict cache for vector cells with arguments (hash and lock per hit) |
| M4 | Facts of one formula and the two typers: vector kind of locals (`kind`, `vector_kind`), scalar kind of expressions (`scalar`) | `vectors.py:385-529,557-750,753-899` | nothing can be lowered; only cells' own returns are converted |
| M5 | Element forms: `v[i]`, `v[i] = x`, `v[i] op= x`, `append`, `len`, `sum(v)`, `sum(v[a:b])`, `v[a:b]`, `[x] * n`, `[]`, tuples of vectors | `vectors.py:918-936,1056-1209,1285-1301,1927-1950` | element reads and writes go through Python objects |
| M6 | Loops: `for x in v`, `for a, b in zip(v, w)`; statement-level list comprehension and `sum(generator)` as C loops | `vectors.py:1303-1371,1401-1447,1891-1924` | per-element boxing in the hot loops (the main part of 21x-32x, estimate) |
| M7 | Loop versioning of `for i in range(a, b)`: one bounds test before the loop, direct buffer access inside | `vectors.py:1449-1556`, `.pxd:184-186` | one unsigned compare per `v[i + c]` (already inline, `.pxd:139-164`) |
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
| C2 | Remove `VecCache` (M3); vector cells with arguments use the existing dict cache | 126 + 57 + 20 + about 25 wiring = **228** | unknown; one dict lookup per hit instead of one load; also removes R6 and the per-instance slot arrays | medium: speed not measured | the models with vector cells that have arguments, one and four workers |
| C3 | Remove loop versioning (M7) | 108 + 4 = **112** | one predictable compare per element; removes the code growth of R9 | low; not measured | the 21x-32x models, run time and generated C size |
| C4 | Remove concatenation (M10, diag-kr): one spelling, falls back to a correct `vec_of(list)` | 7 + 19 + 3 + 56 = **85** | one list per call of the cells that use it | low | the diag-kr model |
| C5 | Powers: delete the second rewrite (M9) and give `powers.py` the element kinds through its existing `operand_kind` hook (`transformer.py:735`) | about **80** | none expected; removes D91 arbitration (`vectors.py:938-979`) | medium: not compiled | models where the report has `powers > 0` |
| C6 | One "native use" predicate instead of two (M12): `_ObjectUses._native` (`vectors.py:1735-1766`) and `RenameMembers._vector_native` (`table_cells.py:442-487`) | about **35** | none; they already differ (R4 exists because only one handles `IfExp`) | low | unit tests of both files |
| C7 | `min`/`max` of a generator (M10) duplicates `reductions.py:109,295-316` with a different typer | 26 + 7 + 8 = **41** | reductions must accept vector operands instead | medium: not compiled | models with `extreme_calls` in the report |
| C8 | Leading sum in a chain (D93): one spelling | 8 + 17 + 6 = **31** | that sum stays a generator | low | the diag-us2 model |
| C9 | `_CheckedFloatReads` repeats the compare that inline `vec_get` already does (`.pxd:139-143`) | **28** | none expected (same C after inlining); not measured | low | compare generated C of one model |
| C10 | `VecSum` is a "private copy" of `_float_sum_state` (`.pxd:9-11,91-115`) | **25** | none | low: a `cimport` across providers | unit tests of sums |
| C11 | Dead or broken: `VecCache.__reduce__` and `_vec_cache_from_items` (`.pyx:290-299`; pickling a vector raises `TypeError`, run), `vec_of` branch for `I64Vec` (`.pyx:93-94`; `list(I64Vec)` raises, run), `_ivec_sum_slow` generator that can only overflow (`.pyx:220-222`), `_Unchecked.mixed` never set (`vectors.py:1503,1471`), unused `node` of `in_lowered_scope`, stale `.pxd:6-7` ("len, index, slice, iteration, ==, buffer, pickle": none exists) | about **20** | none | none | provider unit build |
| C12 | `F64Vec`/`I64Vec` are the same code twice (`.pyx:18-135` and `139-237`) | about **70** with one templated include | none | medium: not compiled | provider unit build |
| C13 | Remove D94 (M11) | 62 + about 12 = **74** | local lists in modules without vector cells stay lists; "identical code elsewhere" becomes true; removes the upstream regression of R1 | low | models whose report has `vector_locals` but no vector cell |

`held_arrays.py` (M14) is not about vectors: it depends on `holds.py` and is best reviewed with table-base.

## 3. Generic

| What | Case | Effect |
|---|---|---|
| No switch | `config.py` has no key for vectors; the only opt-out is a return-type spec per cell (`vectors.py:87`); D94 has none | an upstream user cannot turn the area off |
| Public value changes | `m.S.cf() is m.S.cf()` was `True` (cached list), now `False`: `table_cells.py:257-272` returns `tolist()` per call | identity, in-place edits by the user, and O(n) per Python call |
| Build stops where live works | `x = self.cf(t); x[0] = 0.0` or `.append`: `VectorWriteError` (`vectors.py:1063,1152`) | deliberate, but new for an upstream model |
| Cross-Space reads are slower than upstream (read) | `self.asmp.rates()[t]`: `kind()` knows only `self.<cell>` (`vectors.py:571`), and `_vector_native` returns `False` for a subscript of a `V` call (`table_cells.py:459-487`), so the public method builds a list of n floats per element read | O(n) per read where upstream returned the cached list |
| Fitted spellings | concatenation of comprehensions and `list(range())`; sum leading a chain; `val = [v * g for v in val]` induction (`vectors.py:830-844`); `NAME = range(a, b)` then `NAME.start/.stop` (`706-714`); `ivec_get` in `key_facts.py:357-360`; limits 64 / 2^20 / 128 of M13 | each fires for one diag model's text |
| Rules that do not fire elsewhere | a comprehension with `if`, two `for`s, or not the whole value of a statement (`724-733`, `403,441`); `enumerate(v)`, `v[::2]`, `reversed(v)`, `sorted(v)`; a `sum` whose element is not proven float (`1916`) | reported as fallbacks; each costs a `tolist()` copy per use, so a formula that falls back inside a loop is O(n^2) |
| Loop targets typed in every formula | any formula of a module with one vector cell gets `long long` range targets (`1807,1832`) | not "identical code elsewhere" for such modules |

## 4. Reliable

Blocking by the brief's ranking: R1, R2, R3. R1 is also a regression for unmodified upstream use.

| # | Reproduction | Live | Compiled | Evidence | Fix |
|---|---|---|---|---|---|
| R1 | `out = []`; `out.append(self.rate(i))` in a loop; `out.sort()`; `return out` (also `.extend`, `.reverse`, `.pop`, `.insert`, `.clear`, `.remove`, and `self.fill(out)`) | sorted list | **unsorted**: lowered to `out.tolist().sort()`, the copy is sorted and dropped. Fires in a module with no vector cell too (D94): `xs.sort(); return xs[0]` returns the wrong element | lowering run; compiled equivalent returned `[0,2,4,1,3,0]` against `[9,4,3,2,1,0]` | in `_Facts.visit_Call` (`vectors.py:517-521`) add `('other',)` for a local that is the receiver of any method other than one-argument `append`, or an argument of a call that is not a pure builtin: it stays a list (9 lines; run in the lowering harness on the reproductions: gives `out = []` ... `return vec_of(out)`; unit tests not run). For a cell result (`self.vec(t).sort()`) raise `VectorWriteError` in `_ObjectUses._leave` (4 lines, not run) |
| R2 | Integer cell: `return [self.x(z) for z in range(t)]`, or `out = [0] * n; out[i] = self.x(i)`, with `x` untyped; sample saw ints, a later input gives `2.5` | list holding `2.5` | **`2`**, silently: `r.data[i] = (obj)` / `ivec_set(v, i, obj)` / `ivec_append(v, obj)` coerce an object to `long long`, and Cython truncates a float. Depends on the traced sample. Same for a local `[0] * n` written with an untyped value in a vector-cell module (`vectors.py:793-794` checks only when `strict_locals`) | lowering run (`_mx_v6_r.data[i] = (self.thing(z))`); compiled: sum 45 against 52.5, `ivec_set(v, 0, 2.5)` stores 2 | export `_exact_int` (`.pyx:185-190`) as `ivec_item` and wrap the value in the lowering wherever `scalar(value) != INT`: `1067`, `1106`, `1176`, `1352`, `1190` (about 12 lines, not compiled); or drop `self.strict_locals and` at `794` and send `'A?'` integer comprehensions through `ivec_of` (slower, 6 lines, not compiled) |
| R3 | `v = self.vec(t); return v[0] // 0.1` with `v[0] == 1.0` (any `//` with two proven floats: elements, loop targets) | `9.0` | **`10.0`**: both operands are C doubles and Cython emits `floor(a / b)`, not CPython's remainder-based floor division. The same holds for scalar cells typed `double`, so it belongs to the shared typing too; vectors extends it to list elements | compiled equivalent: literal divisor 10.0, object divisor 9.0 | a provider function with CPython's `float_floor_div` and a rewrite of `//` when `scalar()` is `FLOAT` (about 20 lines, not compiled); `%` already agrees (run) |
| R4 | `c = a if t > 3 else b` with vector locals; `return self.vec(t) if t > 0 else self.vec(0)` in a float vector cell | works | `TypeError: Cannot convert list to F64Vec` at run time: `kind()` types the `IfExp` (`608-611`), `destination` passes it (`1202`), then `_ObjectUses` appends `.tolist()` to both branches because `_native` has no `IfExp` case (`1735-1766`; `table_cells.py:476` has one) | lowering run and compiled | add the `IfExp` case to `_native` (3 lines), or C6 |
| R5 | `out = [None] * n`, fill in a loop, `return out` in a float vector cell | works | build stops: `vec_full(t, None)`, Cython "Cannot assign None to double" | lowering run, Cython run | `kind()` at `586-593`: an element that is not typed and not a number literal gives `None`, the list stays and `vec_of` converts at the return (3 lines, not run) |
| R6 | Vector cell `v(t)` reading `v(t - 1)`, called with `t` beyond the sampled window | cached | never cached outside the window: `put` returns the value unstored (`.pyx:263-264`), multi-parameter key `-1` (`vectors.py:200`). k calls past the window cost O(k^2) formula runs, exponential with two recursive reads. Run time depends on what the sample reached | provider run (`cache_get` after `put(7)` in a window of 4 is `None`) | C2, or a dict for keys outside the window (about 15 lines, not compiled) |
| R7 | `v[1.9]`, `sum(v[0.5:2])`, `range(2.5)` bounds of a lowered comprehension | `TypeError` | index truncated, a value is returned | compiled: `vec_get(v, 1.9)` gives element 1 | same checked conversion as R2 for indices not proven `INT` |
| R8 | `for x in v: pass` then `return x` with `v` empty | `NameError` | an uninitialised C double (`6.9e-310` in the run) | compiled, C compiler warns | initialise or keep the target a Python object when read after the loop |
| R9 | Three nested `range` loops reading `v[i] * v[j] * v[k]` | - | eight copies of the innermost body (2^depth), each inner guard re-tested per outer iteration | lowering run | version only the innermost loop (3 lines) or C3 |
| R10 | Memory: one `VecCache` slot array per cell *per Space instance*, up to 16 MB per cell and 64 MB per class (`vectors.py:126-127`); "the vectors themselves are counted nowhere" (`102`) | - | the budget is per class, not per item count | read | count instances, or C2 |

Not found: no dependence on thread timing (publication is one compare-and-swap, `.pyx:265-269`; a losing thread
drops an equal value), on call frequency, or on the operating system. The compensated sum matches CPython's
for 3.12 and later and is plain before (`.pxd:104-109`), as CPython is. An unassigned vector local raises
`UnboundLocalError` (run), not a crash.

## 5. Table commits

Not a table area. For the vectors commit itself the smallest first commit is M1, M2, M4, M5, M6 and M12 with the
dict cache; M3, M7, M9, M10 and M11 can each follow as a separate commit that can be declined.

## 6. Bottom line

Both figures are estimates from reading; the run-time side of each was not measured (stack install denied).

| Version | Content | Lines of 2,811 |
|---|---|---|
| (a) keeps the measured speed | all mechanisms, minus C1, C4, C6, C9, C10, C11 (about 390 lines), plus the fixes R1-R5 (about 35 lines) | about **2,450** |
| (b) speed within 25-30% | (a) minus C2, C3, C5, C7, C8, C13 (about 565) and C12 (about 70) | about **1,800** |

The risk in (b) is C2 (dict cache instead of `VecCache`) and C3 (loop versioning): each needs one timing on the
models with list cells before it is written into the proposal. Order of work: R1, R2, R3 now; R4, R5, C1 and C11
now (each under half a day, no measurable cost); the rest into the proposal text.
