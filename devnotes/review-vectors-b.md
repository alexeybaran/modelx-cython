# Review r4, area `vectors`, reviewer b

Tree: `review/r4-vectors-b` (= b326334). Stack: CPython 3.14.6 free-threaded, Cython 3.3.1, modelx 636877b,
lifelib 04d69db (source read only; no lifelib model built). Every finding marked *repro* was built from a
2-4 cell model with `python -m modelx_cython exported` and compared with the exported (live) model on the
same calls. Every fix marked *applied* was applied to the tree, the repro re-run, and these 99 tests passed:
`test_vectors`, `test_vector_extrema_typing`, `test_tracer`, `compiled/test_{integer_vector_cells,
leading_vector_sum, local_vectors_without_vector_cells, power_memo_with_vectors, vector_cells,
vector_displays, vector_extrema_and_cache, vector_python_exits, vector_ref_reads, held_array_reads}`.
The fixes were then reverted: this branch holds only this report. Nothing here was timed on a lifelib model.

## 1. Mechanisms (lines on this branch) and what is lost without each
| # | Mechanism | Where | Lines | Lost without it | Users (grep of the 70 lean sources) |
|---|---|---|---|---|---|
| M1 | Traced shape 'V'/'I'/'E'/('T',..) and merge | tracer.py:108-145, 464-472; vectors.py:41-91 | 95 | everything below | all list cells |
| M2 | F64Vec/I64Vec, get/set/append/full/of/slice, `_clip`, sums | _native_vectors.pyx:1-237, .pxd:44-252 | 480 | typed cached value; reads go through list objects | KR, JP, FR, DE, UK list models |
| M3 | VecCache (dense, CAS publish) + plan/budget, multi-param slot | vectors.py:93-222; pyx:240-299; pxd:59-82 | 215 | vector cells fall back to the dict cache + lock | every vector cell with int params |
| M4 | Element typer (`kind`, `elem`, `scalar`, NUM, math) + fixpoint `analyse` | vectors.py:533-900 | 370 | no typed locals at all | all |
| M5 | Fact collector `_Facts` | vectors.py:383-530 | 150 | (needed by M4) | all |
| M6 | Element ops: `v[i]`, `v[i]=`, `v[i] op=`, append, `len`, `sum(v)`, `sum(v[a:b])`, `v[a:b]` | vectors.py:918-936, 1056-1210 | 190 | boxed element access | all |
| M7 | Comprehension / generator-sum expansion to C loops | vectors.py:1303-1372, 1891-1925 | 105 | lists built then copied (`vec_of`) | IP_UK_S, ADE_FR_S, Medical_JP_S, Annuity_JP_S |
| M8 | `for x in v` / `zip(v, w)` loops | vectors.py:1401-1447 | 47 | object iteration | Dep_FR_S, BU_DE_S, ... |
| M9 | Loop versioning (`_Unchecked`, hoisted bounds) | vectors.py:1449-1557 | 110 | one inlined unsigned compare per read (M6 keeps it) | KR interpolation loops |
| M10 | Range/derived index locals as `long long`, `_CheckedFloatReads` | vectors.py:1559-1679 | 120 | Cython infer_types still types most of them | all vector modules |
| M11 | D154 `_ObjectUses`: vector reaching Python code becomes `tolist()` | vectors.py:1685-1794 | 110 | correctness (Python would see an F64Vec) | all |
| M12 | Tuples of vectors ('T') | scattered in M4/M6, `wrap_return` | ~40 | tuple cells stay lists | Dep_FR_S, BU_DE_S |
| M13 | Powers on vector scalars → `_mx_pow` | vectors.py:938-998 | 60 | Cython complex power path | vector element powers |
| M14 | D87 displays `[a, b]` of typed scalars | vectors.py:1211-1226 | 16 | list then `vec_of` | few |
| M15 | D94: typed local lists in modules *without* vector cells | vectors.py:226-290, strict_locals | 75 | Sofort_DE_S bisection loop over `disc/dens/paid` stays boxed | Sofort_DE_S (and any upstream model with `[0.0]*n` or `[]`+append) |
| M16 | diag-kr concatenation `[..] + list(range(..))` | vectors.py:340-347, 423-437, 1228-1283 | 80 | two lists then one copy, once per cached t | CI_KR_S, one cell |
| M17 | diag-kr min/max of a generator over vectors | vectors.py:1373-1399 + 1891-1925 part | 40 | generator on Python objects | Annuity_JP_S, Cancer_KR_S |
| M18 | D93 leading sum of a `+ - * /` chain | vectors.py:1873-1890 + part of 1092-1146 | 35 | generator sum on objects | VUL_US_S, VA_KR_S (per its test) |
| M19 | `vector_ref_reads.py` (opt-in spec key) + `ivec_small_for_reads` | vector_ref_reads.py:1-184; pxd:52-60; transformer.py:743-746 | 197 | DV01 `Market.shock_weight` keeps a Python list | DV01 only (its native_spec) |
| M20 | `held_arrays.py` (typed memoryview held per item) | held_arrays.py:1-126 | 126 | one shared-acquisition slice copy per read | Term_US_SEC; not a list feature |

## 2. Complexity: cuts (none compiled on a lifelib model, none timed)
| Cut | Lines | Expected cost | Risk | Verify (≤10 models) |
|---|---|---|---|---|
| C1 Drop M19 from the vectors commit (DV01-only, opt-in, own scalar-tail interpreter `_ScalarTail` that re-implements a subset of the typer) | 197 + 117 test | DV01 `price` path only; unmeasured | none for the 69 others (key off by default) | DV01 t2 with/without key |
| C2 Move M20 to table-base/holds (it reads memoryview cells, uses `holds.hold_table`, nothing of M1-M18) | 126 out of this commit | 0 | none | Term_US_SEC unchanged C |
| C3 Drop M16 (one cached cell per t in CI_KR_S; fallback is `vec_of` of a Python list) | 80 | one list build per t of one cell; expect <0.5% | low | CI_KR_S t2 |
| C4 Replace `VecSum` (pxd:85-112) by `_float_sum_state.F64Sum` (byte-identical algorithm; pxd itself says "to be unified") | 28 | 0 | low: provider packaging must ship the pxd | C diff of a sum site |
| C5 `_ivec_sum_slow` (pyx:220-223): generator sum only to raise OverflowError; raise directly | 3 | 0 | none | test_integer_vector_cells |
| C6 Stale docs: pxd:6-7 claims a sequence protocol F64Vec no longer has (D154 removed it); pxd:21 names removed `_native_inputs` | 0 (text) | 0 | changes provider C text only | — |
| C7 (b only) Drop M9 loop versioning: reads keep `vec_get`, an inline `(unsigned)i < n` compare that is always predicted | 110 | est. ≤5% on KR interpolation; unmeasured | low | Child_KR_S, LTC_KR_S, Medical_KR_S |
| C8 (b only) Drop M17 and M18 (four models; fallback is object generator over `tolist()`) | 75 | unknown, per-call generator on those cells | low | Annuity_JP_S, Cancer_KR_S, VUL_US_S, VA_KR_S |
| C9 (b only) Drop multi-param VecCache slot (`_cache_key`, 192-202) and keep 1-param | 25 | multi-param vector cells go to dict cache | low | models whose report shows multi-param `vector-dense` |

Duplication of Cython/CPython: M2's `_clip` and `_vec_get_slow` re-implement list slicing/indexing semantics
(needed, since the buffer is not a list). F64Vec/I64Vec are copy-pasted twins (~90 lines); a fused type cannot
parameterise two cdef classes, so no cheap cut. No duplication of pandas or upstream found.

## 3. Generic: fitted to one spelling or behaviour change for upstream users
- M15 (D94) runs in modules **without** vector cells: an unmodified upstream model with `v = []` +
  `v.append(<float>)` or `v = [0.0] * n` now gets a typed C buffer. That is the path of defect R1 below.
- M16/M17/M18 are each fitted to one or two formulas (comments name diag-kr/diag-us2 and the models). A
  `[a for ..] + [b for ..]` concatenation fires; `[..] + [x]` (a display) or `a + list(range(n), )` with a
  keyword does not — fine as fallbacks, but they buy one model each.
- 'E' (empty list in every trace) is normalised to a **float** vector (vectors.py:77-79). A cell that returned
  `[]` in the whole sample and ints later raises for ints above 2**53 (repro) and, by `_exact_float`
  (pyx:69-82, read), returns floats for small ints (`[6.0]` for live `[6]`): sample-dependent type change.
- M19 keys on `type(self) is cls` and a ±2**20 value window: correct, but generic only in name.

## 4. Reliable: differences from live (all *repro*, smallest first in severity order)
| # | Case (minimal) | Live | Compiled | Kind | Fix (applied, tested) |
|---|---|---|---|---|---|
| R1 | `v = []; v.append(0.5*(3-i)+t)` in a loop; `v.sort(); return v[0]` (also `reverse`, `extend`, `pop`; module with **no** vector cells) | 1.5 | 2.5 | **silent, sample-free, upstream regression** | vectors.py:517-521 `_Facts.visit_Call`: a method call other than `append` on a local name marks it `('other',)` (2 lines) |
| R2 | `base(t) = [5_000_000_007 + t, 3]`; `m(t): v = base(t); return (v[0]*v[0]) % 1000` (default `integer_checks: auto`) | 49 | 433 | **silent** (int64 wrap) | tracer.py:943-947 `integer_magnitude`: count list/tuple elements (`if type(value) in (list, tuple): return max(map(integer_magnitude, value), default=0)`, 2 lines) → build goes checked → OverflowError |
| R3 | module with a vector cell; `v = [0.0]*2; v[0] = big(t)` with `big` an int > 2**63; `return int(v[0]) - big(t)` | 0 | -1 | **silent** (int rounded into a double) | vectors.py:788-795: apply the D94 strict test in every module (`self.scalar(value) != FLOAT`/`!= INT`, drop `strict_locals and`); cost on models unmeasured — check the `vector_fallbacks` lists of Dep_FR_S, BU_DE_S, IP_UK_S, ADE_FR_S, DIA_US_S stay empty |
| R4 | `for x in vc(t): pass; return x` with `vc(t) == []` | UnboundLocalError | 3.0e-311 (uninitialised C double) | silent value where live raises | vectors.py:1856-1861: declare the C scalar only when the target is not read outside the loop; not built. (`for i in range(0)` gives 0 vs UnboundLocalError even **without** vectors: general typed-local issue, rest area) |
| R5 | `sum(w(t, s=s) for s in iv(t))`, `[w(t, s=s) for s in iv(t)]` | 1.5 | TypeError: unexpected keyword `_mx_v8_ss` on **every** call | raises where live works, sample-free; build passes | vectors.py:375-381 `_Rename`: keep `Attribute.attr` and `Arg.keyword` (`leave_Attribute`, `leave_Arg`, 4 lines) |
| R6 | `v = [0]*3; v[0] += 0.5` (int vector local, float update) | 1.0 | build stops: Cython "Cannot assign type 'double' to 'long long'" | build stop where live works | vectors.py:454-459 record `(base, value)` of `v[i] op= x`; in `analyse` an `'I'` local updated with a value not proven INT stays a list (6 lines) |
| R7 | float vector cell returning an int vector cell on an unsampled path: `fv(t) = iv(t) if t > 5 else [0.5*t]`, sample t<3 | `[7, 8]` | TypeError: I64Vec not iterable | raises (sample-dependent) | _native_vectors.pyx:93-94 `value = (<I64Vec>value).tolist()` (1 line); then `[7.0, 8.0]`, equal as numbers |
| R8 | `x ** y` on vector floats overflowing (`10.0 ** 400.0`) | OverflowError | inf | silent inf | shared with powers.py (`_mx_sys.pxd:249-251` documents it); not vectors-only; not built |

Equal to live by reading the code against CPython 3.14 (not run): compensated `sum` (first element via
`0 + x`, inf/nan, -0.0), min/max over generators with NaN and ±0.0, zip of unequal lengths, slice
normalisation, negative and out-of-range indexes, loop versioning on empty and negative ranges. Run: a
vector cell called outside its sampled window raises IndexError like dense caches (no silent uncached path). Not checked: concurrent first fill of one `VecCache` slot under
free threading (CAS path read only), Python 3.10-3.13 builds.

Not dependent on the sample: R1, R3, R5, R6. Dependent on the sample: R2 (which mode the build picks), R7, 'E'.

## 5. Table areas
Not applicable.

## 6. Bottom line
| | Lines (area now 2,811: vectors.py 1,950 + vector_ref_reads 184 + runtime 551 + held_arrays 126) | Content |
|---|---|---|
| (a) measured speed kept | **~2,380** (−431: C1 197, C2 126 moved out, C3 80, C4 28; R1-R7 fixes +~20) | M1-M15, M17, M18 |
| (b) within 25-30% (unmeasured) | **~2,170** (further −210: C7 110, C8 75, C9 25) | M1-M8, M10-M15 |

Blocking, fix now (each under 10 lines, applied and tested here): R1 (silent, upstream regression via D94), R2
(silent wrap; integer vectors invisible to `integer_checks: auto`), R3 (silent rounding of an object write),
R5 (keyword/attribute renamed in expanded loops: every call raises). Same day: R6, R7, C4, C5, C6. Proposal text
(not now): R4 needs a rule on loop-target reads after the loop shared with the typed-local work; whether
M9/M17/M18 earn their lines must be timed on the models named in §2 before the vectors commit is cut.
