# Review: non-local logic (cross-cell, sample values, counts) - reviewer a

Stack used: CPython 3.14.6t, Cython 3.3.1, modelx 636877b, gcc 13.3, Linux x86-64. Seven small models were built one
at a time (scripts not committed; every model is given below in full). "ran" = I built and ran it; "read" = taken from
the code and not run. Line counts are function ranges, +-20%. Paths are under `modelx_cython/`.
Abbreviations: FR `frame_reads.py`, RC `row_cells.py`, RN `read_notes.py`, V `vectors.py`, TC `table_cells.py`,
EP `entry_pipeline.py`, IE `inline_entries.py`, B `builder.py`, TR `transformer.py`, T `tracer.py`.

## 0. Blocking: silent differences from live that depend on the sample or on another cell (all ran)

| # | Model (Space P unless said) | Sample | Run | Live | Compiled | Cause |
|---|---|---|---|---|---|---|
| D1 | Ref `rate = 2`; `half = lambda i: rate * i` | `half(0..4)` | `m.P.rate = 0.5; half(3)` | 1.5 | **0** | Reference typed `long long` from the one sampled value (B:860-918, TR:406); Cython attribute store truncates, no check |
| D2 | `Data.df` has int64 column `a`; `Data.T = lambda: df`; `P[pid].ia = lambda t: data.T().loc[pid, "a"] + t` | `P[10].ia(0..9)` | replace `df` by a frame whose `a` is 1.5, 2.5, 3.5; `P[30].ia(1)` | 4.5 | **4** | column kind from the sampled frame (T:663-711); the pandas fallback is cast to the sampled C type (FR:1503-1507) |
| D3 | `rates = lambda: [0.03*t for t in range(5)]`; `srt`: `r = rates(); r.sort(reverse=True); return r[0]` | `srt()` | `srt()` | 0.12 | **0.0** | sampled list of floats makes `rates` a native vector; `.sort` acts on a `.tolist()` temporary (V:1736-1775). `.append` stops the build, `sort/extend/pop/insert/reverse/del` do not |
| D4 | Ref `Kind` (IntEnum), Ref `tab = [Kind.TERM]`; `kind = lambda: tab[0]`; `is_term = lambda: kind() is Kind.TERM` | `is_term()` | `m.V.tab = [1]; is_term()` | False | **True** | `is` lowered to `==` because every sampled result of ANOTHER cell was a member (enums.py:403-456). With one plain int in the sample the test stays `is` and is always False, also where live is True (read) |
| D5 | Ref `scale = 10`; `cube = lambda i: scale*scale*scale*scale*scale*scale*scale*(i+1)` | `cube(0..4)` | `m.P.scale = 1000000; cube(2)` | 3e42 (int) | **-1473332387080306688** | model-wide `integer_checks: auto` chose unchecked: no sampled integer reached 2**31 (config.py:113-118). Documented as a limit, but it breaks the rule "equal or raises" |
| D6 | `isb = lambda x: 0 if isinstance(x, bool) else x` | `isb(1..3)` | `isb(True)` | 0 | **1** | parameter typed from the sample; fold at integer_boundaries.py:234-246 (upstream gives 1 too: typed parameter) |
| D7 | `def z(n): for i in range(n): pass; return i` | `z(1..3)` | `z(0)` | UnboundLocalError | **0** | `infer_types=True` types `i` as C (not caused by the vector switch: same with no vector cell) |

Fixes (not compiled by me unless said): D1 a checked setter per typed Reference (`_mx_exact_int64` exists, ~30 lines)
or type a Reference only when declared; D2 drop the cast on the fallback branch and let the result take the boundary
check that already raises for cells (`_mx_integer_result`, seen raising in my E2 run), ~10 lines; D3 treat every
attribute call other than the known readers on a borrowed vector as a write (V:1062-1161, ~10 lines: build stops);
D4 lower `is` only when the producer is declared enum-valued, otherwise stop the build (an `is` between an erased
integer and a member cannot be right), ~15 lines; D5 make `checked` the default and `unchecked` a declaration
(measured cost below); D6 delete the fold (20 lines) or raise on bool at the boundary; D7 initialise or keep such
locals as objects when read after the loop.

Raises where live works, sample-dependent, guarded (ran): `r = lambda i: tab[i]` sampled on ints, run on 2.5 ->
`TypeError P.r requires an exact signed 64-bit integer`; `ages = lambda: [x for x in src]` sampled empty -> float
vector, run with ints -> TypeError in the reader; duplicate index at run time where the sample was unique ->
`TypeError must be real number, not Series`. These obey the rule (they raise) but each depends on what the sample held.

## 1. Mechanisms (what, where, lines, what is lost without it)

Kind: a = other cells/Spaces/module, b = sampled values, c = counts. Up = exists upstream.

| # | Mechanism | Where | Kind | Lines | Decision it drives | Lost if decided from the cell alone |
|---|---|---|---|---|---|---|
| 1 | Range shared by parameter tuple | B:1005-1017, 1102-1138 | a+b, Up | 80 | array size and IndexError threshold of every cell of the Space with the same parameter names | nothing with a declared size; without one the cell needs a growable store |
| 2 | Sampled bound on hash/dict caches | TR:121-131, 1746; B:728-737 | b | 25 | IndexError beyond sampled max+1 for cells upstream left unbounded | nothing (it only adds an error) |
| 3 | Sampled window + spill, shifted cache | automatic.py:19-42; B:645-770; TR:1653-1699 | b | 200 | physical window, dict spill, negative offset | memory only (window); estimated, not measured |
| 4 | Storage by use; hash capacity; `PREPARE_MIN_EVALUATIONS=8` | automatic.py:88-126; numeric_cache.py:72-84; FR:317-320, 1052; T:218, 341-349 | c | 75 | dense vs hash above 65,536 entries; initial capacity; item-owned frame lowered only if the reader ran >= 8 times per receiver | speed for large sparse rectangles (brief: one mechanism; not measured by me) |
| 5 | Integer mode for the model | config.py:106-118; T:943-966, 1086-1090; cli.py:310-320 | a+b | 150 | `overflowcheck` on every module, checked powers | 1.27x on an integer-only toy (ran: 143 -> 181 ms); brief says 1.00-1.02x for the area |
| 6 | Reference typed from the first traced instance | B:860-918, 1038-1054; T:535-573, 1120-1126 | b, Up | 80 | C type of every read of the Reference in every cell | typed reads of scalars; D1 |
| 7 | Cell argument/result types from traces | T:193-340, 382-520; B:265-590 | b, Up | ~600 | every C type, cache kind, cdef vs def | the compiler itself; an annotation is the local form |
| 8 | Memoryview by usage | usage.py:296-422, 511-531, 671-705; B:480-515 | a, Up | 590 | `const double[:]` vs `object` result from every call site in the model | element reads of arrays |
| 9 | Keyword call anywhere demotes same-named cells | parser.py:292-304; cli.py:366-378 | a, Up | 30 (+ uses) | `def` instead of cpdef; here also removes entries, hits, typed sums, collect leaf | nothing if matched by resolved callee instead of by name |
| 10 | Row-cell inlining decided by all readers | RC:155-265, 312-410; TR:752-777, 850-893 | a | 470 | `R()["f"]` read natively only if no reader in the module uses `R()` otherwise, and no other class mentions the name | brief: row fields are 1,358 of the read sites; per-reader lowering keeps them |
| 11 | Prefix cells, published frames, name uniqueness in the module | TR:779-848; FR:323-382, 820-846; TC:177-191 | a | 320 | lowering onto the parent table; `_mx_prepared_n` sharing; rename of table and vector cells | per-item binding instead of shared (speed, not measured) |
| 12 | Read recorder and vetoes | RN:109-211, 427-476; FR:2366-2395, 1745-1760 | b | 520 | a read whose sampled key selected rows, raised, or gave a non-number stays on pandas; unreached reads are lowered from text | brief: translation 1.5x slower with it; object-level reads have no run-time guard otherwise |
| 13 | Frame schema (levels, unique, column kinds) | T:663-711; FR:294-321, 2466-2480 | b | 50 (+FR) | lowered or pandas; double vs long long; D2 | all table speed unless declared |
| 14 | Key kinds from other cells' traced types | key_facts.py:249-289, 337-343 | a+b | 190 | `is_label` guard emitted or not, typed key code | a guard per read (not measured) |
| 15 | Holds, held arrays | holds.py; FR:588-616, 881-981; held_arrays.py | a | 420 | per-item hold of table, level code, row, array slice; needs item class + arguments | per-read lookup (not measured) |
| 16 | Vector cell from sampled shape | T:111-145, 466-472; V:77-94; B:388-400 | b | 60, gates V | list result stored as F64Vec/I64Vec; empty sample means float | brief: 21x-32x on list models |
| 17 | Module switch: any vector cell types loops and locals of every formula | V:225-286, 1802-1840 | a | 200 | `for i in range` typed, `**` lowering, local lists typed, `strict_locals` | nothing if applied always or per formula |
| 18 | Vector borrow or copy by reader; VecCache budget across cells | TC:442-502; V:105-179, 1693-1793 | a+b | 350 | same class: borrowed; other Space: full copy per call; 64 MB per-Space budget, largest first | - |
| 19 | Enum admission, constants, `is`, argument domain | enums.py:13-33, 86-169, 298-456, 582-624; T:310-325, 542-548 | a+b | 560 | C constants; `is` -> `==`; cache domain = whole class | typed enum compares (speed); D4 |
| 20 | Private entries, exact receiver | IE:16-40, 100-233, 260-402; EP:312-351, 412-483 | a | 440 | `self.g(a)` -> inline cache entry under `type(self) is C` | part of the SEC compiler speed; guarded per site |
| 21 | Range proofs, one-axis clone, cache hits | EP:21-161, 195-288, 367-444, 468-503; cell_entries.py:98-146 | a+b | 290 | caller skips the callee's bounds test when shifted caller window lies in callee window; needs model-wide textual count of `._f_name` | a compare per call (not measured) |
| 22 | Locked entries | locked_entries.py:20-201 | a | 200 | sibling read without re-taking the lock | lock re-entry per read |
| 23 | Integer gate `__getattribute__` | integer_boundaries.py:39-83, 136-152 | a+b | 70 | one sibling with a sampled integer parameter gives the class a wrapper on every Python read | it is the guard |
| 24 | Collect: reach analyses, shared item, bound margin | serial_collect.py:167-354, 443-463, 507-587; factory_reuse.py:115-271; numeric_cache.py:39-69 | a+b | 900 | fast collect on or off from the text of every reachable cell; `margin` from two global sampled maxima | collect piece of the proposal; opt-in keys already exist |
| 25 | Item locals, Space results, vector Ref reads | item_locals.py; space_results.py; vector_ref_reads.py | a+b | 580 | typed item from the sampled class of a Reference or of a producer's result | guarded by exact-type tests |
| 26 | Operand kinds from other cells (powers, sums, rounding, providers) | powers.py:114-141, 225-246; reductions.py:937-992; helper_members.py:25-98; provider_manifest.py:41-71 | a+b | ~900 | `_mx_pow`, typed sum, math provider | brief: reductions 1.13x-4.22x |

## 2-3. Complexity and stability: the flips (ran unless marked)

| # | Smallest unrelated change | Before | After | Surprise (0-3) |
|---|---|---|---|---|
| 1 | add cell `unrelated = lambda t: 2.0*t` and sample `unrelated(60)`; `a`, `b` sampled to 10 | `b(11)`, `a(50)`: IndexError | both return the live value | 3 |
| 3 | sample `b(10)` only | `b` stored as window [10,11) + dict spill, logical [0,11) taken from `a` | - | 1 |
| 4 | project 7 steps instead of 8 per policy in the sample (read) | item-owned frame reads native | all on pandas; with `strict_tables` the build stops | 3 |
| 5 | add Reference `unrelated_id = 3_000_000_000` | `overflowcheck=False`, 143 ms | `overflowcheck=True`, 181 ms, and D5 turns from a wrong value into OverflowError | 3 |
| 6 | sample holds `rate = 2` instead of `2.0` | `half(3)` = 1.5 after `rate = 0.5` | 0 (D1) | 3 |
| 8 | add reader `tot = lambda: data.arr().sum()` | `Data.arr` returns a memoryview to Python callers, `el` reads natively | returns ndarray; `return_type` const double[:] -> object | 2 |
| 9 | `np.sum(x, axis=0)` anywhere (read) | cells named `sum`... any cell whose name equals a keyword-called attribute is cpdef | Python method, entries and typed sums off | 2 |
| 10 | add `dbg = lambda: R()` | `row_cell_inlined: true`, `x.row_fields_inlined: 1` | `false`, `0`, reason "the value is used" | 2 |
| 11 | add a member named `T` to another class of the module (read) | shared prepared frame, cell renamed | per-item bind, walks not lowered; a vector cell then returns F64Vec to Python | 3 |
| 13 | D2; duplicate index in the sample (read): every read of that frame stays pandas | | | 3 |
| 16 | one integer premium in the sample (read) | list result is a native vector | Python list for all readers | 2 |
| 17 | add `w = lambda: [1.0, 2.0]` to any class of the module (read) | loop variables are objects | `long long`; `strict_locals` changes an int result to float | 2 |
| 19 | one plain int among sampled results of `kind` | D4 | `is` always False | 3 |
| 21 | sample stops calling `h(120)` (read) | `g` reads `h` without bounds test | checked call; a same-named cell in another module also removes it | 1 |
| 24 | one local assignment in any reachable cell (read) | reuse collect | fresh root per policy | 2 |

Complexity in one sentence each, as an author would need it: #1 "the largest argument any cell with the same
parameter names received in your sample is the limit for all of them"; #4 cannot be explained without the numbers
8, 16 and 65,536; #10 "a row cell is fast only if nobody anywhere uses the row as a value"; #11 "a table is shared
only if no other class in the file has a member with its name"; #17 "one list cell changes how loops are typed in
every formula of the file"; #5 "one large number anywhere changes the arithmetic of every formula".

## 4. Syntax that makes the decision local (none of these was compiled)

Upstream-style text without the declaration must still compile and be correct; only speed may be lost.

| # | Author writes | Without it | Compiler can delete | Author pays |
|---|---|---|---|---|
| 1-3 | `cells_param_size` (exists), per name once at model level | growable exact store (hash for numeric keys, dict otherwise), no IndexError from the sample | B:1005-1017 sharing by tuple, TR:121-131, window/spill in automatic.py and B: ~250 | 1 line per parameter name per model (the 57 models already declare `t`) |
| 4 | nothing; dense iff declared rectangle <= `max_dense_entries` (exists) | hash store | 75, and the receiver counter in T | a spec key for the few large rectangles |
| 5 | `"integer_checks": "unchecked"` (exists) | checked | T:943-966, 1086-1090, json report, resolve: ~80 | 1 line; 0-27% on integer-heavy formulas if not written |
| 6 | spec `"refs": {"rate": "float"}`, or nothing plus a checked setter | sampled type with a checked setter that raises | 0 lines; fixes D1 | 0-1 line per scalar Reference changed between runs |
| 7, 16, 19 | return annotation in the formula: `-> float`, `-> list[float]`, `-> Kind`; enum-typed parameter `kind: Kind` | sampled type (as upstream), but vector and enum lowering only when annotated | shape merging T:111-145, empty-sample rule, `_traced_enum` enums.py:602-624, `_widen_to_enum`: ~120; fixes D4 and the empty-sample raise | 1 annotation per list or enum cell; modelx keeps annotations in exported text (T:254 already reads them) |
| 8 | `"return_type": "memoryview"` (exists) | `object` | usage.py 590 | 1 spec line per array cell read by element; upstream users lose speed silently unless the key is written: propose as opt-out upstream, so keep for now |
| 9 | nothing; match the resolved callee, not the name | - | 0; removes the flip | - |
| 10 | nothing; lower `self.R()["f"]` at each reader under its existing guard (FR:2648-2750 does this for item classes) | - | `analyse` and `_mentions`: ~250 of RC | - |
| 11 | spec `"tables": {"Data.T": {...}}` naming the frame cell | per-item bind | uniqueness scan TC:177-191 and its consumers: ~60 | 1 line per table |
| 12-13 | the same `tables` entry with index kinds and column kinds: `{"index": ["int"], "unique": True, "columns": {"a": "float"}}`; checked at bind (RT `_decide` already compares) | read stays on pandas, uncast | schema tracing 50, recorder 520 (if every lowered form gets a run-time key test; int levels have `is_label`, object levels need one: unmeasured) | 3-6 lines per table; build could print the entry from the sample for pasting |
| 17 | nothing; type range loops in every formula or in none | - | the module scan V:225-286: ~60 | - |
| 21 | nothing; proofs only against declared sizes | checked call | model-wide `_f_` count EP:16-18, 98-102, 514-518: ~40 | - |

## 5. Not a table area. One remark for the table split: mechanisms 10-13 are what make a table read depend on
other cells and on the sample; a first commit with only "declared table + scalar read + bind-time schema check"
needs none of them.

## Final table, sorted by (lines removable) x (surprise)

| Rank | # | Mechanism | Removable | Surprise | Score | Recommendation |
|---|---|---|---|---|---|---|
| 1 | 12-13 | recorder + sampled schema | 570 | 3 | 1,710 | replace by a declared table schema; fix D2 now |
| 2 | 1-3 | sampled ranges, sharing, window, spill | 250 | 3 | 750 | replace: declared size or growable store |
| 3 | 10 | row cell decided by readers | 250 | 2 | 500 | delete the cross-reader analysis, lower per reader |
| 4 | 16, 19 | sampled list shape, sampled enum class | 120 | 3 | 360 | replace by annotations; fix D3, D4 now |
| 5 | 5 | integer mode | 80 | 3 | 240 | replace: checked unless declared |
| 6 | 4 | count rules | 75 | 3 | 225 | delete, nothing in their place |
| 7 | 11 | name uniqueness in the module | 60 | 3 | 180 | replace by naming the table in the spec |
| 8 | 17 | module switch for loops | 60 | 2 | 120 | delete the switch (always on) |
| 9 | 21 | proofs against sampled windows | 40 | 1 | 40 | keep proofs, base them on declared sizes only |
| 10 | 9 | keyword demotion by name | 0 | 2 | 0 | keep, match by callee |
| - | 6 | Reference types | 0 | 3 | - | keep, add checked setter (D1) |
| - | 8 | memoryview usage | (590) | 2 | - | keep (upstream behaviour); declare later |
| - | 7, 20, 22-26 | traced types, entries, locks, gate, collect, items, kinds | 0 | 0-1 | - | keep: each is guarded by an exact-type or boundary test and flips speed only |

## 6. Bottom line (estimates from the lines above; only the integer mode was timed)

(a) Keeping the measured speed: about 700 lines of non-local logic can go with nothing an author must write
(#4 75, #10 250, #17 60, #21 40, #2 25, D6 fold 20, sharing by tuple and uniqueness scans ~230 once sizes and
tables are named, which the 57 models' specs already do for `t`).
(b) Within 25-30%: about 1,500 lines (add #3 window/spill 200, #5 80 with checked as default, #12-13 570 behind a
declared schema); every remaining non-local rule is then either upstream's or guarded and speed-only.
Not measured: the cost of #3, #4, #21 and of a run-time key test replacing the recorder. Do not build these without
a timing on ten models (suggest: 3 `*_S` with tables, 3 with list cells, 3 `*_US_SEC`, the DV01 example).
