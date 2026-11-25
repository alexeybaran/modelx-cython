# Review r4, cross-cutting area: non-local logic (reviewer b)

Scope: every place where how one cell compiles depends on something outside its own text and declarations:
(a) other cells or Spaces, (b) values seen in the sample, (c) how often something was evaluated.
Stack: free-threaded CPython 3.14.6, Cython 3.3.1, modelx 636877b. Ten small models were built one at a time, each with
`python -m modelx_cython ... --sample --spec`; compiled results were compared with live. Timings are best of 5–15 runs.
Upstream checks were done with `cythonize` at default directives, on a hand-written `.pyx` that copies the generated
method. I could not run the upstream compiler itself: export protocol 2 is not supported there.

The code was mapped by reading. That gave about 45 mechanisms; the table in section 2 merges them into 30 rows.
"Up" means the mechanism already exists in upstream d22426f. "Built" means I made the flip and ran it. "Read" means the
flip was traced in the code only.

## 1. Blocking: silent differences from live, or regressions for upstream use (all built)

| # | Repro (small model) | Expected (live) | Actual (compiled) | Cause and fix |
|---|---|---|---|---|
| B1 | `base()=3_000_000`; `f(): k=base(); return (k*k*k)//(k*k)`; sample `f()` | 3000000 | **950361** (silent wrap) | `integer_boundaries._type_signed_locals` (508-562) annotates `k: longlong` even with `infer_types: False`. In the model-wide `unchecked` mode, `k*k*k` then wraps. Upstream types `k` as an object; Cython default inference gives 3000000 (checked). Adding an unrelated Reference `other_id = 2**31` makes the run raise `OverflowError` instead. Fix (not built): skip `_type_signed_locals` in unchecked mode, about 3 lines. |
| B2 | `last(n): k=-1; for i in range(n): k=i; return float(k+i)`; call `last(0)` | UnboundLocalError | **139681894444223.0** | Cython does not check whether a C loop variable was ever assigned. Two triggers: (i) the default `infer_types=True` (transformer 1105-1112) affects every model; (ii) with `infer_types: False`, an **unrelated** cell `vec` changed from `(1.0, 2.0)` to `[1.0, 2.0]` turns on `module_vector_cells` (vectors.py 225, 1810-1833), which declares `i: longlong`. Built: tuple → raises like live; list → garbage. Fix: type a loop target only if no load of it lies after the loop, about 10 lines. Not built. |
| B3 | `T()` returns a frame; `r(t)=T().loc[t,'c']`; another cell `bump()` does `T().loc[0,'c']=99.0`; run `r(1); bump(); r(0)` | 99.0 | **0.0** | `frame_reads._mutated` (523-558) scans only the reading formula. `r` is lowered (L1) to a copy prepared when the frame was bound, so the edit is not seen. If the same edit is moved into `r`'s own formula, `r` stays on pandas. Fix: run `_mutated` over every formula of the module that calls `T`, and leave `T`'s readers on pandas if any formula edits `T` (about 10 lines). Not built. |
| B4 | IntEnum `Phase`; `status(t)=Phase.ACC if t<3 else 1`; `flag(t)=1.0 if status(t) is Phase.ACC else 0.0` | [1,1,1,0,0,0] | **[0,0,0,0,0,0]** | The traced returns mix an enum member and an int, so `status` returns C `long long` and the `is` test is always False. Unrelated change in the other branch, `1` → `Phase.WDR`: the result becomes correct. Pre-existing (upstream also types the mixed return as long long). This is my reading; I did not run it on upstream. Fix: a cell whose traced returns include an enum member, but not only members, returns `object`; about 4 lines in tracer/normalize. |
| B5 | `f(t, r)=t*r` with `r` a float; sample `f(10, .5)`; call `f(50, .5)` | 25.0 | **IndexError** | `transformer._guard_associative_cache` (121-129) bounds dict and hash caches to the sampled range. Upstream had no bound on a dict cache. It raises even with `automatic_optimizations: False`. Built: replacing the body with `return body` gives 25.0. Delete about 10 lines plus the report fields (admission 70, 300). Tests that name the range: test_builder, test_formula_entry_domains, test_param_bounds, test_vector_extrema_typing. |

Integer mode, measured on an integer-heavy model (200 policies × 600 steps of `//` and `%` index arithmetic), with
the spec forcing each mode: `unchecked` 1.93 ms, `checked` 4.72 ms (2.4x). The model-wide `auto` choice therefore
moves every cell's speed by up to 2.4x. One integer ≥ 2**31 anywhere in the trace flips it: a Reference, an
ItemSpace key, or an int column of any input table. This rules out "always checked" as a free fix for B1. Fix the
locals instead (B1).

## 2. Mechanisms (file:line, decision, own lines, kind, upstream, fallback when it does not fire)

| id | Mechanism | Where | Decides | LOC | Kind | Up | When off |
|---|---|---|---|---|---|---|---|
| M1 | Cache stop pooled per (Space, parameter tuple) | builder 987-1017, 1073-1138 | dense size and accepted domain | 75 | a,b | yes | IndexError |
| M2 | Associative bound | transformer 121-129; builder 728-737 | IndexError on dict/hash cache | 25 | a,b | no | (B5) |
| M3 | Negative arguments, shifted cache | builder 653-679, 697-716 | lower bound, dense vs hash | 40 | b | no | IndexError / hash |
| M4 | Compact window and spill | automatic 19-46; builder 739-780 | physical array from own min/max | 60 | b | no | full array |
| M5 | Dense by use (R-3b) | automatic 76-126 | dense vs hash above 65,536 entries | 40 | c | no | hash store |
| M6 | Numeric-cache capacity | automatic 120-126; numeric_cache 30-84 | primary capacity, reserve/publish | 60 | c,a | no | default 8 |
| M7 | Integer mode `auto` | config 99-118; cli 310-321; tracer 964-990 | overflowcheck in every module | 75 | b | no | (B1) |
| M8 | Typed signed locals | integer_boundaries 508-562 | `k: longlong` from callee kinds | 55 | a | no | object |
| M9 | `called_with_kwargs` by bare name | parser 292-304; cli 368-377; read at ~25 sites | cpdef vs Python method, every lowering of callers | 40 | a | yes | Python method |
| M10 | Usage verdicts | usage 246-532, 671-705; builder 479-515 | memoryview vs object return | 450 | a | yes | object |
| M11 | Operand kinds from callees and Refs | powers 114-245; reductions 937-1018; ordered_arguments 73-132; expression_kinds 183-264 | `**`, min/max, sums native | 300 | a,b | part | Python ops |
| M12 | Enum admission | enums 403-459, 608-624; table_cells 132-157 | `is`→`==`, enum constants, wraps at callers | 90 | a,b | no | `is` as written (B4) |
| M13 | Module vector switch | vectors 225-286, 553, 783-797, 1796-1833 | loop-variable typing in every formula; strict locals | 190 | a | no | object loops (B2) |
| M14 | Vector shape of callees | vectors 539-572, 1693-1795; tracer 111-145 | F64Vec locals, `.tolist()` on Python use | 150 | a,b | no | list |
| M15 | VectorWriteError | vectors 1055-1170 | **build stops** | 30 | a,b | no | — |
| M16 | VecCache window and Space budget | vectors 97-215 | vector cache vs dict, largest first in a 64 MB budget | 120 | a,b | no | dict |
| M17 | Table schema of the callee | tracer 663-713; used by frame_reads | every table-read form | 50 (+core) | a,b | no | pandas, guarded at run time |
| M18 | Read recorder vetoes | read_notes 476; frame_reads 2366-2395, 1745-1760 | list key / only-raised / array argument → pandas | 520 | b | no | native read, TypeError on a list key |
| M19 | Notes keyed by class short name | read_notes 227; frame_reads 132-139 | vetoes pooled across Spaces; missed after inlining | in M18 | a | no | — |
| M20 | Evaluation gate (8 per item) | frame_reads 306-321, 1052 | per-item preparation of an own frame | 12 | c | no | pandas |
| M21 | Row-cell inlining, module-wide name scan | row_cells 312-364; transformer 752-777, 850-893 | row fields read natively | 410 | a | no | Record path |
| M22 | Prefix and parameterised prefix cells, module-unique name | transformer 779-842; frame_reads 323-383 | view, prefix and asof reads | 120 | a,b | no | pandas |
| M23 | Table-cell renames need a module-unique name | table_cells 177-191; frame_reads 820-846 | TableCell, published frame, zip_prefix | 120 | a | no | plain cell |
| M24 | `collect_shared_item` reachability | numeric_cache 39-56 | dense allowed, holds, per-item binding, capacity | 30 | a | no | shared path |
| M25 | Dense proofs by `_f_name` use count | entry_pipeline 54-112 | unchecked dense reads | 60 | c,a | no | checked reads |
| M26 | Private entries need callers; D139 domain | inline_entries 43-57, 236-257 | entry exists; proof domain from sample | 40 | a,b | no | checked call |
| M27 | Forwarded provider results (fixed point) | usage 628-668 | provider view type | 40 | a,b | no | object |
| M28 | Space results, item locals, held arrays | space_results 26-197; item_locals 41-200; held_arrays 207-310 | typed receivers, native result calls | 500 | a,b | no | Python call, guarded |
| M29 | Module source text | transformer 655-672 (`'_mx_canonical_round' not in self._source`), 1069 (`type` binding) | native round, every exact-receiver entry | 15 | a | no | Python |
| M30 | Return and argument types from the sample | tracer 239-292, 426-511 | C types, dict vs dense | — | b | yes (+new rules) | object |

## 3. Stability: the smallest unrelated change that flips a cell (built unless marked "read"), most surprising first

| Rank | Mech. | Change to something else | Effect on the cell | Built |
|---|---|---|---|---|
| 1 | M13/B2 | another cell returns `[1.0, 2.0]` instead of `(1.0, 2.0)` | `last(0)`: UnboundLocalError → **garbage number** | yes |
| 2 | M12/B4 | other branch of the callee returns `1` instead of `Phase.WDR` | `flag` **wrong for t<3** | yes |
| 3 | M7/B1 | unrelated Reference `other_id = 7` → `2**31` | `f()` 950361 (wrong) → OverflowError; all cells 2.4x slower (int model) | yes |
| 4 | B3 | the in-place edit moves from `r` into a sibling cell `bump` | `r(0)` stale: 0.0 vs live 99.0 | yes |
| 5 | M21 | a formula in **another Space** renames a local `q` to `pol` | row cell refused: "used in another class"; 50-policy run 21.3 → 32.6 ms (1.53x) | yes |
| 6 | M9 | a formula writes `pd.Series(...).mean(skipna=True)` | cell `mean` becomes a Python method; `acc` loop 95 → 177 ms (1.85x) | yes |
| 7 | M15 | callee `vec` sampled as a list (not a tuple); `k` writes `x[0]=5.0`, `k` never sampled | **build stops** with VectorWriteError | yes |
| 8 | M1 | unrelated `g(t)` in the same Space sampled to 1000 | `a(500)`: IndexError → 501.0 | yes |
| 9 | M2/B5 | none needed: sample `t ≤ 10` | `f(50, .5)` raises; upstream returns 25.0 | yes |
| 10 | M5 | the sample calls `report()` (400 entries per policy) instead of `total()` (40,200) | `used` dense → hash store with a warning | yes (translate) |
| 11 | M29 | the text `_mx_canonical_round` appears anywhere in the module, even in a comment | native round off for every cell | read |
| 12 | M19 | a list key seen in Space `A._c_Assumptions` | same-text read in `B._c_Assumptions` stays pandas | read |
| 13 | M23/M22 | a Reference or cell named `T` in another Space of the module | `T` loses TableCell; readers lose published binding and views | read |
| 14 | M25 | a cell with the same name in another module | both lose dense proofs | read |
| 15 | M20 | projection length 7 → 8 in the sample | own-frame reads become native | read |
| 16 | M24 | an unrelated collect Space gets a Reference to this ItemSpace | dense → numeric cache; holds and per-item binding off | read |

## 4. The four questions, per group

**Sample ranges (M1–M5, M16, M26).**
- Need: dense arrays are the main speed lever. M4/M5 only save memory or pick between hash and dense above 65,536
  entries. I estimate M5 matters only to models with large rectangles; I did not measure it on lifelib.
- Complexity: 5 interacting rules: pooled stop, own lower, compact window, by-use ceiling, Space budget. Not
  explainable in one sentence.
- Stability: speed flips (#10), and raises flip between "raise" and "works" (#8, #9).
- Syntax: a declared range already exists: `cells_param_size` with `(lower, stop)` pairs and `dense_window`. Make the
  rule "declared range → dense with exactly that range, else hash/dict with no bound". That would delete M2 (25 lines),
  M4 (60), M5 (40) and the auto part of M3 (about 30). Cost to the author: one spec line per parameter name per
  model (`{'t': (0, 1201)}`), usually 2–4 lines. Upstream text without it still compiles; it is slower (dict). M1
  stays for upstream compatibility.
- Recommendation: delete M2 now, because it is a regression. Put the rest in the proposal text.

**Integer mode and typed locals (M7, M8).**
- Need: measured 2.4x on integer-heavy code. For float models I estimate 1.0–1.1x (not measured).
- Complexity: 1 threshold, but it reads every traced integer of the model.
- Stability: #3 (one input value anywhere flips every module).
- Syntax: the spec `integer_checks` already exists. Default to `unchecked` as upstream did, and drop `auto`: deletes
  config 99-118 and the tracer's magnitude collection (about 45 lines). The B1 fix is still required.

**Name-wide and module-wide rules (M9, M13, M21–M23, M25, M29).**
- Need: each guards a real hazard (keyword calls, overrides, shadowing). The match is by spelling, not by binding.
- Complexity: low per rule, but there are about 7 different scopes: model by name, module, class.
- Stability: #1, #5, #6, #11, #13, #14. These are the most surprising, because the trigger is a spelling elsewhere.
- Syntax: none needed; this is precision work on the existing rules:
  - M9: only calls whose receiver resolves to a Space count. Rewrite keyword calls to those Spaces to positional
    (signatures are known). About 30 lines added, about 25 flag checks deleted.
  - M21: `_mentions` should count `self.R`/`x.R` attribute uses, not bare Names (2 lines).
  - M22/M23/M25: key by class hierarchy or fqname instead of short name (about 10 lines each).
  - M29: test the binding, not `self._source` (1 line).
  - M13: make vector typing per formula (only formulas that touch a vector). Delete the module switch and
    `strict_locals` (about 60 lines), and fix B2.

**Vector shape of callees (M14–M16).**
- Need: the brief measured 21–32x on models with list cells.
- Complexity: shape from the sample, plus ownership, plus a budget.
- Stability: #7 stops the build because of another cell's text and the sample.
- Syntax: the author writes a tuple for an immutable vector result, or a return annotation `-> list[float]` that a
  formula may carry. Only the build stop needs fixing: instead of VectorWriteError, demote the callee (keep it a
  list) when any formula writes into its result. That is about 20 lines changed, and upstream text then always
  compiles.

**Table schema and read recorder (M17–M20).**
- Need: the brief measured 15–375x for the forms. The schema is guarded at run time (signature slot), so M17 is
  stable in value.
- M18 costs about 1.5x translation time (brief). Its vetoes protect against list keys (otherwise a TypeError) and
  array arguments.
- Syntax or local alternative: an `is_label` check at run time for every object-level key and for asof arguments,
  falling back to pandas. That makes the vetoes unnecessary and deletes read_notes (476) and the veto code (about 40),
  at the cost of one type test per read. Estimated under 2%, not measured.
- M20: delete. The gate costs 12 lines and flips on projection length; prepare whenever the formula has arguments.

**Kinds, enums, verdicts, providers (M10–M12, M27, M28, M30).**
- Need: M10 and M30 are upstream. M11 and M28 drive the native arithmetic, reductions and result calls (brief:
  reductions median 1.44x). They are guarded at run time where receivers are involved.
- Syntax: a return annotation on the callee (`-> float`, `-> Phase`, `-> int`) would make each caller's decision
  local and would fix B4. Upstream text compiles without it.
- Keep these. Add the B4 rule (mixed enum and int returns → object).

## 5. Ranking: lines removable × instability (instability 3 = silent or build stop, 2 = raise flip, 1 = speed only)

| Mechanism | Removable lines | Instability | Score | Recommendation |
|---|---|---|---|---|
| M18/M19 read recorder vetoes | 516 | 2 | 1032 | replace by a run-time label guard (not measured) |
| M13 module vector switch | 60 (+fix B2) | 3 | 180 | **delete**, type per formula |
| M7 integer `auto` | 45 | 3 | 135 | delete `auto`; default unchecked; **fix B1** |
| M4+M5 compact window, dense by use | 100 | 1 | 100 | replace by a declared range or window |
| M15 VectorWriteError | 30 → demote | 3 | 90 | replace by demotion |
| M3 auto shift | 30 | 2 | 60 | replace by declared `(lower, stop)` |
| M2 associative bound | 25 | 2 | 50 | **delete now** (B5, built) |
| M9 kwargs by name | 25 (net ~0) | 1 | 25 | narrow to Space receivers |
| M20 evaluation gate | 12 | 1 | 12 | delete |
| M21–M23, M25, M29 name scans | ~0 (precision fixes) | 1 | — | keep, fix the scope |
| M6 capacity, M26 entry domain, M27, M28 | 0 | 1 | 0 | keep (speed only, runtime-guarded) |
| M12 enum, M30 types | 0 (+4 fix) | 3 | — | keep, **fix B4** |
| M1, M10 (upstream) | 0 | 2 | — | keep |

Delete with nothing in their place: M2, M20, the `auto` integer mode, the M13 module switch, and the `_source`
text test in M29. Replace by syntax or a declaration: M3–M5 (declared range or window), vector immutability (tuple
or demotion), callee types (return annotation, optional). Keep: M1, M6, M10–M12, M17, M21–M28, with the scope fixes
above.

## 6. Bottom line

- (a) Keeping the measured speed: about 170 lines can go with no expected loss: M2 25, M20 12, the M13 switch 60,
  `auto` integers 45, the M29 text test, and the B5 deletion is built. B1–B4 need about 30 lines of fixes.
  The lifelib impact is estimated, not measured.
- (b) Within 25–30%: additionally M18 to a run-time guard (516) and M3–M5 to declared ranges (130). That is about
  800 lines in all. Translation would be about 1.5x faster (brief's figure for the recorder). The run-time cost of
  the guard is not measured.
