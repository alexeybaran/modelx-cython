# Review r4, area `table-forms`, reviewer `a`

Read in full: `frame_reads.py` (2,876), `row_cells.py` (410), `runtime/_tables.pyx/.pxd` (1,264), the wiring in
`transformer.py:748-954`, `tracer.py:663-712`. Stack as the brief (3.14t, Cython 3.3.1, modelx 636877b, pandas 3.1.1).
Run: three small probe models (23 + 25 + 19 cells, one build at a time), each cell compared with live per item and step;
the same model with `prepared_tables: false` (every difference below vanishes there, so each is caused by this area or
its base); a label fuzz of `PreparedTable` against pandas over 18 index dtypes x 38 key kinds; the fixes of section 4 on
a scratch copy of the compiler, with 76 unit tests and nine table integration files. **Not measured: any speed, and
nothing on a lifelib model** (not allowed here). Every line count is `ast` function length; "forms" below = 1,901 lines
of `frame_reads.py` + `row_cells.py` 410 + 133 of `transformer.py` = 2,444 compiler lines, plus about 300 runtime lines.

## 1. What the code does
| Mechanism (report form) | Where | Lines | Lost without it |
|---|---|---|---|
| Scalar read, literal column, complete key; `R["c"][k]`, `R.c[k]`; object column; Series `S[k]` (L1, L4) | `frame_reads.py:2374-2475`, `_column_read` 2485, `object_read` 2504, `column_keyed` 2270, `leave_Call` 2527 | 213 | 358 sites back to `.loc` (20-100k instructions each, `:1050`) |
| Column chosen by an expression (L1) | `named_read` 2295 | 33 | 28 sites; fires only if all columns are one kind, or the read sits under `float()` |
| Row cell inlined into field reads (L2) | `row_cells.py` all; `transformer.py:752-777, 850-893`; `_row_field` 2690, `_field_read` 2253 | 557 | 1,358 sites: a pandas row per item and reader |
| Row local, `iterrows`, field of a kept row cell (L2) | `_row_locals` 749, `leave_SimpleStatementLine` 2178, `row_local_read` 2227, `iterrows_loop` 2112, `row_cell_field` 2648 | 210 | `row = T.loc[k]` then `row["c"]` builds the row Series |
| Sub-table: prefix cell `P.loc[k,"c"]`, view local, `P(a).loc[...]` (L3) | `prefix_receiver` 354, `param_prefix` 323, `view_assignment` 619, `prefix_of` 723, `view_statement` 1359, `_view` 1404, `view_read` 2144, `_prefix_read` 2604, `audit_subtables` 1133, `transformer.py:779-842` | 398 | 40 sites: a sub-frame built per item, then pandas reads on it |
| Index extreme (L5) | `extreme` 859, `_extreme_part_int` 238, `level_extreme` 1634, `level_walk_extreme` 1674, `extreme_part` 1711, parts of `index_call`/`_builtin` | 135+ | 107 sites; also the C typing of keys built from them (`:392-394`) |
| Ordered lookup: `asof`, `searchsorted` (L6) | `_asof_locals` 480, `asof_statement` 2584, `_asof_read` 2634, `_asof` 1762, `index_call:1592-1631`, runtime `asof_row`, `asof_at`, `search` | 165 + 73 runtime | 0 sites per the brief; 39 in the pinned lifelib text (section 5) |
| Membership (L7) | `leave_Comparison` 2832; prefix form `prefix_membership` 2775 | 45 + 56 | 15 sites |
| Index walk, column walk, `index[i]`, `len/list/sorted/min/max(X.index)` (L8) | `_walk` 1879, `comprehension` 2027, `leave_For` 2084, `index_item` 1526, `index_call` 1545, `_builtin` 1795, `column_assignment` 649, `column_item` 2329, `_hoist_walk` 2561 | 492 | 83 sites |
| Two-statement zip spelling | `zip_prefix` 1919; `:413-417, 466-473, 2742-2743`; runtime `_zip_*`, `zip_prepared_prefix` | 127 + 95 runtime | one model, one cell |
| Key kinds: run-time tuple key, enum-valued local | `tuple_key` 908, `guards` 931, `_enum_key_locals` 1001 | 66 | those keys take the object path or pandas |

## 2. Complexity
| Cut | Lines saved | Expected cost | Risk | Verify (at most ten models) |
|---|---|---|---|---|
| C1 zip spelling and `held_string_selector` plumbing; its test file (241) | 127 compiler, 95 runtime | one cell of one model on pandas, or +15 lines teaching `_walk` (`:1879`) `zip(V.index, V["c"])` (not compiled) | low: guarded by 14 identity checks of pandas internals (`_tables.pyx:546-557, 778-791`) that a pandas release can silently turn off | that model: run time, `_mx_table_reads.json` |
| C2 ordered lookups (if the lean text really has none) | 165 compiler, 73 runtime | none on lean text; on the pinned text 15 models regain a pandas `asof` per step | medium: confirm which text ships | count form `L6` in `_mx_table_reads.json` |
| C3 one recogniser, two copies: `_key_position` 1734 = `_key_part_position` 1016; `_extreme_part_int` 238 repeats `level_walk_extreme` 1680-1700 and `extreme_part` 1717-1725; `int_label` 698 repeats the dispatch of `index_call`; `cached_call` 875 = `transformer._cached_call` 844; `_param_prefix_cells` (`transformer.py:826-833`) re-derives `row_cells._row_expression` | about 70 | none | low | generated `.py` byte-identical |
| C4 the shape `self.<cell>()` tested inline in six places (`:328, 359, 1969, 2657`, `row_cells.py:48, 138`); the "several rows" node tuple written six ways (`:47, 350, 641, 740, 2310, 2355, 2802`; only `:47` has `Set`, `Lambda`) | about 25 | none | low; the unequal tuples are a latent defect | as C3 |
| C5 dead parameters: `_record(view=)` (19 callers pass it, body ignores it, `:1243`); `_guarded(original)` 1501; `_prefix_holds(name)` 2579; `row_field(name)` 956 and `hold_row(assigned)` (`_tables.pyx:714`, always `False` at `:601`); three unused imports of `item_constant` (`:371, 2678, 2717`) | about 20 | none | none | as C3 |
| C6 "first or last row of an ordered view" three times (`:1578-1589, 1610-1615, 1812-1820`) | about 15 | none | low | as C3 |
| C7 Cython workarounds that are not table code and fire only if the model has a traced frame (`:387`): starred loop target (`_starred_target` 1059), `X[True]` (`:2396-2400`) | 30 moved out | none | low | their tests |
| C8 every lowered read keeps its pandas text; a lowered loop keeps a whole second copy of its body (`_branch` 2108), nested walks double again | 0 lines | generated C: probe model 2.21 MB without tables, 3.38 MB with (+53%); `.so` +59% | redesign: one out-of-line pandas function per formula | C size and build time |
Duplicates of pandas are inherent (the keyed lookup). Nothing here duplicates upstream. Tests of the area: 3,803 lines.

## 3. Generic
| Fitted to | Case | Effect on another author |
|---|---|---|
| Receiver = zero-argument cached Cell returning the frame (`_schema_of_call` 294) | `data.tbl.loc[k, "sa"]`, `tbl` a Reference: the upstream lifelib convention | see probe 3 below: not lowered |
| Spelling of the column | `T["sa"].loc[k]`, `T["sa"].at[k]`, `T.loc[k].sa`, `T.iloc[-1]["sa"]`, `len(T)`, `T["sa"].get(k)` | probe 3 below |
| `float()` around a computed field (`:807-810, 2302-2305, 2670`) | `float(T.loc[k, c])` lowered, `T.loc[k, c] * 1.0` not (mixed kinds) | speed depends on spelling |
| One model's text | `zip_prefix`; `item = E.a if c else E.b` (`:1001`); `max(a for a, s in T.index)` (`:1674`); `tbl = data.T()` in every branch (`:279-282`); `T.index.max()[k]` (`:1711`) | each written for a spelling met in one or a few models; about 250 lines |
| Index of integers or text | dates, periods, intervals, categories, bools | wrong today (R1, R2); pandas after the fix |
| Columns float64 or int64 | int32, float32, bool, nullable | object path; wrong rows today (R3) |
| Constants `PREPARE_MIN_EVALUATIONS = 8` (`:1052`), `MAX_PARTS = 6` (`:45`) against `_padded` 5 (`:1321`) and `_MAX_LEVELS = 8` | a prefix of six parts on a deeper index emits a call with the wrong argument count: the build stops | rare |
Behaviour change for an unmodified upstream user (all with default settings, none with `prepared_tables: false`): R1-R7.

## 4. Reliable
Probe = one cell of an item Space; frames in cells of a locked `Data` Space; `T1` int index, columns `sa`,`z` float64, `n`,`zi` int64.
| # | Kind | Minimal reproduction | Live | Compiled | Fix (compiled on the scratch copy unless said) |
|---|---|---|---|---|---|
| R1 | **silent** | `1.0 if "2020-01-01" in TD().index else 0.0`, `TD` on a DatetimeIndex; `1 in TB().index`, `TB` on `[True, False]` | 1.0; False | 0.0; True | labels are looked up by dict equality (`_tables.pyx:99, 253-260`); pandas matches dates and periods by text, intervals by containment, bools by dtype. Prepare only all-int, all-float or all-`str` levels (+4 at `:253`), no bool key on a dict level (+2 at `:99`). Fuzz after the fix: no difference on int, float, text; date-like indexes stay pandas |
| R1b | raises | `TD().loc["2020-02-01", "v"]`; `TI().loc[2.5, "v"]` on an IntervalIndex | 2.0; 2.0 | KeyError | same fix |
| R2 | raises | `TC().loc["a", "v"]`, `TC` on a CategoricalIndex, or an index `["a", 1, 2.5]` | 1.0 | TypeError from `index.max()` (`_tables.pyx:401`) | `prepare` catches only `ValueError` (`:237`): catch `TypeError` too (1 line) |
| R3 | **silent** | `str(TM().loc[policy_id]["n"])`, `TM` = float64 `a`, int64 `n`, int32 `m`; `TU().loc[k]["u1"] + TU().loc[k]["u2"]`, two uint8 columns 200 and 100 beside an int64 one | `'7.0'`; 300 | `'7'`; 44 | `_field_read` takes any non-float64/int64 column as an object column (`:2259`), and the pandas branch is cast to the same C type (`:1507`), so no run-time check can help. Record how pandas types a row in the schema (`tracer.py:681`, +8), use it at `:2259`, decline at `:796` and `:2708`, and check it per table (`_tables.pyx:390, 594`): 20 lines |
| R4 | **silent** | `df = data.T1c(); a = df.loc[k, "sa"]; w = df; w.loc[k, "sa"] = 5.0 + k; return df.loc[k, "sa"]` | 6.0 | 100.0 | `_mutated` sees an edit only through the receiver's own name (`:529-532`): follow locals bound once to it (+9). Not covered by any fix: an edit in another formula, in a callee, or by the user after the first read (base; only `verify_inputs` reports it) |
| R5 | **silent** (live raises) | `v = data.T2().loc["zz" if policy_id == 2 else "a"]; return v.loc[t % 4, "rate"] if policy_id != 2 else 0.0` | KeyError | 0.0 | the view's assignment is dropped once every use is lowered (`:452-457, 566-586`): keep it as `if _mx_vs_v < 0: v = ...` (+8) |
| R6 | raises | `if policy_id != 3: return data.T2().loc[(gcode(), t % 4), "rate"]` / `return 0.0`, `gcode()` raising for item 3; the same with an optional table cell `data.opt()` that raises when unused | 0.0 | ValueError | holds are evaluated before the top-level statement of the read, taken or not (`:446-447, 588-616`). Compiled fix: each hold in `try`, left unresolved on error so the read's pandas branch raises in place (+8, runtime +3). Still early: the receiver and key cells are computed where live may never call them (time; unbounded recursion if they depend on the reading cell). Proposal text: bind inside the read's test |
| R7a | raises | `data.T1().loc[k, "sa"].round(-2)`, `.item()` | 100.0 | AttributeError | the read is a C number: an attribute of it must be the pandas value, as a subscript of it already is (`:2377-2384`): `leave_Attribute` (+8). Through a local (`x = read; x.round(-2)`) it still raises: not fixed |
| R7b | raises | `1.0 / T1().loc[k, "z"]`; `n // zi`, `n % zi`; `z ** 0.5`, z = -4 | inf; 0; 0; nan | ZeroDivisionError; TypeError | not fixed: numpy scalar rules against C rules. The run raises, which the rules allow, but it is new against upstream. Document; or leave a read boxed when it is a divisor or a power base (not compiled) |
| R7c | **silent** | `isinstance(T1().loc[k, "n"], int)`; `T1().loc[k, "n"] ** -1` | False; ValueError | True; 0.142857 | same cause; not fixed |
| R8 | rule: depends on the sample | a read is left on pandas when every sample evaluation raised (`raised_veto` 2366), when the sample saw a mask or list (`:2388-2395`, `_vector_argument` 1754), when the reading formula ran fewer than 8 times per item (`:317-320`), or when the frame's cell was never evaluated (no schema) | - | - | values stay equal (after R1), but the generated code depends on what the sample reached and how often. Replace the vetoes by run-time guards (`is_label` exists for int levels, `:1341-1349`); see probe 3 for a text level |
| R9 | not tested | `S[0]` on a text index under pandas 2 (positional, deprecated) | value | KeyError expected | stack has pandas 3 only |
| R10 | raises | an `int` key of 2**63 or more on an int level | KeyError / False | OverflowError (`_tables.pyx:92-93`) | not fixed; 2 lines |
Checks of the fixes: probes 1 and 2 go from 15 + 10 differing cells to 4 + 3 (R7b, R7c, the local of R7a); 76 unit tests pass
before and after (the first version of the R1 fix broke 3 by refusing a float index; corrected).
INTEGRATION_PLACEHOLDER

PROBE3_PLACEHOLDER

## 5. The split into four commits
- Order is right in outline, with three changes. (1) Complete-key membership (`leave_Comparison` 2832, 45 lines) and the
  one-level extreme (`extreme` 859 and `:2541-2550`, 20 lines) use only base helpers, and the extreme types keys of scalar
  reads (`:392-394`): they belong in commit 1 or 2, not with sub-tables. (2) `_View` (`:1096`) is the shared structure of
  sub-tables, prefix membership, prefix extremes, ordered lookups and walks: the sub-table commit must introduce it whole.
  (3) The tail holds two unlike things: walks (83 sites, wanted) and ordered lookups (0 sites in the brief). Make them two
  commits, so that declining one does not decline the other.
- The count "0 ordered lookups" does not hold for lifelib 04d69db as cloned: its `*_S` folders contain 34 `.index.asof(`
  and 5 `.index.searchsorted(` (15 and 3 models; `grep` over the 70 folders), the "39 before" of the brief. If the lean text is not
  what ships, the tail is not declinable for those models.
- `leave_Subscript` (`:2374-2475`) dispatches every form in one function in a fixed order (rows, extreme part, walks,
  views, prefix cells, frames): each commit has to edit it, so the commits cannot be reviewed independently.
- Smallest first commit: receiver, binding per call and per item, slot by signature, L1 with a literal column and a
  complete key on int or text levels, the pandas branch, the report. About 450 lines of `frame_reads.py`, 450 of the
  runtime, `key_facts.py`, `holds.py`, the schema in `tracer.py`. It needs neither `row_cells.py`, `_View`, held codes
  (`item_fixed` 881), `tuple_key`, nor the recorder: with run-time guards in place of the sample vetoes (R8) the recorder
  is audit only, and the 1.5x translation cost leaves commit 1. Fixes R1, R2, R6 and R7a belong in it.

## 6. Bottom line (compiler lines of the area, now 2,444; runtime 300)
- (a) Same measured speed: **about 2,060 lines** (runtime about 130): C1 to C6, i.e. minus zip 127, ordered lookups 165,
  duplicates and dead parameters 90; conditional on the lean text having no ordered lookup, and on the zip cell being
  served by `_walk`. No speed was measured here.
- (b) Within 25-30%: **about 1,100 lines** (runtime about 60): L1 literal 213, inlined row cells 557, prefix cell read
  (`prefix_receiver`, `_prefix_read`, `_prefix_cells`) 90, one-level extreme 20, complete membership 45, the plain index
  walk (`leave_For`, `comprehension`, index branch of `_walk`) 130; without row locals, views, named columns, tuple and
  enum keys, part and level extremes, column walks. **An estimate from site counts, not a measurement**: one pandas read
  costs more than a whole native step, so a single per-step site of a dropped form breaks the 30% for its model. Decide
  it from evaluations, not sites: add the recogniser's name to each `_record` entry (`:1243`, 1 line) and sum the sample
  counts of `_mx_read_notes.json` per recogniser over ten models; translation alone gives this, no C build.
