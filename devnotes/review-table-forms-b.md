# Review of the table forms (reviewer b)

Tree: `review/r4-brief` b326334. Stack: free-threaded CPython 3.14.6, Cython 3.3.1, modelx 636877b, pandas 3.1.1,
numpy 2.5.5, lifelib 04d69db (text only, no lifelib build). Line numbers are of b326334.
Evidence: five small models built one at a time in a scratch folder (exp1-exp4 below; Projection items
`policy_id`, a frame cell `Data.points()` locked), each compared with live modelx, and with `prepared_tables: False`
to attribute a difference to this area; pandas per-read costs timed on the same interpreter (10k-row frame; 200x120
two-level frame). Two fixes compiled and tested (D1, D3); everything marked "not compiled" was not.

## 1. Mechanisms
Sites: lean text of the 57 models (from the brief). Pandas cost: one read on the same stack, what each read costs
again if the form is dropped (the native read is tens of nanoseconds).

| Form | Sites | Code (`frame_reads.py` unless named) | ~Lines | Lost without it (pandas per read) |
|---|---|---|---|---|
| dispatch shared by the forms | - | `leave_Subscript` 2374-2475, `leave_Call` 2527-2559, `leave_Comparison` 2832-2876, `_lowered` 1504-1512, `_fallback` 2759-2773, `raised_veto` 2366-2372 | 230 | every form |
| L1 scalar read, literal column; L4 `R["c"][k]` | 358 | `_column_read` 2485-2502, `object_read` 2504-2525, `column_keyed` 2270-2293, float()/int() dropped 2553-2558 | 90 | `T.loc[k,"c"]` 11.5 us, `T.at` 8.5 us |
| L2 row field | 1,358 | `_row_locals` 749-813, `leave_SimpleStatementLine` 2178-2225, `row_local_read` 2227-2251, `_field_read` 2253-2268, `row_cell_field` 2648-2688, `_row_field` 2690-2750, `iterrows_loop` 2112-2142 | 340 | row build 13.5 us + 1.2 us per field; exp4: 1.6 ms vs 131 ms (200 items x 240 steps x 2 fields) |
| L2 by inlining a row cell | (same) | `row_cells.py` 1-410; `transformer.py` 752-777, 850-905 | 490 | nothing measurable: see cut C2 |
| column chosen by an expression | 28 | `named_read` 2295-2327, computed field in `row_local_read` 2240-2247 and `row_cell_field` 2668-2688 | 70 | as L1 |
| L3 sub-table (prefix, view, prefix cell, cell of parameters) | 40 | `prefix_receiver` 354-382, `param_prefix` 323-352, `view_assignment` 619-647, `prefix_of` 723-747, `column_assignment` 649-696, `view_statement` 1359-1402, `_view` 1404-1439, `view_read` 2144-2175, `_prefix_read` 2604-2632, `column_item` 2329-2364, `audit_subtables` 1133-1197 | 470 | `M.loc[p]` 45 us + `sub.loc[t,"c"]` 19 us; `M.loc[(p,t),"c"]` 204 us |
| L5 index extreme | 107 | `extreme` 859-868, `_extreme_part_int` 238-265, `extreme_part` 1711-1732, `level_extreme` 1634-1672, `level_walk_extreme` 1674-1709, extremes in `index_call` 1571-1589, 1610-1615, `_builtin` 1802-1824 | 210 | one level 1.3 us; `get_level_values(1).max()` 73 us |
| L7 membership | 15 | `leave_Comparison` 2832-2876, `prefix_membership` 2775-2830 | 100 | one level 0.36 us; two levels 4.1 us |
| L6 ordered lookup (asof) | 0 (39 before) | `_asof_locals` 480-503, `_keyed_use` 505-516, `asof_statement` 2584-2602, `_asof_read` 2634-2646, `_asof` 1762-1784, asof in `index_call` 1616-1631, `_tables.pyx` `asof_row` 681-703, `_last_label` 705-712, `asof_at` 848-860 | 190 | 111 us |
| L8 index walk, `searchsorted`, `X.index[i]`, len/list of an index or column | 83 | `_walk` 1879-1917, `comprehension` 2027-2070, `leave_For` 2084-2105, `_hoist_walk` 2561-2577, `_starred_target` 1059-1080, `_values`/`_builtin`/`_column_builtin` 1786-1839, `index_item` 1526-1543, searchsorted 1592-1609, `_vector_argument` 1745-1760 | 330 | a 120-row walk with a per-row `.loc` 1,387 us; `searchsorted` 13.7 us; `len` 0.17 us; `index[i]` 0.28 us |
| a two-statement zip spelling | "one model" | `zip_prefix` 1919-2025, held string selector 462-473 and 2740-2741, `transformer.py` 939-942, `_tables.pyx` 527-568, 764-800 | 210 | nothing found: no `zip(x.index, x["c"])` in any model formula of lifelib 04d69db (only in its tests) |
| per-item key parts, tuple keys, enum locals | - | `item_fixed`/`_fixed_value` 881-906, `tuple_key`/`guards` 908-934, `_enum_key_locals` 1001-1025 and 2185-2193 | 80 | a key lookup (about 0.1-0.4 us) per read |

## 2. Complexity: what can go
| # | Cut | Lines | Expected cost | Risk | Verify (<= 10 models) |
|---|---|---|---|---|---|
| C1 | the zip spelling and its runtime | ~210 (table row above) | none: no model formula of lifelib 04d69db has it (grep) | low; `test_prepared_zip_prefix.py` goes | grep the 70 exported models; t0 on the model meant, if any |
| C2 | row-cell inlining: `row_cells.Inline`, `analyse`, `_Uses`, `_LocalLoads`, `_Mentions`, `_independent_field_reads` (keep `_row_expression`, `item_constant`, `independent_prefix`), `transformer.py` 752-777 and the snapshot/rollback 850-905, `row_inline` branches in `frame_reads.py` (`row_names`, `sites`, `emitted_inline_assignments`) | ~380 | none measured: exp4 (`claim(t) = model_point()["sa"]*... + model_point()["age"]`) gives byte-identical `_mx_classes.py` with `typed_table_reads: False`; `row_cell_field` 2648 already reads `self.R()["f"]` and `v = self.R(); v["f"]` from the held row | medium: a reader shape only the inliner admits | diff generated C of the 5 models with most L2 sites, `typed_table_reads` on/off; t0 |
| C3 | L6 asof (compiler and runtime) | ~190 | 0 lean sites; the 39 SEC sites go to pandas (111 us each) unless respelled | low | t0 on the SEC models that had asof |
| C4 | `PREPARE_MIN_EVALUATIONS` (306-321, 1050-1052): a per-item frame prepared only if the sample evaluated the formula >= 8 times per item | -5 | unmeasured; replace with a static rule (prepare when the reader has parameters) | low | t0 on models with item-computed frames |
| C5 | `_extreme_part_int` 238-265 repeats the shape checks of `extreme_part` and `level_walk_extreme`; `_key_position` 1734-1743 duplicates `_key_part_position` 1016-1025 | ~35 | none | low | unit tests |
| C6 | one-level forms whose pandas read is < 1.5 us: `len(X.index)`, one-level `k in T.index`, one-level `T.index.max()`, `X.index[i]` | ~80 | unmeasured per model: a per-step site pays the pandas cost per step | medium | t0 on the 10 models with most such sites |

Duplication: `_field_read` 2253-2268 re-implements pandas' interleaved row dtype, and gets it wrong (D1); the fix
asks pandas instead. `index_call` 1545-1632 (88 lines) mixes L5, L6 and L8; splitting it by form is needed for §5.

## 3. Generic
- `tracer.py:685` maps every dtype other than float64/int64 to `obj`; `_field_read` 2259 reads `obj` as "the row is
  an object Series". True only when a column has object dtype; a float32, int32, bool, datetime or category column
  is also `obj` and pandas then builds a float64/int64/object row by its own rule. Fires wrongly on any other author's
  frame with float32/int32 columns (parquet, HDF, `astype`): D1.
- Decided by the sample's call frequency: C4. Safe (only toward pandas): `raised_veto`, `selects_rows` 2390-2396,
  `_vector_argument`.
- Fitted to one spelling: `zip_prefix` (two statements, levels `('obj','int')`, a str selector cell, an f64 column,
  a Data class publishing the frame); `_asof_locals` (top-level `lo = P.index.asof(x)`, every load `P.loc[lo,"c"]`);
  `level_walk_extreme` (`max(a for a, s in R.index)`, no `if`); `extreme_part` (`R.index.max()[k]`, k a literal);
  computed columns only under `float()` (`named_read`, `row_cell_field`); `_enum_key_locals` (`E.a if c else E.b`).
  Not recognised for another author: `T.loc[k].c`, `T.loc[k, c]` with `c` a variable on mixed columns outside
  `float()`, `T.xs(k)`, `T.loc[k][["a","b"]]` (all stay pandas, so correct, only slower).
- Unmodified upstream use changes by default: `prepared_tables` (`config.py:60`) and `typed_table_reads` (`:50`)
  default to true, so D1-D4 reach an upstream user with pandas reads in formulas; each traced frame is copied once
  per bind (columns copied in `_tables.pyx` 375-392), and per item for a frame the item computes.

## 4. Reliable
| # | Class | Minimal case (exp) | Expected (live) | Actual (compiled) | Fix |
|---|---|---|---|---|---|
| D1 | silent, blocking | frame `r`,`a` float64, `b` float32, `n` int32; `row = T.loc[policy_id]; return row["n"] * 100000` and `row["b"] * 3.0`; same with `data.points().loc[policy_id]["n"]` (exp1) | 5000000000.0; 0.30000000447034836 | 705032704.0 (int32 wraps); 0.30000001192092896 (float32 arithmetic) | compiled: `_tables.pxd` +1 (`row_object`), `_tables.pyx` +3: at prepare `row_object = not is_series and n > 0 and frame.iloc[0].dtype == object`; in `_decide` (593) a row signature with an `obj` column gets -1 when the row is not an object row. exp1 then equals live; exp4 (object row, str column) stays native (1.78 vs 1.63 ms); row-field tests 11 passed (§7) |
| D2 | silent | `round(T.loc[policy_id, "r"], 2)`, also `T.at`, `round(row["r"], 2)`; r = 7.215, 0.215, 5.565 (exp1) | 7.22, 0.22, 5.56 (numpy's round of a numpy scalar) | 7.21, 0.21, 5.57 (a C double rounds as a Python float) | with tables off the same formula equals live; at a cell boundary (`round(rate(), 2)`, rate returning the read) it differs with tables off too (exp2): pre-existing there. Not compiled: in `leave_Call`, `round(<typed read>, n)` takes the read's pandas value, as `leave_Subscript` 2377-2384 already does for a subscript; or box the read as `numpy.float64` for that consumer |
| D3 | silent | two-level frame; `v = T.loc[policy_id]` then `if policy_id < 3: return v.loc[1, "c"]` and `return 0.0`; policy 4 absent from the frame (exp3) | KeyError | 0.0: the view's assignment is dropped (`_drop_unused_locals` 566, `swaps` 1401) | compiled: in `view_statement` before 1400 append `if _mx_vs_<v> < 0: <the assignment as written>` (3 lines): pandas raises there, as live; native path unchanged. exp3 then raises KeyError; 22 tests pass (§7) |
| D4 | raise where live works | `T.loc[k,"r"] / T.loc[k,"z"]` with z = 0; `T.loc[k,"neg"] ** 0.5` with neg < 0 (exp2) | inf; nan | ZeroDivisionError; TypeError (complex) | allowed by the rules (it raises); new against tables off. Document, or keep a numpy scalar where the consumer is `/`, `**` (not compiled) |
| D5 | sample / frequency | C4: native or pandas for a per-item frame depends on how often the sample called the reader | same either way | same values, different speed | C4 |
| D6 | thread timing, not reproduced | per-item binding `self.<table> = ...; self.<state> = 1` (612-615) are plain stores with no release order; only matters if two threads compute one item | - | - | question for the table-base reviewer (`holds.py`); not tested |

## 5. The proposed split into four commits
The order base -> rows -> sub-tables (+ extremes, membership) -> tail follows the code's coupling, with changes:
- Walks are not a tail that can be declined: 83 lean sites, and a pandas walk with a per-row read cost 1.4 ms for 120
  rows; `searchsorted` (5 lean sites: `sub.index[sub.index.searchsorted(x, side="right")]` in CI_UK_S, IP_UK_S,
  ADE_FR_S) is labelled L8 and lives with them. Put walks in commit 3 (they need `_View`, `_view_parts`,
  `_walk_hold` from it) and make the tail L6 asof + the zip spelling (C1, C3), both with no lean site.
- `index_call` must first be cut into an extremes part, an asof part and a walk part.
- Rows carry `named_read` (both use the whole-table kind check -4) and `row_cell_field`; `row_cells.py` inlining
  should not be proposed (C2).
- Smallest first commit: L1 and L4 on a whole frame of a unique index: `_column_read`, `object_read`,
  `column_keyed`, `_key`, `_read_args`/`code_of`/`key_part` without `tuple_key`, enum locals and per-item code holds,
  `_lowered`/`_fallback`, `_mutated`/`_plain`, the L1 branch of `leave_Subscript`, the float()/int() drop, the
  report; about 600 lines of `frame_reads.py` with the base (not counted exactly, not built). It carries D2/D4 already.

## 6. Bottom line
(a) Keep the measured speed: base + L1/L4 + L2 via `row_cell_field` + named columns + sub-tables + walks + extremes
+ membership; drop C1 (~210), C2 (~380), C3 (~190, if the 39 SEC asof sites are respelled or accepted on pandas),
C4, C5 (~35): about 810 lines out of ~2,000 form lines plus `row_cells.py`. C1 and C2 have no measured cost (C2:
identical generated code on exp4); C3 costs nothing on the 57 lean models.
(b) Within 25-30%: also C6 (~80) and the one-spelling recognisers `extreme_part`, `level_walk_extreme`,
`_extreme_part_int`, `_enum_key_locals`, runtime tuple keys (~170): about 1,060 lines. Not measured on models (no
lifelib builds here); the per-read pandas costs in §1 bound the loss: each dropped form costs at most its pandas time
per evaluation. Verify by t0 on the ten models with most sites of the dropped forms.
Blocking now: D1 and D3 (fixes above, compiled), D2 for in-formula sites (fix not compiled).

## 7. What was run
- exp1 (D1, D2), exp2 (D2 at a boundary, D4), exp3 (D3), exp4 (C2, speed): each built with and without
  `prepared_tables`, values compared with live.
- With the D3 fix: `integration_tests/test_table_safety_net.py`, `test_table_fuzz.py` (default tier),
  `modelx_cython/tests/test_diag_run8_row_cells.py`, `test_read_audit.py`: 22 passed.
- With the D1 fix: `test_diag_run8_row_fields.py`, `test_diag_run8_row_fields_derived.py`, `test_table_safety_net.py`:
  11 passed.
- Neither fix is committed here (this branch carries only this report).

## 8. The two compiled fixes (against b326334)
D3, `frame_reads.py` in `view_statement`, before `view.setup = [*statements, statement]` (1400):
```python
        # an absent prefix (or a table that is not prepared) evaluates the assignment as written: pandas raises
        # its KeyError there, as live, even when no read of the view runs
        statements.append(cst.If(test=cst.parse_expression(f'{start} < 0'), body=cst.IndentedBlock(body=[statement])))
```
D1, `runtime/_tables.pxd` in `PreparedTable` after `ordered`, and `runtime/_tables.pyx` after
`table.objects = tuple(objects)` (399) and before `if result and columns is not None:` in `_decide` (593):
```python
    cdef readonly bint row_object          # pandas builds a row of this frame as an object Series
    table.row_object = (not table.is_series and n > 0 and frame.iloc[0].dtype == object)
        if result and columns is not None and not table.row_object and any(k == COLUMN_OBJECT for _, k in columns):
            result = False      # a row of float32/int32/bool... columns is not an object row: its dtype is pandas'
```
D1 costs one pandas row build per prepared table (not per read). An empty frame gets -1 for row reads with an
`obj` column; its row reads raise KeyError on pandas as before.

## 9. Corrections (after the review)
- Ordered lookups are not without lean sites: lifelib 04d69db `*_S` text holds 34 `.index.asof(` in 15 models and
  5 `.index.searchsorted(` in 3 (my count of 0 came from a grep with an unescaped parenthesis). In CI_UK_S
  `pivot_interp` (a per-step cell) reads a sub-table, `asof`, `searchsorted` and two `.loc` per evaluation: on pandas
  that is about 170 us per step (§1 costs). Cut C3 would cost far more than 3% on those models: not done.
- Row-cell inlining (C2): translate-only of KLV_DE_S, Dep_FR_S, VA_US_S with the shipped sample and spec, inlining
  on and off. Dep_FR_S and VA_US_S: identical generated code (only the report's inlining flags differ). KLV_DE_S:
  `expenses_pp(t)` and `commissions(t)` differ (`cost = self.cost_row()`, then `cost["..."]`: inlined, a row local
  with a branch per call; not inlined, the held row read directly), the same 62 native reads otherwise. Not
  identical, so the cut was not made.
