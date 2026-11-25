# Review: table-base (reviewer a)

Tree b326334. Stack: free-threaded CPython 3.14.6, Cython 3.3.1, modelx 636877b, pandas 3.1.1, numpy 2.5.5.
Evidence: two models of my own built with `mx2cy` (12 and 9 cells; compiled against live), the runtime
`_tables` built alone (probe of 60 keys on 20 index kinds), `tests/test_prepared_tables.py` (24 pass on the
tree). No lifelib model was built: every speed statement about the 70 models is the brief's, not mine.
Short names: FR `frame_reads.py`, KF `key_facts.py`, RN `read_notes.py`, TC `table_cells.py`,
PYX/PXD `runtime/_tables.pyx/.pxd`. "Tested" = compiled and run here; "not compiled" = read only.

## 1. Mechanisms
| # | Mechanism | Where | Lines | Lost without it |
|---|---|---|---|---|
| 1 | Schema of a traced frame (level kinds, column kinds, unique) | `tracer.py:663-710`, `usage.py:575-622` | 60 | no receiver is known: everything stays pandas |
| 2 | PreparedTable build: level codes, combined code, grid, typed column copies | PYX 241-414, PXD 19-113 | 270 | all native reads |
| 3 | Keyed lookups: `code_ll`, `code_obj`, `row_codes`, `kcode` by C type, `f64_*`/`i64_*`/`contains_*` | PXD 62-103, 136-260; PYX 59-119, 725-743 | 210 | all native reads |
| 4 | One table per frame object (`prepared`, weak reference, lock) | PYX 425-470 | 45 | preparation repeated per call |
| 5 | Per-item preparation of a frame the item computes (`bind_own`, per-thread signatures, 8-evaluation rule) | PYX 473-524; FR 306-321, 609-611, 1050-1052 | 75 | reads of item-computed frames stay pandas |
| 6 | Signatures and slots: the traced schema is re-checked on the run-time table, once per table | PYX 570-623; PXD 355-358; FR 848-873; `cli.py:467-473` | 100 | a frame of other kinds at run time would be read wrongly; this is the part that makes "does not depend on the sample" true for values |
| 7 | Publication by the Data cell (`_mx_ptable_`, `_mx_prepared_`), renamed C entry, `TableCell` | TC 45-129, 166-198, 213-282; `runtime/table_cells.py` | 230 | each item would take the map lock once per receiver |
| 8 | Per-item holds (table, state, key code, row) and their reset | `holds.py`; FR 588-616, 945-990 | 180 | a lookup and reference per read |
| 9 | Receiver and once-bound locals; in-place edit by the formula keeps pandas | FR 267-321, 518-557 | 95 | `T = data.t()` spelling; wrong reads of an edited frame |
| 10 | Key typing: `is_int`, `fits_long_long`, enum constants, tuple keys checked at run time, parts fixed per item | KF 419; FR 881-1045, 1262-1356 | 680 | object lookup per key part (dict or boxed int) |
| 11 | Scalar read L1, membership L7, `float()`/`int()` dropped, pandas fallback text | FR 2374-2559, 2752-2773, 2832-2876 | 260 | the 358 scalar sites; base of every form |
| 12 | Recorder of the sample's reads | RN 476; `cli.py:146-154` | 490 | three vetoes (FR 1754, 2366-2372, 2388-2395) and the audit |
| 13 | Audit and reports | FR 141-215, 1107-1197, 1243-1260; `cli.py:474-523` | 240 | `_mx_read_audit.json`, `strict_tables` |
| 14 | Check mode `verify_inputs` | PYX 635-668 | 34 | no way to detect an in-place edit |
| 15 | Runtime of the forms kept in the base file (views, walks, asof, extremes, ZIP) | PYX 151-162, 527-567, 681-711, 746-906; PXD 264-351 | 390 | belongs to commits 2-4 |

## 2. Complexity
| Cut | Lines | Expected cost | Risk | Verify on ten models |
|---|---|---|---|---|
| C1 Recorder and audit become opt-in (`--read-audit`), out of the base commit | -730 from default path (RN, FR audit, cli) | translation 1.5x faster (brief's figure); run time: not measured | a place that only raised in the sample is lowered: its error can be another type than pandas'; the "several rows" veto is already covered at run time by `is_label` (FR 1341-1349) | build with and without; diff `_mx_table_reads.json` and `_mx_classes.py`; t2 |
| C2 Dead asof arrays: `asof_keys`, `asof_labels`, `asof_position`, `PreparedTable.asof` are read by no generated code (only `tests/test_prepared_tables.py::test_asof`); built for every flat numeric index (PYX 407-413) | -30 | none; saves memory (see C3) | none | tested: 23 of 24 runtime tests pass, the 24th tests the removed method |
| C3 Object columns copied with one `column.iat` call per value (PYX 389) | +3 | measured with C2 on a 200,000 x 13 table (8 float, 4 text, 1 int): prepare 2.7 s -> 0.21 s, retained 49.6 MB -> 35.2 MB | values and types equal on the probe; only object and `str` dtype take the fast path | time of first read of the model point table |
| C4 `_1` variants (`f64_1`, `i64_1`, `contains_1`, `row_1`, `_row_1`; PXD 222-260) duplicate `_c` with count 1, which the C compiler folds after inlining; `lcode` duplicates `kcode[long long]` | -45 | expected none; not compiled, not measured | low | instruction counts |
| C5 `bind_own`, `_fresh`, `_thread_signatures`, the 8-evaluation rule: use `bind_ref` for every receiver | -75 | unknown: I do not know which of the 57 models read item-computed frames | a frame per item registered in the shared map (lock once per item) | count `bind_own` in generated text of the ten; t2 |
| C6 Publication decoupled from the rename (TC 104 requires `n in mapping`; TC 188 drops a name two classes share) | about -20 | fixes R7; not compiled | low | generated text unchanged where names are unique |
| C7 ZIP runtime (PYX 27-34, 527-567, 764-799; one model) moves to the tail commit | -95 from base | none | none | the one model |
| C8 `view_start`/`view_stop` repeat `view_range` (PYX 802-820) | -20 (forms) | one search more where both ends are asked; not measured | low | t2 |

Duplicates of pandas or Cython: none worth cutting; the fused `kcode` (PXD 149-175) is the right use of Cython
typing and makes most of `KeyFacts.is_int` (KF 300-374) an optimisation, not a need (see 6b).

## 3. Generic
| # | Fitted to these models or one spelling | Case |
|---|---|---|
| G1 | A receiver is only a zero-argument cached Cell call (FR 294-304). A frame held by a Reference is never one | `examples/BasicTerm_S_nomx/_mx_classes.py:393` `self.mort_table[...][...]`, `:382` `self.model_point_table.loc[self.point_id]`: upstream's own example gets no fast read |
| G2 | Name shared by two classes switches the fast read off (R7) | `Calc.dup` beside `Data.dup` |
| G3 | `PREPARE_MIN_EVALUATIONS = 8` (FR 1052) decides from call counts of the sample (`tracer.py:218`) | breaks the rule "nothing may depend on call frequency"; values are equal, speed is not |
| G4 | Limits: 6 key parts (FR 45), 8 levels, grid 65,536 (PYX 24-26), columns exactly float64/int64 (float32, int32, bool, nullable go the object way) | reported, not wrong |
| G5 | Only `.loc`/`.at` with a literal column and a complete key; `T["c"][k]`, `T.loc[k]["c"]`, `T.c[k]` are other commits or nothing | spelling |

Changes for an existing upstream user (`prepared_tables` is on by default, `config.py:60`):
U1 a frame cell can raise at run time (R2); U2 typed reads lose numpy scalar behaviour (R4);
U3 in-place edits are no longer seen (R3); U4 every frame cell of a class without items is copied when first
computed, once any read of the model is lowered (2.7 s and +74% memory for the table of C3), even a table no
read uses natively; U5 the sample runs through an instrumented loader (R6) and translation is 1.5x slower;
U6 `space.cell` of a frame cell is a property giving a `TableCell`, not a bound method (TC 247-250).

## 4. Reliable
All "tested" rows: minimal model, live against compiled, same process.
| # | Kind | Reproduction | Live | Compiled | Fix |
|---|---|---|---|---|---|
| R1 | silent | `Data.dt()` = frame with `pd.to_datetime([...])` index; `(str(y) + '-01-01') in data.dt().index` | `True` | `False` | F2 |
| R1b | raises | same table `data.dt().loc[str(y) + '-01-01', 'q']`; IntervalIndex `band().loc[5, 'q']`; also Period and Timedelta text keys (runtime probe) | 1.0 | `KeyError: None` | F2 |
| R2 | raises, no read needed | `Data.cat()` = frame with unordered `CategoricalIndex`; `Data.mix()` with index `[1, 'a', 2.5]`; a cell only takes `len(data.cat())` | 2, 3 | `TypeError` from `index.max()` (PYX 401) inside the bind of the cell; uint64 label above 2**63: `OverflowError` | F1 |
| R3 | silent | `flat()` returns the Reference frame `SRC`; read `plain(2)`; `SRC.loc[23, 'q'] = 99.0`; first call of `plain(3)` | 99.0 | 4.0 (copy of the first bind; documented in `doc/source/table_reads.md`, check mode is opt-in) | proposal P1 |
| R4a | raises | `data.flat().loc[20 + t, 'q'].round(1)`; `...loc[20 + t, 'n'].item()` | 0.2; 2 | `AttributeError: 'float' object has no attribute 'round'` | F4 (not compiled) |
| R4b | raises | `1.0 / data.flat().loc[20 + t, 'q']` with value 0.0; `5 // ...'n'`, `5 % ...'n'` with 0 | `inf`; 0; 0 (numpy) | `ZeroDivisionError` | P2 |
| R4c | silent | `isinstance(data.flat().loc[20 + t, 'n'], int)` | `False` | `True` | F4 or documented boundary |
| R5 | silent where live raises | float level, key `True`; bool level, key 1, 0, 1.0; two levels `('a', True)` on a float level (runtime probe, `code_obj` PYX 99) | `KeyError` | the value of label 1.0 / True | F3 |
| R6 | sample | formula `x = {(1,): 5.0}; return x[t,]` | 5.0 | the recorder evaluates `x[t]` (RN 299): the sample raises `KeyError`; with a tolerant sample the cell is typed from a raise. Also `self.__m(t)` is called unmangled (RN 317) | F5 |
| R7 | speed, by name and by sample | item cell `data.dup().loc[1 + t, 'q']`, `Calc` also has a member `dup`, 3 evaluations per item | - | read left on pandas in a per-step cell; report says "no rule recognises this form"; with 8 evaluations each item would copy the whole shared table (`bind_own`) | C6 |
| R8 | error type | int level, key `2**70` | `KeyError` | `OverflowError` (PYX 92-93) | F3 |
| R9 | threads, not measured | in a class without items every formula call takes the global `RLock` (PYX 448, via `bind_ref`) | - | contention under workers | hold the published table as items do |

Fixes, lines:
- F1 (tested): PYX 235-238 and 226-230 `except Exception` instead of `except ValueError`: any failure of the
  preparation leaves the reads on pandas.
- F2 (tested): PYX before 323: `for plain in (index.levels if index.nlevels > 1 else (index,)):` raise
  `ValueError` unless `type(plain) in (pd.Index, pd.RangeIndex)` and `plain.inferred_type in ('integer',
  'string', 'floating', 'mixed-integer-float', 'empty')`. Datetime, period, timedelta, interval, categorical
  and bool levels then stay pandas (they are found by pandas' own rules, not by equality of the label).
- F3 (tested): PYX 99 `if isinstance(key, (bool, np.bool_)): return -1` before the dict lookup; PYX 92 return
  -1 for an int outside 64 bits. Cost: one type test per object-level key; not measured on a model.
- F1-F3 together: 51-line diff, the 24 runtime tests pass, every difference of the probe is gone except a
  partial tuple through the Python-only `value()` and a bool as a run-time column name (`name_position`,
  PYX 73-85: `True` finds column `1`; table-forms).
- F4 (not compiled): FR 2374 `leave_Subscript`: when the parent of the read is an attribute access, decline
  (record the reason) instead of typing it; needs the parent map the facts already hold.
- F5 (tested on the rewritten text): RN 299 `one = len(parts) == 1 and not isinstance(updated.slice[0].comma,
  cst.Comma)`; a tuple otherwise. The same shape test is missing in FR 98-104 (`_index_parts`; not tested).
- P1 (proposal): R3 cannot be fixed cheaply while values are copied. Options: (i) keep the copy and say in the
  proposal text that a compiled model reads inputs as of first use, with `verify_inputs` as the check; (ii)
  read float64/int64 values through pointers into pandas' own blocks with a row indirection (one more load
  per read; cost not measured, and pandas may reallocate a block on a copy-on-write edit, so the pointer must
  be re-taken when the block object changes). I would propose (i) upstream and state it as a contract.
- P2 (proposal): R4b is the same boundary upstream already has for a typed cell result; with tables it reaches
  a read used directly as a divisor. Either document it, or emit the division of a typed read under C
  division rules for doubles (not compiled; integer division must keep raising or keep numpy's 0).

Outside my area, seen in passing: `Calc.plain(9)` raises `IndexError ... Expected t in [0, 4)` where live
raises `KeyError` (cache domain taken from the sample).

## 5. The split into four commits
The split by form is right for the compiler side, but not for the runtime: PYX/PXD hold 390 lines used only by
views, walks, asof, extremes and ZIP (row 15), and the recorder is not needed by the base at all. Proposed:
| Commit | Content | Lines (estimate) |
|---|---|---|
| 1 base | rows 1-4, 6-9, 11 of section 1, key typing reduced to `kcode` by C type plus `is_label`, report `_mx_table_reads.json`, F1-F3 | about 1,700 |
| 1b | static key typing (`is_int`, held key codes, enum constants, tuple keys), as a pure speed commit with its measured value | about 500 |
| 2 rows | row cells and row fields, `hold_row`, `object_field`, `field_f64` | as planned |
| 3 sub-tables, extremes, membership of prefixes | `prefix_range`, `view_*`, `max_ll`, parts of extremes | as planned |
| 4 tail | ordered lookups, index walks, ZIP, per-item preparation (C5) | declinable |
| separate | recorder and audit (C1), opt-in | 730 |
Smallest first commit: one-level and multi-level `R.loc[key, "c"]` / `R.at[...]` of float64/int64 with a
complete key written as a tuple display, Series `R.loc[k]`, `key in R.index`; receiver a Data frame cell or a
local bound once to it; table published by the Data cell and held per item; slots by signature. It needs none of:
recorder, audit, `bind_own`, held codes, tuple keys checked at run time, enum key locals, object columns.

## 6. Bottom line
Base today (rows 1-14, without row 15 and without the enum and vector parts of TC): about 3,250 lines.
- (a) same measured speed: about 2,400 lines = today minus C1 (recorder and audit opt-in, 730), C2, C4, C5,
  C6. C2/C3 tested; that C1, C4, C5 cost no run time is expected, not measured.
- (b) within 25-30%: about 1,700 lines = (a) minus static key typing and per-item held key codes (commit 1b,
  about 500), minus the `_mx_prepared_` publication methods if a per-item `bind_ref` is accepted (about 85),
  minus `verify_inputs`. The 25-30% is my estimate from the generated text (a key part then costs one
  `kcode` on a boxed or C value instead of a held code); it is not measured and must be before it is quoted.

Blocking by the brief's ranking: R1 (silent), R2, R4a (regressions for unmodified upstream use), R3 (silent,
documented, needs the owner's decision), R5 (silent where live raises), R6 (valid model fails to translate).
