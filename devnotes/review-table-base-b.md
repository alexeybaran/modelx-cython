# Review: table-base (reviewer b)

Branch `review/r4-table-base-b`, base b326334. Stack: CPython 3.14.6t, Cython 3.3.1, modelx 636877b, lifelib
04d69db, pandas 2.x. Probes are small scratch models (two to thirteen cells), built one at a time with
`python -m modelx_cython <pkg> --sample sample.py --spec spec.py`, and compared with live modelx. No lifelib model
was built. "Built" below means the change was compiled and the probe rerun; "not built" means reasoning only.
Line numbers are at b326334.

## 1. What the code does
| # | mechanism | where | lost without it |
|---|---|---|---|
| M1 | PreparedTable build: one code per level label (dense int, sorted int, dict for other labels), combined codes sorted, grid when product <= 65,536, f64/i64 blocks, object columns as tuples | `runtime/_tables.pyx:241-416` | every native read |
| M2 | inline lookups by level codes: `lcode/ocode/kcode`, `f64_1/_c`, `i64_1/_c`, `contains_1/_c`, `missing_codes` (KeyError) | `_tables.pxd:136-246`, `_tables.pyx:725-745` | the read itself (P4: tens of ns vs about 9 us per pandas read) |
| M3 | read signatures and slots: per model, (levels, columns, label, kind) ids written to `_mx_table_signatures.py`; each prepared table decides once which reads it serves (`_decide`), every read tests `slot(ref, id) > 0`, else pandas | `_tables.pyx:570-624`, `_tables.pxd:351-358`, `frame_reads.py:848-873`, `cli.py:497-503` | the guard that keeps a run-time frame of other kinds on pandas (contract test "float column -> int column") |
| M4 | binding: global `id(frame) -> (weakref, table)` cache under an RLock | `_tables.pyx:425-470` | per-call preparation in non-item classes |
| M5 | holds: per-item fields `_mx_tp_n/_mx_tq_n` (table, state), held key codes and rows (`hold_code`, `hold_row`); Data frame cells publish their table at bind (`_BindFrame`, `_mx_prepared_<name>`) | `frame_reads.py:588-617, 945-990`, `holds.py` 103, `table_cells.py:45-104, 213-283`, `_tables.pyx:671-723` | probe P4: 208 ns/call held vs 385 ns/call per-call lookup (two reads per formula, called from Python); pandas 17,700 ns |
| M6 | `bind_own` + per-thread signature copies, gated by `PREPARE_MIN_EVALUATIONS = 8` sampled evaluations per item | `_tables.pyx:473-525`, `frame_reads.py:306-321, 1050-1052` | native reads of frames an item computes itself |
| M7 | receivers: zero-argument cached cell calls (or a local bound once to one) with a traced frame schema; prefix and parameter cells | `frame_reads.py:294-383`, `tracer.py:193-218, 341-350, 663-711` | everything; References are not receivers (see G2) |
| M8 | key emission: typed key parts (`kcode` fused on the Cython type), `is_label` when not known int, enum constants, tuple keys checked at run time, item-fixed parts held | `frame_reads.py:881-1045, 1262-1357`, `key_facts.py` 419 | C-integer lookups without boxing |
| M9 | read recorder: wraps every subscript/call/`in`/loop of the formula text during the sample; notes per site | `read_notes.py` 476, `cli.py:148-154, 386` | three sample vetoes (`frame_reads.py:1745-1760, 2366-2372, 2388-2395`) and the reports |
| M10 | audit and reports: `_mx_read_audit.json`, `_mx_read_notes.json`, `_mx_table_reads.json` | `frame_reads.py:141-216, 1133-1198`, `cli.py:437-439, 489-525` | diagnostics only; no generated code depends on them |
| M11 | verify mode (`verify_inputs`): re-prepare and compare at each bind | `_tables.pyx:635-669` | the only detector of in-place edits (opt-in) |
| M12 | API: `space.cell` becomes a `TableCell` property, the C entry renamed `_mx_cell_<name>` | `table_cells.py` (shared with enums/vectors), `runtime/table_cells.py:51-71` | `cell[()]` parity; not speed |
| M13 | data used only by forms but built in M1: `pos_row`, `ordered_within`, `level_names` (walks), `asof_keys/labels` (asof), `max/min_parts` (extremes), name map (column by expression), `_zip_axes` | `_tables.pyx:282-301, 351-364, 403-415, 27-34, 527-568, 764-800` | belongs to the form commits, not to the base |

## 2. Complexity: what can go
| cut | lines | expected cost | risk | how to verify (<=10 models) |
|---|---|---|---|---|
| C1 recorder + audit (M9, M10). Replace the `selects_rows` veto by the run-time `is_label` on object-level parts (R3 fix, built); drop `raised_veto` (speed only); `_vector_argument` keeps its text test | ~660 (`read_notes.py` 476, audit 141 lines, cli ~45) | none expected on run time; translation about a third faster (owner's 1.5x) | lose a developer report; sites that only raised in the sample become native (still KeyError as pandas) | diff the generated `.pyx` of 10 models: lowered sites a superset; t0 equal within 0.2%; values equal live |
| C2 global cache + RLock (M4): a Data cell's bind prepares fresh (as `bind_own` does), and non-item classes read the published table as items already do | ~35 | P4 suggests a non-item read path near 208 instead of 385 ns/call | fixes R1; item-computed frames shared by identity are prepared per item | not built; run P1 stale case, test_table_stress, t0 on 3 models with non-item readers |
| C3 ZIP terminal form, runtime part (in M13) plus `zip_prefix` and `builtin_zip_walk` | ~90 runtime + ~110 compiler | one model loses that form | none for other models | forms reviewers own the measurement |
| C4 Python access methods of PreparedTable (`key_codes/row/value/contains/asof`), used only by tests | ~50 in `_tables.pyx:172-222` compiled into every model | none | move into `tests/test_prepared_tables.py`, which already compiles a pyx | run that test |
| C5 open-addressing column name map: duplicates the CPython dict; `PyDict_GetItemWithError(column_positions, label)` is borrowed, and positions are small immortal ints | ~40 | unknown on free-threaded contention (the reason given in the docstring) | not measured | t2 on the models with the 28 column-by-expression sites |
| C6 `kcode` already dispatches on the Cython type and `is_label` folds to True for C types: the int-level special cases of `key_part` (`int_argument`, `fits_long_long`, part of `is_int`) could go | ~100-150 of `key_facts.py` | none if Cython infers the walrus key local as before | redesign, not built | generated C of 10 models instruction-equal |
| C7 TableCell API (M12) out of the base: the published table needs `_mx_prepared_<name>` only, not the rename of the public name | ~120 | none on speed | upstream users keep `space.cell` as a method | not built |
Nothing in the base duplicates pandas; M1 is the only copy and is needed. Cython has no equivalent of M2-M5.

## 3. Generic
| # | fitted where | case | behaviour on another model |
|---|---|---|---|
| G1 | the value of every native read is a C `double`/`long long`, cast also on the pandas branch (`frame_reads.py:43, 1504-1512`) | any formula that divides, rounds or calls a method on a read | see R2; upstream kept the numpy scalar inside the formula (by reading: upstream has no table lowering, so `T.loc[...]` stays a pandas call returning `numpy.float64`) |
| G2 | receivers must be zero-argument cell calls (`frame_reads.py:294-321`) | upstream lifelib `BasicTerm_S/Projection/__init__.py:437` reads the Reference: `model_point_table.loc[point_id]`; the lean models read `data.model_point_table()` (e.g. `Basis_DE_S:406`). Probe P1 `direct`: `SRC2.loc[20 + t, 'q']` stays pandas | an unmodified model whose tables are References gets no table speed-up; the speed depends on rewriting into Data cells |
| G3 | index lookup assumed to be dict equality for every non-int level (`_tables.pyx:241-261`) | IntervalIndex, DatetimeIndex with text keys, big ints | R4 |
| G4 | `PREPARE_MIN_EVALUATIONS = 8` from the sample's count per item | an item formula sampled 7 times stays pandas, sampled 8 times is native | speed depends on sample frequency (results equal) |
| G5 | ZIP admission: 2 levels, `str` level 0, `int64` level 1, `float64` column, empty `attrs`, pandas methods not patched | one model | none of it fires elsewhere |
| G6 | upstream behaviour change: `space.flat` is a `TableCell` property, not a bound method (`table_cells.py:247-250`) | `type(m.Data.flat)`, `m.Data.flat.__doc__`, `inspect.signature` | call and `[()]` still work |

## 4. Reliable
All reproductions are in scratch probes P1-P3; live is modelx 636877b on the same frames.

| # | class | minimal reproduction | expected (live) | actual (compiled) | fix |
|---|---|---|---|---|---|
| R1 | **silent** | Data cell `viaref(): return SRC`; Calc `T = data.viaref(); return T.loc[20 + t, 'q']`. `m1 = Model(); m1.Data.SRC = F; m1.Calc.viaref(0)`; then `F.loc[20, 'q'] = 99.0`; `m2 = Model(); m2.Data.SRC = F; m2.Calc.viaref(0)` | 99.0 | 0.001 (the copy prepared for m1: global cache `_tables.pyx:445-461` keyed by `id(frame)`) | C2 (not built). The verify-mode message (`_tables.pyx:~645`) and `table_reads.md:65-75` tell the user to do exactly this, so both are wrong today |
| R2a | **silent** | `round(T.loc[20 + t, 'q'], 4)`, q = 0.02545 | 0.0254 (numpy rounding of `numpy.float64`) | 0.0255 (Python `round` of a double); 4/4 values differ by 1e-4; 1.6% of (5-digit value, 2-4 digits) pairs differ | box the read as `numpy.float64` when its value reaches `round` (see R2 fix) |
| R2b | **silent** | `round(T.loc[20 + t, 'q'] * 1.0, 4)` | as R2a | as R2a: the numpy scalar is lost through arithmetic | same; needs the flow of the value, not only the parent node |
| R2c | **silent** | `T.loc[20 + t, 'n'] ** -1` (int64 column) | ValueError (numpy) | 1.0, 0.5, ... | same |
| R2d | **silent** | `type(T.loc[k, 'q']).__name__`, `isinstance(T.loc[k, 'q'], np.floating)` | float64 / True | float / False | same |
| R2e | raise | `T.loc[k, 'q'] / T.loc[k, 'z']`, z = 0.0; also `%`, and int64 `n / zn`, `n // zn` | inf / nan / 0 (numpy warning) | ZeroDivisionError | guard: `a / b if b != 0 else <numpy op on boxed a>`; one compare, no cost when b != 0 |
| R2f | raise | `T.loc[k, 'q'].round(3)`, `T.loc[k, 'n'].item()` | works | AttributeError: the read is lowered under an Attribute (`(...double...).round(3)`) | decline (or box) when the read is the value of an Attribute; one check in `leave_Subscript` (`frame_reads.py:2374`) |
| R2g | raise | `T.loc[k, 'q'] ** 0.5` with q < 0; `T.loc[k, 'n'] * 2**62` | nan; wraps | TypeError; OverflowError | box under `**`; overflow as integer area decides |
| R3 | raise, depends on the sample | text index; key `KEYS[t]` traced as `str`; at run time `KEYS[1] = ['A', 'B']` or `slice('A', 'B')` | a Series | TypeError (unhashable list in `code_obj`) / KeyError(None): object-level parts get no `is_label` test (`frame_reads.py:1341`) | **built**: apply the `is_label` condition also to `how == 'object'` parts that are not literals (3 lines); now falls to pandas (then fails only at the cell's traced float return, as any typed cell) |
| R4a | raise | `IntervalIndex.from_breaks([20, ..., 24])`, `T.loc[20.5 + t, 'q']` | 0.1 ... | KeyError (dict lookup of 20.5) | **built**: `_build` refuses a level that is not exactly `pd.Index`/`pd.RangeIndex` (4 lines at `_tables.pyx:317`) |
| R4b | raise | `date_range(...)` index, key `'2020-01-0' + str(t + 1)` | 0.1 ... | KeyError | same fix, built |
| R4c | raise | object index `[1, 2**64]` | 0.1 | OverflowError at bind (`prepare` catches only ValueError, `_tables.pyx:228, 237`) | **built**: catch OverflowError too (2 lines) |
| R5 | thread timing (not reproduced) | two threads evaluating the same item's formula: both see `state == 0`; the second store of `_mx_tp_n` drops the first table while the first thread reads through the borrowed pointer; for a `bind_own` table that pointer is then freed | - | possible use after free | outside the supported use (collects split items); a compare-and-set of the state or a lock on the hold path closes it |
| R6 | speed depends on sample | G4, and the three vetoes of M9 | - | - | C1, static gate instead of sampled counts |

Built fixes R3 and R4: the probes then equal live on every case listed; `test_prepared_tables`, `test_read_notes`,
`test_read_audit`, `test_tracer` pass (65). The integration probe result with the fixes is in section 7.
R2 fix (not built): emit the native lookup unchanged, but box the value (`numpy.float64(x)` / `numpy.int64(x)`,
about 30-50 ns) unless every use of it, through `+ - *`, unary minus, parentheses and locals bound once
(`key_facts._Formula.loads`/`parents` already give this), ends in a comparison, `float()`, `int()`, `abs()`, a key
part of another read, or the return of a cell with a C result (the documented boundary). Division and `%` by a
divisor that is not a nonzero literal take the guarded form of R2e instead of boxing. Cost on the 57 lean models
is not measured; their text already wraps rounded table values in `float(...)` (ULSG_US_S:862,
Immediate_KR_S:552), which suggests few sites are boxed. Verify: count boxed sites per model, t0 on the 3 models
with the most.

## 5. The four-commit split
The split is right in direction, with three changes:
- Membership (L7, 15 sites) shares the key path of scalar reads (`contains_1/_c`, kind -1): put it in the base,
  not with sub-tables. It adds about 40 lines.
- The base commit must not carry forms' data: M13 (`pos_row`, `ordered_within`, `level_names`, `asof_*`,
  `max/min_parts`, the name map, `_zip_axes`) moves into the commit of its form, with its `_decide` kinds
  (-2 extremes, -3 walks, -4 rows, -5/-6 prefixes, -7 zip).
- The recorder and audit (M9, M10) should be in none of them (C1). If kept, they are a separate tail commit.
- Rows (1,358 sites) are the second commit; the ordered-lookup and walk tail can be declined (0 lean sites for
  ordered lookups).

Smallest first commit (estimate from the line map, not built): M1 without M13 (~180), M2 (~150 pxd), M3 kinds
>= 0 and -1 (~60), M5 (holds, published Data tables, `hold_code`; ~250 across `frame_reads`, `holds.py`,
`table_cells`), M7 without prefix/parameter cells (~150), M8 (~400 incl. `key_facts`), L1 literal column and L7
in `leave_Subscript`/`_column_read`/`leave_Comparison` (~250), tracer schema (~80), wiring (~40): about
1,550 lines, with the R1/R2/R3/R4 fixes.

## 6. Bottom line
(a) Keep the measured speed: keep M1-M3, M5, M7, M8; cut C1 (-660), C3 runtime (-90), C4 (-50), C2 (-35):
about -835 of roughly 3,500 base lines, no expected run-time cost (C2 likely faster for non-item readers).
(b) Within 25-30%: (a) plus C5 (-40) and C6 (-100 to -150); both unmeasured. Item holds (M5) must stay: without
them the per-call binding cost 385 vs 208 ns per formula call in P4 (1.85x on a read-heavy formula), more than 30%.
Blocking now: R1 (silent, contradicts the documentation) and R2a-d (silent; regression against upstream for an
unmodified model that rounds or divides table values). R3/R4 fixes are built and are 9 lines.

## 7. Runs
- Table-base unit tests: 65 passed. The full unit suite was stopped at 8% (each compiled test builds a model;
  too slow here); no failure had appeared.
- P1 (13 cells), P2 (7), P3 (2), P4 (timing, 2 spaces), each compared with live for t = 0..3.
