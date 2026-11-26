# R4 P3: reconciling the syntax bridge onto line A (plan only)

Line A = `origin/r4/p2-collect-3` 90da50b (its tree is `pr/fast-sec` 5982ebe + 692 lines of proposal 2; the two are
not ancestors of each other, `git diff --stat` between them is collect.py, config, parser, space_results, transformer).
`r4/p1-math` (146 lines) is not pushed and was not read: it is assumed in line A and every math statement below is
conditional on it. Line B = `origin/r4/integration` 8c03194. Fork point of the two: 3bc7be5; line A was cut from the
full tree by deletion (eec6755, -19,588 lines) and then rewritten (overflow flag, translation time, thin collect).
Method: `git diff`, an import/name check of every line-B module against line A's definitions (script, not kept),
signature comparison of shared modules, `devnotes/r4-test-layers.md`, the review notes on `review/r4-*`, and the
line-A reports `r4-s2-report.md`, `r4-p2-collect-report.md`. Nothing was built or run. Line numbers are line B's
unless marked `A:`. "Lines" are non-test lines.

## 1. Modules (non-test files of `modelx_cython/`)

**Identical (10):** `__init__`, `__main__`, `_numeric_cache.pyx`, `_wide_numeric_cache.pxd/.pyx`, `consts`,
`exceptions`, `locked_entries`, `monkeytype_tracing`, `typedefs`.

**Line A only (2):** `collect.py` 276 (thin collect + reused receiver, proposal 2), `native_sources.py` 31 (installs a
.pyx/.pxd pair under `_mx_helper_<hash>`; replaces line B's `helpers.install`).

**Line B only (34 files)**, by fate in proposal 3:

| group | files (lines) | fate |
|---|---|---|
| bridge, new | `enums` 438 (B1); `reductions` 942, `_float_sum_state.pxd` 233, `runtime/_native_streams.pxd` 28 (B2); `vectors` 2066, `runtime/_native_vectors.pxd/.pyx` 252+363 (B3); `key_facts` 402, `runtime/table_cells.py` 63, `runtime/__init__.py` 1 (B4); `row_cells` 410 (B5, 30 lines B4); `read_notes` 477 (B4 ~300 / B7 ~177) | bring |
| bridge, shared by layers | `frame_reads` 2659 (B4 ~1190, B5 ~307, B6 ~975, B7 ~188); `runtime/_tables.pxd/.pyx` 352+701 (B4 ~761, B5 ~55, B6 ~237); `table_cells` 534 (B3 ~250, B4 ~284) | bring, split |
| P1 residue line A lacks | `rounding.py` 35, `_rounding.pxd` 61 | bring as B0 (section 4) |
| old P1 equivalents | `checked_arithmetic` 187, `helpers` 385, `helper_members` 99, `provider_build_cache` 144, `_call_layout.pxd` 12, `runtime/_native_math.pxd/.pyx` 35+180, `item_locals` 200 | do not bring (section 3) |
| old collect | `serial_collect` 1137, `collect_adapter` 88, `factory_individuals` 141, `factory_reuse` 271, `runtime/factory_reuse.py` 22 | do not bring |
| orphan | `runtime/LICENSE.lifelib.txt` 20: no remaining runtime file cites lifelib (the licensed `_native_inputs` left in C1) | do not bring unless a ported file needs it |

**In both, different (27).** Line A's version is kept in every case (it owns entries, caches, holds, integers,
collect, translation time, docs). The last column is what line B adds on top that the bridge needs.

| module | A / B lines | diff +/- | what the bridge needs from B (layer, ~lines); the rest of the diff is B's older form |
|---|---|---|---|
| `transformer.py` | 1620 / 1930 | +400/-90 | B1 10 (`code_enums.prepare` 607, storage 313/1139, `bind_properties` 1177, `fixed_refs` in `_integer_reference_paths` 103); B2 22 (reductions block in `_rewrite_powers` 650-675); B3 50 (`vectors.lower` 636, cache decl 346/1105/1408, `cache_body` 1655, providers 208/605); B4 97 (`_table_reads*`, `_lower_frames`, `_pandas_reads`, `_table_reads_kept` at the three Python-method exits, `table_cells` pxd/rename 317/377/522, import 1008); B5 70 (`_row_cells`, `_table_reads_with_row_fields` 804-847); B6 63 (`_prefix_cells`, `_param_prefix_cells` 733-802); B0 10 (round 642-649, 987) |
| `tracer.py` | 1097 / 1257 | +222/-62 | B1 75 (IntEnum value info, enum-argument widening); B3 100 (`vector_shape`, `merge_vector_shapes`); B4 60 (`frame_schema`, `_column_kind`, `receiver_calls`/`_count_receiver`) |
| `admission.py` | 228 / 525 | +320/-23 | B2 15 (reduction keys); B3 30 (`vector_refusal`, cache); B4 200 (`TableReadSite`, `table_read_sites`, `untyped_read_warning`, `expressions_*`, `PANDAS_RECEIVER`) |
| `cli.py` | 619 / 718 | +121/-22 | B1 5 (`enums.plan` over all sources before parsing); B4 45 (`run_sample(record_reads=)`, `_mx_table_signatures.py`, `_mx_table_reads.json`, `table_cells.finish` as last pass); B7 40 (`audit_subtables`, `_mx_read_audit.json`, per-step warnings, `strict_tables`) |
| `config.py` | 347 / 427 | +93/-13 | B4 ~20 (`prepared_tables`, `strict_lowering`, env `MX_CY_STRICT_LOWERING` for tests); B5 3 (`typed_table_reads`, only if the inliner comes); B7 ~8 (`strict_tables`, `verify_inputs`); B0 3 (`native_math` guard of round, unless p1-math has its key) |
| `_mx_sys.pxd` | 626 / 611 | +95/-110 | B1 38 (`_mx_check_enum_schema` 217, `_mx_enum_at` 232, `_mx_enum_code` 237); B0 25 (`_mx_round_as_index` 79, `_mx_rounded_index` 91, `_mx_bare_round_index` 97, cimport of `_rounding` 9) |
| `integer_boundaries.py` | 1233 / 879 | +186/-540 | B1 ~20: the enum argument check in the call gate (`enums` dict, `_mx_enum_code`, 46-57, 83-85, 145) rebuilt on A's `_CALL_GATE`; B2-B4: entries in A's `_PROVIDER_RETURNS` (A:112) for the C-number functions of each new header (`_tables.pxd` 43, `_native_vectors.pxd` 21, `_float_sum_state.pxd` 8) |
| `builder.py` | 1154 / 1187 | +39/-6 | B3 25 (`vector_shape`, `uses_vector_cache`, vector return type); B1 4 (`enum`, `enum_safe` on Reference info) |
| `usage.py` | 622 / 655 | +45/-12 | B4 25 (`pandas_results`, `frame_schemas`, `_pandas_result`, `pandas_type_id`); keep A's `position`/wrapper |
| `holds.py` | 77 / 103 | +34/-8 | B4 20 (kinds `prepared`, `prepared_slot`, `prepared_code`, `prepared_row` and their declarations); not `reset_lines` |
| `space_results.py` | 207 / 197 | +18/-28 | B0 10 (rounded index conversion, with `_rounding`) |
| `ordered_arguments.py` | 129 / 145 | +36/-20 | nothing for speed (the min/max report via `reductions.untyped_entry` is optional, ~30) |
| `numeric_cache.py`, `held_arrays`, `parser`, `item_roots`, `inline_entries`, `automatic`, `build`, `powers`, `reparse_check`, `scopes`, `syntax`, `entry_pipeline`, `cell_entries`, `value_storage`, `_numeric_cache.pxd` | | small | nothing: older forms, labels in comments, old collect (`collect_shared_item`, `classify_shared_items`, `collect_cache_classes`), old integers (`checked_integer_powers`), dense-by-use, `exact_guards`/`item_entries` (dead or item_locals) |

Also outside the package: `pyproject.toml` needs `"modelx_cython.runtime"` package data (B4); line B pins
`modelx>=0.33.1.dev0`, line A `modelx`.

## 2. Per layer: what it needs and where it plugs into line A

| layer | line-B modules and hunks | lines | line-A interfaces it plugs into | as line B expects? |
|---|---|---|---|---|
| B0 (not a layer in B's record; P1 there) | `rounding.py`, `_rounding.pxd`, round helpers in `_mx_sys.pxd`, `space_results` and `transformer` round hooks; header install; `runtime/` package | ~150 | `_rewrite_powers` (A:608), `leave_Module` imports (A:672), `native_sources.install` | **no**: A has no native round at all; `native_sources.install` requires a .pyx for every provider (A:24-30), line B's `_native_streams` and `_float_sum_state` are header-only (B `helpers.install` 366-385 skips the .pyx) |
| B1 labels | `enums.py`; transformer 10; tracer 75; `_mx_sys` 38; builder 4; cli 5; admission 3; enum gate in integer guard 20 | ~590 | source pass before parsing in `cli.main_handler`; PXD/`leave_ClassDef` declarations; the integer call gate | **no**: (1) the enum gate extends line B's `_CALL_GATE` with an `enums` dict; A's gate (A:37) has a different shape; (2) `readonly_refs` storage must survive A's reused receiver (`__init__` + `_mx_copy_refs` then reset: A `collect.py` 98-111, 188-211) |
| B2 reductions | `reductions.py`, `_float_sum_state.pxd`, `_native_streams.pxd`; transformer 22; admission 15 | ~1250 | `_rewrite_powers` pass list, `helper_providers`, header install, integer guard | **no**: (1) `from .helpers import module_name` (reductions:8) -> `native_sources.module_name`; (2) `helper_reference` (reductions:888) reads `helper_members` that line B's tracer puts on module References: that is the math registry line A removed; re-point to p1-math's member resolution (unread); (3) A's flag pass must accept generated reduction code (A:122-137 already lists `StopIteration` handlers as harmless) |
| B3 vectors | `vectors.py`, `_native_vectors.*`; `table_cells.py` vector half (`vector_cells`, public tuple methods, `RenameMembers` exits); tracer 100; builder 25; transformer 50; admission 30 | ~3140 | PXD cache declaration, `__init__` initializers, cached body, providers; collect reset | **partly**: (1) `helpers.module_name` (vectors:53); (2) `from .reductions import _EMPTY` (vectors:1478): B3 needs B2; (3) `VecCache` field is reset to None by A's reset (A `collect.py` 197-201, found by scanning package pxds) and rebuilt by `__init__`: matches A's numeric caches, unverified; (4) vector `ivec_*` long long results unknown to `_PROVIDER_RETURNS` |
| B4 table base + L1/L4 | `frame_reads` ~1190 (dispatch, receiver, keys, holds, slots, L1/L4, numpy-scalar boxing); `key_facts`; `_tables` ~761; `runtime/table_cells.py`; `table_cells` ~284; **recorder core of `read_notes` ~300**; `row_cells.item_constant` ~30; holds 20; usage 25; tracer 60; transformer 97; admission 200; cli 45; config 20 | ~3500 | `holds.HoldTable`, `_rewrite_powers`, `leave_FunctionDef` Python-method exits, PXD declarations, last pass after the integer guard (cli A:403-418), resolver in `usage.py` | **no**: (1) `frame_reads` imports `read_notes` at module top (39) and decides by the recording (`raised_veto` 2188-2194, `selects_rows` 2215-2222): the recorder is B4, not B7, or those reads are lowered against what the sample saw; (2) `run_sample` has no `record_reads` in A; (3) `frame_reads.py:41` provider `runtime._tables` needs the `runtime` package and package data; (4) compiled modules import `modelx_cython.runtime.table_cells` (table_cells:29): a table model needs the compiler at run time, against A's upstream-regression test (no modelx_cython at run time); (5) `table_cells.finish` renames C members of table cells and References (D101a) after A's integer guard, which adds `_reference_write_gate` (A integer_boundaries) for integer References: order to be checked; (6) `_tables.pxd` C-number functions absent from `_PROVIDER_RETURNS` |
| B5 rows (L2) | `frame_reads` ~307 (`row_cell_field`, `_row_field`, `row_local_read`, `_field_read`, `_row_locals`, `iterrows_loop`); `_tables` `hold_row` ~55; optionally the inliner: `row_cells.py` ~380 + transformer 70 | ~360 without the inliner, ~810 with | as B4 | yes once B4 is in; `review-table-forms-b` C2: the inliner gives byte-identical generated code on its case (`typed_table_reads: False`) |
| B6 sub-tables, extremes, membership, walks | `frame_reads` ~975 (L3, L5, L7, L8, comprehensions); `_tables` ~237 (`prefix_range`, views, `view_start/stop`, `position`); transformer 63 | ~1275 | as B4 | yes, but `_vector_argument` (searchsorted over a vector) needs B3 |
| B7 audit and reports | `read_notes` ~177 (`as_json`, `by_text` reports), `frame_reads` `audit`, `audit_subtables`, `without_fallbacks` ~188; cli 40; config 8 | ~410 | cli report writing | yes |

Tests per layer (line B, `r4-test-layers.md`): B1 1681, B2 1436, B3 1778, B4 587, B5 556, B6 2593, B7 367, join 1556.
The join files need collect; on line A they must be rewritten to the thin collect hook (development modelx with
`_mx_collect_hook`), not to line B's `collect_adapter` protocol 2.

Interfaces checked and found as line B expects: `scopes` (`CstFacts`, `FormulaTransformer`, `Names`,
`OPAQUE_SCOPES`, `has_scope_effects`, `lexical_nodes`), `syntax` (`arguments`, `code`, `literal_value`, `walk`,
`dotted_name`), `usage` (`RESOLVED`, `UNKNOWN`, `_extract_self_chain`), `automatic.max_dense_entries`,
`powers` (`KIND_*`, `formula_local_kinds`, `kind_of_type_expr`, `OperandKind`), `holds.HoldTable`,
`_mx_sys` `_mx_pow`, `_mx_load_flag`, `_mx_store_flag`, `_mx_integer_result`, `_mx_exact_int64`.
A's `_FirstRead` lazy local kinds (A:520, 631) stays; line B's reductions call `formula_local_kinds` themselves.

## 3. Line-B code proposals 1-2 make redundant (must not come across)

| what | where (line B) | lines |
|---|---|---|
| old collect | `serial_collect` 1137, `collect_adapter` 88, `factory_individuals` 141, `factory_reuse` 271, `runtime/factory_reuse.py` 22; hooks: transformer `add_methods` 1178-1182, `serial_shapes`/`collect_workers` 574-595, memset/malloc imports 977-983, `collect_shared_item` branches 1237/1657; parser `collect_cache_classes`; cli `CollectRuntime`, `serial_collect.analyze`; numeric_cache `classify_shared_items`; holds `reset_lines`; held_arrays, builder `collect_shared_item`; `_mx_sys` `_mx_collect_workers_override`, `_mx_collect_dispatch_state` 601-602; config `parallel_workers`, `serial_collect`, `parallel_collect_*`, `param_bound_checks`; `table_cells._reset_lines` 190-193 (dead, imports serial_collect) | ~1840 |
| old integer checks | `checked_arithmetic` 187; line B's `integer_boundaries` (A's 1233 wins); config `integer_checks`, `UNCHECKED_LIMIT`, `resolve_integer_checks` 101-119; cli integer mode 311-320, 467; transformer `CheckedArithmetic` 628/640 and `overflowcheck=` directives 1044-1053; powers `checked_integer_powers`; tracer `_integer_magnitudes`/`integer_magnitude`; `_mx_sys` `_mx_divisible*` 306-323; config `native_divisibility` | ~330 + 879 |
| helper/provider registry | `helpers` 385 (keep the idea of `install` for headers only: 6 lines in `native_sources`), `helper_members` 99, `provider_build_cache` 144, `_call_layout.pxd` 12, `runtime/_native_math.*` 215 (p1-math replaces), transformer `_bind_declared_helpers` 933-947 and its 4 call sites, tracer `module_helpers` | ~880 |
| options and sizing line A removed | dense by use (`automatic.dense_by_use`, `filled_per_receiver`, `DENSE_USE_FACTOR`; admission 168-197); `item_locals` 200 (A 341bf54: no SEC model; see risk R10); inline_entries `exact_guards`/`item_entries` 39 (dead per `review-reductions-b` C1); builder `_can_bind_native_result` (A's `_plain_formula`); min/max report in `ordered_arguments` (optional) | ~330 |

About 4,250 non-test lines of line B are not brought.

## 4. Order of work: first measurable state soonest

**The single layer that matters most is B4.** Every one of the 57 regional models reads pandas tables (the review
brief: table base is "every model"); the collect report on line A measured KLV_DE_S at about 50 ms per policy against
45 us on line B, "its table reads stay on pandas until the syntax bridge" (~1000x). The other layers are worth
1.13x-4.22x (reductions, median 1.44x), 21x-32x only on models with list cells (vectors), about 2x on enum-tuple sums
(labels). So B4 first, even though line B numbers it fourth: nothing in B4 imports B1-B3 except two lazy uses
(`key_facts.py:340` vectors alias, `table_cells.py:385/433` vector cells) that are guarded by "the module has
vector cells".

**Scalar reads alone (B4 as numbered) are not a useful end state.** By site count in the 57 lean texts
(`review-table-forms-b`): L1/L4 scalar 358, **L2 row field 1,358**, L3 sub-table 40, L5 extreme 107, L7 membership
15, L8 walk 83, column by expression 28. A row field left on pandas costs a row build (13.5 us) plus 1.2 us per
field per item and reader. With L1/L4 only I expect no model to reach line-B speed (not measured).

*Superseded by the measured census in section 7: the first slice must also take B6's index extremes (L5/L7).*

**Recommended first slice (B0-lite + B4 + B5 without the inliner):** header install and `runtime` package (~10),
B4 (~3,500, recorder core included), the L2 held-row reads (~360), no row-cell inliner, no audit, no sub-tables.
About 3,900 lines plus the B4 and B5 tests (1,143). Expected to reach line-B speed on models that have no vector
cells, no L3/L5/L7/L8 site in a per-step cell, and no generator reduction in a per-step cell. From the review notes,
these are **excluded** by name: vectors (IP_UK_S, BU_DE_S, Medical_JP_S, VA_KR_S, WP_UK_S, ADE_FR_S, Dep_FR_S,
Sofort_DE_S, CI_KR_S, Child_KR_S, LTC_KR_S, Medical_KR_S, Annuity_JP_S, Cancer_KR_S, VUL_US_S); sub-tables and walks
(CI_UK_S, IP_UK_S, ADE_FR_S); `return sum(...)` in double cells (Basis_DE_S, BU_DE_S, Index_DE_S, Medical_JP_S,
Riester_DE_S, Sofort_DE_S). Models whose only gap is labels (KLV_DE_S, FIA_US_S, RILA_US_S, EC_FR_S use members)
should come within a small factor. The full list of the remaining ~35 cannot be read from the repository (no per-model
form table exists; lifelib is not here): settle it with experiment E1 before choosing the measurement set.

**Then:** B6 (finishes the table models that walk or slice; same owner as B4/B5), B0 rounding (correctness of
`round` on table values, D2 in `review-table-forms-b`, and speed), B2 then B3 (B3 needs B2's `_EMPTY`), B1, B7 last
(reports only). The final series can still be presented in the owner's order B1..B7 if each commit is re-checked on
its own; building in the order above gives a measurable tree after the first ~3,900 lines.

## 5. Risks and unknowns (cheapest experiment for each)

| # | risk / unknown | cheapest experiment |
|---|---|---|
| E1 | which of the 57 need which layer | on line B, translate only (no C build) all 57 with the shipped sample and spec; per model read `_mx_table_reads.json` (form per site), `_mx_read_audit.json` (per-step), `_mx_admission.json` (reductions, vector cells, enum). 37-121 s per model (`labels-rule-w5.md`), about 1.5 h on one machine |
| R1 | A's overflow flag over bridge-emitted integer code: helper calls not in `_PROVIDER_RETURNS` (72 C-number functions in the new headers) are not known integers, so arithmetic on them is either unflagged C (silent wrap) or object; a construct the flag pass cannot cover sends the whole formula to `overflowcheck_fallback` (slower than line B) | after B4 and after B3, build `test_integer_vector_cells`, `test_prepared_table_integer_keys`, `test_table_cells_api`; grep generated `.py` for `_mx_flag_` next to `ivec_get`/table reads and read `_mx_integer_checks.json` `overflowcheck_fallback` |
| R2 | the recorder is needed by B4 (vetoes), and costs translation time (brief: about 1.5x) | on line B, run the B4-B6 tests with `record_reads=False` forced in `cli.main_handler`; the failures list what the recording decides |
| R3 | A's reused collect receiver resets every declared field generically (objects to None, C values to zero bytes). Hold codes are "one plus" encoded (0 = unbound), so that matches by reading; enum `readonly_refs` storage and renamed table References (D101a) are reset to None and must be refilled by `_mx_copy_refs` through the generated properties | build the join halves `test_held_table_handles_join`, `test_table_cells_api_join`, `test_enum_results_join` on the port with A's collect at 1 and 4 workers; values equal live |
| R4 | tracer conflicts: A restructured the hot paths (per-code-object decisions `MxCodeFilter._decide`, one `f_locals` view, `_add_arg_vals`, `_ReturnTypeMerger`); B1/B3/B4 hooks go into the same functions; translation time may regress | translate-only of three models (KLV_DE_S, VA_US_S, IP_UK_S) on A + hooks against A alone, wall time of the trace phase |
| R5 | run-time dependency: table models import `modelx_cython.runtime.table_cells`, against A's "no modelx_cython at run time" test | build one table test model, import it in a venv without modelx_cython; alternative to test: install the runtime module through `native_sources` like the caches |
| R6 | modelx version: line B's tests from the labels merge on run on development modelx `r4/modelx-integration` (stored enum module); A runs on upstream modelx | run the B1 test files on upstream modelx main before porting B1 |
| R7 | Python 3.10 (A's floor): `review-labels-a` 4.10, `str()` of an IntEnum member differs on 3.10 (silent); StrEnum exists only from 3.11 | the B1 tests on 3.10 |
| R8 | p1-math is unread: reductions resolve math members through line B's `helper_members` on References | read `r4/p1-math` once pushed; port `test_native_module_math` and one reduction over `math.exp` |
| R9 | rounding: line A has none; `round` over table values must keep live's numpy rounding (D2, silent in line B) | count per-step `round(` sites in E1; run `test_round_threads` and the D2 case on the port |
| R10 | `item_locals` (removed by A as unused by SEC) may matter on regional models | in E1, sum `typed_item_locals` from the build reports of line B |
| R11 | target speed was measured on line B with its old collect (reuse, native sums); A's collect differs (4-worker +15-38% before the reused receiver) | measure KLV_DE_S and VA_US_S books on the first slice with A's collect, as `r4-p2-collect-report.md` did |
| R12 | `table_cells.finish` renames members after A's integer guard and before A's `lower_locked_entries`; A's held Space results pin `root._v_<producer>` by name | B4 tests `test_table_cells_api`, `test_held_table_handles` and A's `test_space_item_results` on the port |
| R13 | options: line A removed options; B4 brings `prepared_tables`, `strict_lowering`, `typed_table_reads`, `strict_tables`, `verify_inputs` and two env switches | owner decision; recommended: `prepared_tables` (off switch) and `strict_lowering` (tests) only |

## 6. Size and parallel work

| layer | new files | hunks in shared files | total to bring | tests (B) | depends on |
|---|---|---|---|---|---|
| B0 | 97 (`rounding`, `_rounding.pxd`) | ~55 | ~150 | in P1 files | - |
| B1 | 438 | ~155 | ~590 | 1681 | - |
| B2 | 1203 | ~45 | ~1250 | 1436 | B0 (header install) |
| B3 | 2681 | ~460 (table_cells 250) | ~3140 | 1778 | B2 |
| B4 | ~2750 (frame_reads part, key_facts, _tables part, runtime, recorder) | ~760 (table_cells 284, admission 200) | ~3500 | 587 | B0 |
| B5 | ~410 (frame_reads, _tables parts) | ~0 / 70 with inliner | ~360 / ~810 | 556 | B4 |
| B6 | ~1212 | 63 | ~1275 | 2593 | B4, B3 (searchsorted on a vector) |
| B7 | ~365 | ~48 | ~410 | 367 | B4 |
| total | | | ~11,100 (~10,700 without the row inliner) | 8,998 + join 1,556 | |

Against line B's own estimate for B1-B7 (10,918, `r4-test-layers.md`), the difference is the B0 residue (+150), the
recorder moved from B7 to B4, the inliner made optional and integer-gate adaptations; the -4,250 of section 3 are
line B's P1/collect code that line A already replaced.

**Shared files per layer:** `transformer.py` (all), `tracer.py` (B1, B3, B4), `admission.py` (B1-B4), `cli.py`
(B1, B4, B7), `config.py` (B0, B4, B5, B7), `integer_boundaries.py` (B1 gate; B2-B4 `_PROVIDER_RETURNS`),
`_mx_sys.pxd` (B0, B1), `table_cells.py` (B3, B4), `frame_reads.py` and `runtime/_tables.*` (B4-B7),
`native_sources.py` (B0), `builder.py` (B1, B3), `holds.py`, `usage.py` (B4).

**Parallel plan (three porters and one integrator):**
- Integrator owns `transformer.py`, `tracer.py`, `admission.py`, `cli.py`, `config.py`, `integer_boundaries.py`. Each
  porter hands in their new files plus a short patch of hunks for these six, against line A's head; the integrator
  applies them in series. These hunks are small (10-200 lines per file per layer) and sit in separate functions
  (`_rewrite_powers` pass list, PXD declarations, `leave_ClassDef`, report writing).
- Porter 1: B0, then B1, then B2 (`rounding`, `enums`, `reductions`, headers, `_mx_sys.pxd`). No file in common with
  porter 3.
- Porter 2: B3 (`vectors`, `_native_vectors.*`, the vector half of `table_cells.py`). Starts after B2 is in, or ports
  against B2's tree; shares only `table_cells.py` with porter 3, which creates it in B4 (table half first, vector
  half added by B3).
- Porter 3: B4, B5, B6, B7 in that order, one owner of `frame_reads.py`, `_tables.*`, `read_notes.py`, `key_facts.py`,
  `row_cells.py`, `holds.py`, `usage.py`. These cannot be split between people: B4-B7 all edit `frame_reads.py` and
  `_tables.*`.
- Porters 1 and 3 can start the same day; porter 2 starts once `reductions.py` exists. The first measurable state
  (section 4) is porter 3's B4 + B5-lite plus porter 1's header-install lines.

## 7. E1: census of the 57 regional models on line B (measured, translate only)

Line B 8c03194 (`python -m modelx_cython <export> --translate-only`), development modelx `r4/modelx-integration`
df3ced9, pandas 2.3.3, CPython 3.13 (GIL build). Texts: lifelib `r4/l1-lean-b` c98ff5c9 (the bridge's text,
generated from upstream 20d4dae) and upstream lifelib main 20d4dae (original text). Each model is exported with
`mx.read_model(...).export(...)`. The sample is its product `run.py` with `model` bound to the export. Seven run.py
files (Sofort_DE_S, IncomeTerm_JP_S, Cancer_KR_S, Immediate_KR_S, Medical_KR_S, Term_KR_S, VA_KR_S) list
`model.Projection.cells`, which an export lacks; there the public attributes were listed instead. The spec is the
lean product's `spec.py` for both texts (upstream ships none). All 114 translations exit 0, at 28-121 s each. A layer
is needed when line B's output for the model uses it: B4 any lowered read; B5 an L2 site; B6 L5/L7 or L3/L8;
B1 `_mx_enum_at`, `_mx_definition_*`, enum domains or members rewritten to integers; B3 `_native_vectors` calls or
vector cells; B2 `native_reductions`; B0 native `round`; math `_native_math` calls (p1-math, assumed in line A).
This counts mechanisms; speed and values were not measured.

**The row inliner is not needed.** All 57 lean models were translated again with `typed_table_reads: false`. Each
of the 57 leaves exactly the same reads on pandas, per step and per policy, as with the inliner. B5 can come
without `row_cells.py`'s inliner, about 450 lines.

Lowered sites (both texts nearly equal): L1 350 (274 per step), L2 1,386 (308 per step), L3 41, L5 114 (106 per
step), L7 14, L8 51 (39 per step); no L4. **Extremes are per-step reads**, so L5/L7 belong in the first slice.

**Models in line-B form**, that is, every layer line B uses for the model is present. In brackets: of these, how
many also leave no per-step pandas read on line B itself. Math is assumed present. Each value is without B0 /
with B0.

| set of layers | lean text | original text |
|---|---|---|
| (a) B4+B5 (no inliner) | 0 [0] / 0 [0] | 5 [4] / 7 [6] |
| (b) + B1 | 5 [3] / 7 [5] | 5 [4] / 7 [6] |
| (c) + B3 | 5 [3] / 7 [5] | 5 [4] / 7 [6] |
| (d) + B2 | 9 [6] / 11 [8] | 9 [8] / 11 [10] |
| (e) + B6 | 50 [21] / 57 [24] | 50 [27] / 57 [30] |
| order by need: B4+B5+L5/L7 | 0 / 0 | 17 [14] / 20 [17] |
| ... + L3/L8 (rest of B6) | 1 / 1 | 34 [20] / 38 [23] |
| ... + B1 | 33 [15] / 36 [18] | 34 / 38 |
| ... + B2 | 42 [20] / 47 [23] | 45 [26] / 52 [29] |
| ... + B3 | 50 [21] / 57 [24] | 50 [27] / 57 [30] |

| layer | B0 | B1 | B2 | B3 | B4 | B5 | L5/L7 | L3/L8 | math |
|---|---|---|---|---|---|---|---|---|---|
| models (lean / original) | 7 / 7 | 56 / 0 | 12 / 14 | 10 / 5 | 57 / 57 | 57 / 57 | 39 / 39 | 28 / 28 | 18 / 18 |

The original text has no enum members, so it never needs B1. B1 is needed by every lean model except Term_US_S.
Medical_KR_S and VUL_US_S take B2 on the original text where they take B3 on the lean text. Even on line B, 33 lean
and 27 original models keep at least one per-step pandas read. These are listed in brackets as lean/original:
Cancer_KR_S 14/10, Child_KR_S 12/9, Cancer_JP_S 11/11 and VA_KR_S 11/10 have the most.

Per model (26 groups by identical needs, both texts, per-step pandas reads left on line B): `devnotes/r4-p3-census.md`.
**Revised order** (it replaces section 4's first slice):
1. B4, then B5 without the inliner, then B6's L5/L7 (about 310 lines of `frame_reads` plus the extremes of
   `_tables`). On the original text this brings 17 models to line-B form, 20 with B0.
2. The rest of B6: 34 (38).
3. B1. It is needed only for the lean text, where it lifts the count from 1 to 33 (36).
4. B2: 45 (52) original, 42 (47) lean.
5. B3 last: all 57 with B0. B0 rounding (~150 lines, 7 models on either text) comes with step 1.

**Smallest B4 that keeps the recorder's two decisions** (by reading, not compiled): about **3,150 lines**, not 3,500.
- **Kept from `read_notes.py` (about 220 of 477):** the import hook (`_Loader`, `_Finder`, `instrument`, `_walk`,
  `recording`), the subscript and method-call branches of `_Wrap`, `r_sub`, `r_call`, `_note`, `_shape`, a plain
  `kind`, `normal`, `_balanced`, `by_text` without table counts, `only_raised`, and `selects_rows`. `raised_veto`
  is called on subscripts and calls (`frame_reads.py` 2228, 2365, 2625), and `selects_rows` on subscripts (2215).
- **Left for B7 from `read_notes.py`:** `r_in`, `r_iter`, `r_it`, `r_v` and their wrapping, the provenance
  `_follow`, `as_json`, `table_read`, and `scalar_table_read`.
- **Also left for B7:** `admission.py`'s table-read sites and warnings (about 200); the transformer's
  `_pandas_reads`, `_table_reads_kept` and `_table_reads_report` (about 35); cli's `_mx_table_reads.json`,
  `_mx_read_notes.json` and `_mx_read_audit.json` (about 25); and `strict_tables` and `verify_inputs` (about 10).
- **Kept in B4:** `_mx_table_signatures.py`, because the runtime reads it (`_tables.pyx` 426-433).
- **Risk:** without provenance, a key that was a traced table column reads as `series`. That is still a selector,
  so the decision should be unchanged. Check it by running the B4-B6 tests with the reduced recorder.

## 8. Original text against the fork text (UC_FR_S, Dep_FR_S, Child_KR_S at 856b984): `devnotes/r4-p3-original-text.md`
