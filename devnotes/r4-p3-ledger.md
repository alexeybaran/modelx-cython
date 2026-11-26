# R4 P3 ledger: from the reference 633c9be to `r4/integration` 531e818

This covers the 125 commits from the reference 633c9be ("the reference") to `r4/integration` 531e818. Non-test
lines over that range: +2,036/-7,543, of which +1,352/-6,651 come up to 856b984.

## Method

By reading `git log` bodies, `git diff` per commit, and the test files each commit adds. Nothing was built or run.

**Whether a bridge branch has a change** was checked two ways:
- the share of the commit's added non-comment lines found in the same file of `r4/p3-tables`, `r4/p3-views`,
  `r4/p3-reads` and `r4/p3-reductions` (the runtime `_tables` is mapped to the bridge's root-level `_tables`);
- where the bridge reworked the code, by reading the equivalent code there.

**Classes:**
- **F:** the reference gives a wrong value, a wrong exception, a crash or a silent difference from live in the stated
  case.
- **S:** a simplification or removal, with no behaviour change claimed.
- **H:** known harmful to speed or to a model.
- **T:** tests, records or documentation only.
- **N:** a new capability. This is not one of the requested classes; it is added because some commits are neither a
  fix nor a simplification.

**"Bridge" column:**
- **yes:** the code is in the bridge branches as committed.
- **yes (adapted):** equivalent code is in the bridge.
- **n/a:** the mechanism does not exist on line A.
- **RE-APPLY:** the change must go on top of a module transplanted from the reference.

## Ledger

| # | commits | what it did | class | layer | bridge / transplant | lines; test |
|---|---|---|---|---|---|---|
| 1 | 5dd6091 | dense cache values left unwritten at allocation (S1) | N | integers/P1 | in line A (proposal 1) | 236/14 |
| 2 | 92c603c | scrubbed pickle copy returns its type reference: a leak under `CYTHON_USE_TYPE_SPECS` | F | P1 | in line A | 4/1; test_value_storage (50 reductions leave no block) |
| 3 | d12c0fa | copy of a scrubbed item under its critical section: a member could be replaced and freed while the copy borrowed it (free-threaded) | F | P1 | in line A | 100/28; test_value_storage (concurrent copies) |
| 4 | 834913a, 3bc7be5 | tests of value storage and locked Spaces | T | P1 | in line A | - |
| 5 | e0de350, 674f54c, ba14994, 8c92ac4, eafa5c5, 2a468ca, 311e3c0 | remove the helper table lane, the helper-package fallbacks and registered providers (C1) | S | other | n/a (line A never had them); transplants must drop `helpers` imports | -3,780 |
| 6 | e2d9aa2, 0d95b2b, 15454b7, 1b5d2e2, 09cbd68 | enum literal pass replaces the reference's enum mechanism | H (stops VA_KR_S) | labels | not brought | 240/802 |
| 7 | 1868bc0 | (inside the literal pass) `.name` of an integer by checked lookup: a negative integer wrapped to another member's name | F of H code | labels | not applicable as code; check the case on the transplant | 13/10; test_enum_literals, test_diag_mt_shared_refs |
| 8 | bae54ee | (inside the literal pass) members stay members under repr, type, isinstance, getattr, id, `!r`, `%`, as stored keys; identity tests are build errors | F of H code | labels | not applicable as code; check the cases on the transplant | 84/23; test_d98_enum_identity |
| 9 | cb7cd2c | an IntEnum with its own `__str__`/`__format__`/`__repr__`/`__hash__` is not enum-safe: `str(member)` gave the integer's text where live gives the class's own | F | labels | **RE-APPLY** (tracer `enum_safe`, also in the reference) | 3/1; no test in the commit: write one (an IntEnum with its own `__str__`, `str(K.A)` in a formula, equal to live) |
| 10 | 1616462 | a literal integer tuple in a native sum loops as a C integer (2.05x on member-tuple sums) | N | reductions | yes (adapted: 19ee271, tuples as C values) | 30/62 |
| 11 | a58f302 | the same loop kept live's Python integers when the variable is in arithmetic (C overflow where live does not) | F | reductions | superseded by line A's overflow flag (raises naming the cell) | 12/2 |
| 12 | a4529f6, 65ae191, bfb3c31, a2df0bc, d5758f6 | test split by layer, direct-route twins, layer record, live/compiled helper | T | - | bridge has its own tests | - |
| 13 | bcddcca | `sum(f(t=t) for t in ...)` renamed the keyword: TypeError (or a parser error over a literal tuple) | F | reductions | yes | 4/0; test_reduction_keyword_arguments (in bridge) |
| 14 | bd11687 | a reduction inside a vector subscript was lifted out of a call an earlier lowering put in: TypeError on every call | F | reductions | yes (code); test needs vectors | 17/4; test_reduction_in_vector_subscript: **bring after the vectors transplant** |
| 15 | bef4b7a | `all([...])`/`any([...])` tested every element: ValueError where live stops at the deciding element | F | reductions | yes | 4/1; test_reduction_boolean_list (in bridge) |
| 16 | c658d20 | `max((n, 2))` unrolled without binding its items: build stopped | F | reductions | yes | 4/1; test_reduction_direct_literals (in bridge) |
| 17 | 5f6898e | `sum(x.real for x in (1, 2))` wrote `1.real`: parser error | F | reductions | yes | 1/1; test_reduction_integer_attribute (in bridge) |
| 18 | 11c2a96 | the exact sum state follows the interpreter's float loop before 3.14 (1e16+1+1-1e16: live 0.0, compiled 2.0) | F | reductions | yes (adapted: `_reductions.pxd`, PY_VERSION_HEX) | 21/7; test_exact_sum_state (in bridge) |
| 19 | 952ce76, da0fcf1, c6b1954 | one reduction lowering: specialised double-sum and all lanes and `_native_iterables` removed | S | reductions | yes (the bridge has one lowering) | -394 |
| 20 | cb1f944, a0916bc, 31c37de, f995c82 | a provider with only a .pxd is installed as a header; streams is a header | S | other | yes (adapted: `native_sources` in p3-reductions) | 10/10 |
| 21 | e1b0642 | an index pandas looks up by its own rules (dates, periods, intervals, categoricals, bool, mixed labels) left to pandas: KeyError or False where live finds the key | F | tables | yes (8/10 lines) | 12/4; test_prepared_tables: **missing in bridge** |
| 22 | c2c8c74 | a bool key found 1.0 on a float level; an integer beyond 64 bits raised OverflowError where pandas raises KeyError | F | tables | yes | 5/1; test_prepared_tables: missing |
| 23 | c69b0f9, 23c4819 | a frame edited in place and given to a new model was read from the earlier model's prepared copy: stale values, silently | F | tables | yes (adapted: a generation per model, `renew`) | 78/34; test_prepared_tables, test_table_runtime_contract: **missing** |
| 24 | dc43e8f | a row pandas widens to one dtype kept a field as stored (numpy.int32): not live's value, could wrap | F | tables | yes | 5/0; test_prepared_tables: missing |
| 25 | 7476842 | copy an object or text column in one pass (preparation 2.7 s to 0.3 s) | S | tables | yes | 5/1 |
| 26 | 6805238, 95281b7, 61e3665 | remove the unused ordered-lookup arrays, per-thread signature copies and test-only lookups from the runtime | S | tables | n/a (the bridge's `_tables` was rewritten smaller) | -119 |
| 27 | c6bc317, 34c84ca, 856b984 | a read meeting numpy's rules is boxed as the numpy scalar: `round(T.loc[k,'q'],4)` 0.0255 against live 0.0254, `int64 ** -1` gave 0.5 where live raises; the merge repair 856b984 | F | tables | yes | 147/1; test_table_numpy_values (in bridge) |
| 28 | ea657a7, 967e2e8, b87bf4b | table tests and layer record | T | - | - | - |
| 29 | 030de71 | R1: a local list changed in place (sort, del, passed to a callee, aliased) was a typed vector: the change reached a copy | F | vectors | **RE-APPLY** | 83/5; test_vector_list_mutators, helper vector_live.py |
| 30 | 1eef6ff | R2: a value not proven an integer entering an integer vector was truncated (2.5 to 2) silently | F | vectors | **RE-APPLY** (vectors.py and runtime `ivec_item`) | 58/6; test_integer_vector_writes |
| 31 | c51c847 | R3: the same for float vectors (an int above 2**53 rounded silently) | F | vectors | **RE-APPLY** (vectors.py and runtime `vec_item`) | 29/6; test_float_vector_writes |
| 32 | 95e73b6 | R4: a vector branch of a conditional was converted to a list: TypeError on every call | F | vectors | **RE-APPLY** | 2/0; test_vector_conditionals |
| 33 | a7bcf56 | R5: `[None] * n` made `vec_full(n, None)`: build stopped | F | vectors | **RE-APPLY** | 8/0; test_vector_placeholders |
| 34 | 7b5c27c | renaming an expanded loop target renamed keywords and attributes (`w(t, s=s)`, `math.e`): TypeError | F | vectors | **RE-APPLY** | 8/0; test_vector_loop_renames |
| 35 | e79b132 | `self.<Ref>.<cell>(...)` of a vector cell read natively through the Reference (a new list per read before) | S (speed; makes 36 possible) | vectors | **RE-APPLY** before 36 | 36/11; test_vector_reference_reads |
| 36 | ec9ef8b, 714b127, 89f7ebf, b40a095 | remove `vector_ref_reads.py`, its provider helper, the `native_vector_ref_reads` key and doc | S | vectors | **re-apply later as a deletion**, after 35 | -198 |
| 37 | 131cd37 | `v = [0] * n; v[i] += 0.5` stopped the build (double into long long) | F | vectors | **RE-APPLY** | 8/0; test_integer_vector_updates |
| 38 | e82916d | `vec_of` called `list()` on an I64Vec: TypeError for a float vector cell returning an integer vector on an unsampled path | F | vectors | **RE-APPLY** (runtime `_native_vectors.pyx`) | 1/1; test_float_vector_of_integers |
| 39 | 1c3db1e | remove the concatenation spelling of vector returns (one CI_KR_S cell, below 0.5%) | S | vectors | re-apply later as a deletion | -90 |
| 40 | c410929 | vector sums use `_float_sum_state`'s F64Sum instead of a private copy | S | vectors | re-apply later, onto the bridge's `_reductions.pxd` | 15/44; test_vectors |
| 41 | 5b18018, 21675c3 | marker and layer record | T | - | - | - |
| 42 | 0f1837a | every build refused an export without collect protocol 2: released modelx exports could not compile | F | collect | n/a (no `collect_adapter` on line A) | 23/4 |
| 43 | e37f3e8 | every module imported `modelx_cython.runtime.table_cells`: a table-less model needed the compiler at run time | F | tables | n/a (the bridge copies `_tables` into the package, no runtime import) | 18/1; test_import_without_compiler |
| 44 | 7dbe2dc, 5c27c5c | delete the round memo (`RoundedInputs`, `_native_rounding`) | S | rounding/math | n/a (line A has no rounding; bring B0 from the head, which lacks the memo) | -254 |
| 45 | 128d4b4 | `math.ceil(x)` inside `lambda x:` took the formula parameter's type: Cython refused the model | F | rounding/math | check against r4/p1-math (not read; none of its lines are in the bridge) | 28/7; test_math_in_nested_scopes |
| 46 | 5a58734, 2a62677, 7d4831c, 49f986f, f062708 | delete typed min/max with `expression_kinds`, object providers and the callable math path (1.00-1.02x) | S | rounding/math, other | n/a (registry absent on line A) | -879 |
| 47 | 5b846c6 | layer record | T | - | - | - |
| 48 | 288fa5a | an absent prefix of a view whose reads were all lowered returned a value where live raises KeyError | F | views | yes (adapted, `_finish_views`) | 3/0; test_forms_view_keyerror: **missing in bridge** |
| 49 | a0230c9 | `w = df; w.loc[k, c] = v` left the reads of `df` on the stale prepared copy | F | tables | yes | 23/6; test_forms_alias_edit (in bridge) |
| 50 | 9fa932f | `isinstance(T.loc[k,"n"], int)` was True on the C integer (live: numpy.int64) | F | tables | yes | 15/5; test_forms_scalar_type (in bridge) |
| 51 | 07a3b8a | a hold evaluated before a branch that never reads the table raised where live returns | F | tables | yes (adapted, `bind_ref`) | 13/5; test_forms_hold_branch (in bridge) |
| 52 | cff460d | a non-literal key part on a text level was looked up as one label: TypeError where live selects rows | F | tables | yes | 11/5; test_forms_list_key (in bridge) |
| 53 | 6154c3e | the recorder passed `k` for `x[k,]`: the sample evaluated another read | F | tables (recorder) | n/a until the recorder comes (no `read_notes` in the bridge) | 2/1; test_read_notes |
| 54 | 6551a27 | an item's own frame was prepared by the sample's call count (code depended on the sample) | F | tables | yes (adapted, f551058: decided by text) | 7/11; test_forms_own_frame (in bridge) |
| 55 | efae68a, 2d51e1b, da8c6ce | dead parameters and imports, one test for several-rows key parts, one place for extreme spellings | S | tables | not in bridge (5/53, 3/9, 18/36): re-apply later where the code matches | -150/+108 |
| 56 | 54504f4, 618b525, 19c3147, 1453506 | remove the zip spelling and the ordered lookups (asof, searchsorted), compiler and runtime | H | tables | not brought; the bridge re-added asof (95faf03, test_forms_asof) | -500 |
| 57 | 3035e52, 0a2d914, 29340c9, ee945fd, b6056ca | tuple experiment: tuples of numbers built as sequences are vectors beside lists; Python gets tuples | N (lean text: 12 tuple cells) | vectors | optional after the vectors transplant | 276/20; test_tuple_vectors |
| 58 | d8c8637, a4ce7e8, 905c984, df43c42, fe8641f, 4080d3b, 3457f65, 1ef5a8d, 07be3ac, b446796, 61602c4 | tuples only: lists become Python objects, D94 and the R1 analysis removed | H (IP_UK_S doubled) | vectors | not brought | 250/401 |
| 59 | d23f373, ff2390c, ad12ac3 | join twins and layer records | T | - | - | - |
| 60 | bbca0b6, 58efd1f, ed1877a, 480c6c0, 424b4d9, 8c03194 | enum members by use (StrEnum, held parameters, enum gate in the integer guard) | H (KLV_DE_S 44%) | labels | not brought | 457/326 |
| 61 | 4b96b21, 531e818 | tests: a changed enum definition refused at load; IntEnum items in a native loop | T | labels | bring after the labels transplant, if the reference passes them | test_enum_by_type, test_audit_runtime_b2, test_diag_scale_int_items |

**Counts (rows):**

| class | rows | of which |
|---|---|---|
| F | 35 | 30 straight fixes; 2 fixes of H code (rows 7, 8); 3 superseded or not applicable on line A (rows 11, 42, 43) |
| S | 12 | |
| H | 4 | |
| T | 7 | |
| N | 3 | |

The F rows by layer: tables 12 (one of them the recorder), views 1, vectors 8, reductions 7, labels 3, P1 2, collect 1, rounding/math 1.

## What the transplants need besides re-applied fixes (by reading)

- **Labels from the reference** need its `enums.py` (625 lines) and the reference's enum hunks elsewhere:
  - in `transformer.py`: `code_enums.declarations`, `Constants`, `prepare`, `storage`, `bind_properties`, at
    reference lines 218, 318, 622, 704, 1216 and 1266;
  - the enum-keyed table paths e2d9aa2 removed: `frame_reads` about 76 lines, `key_facts` 23, `table_cells` 115,
    `runtime/table_cells` 19, `admission` 10. The bridge's table code comes from 856b984, after e2d9aa2, so it lacks
    them.

  The reference's `enums.py` imports `reductions.formula_lines` and `untyped_entry`, which the bridge's
  `reductions.py` has. The reference has no enum gate in the integer guard.
- **Vectors from the reference** need `vectors.py` (1,950 lines), `vector_ref_reads.py` (184) and
  `runtime/_native_vectors.*` (299 + pxd). Its imports `from .helpers import module_name` become
  `native_sources.module_name`. `held_arrays` is in line A.

## Re-apply after each transplant, in order, with tests

**After the labels transplant:**
1. cb7cd2c, `tracer.py` `enum_safe`, 3 lines. Write a test: an IntEnum with its own `__str__`, `str(member)` in a
   formula, equal to live.
2. Run the cases of 1868bc0 and bae54ee against the transplant. If the reference shows those defects, fix them
   there:
   - a negative integer's `.name`;
   - repr, type, isinstance, id, `!r` and `%` of a member, and identity tests.

   Tests: test_enum_literals (the `.name` cases), test_d98_enum_identity.
3. Bring 4b96b21's tests: test_enum_by_type (a changed definition refused at load), test_audit_runtime_b2,
   test_diag_scale_int_items.

**After the vectors transplant:**
1. 030de71 R1, with test_vector_list_mutators and vector_live.py.
2. 1eef6ff R2, with test_integer_vector_writes.
3. c51c847 R3, with test_float_vector_writes.
4. 95e73b6 R4, with test_vector_conditionals.
5. a7bcf56 R5, with test_vector_placeholders.
6. 7b5c27c, with test_vector_loop_renames.
7. e79b132, with test_vector_reference_reads.
8. 131cd37, with test_integer_vector_updates.
9. e82916d, with test_float_vector_of_integers.
10. The test of bd11687, test_reduction_in_vector_subscript (the fix is already in the bridge's reductions).
11. Optional: the tuple experiment 3035e52 + 0a2d914 + 29340c9, with test_tuple_vectors.

**Tables (the code is already in the bridge; only tests are missing):** test_prepared_tables (rows 21-24),
test_table_runtime_contract (row 23), test_forms_view_keyerror (row 48), adapted to the bridge's `_tables`. Row 53
waits for the recorder.

**Check:** 128d4b4 against r4/p1-math.

## Simplifications to re-apply later as deletions

1. Vectors, after e79b132: remove `vector_ref_reads.py`, its provider helper and the `native_vector_ref_reads` key
   (714b127, ec9ef8b, 89f7ebf, b40a095).
2. Vectors: remove the concatenation spelling (1c3db1e).
3. Vectors: F64Sum shared with the reductions' state (c410929, onto `_reductions.pxd`).
4. Tables: efae68a, 2d51e1b and da8c6ce, where the bridge's `frame_reads` still has the code they cut.
5. Labels: the helper-marker comment of e0de350, and drop any `helpers` reference the reference's modules carry.
6. Not needed on line A: rows 5, 19, 20, 26, 44 and 46, already absent there or already done in the bridge.
