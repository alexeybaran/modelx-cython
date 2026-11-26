# r4/p3-inverse-adds: state

Branch cut from r4/p3-inverse a2df257. Additions are taken from the proposal side (4eca864, re-checked against
r4/p2-collect-7 58ff45d: the files touched here are unchanged there, except the Reference-write check, which is
taken from 58ff45d).

## Commits

| commit | addition |
|---|---|
| 3ee996b | native math lowering (native_math.py, `_mx_math_*` in `_mx_sys.pxd`, `is_math`); inert where the tree's helper path lowers the same calls first |
| 9463a2e, reverted by cb24e0d | per-Cell Python entry (withdrawn on the proposal line) |
| 8721581 | `_fast_item_call` keeps `_mx_apply_formula` (dynamic References) |
| 1c38727 | value_storage wiring (value arrays last, unwritten at allocation) |
| af32041 | shared_facts / kept_children |
| 01f6bef | parse_source |
| 616a04f | positions resolved only where a line is reported; cell_functions / cell_locations |
| 5d0cd32 | tracer per-code-object caches |
| f2632fa | rewrite_powers returns at once without a power |
| ea74809 | re-parse check compares the built tree first |
| eaeca92 | Reference-write check (`_mx_integral`, `__setattr__`), with test_python_integer_boundary |
| 9241355, reverted by e8c14e3 | native_math.py as the lowering path (held: it takes a reference per call; the helper path passes the Reference borrowed) |
| f9cd640 | helpers takes module_name and read_resource from native_sources |
| c168271 | `_ReturnTypeMerger` keeps a scalar type without rebuilding it (list and tuple keep the vector-shape merge) |
| 7e75eb8 | `_FirstRead` local kinds; expression_kinds reads them through get |

## Checks

- Environment: PyPI modelx 0.33.0. Attaching the modelx and lifelib forks was refused, so no fork text, no
  pr/collect modelx, no live comparison and no instruction counts. Ratchet and SEC figures: not measured.
- After each commit: unit suite failures equal those at c342db9 (22 failed, 7 errors, on this modelx). Compiled
  subset (dense, value storage, entries, native math, Python call entry, integer boundaries): no new failure;
  test_value_storage and test_python_integer_boundary pass after their commits and fail before.
- Pre-existing compiled failures: test_shared_dense_publish, test_dense_by_range (ceiling), test_uncached_entries,
  test_shared_item_reads (four threads), test_native_math_calls (2; the helper path lowers first),
  test_python_call_entry ('dynamic': needs a modelx that exports `_mx_apply_formula`).
- After eaeca92: test_native_integer_boundaries[reference] fails. That tree test expects Cython's truncation
  of a float written to an integer Reference, which the Reference-write check replaces with TypeError; the
  proposal deleted this test file in aa7853d.

## Not applied

- Dense range check in unsigned arithmetic (entry_pipeline `_unsigned_check`): this tree compiles with
  `overflowcheck=True` in checked mode, under which the unsigned difference trips Cython's check. After the
  integer removal.
- Power inside min/max typed complex: already present; the proposal's test_powers and test_ordered_arguments
  pass on this tree.
- native_sources.install in place of helpers.install: helpers.install also writes the build guard of object
  providers (provider manifests), which native_sources.install lacks; kept.
- Native math as the path in use: held until the proposal's path passes the Reference borrowed.
- maybe_unbound / `_unbound_locals`, isinstance folding: part of formula_passes.py, which replaces
  guard_integer_boundaries on the proposal line.
- shared_facts around the integer guard: left with the guard.
