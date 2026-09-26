# Experimental lifelib helper support

This implementation belongs to `development/helpers`. It adds generic helper
support to the fork's `project-base` (`0f3e5649669fea3d42ca204aef7808c10d66d417`),
separately from `development/core-compiler`.

Enable the adapter with `{'experimental_helpers': True}` in a compiler spec.
Trace all required policies and the union of calculation/report entry points before
compiling. The adapter recognizes `lifelib.helpers.lookup` and
`lifelib.helpers.lookups.power`. It also recognizes their historical
`mxcf_kernel` module names so retained experiments remain usable.

The retained implementation provides:

- Enum/IntEnum object identity at cell boundaries and guarded member bindings.
- Observed scalar NumPy bool/float normalization.
- Full-table preparation into Curve, CurveTable, FieldTable and group-label helpers.
- Guards for helper, owning Space and table bindings; unsupported calls remain
  ordinary Python calls.
- Generated helper inventory and packaged Cython `.pxd` runtime declarations.

The preparation uses immutable snapshots, not selection by a particular policy or
scenario. In-place mutation of a prepared source table is not tracked; replacement
and normal cell caching semantics must be respected. This is a retained experimental
contract, not an assertion of production readiness.

## Provenance and integration

Imported from the helper-only delta between the historical project's
`upstream-pristine` (`a6e4b8c062b3b9cdfd3b97577274fc8d44432dce`) and
`feature/helpers` (`74c042659fa610ee11166403c278795c6b16cd3d`). That helper compiler
matches the exact Linux control (`c7fd7137d89403583c570502da5b9305e84603c3`).
The H09/prototype/previous compiler implementations are not imported.

The historical imported pristine source differs from this fork's project-base:
negative-argument cache exclusion/minimum tracking and local-exponent rewriting
are absent here. They are deliberately excluded from this helper delta. The tracer
patch was adapted only to remove context for that unrelated minimum tracking.

Integration adds the lifelib module names alongside historical names, ensures
runtime `.pxd` files are included in distributions, and exercises both namespaces
in the focused live/export/native helper test. Existing upstream tests are unchanged.

The corresponding lifelib helper commit
`fb8534fd8f9312bf844b026986c7a9dcbfd70747` updates `MYGA_US_S` itself, including its
helper modules and public Enums. Modelx uses the unchanged fork project-base
`4fa720f53776f20c5a07c6d5025f705dc3c9ce0e`. Pin the compiler and lifelib commits
together for experiments; do not combine this adjusted model with an unrelated
compiler control and treat the result as accepted performance evidence.

Run `python -m pytest modelx_cython/tests/test_h06_helpers.py` with a working native
C compiler. The focused test builds outside the source tree and checks live,
exported and compiled behavior, identity guards and table preparation ownership.
On Windows use a process-local MSVC environment and a short, new `--basetemp` path.

Import verification on native Windows (2026-09-26): **3 focused tests passed**,
including native builds for both helper namespaces, using Python 3.13.5,
Cython 3.2.4 and MSVC 14.44. The companion lifelib checks passed all 68 tests.

No benchmark is rerun for this import. Earlier Windows/Linux receipts retain their
original source identities and do not establish performance for this fork revision.
The full upstream CI and a compiled MYGA campaign are not claimed for this commit.
