# Experimental native frame backend

This package is an in-progress transfer of the retained `prototype_0.8.2a1`
emitter and runtime into modelx-cython. It is kept off the accepted compiler
branch. The source snapshot is
`C:\mxw-01a0df78\h\references\prototype_0.8.2a1\src\mxcf_kernel`;
`modelx_frontend_native_mode.py` comes from the retained H06 frontend patch.

The generated kernel has one formula implementation and one cache layout per
Cell. `CompiledKernel.run_scalar` traverses policies with a reusable working
context. `CompiledKernel.fork_item` gives a directly accessed ItemSpace an
independent context with the same generated Cell entries. The two routes
therefore differ in **ownership**, not in formula or cache implementation.

The opt-in `mx2cy --frame-collect --no-spec` route can read preserved formula
sources from a standard exported package and build a separate frame package.
It does not replace the normal `mx2cy` translation path. Its experimental
facade accepts one flat policy Space and a `collect_outputs` declaration with
numeric `sum` leaves that directly call zero-argument Cells over the complete
policy-table domain. Other output shapes and domains remain outside this frame
route. The MYGA bridge still uses a toolkit-only scalar Cell and the retained
formula/type specification to check performance. Neither route is an accepted
compiler implementation or proof of general exported-package compatibility.

The validated DV01/CS01 demo is a concrete integration gap: it has separate
`Actual`, `Valuation`, and `Outer` Spaces, dynamic per-policy references,
time-indexed `sum` and `individual` results, and a nested `Actual.collect()`
dependency. It runs through the ordinary exported/Cython `collect()` path,
not this frame backend. The 105-policy DV01 timing must not be compared as a
matched frame-backend result. Bringing that demo onto the frame path requires
one source-preserved multi-Space graph with the existing `collect()` output
contract and nested dependency, followed by full result and CI validation.

Windows portability edits in this snapshot are limited to UTF-8 generated
source reads/writes and a package-local generated facade import. The toolkit
campaign records the generated and loaded binary identities, full MYGA
cash-flow validation, and a focused context-ownership check.
