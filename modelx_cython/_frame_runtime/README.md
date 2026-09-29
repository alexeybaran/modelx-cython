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
facade accepts one flat policy Space. Direct zero-argument numeric `sum` leaves
use the original scalar path. Other `collect_outputs` shapes use one generated
Cython object Cell per policy; its result is detached and folded immediately,
including time-indexed `sum` and `individual` sections. The policy table may
also provide object-valued references prepared for each policy. The MYGA bridge
still uses a toolkit-only scalar Cell and the retained
formula/type specification to check performance. Neither route is an accepted
compiler implementation or proof of general exported-package compatibility.

The validated DV01/CS01 demo remains a connected-Space integration gap: it has separate
`Actual`, `Valuation`, and `Outer` Spaces, dynamic per-policy references,
time-indexed `sum` and `individual` results, and a nested `Actual.collect()`
dependency. A toolkit-only seven-policy probe now compiles `Actual` through the
frame backend using explicitly prepared policy records and ordinary exported
Cython signatures. Its 36-month cash flows and annual states match the retained
DV01 fixture. This does not yet compile `Valuation` and `Outer` together or
prepare their references on each collect call. `Valuation` also has a policy
selector in its second required Space parameter, which the flat frontend does
not yet accept. The 105-policy DV01 timing remains an ordinary compiler result,
not a matched frame-backend result.

Windows portability edits in this snapshot are limited to UTF-8 generated
source reads/writes and a package-local generated facade import. The toolkit
campaign records the generated and loaded binary identities, full MYGA
cash-flow validation, and a focused context-ownership check.
