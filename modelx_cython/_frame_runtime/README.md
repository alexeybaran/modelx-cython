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

This package is not yet connected to the standard `mx2cy` exported-model
pipeline. Its experimental facade now provides `collect()` for a live modelx
declaration consisting of numeric `sum` leaves that directly call zero-argument
Cells over the complete policy-table domain. Other output shapes and domains
need the general collect implementation. The MYGA bridge still uses a
toolkit-only scalar Cell and the retained formula/type specification to check
performance. Neither route is an accepted compiler implementation or proof of
normal exported-package compatibility.

Windows portability edits in this snapshot are limited to UTF-8 generated
source reads/writes and a package-local generated facade import. The toolkit
campaign records the generated and loaded binary identities, full MYGA
cash-flow validation, and a focused context-ownership check.
