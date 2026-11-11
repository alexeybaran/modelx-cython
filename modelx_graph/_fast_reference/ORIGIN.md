# Part-1 private fast-reference backend

This directory is a temporary, private copy/adaptation of source from the user-supplied artifact:

`modelx_semantic_compiler-0.4.0.dev0-py3-none-any.whl`

It exists only to make the v0.19 Part-1 backend experiment reproducible inside the clean-room handoff. The copy was modified to enforce single-thread serial execution: no `prange`, no OpenMP compile/link flags, and `threads != 1` is rejected.

The package is **not** the correctness frontend and is not intended to survive Part 2. `CanonicalExecutionPlan` remains authoritative. Once register-region, frozen-reference, scalar-sequence and ordered-reduction optimizations are implemented in the canonical backend, this adapter should be removed.
