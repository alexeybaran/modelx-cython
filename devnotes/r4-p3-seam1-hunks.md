# Seam 1: proposal-side hunks still to reconcile after the collect swap

Branch `r4/p3-inverse` = `r4/p3-inverse-trial` 0b77986 (the compiler tree of 633c9be on the collect head 4eca864) plus:
- 5b4b4a3, which removes the handover collect side;
- c342db9, which wires the proposal's `collect.py` (verbatim, as 074fdee and 15a5255 do).

The table counts the hunks of `git diff -U0 c342db9 4eca864` in each file present in both trees (non-test). That is
the proposal side, 4eca864, against this branch. Each cell is "hunks (changed lines)".

How a hunk is classed:
- **(i) integer logic:** the hunk's lines name integer checks or flags (overflow, `_mx_flag`, `exact_int64`, signed,
  checked, guard, divisibility). Ignored, because it is being removed.
- **(ii) proposal-side addition to re-apply:** the hunk mostly adds lines (added lines at least twice the removed
  ones).
- **(iii) proposal-side removal or rewrite where a side must be chosen:** everything else.

The classing is by keywords and line counts, not by reading each hunk. The comment column names the functions the
hunks touch (from the hunk headers); it is not a reading of what each hunk does.

| file | (i) | (ii) | (iii) | where (from the hunk headers) |
|---|---|---|---|---|
| `integer_boundaries.py` | 47 (738) | 5 (20) | 18 (50) | ii: import libcst as cst, _overflow_handler, _GuardIntegers, _guard_divisions; iii: from .powers import KIND_FLOAT, OperandKind, kind_, _python_call_gate, __getattribute__, _GuardCalls |
| `transformer.py` | 6 (78) | 13 (52) | 62 (477) | ii: PXDGenerator, ModuleTransformer, _fast_item_call; iii: from libcst.metadata.scope_provider import Builtin, from modelx_cython import numeric_cache, from modelx_cython.consts import, _integer_reference_paths |
| `_mx_sys.pxd` | 9 (463) | 0 (0) | 3 (17) | ii: -; iii: from cpython.unicode cimport PyUnicode_Check, PyUn, cextern from *, cinline object _mx_generic_getattr |
| `admission.py` | 2 (6) | 0 (0) | 27 (370) | ii: -; iii: import re, describe, OVER_LIMIT_WARNING =, over_limit_warning |
| `tracer.py` | 8 (184) | 10 (40) | 27 (122) | ii: import inspect, _PLAIN_RETURNS = frozenset, _add_arg_vals, _ReturnTypeMerger; iii: import operator, import itertools, from contextlib import contextmanager, import numpy as np |
| `scopes.py` | 0 (0) | 4 (178) | 2 (4) | ii: from collections import Counter, has_scope_effects, CstFacts; iii: from libcst.metadata import ExpressionContext, Exp, lexical_nodes |
| `usage.py` | 0 (0) | 1 (6) | 15 (142) | ii: UsageVisitor; iii: to a cells poison every cells with a matching name, from dataclasses import dataclass, field, from modelx_cython.parser import ParentScopeAddin, CellsResolver |
| `ordered_arguments.py` | 4 (69) | 0 (0) | 10 (77) | ii: -; iii: import libcst as cst, _OrderedArguments, preserve_argument_order |
| `cli.py` | 3 (26) | 3 (6) | 13 (93) | ii: from modelx_cython.entry_pipeline import lower_ent, from modelx_cython.syntax import read_literal, main_handler; iii: import subprocess, from modelx_cython.usage import, from modelx_cython.integer_boundaries import guard, increment_backups |
| `builder.py` | 1 (15) | 1 (1) | 13 (106) | ii: CombinedRefInfo; iii: CombinedCellsInfo, CombinedRefInfo, ClassInfo |
| `inline_entries.py` | 0 (0) | 0 (0) | 20 (89) | ii: -; iii: _eligible, _Calls, _referenced_entries, add_inline_entries |
| `config.py` | 8 (75) | 0 (0) | 4 (7) | ii: -; iii: from typing import Union, Sequence, Mapping, CompilerPolicy, TransSpec |
| `build.py` | 0 (0) | 3 (47) | 7 (17) | ii: cell_locations, cythonize_model, create_setup; iii: from .syntax import statements, walk, cell_locations, create_setup |
| `reparse_check.py` | 0 (0) | 5 (33) | 6 (23) | ii: _operands, _children, _first_difference, check; iii: _children, _first_difference |
| `syntax.py` | 0 (0) | 4 (50) | 1 (2) | ii: from dataclasses import fields, import libcst as cst, local_assignment; iii: walk |
| `holds.py` | 0 (0) | 0 (0) | 8 (51) | ii: -; iii: reset with the item's per-policy caches., _KINDS = {, HoldTable, pxd_declarations |
| `automatic.py` | 0 (0) | 0 (0) | 4 (47) | ii: -; iii: max_dense_entries, dense_allowed, primary_capacity |
| `space_results.py` | 1 (12) | 2 (29) | 3 (5) | ii: pin_function, lower; iii: from .typedefs import primitive_kind, _traced_target, _HeldCalls |
| `powers.py` | 3 (7) | 1 (5) | 8 (16) | ii: rewrite_powers; iii: OperandKind, rewrite_powers, _LocalKind, _declared_c_kinds |
| `item_roots.py` | 0 (0) | 0 (0) | 4 (22) | ii: -; iii: release_item, functions |
| `entry_pipeline.py` | 1 (10) | 0 (0) | 3 (6) | ii: -; iii: _bounds, _emit |
| `held_arrays.py` | 0 (0) | 0 (0) | 3 (11) | ii: -; iii: every evaluation: an atomic increment and decremen, memoryview; the formula has parameters, lower |
| `numeric_cache.py` | 0 (0) | 0 (0) | 2 (9) | ii: -; iii: import libcst as cst, exclusively_owned |
| `parser.py` | 0 (0) | 2 (4) | 0 (0) | ii: import libcst.matchers as m, ModuleVisitor; iii: - |
| `_numeric_cache.pxd` | 0 (0) | 0 (0) | 1 (2) | ii: -; iii: cunion Bits |
| `cell_entries.py` | 0 (0) | 0 (0) | 1 (2) | ii: -; iii: EntryModule |

**Totals over 26 files:** (i) 93 hunks (1,683 lines), (ii) 54 hunks (471 lines), (iii) 265 hunks (1,767 lines).

Files on one side only are not counted. They are:
- proposal only: `native_sources.py`, `native_math.py`, `value_storage.py` (present, unwired);
- handover only: everything 633c9be has beyond the proposal, the bridge layers included.

Collect after c342db9: the proposal's `test_collect.py` gives values equal to the export and to live, bit for bit, at
one and four workers. Its bound on the items left after a four-worker collect fails, 7 against at most 6, on this
tree's `item_roots`. Direct mode is the subject of seam 1.

## Checks after each commit (direct mode, fork text 04d69db, its spec)

The gate could not be used: attaching the benchmark toolkit was refused by a permission check. The figures below
come from a substitute measurement, so they are not the gate's numbers and are not comparable to the handover
references 833,521 and 23,171,472.

Setup:
- **Run:** one process per build, `PYTHONHASHSEED=0`.
- **What is counted:** callgrind collects only inside the compiled `Projection.net_cf` entry and the item
  construction (`--toggle-collect`).
- **Workload:** over the book, each policy is built and `net_cf(t)` is read for t in `range(proj_len())`.
- **Figure:** collected instructions ÷ policies. This includes the first read's table preparation, once per run.
- **Sample:** the shipped `sample.py` calls `collect_outputs`, which the 04d69db text does not define. Every build
  used one generic sample instead: every policy, every public zero-argument cell, `collect_outputs` where defined.
- **Live:** every `net_cf(t)` value of every policy is compared with live modelx `pr/collect` 1aa21f4, at rel 1e-10
  and abs 1e-7.

| tree | modelx | Term_UK_S (8 policies) | Dep_FR_S (11 policies) | values against live |
|---|---|---|---|---|
| 0b77986 (baseline) | handover 636877b | 3,696,426 | 24,293,844 | 0 mismatches (2,340 and 5,340 values) |
| 5b4b4a3 | pr/collect 1aa21f4 | 3,707,505 (+0.30%) | 24,295,905 (+0.01%) | 0 mismatches |
| 5b4b4a3 | handover 636877b | 3,693,965 (-0.07%) | - | 0 mismatches |
| c342db9 | pr/collect 1aa21f4 | 3,690,576 (-0.16%) | 24,291,555 (-0.01%) | 0 mismatches |

**A first method was discarded:** the whole process, four passes minus one, with `del Projection[k]` between
passes.
- On the handover export, a deleted item comes back with its caches filled. `_f_net_cf` ran 2,340 times in four
  passes, against 9,360 on the pr/collect export.
- That method showed +33%, an artefact of the export rather than of the compiler.
