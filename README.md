# modelx-cython extension (experimental)

A standalone add-on to unmodified official modelx 0.33.0 and modelx-cython
0.1.0. It exports modelx formulas, uses modelx-cython's frontend, and adds native
lowering for supported arithmetic, finite categories, table lookups and loops.
Unsupported formulas can use explicit Python fallbacks; some models remain
unsupported. This is a research compiler, not a production assurance of coverage.

## Install

Use Python 3.12 (tested), a C/C++ compiler and an isolated virtual environment.
Windows requires Visual Studio Build Tools with the C++ workload. Linux requires
a compatible C/C++ compiler; this release was validated on Windows only.

    python -m pip install -c tested_requirements.txt ./modelx_cython_extension-0.1.0+phase2r1b-py3-none-any.whl lifelib==0.17.1

Or install this source directory with `python -m pip install .`.
No upstream package files are edited. Lifelib data and any optional model-specific
dependencies are installed separately. There is no dependency on a research ZIP,
benchmark directory or precompiled machine-specific binary.

An optional modified modelx-cython build is also included in its own folder,
with source, patch, license files and an explanation. It reproduces the compiler
source behind the frozen modelx-cython baseline. The updated extension census
used the official wheel; the optional build has separate compatibility validation.
See `optional_modelx_cython/README.md` before choosing that build.

## Example

```python
import tempfile
from pathlib import Path
import lifelib
import modelx as mx
import numpy as np
from modelx_graph.valuation import build_valuation

folder = Path(tempfile.mkdtemp()) / "basiclife"
lifelib.create("basiclife", str(folder))
model = mx.read_model(folder / "BasicTerm_S")
keys = [1, 2]  # Choose the full requested portfolio for actual use.
expected = np.array([model.Projection[k].pv_net_cf() for k in keys])
valuation = build_valuation(model, model.Projection, "pv_net_cf", keys,
                            folder / "compiled")
model.close()
valuation.reset()
np.testing.assert_allclose(valuation.calculate(), expected, rtol=1e-9, atol=1e-7)
valuation.reset()                 # caches reset, snapshot retained
result = valuation.calculate()    # repeated full-portfolio calculation
```

`build_valuation` performs preparation, export, trace, native compilation and input
binding. `reset` starts a new calculation against the same snapshot. `calculate`
does not reset automatically. Measure both operations for repeated valuation
costs; compilation/binding are separate. Rebuild after changing source inputs.
Each build uses a distinct compiled module so resetting one build does not reset
another. A single valuation is not safe for concurrent calls. Shared-module
instance state and cheap rebinding remain future work.
Treat source input objects as immutable while a valuation is in use; this API
does not promise general copy-on-write ownership of arbitrary Python objects.

The output Cells must take no arguments. For an output with arguments, define a
readable zero-argument wrapper Cells before building. Results include only the
requested output dependency closure, not every formula in the model.

`sample_keys` optionally chooses trace keys separately from run keys. Sampling is
not a proof of unseen branches: always compare the complete run to modelx.
Ambiguous classes, unsupported expressions or incomplete domains may be rejected.
Use `valuation.metadata` and the generated manifests to inspect native/Python
coverage; do not infer native execution merely from successful compilation.

The package preserves the existing experimental compiler internals to avoid a
large unrelated refactor. The supported entry point is `modelx_graph.valuation`;
`modelx_graph.api.compile_exported` also accepts an existing export and sample.

## Phase 1 semantic compiler boundary

This build introduces a backend-neutral semantic formula IR boundary and an
explicit ordered semantic-pass pipeline in the fast modelx-cython path. Existing
normalizers and Cython lowering are unchanged semantically; the new boundary makes
semantic normalization the single input to later demand analysis/physical lowering.

The plan manifest now includes `semantic_formula_ir` entries with deterministic AST
fingerprints, ordered pass records, and normalized semantic operator IDs. This is
preparatory architecture for generic lookup/table IR in later phases; it does not
add MYGA-specific rules or claim additional native coverage by itself.


## Phase 2 generic as-of lookup lowering

This build adds a backend-neutral `AsofLookupIR` and lowers bounded grouped
"latest key <= query" table access through the existing prepared native step-lookup
runtime. Missing-domain behavior is preserved by native guards when static range
proof is incomplete. The implementation contains no MYGA/product-specific matching.

On the full seven-policy MYGA validation workload, the selected closure changes
from 52 N0 / 45 N1 / 3 P to 99 N0 / 0 N1 / 2 P: `scenario_rate` becomes native and
its 45 dependent N1 cells become N0. The two remaining P cells are the pandas
`result_cf` / `census_output` boundary. See `PHASE2_IMPLEMENTATION.md` for details.

## Phase 2 lookup physicalizer refactor

This build preserves Phase 2 semantics while removing duplicated StepLookup1D
physicalization.  Both the legacy explicit step-lookup syntax and first-class
`AsofLookupIR` now share one row-prefix planner, one finite column planner, one
prepared leaf builder and one finite dispatch emitter.  The existing
`NormalizedLookup1D` / `__step_lookup_1d__` runtime ABI is unchanged.

The refactor is intentionally behavior-preserving: MYGA's selected closure remains
99 N0 / 0 N1 / 2 P and `scenario_rate` remains native.  See
`LOOKUP_REFACTOR.md` in the source handoff for the structural validation.


## Semantic cleanup Phase A

See `PHASE_A_SEMANTIC_FACADE.md` for the behavior-preserving `ValueSpec` / `AxisDomain` / `CellSignature` / `LoweringPlan` facade.


## Semantic cleanup Phase B

Phase B adds a bounded path-sensitive finite-domain category analyzer. It tracks
exact finite `str`/`None` values through membership/equality tests, boolean
short-circuiting, conditionals and simple assignments. Opaque non-category
control explores both feasible paths, while unsupported operations on a tracked
category remain fail-closed. Direct category evidence also propagates through
bounded literal loops. This recovers Child_KR_S without product-specific rules and
raises the verified census to 58/71 (47/47 regional insurance). See
`PHASE_B_FINITE_DOMAIN_CATEGORY.md`.
