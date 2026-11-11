# Phase B — finite-domain category analysis

Version: `0.1.0+phase2r1b`

Phase B replaces the category-guard admission special cases with a bounded,
path-sensitive finite-domain evaluator. The acceptance case is `p34 Child_KR_S`,
but the implementation contains no product names or hard-coded cause domains.

## Semantic rule

Category demand remains callsite-driven and fail-closed. A tracked category value
is an exact finite set of `str`/`None` labels. Predicates are evaluated in a
three-valued domain:

- `TRUE` — the current finite environment proves the condition;
- `FALSE` — it proves the condition false;
- `UNKNOWN` — both paths remain feasible.

`UNKNOWN` is not automatically a failure. The evaluator separately marks whether
uncertainty came from an unsupported operation *on a tracked category*. Opaque
non-category predicates therefore branch conservatively, while expressions such as
an unsupported category method call still fail closed.

Supported finite semantics include:

- exact `str`/`None` values and aliases;
- `==`, `!=`, `is`, `is not`;
- `in` / `not in` over finite literal collections;
- `and`, `or`, `not` with Python short-circuit logic;
- `if` / `elif` and conditional expressions;
- simple assignments preserving exact finite values;
- `return`, `raise` and guard termination;
- bounded literal `for` loops for direct callsite category evidence.

## Implementation

`modelx_graph/finite_domain_category.py` is backend-neutral. It knows nothing about
modelx Cells, category integer codes, products or physical ABIs. Its main types are
`FiniteValue`, `TruthValue` and `PredicateResult`.

`modelx_graph/modelx_cython_category_abi.py` now uses this evaluator for category
path acceptance and for bounded direct-call evidence. Existing provider/table and
exhaustive-literal evidence remains because it answers a different question:
*which finite labels are candidates?* The new evaluator answers whether demanded
labels have valid reachable source paths.

The category proof also stops cleanly when a finite label is forwarded into a
parameter that modelx-cython traced as Python/object semantic and for which no
native category ABI exists. That is not an ABI hole: the detached Python helper
continues to carry the original Python label. Helper admissibility is checked by
the existing detached-P planner separately.

`modelx_graph/mixed_recursive_native.py` received one generic correctness fix
exposed by p34: names bound by `import`, `from ... import ...`, nested function and
class definitions are local Python bindings and are no longer misclassified as
modelx references.

## p34 behavior

The previous blocker was a valid non-terminal guard such as:

```python
if broad_def() and cause in ("cerebral", "cardiac"):
    ...
return rate
```

For `cause="burn"`, finite membership proves the right operand false, making the
whole `and` false even though `broad_def()` is opaque. For
`cause="cerebral"`, membership is true and the opaque predicate leaves both paths
feasible. Both cases can reach the valid return; neither requires a product rule.

Further downstream, p34 also passes category values from bounded literal loops.
The direct-call evidence walker now propagates those finite loop values generically.

## Validation

- source tests: **33/33 PASS**;
- p34: **PASS**, N0/N1/P = **38/30/19**;
- p34 max absolute error: **3.725290298461914e-09** live and detached;
- p21: **0.0** live/detached error;
- p30: **0.0** live/detached error;
- p38: **0.0** live/detached error;
- p52: **3.637978807091713e-12** live/detached error;
- full census: **58/71 PASS**;
- direct insurance: **56/65 PASS**;
- regional insurance (FR/JP/KR/UK/US): **47/47 PASS**;
- status changes vs Phase A: exactly one, p34 ERROR -> PASS.

Phase B does not change ranked-axis semantics, dynamic-Space invocation, memo ABIs
or the Phase-A semantic facade. Phase C should make ranked axis identity
first-class and use p10 as its first acceptance case.
