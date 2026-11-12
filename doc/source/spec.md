# Spec file reference

A *spec file* supplies translation parameters that cannot be observed from
the sample run, and lets you override inferred types.  By default `mx2cy`
reads `spec.py` in the current directory; use `--spec` to point to a
different file, or `--no-spec` to translate without one.

## File format

The file must contain a single Python **dict literal**.  It is read as
UTF-8 with {py:func}`ast.literal_eval`, so only literals are allowed —
no imports, expressions, or function calls.  Despite the `.py`
extension, the file is data, not an executable module.

```python
{"spaces":
     {"Projection":
          {"cells_param_size":
               {"t": 241},
           "cells":
               {"disc_factors":
                    {"return_type": "object"}}
           }
      }
 }
```

The dict mirrors the space tree of the model.  The top level corresponds to
the model itself, and each `"spaces"` key maps child space names to nested
dicts of the same shape:

```python
{"spaces":
     {"Parent":
          {"spaces":
               {"Child":            # Parent.Child
                    {"cells_param_size": {"t": 100}}}}}}
```

## Keys

### `spaces`

Maps child space names to the spec dicts of those spaces.  Spaces that are
not listed use no spec.

### `cells_param_size`

Declares the sizes of the per-cells cache arrays allocated for cells with
integer parameters. For an array-backed cell, the translator needs an
upper bound for each integer parameter at translation time.

The value is a dict that maps a parameter name — or a tuple of parameter
names for cells with multiple parameters — to an exclusive upper bound,
or a tuple of bounds. The lower bound is zero or the lowest negative
value observed for the cell and parameter:

```python
{"cells_param_size":
     {"t": 241,              # cells with parameter (t): t in range(241)
      ("i", "j"): (3, 6)}}   # cells with parameters (i, j)
```

The entry applies to every cells in the space with that exact parameter
tuple.

If the sample run passes an argument value equal to or larger than the
declared size, the size is automatically raised to the observed maximum
plus one, and the adjustment is logged at `INFO` level.  For parameter
tuples with no spec entry, the upper bound is the observed maximum plus
one. Calling an array-backed cell outside its accepted range raises
{py:exc}`IndexError`. To extend the negative range, include the required
negative indices in the sample.

### `cells`

Maps cells names to per-cells settings.  Currently one setting is defined:

`return_type`
: Overrides the return type inferred from the sample run.  A
  `return_type` setting takes effect only when the sample run collected
  type information for the cells; for a cells the sample never called
  it is ignored and the return type is `object` (only the
  `"memoryview"` setting logs a warning when ignored).  One of:

  * `"bool"`, `"int"`, `"float"`, `"str"` — declare the corresponding
    scalar type (`int` and `float` map to C `long long` and `double`).
  * `"object"` — fall back to a generic Python object.  Useful when the
    inferred type is too narrow, for example when a cells returns a NumPy
    array that consumers use as a whole array.
  * `"memoryview"` — force a typed-memoryview return type for a cells
    that returns a real-valued NumPy array.  This overrides the automatic
    usage analysis (see {ref}`memoryview-analysis`).  Translation fails
    with an error if the cells does not return a real-valued array; the
    setting is ignored with a warning if the sample run collected no type
    information for the cells.

```python
{"cells":
     {"disc_factors": {"return_type": "object"},
      "arr_whole": {"return_type": "memoryview"}}}
```

(deprecated-keys)=
## Deprecated keys

`cells_params`
: Older form of `cells_param_size`, wrapping each size in a nested dict
  under a `"size"` key.  Still accepted, but `cells_param_size` takes
  precedence when both are present:

  ```python
  {"cells_params":
       {"t": {"size": 241},
        ("i", "j"): {"size": (3, 6)}}}
  ```

(memoryview-analysis)=
## Return types of array-valued cells

A cells whose sampled return value is a real-valued NumPy array is normally
given a typed-memoryview return type (`const double[:]`, for example),
which makes element access from other compiled cells fast.  A memoryview,
however, supports only element access; whole-array operations (arithmetic,
NumPy methods, passing the array to functions) need the value as a regular
object.

`mx2cy` analyzes how each array-returning cells is used *inside the model*
and automatically falls back to an `object` return type when a use other
than element access is found; the decision is logged at `INFO` level.  Uses
in external scripts cannot be seen by this analysis.  If your own code
consumes an array-returning cells as a whole array and the compiled model
hands you a memoryview, set `"return_type": "object"` for that cells;
conversely, `"return_type": "memoryview"` forces the memoryview even when
the analysis would fall back.

## Automatic cache and collection policy (experimental)

`automatic_optimizations` defaults to `True`. The compiler uses the same
policy for every model; no per-Cell cache window, capacity or memo setting is
required. The existing representative sample supplies types and working-set
observations. It does not supply timing feedback to the native compiler, and
the build uses no PGO.

For eligible native scalar results and integer arguments, the compiler chooses
dense storage within a common 65,536-entry resource ceiling. An observed window
may reduce physical storage when its saved data/readiness bytes cover the added
pointer and cold dictionary. This never narrows the logical key domain:
keys in that domain outside the physical window retain exact spill storage. A nearly full
window stays full, preserving its simple range proofs.

Eligible native numeric associative keys/results use the native store; other
signatures retain Python dictionaries. An exclusively owned receiver gets a
power-of-two primary table sized for at most 50% observed load, capped at 4,096
entries, with growing exact overflow storage. The synchronized path remains
for shared receivers. Published cache entries survive normal model pickling.
The ceilings are common resource policies, not domain or correctness limits.

Range proofs, receiver propagation, one-axis checks, guarded arithmetic and
exact-operand power reuse apply under their existing safety checks.
`native_math` defaults to `true` and enables guarded builtin `round` lowering.
Set it to `false` for ordinary Cython compilation of rounding.
`native_cache_hits` defaults to `"auto"`: scalar hits and dense hits that already
have a proven index can read the existing cache fields directly. Every miss
keeps the original checked call and Cython error handling; public methods and
virtual overrides retain their dispatch behavior. No formula-length rule is
used to select private formula calls.

Eligible scalar sum collections use a fresh receiver by default. An explicit
Space contract `serial_collect="reuse"` additionally declares independent
policies, confined mutable state and stable Reference bindings. Under that
contract the automatic policy can reuse one receiver serially or per parallel
chunk and reuse private lookup resources. This semantic declaration cannot be
inferred from the timing sample. Custom lifecycle hooks and unsupported shapes
keep ordinary collection. Chunk totals merge as workers finish, so floating
point totals can differ by roundoff.

Set the Space option `parallel_collect_order="domain"` to merge reused chunks
in domain order. It requires parallel reuse. The default is `"completion"`.
Domain order is reproducible for a fixed worker count; changing chunk boundaries
can still change floating-point sums. Both choices propagate worker failures in
domain order. Ordinary collection always retains its existing ordered behavior.

Native collection uses the exporter's version 2 evaluator hook. The compiler
adds the native evaluator and its worker policy to the generated Space. The
exporter owns validation, dispatch and the result cache. Re-export collection
models with current modelx before translating older export formats.

Individual switches and legacy per-Cell options remain available for controlled
ablations and compatibility diagnostics. `automatic_optimizations=False`
disables automatic selection and the default dense miss binding on each
platform. Set `dense_miss_binding=True` to request that mechanism in an
otherwise manual configuration.
`native_cache_hits` also accepts `"none"`, `"scalar"`, `"dense"` and `"both"`;
non-`"none"` modes require `dense_miss_binding`. Checked dense hits can repeat
domain work, so the automatic mode admits the existing proven-index form.

## Registered helper providers

The shared safety and provider ABI rules are documented in
[Native specialization contracts](native_contracts.md).

An exported Python function Reference may declare `__modelx_cython_helper__` as a
`(provider, operation)` tuple. The provider names a Python package resource
with matching Cython `.pxd` and `.pyx` files. The operation receives the
current callable first. It must delegate to that callable when its
identity, argument types, or table layout fall outside the native path.

A provider owns the table representation and the rules for exact keys,
predecessors, column identifiers, grouped keys, and defaults. The Python
Reference remains the behavioral oracle for live modelx, Python export,
and compiled calls. The compiler checks operation names and only selects
numeric capabilities where the generated call has a justified signature.
It leaves user-defined key expressions and arbitrary generators on the
ordinary path when their types or bindings are uncertain.
