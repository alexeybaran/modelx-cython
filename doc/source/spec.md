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
integer parameters.  A cells whose parameters are all integers and whose
values are numeric scalars (not arrays) is backed by a fixed-size C array
indexed by its parameters, so the translator must know each parameter's
maximum size at translation time.

The value is a dict that maps a parameter name — or a tuple of parameter
names for cells with multiple parameters — to the array size, or tuple of
sizes.  A size of `n` means the parameter takes values in `range(n)`:

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
tuples with no spec entry, sizes are derived entirely from the observed
maximum values.  Calling a compiled cells with an index outside the
declared range raises {py:exc}`IndexError`.

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

## Experimental cache and collection options

Cells that need an associative cache use a Python dictionary by default.
For eligible scalar Cells, set `native_numeric_cache` in the Cell spec to try
the native key store. A Cell spec with `numeric_cache_capacity` also opts in unless
`native_numeric_cache` is false. Use the default for compiled models that
need to be pickled.

`dense_bounds_proofs` is an explicit top-level opt-in. The compiler still
uses checked cache entries and exact-receiver dispatch with the default.

Set `serial_collect` to `"reuse"` on an eligible leaf Space with one scalar
sum output to reuse a receiver. Set `parallel_collect_reuse` on the same Space
to reuse one receiver per chunk when `parallel_workers` exceeds one. Each
policy must be independent, its receiver and mutable cache must stay inside
collect, and Reference bindings must stay stable during the calculation.
The existing chunk partition and result merge order apply. Shapes outside
the admitted contract use the ordinary collection implementation.

## Registered helper providers

An exported callable Reference may declare `__modelx_cython_helper__` as a
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
