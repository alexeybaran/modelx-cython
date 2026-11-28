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

Declares exclusive upper bounds for cached integer coordinates. For an
array-backed cell, the translator also uses the bounds to size its storage.

The value is a dict that maps a parameter name — or a tuple of parameter
names for cells with multiple parameters — to an exclusive upper bound,
or a tuple of bounds. The lower bound is zero or the lowest negative
value observed for the cell and parameter:

```python
{"cells_param_size":
     {"t": 241,              # cells with parameter (t): t in range(241)
      ("i", "j"): (3, 6)}}   # cells with parameters (i, j)
```

A bound may also be a `(lower, stop)` pair, which declares the parameter's
range as `range(lower, stop)`: storage covers only that range (a parameter
such as a birth year needs none below `lower`), and an argument below `lower`
raises like one at or beyond `stop`:

```python
{"cells_param_size": {"year": (1900, 2100)}}
```

A tuple entry applies to each cached Cell in the space with that exact parameter
tuple. A single-name entry applies to Cells with that one parameter and also
bounds that parameter's axis of every other cached Cell in scope whose
parameter tuple has no tuple entry, so `{"t": 241}` covers `pv(t)` and
`claims(t, kind)` alike. In a mixed signature, only native signed integer
coordinates use the declared bounds. Other keys keep their own lookup semantics.

`cells_param_size` may also be given at the model level (the top of the spec)
and in any enclosing Space; a Space inherits those entries, and the nearest
declaration of a key wins. Declare a bound once, at the highest level that
covers it:

```python
{"cells_param_size": {"t": 1441},              # every Space of the model
 "spaces":
     {"Projection":
          {"cells_param_size": {"x": 121}}}}     # this Space and its children
```

A declared axis is allocated at its declared size. A Cell whose declared
array would exceed the dense storage ceiling (`max_dense_entries`) uses the
hash store instead, and the admission report says so. The report lists, per
Cell, the axes whose size is declared and where.

If the sample run passes an argument value equal to or larger than the
declared size, the size is automatically raised to the observed maximum
plus one, and the adjustment is logged at `INFO` level.  For parameter
tuples with no spec entry, the upper bound is the observed maximum plus
one; a sampled argument below a declared lower bound lowers it the same way.
Calling a cached Cell with an integer coordinate outside its accepted range
raises {py:exc}`IndexError` naming the Cell, each argument's accepted range and
the key `cells_param_size`. To extend the negative range, include the required
negative indices in the sample or declare a `(lower, stop)` pair.

### `cells`

Maps cells names to per-cells settings: `return_type`, below, and the cache
settings described under {ref}`cache-settings`.

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

### `collect_workers`

Model level only. The number of worker threads that the compiled `collect()` of a Space declared with
modelx's `Space.set_collect` uses by default: a positive integer, `1` if absent. See {ref}`collect`.

```python
{"collect_workers": 4}
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

(cache-settings)=
## Cache settings

The compiler chooses each Cell's cache from the sample's types and from the
declared or sampled range of its arguments. These settings are optional.

A cached Cell whose results are numeric scalars and whose arguments are
integers gets *dense* storage, an array inside the Space object indexed by
its arguments, when the range's rectangle has at most `max_dense_entries`
entries (65,536 by default). With automatic optimizations enabled and no
explicit limit, a cache with two or more parameters can also stay dense if
its rectangle has at most 16 times the entries filled by one receiver in the
sample. This affects storage selection. Keys in the logical range outside a
smaller physical window keep exact spill storage. Numeric keys and results
otherwise use a native hash store. Other signatures use Python dictionaries.

`max_dense_entries`
: Space or per-Cell setting: a positive integer limiting automatic dense
  storage. An explicit limit disables the density exception for that Cell.
  The build reports the selected storage and the cell's range.

`dense_window`
: Per-cells setting: half-open `(lower, stop)` physical windows, one per
  parameter in parameter order, for a dense cache. A window must fit inside
  the logical range and may exceed the automatic ceiling.

`native_numeric_cache`
: Per-cells setting: `true` or `false` overrides the automatic choice of the
  native hash store for numeric keys and results.

`numeric_cache_capacity`
: Per-cells setting: the initial capacity of a privately owned native hash
  store, a power of two from 2 to 4,096 (by default twice the entries of the
  declared or sampled range, at most 4,096; the table grows past it). Setting
  it also selects the native store unless `native_numeric_cache` is `false`.

`memoize_power`
: Per-cells setting: `true` keeps a one-entry memo of a real power's operands
  and result for a Cell of a privately owned Space (default: on with
  `automatic_optimizations`).

`native_locked_result_calls`
: Space setting for a locked Space: `true` lets a Cell call another Cell's
  result natively when that result is an ItemSpace of the exact class the
  sample saw; any other result takes the ordinary call.

## Integer arithmetic

Integers traced as `int` are C `long long` values, and their arithmetic is
Cython's, as on upstream: `+`, `-`, `*`, unary minus, `<<` and an integer
`**` wrap silently when the exact result leaves 64 bits; `//` and `%` keep
Python's rounding, `LLONG_MIN // -1` raises {py:exc}`OverflowError` and
`LLONG_MIN % -1` stops the process (a C division trap). An `int` beyond 64
bits passed from Python raises {py:exc}`OverflowError`, and a `float`
passed from Python where an integer was traced is truncated, as on
upstream. A value written from Python to an integer Reference must be an
integer or a `float` that is a whole number: a fractional `float` or a
non-number raises {py:exc}`TypeError` naming the Reference. Keep integer
values well inside 64 bits, or let the sample pass values that make the
Cell an object.

`infer_types`
: `true` (default) compiles generated model modules with Cython's C type
  inference for untyped locals; `false` keeps Cython's default.

## Compiler switches

`automatic_optimizations` (default `true`) turns on the policy above for
every model. The following switches default to it and exist for diagnosis
and comparison: `dense_bounds_proofs`, `dense_one_axis`,
`dense_miss_binding`, `propagate_exact_receiver` (requires
`dense_miss_binding`), and `native_cache_hits` (`"auto"` by default; also
`"none"`, `"scalar"`, `"dense"` or `"both"`; any but `"none"` requires
`dense_miss_binding`). They change how a cache hit is reached, never a value:
every miss keeps the checked call and Cython's error handling.
