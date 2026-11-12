# Native providers for Cell results

A Python class can declare a native representation for a Cell that returns its
instances. The Python class remains the semantic owner and works without Cython
or a compiled provider. Declare the protocol version, provider resource name,
provider ABI, and semantic type identity on the exact owner class.

```python
class Curve:
    __modelx_cython_type__ = (1, 'example.rates', 'rates-v1', 'example.Curve')
```

Put `rates.provider.json` beside `rates.pyx` and `rates.pxd` in the `example`
package. The manifest connects the owner to the provider's Cython types and
symbols. Provider symbol names are local to the package.

```json
{
  "protocol": 1,
  "provider": "example.rates",
  "abi": "rates-v1",
  "types": {
    "example.Curve": {
      "view_type": "CurveView",
      "bind": "bind_curve",
      "members": {
        "rate_at": {"signature": "int64_label_to_float64", "native": "read_rate"}
      }
    }
  }
}
```

The `.pxd` declares `bind_curve(object) -> CurveView` and
`read_rate(CurveView, long long) -> double` as Cython functions. The compiled
provider publishes `__modelx_cython_provider__ = (1, 'example.rates', 'rates-v1')`.
At model import, the generated package checks that declaration and a fingerprint
of the provider manifest, implementation and `.pxd` before loading native Cells.

The binder retains the owner for the lifetime of the view. For a local view
received through a forwarded Cell, it returns the same view. For a foreign
wrapper, it can obtain the Python owner through `__modelx_cython_owner__()`,
validate the exact owner type, and create a view local to the current model.
Providers should expose the owner through the same method for callers who need
the Python representation. The Python-facing view must also support the owner
methods a formula may call through its ordinary object path. The binder raises
a clear error when a typed Cell returns a value outside the declared domain.

The compiler uses a consistent exact owner type observed in the sample to bind
the result. An explicit `object` return annotation or `return_type` setting
keeps the Cell on the ordinary result path. A missing optional manifest also
keeps the ordinary path. The `int64_label_to_float64` member signature currently
lowers a direct member call on a resolvable typed Cell result when the label is
a signed 64-bit integer literal. Other arguments use the Python-facing method.

## Reusing compiled provider binaries

Set `MODELX_CYTHON_PROVIDER_CACHE` to a build-cache directory to reuse a native
provider when another build has the same extension identity, source and
declaration dependencies, Python ABI, Cython version, compiler and flags. The
build copies the provider binary into the generated model package. The model
loads that local binary when the build cache is unavailable. Distinct qualified
extension names have distinct cache entries.
