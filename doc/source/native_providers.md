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
      "bind": "bind_curve"
    }
  }
}
```

The `.pxd` declares `CurveView` and the Cython function
`bind_curve(object) -> CurveView`. The compiled provider publishes
`__modelx_cython_provider__ = (1, 'example.rates', 'rates-v1')`.
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
the result. An explicit `object` return annotation or an explicit `return_type`
setting leaves the Cell on its declared result path. A missing manifest leaves
the ordinary path in place. A view can declare `"admission": "declared_or_forwarded"`
to require an explicit owner annotation, a resolved forwarding Cell with the
same provider, ABI and semantic type, or a direct return from a registered
constructor whose `__modelx_cython_result_type__` matches the observed type
and provider. The default is `"observed"`. Set `"bind_cell": true` when the
binder takes the Cell's qualified name as a second argument for diagnostics.

## Reusing compiled provider binaries

Set `MODELX_CYTHON_PROVIDER_CACHE` to a build-cache directory to reuse a native
provider when another build has the same extension identity, source and
declaration dependencies, Python ABI, Cython version, compiler and flags. The
build copies the provider binary into the generated model package. The model
loads that local binary when the build cache is unavailable. Distinct qualified
extension names have distinct cache entries.
