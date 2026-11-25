# Reusing compiled provider binaries

The compiler copies its native providers (the `.pyx`/`.pxd` pairs of
`modelx_cython.runtime`) into each generated model package and compiles them
with the model.

Set `MODELX_CYTHON_PROVIDER_CACHE` to a build-cache directory to reuse a native
provider when another build has the same extension identity, source and
declaration dependencies, Python ABI, Cython version, compiler and flags. The
build copies the provider binary into the generated model package. The model
loads that local binary when the build cache is unavailable. Distinct qualified
extension names have distinct cache entries.
