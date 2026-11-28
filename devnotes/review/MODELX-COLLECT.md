# Add Collect to evaluate only the requested outputs across policies

`Space.set_collect(over=..., outputs=...)` declares a coordinate domain and a zero-argument Python function that returns selected individual values and sums. Use `space.collect(...)` to supply remaining Space arguments. Collected series support both `series(key)` and `series[key]`. Direct ItemSpace calls remain available on the same Space.

The live evaluator uses temporary policy items. Shared Cell and Reference reads attach to the Collect result's dependency node. Changes to input values invalidate that result and its consumers. Direct calls keep their usual caches and traces. Collect declarations survive a model save and read.

The exported Python package supplies the same API. Free-threaded execution with the GIL off can use up to `_mx_collect_workers` on eligible calls. Models with no Collect declaration retain the plain export runtime. A versioned hook lets a compiler provide native dispatch and mixed native and Python result fields. The pure Python runtime remains usable on its own.

The commits separate live evaluation, serialization, export, the API reference, and review version metadata. Review or merge dynamic Space Reference export separately first. Test the compiler integration against a distribution containing this API before its dependency is released. The review version `0.33.1.dev0` is a prerelease build for the paired installation. The maintainer can version the eventual upstream release independently.
