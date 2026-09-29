# Experimental source-preserved collect frame

The `--frame-collect` mode builds an exported model through the experimental
native frame generator. It is opt-in and leaves the established `mx2cy` route
unchanged. The export must include modelx's `_mx_formula_sources.json`; the
compiler verifies those formula hashes, runs the normal `--sample` trace, and
uses that trace to supply types to Cython. A direct `lookup.key(data.table(),
...)` return also carries its explicit table-key context. The compiler does
not copy types from the historical prototype.

For the currently supported subset:

```text
mx2cy Model_nomx --sample sample.py --no-spec --frame-collect \
  --frame-point-table model_point_table --frame-steps 480
```

This writes `Model_nomx_cy` and a retained `Model_nomx_cy_sources` directory.
It refuses to overwrite either path. Unsupported source or collection shapes
fail explicitly. At present the frame accepts a flat, independent-policy
Space with a full policy-table collection domain and direct zero-argument
numeric `sum` leaves. It does not yet compile `individual` outputs, nested
results, or multiple interacting Spaces, so the DV01/CS01 example is **not**
an integration pass at this stage. Its 100-policy compiled run should be used
only after those contracts are implemented and separately validated.

The frame uses Cython to compile the source-preserved Cell formulas. Its
selective real-power handling follows the established compiler's typed
fractional-power rule; unresolved and integer powers retain their original
route. Type context from a sample is a build assumption, so the resulting
artifact still requires correctness checks across the intended input domain.
