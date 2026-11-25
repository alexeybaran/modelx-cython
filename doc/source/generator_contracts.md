# Generators and reductions

Write standard Python generators and reductions in Cell formulas. Use `range`
for integer domains and `sum` for numerical totals. A start value such as
`0.0` expresses the desired type for an empty total.

The compiler can lower a floating-point sum to a native loop when the
iterable, yielded values, and evaluation order satisfy its analysis. For
other cases the generator keeps its ordinary Python behavior. The native
path checks the reduction callable before use, so rebinding a callable
continues to use that callable.

Use pandas DataFrames and Series for input tables. Ordinary pandas selection
expressions retain their public results.
