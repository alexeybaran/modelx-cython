# Tuple experiment (r4 w1-vectors)

Marker: the commits after this one are the experiment; the commits before it are the fixes and deletions.

Rule tried: a tuple of floats or integers BUILT AS A SEQUENCE (`tuple(<generator or list>)`, `(x,) * n`,
or those concatenated with other tuples, `(x,) + tuple(...)`) returned by a cell is a vector cell,
immutable for readers by construction; Python code gets an equal tuple. A literal display `(a, b)`,
a tuple used as a cell argument or a dictionary key, and a small record stay Python tuples.

Real shapes: lifelib r4/l1-lean-b, IP_UK_S, `claim_rate_vectors` and `sick_cohorts`, repeated small in
`modelx_cython/tests/compiled/test_tuple_vectors.py`.

Pass conditions: (a) those shapes compile as vectors and equal live; (b) records, argument tuples and key
tuples are untouched (generated code unchanged for them); (c) the experiment removes more lines than it
adds, counting the copy and ownership code a tuple makes unnecessary. Results: below, at the end.
