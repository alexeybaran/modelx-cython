# Expression table helpers: implementation contracts

This feature branch descends from qualified compiler `299d8c10` and consumes
the shared module-helper resolver from generator checkpoint `6c4e7b4`.
It is not an all-model native qualification or a core promotion.

## Authored expressions and modes

Use the existing `modelx_helpers.immutable_inputs.Table` and `Record`, prepared explicitly from public
pandas input Cells. The same formulas/helpers run in live modelx and exported
Python. Native providers replace admitted interactions when compiled.

```python
# Data preparation Cell; original public mort_table() remains a DataFrame.
def prepared_mortality():
    return inputs.table(mort_table(), columns=('mort_rate',))

# Projection expression; the whole original key is evaluated in source order.
q = float(tables.lookup(data.prepared_mortality(),
                       (sex(), smoker(), age(t)), 'mort_rate'))
```

There is no formula interpreter, product-specific compiler configuration, new
table framework, or public helper per key arity. The public interaction remains
`lookup(table, key, column[, default])`. `previous` retains its ordered-key
contract. `keys(table)` exposes original input order rather than sorted order.

Whole-value replacement invalidates dependent Cells in live modelx. Mutations
inside a pandas value are not automatically tracked. Exported and compiled
models require explicit cleanup, including their normal retained input/cache
rules. Prepared native resources are disposable; financial Cell caches follow
the ordinary model contract.

## Contextual conversion and registry

The shared `helper_reference` resolver handles direct and module-member
References. Metadata records never freeze the current callable. Contextual
lowering receives proven builtin `float` scope information and visits one
formula scope. Child generators, comprehensions, lambdas and functions retain
their semantics until separately lowered by the generator lane. The reusable
`ContextualFloat` matcher runs after admitted generator lowering and before
general helper calls. It supports expressions in assignments, arithmetic,
branches and native reduction payloads, not just formula returns.

`float_numeric` is one validated provider identifier. The provider receives
current conversion, current helper, LookupContext and original arguments.
Its fallback is exactly `conversion(helper(*args))`; an implicit C-double cast
would incorrectly reject a replacement helper returning the string `'2.5'`.
The scalar ABI accepts a signed-int64 key, so a boolean key must not be sent
through it: that would change the key's type under callable replacement.

`keyed_float_numeric` is five validated private provider identifiers for key
arities zero through four. This is one optional registry capability. A wrapper
receives current conversion, current helper, context, table, individual key
components in source order, a tuple-key flag, column and optional default.
Each component is a fused int64, boolean or object argument. Object components
retain a strong reference throughout the call; exact Unicode admits native
hash/comparison while subclasses and other objects preserve fallback identity.
Scalar text uses arity one with the tuple flag false; `(text,)` uses true.
Arity zero represents the empty tuple only. Mixed bool/int min/max expressions
are declined when their original result type cannot be proven.

Wrappers pack primitive components on the C stack. They do not allocate query
tuples or box numeric components on the admitted path. Full keys are rebuilt
only for the original fallback or a missing-key exception. This preserves the
evaluation of later key components before the lookup can raise.

## Native resource and semantic boundaries

The keyed view belongs to the existing owned Table. It preserves original rows
and columns and uses native arrays/open addressing for fixed-shape text/int64/
boolean keys, including empty tables and empty tuple keys. Bool/int equality
matches the original Table contract. Unsupported key shapes decline admission.
This is not a general implementation of arbitrary pandas indexing semantics;
source adjustments must establish equivalence for the original expression's
key domain, coercions and unique axes.

An explicit float-context resource prepares conversions of exact immutable
float/int/bool/string values. Errors remain deferred until that value is read.
Other original scalars retain identity, including pandas NA/NaT singletons;
conversion of the selected scalar retains its original exception or user
conversion. Native row/column access does not delegate to Python lookup.
Preparation is accounted separately from repeated calculation.

Current callable identities guard every admitted entry. Replacements use the
original conversion/helper order, argument types and default behavior. Native
key search precedes column errors, matching Table.lookup. A context tracks
table/column identity; public concurrent contexts use the existing critical
section convention, and private access follows existing worker ownership.
Retired resources are released after the critical section. Table and context
pickle reducers omit these caches; another process rebuilds hash-seeded views.
Provider source/ABI is copied and built together with each compiled package.

CSV parsing, explicit owned input construction, and final public pandas object
construction are recorded boundaries, as in adjusted MYGA/VA. Public labels,
column order, index classes/names, dtypes, flags and missingness must match.
Pandas arithmetic/grouping inside reports is a calculation/transformation gap
until implemented and evidenced; it is not hidden in materialization.

## Evidence and remaining qualification

The all-57 scalar phase-1 campaign preserves 2,123 original/live/exported outputs,
with all exported output pickle bytes identical to their original counterparts.
It adjusts only 117 Cells/118 scalar reads. This establishes partial source and
financial parity, not complete operation coverage. A compiled UK probe retained
Python helper calls before shared namespace wiring; that failure remains evidence.
Raw module Reference whole-model pickle is also an existing baseline limitation
under investigation, separate from Table/context serialization.

Qualification requires operation/source/test/generated-Cython/C evidence,
active-path checks, full original-dataset financial comparison, lifecycle and
serialization checks, relevant untouched suites, then serial matched same-VM
preparation/calculation/report/lifecycle timing. Timing across VMs or Windows
and Linux cannot establish a regression ratio. No special model settings,
forced inlining or PGO are permitted.
