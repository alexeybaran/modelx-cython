# Generator and reduction contracts

Development status: incomplete native coverage, pending the local qualified
compiler checkpoint and full regional qualification. This document distinguishes
implemented admission from the remaining design; it is not an acceptance receipt.

## Authoring interface

Keep ordinary Python generator/comprehension syntax. Start with the four existing
expression helpers, whose live/export implementations define their contracts:

| Expression | Contract | Actuarial example |
|---|---|---|
| `sum_f64(values)` | `sum(values, 0.0)`, including the interpreter's floating summation behavior | `sum_f64(net_cf(t) * disc(t) for t in range(proj_len()))` |
| `any_true(values)` | `any(values)`; consume through the first truthy item | `any_true(is_shortfall(t) for t in range(proj_len()))` |
| `all_true(values)` | `all(values)`; consume through the first false item | `all_true(abs(residual(t)) <= tolerance for t in range(proj_len()))` |
| `max_value(values)` | `max(values)`; first greatest original object; empty iterable raises | `max_value(av_pp(t) for t in range(proj_len()))` |

These are current one-positional-argument helpers. No `key`, `default` or `start`
extension is implied. Ordinary `sum(values)` retains its distinct integer start,
return types, large-integer and mixed-numeric behavior; replacing it with
`sum_f64` requires a demonstrated compatible model domain. In particular, a
sample of float results does not justify changing all legal integer inputs.

No additional helper is yet agreed with the pandas lane. Builtin `min`, `next`,
`sorted`, tuple/set/dict construction and pandas materialization retain their
Python contracts until corresponding generic native mechanisms are qualified.
Creating a separate helper for each business formula is excluded.

## Evaluation boundaries

1. Evaluate the consumer callable, then its arguments in source order. A
   generator expression evaluates its outermost iterator expression immediately.
   Its payload, filters and subsequent iterators execute only as consumed.
2. Preserve each filter's order, inner-loop placement, tuple unpacking and
   short-circuit behavior. An empty outer loop never evaluates inner bounds.
3. Generator variables have their own scope. They cannot overwrite a same-named
   outer variable or capture a generated identifier. Captured mutable bindings
   remain late-bound; moving a closure requires an explicit lexical proof.
4. A generator is a one-shot iterator. A replacement helper may retain it,
   partially consume it, consume it twice, or raise its own exception. Such a
   replacement receives the original lazy generator, not an eager list.
5. An exception thrown while evaluating the outer iterator is propagated
   directly. `StopIteration` escaping the generator body becomes
   `RuntimeError('generator raised StopIteration')`, with that original exception
   as cause and context. Other exceptions preserve identity and order.
6. Numeric loops must preserve empty/descending ranges, zero-step errors,
   arbitrary Python integers, index conversion and overflow behavior. Native
   machine-integer admission is a proof obligation for every arithmetic
   intermediate and terminating increment, not just sampled endpoints.

## Current composed primitive lowering

Registered `all_true`, `any_true`, `sum_f64` and `max_value` can use composed typed loops when all of
these conditions hold:

- The formula returns the reduction directly, has a safe lexical scope and a
  cold generator fallback, and has a native boolean or double result as appropriate.
- The receiver is exact and the current callable passes the provider's identity
  check. A subclass or replacement callable keeps the generator path.
- Clauses use builtin `range` with primitive proved integer arguments and a
  unit step. Filters and payloads have proved primitive truth semantics.
- Generator targets are distinct, hygienically renamed and separate from
  formula parameters. Unknown captures, unpacking, aliases and iterators decline.
- Integer arithmetic is not introduced merely because a callee's result was
  traced as primitive. Mixed int64/float comparisons decline when conversion
  could change Python comparison results. `abs` is native only for proved
  double values and the builtin binding.

The fast branch evaluates outer range arguments once before the generator-body
exception boundary. It evaluates inner bounds and filters on demand inside that
boundary. Empty identities are `True` for all, `False` for any, and float zero for
sum. Maximum stores the first value, compares later values strictly, and preserves
the first NaN and signed-zero tie behavior. Its empty-error boundary delegates to
the registered callable so the Python release's exact exception message is
preserved. Sum reuses the existing provider's compensated F64Sum state; no second
summation algorithm is introduced. The terminating
increment of a unit-step half-open int64 range is representable. General steps
retain compatibility until a separate checked progression mechanism is proven.

Existing singleton/captured sum and bounded max/any mechanisms remain separately admitted. Their
presence does not imply arbitrary-range, arbitrary-capture or arbitrary-iterator
coverage. Object-capable Cython is useful compatibility but does not by itself
close a primitive native-coverage row.

## Remaining regional mechanisms

The source census requires ordinary and float-start sums, all/any, min/max,
filtered extrema over index keys, a nested care-state reduction, tuple/set/sorted
materialization, a `next` over zipped adjacent ages, and list/dict comprehensions.
Most list comprehensions construct pandas reports, so ownership and boundary
contracts must be coordinated with the pandas lane before model edits.

Complete qualification requires a source-level operation row, any source
adjustment, exact semantic test, generated Cython/C evidence, loaded extension
identity, full matched financial outputs and same-VM performance evidence for
every regional model. Unresolved indirect calls and compatibility fallbacks stay
explicitly open. Windows acceptance and core promotion remain separate.
