# Generator and reduction contracts

Development status: incomplete native coverage. The baseline is qualified local
compiler `299d8c10`; generator changes still need full regional native qualification.
This document distinguishes
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

For saved models, bind a module Reference such as `generators =
lifelib.typed_inputs` and write `generators.all_true(...)`. Module References
serialize as imports, preserving regional formulas-only folders; individual
function References can require embedded pickle data. The compiler resolves
registered module members through the same capability registry as direct
References and passes the current member callable to the provider. Module aliases
and ordinary Reference replacement remain checked. This source-saving contract
does not add runtime pickle support for module objects: exported Python and
compiled packages retain the existing module-object pickle limitation. A fresh
model can be constructed through its normal model class.

No additional reduction helper is required for builtin `all`, `any`, `min` and
`max`: proved builtin bindings share the generic primitive lowering and identity
provider. Integer extrema preserve integer results. Builtin `bool` around a
boolean reduction is removed only from the admitted fast branch; fallback keeps
the conversion, including replacement-helper truth effects and exceptions.
Ordinary `sum`, `next`, `sorted`, general tuple/set/dict construction and pandas
materialization still have open native-coverage rows.
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
- Clauses use builtin `range` with primitive proved signed-int64 arguments and
  any nonzero signed-int64 step, homogeneous literal tuples, or proved singleton bindings. Filters and payloads have proved
  primitive truth/numeric semantics.
- Literal tuples preserve order and exact homogeneous bool/int64/float/string
  values. Their C counter indexes an immutable tuple; string tokens pass as the
  original objects to compatible typed Cell arguments. Literal str/bytes/None
  arguments are admitted only for matching string/bytes/object parameters.
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
increment of a unit-step half-open int64 range is representable. Other steps
use a C progression state whose unsigned endpoint distance proves the next
signed value is representable. It stops before computing an out-of-range final
increment, including a step of `-2**63`. Zero-step errors are eager for the outer
range and lazy within the PEP479 boundary for inner ranges.

A single immediately preceding double-valued binding can feed composed reductions.
The original assignment executes eagerly in place; its value is passed into the
retained fallback rather than recomputed. Scope-proved double Cells and double
min/max expressions can establish that binding. Multi-argument double min/max
payloads use strict-comparison C helpers. Cython 3.2.9 extrema intrinsics can
reorder effectful operands; ordinary nested C calls retain the original operand
order and first-value ties. A captured min/max assignment records its current
callable and admission before evaluating operands; its fallback retains the
original captured object. The native loop unboxes the admitted capture once.
Unknown, reassigned-parameter and arbitrary-object bindings decline.

Existing bounded max/any mechanisms remain separately admitted. Their
presence does not imply arbitrary-precision bounds, arbitrary-capture or arbitrary-iterator
coverage. Object-capable Cython is useful compatibility but does not by itself
close a primitive native-coverage row.

## Shared namespace and provider metadata

`helper_members.module_helpers` captures registered functions of an exact module
without invoking module `__getattr__`. `helper_reference(func, refs)` is the one
resolver for `self.helper` and `self.namespace.helper`, shared with numeric
lowering. Records contain provider identifiers and validated capabilities, not
frozen callable objects. Unknown namespaces and unproved lexical receivers decline.

Two capability shapes are agreed with the pandas lane and validated in the sole
`helpers.CAPABILITIES` registry:

| Suffix | Internal field | Contract |
|---|---|---|
| `iteration` | `iteration_helper` | Four identifiers: admission operation, owning span class, count operation and signed-int64 key access operation |
| `float_numeric` | `float_numeric_helper` | Contextual conversion operation receiving current conversion, current helper, context and original arguments |

The integer key-span preserves original key order, empty input and owner lifetime.
Admission never invokes an unsupported callable; fallback must reuse captured
arguments and must not repeat table expressions. The original outer iterator's
errors remain eager. Unsupported key domains remain explicit coverage gaps.
Resources are disposable and private to the compiled provider package.

The reduction pass consumes the iteration capability for an outer keys call,
composed with inner signed-int64 ranges and singleton bindings. It evaluates the current
consumer, key producer and table argument once, in that order. Unsupported
receivers, callables or key domains build the original generator over the already
captured outer iterable; they never repeat the table expression. The native branch
uses a Py_ssize_t C counter and signed-int64 key access, without a Python iterator.
Nested key-span producers remain outside this admission until separately tested.

Contextual float fallback evaluates `conversion(helper(*args))` in source order.
It cannot simply coerce a replacement helper's result through a C-double ABI:
Python `float('2.5')`, for example, has different conversion semantics. Pandas owns
the contextual numeric/provider implementation; the generator lane owns registry,
namespace resolution and generator consumption. Registry reservation alone is not
evidence that either capability is fully implemented or qualified.

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
