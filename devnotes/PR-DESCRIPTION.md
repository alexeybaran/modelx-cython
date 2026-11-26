# Faster compiled models

This change makes the compiled package of an unmodified modelx model run faster than the current
compiler's output, with the values live modelx returns. Models need no change. The lifelib SEC models
build from their released `sample_sec.py` and `spec_sec.py`, and CI compares Term_US_SEC, MYGA_US_SEC
and SPIA_US_SEC with live modelx on every point (see Results).

## What the change does, mechanism by mechanism

1. **Shared syntax, scope and build helpers.** The translation passes used to rebuild expression
   contexts, scopes and definite assignment for every formula. These are now computed once per formula
   tree and shared between passes. The generated `setup.py` turns off floating-point contraction, so
   the C compiler cannot fuse operations into results that differ from Python's. A Cython error is
   reported against the Cell that caused it, and the Cell's source lines are located only when Cython
   reports one. The translated source is parsed again before it is built.
2. **Value arrays without zero fill.** Cell values live in arrays allocated without initialisation,
   plus a separate "written" flag. Building a model point no longer pays for clearing memory that is
   about to be overwritten. Results that a Space holds are kept in arrays released with their owner.
   The runtime helpers in `_mx_sys.pxd` arrive whole with this first user; the later mechanisms use
   them too.
3. **Dense numeric caches.** A Cell keyed by integers gets a dense native array over the range declared
   in `cells_param_size`, or over the range the sample run reached. Wider keys use an open-addressing
   table. The two cache modules ship as `.pyx` sources with the package.
4. **Typed formula entries.** Each Cell gets a typed entry. The entry checks its arguments against the
   cache's domain once, then calls a private formula entry on an exact receiver. Calls between compiled
   Cells use these entries directly.
   - Items of locked Spaces are read without a dictionary lookup.
5. **Powers.** When a power's operands are known to be real, it is computed with C `pow`, with a
   one-entry memo per Cell. Where Python's semantics depend on the order arguments are evaluated in,
   that order is kept.
6. **Math-module calls.** A Reference can hold the standard `math` module. A call through it to `log`,
   `exp`, `sqrt`, `erf` or `isinf` then runs the C library's function wherever that gives CPython's own
   result: `log` for x > 0, `sqrt` for x ≥ 0, `exp` for a finite result, and `erf` and `isinf` always.
   Everywhere else the Reference's own function runs. Domain errors, overflows, infinities, NaNs, ints
   beyond 64 bits and a rebound Reference therefore behave exactly as in live modelx. A user module that
   happens to be called `math` is not affected.
7. **Held items of shared Spaces.** A Space result read through a shared Space keeps its items held by
   the compiled model. They are not rebuilt on every read.
8. **Translation.** The transformer, builder and command line apply mechanisms 1–7. This includes fast
   construction of ItemSpace items, also when the Space has dynamic References. The tracer records the
   range of each argument and which calls raised. The spec gains `cells_param_size` bounds.

   This is one commit rather than one per mechanism. Every mechanism is emitted from the same class and
   entry generators in `transformer.py`, and the command line runs the passes as one pipeline. The
   earlier commits add each mechanism's module. This commit is where they start being used, so a
   reviewer reads each module together with its part of this commit.
9. **Tests.** Unit tests for each mechanism. Compiled tests build small models and compare them with
   live modelx, including one that runs the generated package with `modelx_cython` blocked from import.
10. **CI.** Unit, compiled, allocation and SEC lanes, described below.
11. **Documentation.** The compiled contract and the spec keys.

## Integers

Integers are compiled as upstream compiles them: a value traced as an int is a C `long long`, with
Cython's own arithmetic and conversions. A Cell with integer parameters is called from Python as on
upstream, so a fractional float passed for an integer parameter is truncated, as upstream does.

Past 64 bits, `+`, `-`, `*`, unary minus, `<<` and `**` wrap silently, `LLONG_MIN // -1` raises
`OverflowError`, `LLONG_MIN % -1` terminates the process, and an integer beyond 64 bits passed from
Python raises `OverflowError`. This model shows the first case:

```python
# make.py: export a one-Cell model and its sample
import modelx as mx
m = mx.new_model('Wrap')
s = m.new_space('S')
s.x = 1                      # an integer Reference
s.new_cells('f', formula=lambda: x * 2)
m.export('Wrap_nomx')
open('sample.py', 'w').write('from Wrap_nomx import Wrap\nWrap.S.f()\n')
```

```console
$ python make.py && mx2cy Wrap_nomx
$ python -c "from Wrap_nomx_cy import Wrap; Wrap.S.x = 2**62; print(Wrap.S.f())"
-9223372036854775808
```

Live modelx returns 9223372036854775808 for the same call.

The one addition to upstream: a fractional float or a non-number written from Python to an integer
Reference raises `TypeError` naming the Reference. Upstream truncates it, so a Reference set to 0.5
stores 0. There is no range check.

## What a model author can rely on

- **Values.** The compiled package equals live modelx within a relative tolerance of 1e-10 and an
  absolute tolerance of 1e-7. A raise must match the same raise. Where the compiler cannot prove an
  operation safe, the build stops or the call raises at a named Cell. Integer arithmetic is the
  exception: as on upstream, it wraps past 64 bits (see Integers).
- **Ranges.** A call outside a Cell's declared or sampled range raises `IndexError`. The message names
  the Cell, each argument's range, and the spec key `cells_param_size`, which widens the range.
- **No run-time dependency.** The generated package imports nothing from `modelx_cython` at run time.
- **One declaration.** The only thing an author may need to declare is `cells_param_size`, when a run
  calls a Cell with arguments beyond those the sample reached. Everything else is chosen from the types
  the sample produced and from the declared or sampled ranges.

## Behaviour that differs from the current compiler

1. A Cell called outside its declared or sampled range raises `IndexError` naming the Cell, the ranges
   and `cells_param_size`. The current compiler's dense caches raise a bare
   `IndexError("array index out of range")`, and its dictionary caches accept any argument. The range
   check now applies to every cache, so a value is never read from storage that was not sized for it.
2. A variable left unbound by a loop that never ran raises `UnboundLocalError`, as live modelx does. The
   current compiler returned 0, because a C local has no unbound state. Such a local now stays a Python
   object.
3. A real power uses C `pow` with a one-entry memo (`_mx_memo_pow`), so the generated source text of
   powers changes.
4. A fractional float or a non-number written from Python to an integer Reference raises `TypeError`
   naming the Reference; the current compiler truncates it.
5. The declared Python floor is 3.10. The current package declares 3.8 or later. CI covers 3.10 to
   3.14t, upstream lifelib requires 3.10, and nothing here is written or tested for lower versions.

### Upstream tests changed, and why

| test | change | reason |
|---|---|---|
| `tests/samples/various_types/assert_cy.py`, `assert_cy_no_sample.py` | calls at argument 11 and (3, 4) now expect `IndexError` | difference 1: these arguments lie outside the sampled range of a dictionary cache |
| `test_samples.py::test_fractional_power` | the `int_pow` signature may carry `except? -1`; `2 ** -t` is matched anywhere in the method instead of as `return 2 ** -t`; a run check of `int_pow(0) == 1` and `int_pow(1) == .5` was added | a typed numeric Cell can propagate a raise from its formula, and the method's statements around the power changed; the run check keeps the values tested |
| `test_samples.py::test_locked_spaces` | classes are extracted with `ast` instead of splitting the text on `class` | the generated module has more classes, and a plain text split no longer isolates one class |
| `test_uslib.py::test_mx2cy_with_uslib` | assertions on literal source text are replaced by an AST match of the operands of `_mx_pow` and `_mx_memo_pow` calls | difference 3: the generated method places these calls behind an exact-receiver dispatch and may use the memoized power, so the spelling changed while the operands did not |
| `test_powers.py` | the `_rewrite` helper takes the new keyword arguments and more Cells and References | the rewrite now uses the kinds of locals, Cells and References |
| `test_tracer.py` | `compiled_codes` takes a source argument; a section comment was removed | so the new raised-call and range tests can reuse the helper; no assertion was removed |
| `test_builder.py`, `test_cli.py`, `test_config.py`, `test_parser.py` | additions only | new spec keys and diagnostics |

No upstream test was deleted.

## Supported Pythons and compilers

CI (`.github/workflows/python-package.yml`) runs CPython 3.10, 3.12 and 3.14, plus free-threaded 3.13t
and 3.14t, on the `ubuntu-latest`, `windows-latest` and `macos-latest` runners, with each runner's
default C compiler (GCC, MSVC and Clang). Cython is not pinned: pip installs the newest release that
satisfies the package's `Cython>=3.2.4`. Each lane runs four suites:

- **unit:** the unit tests.
- **compiled:** the compiled tests and the integration tests.
- **allocation:** the unwritten-value tests, including two threads copying a locked Space while it fills.
- **sec:** Term_US_SEC, MYGA_US_SEC and SPIA_US_SEC built with `mx2cy` and compared with live modelx on
  every point.

Free-threaded lanes assert that the GIL stays disabled after the compiled package is imported. Modelx
and lifelib come from PyPI.

## Results

**Values.** `python .github/scripts/sec_compare.py <work dir> <model> <product>` builds a released
lifelib SEC model with `mx2cy` and runs lifelib's own check against live modelx: every point swept,
every cached value and raise compared, at lifelib's relative tolerance of 1e-12. The CI `sec` lane runs
it for Term_US_SEC, MYGA_US_SEC and SPIA_US_SEC.

**Calls from Python.** At 1c3f228, measured as instructions per operation with loop overhead
included, upstream against this change:

| operation | upstream | this change |
|---|---|---|
| `p.rate(5)`, a cached call to a Cell with one integer parameter | 789 | 724 |
| a write from Python to an integer Reference | 731 | 798 |

The write costs more because of the type check described under Integers.

**Compilation time.** Translation does more analysis than before: ranges, local kinds, admission of
typed entries and the re-parse check. The C build compiles more code per Cell: typed entries and dense
caches. Both stages take longer than on the current compiler.

## Commits and line counts

To be updated at freeze.

## CI

To be updated at freeze.
