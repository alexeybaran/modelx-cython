# Faster compiled models

This change makes the compiled package of an unmodified modelx model run faster than the current
compiler's output, with the values live modelx returns. On the SEC models, a policy took 4.4 to 6.2
times less time than on the current compiler, with up to 4.8 times less peak memory (measured at
53104d2; see Results). Models need no change. The lifelib SEC models build from their released
`sample_sec.py` and `spec_sec.py`, and CI compares Term_US_SEC, MYGA_US_SEC and SPIA_US_SEC with live
modelx on every point.

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
5. **Powers.** Upstream's `powers.py` rewrites a power to C `pow` when its operands are provably real
   Cells, References or parameters. This change adds the following, each a separate proof or rewrite.
     Figures in this list come from one developer session and are unverified by a second
     measurement: callgrind instructions per policy on the SEC model's result Cell, single-threaded
     BLAS, gc off, two runs, noise about 2%.
   - **One-entry memo per Cell.** A real power of a Cell that owns its result goes through
     `_mx_memo_pow`, which keeps the last operands and result, so a repeated power costs a comparison.
     With the memo on rather than off, instructions fall 2.9% on Term_US_SEC, 2.7% on
     MYGA_US_SEC and 0.5% on RILA_US_SEC.
   - **Kinds of locals.** A local assigned only real values is real, so a power of it is rewritten too.
   - **Running totals in loops.** A local accumulated with real terms inside a loop keeps its real kind,
     so powers of the total are rewritten. With this on rather than off, DIA uses 5.5% fewer
     instructions, and boxed floats per policy fall from 19,021 to 14,620.
   - **Conditional expressions and negative exponents.** `a if c else b` is real when both branches
     are. An integral base raised to a negative integer is treated as fractional, so it takes the real
     path, not an integer power.
   - **Indexed typed arrays and mutually dependent locals.** One index per dimension of a Cell's
     typed-memoryview result gives an element of its C type. Locals that depend on each other are
     proved real together. Together, measured as 6c04fe4 against e67f27a (6c04fe4 has the same tree as
     503b07c), DIA falls from 9.909M to 8.755M instructions per policy (−11.6%), and boxed floats per
     policy fall from 14,620 to 4,953. The other models change by +0.1% (Term), +0.6% (MYGA) and −0.2%
     (RILA).
   - **Argument order.** Where Python's semantics depend on the order in which arguments are evaluated,
     that order is kept. Term_JP_S, Term_KR_S and CI_KR_S compile only with this: on upstream, a float
     power inside `min` or `max` is typed complex by Cython 3 and the build stops.
6. **Math-module calls.** A Reference can hold the standard `math` module. A call through it to `log`,
   `exp`, `sqrt`, `erf` or `isinf` then runs the C library's function wherever that gives CPython's own
   result: `log` for x > 0, `sqrt` for x ≥ 0, `exp` for a finite result, and `erf` and `isinf` always.
   The Reference is passed to the native call as a borrowed pointer and compared with the `math` module
   by pointer. A rebound Reference, and every case outside those domains, takes the fallback: the
   Reference's own function runs. Domain errors, overflows, infinities, NaNs and ints beyond 64 bits
   therefore behave exactly as in live modelx. A user module that happens to be called `math` is not
   affected. On RILA_US_SEC with four workers, a run fell from 0.151 s to 0.063 s. That was measured
   as 1c3f228 against 6c04fe4 on free-threaded CPython 3.14 with four virtual cores, each worker running
   its own model, best of five. It comes from one developer session; a second session's five
   alternating rounds are pending.
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
$ python make.py && mx2cy Wrap_nomx --no-spec
$ python -c "from Wrap_nomx_cy import Wrap; Wrap.S.x = 2**62; print(Wrap.S.f())"
-9223372036854775808
```

Live modelx returns 9223372036854775808 for the same call. This was run at 503b07c, where writing
0.5 to `x` raises `TypeError: S.x requires an integer, got 0.5`.

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

Every value below equals live modelx in every run.

The independent benchmark (per-policy times, peak memory and compile times) and the DV01 parallel book
were taken at 53104d2, a development commit that is not part of this series. It still had integer
overflow checks and a per-Cell Python entry; the final series has neither. The other tables were measured on code without them,
at the commit each one names.

### Values

`python .github/scripts/sec_compare.py <work dir> <model> <product>` builds a released lifelib SEC
model with `mx2cy`. It then runs lifelib's own check against live modelx: every point swept, and every
cached value and raise compared, at lifelib's relative tolerance of 1e-12. The CI `sec` lane runs it
for Term_US_SEC, MYGA_US_SEC and SPIA_US_SEC.

### SEC models, per policy (at 53104d2)

Machine: an 8-core AMD Ryzen 7 5800H under WSL2, free-threaded CPython 3.14.8. Method: both compilers
built the same models and ran identical books of policies. Rounds alternated between the two
compilers, and the figures are medians of five rounds. Time is wall-clock microseconds per policy on
the book-total cell. Peak memory is the process's maximum resident set.

| model | µs per policy, upstream | µs per policy, candidate | speed-up | peak memory MiB, upstream | peak memory MiB, candidate |
|---|---|---|---|---|---|
| Term_US_SEC | 653.1 | 149.2 | 4.38x | 126.7 | 97.8 |
| MYGA_US_SEC | 1,871.8 | 300.6 | 6.23x | 256.0 | 104.0 |
| RILA_US_SEC | 1,993.3 | 444.9 | 4.48x | 493.9 | 103.9 |

### The borrowed math pointer: 7edf07b against 1c3f228

Method: the `sec` gate tier. Per policy, a fresh item is built, evaluated cold and released, and
callgrind counts the instructions. Numeric libraries are single-threaded. Upstream modelx and lifelib
20d4dae. Compiler 7edf07b against 1c3f228, which is the same code without the borrowed math pointer.

| model | 1c3f228 | 7edf07b | ratio |
|---|---|---|---|
| Term_US_SEC | 998,446 | 998,424 | 1.000 |
| MYGA_US_SEC | 2,337,714 | 2,337,714 | 1.000 |
| RILA_US_SEC | 3,616,637 | 3,480,441 | 0.962 |

### Twelve SEC models against upstream (at 503b07c)

A run of all twelve SEC models on 503b07c against upstream d22426f, on CI runner machines, is in
progress.

| model | upstream Ir/policy | proposal Ir/policy | ratio | peak MB | translate s | build s |
|---|---|---|---|---|---|---|
| (run in progress) | | | | | | |

### Scaling with workers

This uses the same harness as the RILA math figure above, on an idle machine (load under 0.3) with
four virtual cores and 32 policies per worker. From one to four workers:
- DIA went from 0.711 s to 0.241 s (2.95x) at e67f27a, and reached 3.24x at 6c04fe4.
- RILA_US_SEC reached 3.08x.

On the same machine, a share-nothing Python loop loses 11% per thread at four threads. These figures
come from one developer session. An independent run on an eight-core machine is pending.

| model | 1 worker | 4 workers | 8 workers (eight-core machine) |
|---|---|---|---|
| DIA (6c04fe4) | 1.00x | 3.24x | (pending) |
| RILA_US_SEC | 1.00x | 3.08x | (pending) |

### Parallel book: SEC DV01 example (at 53104d2)

Same machine as the per-policy table. 1,050 policies from the lifelib DV01 example, six alternating
rounds. Each figure is the median wall time of the whole book at the given number of workers.

| workers | seconds | speed-up over one worker | peak memory MiB |
|---|---|---|---|
| 1 | 12.17 | 1.00x | 252 |
| 4 | 4.54 | 2.68x (2.31–2.87 across rounds) | 344 |
| 8 | 4.10 | 2.97x | 386 |

### Breadth: every lifelib model (at 503b07c)

A complete census on the final code is running. It covers all 95 driven models of upstream lifelib
20d4dae, compiling each with this branch's compiler at 503b07c and with upstream's at d22426f, both
on modelx fec516d. Both compilers run on the same machine for each shard, spread over twelve machines.

| | count / median |
|---|---|
| models that build and equal live modelx on both | (census in progress) |
| models that run only on this compiler | (census in progress) |
| models that fail identically on both | (census in progress) |
| SEC models, instructions per policy, this / upstream | (census in progress) |
| regional models, instructions per policy, this / upstream | (census in progress) |
| memory per policy, this / upstream | (census in progress) |
| translation time, this / upstream | (census in progress) |
| C build time, this / upstream | (census in progress) |

An earlier census, at 5982ebe (an earlier commit of this series, which still had integer overflow
checks), found no model that the current compiler runs failing on this one.

### Calls from Python (at 503b07c)

Method: a one-Space model with a Reference `x = 1` and a Cell `rate(t) = 0.01 * t + x`, sampled at
t = 0 to 9 and built with `mx2cy --no-spec` by each compiler. The script then runs `p.rate(5)` (a
cached call) or `p.x = 1` (a write to the integer Reference) in a Python loop. Each figure is callgrind's
total instructions for 200,000 iterations minus those for none, divided by 200,000, so loop overhead is
included. Repeat runs differ by about 2%. Run on Linux x86-64 with free-threaded CPython 3.14 and Cython 3.3.1.

| operation | upstream | this change |
|---|---|---|
| `p.rate(5)` | 906 | 821 |
| `p.x = 1` | 818 | 907 |

The write costs more because of the type check described under Integers.

### Compilation time (at 53104d2)

Translation runs `mx2cy` up to the generated sources. The C build is Cython plus the C compiler.
Translation does more analysis than before: ranges, local kinds, admission of typed entries and the
re-parse check. The C build compiles more code per Cell: typed entries and dense caches.

Same machine as the per-policy table, one build of each model with each compiler:

| model | translate s, upstream | translate s, candidate | C build s, upstream | C build s, candidate | total, candidate / upstream |
|---|---|---|---|---|---|
| Term_US_SEC | 12.3 | 28.6 | 57.7 | 73.1 | 1.45x |
| MYGA_US_SEC | 40.9 | 88.1 | 130.3 | 154.0 | 1.41x |
| RILA_US_SEC | 60.9 | 127.9 | 160.9 | 190.8 | 1.44x |

Translation and C build times across all models are reported with the census above.

## Commits and line counts (upstream d22426f to 503b07c, 14 commits)

"Compiler" is non-test code under `modelx_cython/`. "Other" is CI, documentation and packaging.
"Tests" is everything under `modelx_cython/tests/` and `integration_tests/`. Counts are from
`git show --numstat` per commit.

| commit | title | compiler + / − | other + / − | tests + / − |
|---|---|---|---|---|
| 19a063d | Add shared syntax, scope and build helpers | +882 / −2 | +0 / −0 | +0 / −0 |
| d2ff936 | Add value arrays without zero fill and held arrays | +589 / −1 | +0 / −0 | +0 / −0 |
| c8a519d | Add dense numeric caches sized from the declared or sampled range | +842 / −0 | +6 / −4 | +0 / −0 |
| 056cc5f | Add typed formula entries checked once per call | +1,345 / −0 | +0 / −0 | +0 / −0 |
| 534ee00 | Prove powers real from locals, memoise C pow, keep argument order | +444 / −26 | +0 / −0 | +192 / −3 |
| 0e1315a | Keep Python's semantics of locals under C typing and check integer Reference writes | +253 / −1 | +0 / −0 | +0 / −0 |
| 412eb13 | Add held item results of shared Spaces | +190 / −0 | +0 / −0 | +0 / −0 |
| 58f1eb0 | Translate Cells through typed entries and dense caches | +1,233 / −252 | +0 / −0 | +279 / −37 |
| d1fbd12 | Add tests for entries, caches and powers | +0 / −0 | +0 / −0 | +4,474 / −0 |
| e33041d | Run unit, compiled and SEC model comparisons in CI | +0 / −0 | +159 / −126 | +0 / −0 |
| e6049d2 | Document the compiled contract, spec keys and integer arithmetic | +1 / −1 | +187 / −39 | +0 / −0 |
| 065a6c1 | Compile calls of the math module through a Reference to the C library | +146 / −0 | +0 / −0 | +107 / −0 |
| 5eef6ff | Compare the math domain error's text with the running interpreter's | +0 / −0 | +0 / −0 | +5 / −1 |
| 503b07c | Pass the math Reference to native math calls as a borrowed pointer | +26 / −18 | +0 / −0 | +10 / −3 |
| total | 37 compiler files, 10 other, 70 test files | +5,932 / −282 | +352 / −169 | +5,063 / −40 |

All files: 117 changed, +11,347 / −491.

## CI

These are runs of `.github/workflows/python-package.yml`, read from each run's job list. Each lane runs
the unit, compiled, allocation and sec suites, and every lane that executed passed all four. "No
runner" means the job was never assigned a runner and was cancelled after waiting. Between these
commits the code changed only as the commit table shows, so a lane that passed on an earlier commit is
evidence for that commit, not for the final one.

| run | commit | executed and passed | no runner |
|---|---|---|---|
| 45831453108 | f86f38d | Linux 3.10, 3.12, 3.14, 3.13t, 3.14t; Windows 3.10, 3.12, 3.14, 3.13t, 3.14t; macOS 3.13t | macOS 3.10, 3.12, 3.14, 3.14t |
| 45833870838 (two attempts) | 1c3f228 | Linux and Windows, all five versions; macOS 3.10, 3.12, 3.14, 3.13t | macOS 3.14t |
| 45851243863 | 7edf07b | Linux and Windows, all five versions; macOS 3.10 | macOS 3.12, 3.14, 3.13t, 3.14t |
| 45869162145 | 503b07c (final) | (queued) | (queued) |

macOS / free-threaded 3.14 was not assigned a runner on any of these runs, so it is untested.
