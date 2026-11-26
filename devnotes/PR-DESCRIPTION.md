# Faster compiled models with checked values

This change makes the compiled package of an unmodified modelx model run several times faster than the
current compiler's output. On the SEC models, a policy takes 4.4 to 6.2 times less time and up to 4.8
times less peak memory. Every value equals live modelx. Models need no change. The lifelib SEC models
build from their released `sample_sec.py` and `spec_sec.py`. In a census of 95 lifelib models, no model
that the current compiler runs fails on this one. Nine models that the current compiler cannot run now
build and match live modelx.

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
   The runtime helpers in `_mx_sys.pxd` arrive whole with this first user. The later mechanisms (typed
   entries, powers, math calls, integer conversions and held Space results) use them too.
3. **Dense numeric caches.** A Cell keyed by integers gets a dense native array over the range declared
   in `cells_param_size`, or over the range the sample run reached. Wider keys use an open-addressing
   table. The two cache modules ship as `.pyx` sources with the package.
4. **Typed formula entries.** Each Cell gets a typed entry. The entry checks its arguments against the
   cache's domain once, then calls a private formula entry on an exact receiver. Calls between compiled
   Cells use these entries directly.
   - A Cell with integer parameters that is called from Python returns a small per-Cell Python entry.
     The entry has the Cell's parameters, defaults, name and documentation. It converts each integer
     argument exactly to 64 bits and calls the compiled method at C level.
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
7. **Integer handling.** See the next section.
8. **Held items of shared Spaces.** A Space result read through a shared Space keeps its items held by
   the compiled model. They are not rebuilt on every read.
9. **Translation.** The transformer, builder and command line apply mechanisms 1–8. This includes fast
   construction of ItemSpace items, also when the Space has dynamic References. The tracer records the
   range of each argument and which calls raised. The spec gains `cells_param_size` bounds and
   `integer_checks`.

   This is one commit rather than one per mechanism. Every mechanism is emitted from the same class and
   entry generators in `transformer.py`, and the command line runs the passes as one pipeline. The
   earlier commits add each mechanism's module. This commit is where they start being used, so a
   reviewer reads each module together with its part of this commit.
10. **Tests.** Unit tests for each mechanism. Compiled tests build small models and compare them with
    live modelx, including one that runs the generated package with `modelx_cython` blocked from import.
11. **CI.** Unit, compiled, allocation and SEC lanes, described below.
12. **Documentation.** The compiled contract, the spec keys and the integer modes.

## Integer handling

The spec key `integer_checks` takes `auto` (the default), `checked` or `unchecked`.

- **`unchecked`** compiles C integer arithmetic as plain C, as the current compiler does. A model can
  then wrap silently when an intermediate value exceeds 64 bits, for example when a run reaches
  magnitudes that the sample did not suggest. This is the same as the current compiler.
- **`checked`** compiles C integer arithmetic with Cython's overflow checking. An overflow raises
  `OverflowError` naming the Cell. Checked models are markedly slower, because every integer operation
  is tested.
- **`auto`** picks `unchecked` when the largest integer seen in the sample is below 2**31, and
  `checked` otherwise.

In every mode, integers that cross between Python and compiled code are checked exactly. This covers
arguments, results, ItemSpace keys and References typed as C integers. A float, a NaN or an int beyond
64 bits raises `TypeError` or `OverflowError` naming the Cell or Reference. Nothing is truncated.

## What a model author can rely on

- **Values.** The compiled package equals live modelx within a relative tolerance of 1e-10 and an
  absolute tolerance of 1e-7. A raise must match the same raise. Where the compiler cannot prove an
  operation safe, the build stops or the call raises at a named Cell. The one exception is integer
  arithmetic in unchecked mode, which can wrap as described above. Use `checked` for models whose
  integers can grow beyond what their sample shows.
- **Ranges.** A call outside a Cell's declared or sampled range raises `IndexError`. The message names
  the Cell, each argument's range, and the spec key `cells_param_size`, which widens the range.
- **No run-time dependency.** The generated package imports nothing from `modelx_cython` at run time.
- **One declaration.** The only thing an author may need to declare is `cells_param_size`, when a run
  calls a Cell with arguments beyond those the sample reached. Everything else is chosen from the types
  the sample produced and from the declared or sampled ranges.

## Behaviour that differs from the current compiler

1. In checked mode, integer arithmetic beyond 64 bits raises `OverflowError` naming the Cell. The
   current compiler always wraps. The default `auto` selects checked mode for samples whose integers
   reach 2**31.
2. A Cell called outside its declared or sampled range raises `IndexError` naming the Cell, the ranges
   and `cells_param_size`. The current compiler's dense caches raise a bare
   `IndexError("array index out of range")`, and its dictionary caches accept any argument. The range
   check now applies to every cache, so a value is never read from storage that was not sized for it.
3. A float, a NaN or an int beyond 64 bits passed where an integer was traced (an argument, an ItemSpace
   key, a Reference write) raises naming it. The current compiler truncated a float written to an
   integer Reference.
4. A variable left unbound by a loop that never ran raises `UnboundLocalError`, as live modelx does. The
   current compiler returned 0, because a C local has no unbound state. Such a local now stays a Python
   object.
5. A real power uses C `pow` with a one-entry memo (`_mx_memo_pow`), so the generated source text of
   powers changes.
6. The declared Python floor is 3.10. The current package declares 3.8 or later. CI covers 3.10 to
   3.14t, upstream lifelib requires 3.10, and nothing here is written or tested for lower versions.

### Upstream tests changed, and why

| test | change | reason |
|---|---|---|
| `tests/samples/various_types/assert_cy.py`, `assert_cy_no_sample.py` | calls at argument 11 and (3, 4) now expect `IndexError` | difference 2: these arguments lie outside the sampled range of a dictionary cache |
| `test_samples.py::test_fractional_power` | the assertions on the integer power's signature and source follow the new forms | differences 1 and 5 |
| `test_samples.py::test_locked_spaces` | classes are extracted with `ast` instead of splitting the text on `class` | the generated module has more classes, and a plain text split no longer isolates one class |
| `test_uslib.py::test_mx2cy_with_uslib` | the `_mx_pow(...)` and `**` source assertions are replaced by assertions on the memoized power forms | difference 5 |
| `test_powers.py` | the `_rewrite` helper takes the new keyword arguments and more Cells and References | the rewrite now uses the kinds of locals, Cells and References |
| `test_tracer.py` | `compiled_codes` takes a source argument; a section comment was removed | so the new raised-call and range tests can reuse the helper; no assertion was removed |
| `test_builder.py`, `test_cli.py`, `test_config.py`, `test_parser.py` | additions only | new spec keys and diagnostics |

No upstream test was deleted.

## Supported Pythons and compilers

CPython 3.10, 3.12 and 3.14, plus free-threaded 3.13t and 3.14t, with Cython 3.3.1. The lanes are Linux
with GCC on x86-64, Windows with MSVC 14.52 on x64, and macOS with Clang on arm64. Each lane runs four
suites:

- **unit:** the unit tests.
- **compiled:** the compiled tests and the integration tests.
- **allocation:** the unwritten-value tests, including two threads copying a locked Space while it fills.
- **sec:** Term_US_SEC, MYGA_US_SEC and SPIA_US_SEC built with `mx2cy` and compared with live modelx on
  every point.

Free-threaded lanes assert that the GIL stays disabled after the compiled package is imported. Modelx
and lifelib come from PyPI.

## Results

Every value below equals live modelx in every run.

### SEC models, per policy

Measured on an 8-core AMD Ryzen 7 5800H (WSL2) with free-threaded CPython 3.14.8. Both compilers ran
identical books. Figures are medians of five alternating rounds.

| model | µs per policy, upstream | µs per policy, candidate | speed-up | peak memory MiB, upstream | peak memory MiB, candidate |
|---|---|---|---|---|---|
| Term_US_SEC | 653.1 | 149.2 | 4.38x | 126.7 | 97.8 |
| MYGA_US_SEC | 1,871.8 | 300.6 | 6.23x | 256.0 | 104.0 |
| RILA_US_SEC | 1,993.3 | 444.9 | 4.48x | 493.9 | 103.9 |

### Parallel book: SEC DV01 example

1,050 policies on the same machine, six alternating rounds:

| workers | seconds | speed-up over one worker | peak memory MiB |
|---|---|---|---|
| 1 | 12.17 | 1.00x | 252 |
| 4 | 4.54 | 2.68x (2.31–2.87 across rounds) | 344 |
| 8 | 4.10 | 2.97x | 386 |

### Breadth: every lifelib model

Upstream lifelib main has 95 models, and each was driven through both compilers on one machine per
model.

- **No regressions.** No model that the current compiler runs fails on this one. 73 models build and
  equal live modelx on both.
- **SEC models.** The 12 SEC models run at a median 0.30 of the current compiler's instructions per
  policy, with a range of 0.22–0.52.
- **Regional models.** 47 regional models run at a median 0.99 of the current compiler's instructions
  per policy (0.92–1.05), with a median 0.87 of its memory per policy.
- **Models only this compiler runs.** Nine regional models hit two defects in the current compiler:
  - A float power inside `min` or `max` is typed complex by Cython 3, so the build stops: Term_JP_S,
    Term_KR_S and CI_KR_S.
  - A dense cache that starts at 0 raises `IndexError` for a negative argument before the formula runs:
    BU_DE_S, KLV_DE_S, Dep_FR_S, Euro_FR_S, IncomeTerm_JP_S and VA_US_S.
- **Models neither compiler runs.** Thirteen models fail identically on both, for reasons outside the
  compiler: workbook inputs, old model formats, a missing optional package, and one report column
  returned as integers.

### Known costs

- **LTC_JP_S, 5% more instructions per policy.** This is the only model more than 3% above the current
  compiler. A two-argument Cell gets a hashed cache where the current compiler allocates a dense
  948 × 948 array per policy, about 7 MB each. Peak memory falls from 12.5 GB to 9.7 GB.
- **Calls from Python with integer arguments.** A call from Python into a compiled Cell with integer
  parameters costs about 1,180 instructions, against 634 on the current compiler. The difference is the
  exact check of each integer argument. Calls between compiled Cells do not pay it. One top-level call
  driving a whole projection (BasicTerm_SE) uses 3.7% fewer instructions than on the current compiler.

### Compilation time

Translation runs `mx2cy` up to the generated sources. The C build is Cython plus the C compiler.
Translation does more analysis than before: ranges, local kinds, admission of typed entries and the
re-parse check. The C build compiles more code per Cell: typed entries and dense caches.

Same machine as the per-policy table:

| model | translate s, upstream | translate s, candidate | C build s, upstream | C build s, candidate | total, candidate / upstream |
|---|---|---|---|---|---|
| Term_US_SEC | 12.3 | 28.6 | 57.7 | 73.1 | 1.45x |
| MYGA_US_SEC | 40.9 | 88.1 | 130.3 | 154.0 | 1.41x |
| RILA_US_SEC | 60.9 | 127.9 | 160.9 | 190.8 | 1.44x |

Across the census, translation takes a median 1.55x the current compiler's time for regional models
and 1.70x for SEC models. The C build takes 1.29x–1.30x.

## Commits, line counts and CI (to be updated at freeze)

| # | commit | non-test lines + / − | test lines + / − |
|---|---|---|---|
| | | | |

| OS / compiler | Python | unit | compiled | allocation | sec |
|---|---|---|---|---|---|
| Linux / GCC | 3.10, 3.12, 3.14 | | | | |
| Linux / GCC | 3.13t, 3.14t | | | | |
| Windows / MSVC 14.52 | 3.10, 3.12, 3.14 | | | | |
| Windows / MSVC 14.52 | 3.13t, 3.14t | | | | |
| macOS / Clang arm64 | 3.10, 3.12, 3.14 | | | | |
| macOS / Clang arm64 | 3.13t, 3.14t | | | | |
