# Faster compiled models with checked values

This change makes the compiled package of an unmodified modelx model run several times faster than the
current compiler's output. On the SEC models, a policy took 4.4 to 6.2 times less time and up to 4.8
times less peak memory, measured at 53104d2, before the integer overflow checks were removed (see
Results). Every value equals live modelx. Models need no change. The lifelib SEC models build from their
released `sample_sec.py` and `spec_sec.py`. In a census of 95 lifelib models, taken at 53104d2, no model
that the current compiler runs failed on this one. Nine models that the current compiler cannot run
built and matched live modelx.

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
     The entry has the Cell's parameters, defaults, name and documentation. An int, or a float that is a
     whole number, goes on to Cython's own conversion to a C `long long`; a fractional float or a
     non-number raises `TypeError` naming the Cell (see Integers). The entry then calls the compiled
     method at C level.
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
Cython's own arithmetic and conversions. Past 64 bits, the following was observed: `+`, `-`, `*`, unary
minus, `<<` and `**` wrap silently; `LLONG_MIN // -1` raises `OverflowError`; `LLONG_MIN % -1`
terminates the process; an integer beyond 64 bits passed from Python raises `OverflowError`.

One addition to upstream: a fractional float or a non-number passed from Python for an integer Cell
parameter, or written to an integer Reference, raises `TypeError` naming the Cell or the Reference.
Upstream truncates it: 1.5 is taken as 1, and a Reference set to 0.5 stores 0. There is no range check.

## What a model author can rely on

- **Values.** The compiled package equals live modelx within a relative tolerance of 1e-10 and an
  absolute tolerance of 1e-7. A raise must match the same raise. Where the compiler cannot prove an
  operation safe, the build stops or the call raises at a named Cell.
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
4. The declared Python floor is 3.10. The current package declares 3.8 or later. CI covers 3.10 to
   3.14t, upstream lifelib requires 3.10, and nothing here is written or tested for lower versions.

### Upstream tests changed, and why

| test | change | reason |
|---|---|---|
| `tests/samples/various_types/assert_cy.py`, `assert_cy_no_sample.py` | calls at argument 11 and (3, 4) now expect `IndexError` | difference 1: these arguments lie outside the sampled range of a dictionary cache |
| `test_samples.py::test_fractional_power` | the assertions on the integer power's signature and source follow the new forms | difference 3 |
| `test_samples.py::test_locked_spaces` | classes are extracted with `ast` instead of splitting the text on `class` | the generated module has more classes, and a plain text split no longer isolates one class |
| `test_uslib.py::test_mx2cy_with_uslib` | the `_mx_pow(...)` and `**` source assertions are replaced by assertions on the memoized power forms | difference 3 |
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

The independent benchmark (the SEC per-policy table and the compilation times), the DV01 parallel
book and the 95-model census were taken at 53104d2, before the integer overflow checks were removed
(aa7853d). Instructions per policy, single-threaded, from 53104d2 to f86f38d:

| model | 53104d2 | f86f38d |
|---|---|---|
| Term_US_SEC | 6.49M | 6.06–6.27M |
| MYGA_US_SEC | 7.90–8.05M | 7.73–7.74M |
| RILA_US_SEC | 6.44–6.50M | 6.32–6.33M |

### SEC models, per policy (at 53104d2)

Measured on an 8-core AMD Ryzen 7 5800H (WSL2) with free-threaded CPython 3.14.8. Both compilers ran
identical books. Figures are medians of five alternating rounds.

| model | µs per policy, upstream | µs per policy, candidate | speed-up | peak memory MiB, upstream | peak memory MiB, candidate |
|---|---|---|---|---|---|
| Term_US_SEC | 653.1 | 149.2 | 4.38x | 126.7 | 97.8 |
| MYGA_US_SEC | 1,871.8 | 300.6 | 6.23x | 256.0 | 104.0 |
| RILA_US_SEC | 1,993.3 | 444.9 | 4.48x | 493.9 | 103.9 |

### Parallel book: SEC DV01 example (at 53104d2)

1,050 policies on the same machine, six alternating rounds:

| workers | seconds | speed-up over one worker | peak memory MiB |
|---|---|---|---|
| 1 | 12.17 | 1.00x | 252 |
| 4 | 4.54 | 2.68x (2.31–2.87 across rounds) | 344 |
| 8 | 4.10 | 2.97x | 386 |

### Breadth: every lifelib model (at 53104d2)

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
- **Calls from Python with integer arguments.** At 53104d2 a call from Python into a compiled Cell with
  integer parameters cost about 1,180 instructions, against 634 on the current compiler. Re-measured at
  f86f38d (callgrind, CPython 3.13 with the GIL, Linux x86-64, a one-parameter Cell, 200,000 cached
  calls, loop overhead included in both): `p.rate(5)` costs 1,457 instructions against 892 on the current
  compiler; with the entry held in a local (`f = p.rate`, then `f(5)`), 858 against 847. The difference
  is the per-Cell entry returned on each attribute read, not the argument check. Calls between compiled
  Cells do not pay it. One top-level call driving a whole projection (BasicTerm_SE) used 3.7% fewer
  instructions than on the current compiler, at 53104d2.

### Compilation time

Translation runs `mx2cy` up to the generated sources. The C build is Cython plus the C compiler.
Translation does more analysis than before: ranges, local kinds, admission of typed entries and the
re-parse check. The C build compiles more code per Cell: typed entries and dense caches.

Same machine as the per-policy table, at 53104d2:

| model | translate s, upstream | translate s, candidate | C build s, upstream | C build s, candidate | total, candidate / upstream |
|---|---|---|---|---|---|
| Term_US_SEC | 12.3 | 28.6 | 57.7 | 73.1 | 1.45x |
| MYGA_US_SEC | 40.9 | 88.1 | 130.3 | 154.0 | 1.41x |
| RILA_US_SEC | 60.9 | 127.9 | 160.9 | 190.8 | 1.44x |

Across the census, translation takes a median 1.55x the current compiler's time for regional models
and 1.70x for SEC models. The C build takes 1.29x–1.30x.

## Commits and line counts (upstream d22426f to f86f38d)

| # | commit | subject | non-test lines + / − | test lines + / − |
|---|---|---|---|---|
| 1 | 5364738 | Add shared syntax, scope and build helpers | +882 / −2 | +0 / −0 |
| 2 | 39b41e4 | Add value arrays without zero fill and held arrays | +864 / −1 | +0 / −0 |
| 3 | 9c83429 | Add dense numeric caches sized from the declared or sampled range | +848 / −4 | +0 / −0 |
| 4 | 4047122 | Add typed formula entries checked once per call | +1,348 / −0 | +0 / −0 |
| 5 | 4c18687 | Compute real powers with C pow and keep argument order | +528 / −33 | +190 / −4 |
| 6 | c52cee7 | Add overflow-flagged integer arithmetic and checked conversions | +1,233 / −0 | +0 / −0 |
| 7 | 5b069d1 | Add held item results of shared Spaces | +195 / −0 | +0 / −0 |
| 8 | 19e9e10 | Translate Cells through typed entries, dense caches and checked integers | +1,299 / −254 | +281 / −37 |
| 9 | 9c604f3 | Add tests for entries, caches, integer checks and powers | +0 / −0 | +5,654 / −0 |
| 10 | a082f0a | Run unit, compiled and SEC model comparisons in CI | +159 / −126 | +0 / −0 |
| 11 | 5982ebe | Document the compiled contract, spec keys and integer checks | +185 / −40 | +0 / −0 |
| 12 | bf9c28f | Compile calls of the math module through a Reference to the C library | +146 / −0 | +107 / −0 |
| 13 | 83a8915 | Compare the math domain error's text with the running interpreter's | +0 / −0 | +5 / −1 |
| 14 | 53104d2 | Check a Python call's integer arguments in a per-Cell entry instead of a gate object | +42 / −60 | +89 / −0 |
| 15 | aa7853d | Remove the integer overflow checks and compile integers as upstream does | +290 / −1,602 | +25 / −1,261 |
| 16 | 3d8deae | Compare the dynamic ItemSpace call with what modelx exports | +0 / −0 | +5 / −1 |
| 17 | f86f38d | Raise TypeError for a fractional float passed from Python for a C integer | +62 / −10 | +65 / −0 |
| | d22426f..f86f38d | net, upstream to head | +6,407 / −458 | +5,158 / −41 |

Lines are counted with `git show --numstat` per commit and `git diff --numstat d22426f f86f38d` for the
net row. A path under a `tests` directory, under `integration_tests`, or a file named `test_*` counts as
a test line; every other path, CI configuration and documentation included, as a non-test line.

CI: run 45831453108 on f86f38d, pending.

| OS / compiler | Python | unit | compiled | allocation | sec |
|---|---|---|---|---|---|
| Linux / GCC | 3.10, 3.12, 3.14 | pending | pending | pending | pending |
| Linux / GCC | 3.13t, 3.14t | pending | pending | pending | pending |
| Windows / MSVC 14.52 | 3.10, 3.12, 3.14 | pending | pending | pending | pending |
| Windows / MSVC 14.52 | 3.13t, 3.14t | pending | pending | pending | pending |
| macOS / Clang arm64 | 3.10, 3.12, 3.14 | pending | pending | pending | pending |
| macOS / Clang arm64 | 3.13t, 3.14t | pending | pending | pending | pending |
