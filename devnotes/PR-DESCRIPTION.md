# Faster compiled models with checked values

This change makes the compiled package of an unmodified modelx model run several times faster than the
current compiler's output, and it checks that the result can be trusted: the compiled package returns the
values live modelx returns, or it raises at a named Cell. Models need no change. The lifelib SEC models
(Term_US_SEC, MYGA_US_SEC, SPIA_US_SEC and the others) build from their released `sample_sec.py` and
`spec_sec.py`.

Non-test source under `modelx_cython` grows from 6,300 to 13,211 lines (`.py`, `.pyx`, `.pxd`).

## What the change does, mechanism by mechanism

Line counts are the non-test lines each commit adds or removes.

1. **Shared syntax, scope and build helpers** (+844 / −2). The translation passes used to rebuild
   expression contexts, scopes and definite assignment for every formula. These are now computed once per
   formula tree and shared between passes. The generated `setup.py` turns off floating-point contraction,
   so the C compiler cannot fuse operations into results that differ from Python's. A Cython error is
   reported against the Cell that caused it. The translated source is parsed again before it is built.
2. **Value arrays without zero fill** (+315 / 0). Cell values live in arrays allocated without
   initialisation, plus a separate "written" flag. Building a model point no longer pays for clearing
   memory that is about to be overwritten. Results that a Space holds are kept in arrays released with
   their owner.
3. **Dense numeric caches** (+848 / −3). A Cell keyed by integers gets a dense native array over the range
   declared in `cells_param_size`, or the range the sample run reached. Wider keys use an open-addressing
   table. The two cache modules ship as `.pyx` sources with the package.
4. **Typed formula entries** (+1,348 / 0). Each Cell gets a typed entry. The entry checks the arguments
   against the cache's domain once, then calls a private formula entry on an exact receiver. Items of
   locked Spaces are read without a dictionary lookup.
5. **Powers** (+529 / −33). When a power's operands are known to be real, it is computed with C `pow`,
   with a one-entry memo per Cell. An integer power of C integers goes through an exact helper that
   raises beyond 64 bits. Where Python's semantics depend on the order arguments are evaluated in, that
   order is kept.
6. **Integer safety by overflow flag** (+1,809 / −1). C integer `+`, `-`, `*` and unary minus return the
   wrapped result and set a flag. GCC and Clang use `__builtin_*_overflow`. MSVC uses unsigned
   arithmetic, `_mul128` on x64 and `__mulh` on ARM64. The formula tests the flag once before it returns,
   and before any loop or call that receives a flagged value. If the flag is set, it raises
   `OverflowError` naming the Cell.
   - A formula containing an operation whose C type cannot be determined keeps Cython's
     `overflowcheck` for that formula alone.
   - `//`, `%`, shifts and integer powers keep Python's results through exact helpers.
   - Arguments, results, ItemSpace keys and References typed as C integers cross a checked conversion.
     Nothing is truncated.
   - `integer_checks: "unchecked"` in the spec turns the checks off and generates plain C, as before.
7. **Held items of shared Spaces** (+195 / 0). A Space result read through a shared Space keeps its items
   held by the compiled model. They are not rebuilt on every read.
8. **Translation** (+1,319 / −255). The transformer, builder and command line apply mechanisms 1–7. This
   includes fast construction of ItemSpace items, also when the Space has dynamic References. The tracer
   records the range of each argument and which calls raised. The spec gains `cells_param_size` bounds
   and `integer_checks`.
9. **Tests** (+5,841 test lines). Unit tests for each mechanism. Compiled tests build small models and
   compare them with live modelx, including one that runs the generated package with `modelx_cython`
   blocked from import.
10. **CI** (+160 / −126). Unit, compiled, allocation and SEC lanes, described below.
11. **Documentation** (+959 / −39). The compiled contract, the spec keys and the integer checks.

## What a model author can rely on

- **Values.** The compiled package equals live modelx within a relative tolerance of 1e-10 and an
  absolute tolerance of 1e-7. A raise must match the same raise. Where the compiler cannot prove an
  operation safe, the build stops or the call raises at a named Cell. It never returns a different number.
- **Integers.** Integer arithmetic beyond 64 bits raises `OverflowError` naming the Cell. A float, a NaN
  or an int beyond 64 bits passed where an integer was traced raises `TypeError` or `OverflowError`
  naming the Cell or Reference.
- **Ranges.** A call outside a Cell's declared or sampled range raises `IndexError`. The message names
  the Cell, each argument's range, and the spec key `cells_param_size`, which widens the range.
- **No run-time dependency.** The generated package imports nothing from `modelx_cython` at run time.
- **One declaration.** The only thing an author may need to declare is `cells_param_size`, when a run
  calls a Cell with arguments beyond those the sample reached. Everything else is chosen from the types
  the sample produced and from the declared or sampled ranges.

## Behaviour that differs from the current compiler

1. Integer arithmetic beyond 64 bits raises `OverflowError` naming the Cell. The current compiler wraps
   silently, which gives a wrong value.
2. A Cell called outside its declared or sampled range raises `IndexError` naming the Cell, the ranges
   and `cells_param_size`. The current compiler's dense caches raise a bare
   `IndexError("array index out of range")`, and its dictionary caches accept any argument. The range
   check now applies to every cache, so a value is never read from storage that was not sized for it.
3. A float, a NaN or an int beyond 64 bits passed where an integer was traced (an argument, an ItemSpace
   key, a Reference write) raises naming it. The current compiler truncated a float written to an integer
   Reference.
4. A variable left unbound by a loop that never ran raises `UnboundLocalError`, as live modelx does. The
   current compiler returned 0, because a C local has no unbound state. Such a local now stays a Python
   object.
5. An integer power of C integers goes through a checked helper that raises beyond 64 bits. The current
   compiler used Cython's C power, which wraps. A real power uses C `pow` with a one-entry memo
   (`_mx_memo_pow`), so the generated source text of powers changes.

### Upstream tests changed, and why

| test | change | reason |
|---|---|---|
| `tests/samples/various_types/assert_cy.py`, `assert_cy_no_sample.py` | calls at argument 11 and (3, 4) now expect `IndexError` | difference 2: these arguments lie outside the sampled range of a dictionary cache |
| `test_samples.py::test_fractional_power` | the `int_pow` signature may carry `except? -1`; `return 2 ** -t` becomes `2 ** _mx_sys._mx_flag_neg(t, …)` | differences 1 and 5: the negation goes through the overflow flag, and a numeric Cell can propagate a raise |
| `test_samples.py::test_locked_spaces` | classes are extracted with `ast` instead of splitting the text on `class` | the generated module has more classes, and a plain text split no longer isolates one class |
| `test_uslib.py::test_mx2cy_with_uslib` | the `_mx_pow(...)` and `**` source assertions are replaced by assertions on the memoized and integer power forms | difference 5 |
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

| OS / compiler | Python | unit | compiled | allocation | sec |
|---|---|---|---|---|---|
| Linux / GCC | 3.10, 3.12, 3.14 | | | | |
| Linux / GCC | 3.13t, 3.14t | | | | |
| Windows / MSVC 14.52 | 3.10, 3.12, 3.14 | | | | |
| Windows / MSVC 14.52 | 3.13t, 3.14t | | | | |
| macOS / Clang arm64 | 3.10, 3.12, 3.14 | | | | |
| macOS / Clang arm64 | 3.13t, 3.14t | | | | |

## Translation time

Translation does more analysis than before: ranges, local kinds, admission of typed entries and the
re-parse check. The shared analysis of mechanism 1 computes contexts and scopes once per tree, and local
kinds are analysed only when a formula has a classified operand. The second table under Results compares
translation and build time of this change (candidate) against the current compiler (upstream) on the
same machine.

## Results

Instructions are counted with callgrind over the compiled entries of a policy run. Wall seconds are for
the book-total cell. Peak memory is the maximum resident set during that run.

| model | instructions, upstream | instructions, candidate | wall s, upstream | wall s, candidate | peak memory MB, upstream | peak memory MB, candidate |
|---|---|---|---|---|---|---|
| Term_US_SEC | | | | | | |
| MYGA_US_SEC | | | | | | |
| SPIA_US_SEC | | | | | | |

| model | translate s, upstream | translate s, candidate | build s, upstream | build s, candidate |
|---|---|---|---|---|
| Term_US_SEC | | | | |
| MYGA_US_SEC | | | | |
| SPIA_US_SEC | | | | |
