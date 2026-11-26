# Faster compiled models

This change reduces the cost of running compiled modelx models. On the twelve released lifelib SEC models, the current compiler used 2.20× to 4.94× as many instructions per projection as this change, with a median of 3.75×. The measured values matched live modelx across all 24 runs. The released `sample_sec.py` and `spec_sec.py` files build those models without edits, and the CI SEC suite compares Term_US_SEC, MYGA_US_SEC and SPIA_US_SEC with live modelx on every point.

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
   Reference's own function runs. This preserves the module's handling of domain errors, infinities
   and NaNs. A user module that happens to be called `math` continues to use its own functions.
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
  absolute tolerance of 1e-7. A different error type is accepted where both the live and compiled
  calls raise. Arithmetic past 64 bits follows upstream's integer behavior (see Integers).
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
5. The declared Python floor is 3.10. The current package declares 3.8 or later. The CI matrix targets 3.10 to
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

The CI matrix (`.github/workflows/python-package.yml`) targets CPython 3.10, 3.12 and 3.14, plus free-threaded 3.13t
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

The dedicated SEC instruction runs matched live modelx. The census reports all observed outcomes and inherited failures below.

### Values

`python .github/scripts/sec_compare.py <work dir> <model> <product>` builds a released lifelib SEC
model with `mx2cy`. It then runs lifelib's own check against live modelx: every point swept, and every
cached value and raise compared, at lifelib's relative tolerance of 1e-12. The CI `sec` lane runs it
for Term_US_SEC, MYGA_US_SEC and SPIA_US_SEC.

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

Method: this branch's compiler at 503b07c against upstream's at d22426f, on a cloud VM with an Intel
Xeon at 2.10 GHz and free-threaded Python 3.14.6. Both use modelx fec516d and lifelib 20d4dae. Each
model runs the given number of projections by the loop route, and callgrind counts the instructions per
projection. In all 24 runs the values equal live modelx (maximum absolute difference 0). Peak MB is the
peak resident memory of the measured run of that many projections.

| model | projections | upstream Ir/projection | this change Ir/projection | upstream / this | peak MB, upstream | peak MB, this change |
|---|---|---|---|---|---|---|
| Term | 204 | 4,051,402 | 998,454 | 4.06x | 193 | 119 |
| SPIA | 220 | 6,555,090 | 2,433,060 | 2.69x | 358 | 122 |
| WholeLife | 220 | 7,521,839 | 2,028,906 | 3.71x | 368 | 122 |
| VUL | 207 | 21,776,962 | 9,890,900 | 2.20x | 1,671 | 127 |
| FIA | 216 | 11,126,173 | 2,906,594 | 3.83x | 824 | 124 |
| VA | 207 | 17,745,439 | 4,674,314 | 3.80x | 1,207 | 130 |
| UL | 207 | 11,289,671 | 2,655,970 | 4.25x | 487 | 122 |
| IUL | 204 | 33,126,381 | 9,321,880 | 3.55x | 1,090 | 125 |
| ULSG | 207 | 11,725,569 | 3,451,453 | 3.40x | 650 | 123 |
| MYGA | 210 | 10,270,358 | 2,357,352 | 4.36x | 622 | 124 |
| RILA | 216 | 17,187,850 | 3,479,318 | 4.94x | 1,436 | 128 |
| DIA | 208 | 7,283,855 | 2,359,605 | 3.09x | 464 | 125 |

Across the twelve models, upstream uses 2.20x to 4.94x the instructions of this change
(median 3.75x).

### Paired wall time

No paired SEC wall-time comparison met the repeatability threshold. On a fixed 8,000-policy Term book, the candidate and upstream medians were 2.451 s and 6.736 s. The paired ratio ranged from 0.303 to 0.560 and the identical-build ratio ranged from 0.512 to 1.070, including the retry. These timings are diagnostic. The other eleven SEC products have no published same-book paired timing. [Local pilot and timing method](LOCAL-TIMING.md)

### DIA first-use preparation

On the eight-core Ryzen 7 5800H under WSL2, DIA_US_SEC scaled 1.73× from one to four workers on a 5,000-policy book at `503b07c`. The direct route includes the first projection's premium-schedule preparation. The model's Data Cells `premium_array`, `premium_rows` and `premium_last` prepare those tables serially in pandas. All runs matched live modelx.

### Broad census of released lifelib models

The census records outcomes for **all 95 driven models**. The catalog has another 10 entries without a modelx driver. Each comparison uses the measured compiler `503b07c6f13d646ca4a15b020a18aea7c677a205` against upstream `d22426fbac8637f84510902c2fa33b6466e5acb4`, with modelx `fec516d7e9b5813e7d40819a72731bd35fc2972f` and lifelib `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4` on both sides. Eleven shards used the same machine for each pair. The remaining shard uses an earlier upstream run on a matching CPU type. The final head adds two test files.

| Published outcome | Models |
|---|---:|
| Both compilers pass the live comparison | 73 |
| Candidate passes, baseline fails | 9 |
| Matching failure class on both | 13 |
| Baseline passes, candidate fails | 0 |

The nine candidate-only passes are BU_DE_S, Dep_FR_S, Euro_FR_S, IncomeTerm_JP_S, KLV_DE_S and VA_US_S, where baseline raises `IndexError: array index out of range`, and Term_JP_S, CI_KR_S and Term_KR_S, where baseline Cython rejects complex powers as unordered. The shared failures are four ExcelWorkbook translation errors, one ModuleIO translation error, five value-comparison failures involving missing data files, the WholeLife_JP_S report dtype mismatch, the BasicBonds live import of missing QuantLib, and the fastlife modelx export error. The cause of the missing-file failures remains unverified.

| Candidate ÷ baseline | Comparable models | Range | Median |
|---|---:|---:|---:|
| SEC instructions per policy | 12 | 0.2024–0.4542× | 0.2664× |
| Regional instructions per policy | 46 | 0.9201–1.0476× | 0.9898× |
| Legacy sweep instructions per evaluated value | 7 | 0.8072–1.0022× | 0.9639× |
| Translation time | 66 | 0.8098–1.9078× | 1.3804× |
| C build time | 66 | 1.0475–1.4052× | 1.2367× |
| Incremental memory per policy | 52 | 0.2125–0.9939× | 0.8428× |
| Peak RSS at equal book size | 26 | 0.2336–0.9902× | 0.8608× |

The 65 instruction ratios require matching CPU types, counted books and successful live comparisons. Cancer_KR_S passed live on both compilers and has no matched instruction ratio in the census because the counted books differed. Wall, memory and build-time ratios exclude the cross-host shard. Peak RSS ratios also require equal book sizes. The separate SEC table above uses its own measurement runs and sample sizes.

**LTC_JP_S** used 514,431,746.9 candidate instructions per policy versus 491,080,255.2 baseline, a **4.755% increase** on the same 20-policy book in shard 2. Both values matched live exactly. This is the only published comparable instruction count more than 3% above baseline. A local exact-pin reproduction measured a 5.03% increase. The candidate stores `Projection.care_surv(s, t)` in a native hash cache, replacing a 948-by-948 dense value array and its flags per item. The two hash helper functions accounted for roughly the net increase in the callgrind profile. The published census estimated about 22% less incremental memory per policy. [LTC cache attribution](LTC-JP-S.md)

[Full census and shard evidence](PROPOSAL-1-CENSUS.md)

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

## Commits and line counts (15 linear commits)

The head `7957dc3beb9237c779872bb479c2c9978b0ed296` is 15 single-parent commits ahead of upstream `d22426fbac8637f84510902c2fa33b6466e5acb4`. The published census, the twelve-model SEC instruction table, and the standalone CI run use `503b07c6f13d646ca4a15b020a18aea7c677a205`. The final commit adds `test_power_accumulator_shapes.py` and `test_python_call_no_entry.py`, totaling 99 lines. Historical development measurements retain the commits identified in their sections. The head's recorded unit suite had 409 passes and 8 skips, with six further unit tests requiring lifelib in that environment. [Commit series](https://github.com/alexeybaran/modelx-cython/compare/d22426fbac8637f84510902c2fa33b6466e5acb4...7957dc3beb9237c779872bb479c2c9978b0ed296), [final test-only change](https://github.com/alexeybaran/modelx-cython/compare/503b07c6f13d646ca4a15b020a18aea7c677a205...7957dc3beb9237c779872bb479c2c9978b0ed296)

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
| 7957dc3 | Test power shapes over an accumulator and Python calls of integer Cells | +0 / −0 | +0 / −0 | +99 / −0 |
| total | 37 compiler files, 10 other, 72 test files | +5,932 / −282 | +352 / −169 | +5,162 / −40 |

All files: 119 changed, +11,446 / −491.

## CI

The standalone run on measured commit `503b07c` passed all four suites in 12 of 15 jobs. Three macOS jobs were cancelled before running any steps. The later Collect verification run on `813c04f`, which includes the measured Fast SEC commit, passed 13 of 15 jobs. The two macOS free-threaded jobs ran and passed there. The regular macOS 3.14 job never received a runner in either of these runs. [Standalone workflow](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145), [Collect verification workflow](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273)

| OS | CPython | Fast SEC on 503b07c | Fast SEC plus Collect on 813c04f |
|---|---|---|---|
| Linux | 3.10 | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742997) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738706) |
| Linux | 3.12 | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079743049) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738839) |
| Linux | 3.14 | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742969) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738733) |
| Linux | 3.13t | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742703) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738910) |
| Linux | 3.14t | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079743060) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738775) |
| Windows | 3.10 | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742878) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738880) |
| Windows | 3.12 | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079743097) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738974) |
| Windows | 3.14 | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742882) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738815) |
| Windows | 3.13t | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742971) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079739018) |
| Windows | 3.14t | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079743084) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738708) |
| macOS | 3.10 | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742299) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738902) |
| macOS | 3.12 | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742739) | [No runner](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738920) |
| macOS | 3.14 | [No runner](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742798) | [No runner](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738344) |
| macOS | 3.13t | [No runner](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742802) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738782) |
| macOS | 3.14t | [No runner](https://github.com/alexeybaran/modelx-cython/actions/runs/45869162145/job/140079742824) | [Pass](https://github.com/alexeybaran/modelx-cython/actions/runs/45869161273/job/140079738755) |
