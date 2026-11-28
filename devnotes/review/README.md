# Fast SEC and Collect review package

The two compiler patches can be reviewed in sequence. Fast SEC is an independent change for ordinary compiled models. Collect then adds a portfolio API to modelx and uses it in the compiler. The optional SEC DV01 example is a separate consumer of those packages.

| Review | Base | Head | Commits | Description |
| --- | --- | --- | ---: | --- |
| Fast SEC compiler | `d22426fbac8637f84510902c2fa33b6466e5acb4` | `eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df` | 8 | [Draft](COMPILER-FAST-SEC.md) · [Diff](https://github.com/alexeybaran/modelx-cython/compare/d22426fbac8637f84510902c2fa33b6466e5acb4...eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df) |
| modelx dynamic Reference prerequisite | `fec516d7e9b5813e7d40819a72731bd35fc2972f` | `7e6ac7a4018248c373a7c43ede951363a1d4cf10` | 2 | [Diff](https://github.com/alexeybaran/modelx/compare/fec516d7e9b5813e7d40819a72731bd35fc2972f...7e6ac7a4018248c373a7c43ede951363a1d4cf10) |
| modelx Collect | `7e6ac7a4018248c373a7c43ede951363a1d4cf10` | `f83a55a0ca3b7f15896184234f2b8fd06a1da215` | 5 | [Draft](MODELX-COLLECT.md) · [Diff](https://github.com/alexeybaran/modelx/compare/7e6ac7a4018248c373a7c43ede951363a1d4cf10...f83a55a0ca3b7f15896184234f2b8fd06a1da215) |
| Compiled Collect | `eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df` | `0cb1cb7f9dfeac4bff45df4ff0b96780f2cd725f` | 4 | [Draft](COMPILER-COLLECT.md) · [Diff](https://github.com/alexeybaran/modelx-cython/compare/eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df...0cb1cb7f9dfeac4bff45df4ff0b96780f2cd725f) |
| Optional lifelib Direct example | `fbafbaab44493f77bdd015ad09f0d770a6488ded` | `c9284866839b79e880b747751e7ae2bbc213e3d9` | 1 | [Draft](LIFELIB-DV01.md) |
| Optional lifelib Collect example | `c9284866839b79e880b747751e7ae2bbc213e3d9` | `a050c361f92ba2095d91cc7172674186e3d39481` | 1 | [Draft](LIFELIB-DV01.md) |

The final compiler diffs against the measured `1994dc2` and `9076e2f` trees contain four files each, all installation metadata, documentation or CI. The modelx Collect implementation matches `d41b2d2` byte for byte outside its version tuple. The commits bring together the implementation and its behavioral coverage without rewriting existing public branches. The optional example has its own release and test evidence.

## Validation

- On WSL2 CPython 3.14.8t, built both compilers' wheels from their sdists and installed the two exact package pairs in separate fresh environments. Pip resolved dependencies normally and `pip check` passed. The build used Cython 3.3.2. Native sources expected in the sdist and wheel were present. The small example passed in live modelx, exported Python and native code with the GIL off. The Collect example checked both series access forms and retained Direct values on the same Space.
- Both installed examples also ran as exported Python and as compiled packages with imports of `modelx` and `modelx_cython` blocked.
- The scalar power qualification combines paired instruction counts and emitted source comparisons. FIA's generated sources are identical. MYGA and DV01 add native calls only at geometric MVA expressions. Existing power caching decisions match the controls. [Pinned scalar qualification archive](https://github.com/alexeybaran/modelx-cython/blob/46d1330/devnotes/review/PERFORMANCE.md)
- All 12 exact hosted jobs passed. They installed the source pin pairs together on Linux, Windows and macOS for Python 3.12 and free-threaded 3.14. They ran both compiler suites, the Collect companion suites and released SEC examples. [Exact review workflow](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105)
- The scaling refresh uses the review heads in the table. Compiled Direct and Collect cover all 12 released SEC products, BasicTerm_SC and the full MYGA DV01 report at one and four workers. The MYGA retained-state and live/exported comparison is in the Collect draft. [Updated compiled benchmark tables](SEC-BUILD-BENCHMARKS.md) · [MYGA memory and API](COMPILER-COLLECT.md)

## Repeat the installed example

Use separate free-threaded CPython environments for the two rows. Check out each source pin and use absolute paths. Set `PYTHON_GIL=0` before running. The Direct modelx pin is upstream `fec516d7e9b5813e7d40819a72731bd35fc2972f`. The Collect pin is the review companion above. The same script also creates the exported sample and the compiled assertion file.

```sh
python -m pip install "$MODELX_SOURCE" "$COMPILER_SOURCE[free-threaded]"
python -m pip check
python /path/to/installed_example.py direct  # use collect for the Collect pair
python sample.py
mx2cy portfolio_nomx --sample sample.py --no-spec
python check.py
```

The versioned modelx review checkout satisfies the compiler dependency in a source install. For a normal public installation and upstream CI with the standard package index, the Collect-capable modelx release must be available first. The review version is a separate commit so release numbering stays with the maintainer.

For the Collect pair, a fresh free-threaded environment can install the two pinned Git sources together.

```sh
python -m pip install \
  'modelx @ git+https://github.com/alexeybaran/modelx.git@f83a55a0ca3b7f15896184234f2b8fd06a1da215' \
  'modelx-cython[free-threaded] @ git+https://github.com/alexeybaran/modelx-cython.git@0cb1cb7f9dfeac4bff45df4ff0b96780f2cd725f'
```

## Submission order

1. Review Fast SEC against compiler main. Current upstream main is its base. Smaller upstream changes already open touch the same translator or spec paths. In particular, reconcile the `use_libm` flag with native dispatch if [compiler PR #60](https://github.com/fumitoh/modelx-cython/pull/60) lands first. Keep upstream's chosen API and rerun the affected emitted-source and correctness tests on the resulting base.
2. Review the modelx dynamic Reference export independently, then the Collect API, isolation and export hook. Release a versioned modelx package containing the hook before requesting merge of compiled Collect.
3. Review the compiler Collect diff against Fast SEC. Treat the optional lifelib example as a consumer after the corresponding compiler and modelx dependencies are available.

The earlier broadly scoped [compiler PR #62](https://github.com/fumitoh/modelx-cython/pull/62), [modelx PR #288](https://github.com/fumitoh/modelx/pull/288) and [lifelib PR #191](https://github.com/lifelib-dev/lifelib/pull/191) are separate. Use these split branches to discuss replacement or submission order with the maintainer.

The earlier Fast SEC and compiled Collect descriptions preserve the original benchmark runs, commit counts, activation rules and guarantees under their own pins. The original combined proposals also document the 57 converted Regional products, the older export protocol and the earlier Windows attribution, per-policy and per-scalar reports. [Original Fast SEC description](https://github.com/alexeybaran/modelx-cython/blob/c39337225c8d4097bb93b4cd540c547e569ecaa4/devnotes/PR-DESCRIPTION.md) · [Original compiled Collect description](https://github.com/alexeybaran/modelx-cython/blob/be40323abe7e5c5f8d12a7f1dbe7b2d89d77f5d9/devnotes/PR-DESCRIPTION-collect.md) · [Earlier Windows attribution](https://github.com/alexeybaran/modelx-benchmark-toolkit/blob/90ee8693a99ca1dfda785c2f2da2c577354a3c65/parallel/20261113/round3/windows-census/report-6379886/ATTRIBUTION-COLD.md)

### Open upstream compiler changes

These were checked with `git merge-tree` against the Fast SEC review head. Its base remains current upstream main.

| Upstream PR | Proposed behavior | Merge overlap and retained behavior |
| --- | --- | --- |
| [#55](https://github.com/fumitoh/modelx-cython/pull/55) | Widen a mixture of Python `float` and NumPy `float64` arguments to C `double`. Log the other mixed-type fallbacks. | The tracer auto-merges. Combine the conflicting tracer regression coverage, new tests and release notes. Check the merged widening through typed entries. |
| [#56](https://github.com/fumitoh/modelx-cython/pull/56) | Retain Python object returns for unsupported NumPy buffers and zero-dimensional arrays. | The builder auto-merges. Keep both test groups, including the independence of public object returns and eligible native memoryviews or held buffers. |
| [#57](https://github.com/fumitoh/modelx-cython/pull/57) | Validate names before sampling and keep the prior translated output through preflight. | `cli.py` changes both directory replacement and translation order. Integrate the preflight before rotation and keep both ordinary translation and compile-only paths. |
| [#58](https://github.com/fumitoh/modelx-cython/pull/58) | Validate and apply model-wide Cython directives in the spec. | Both sides edit spec validation, option defaults and generated build setup. Preserve documented upstream override behavior and reject incompatible directives. |
| [#59](https://github.com/fumitoh/modelx-cython/pull/59) | Apply per-Cell `param_type` overrides. | The combined builder must feed the override into public signatures, storage choices and power classification. Retain Fast SEC's distinct checks on rebound formula parameters and require usable bounds before selecting dense storage. |
| [#60](https://github.com/fumitoh/modelx-cython/pull/60) | Add opt-in `use_libm` for `math.exp`, `math.log` and `math.pow`. | Both sides change runtime helpers, the power pass, math Reference facts and dispatch. Reconcile the maintainer's chosen spec API and defaults, use one lowering path, and retest native arithmetic, rebinding, domain fallback and generated code. |
