# Fast SEC and Collect review package

The two compiler patches can be reviewed in sequence. Fast SEC is an independent change for ordinary compiled models. Collect then adds a portfolio API to modelx and uses it in the compiler. The optional SEC DV01 example is a separate consumer of those packages.

| Review | Base | Head | Commits | Description |
| --- | --- | --- | ---: | --- |
| Fast SEC compiler | `d22426fbac8637f84510902c2fa33b6466e5acb4` | `eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df` | 8 | [Draft](COMPILER-FAST-SEC.md) · [Diff](https://github.com/alexeybaran/modelx-cython/compare/d22426fbac8637f84510902c2fa33b6466e5acb4...eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df) |
| modelx dynamic Reference prerequisite | `fec516d7e9b5813e7d40819a72731bd35fc2972f` | `7e6ac7a4018248c373a7c43ede951363a1d4cf10` | 2 | [Diff](https://github.com/alexeybaran/modelx/compare/fec516d7e9b5813e7d40819a72731bd35fc2972f...7e6ac7a4018248c373a7c43ede951363a1d4cf10) |
| modelx Collect | `7e6ac7a4018248c373a7c43ede951363a1d4cf10` | `f83a55a0ca3b7f15896184234f2b8fd06a1da215` | 5 | [Draft](MODELX-COLLECT.md) · [Diff](https://github.com/alexeybaran/modelx/compare/7e6ac7a4018248c373a7c43ede951363a1d4cf10...f83a55a0ca3b7f15896184234f2b8fd06a1da215) |
| Compiled Collect | `eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df` | `0cb1cb7f9dfeac4bff45df4ff0b96780f2cd725f` | 4 | [Draft](COMPILER-COLLECT.md) · [Diff](https://github.com/alexeybaran/modelx-cython/compare/eb7fcc41fcf07e94938d06e9433dc0bc42b3e4df...0cb1cb7f9dfeac4bff45df4ff0b96780f2cd725f) |

The final compiler diffs against the measured `1994dc2` and `9076e2f` trees contain four files each, all installation metadata, documentation or CI. The modelx Collect implementation matches `d41b2d2` byte for byte outside its version tuple. The commits bring together the implementation and its behavioral coverage without rewriting existing public branches. The optional example has its own release and test evidence.

## Validation

- On WSL2 CPython 3.14.8t, built both compilers' wheels from their sdists and installed the two exact package pairs in separate fresh environments. Pip resolved dependencies normally and `pip check` passed. The build used Cython 3.3.2. Native sources expected in the sdist and wheel were present. The small example passed in live modelx, exported Python and native code with the GIL off. The Collect example checked both series access forms and retained Direct values on the same Space.
- The paired instruction gate covers seven rows on the same ordered books. DV01 Direct improves by 0.444% on Fast SEC and 0.447% on the compiler with Collect. DV01 Collect improves by 0.457%. FIA is effectively flat on both routes. MYGA improves by 0.446% in Direct and 0.502% in Collect. The mixed DV01 export exactly reproduces the package that was measured. FIA is byte-identical and the generated MYGA and DV01 arithmetic changes are limited to geometric MVA. Existing power caching decisions match the controls.
- The exact hosted workflow installs the source pin pairs together on Linux, Windows and macOS and runs both compiler suites, the Collect companion suites and the released SEC examples. This workflow is separate from the code proposed upstream. [Exact review workflow](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105)

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

## Submission order

1. Review Fast SEC against compiler main. Current upstream main is its base. Smaller upstream changes already open touch the same translator or spec paths. In particular, reconcile the `use_libm` flag with native dispatch if [compiler PR #60](https://github.com/fumitoh/modelx-cython/pull/60) lands first. Keep upstream's chosen API and rerun the affected emitted-source and correctness tests on the resulting base.
2. Review the modelx dynamic Reference export independently, then the Collect API, isolation and export hook. Release a versioned modelx package containing the hook before requesting merge of compiled Collect.
3. Review the compiler Collect diff against Fast SEC. Treat the optional lifelib example as a consumer after the corresponding compiler and modelx dependencies are available.

The earlier broadly scoped [compiler PR #62](https://github.com/fumitoh/modelx-cython/pull/62), [modelx PR #288](https://github.com/fumitoh/modelx/pull/288) and [lifelib PR #191](https://github.com/lifelib-dev/lifelib/pull/191) are separate. Use these split branches to discuss replacement or submission order with the maintainer.
