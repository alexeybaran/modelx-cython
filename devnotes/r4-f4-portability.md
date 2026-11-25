# R4 F4: portability of the layer-1 trial cut (GitHub Actions)

Branch `verify/r4-layer1-trial` = `r4/s2-layer1-trial` (56e01f9) + workflow commits + one test-only commit.
No compiler source changed. Upstream modelx 0.33.0 and lifelib 0.17.4 from PyPI (0.33.0 is modelx main's version).

## What each lane runs (`.github/workflows/python-package.yml`, one job per lane)
- unit: all of `modelx_cython/tests` except `compiled/` (with lifelib, so its sample tests run).
- compiled: `modelx_cython/tests/compiled` + `integration_tests`, 116 tests: the 129 of the report less the 11
  below and the 2 allocation tests run separately.
- allocation: `test_value_storage.py::test_unwritten_values_after_pickle_and_copy` (shipped, poisoned with
  0xA5 + heap types, and fallback builds; copy, pickle, deepcopy, scrubbed reduce) and
  `::test_unwritten_values_of_a_locked_space` (two threads copying a locked Space while it fills; the thread
  part asserts only on free-threaded builds).
- sec: `.github/scripts/sec_compare.py` builds released lifelib `Term_US_SEC` and `MYGA_US_SEC` (the latter
  builds the numeric-cache `.pyx` helper and the divisibility checks) with their `sample_sec.py`/`spec_sec.py`
  and runs lifelib's own C4 (`sec_support.compiled_mismatches`: every point swept, every cached value and
  raise compared, compiled run in a child) at lifelib's non-exact tolerance, rel 1e-12.

## Matrix (final run 4, f34086e, unless noted)
| lane | unit | compiled | allocation | sec |
|---|---|---|---|---|
| ubuntu 3.10 / 3.12 / 3.14 | pass | pass | pass | pass |
| ubuntu 3.14t / 3.13t | pass | pass | pass | pass |
| windows (MSVC 14.52) 3.10 / 3.12 / 3.14 | pass | pass | pass | pass |
| windows 3.14t / 3.13t | pass | pass | pass | pass |
| macos (clang, arm64) 3.10 (run 2), 3.12 | pass | pass | pass | pass |
| macos 3.14t (run 3) | pass | pass | pass | pass |
| macos 3.14 / 3.13t | pending (run 4) | | | |

Free-threaded lanes assert `Py_GIL_DISABLED` and `sys._is_gil_enabled() is False` before testing.
## Real failures (compiler defects)
None. No lane failed a test for a reason in the compiler. In particular the allocation change installs its
custom `tp_alloc` (layout check passes) on gcc/x86-64, MSVC/x64 and clang/arm64, on 3.10 to 3.14t; the
fallback build warns and zero-fills on all three; the numeric caches build as package `.pyx` everywhere.

## Failures that were not the compiler, and what was done
1. Test assumption, fixed in the test (commit 9461d43, test-only, `compiled/test_value_storage.py`):
   - 3.10 and 3.12, all OS: `AttributeError: module 'sys' has no attribute '_is_gil_enabled'` in the thread
     probe. Fix: `getattr(sys, '_is_gil_enabled', lambda: True)()`, as the cut already does elsewhere.
   - Windows, all versions: `assert ... '-DMX_POISON_UNWRITTEN_VALUES' in build.stdout` failed: setuptools
     gives `CPPFLAGS` to unix-style compilers only, so MSVC built the "poisoned"/"fallback" models unpoisoned.
     Fix: under MSVC the macros also go in `CL` (cl reads it, accepts `-D`), and the log-echo check is skipped
     (cl does not echo `CL`); the fallback warning and heap-type assertions still prove the macros applied.
     Proposal for the real cut: give the test (or `build.py`) one portable way to add defines, e.g.
     `define_macros` from an environment variable, rather than relying on `CPPFLAGS`.
2. Harness: lifelib 0.17.4 runs its compiled child as `python -P` (3.11+); on 3.10 `sec_compare.py` drops it.
3. Toolchain/packaging (workflow fixes only):
   - 3.13t: newest libcst has no cp313t wheel and does not build (PyO3: no free-threaded below 3.14); newest
     scipy/numpy/pandas lack some cp313t wheels, and numpy/pandas built from source on Windows 3.13t failed
     to load (`DLL load failed while importing groupby`). Fix: `--only-binary scipy,libcst,numpy,pandas`
     (resolves to libcst 1.8.6, scipy 1.17.1, numpy 2.4.6, pandas 3.0.3 on 3.13t).
   - macOS free-threaded: a job-wide `PYTHON_GIL=0` aborted setup-python's macOS installer (it runs its own
     non-free-threaded Python). Fix: set `PYTHON_GIL=0` via `$GITHUB_ENV` after setup.
   - macOS runners: jobs waiting more than 15 minutes for a runner were cancelled (runs 1-4). macOS lanes are
     now a separate job, one at a time; a lane cancelled this way is shown as "not run".
4. Numeric note (not a failure here): with lifelib's bit-exact C4 (its default on x86-64), MYGA_US_SEC
   differs from modelx in the last bit on a few values (seen locally, Linux x86-64, gcc, Python 3.13:
   e.g. `net_cf(46)` -174.98694080700818 vs ...815). The cut's report and this check use a tolerance; if the
   real cut is to meet lifelib's exact C4, that difference needs a look (FP order in the power rewrite or
   checked arithmetic is the likely place).

## Tests left out, and why
- 11 compiled tests that build models with `new_space(collect_over=...)`, development modelx API absent from
  upstream modelx (as in the cut's report): `test_compiled_contract.py::test_exported_contract_and_collection`
  [1, 2, 4]; `test_input_scenarios.py::test_new_input_scenario_after_collect`;
  `test_locked_numeric_domains.py::test_locked_native_calls_and_dependent_collect_domain`;
  `test_over_limit_hash_store.py::test_over_limit_cell_is_one_hash_store_on_both_routes` and
  `::test_removed_spec_key_reuse_dense_windows_is_refused`;
  `test_raised_sample_calls.py::test_raised_sample_calls_keep_the_returned_type`;
  `test_shared_dense_publish.py::test_shared_dense_cache_publishes_its_flags`;
  `test_uncached_entries.py::test_uncached_cells_take_a_direct_entry_and_equal_live`;
  `test_value_storage.py::test_unwritten_values_on_collect_and_direct_routes`.
- Python 3.9 lanes dropped: below the 3.10 floor (pyproject still says `>=3.9`; not changed here).
- lifelib's own `pytest -m cython` suite: it needs `tools/build_sec_cy.py`, absent from both the wheel and
  lifelib main, so the C4 check is driven by `sec_compare.py` instead.

## Runs (https://github.com/alexeybaran/modelx-cython/actions/runs/ + id)
1: 45621453975 (found the 3.10/MSVC test issues); 2: 45624825505; 3: 45628540754; 4: 45633805590 (final).
