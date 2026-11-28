# Speed up compiled scalar models with guarded native Cell execution

Compiled scalar insurance models spend much of their time looking up Cell values and crossing Python call boundaries. This change emits typed Cell entries and chooses native storage from the Cell's argument types, ranges and observed usage. Exact receivers take the native path. The existing public interfaces and supported fallback paths remain available.

## Changes to review

- Use typed entries, dense caches with publication flags, and native numeric key stores. Keep shared reads synchronized for Spaces explicitly locked at export.
- Lower supported real powers, rounding and calls through the standard `math` Reference. Keep existing one-entry power memoization for eligible private Items. Apply the checked real power helper directly to additionally proved scalar arithmetic returns.
- Report declared or sampled out-of-range arguments with the Cell and its range. Preserve Python's unbound-local behavior and reject fractional writes to integer References. The signed-64-bit arithmetic contract is unchanged.
- Package the native sources in the wheel and sdist. The Python floor is 3.10. On free-threaded Python, install `modelx-cython[free-threaded]` to select Cython 3.3.2 or later. Standard Python retains the ordinary Cython 3.2.4 floor.

## Evidence

All 12 released SEC products built and matched live modelx. In the 10 equal-book pairs that also built upstream, Fast SEC used 69–77% fewer instructions per policy and median elapsed time was 4.0–6.6× faster. The published table includes translation, C build time and peak memory. The upstream DIA and VUL controls failed in Cython with `complex types are unordered`. The separate SEC DV01 example requires dynamic Reference export and has no upstream performance ratio. All 12 exact review CI jobs passed. [Benchmark refresh](REFRESH.md) · [Exact review CI](https://github.com/alexeybaran/modelx-cython/actions/runs/46147767105)

The earlier census remains available with its original pins and limitations. It documented an LTC Japan allocation and instruction-count trade involving `Projection.care_surv`. The owner accepted the earlier result. The current release comparison covers supported SEC workloads. [Earlier full census](https://github.com/alexeybaran/modelx-cython/blob/c39337225c8d4097bb93b4cd540c547e569ecaa4/devnotes/PR-DESCRIPTION.md)

## Review order

Start with the typed execution and storage commits, then read the math and power commits with their fixtures. The final installation commit changes requirements, documentation and CI installation only. The compiler can be reviewed on `modelx-cython` main with modelx 0.33.0. The optional SEC DV01 example also needs the separate dynamic Space Reference export support.

The existing `use_libm` proposal intersects the native math emitter. The other open proposals around parameter types, compiler directives, and reserved-name validation intersect translation. This branch is based on current main. [Native math proposal](https://github.com/fumitoh/modelx-cython/pull/60) · [Parameter types](https://github.com/fumitoh/modelx-cython/pull/59) · [Directives](https://github.com/fumitoh/modelx-cython/pull/58) · [Name validation](https://github.com/fumitoh/modelx-cython/pull/57)
