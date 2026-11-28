# Speed up compiled scalar models with guarded native Cell execution

Compiled scalar insurance models spend much of their time looking up Cell values and crossing Python call boundaries. This change emits typed Cell entries and chooses native storage from the Cell's argument types, ranges and observed usage. Exact receivers take the native path. The existing public interfaces and supported fallback paths remain available.

## Changes to review

- Use typed entries, dense caches with publication flags, and native numeric key stores. Keep shared reads synchronized for Spaces explicitly locked at export.
- Lower supported real powers, rounding and calls through the standard `math` Reference. Keep existing one-entry power memoization for eligible private Items. Apply the checked real power helper directly to additionally proved scalar arithmetic returns.
- Report declared or sampled out-of-range arguments with the Cell and its range. Preserve Python's unbound-local behavior and reject fractional writes to integer References. The signed-64-bit arithmetic contract is unchanged.
- Package the native sources in the wheel and sdist. The Python floor is 3.10. On free-threaded Python, install `modelx-cython[free-threaded]` to select Cython 3.3.2 or later. Standard Python retains the ordinary Cython 3.2.4 floor.

## Evidence

The earlier matched census covered all 12 SEC products. They used **0.202–0.454×** upstream's instructions per policy. In the wider 95-model census, 73 passed on both compilers, nine passed only with the candidate, and 13 had the same failure class. The largest measured Regional increase on that earlier compiler was LTC Japan at **+4.76%**. The observed dense-cache policy in this series addresses the identified hash-cache path. This description makes no updated full-census claim. [Earlier compiler and base](https://github.com/alexeybaran/modelx-cython/compare/d22426fbac8637f84510902c2fa33b6466e5acb4...503b07c6f13d646ca4a15b020a18aea7c677a205)

The later local WSL2 qualification measured 10 comparable SEC Direct workloads from matched books. At the measured Direct pin, instructions fell 69–77% across available upstream pairs, and the five-round elapsed measurements also improved. The upstream controls for DIA and VUL did not build under this local toolchain. SEC DV01 depends on a separately exported dynamic-Reference example and has no comparable upstream compile result.

| Final scalar change versus the measured Direct tree | Policies | Instruction change |
| --- | ---: | ---: |
| FIA SEC | 32,700 | −0.010% |
| MYGA SEC | 34,300 | −0.446% |
| SEC DV01 | 1,050 | −0.444% |

The final scalar comparison used three alternating fresh-process pairs on a Ryzen 7 5800H under WSL2, with free-threaded CPython 3.14.8 and the GIL off. FIA's generated class and runtime sources match byte for byte. MYGA and SEC DV01 change only the geometric MVA power expressions. Every DV01 run matched all 72 report values bit for bit. No new isolated elapsed-speed claim is made for the scalar change.

## Review order

Start with the typed execution and storage commits, then read the math and power commits with their fixtures. The final installation commit changes requirements, documentation and CI installation only. The compiler can be reviewed on `modelx-cython` main with modelx 0.33.0. The optional SEC DV01 example also needs the separate dynamic Space Reference export support.

The existing `use_libm` proposal intersects the native math emitter. The other open proposals around parameter types, compiler directives, and reserved-name validation intersect translation. This branch is based on current main. [Native math proposal](https://github.com/fumitoh/modelx-cython/pull/60) · [Parameter types](https://github.com/fumitoh/modelx-cython/pull/59) · [Directives](https://github.com/fumitoh/modelx-cython/pull/58) · [Name validation](https://github.com/fumitoh/modelx-cython/pull/57)
