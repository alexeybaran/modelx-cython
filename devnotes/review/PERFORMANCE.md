# Matched SEC benchmark results

This page is the archive for the earlier performance qualification. See the [current refresh](REFRESH.md) for measurements on the review heads.

These results compare upstream compiler d22426f with measured Fast SEC compiler 5186502, using modelx fec516d and lifelib 20d4dae. The final shared scalar adjustment has its own paired results below. The review packaging changes leave compiler source identical to the final measured source.

WSL2 on the same eight-core Ryzen 7 5800H, free-threaded CPython 3.14.8 with GIL off. Equal ordered books within each pair. Wall medians are five alternating fresh-process rounds and instructions are separate hardware-counter passes. Runs compare required values with live modelx.

| SEC product | Policies | Candidate / upstream instructions | Upstream → candidate wall seconds | Upstream → candidate compile seconds |
| --- | ---: | ---: | ---: | ---: |
| FIA | 32,700 | 0.306 | 59.412 → 11.566 | 203.9 → 275.3 |
| IUL | 10,500 | 0.270 | 59.115 → 10.980 | 204.6 → 237.8 |
| MYGA | 34,300 | 0.247 | 70.630 → 11.969 | 142.3 → 195.8 |
| RILA | 27,000 | 0.249 | 56.719 → 12.541 | 189.8 → 268.2 |
| SPIA | 39,700 | 0.300 | 55.157 → 11.851 | 111.6 → 131.3 |
| Term | 95,700 | 0.279 | 67.703 → 15.667 | 56.9 → 65.0 |
| ULSG | 27,100 | 0.300 | 57.670 → 10.938 | 125.3 → 162.4 |
| UL | 28,600 | 0.267 | 62.126 → 10.268 | 110.8 → 126.0 |
| VA | 24,100 | 0.228 | 74.157 → 11.991 | 296.5 → 352.9 |
| WholeLife | 41,800 | 0.284 | 58.301 → 10.712 | 112.7 → 111.2 |

The local upstream DIA and VUL packages failed during native code generation. The upstream compiler also could not translate the SEC DV01 export. Those tests supply no upstream speed ratio. Compile time combines translation and C build and excludes export.

## Final scalar power adjustment

Compare each adjusted stack with its own previous head on the same book and route. Three alternating fresh-process pairs for each row. Outcomes checked against live modelx or the sealed 72-value DV01 oracle.

| Model | Route | Policies | Instructions before → after | Paired change |
| --- | --- | ---: | ---: | ---: |
| FIA | Direct | 32,700 | 82,293,531,846 → 82,285,110,369 | −0.0102% |
| FIA | Collect 1w | 32,700 | 75,681,556,195 → 75,682,144,108 | +0.0008% |
| MYGA | Direct | 34,300 | 71,685,807,758 → 71,365,943,082 | −0.4461% |
| MYGA | Collect 1w | 34,300 | 64,709,009,490 → 64,384,626,401 | −0.5016% |
| DV01 | Fast SEC Direct | 1,050 | 87,262,232,687 → 86,874,971,925 | −0.4438% |
| DV01 | Mixed package Direct | 1,050 | 86,716,355,293 → 86,328,349,237 | −0.4474% |
| DV01 | Mixed package Collect 1w | 1,050 | 84,932,481,459 → 84,544,266,416 | −0.4571% |

The compared compiler commits are 5186502 versus 1994dc2 for Fast SEC and 3cd839d versus 9076e2f for the mixed package. DV01's mixed package reuses a native build only after its normal emitted package and all class/runtime build inputs matched byte for byte. FIA's generated classes and system modules match byte for byte. MYGA and DV01 add only the intended native MVA power calls. These counts do not claim an isolated elapsed-speed gain for the scalar change.

## Historical scope

The 95-model census and MYGA retained-state RAM results in the review descriptions are separately measured campaigns. The earlier Fast SEC census compiler was 503b07c and the MYGA common-book memory compiler was 6f0559a with modelx d8b6fbe and lifelib 20d4dae. They illustrate the mechanisms and carry their original pins.
