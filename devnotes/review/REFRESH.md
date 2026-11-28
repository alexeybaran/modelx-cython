# Performance refresh on the review heads

The refresh runs on a WSL2 laptop with eight physical Ryzen 7 5800H cores, free-threaded CPython 3.14.8 and Cython 3.3.2. It verifies the GIL stays off. The [review branches](README.md) supply the exact source pins. [Earlier measurements](PERFORMANCE.md) retain their original toolchain and source pins.

| Workload | Matched comparison | Measurements |
| --- | --- | --- |
| All 12 released SEC products | Fast SEC and upstream compiler `d22426fbac8637f84510902c2fa33b6466e5acb4` with modelx `fec516d7e9b5813e7d40819a72731bd35fc2972f` and unchanged lifelib `20d4daef3281d538b3fbda6d0fb951fcceb3a6f4` | Direct wall time per policy, process CPU, native instructions, peak RSS, translation and C build time. A failed upstream build gets an explicit row. |
| MYGA SEC | Direct and Collect in live modelx, exported Python and compiled code. Separate compiler Direct comparison between Fast SEC and Collect heads. | Equal-book time with Direct policy deletion. Separate retained-state peak RSS with no deletion. Exported and compiled four-worker timing. |
| Full SEC DV01 and CS01 report | Fast SEC Direct and Collect-branch Direct and Collect on the corresponding rebased examples and the same ordered 1,050-policy book. | One and four workers. Wall time, process CPU, peak RSS and one-worker instructions. The timed Collect route calls a portfolio report Cell with the CSV input directory. |

Each matched elapsed comparison uses five alternating fresh-process rounds with no active builds. One-worker runs are pinned to distinct physical cores. Four-worker results use an isolated run on four physical cores. Instructions come from separate one-worker hardware-counter runs with event coverage checked. Each Direct policy value and Collect total is checked against live outputs. The full DV01 benchmark checks all 72 DV01 and CS01 values. Peak RSS uses the child's own high-water mark.

The SEC books are sized for several seconds on Fast SEC. The compiled MYGA comparison uses the same book at both worker counts. The higher-policy MYGA memory comparison covers exported and compiled modes. Live Direct retains a much larger per-policy dependency graph, so the live memory comparison uses the smaller common book.
