# Collect: the lifelib side and the evidence

## Patch series (on lifelib `pr/sec-dv01` 0d48fb09; `git am *.patch`)

| patch | change |
|---|---|
| 0001 | `dv01_sec/model.py` +5: `Valuation.set_collect` over `point_indices`, outputs the liability PV and, on the base curve, the cash flows |
| 0002 | `dv01_sec/compiled.py`: `Data` and `Actual` join `ESG` and `Market` in the export's `locked_spaces` |
| 0003 | `dv01_sec/report.py` +26 (`collect_report`, 21 lines of code) and `--collect` on `run_dv01_sec.py`; the pooled driver stays the default |
| 0004 | test: the report through collect equals the pooled driver's bit for bit and live modelx, 1 and 4 workers, shipped book (16 passed) |
| 0005 | `dv01-sec.md`: both routes, the locked Spaces, time and memory at 1,050 policies |

Requires modelx with `Space.set_collect` and the compiled collect of this repository.

## DV01, 1,050 policies (medians of 5, wall s / CPU s / peak MB)

| route | 1 worker | 4 workers | scaling |
|---|---|---|---|
| pooled driver | 9.97 / 9.9 / 252 | 3.80 / 13.4 / 352 | 2.62x |
| collect | 9.81 / 9.7 / 1120 | 3.75 / 12.5 / 1099 | 2.62x |
| collect without the borrowed read | 10.31 / 10.2 / 1120 | 7.79 / 28.6 / 1099 | 1.32x |

DV01 129960.61614055187 in every run; on 70 policies every route equals live modelx bit for bit. Collect holds the
1,050 `Actual` ItemSpaces across the 39 collects (each close reads them); the pooled driver deletes each policy's
when it is done. Collect's gain is less model code, not speed.

## The twelve `*_US_SEC` models (medians of 3, wall s / peak MB)

Collect declared in a measurement script (`measure/prep_sec.py`: `book` = the table's positions, outputs
`{"sum": {"net_cf": net_cf_total()}}`, `Book.total` the book-total cell; nothing in lifelib changed), exported with
`Data` locked. Books tiled from the shipped table (per-point side tables tiled with it) to the size at which the
direct route, keeping every item, fits 4 GB. Values: on the shipped book collect (1 and 4 workers, with and without
the borrowed read) equals live modelx bit for bit on all twelve; on the tiled book collect and the direct route at one
worker give one value. Direct at four workers adds four partial sums and differs in the last bits.

| model | points | direct kept 1w | direct kept 4w | collect 1w | collect 4w | scaling direct / collect | collect without borrowed read 1w / 4w (scaling) | direct without, 4w (scaling) |
|---|---|---|---|---|---|---|---|---|
| DIA_US_SEC | 2229 | 1.67/1747 | 0.79/1750 | 0.87/100 | 0.51/108 | 2.13 / 1.73 | 0.87 / 0.53 (1.65x) | |
| FIA_US_SEC | 1156 | 1.05/1802 | 0.40/1804 | 0.28/99 | 0.12/108 | 2.64 / 2.30 | 0.29 / 0.13 (2.33x) | |
| IUL_US_SEC | 792 | 1.45/1946 | 0.54/1949 | 0.70/101 | 0.27/114 | 2.71 / 2.65 | 0.74 / 0.26 (2.88x) | |
| MYGA_US_SEC | 1552 | 1.18/1740 | 0.40/1742 | 0.33/97 | 0.14/104 | 2.95 / 2.27 | 0.34 / 0.14 (2.44x) | 0.40 (2.83x) |
| RILA_US_SEC | 863 | 1.34/1396 | 2.13/1398 | 0.72/101 | 2.05/111 | 0.63 / 0.35 | 0.73 / 2.01 (0.37x) | |
| SPIA_US_SEC | 3299 | 1.52/1824 | 0.46/1826 | 0.65/96 | 0.23/103 | 3.32 / 2.81 | 0.77 / 0.23 (3.35x) | |
| Term_US_SEC | 8117 | 1.70/2522 | 0.62/2523 | 0.65/94 | 0.23/100 | 2.76 / 2.79 | 0.61 / 0.22 (2.75x) | 0.60 (3.09x) |
| ULSG_US_SEC | 1449 | 1.20/1815 | 0.40/1818 | 0.42/95 | 0.17/104 | 3.00 / 2.47 | 0.50 / 0.18 (2.75x) | |
| UL_US_SEC | 2046 | 1.37/2083 | 0.47/2085 | 0.48/95 | 0.19/103 | 2.90 / 2.53 | 0.45 / 0.19 (2.43x) | |
| VA_US_SEC | 954 | 1.39/2020 | 0.95/2022 | 0.49/104 | 0.16/117 | 1.46 / 3.01 | 0.43 / 0.17 (2.57x) | 0.97 (1.47x) |
| VUL_US_SEC | 817 | 2.33/2322 | 2.49/2339 | 1.53/101 | 0.49/116 | 0.94 / 3.10 | 1.58 / 0.50 (3.14x) | |
| WholeLife_US_SEC | 3178 | 1.56/2363 | 0.59/2364 | 0.54/95 | 0.22/102 | 2.62 / 2.39 | 0.53 / 0.24 (2.25x) | |

The books are small (memory bounds the direct route), so four-worker runs take 0.1-0.5 s and their scaling
includes thread start and noise. RILA_US_SEC runs slower on four workers than on one on both routes; the cause is
outside collect and was not investigated. VA_US_SEC's `sample_sec.py` calls every public function of the Space class
on an item, which now includes `collect` (modelx raises for an ItemSpace); the measurement copy skips it.

## Scripts (`measure/`)

`prep_sec.py` (copy, tile, declare, export), `tile_sec.py` / `tile_side.py` (books), `sample_c.py` (the model's own
sample, then the book total), `sec_one.sh` (prepare and build with both compilers), `sec_size.py` (book size),
`sec_bench.sh` and `run.py` (fresh process per run), `dv01run.sh` (DV01 runs). Paths are those of the measurement
machine.
