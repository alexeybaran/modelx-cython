# Collect re-measured on 58ff45d (r4/p2-collect-7)

Compiler 58ff45d, modelx pr/collect 1aa21f4, lifelib pr/sec-dv01-collect 6a456523 (Data and Actual locked), the
same 1,050-policy book as before (the seven new-business points repeated), free-threaded CPython 3.14.6, 4 vCPU.
Fresh process per run; the runner's own compiled child; CPU and peak RSS of that child.

## DV01, 1,050 policies: 5 alternating rounds, median (range)

| route | workers | wall s | CPU s | peak MB | scaling |
|---|---|---|---|---|---|
| direct (pooled driver) | 1 | 16.27 (14.97-18.05) | 17.3 | 253 | |
| direct (pooled driver) | 4 | 5.09 (4.41-5.23) | 19.8 | 353 | 3.20x |
| collect | 1 | 12.85 (11.82-13.37) | 13.9 | 1121 | |
| collect | 4 | 4.89 (4.74-5.16) | 18.0 | 1100 | 2.63x |

DV01 129960.61614055187 and CS01 157856.97087213397 in every run, as before (bit for bit).

The machine is slower today than at the earlier measurement: the earlier build (`r4/p2-collect-3`, same book), re-run
now, takes 5.16 s for collect at four workers where it took 3.75 s. Interleaved on the same machine, 3 rounds, median:

| route | workers | earlier build | 58ff45d | change |
|---|---|---|---|---|
| direct | 1 | 15.31 | 14.34 | -6% |
| direct | 4 | 5.05 | 4.48 | -11% |
| collect | 1 | 14.41 | 12.27 | -15% |
| collect | 4 | 5.16 | 4.86 | -6% |

Peak memory is unchanged: collect 1.10-1.12 GB, direct 0.25 / 0.34-0.35 GB. Nothing is worse than before.

## RILA_US_SEC and DIA_US_SEC collect, 3 rounds interleaved with the earlier build, median

| model | points | build | 1 worker s | 4 workers s (range) | scaling |
|---|---|---|---|---|---|
| RILA_US_SEC | 863 | earlier | 1.081 | 1.313 (1.302-1.362) | 0.82x |
| RILA_US_SEC | 863 | 58ff45d | 0.441 | 0.528 (0.494-0.535) | 0.84x |
| DIA_US_SEC | 2229 | earlier | 1.542 | 0.712 (0.680-0.766) | 2.17x |
| DIA_US_SEC | 2229 | 58ff45d | 1.159 | 0.706 (0.597-0.883) | 1.64x |

Values equal between builds (RILA -38020248.73820954, DIA -652661961.9109885). RILA runs 2.5x faster on one worker
with native math, but four workers are still slower than one; the direct route behaves the same way, so the cause is
outside collect and is not explained here. DIA's four-worker time is unchanged, so its scaling falls as its one-worker
time falls.
