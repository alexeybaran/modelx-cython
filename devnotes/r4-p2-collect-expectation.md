# Thin collect: what I expect before measuring

Written before the first measurement; the report compares against it.

## Lines (non-test)

| piece | expected lines |
|---|---|
| declaration: the `_mx_collect_cache` slot of the compiled class (parser + pxd) | 6 |
| hook: `_mx_collect_hook` and its version, generated from the exported `collect` and `_mx_collect_item` | 60 |
| releasing evaluator: items in domain order, outputs checked and added by the export's own `prepare_output` and `Accumulator`, a created item released by the existing root release | 20 |
| worker threads: shared index, outputs added in domain order as they complete, first failing coordinate raised | 35 |
| the refusal of unlocked shared Spaces with more than one worker | 15 |
| spec key `collect_workers`, wiring in transformer and config | 15 |
| total | about 150 |

Nothing else of the old implementation: no reused receiver, no native sums, no chunk merging, no
factory, no protocol adapter, no per-policy bound checks.

## Speed

- Per item the thin path adds a Python-level item construction (the exported `_mx_collect_item`), the
  `collect_outputs()` dict, `prepare_output`, one `Accumulator.add` and, with workers, two lock rounds:
  a few microseconds against milliseconds of projection per policy. Expected: within 1% of the old
  collect at one worker on both models.
- The old implementation reuses one receiver per worker; the thin one allocates and frees an item per
  policy (its dense caches). Expected cost 0-3% at four workers; peak memory equal or lower (one item
  per worker, no chunk buffers).
- Four-worker scaling: 3.0x-3.6x on four CPUs, as the old one, since the items share nothing but locked
  inputs.
- Sums are added in domain order, so values are identical at one and four workers and equal to live
  bit for bit.

Risk I see: allocation of a fresh item per policy under free-threaded malloc contention at four
workers; if it exceeds 3%, the reused receiver is the piece to measure back.
