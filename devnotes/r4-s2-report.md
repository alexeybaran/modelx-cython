# R4 S2: trial cut of layer 1

Branch `r4/s2-layer1-trial`, cut by deletion from `r4/s1-zeroing` (3bc7be5) and diffed against upstream
`d22426f`. Runs on upstream modelx main (fec516d, 0.33.0) and upstream lifelib main (20d4dae), Python 3.14t.

## Expectation (as set in the brief, unchanged)

It builds; all twelve SEC models match live modelx (rel 1e-10, abs 1e-7); instructions per policy of
Term_US_SEC and MYGA_US_SEC equal the full tree's within 1%.

## Results

@@RESULTS@@

## How it was measured (gate tool not used)

The gate toolkit could not be read: attaching `modelx-benchmark-toolkit` was refused by a permission check,
so `RUN.md` and the gate were not used. Development modelx 636877b is not reachable either. Instead:

- Build: copy the product folder, `mx.read_model` + `export` with upstream modelx, `mx2cy` with the product's
  `sample_sec.py` and `spec_sec.py`.
- Values: on live modelx, every no-argument cell, then every `t` cell for `t = 0 .. proj_len()`, then every
  cached value at every argument (lifelib's `full_sweep` + `_cached_values`); the same calls on the compiled
  package in a child, 256 MiB thread stack. rel 1e-10 / abs 1e-7; kinds must match; a raise must meet a raise
  of the same type (message not compared).
- Instructions per policy: callgrind collecting only inside the compiled `_mx_new_root__c_Projection`,
  `_c_Projection._f_result_cf` and `_mx_release_item__c_Projection`; a pass builds every point, calls
  `result_cf()`, deletes it; ipp = (I(4 passes) - I(1 pass)) / (3 x points), so first-pass reads cancel.
  Whole-process counts were 1-2% noisy run to run (free-threaded interpreter), this window is not.
- Full tree on the same machine: 3bc7be5 refuses an upstream export (`collect_adapter.require_current`:
  export protocol 2). For the baseline that one check was disabled in the installed copy only; nothing else
  changed. With an upstream export no collect is generated, so this is the full tree's non-collect path;
  its table and vector providers were active (Term built `_tables` and `_native_vectors`).

## Lines (non-test, .py/.pyx/.pxd under modelx_cython)

@@LINES@@

## Kept outside the layer-1 list

@@KEPT@@

## Dependencies on modelx behaviour not in upstream modelx

@@MODELX@@

## Configuration keys left

@@KEYS@@

## Tests

@@TESTS@@
