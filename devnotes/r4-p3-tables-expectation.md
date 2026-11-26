# Table reads on the fast line: what I expect, before measuring

Branch `r4/p3-tables`, on `r4/p2-collect-3` (90da50b). Written before the first build of a table model with the
ported code. Source of the port: `r4/integration` (8c03194), reconciled as `devnotes/r4-p3-reconcile.md` on
`review/r4-p3-reconcile` describes, with one change of scope made since: the read recorder (`read_notes.py`) is
not ported. Its two decisions (a read whose every sample evaluation raised, and a read that took a mask, list or
slice in the sample, stay on pandas) are replaced by what already decides without it: the text-only rule (a key
display that selects several rows stays on pandas) and the run-time single-label test of a key part
(`is_label`), which sends a read to pandas whenever the part is not one label. Membership (`k in T.index`)
gets the same run-time test, so an unhashable or several-rows key raises or answers as live does.

## What comes, in three commits (non-test lines)

| commit | pieces | expected lines |
|---|---|---|
| table base | runtime `_tables.pyx/.pxd` (preparation, slots per read signature, level codes, keyed reads of float64, int64 and object columns, numpy-scalar boxing); `frame_reads.py` (receiver by traced type, holds per item, keyed reads L1, key facts); `key_facts.py`; frame schema in the tracer, resolver, hold kinds, transformer and cli wiring, spec keys `prepared_tables` and `strict_lowering`; the C-number functions of `_tables.pxd` registered with the integer guard | 2,000 |
| held-row reads | the row of a key fixed per item held per item, fields read from the prepared blocks (`T.loc[k]["c"]`, `row = T.loc[k]` then `row["c"]`); no row-cell inliner | 350 |
| index extremes | `T.index.max()`/`.min()` (one level, several levels by part, by level), of a prefix of the whole table, and `k in T.index` with a complete key or a prefix | 450 |

About 2,800 lines in all, under the 3,600 asked for: the recorder (about 220 lines), the build reports and the
table-cell API of line B (renamed C entries, `TableCell` properties, prepared-at-bind publishing: about 300
lines) are not needed for speed. An item binds a Data frame by identity once per item instead of reading a
published field: one dictionary lookup per frame and item.

The runtime is copied into the model package as the numeric caches are (`native_sources.install`, a .pyx/.pxd
pair): the compiled model imports nothing from modelx_cython at run time, and a new model instance starts a new
generation of prepared tables by calling the copied runtime directly. No header-only install is needed in this
slice.

## Speed I expect (per policy: a new item, `result_cf()`, the item deleted; this machine, Python 3.13)

Measured before the port (line A today and line B, same machine, `result_cf()` of every policy of the shipped
book; noisy, the suite was running beside them): KLV_DE_S 72 ms line A, 0.74 ms line B; Term_US_S 74 ms line A,
0.63 ms line B.

- KLV_DE_S uses only forms of this slice (L1 15, L2 36, L5 5 of which three of a prefix, L7 1 on line B): I expect
  line B's time within 15%, about 0.7 to 0.85 ms, about 90 to 100 times faster than line A.
- Term_US_S has one walk (`iterrows()` of the shock lapse table, form L8, once per policy), which this slice
  leaves on pandas with its three row fields; everything else is in the slice. I expect 0.8 to 1.3 ms: line B's
  time plus one pandas walk per policy.
- Term_US_SEC and MYGA_US_SEC: generated code byte-identical (their pandas reads are all in zero-argument
  formulas of a Space without items, which run once per model and stay on pandas, or select several rows).
- Translation time: the frame schema is read from traces already kept, so I expect the translation of a table
  model within 10% of line A's.
