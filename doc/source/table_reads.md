# Table reads in compiled models

**Status:** this page describes the target of the table redesign (steps T-1 to T-7). Until step T-3, compiled models
still use the older table layer, and the probe test lists where it differs from live modelx
(`integration_tests/test_table_probe.py`, `KNOWN`).

## Rule

The compiler makes the reads it recognises fast by preparing typed arrays from a table when the table is bound.
Every other operation runs on the real pandas object. A formula always sees the user's own DataFrame or Series; no
other object stands in for it, at build time or at run time.

## Reads made fast

A read is recognised by the traced type of its receiver (a pandas DataFrame or Series, from any source) and by the
pandas operation, never by how it is spelled:

| form | read |
|---|---|
| L1 | `.loc[key, col]` / `.at[key, col]` with a full key (flat or MultiIndex); a Series `.loc[k]`, `.at[k]`, `[k]` |
| L2 | `.loc[key]` (a row) read only by field, also across cells (the model-point row) |
| L3 | a partial-key `.loc[prefix]` followed by a keyed read |
| L4 | a column by name followed by a key, `T["c"][k]` |
| L5 | `.index.max()`, `.index.min()` and their parts |
| L6 | `.index.asof(x)` |
| L7 | `key in T.index` |

A read is made fast only on a table with a unique index, with key parts whose compiled types match the index
levels, and with a column label known at build time. Anything else is left to pandas.

## The build report (`_mx_table_reads.json`)

One entry per cell that reads a table, one record per read site:

```json
{
  "Calc.premium": [
    {"line": 3, "expression": "T.loc[age, 'q']", "receiver": "Data.mort_table", "form": "L1"},
    {"line": 4, "expression": "T.loc[T['q'] > x]", "receiver": "Data.mort_table", "form": "pandas",
     "reason": "a boolean mask: not a recognised read"}
  ]
}
```

- `form` is one of `L1` to `L7`, or `pandas` for a read left to pandas.
- `reason` is given for every `pandas` entry: the operation is not a recognised form, or the index is not unique,
  or a key type does not match a level, or the column is not known at build time.
- The report states what the build decided and why. It does not claim that a result equals live modelx: that
  claim is checked by the tests, not by the build.

## Input changes

Compiled models follow live modelx:

- assigning a new frame to a Data cell (`Data.T[()] = frame`) replaces it at once: no prepared copy of the old frame
  is used afterwards;
- results already cached stay cached until the dependent Space is cleared (`Space.clear_all()`), and are then
  recomputed from the new frame;
- `Model.clear_all()` discards assigned inputs (the cell recomputes its formula), as on live modelx;
- an in-place edit of a bound frame is applied by clearing the dependent Space, as on live modelx. Before the clear,
  reads made fast use the copy taken when the table was bound. An opt-in check mode (spec key `verify_inputs`,
  step T-4) detects the edit and raises.
