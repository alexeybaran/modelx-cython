# Integer-enum labels on the fast line: report

Branch `r4/p3-labels` = `r4/p3-views` (18c6b03) + the commits below. Source: the full compiler at 633c9be
(`enums.py` and the enum hunks of its transformer, tracer, builder, admission, frame_reads, key_facts and
table_cells). A first port from 856b984 (the integer-literal pass) was dropped before it was pushed, when the
source changed. The expectation, `r4-p3-labels-expectation.md`, was written against that first port. Only its
"upstream text and SEC models must not move" part still applies, and that part holds.

Stack: CPython 3.13.18, Cython 3.3.2, pandas 3.0.5, numpy 2.5.3, modelx 0.33.0 (upstream), lifelib main
20d4dae. The machine has 4 vCPUs.

## Commits

| commit | kind | what |
|---|---|---|
| Map IntEnum definitions to C constants of the class module | transplant | `enums.py` (633c9be). Definitions are read-only and refused at load when redefined. Members become `cdef enum` constants in integer positions (comparisons, `is` with a typed value of the same class, `in` over member tuples, C-integer arguments and returns, member loops, member locals). Object membership compiles to a chain of identity and `==` tests. Table keys: a member key part is its constant (`lcode`), and member key locals are bound to constants. |
| Give an IntEnum parameter its class's member range as its domain | transplant | tracer `enum_args` / `_widen_to_enum`; `enum_domains` in the report |
| Give Python the live member of an IntEnum-valued Cell | transplant (wiring differs) | 633c9be's enum results: `enum_cells`, the object-use wrapping of `RenameMembers`, `enum_member`. Python callers get the member through `__getattribute__`. |
| Give live's value where an enum value held as a C integer is used as an object | fix | the cases of 1868bc0 and bae54ee, checked on the transplant; three of them differed from live (below) |
| Test that an IntEnum with its own str and format shows as live | fix (test) | cb7cd2c |
| Test that a class redefining an enum definition is refused at load | test | 4b96b21's test_enum_by_type case |

## The fixes: did the reference already behave as live?

- **cb7cd2c** (`enum_safe` also requires IntEnum's own `__str__`, `__format__`, `__repr__` and `__hash__`):
  - The check came in with the first commit, copied from the tracer of 856b984.
  - The new test also passes with the reference's weaker check (only `__eq__` / `__ne__`). The constants replace
    members only in integer positions, so `str()` of a member keeps the class's text there.
  - **The reference was already correct on this case.**
- **1868bc0 and bae54ee** (repr, type, isinstance, getattr, `!r`, `%`, stored keys, `.name`, identity):
  - Checked with the cases of 856b984's `test_enum_object_uses`.
  - **The reference was correct for:** repr, type, isinstance, getattr, `!r`, `%`, stored keys and identity
    (identity is lowered as `==` with a value of the same class; the reference's own identity test passes).
  - **It differed from live for:**
    - `.name` of a child Space's enum Cell (`O.pick().name`) and of a local bound to one: AttributeError.
    - `.name`, `str()` and `repr()` of a parameter traced as one class: AttributeError, or the integer's text.
    - `.name` of an integer that is no member's (`pick(-1)` returning `E.A - 1`): the reference raises
      ValueError from `cls(value)`; live raises AttributeError.
  - Fixed in the fix commit.
  - A member of one class passed to a parameter traced as another, and used there as an object, now stops the
    build naming the caller. Its class is lost in the C integer.
- **4b96b21:**
  - The refusal at load is brought.
  - Its `test_audit_runtime_b2` case (a class redefining HTTPStatus accepted) belongs to the rule by use: under
    the reference HTTPStatus is a definition and the redefinition is refused.
  - Its `test_diag_scale_int_items` case needs the native tuple loops of the reductions layer, which this base
    does not have.

## Deviations from 633c9be, with reasons

1. **The load check** `_mx_check_enum_schema` is a function emitted into the class module, not part of
   `_mx_sys.pxd`. Every model ships a copy of `_mx_sys.pxd`, so the SEC models would not stay byte-identical.
2. **`formula_lines` and `untyped_entry`** are copied into `enums.py`. They live in `reductions.py`, which this
   base does not have.
3. **Enum modules only.** The `Constants` pass and its report keys (`native_enum_*`, `native_expressions`,
   `expressions_untyped/object`) run only in modules with enum definitions.
   - The pass's object-membership chain would otherwise change the generated code of models without enums.
   - `native_expressions` counts enum memberships only; reductions, min/max and math are not in this base.
   - `enum_domains` is written only for Cells with an enum parameter.
4. **Members at the Python boundary.**
   - The reference renames each enum Cell's C member (`_mx_cell_<name>`, `table_cells.RenameMembers`) and adds a
     Python method that returns `runtime.table_cells.enum_member(...)`.
   - This base has no member renaming. `enum_results.py` keeps the reference's object-use logic and gives Python
     callers the member through `__getattribute__`, wrapping the integer arguments' entry where one exists.
   - `enum_member` is emitted into the model, because no compiled module imports `modelx_cython` at run time.
   - It returns the integer for a value that is no member's (fix commit).
5. **Not ported**, because the layers they serve are absent from this base:
   - the enum constants given to `NativeReductions`;
   - the `tuple_key` condition of membership.
6. **Tests:**
   - The `legacy_marker` case of `test_enum_by_type` is dropped: it needs `modelx_helpers`.
   - `test_native_enums` reads `native_enum_constants` with `.get`: the key is absent from models without
     definitions.
   - `test_enum_results` is split into a direct-route twin and its collect half.
   - The enum cases of `test_audit_contracts` are in `test_enum_audit_cases.py`.
   - Internal labels are removed from file names, comments and docstrings.
   - One earlier test, `test_constant_key_reads`, asserts the enum constant again, as in 633c9be. It was rewritten
     in slice 1 to the member-as-object form.

## Checks

- **Suite:** the full suite passes on the transplant head (bb76dad): 624 passed, 18 skipped. The suite of the fix
  head is in the section below.
- **Collect halves**, skipped for want of `set_collect`: `test_enum_parameter_domain_collect`,
  `test_enum_table_keys_collect`, `test_enum_results_collect`.
- **SEC models:** Term_US_SEC and MYGA_US_SEC give byte-identical generated files (sha256 of every
  .py/.pxd/.pyx/.json) on the base and on this branch.
- **Upstream-text models:** KLV_DE_S, Term_US_S, UC_FR_S, VA_US_S and IP_UK_S also give byte-identical generated
  files. Their values and run times therefore cannot move.
- **Probe model** (IntEnum members as table keys, parameters, comparisons, loop items, a member Reference, a
  member-returning Cell and `.name`): every value and result type equals live. On the base, `.name` of the
  member-returning Cell raised AttributeError, and the Cell returned `int` to Python.
- **Fix head (2fd55cf):** full suite 630 passed, 18 skipped.
- **Timing.**

| measure | base 18c6b03 | this branch | line B 856b984 |
|---|---|---|---|
| probe: new item, sum of `net_cf(t)`, item deleted (ms per policy, median of 3 interleaved runs) | 0.101 | 0.085 | 0.094 |
| Term_US_SEC translate (s, two runs) | 17.7 / 20.1 | 19.3 / 20.8 | - |
| MYGA_US_SEC translate (s, two runs) | 56.4 / 58.1 | 54.2 / 59.6 | - |

  - The run-to-run spread is wide on this machine (one probe run per tree was about 0.17 ms).
  - The SEC translate times are within that noise, and their generated files are identical.

- **Not measured:** speed against 633c9be. It needs a modelx with export protocol 2, so it does not build here.
