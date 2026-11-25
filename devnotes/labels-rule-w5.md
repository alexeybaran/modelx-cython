# Enum members by use: verified and shrunk (branch `r4/w5-labels`)

Base: `review/r4-labels-a` e3dba9c. Stack: CPython 3.14.6t, Cython 3.3.1, modelx `r4/modelx-integration` df3ced9.
Target set before starting: `enums.py` under 450 lines with no tested behaviour lost. Result: 842 -> 438 lines.

## Verify (on e3dba9c)
- Suites: enum unit tests 101; enum and tracer units on Python 3.11, 136; compiled enum and gate files 32.
  The unit suite gave 719 passed, 6 failed; all 6 failed only because `lifelib` was not installed.
- Translate-only, exit 0: IP_UK_S, KLV_DE_S, FIA_US_S (lifelib `r4/l1-lean-b`, StrEnum, enum module stored with
  the model); VA_KR_S and Child_KR_S (04d69db, IntEnum); Child_KR_S (`r4/l1-lean`).
- Two defects found in that text, both reproduced on a compiled model (`test_enum_stored_module.py` fails on
  e3dba9c, passes now):
  1. The call gate imported the class by module name. A module stored with the model is `'<unnamed module>'`,
     so every Python call to a Cell with an enum parameter raised `ModuleNotFoundError`. That is 4 Cells in
     KLV_DE_S, 13 in FIA_US_S and 11 in Child_KR_S.
  2. `claims(t, kind=None)` (KLV_DE_S): the builder made `kind` an object, but the pass held it as a position.
     As a result `claims(t)` and `claims(t, Kind.X)` raised `TypeError` from Python. In IP_UK_S the sample
     passes `None`, so no family is held there at all.

## The rule now (one path for both kinds)
- A *family* is an IntEnum whose values are 0..n-1, or a StrEnum, whose `==`, `!=` and `__hash__` are the
  base's. All 637 IntEnum classes in both lifelib trees are 0..n-1. A member stands for its position, which for
  an IntEnum is its value.
- Only a Cell parameter is held: one traced as members of one family (or values equal to members), in a Space
  with a Reference to the class. It is an integer in three uses: `==`, `!=` and `in` against members, values or
  a held name of the class; as an argument of a parameter held for the class; and in a loop over a display of
  members whose variable has no other use. Every other use is `_mx_enum_at(self.<Ref>, k)`.
- Values from elsewhere pass `_mx_enum_code(cls, v, strict, 'Space.cell')` (formulas and the Python gate). A
  parameter is *strict* if a use shows the member's class, or if it gives the parameter on to a strict one.
  Uses that show the class: an attribute, a call argument, `is`, storing, returning. Uses that don't:
  operands of arithmetic and of comparisons other than `is`, truth tests, f-string fields, read subscripts.
  Strict parameters take members only; other parameters also take positions and values equal to a member, as
  modelx compares them. Anything else raises `ValueError` naming the Cell.
- A parameter is released to an object in five cases: it is bound again; it is seen where `self` is another; it
  has a default that is no member's value; its formula is uncached and has a closure (a Python method with
  untyped parameters); or it is strict while the sample also gave it plain values. Cell
  results, Reference values and item-Space arguments that are members stay members (tracer).
- The gate reaches the class as `getattr(space, Ref)`, so no import is needed.

## Why it was 842 lines, and what went
Two holding schemes: IntEnum by value (any values, results and References too) and StrEnum by position. The
broad integer rule (arithmetic, subscripts, numeric builtins, truth tests, `match`, numeric results) needed
class certainty for every held value. That meant `needs`, `flows`, `forced` and `checks`, with a fixed point
across modules, and `_mx_enum_member`. Locals bound to members had their own tracking, and a hand-written
scope tracker was about 150 lines. Replacements: one position path; the three uses, where each held use is
either an integer or converted on the spot; strictness as the only cross-Cell fact (a short fixed point); and
libcst `ScopeProvider` and `ParentNodeProvider` instead of the scope code. `_mx_sys.pxd` went from four enum
functions to two.

## Speed
The previous session's micro models are not in the repository. I rebuilt them from the review's description
(`devnotes/labels-micro-bench.py`: a three-way `==` chain with `in`, about 1.3 million evaluations; and `sum`
over a literal tuple of members). Fresh processes, alternating 15454b7 and this branch, 25 runs each:

| model | 15454b7 min / median | this branch min / median | ratio min / median |
|---|---|---|---|
| micro1 | 0.3999 / 0.4281 s | 0.3958 / 0.4354 s | 0.990 / 1.017 |
| micro2 | 0.8426 / 1.0059 s | 0.8662 / 0.9687 s | 1.028 / 0.963 |

The generated formula code of both is identical to 15454b7's; only the Python call gate differs.
Lost: D31 (a Reference holding a member, compared in a loop through a local alias, SPIA_US_S
`commuted_value`) is a Python comparison again. On e3dba9c it applied to IntEnum only, never to StrEnum
text. Measured on its own micro model (a 20-step loop comparing a member Reference through a local, 4 million
comparisons): 0.058 s on 15454b7, 0.566 s here, about 10x. Restoring it means holding References and locals bound
to them again, which the old code did in about 100 lines; not done.
In the text going forward (lifelib `r4/l1-lean-b` 1956487, all 56 StrEnum models) no Reference holds an enum
member. A scan loaded each model and checked every formula for a member Reference compared directly or through a
local, and found none. SPIA_US_S's `commute_disc_convention` is now the text `"compound"`, compared with text
literals. So D31 has no instance there and stays lost.

## Behaviours not equal to live
- A value that is no member's value, given to a held parameter, raises `ValueError` at the named Cell where
  modelx may give a value (`claims(1, "NOPE")`, live `0.0`).
- A strict parameter given anything but a member of its class raises `ValueError` at the named Cell. This
  covers a text, a member of another class, a float, and a plain integer. Live then gives a value, raises
  `AttributeError`, or returns another call's cached value: modelx keys its cache by `==`.
- A value equal to a member, given to a parameter that is not strict, is the member in the uses that don't show
  the class. Arithmetic, comparison, truth and f-string give the same; so does a read subscript, as long as
  the container doesn't key the member and the text apart.
- An integer or a bool given to a StrEnum parameter raises `ValueError` at the named Cell (it is equal to no
  member; live takes the non-matching branch). The cost: a Python subclass of a compiled Space, which passes
  positions through the gate, raises there too. For an IntEnum an integer or bool is its value, equal to live.
- Gone from the previous list: a text read as a member (now a clear error), a Cell returning an IntEnum's
  integer (it returns the member), and `TypeError`s without the Cell's name.

## Tests edited or replaced (reasons)
- `test_enum_literals.py`: rewritten for the rule, 101 -> 81 tests. It keeps: members in object uses stay;
  comparisons, `in`, loops and arguments become positions; held values are converted in every other use;
  shadowing; text-valued families; text defaults; the tracer's decisions. Removed with the old rule:
  arithmetic, subscript, builtin, `match`, result and local integer cases; import of the class; item-Space
  conversion; the rebound-parameter build error (now an object); `enum_safe` and `text_safe` (now `family`).
  Added: strictness and its propagation, the gate table, releases (default, rebinding, other `self`, star
  call, no Reference, plain values), and `-1` for a literal of no member.
- `test_audit_contracts.py`: two cases now use a family (OK=0, FAILED=1) with a held parameter. Before, the
  literal came from a non-family class compared with an untyped value. Each keeps its intent.
- Compiled tests, where every probe still compares with the exported model:
  - `test_enum_uses`: `pick_nm(0)` now equals live and moved to the probes. The two remaining raises are
    `ValueError` (was `TypeError`).
  - `test_enum_text_values`: a text given to a Cell that reads `.name` or tests `is` now raises `ValueError`.
    Before, `nm("LAPSE")` gave a value where live raises, and `ident("MATURITY")` equalled live only by luck
    of the cache.
  - `test_enum_object_uses`: `TypeError` -> `ValueError`.
  - `test_enum_results`: results are members, not ints.
  - `test_enum_by_type`, `test_enum_containers`, `test_diag_scale_enum_conditionals`: classes changed to
    0..n-1 so the pass still runs; generated-text assertions state the new forms.
  - `test_native_enum_membership`, `test_native_enums`, `test_enum_reference_compare`: `HTTPStatus` is no
    family. Members stay and the Reference is ordinary, so the substitution, read-only and `long long`
    assertions were replaced by their new forms.
- New: `test_enum_stored_module.py`. It covers a stored module, a `None` default, a cached and an uncached Cell
  with a closure (the uncached one was silently wrong before the release), and texts, ints and floats from
  Python.

## Results on this branch
- Unit suite on 011cfeb (`lifelib` and `psutil` installed): 694 passed, 2 failed, 8 skipped. Both failures
  were from `psutil` missing for `benchmark.py`; with it, both pass. On the last commit, the enum, tracer and
  audit units (130) and both lifelib sample builds pass: 132 passed, 2 skipped. On Python 3.11: 130 passed.
- Compiled enum and gate files on the last commit: 45 passed.
- Translate-only on the last commit, exit 0: IP_UK_S 38 s, KLV_DE_S 64 s, FIA_US_S 84 s, Child_KR_S (l1-lean)
  116 s, VA_KR_S (04d69db) 37 s, Child_KR_S (04d69db) 121 s. Held parameters (strict): IP 0, KLV 3 (0),
  FIA 13 (0), Child 14 (11), Child at 04d69db 15 (6). No gate imports a module.
  - IP_UK_S holds nothing: its only family parameter, `claims(t, kind=None)`, takes `None`.
  - VA_KR_S at 04d69db is thinly traced: its shipped sample calls `collect_outputs`, which this text lacks
    (I made the call a no-op), and five deep Cells raise `RecursionError` in the sample. So only translation
    is shown there.

## Not shrunk, and why
- `prepare` and `bind_properties` (about 45 lines): the read-only definition is needed whenever members are in
  the code.
- Callee resolution across Spaces (about 20 lines): the transformer calls `self.<Space>.<cell>` directly, so
  an unconverted member would reach a C integer.
- `integer_boundaries.py` skips Cells with closures (`integer_parameters`), so they get no gate. The new test
  covers such a Cell (`scaled`) and passes.
