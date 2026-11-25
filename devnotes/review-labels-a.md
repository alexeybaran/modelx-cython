# Review of `labels` (IntEnum members as integer literals), reviewer a

Diff `3bc7be5..15454b7`. Stack: free-threaded CPython 3.14.6, Cython 3.3.1, modelx 636877b; one check on CPython 3.10.20.
Every case marked **built** was reproduced by building one small model and comparing the exported model with the compiled
one in separate processes. **pass** means reproduced on the output of `enums.literals` only. Enum classes used below:
`Leg{CALL=0,PUT=1}`, `Kind{X=0,Y=1}`, `Pretty{M=0,F=1}` with its own `__str__`/`__format__`, `Ord{LOW=2,HIGH=1}` with its own
`__lt__`. Formulas are written as the model author writes them. `devnotes/review-labels-a.patch` (57 lines added, 13
removed, applies to 15454b7) holds the fixes marked **patch**. With it `test_enum_literals.py` and `test_tracer.py` give
117 passed, 1 failed (it asserts a build error the patch removes on purpose, `Kind.A.real`). The nine compiled enum test
files with `test_audit_contracts.py` gave 148 passed, 2 failed: one asserts a build error the patch turns into a
live-equal result (`pretty_param`); one imports `modelx_helpers`, which is not in the brief's stack. That compiled run
was made before the last three small parts of the patch (G4, the attribute of a member in place, the 3.10 constant);
those were checked on the unit tests and on the pass output only. The patch was not run on the 53 models.

## 1. What the code does

| # | Mechanism | Where | Lines | Lost without it |
|---|---|---|---|---|
| 1 | Substitute `self.<Ref>.<member>` by its integer, before parsing for any other pass | `enums.py:66-86`, `204-210`, `314-320`; `cli.py:349` | 35 | Measured on a micro model (three-way `if k == Kind.X` chain, `k in (...)`, about 1.3 million Cell evaluations): 0.0126 s with the pass, 0.248 s with it switched off: **20x** |
| 2 | Scope tracking: which `self` is the Space, which names are shadowed | `enums.py:116-202` | 87 | Wrong substitution under a rebound or nested `self` |
| 3 | Displays keep members unless iterated in place, right of `in`, or a subscript key | `enums.py:283-312` | 30 | `leg.name` after a tuple of members travels (RILA_US_S `result`, 1 site) |
| 4 | Literal integer tuple in a native sum loops as a C integer, a `range` when consecutive | `reductions.py:48-74`, `1180`, `1228-1229`, `1244` | 35 | Measured on `sum(claims(t, k, s) for k in (A, B, C))`: 0.0256 s against 0.0525 s: **2.05x**. 41 such sites in the 57 lean models. Replaces the 44-line `enum_range` |
| 5 | `.name` by a `{value: name}` table and `.value`, for members, parameters and Cell results traced as one class, and locals bound only to members | `enums.py:89-113`, `154-175`, `212-234`, `321-339` | 75 | `phase(t).name` in FIA_US_S (1 site). No lean model reads `.name` of a parameter or local, or `.value` |
| 6 | Build errors: class shows (`repr`, `type`, `%r`...), other attributes, class mismatch at a call | `enums.py:236-263`, `331-333` | 32 | Silent differences of the first review |
| 7 | Build error on identity tests | `enums.py:265-281` | 17 | Same. No lean model has an identity test other than with `None` |
| 8 | Definition is read-only in the compiled model | `enums.py:18-62`, `_mx_sys.pxd:217-229`, `transformer.py:321,622,1209,1261` | 62 | Rebinding the Reference would change displays but not literals |
| 9 | "Enum-safe" test; argument domain widened to the whole class | `tracer.py:557-561`, `317-332` | 21 | Substituting a class with its own operators; dense cache only over sampled members |

Total about 395 lines. Rows 1, 2, 3, 4 (187 lines) carry the speed. Rows 5 to 8 exist only because values are then held as
integers.

## 2. Complexity

| Cut | Lines saved | Cost | Risk | Verify |
|---|---|---|---|---|
| A. Replace `.name`/`.value` of an integer-held value by a build error that says "write `Kind(x).name`" (works today: `self.Kind(k).name` is left alone). Removes the table, `_traced_call` result names, the class-mismatch error and the return check of 4.6 | about 25, and three sample-dependent defects of section 4 | One line of model text: FIA_US_S `phase(t).name` becomes `Phase(phase(t)).name`. RILA is untouched (objects from a tuple) | Low. Not built | FIA_US_S and RILA_US_S |
| B. `_Table.plain` and `through_str` become one interpreter constant once arguments of a non-enum-safe class stay objects (4.5, patch) | 8 | None | Low. Not built | `test_enum_object_uses.py` |
| C. One `CstFacts` per method instead of two (`enums.py:141` and `156`) | 0 | Pass time on a synthetic 3,600-line module with a member in every method: 4.93 s to 3.70 s, same output (measured by sharing the object); parsing alone is 0.60 s | None | Any model: same generated text |
| D. `_SHOWN` half for members (`kept`) and `visit_AssignTarget` go away under the redesign of 4.9 | about 12 | See 4.9 | Redesign, not built | Ten models with member arguments |
| E. Build errors reach the author as a 40-frame libcst traceback ending in `TypeError`; catch in `cli.py` and print the one line | +3 | None | None | Any failing case of `test_enum_object_uses.py` |

Nothing above 25 lines can go. Rows 1 to 4 duplicate nothing in Cython or CPython. Row 8 was not challenged by experiment.
`found.update(enumerate(...))` (`enums.py:112`) keys parameters by position in the same dict as names; correct, hard to read.

## 3. Generic

| Finding | Case | Status |
|---|---|---|
| G1. The rule is one spelling: `self.<Ref>.<member>` with the class in a Reference of the same Space | `enums.Leg.CALL` (a module or a Space as namespace), `E = Leg; E.CALL`, `Leg["CALL"]`, `Leg(1)` stay objects. Correct, but none of the speed of row 1 | By design; say so in the release notes |
| G2. Upstream already holds a parameter or result traced as any `int` subclass as a C integer (`tracer.py:190,233` at d22426f). So "a Cell that returns a member returns its integer" is not new | `lambda: Leg.PUT` returns `1` upstream too | Not a regression |
| G3. New for an unmodified upstream model: every case of 4.2, 4.4, 4.7, 4.8, 4.9 worked upstream, because a member written in a formula was an object there | See section 4 | Blocking by the brief's rule |
| G4. Identity error fires on any `a is b` in a module that has an enum Reference in any Space | `type(x) is np.ndarray`, `x is not pd.NaT`, `x is NotImplemented`, `x is self.sentinel`: all stop the build (pass), with the advice "compare with ==", which is wrong for the first three | patch exempts `type(...)`, `module.attr`, `NotImplemented`, `Ellipsis`; `x is self.sentinel` and `a is b` still stop |
| G5. `.name` by table trusts the sample for the class (4.6) | | See cut A |

Would the build errors annoy an author writing ordinary Python? In the lean text of the 57 models: no site hits any of
them. For another author, ranked by how ordinary the spelling is: (1) identity, G4; (2) a method or property the author
put on the enum (`Leg.CALL.sign`): stops the build although a member written in place could simply stay an object (patch);
(3) `k.bit_length()` and other `int` attributes stop the build although the integer gives the same answer (not patched);
(4) `isinstance(kind(), int)` stops the build although both sides give `True`. The other errors (`repr(k)`, `type(k)`,
class mismatch) are rare spellings and the messages name the Cell and the expression. They are acceptable.

## 4. Reliable

Silent means compiled returns a different value with no error. All were built unless marked.

| # | Formula (sample reaches it unless said) | Live | Compiled | Kind | Fix |
|---|---|---|---|---|---|
| 4.1 | `lambda: Ord.LOW < Ord.HIGH`; also `lambda k: k < Ord.HIGH` | `True` | `False` | Silent. `enum_safe` (`tracer.py:557-561`) checks `==`, `!=`, `str`, `format`, `repr`, `hash` only; ordering, arithmetic, `__bool__`, `__index__`, and a `_value_` that is not the integer are not checked | patch: every dunder of `int` plus `__str__` must be `IntEnum`'s, and `m.value == int(m)`; 9 lines. Verified on 3.10 and 3.14: plain classes pass, `Ord`, `Pretty` and a class with a text `_value_` fail |
| 4.2 | `lambda: [repr(x) for x in (Leg.CALL, Leg.PUT)]`; `[isinstance(x, Leg) for x in (...)]` | `['<Leg.CALL: 0>', ...]`; `[True, True]` | `['0', '1']`; `[False, False]` | Silent. A comprehension variable over a display iterated in place is an integer but is not known as an enum value (`_member_locals`, `enums.py:154-168`, sees only `=` and `for` statements over tuples) | patch: type comprehension variables and `for` over list and set displays; 12 lines. Both become build errors |
| 4.3 | `lambda: [x.name for x in (Leg.CALL, Leg.PUT)]`; `for x in [Leg.CALL, Leg.PUT]: out.append(x.name)` | `['CALL', 'PUT']` | `AttributeError: 'int' object has no attribute 'name'` | Raises where live works; same cause as 4.2 | Same patch: both equal live |
| 4.4 | `lambda: "{!r}".format(Leg.CALL)`; `lambda k: str((k, 1))`; `lambda s: "sex={}".format(s)` with `s` a `Pretty` | `'<Leg.CALL: 0>'`; `'(<Leg.PUT: 1>, 1)'`; `'sex=F'` | `'0'`; `'(1, 1)'`; `'sex=1'` | Silent. `_SHOWN` (`enums.py:104`, `245-247`) knows `repr(x)` and `%`, not `str.format` and not a display under `str`/`repr` | patch: 10 lines. First equals live, second is a build error, third equals live by 4.5 |
| 4.5 | `lambda s: "sex=" + str(s)` with `s` a `Pretty` is a build error today; `lambda k: k < Ord.HIGH` is silent | | | A parameter traced as a class that is not enum-safe is still typed as a C integer (upstream behaviour, `tracer.py:295-301`) | patch, 3 lines: such an argument is `object`. Built: `.pxd` shows `object k`; `str(s)`, `s.name`, `k < Ord.HIGH` equal live; a safe class keeps `long long k`. The result side (`lambda t: Pretty.F`) is not patched |
| 4.6 | `nm = lambda k: k.name`, sample calls `nm(Leg.CALL)`; `cross = lambda: [nm(x) for x in (Kind.X, Kind.Y)]` not sampled. `pick = lambda t: Leg.PUT if t > 0 else Kind.X`, `pick_nm = lambda t: pick(t).name`, sample calls `pick_nm(1)` | `['X', 'Y']`; `pick_nm(0) == 'X'` | `['CALL', 'PUT']`; `'CALL'` | Silent, and decided by what the sample reached: the name table is the traced class's (`enums.py:107-113`), the class check sees only arguments with a known class (`enums.py:249-253`) | patch: `cross` becomes a build error through 4.2; a `return` of another class's member is a build error (11 lines). A value of another class arriving by a route the pass cannot see stays wrong. Cut A removes the cause |
| 4.7 | `un = lambda k: k.name`, `call_un = lambda t: un(Leg.PUT) if t > 0 else ""`, sample calls `call_un(0)` only | `call_un(1) == 'PUT'` | `AttributeError` | Raises where live works, decided by the sample: an untraced Cell is not rewritten, its caller passes `1` | 4.9 |
| 4.8 | In a child Space with `par` the parent: `lambda: par.legs()[0] is par.first()`, `legs = lambda: (Leg.CALL, Leg.PUT)`, `first = lambda: Leg.CALL` | `True` | `False` | Silent. The identity error is per module (`self.members`, `enums.py:277`); a child Space is another module and has no enum Reference of its own | Not patched. Take the condition from all modules: 2 lines, not built |
| 4.9 | `label` a function Reference `def label(leg): return leg.name`: `lambda: label(Leg.PUT)`. `out.append(Leg.CALL)` then `[x.name for x in appended()]`. `Sub[Leg.PUT].val()` with `Sub` an item Space, `val = lambda: leg.name`. `x = Leg.PUT if t > 0 else None; x.name` | `'PUT'`, `['CALL']`, `'PUT'`, `'PUT'` | `AttributeError` in all four; after the third, `Sub[Leg.PUT].val()` from Python fails too, because the item Space keyed `1` is reused | Raises where live works. A member handed to code the pass cannot see arrives as a bare integer | Redesign, not built: substitute only where the consumer is known to take an integer (operand of a comparison or of arithmetic, argument of a traced Cell whose parameter is an integer or that class, element of an in-place display, subscript key) and keep the member elsewhere. The hot forms of the lean text are all on the substituted side (41 loop displays, comparisons, Cell arguments; no member is passed to a method in any lean model), so no speed change is expected. Not measured |
| 4.10 | On CPython 3.10: `lambda: str(Leg.PUT)`, `lambda k: "leg=" + str(k)`, `lambda k: "%s" % k` | `'Leg.PUT'`, `'leg=Leg.PUT'`, `'Leg.PUT'` | `'1'`, `'leg=1'`, `'1'` | Silent, by Python version (built on 3.10.20). `plain` (`enums.py:99`) compares methods with `IntEnum`'s, and on 3.10 `IntEnum.__str__` shows the class | patch, 4 lines: `plain` by behaviour (`str(m) == str(int(m))`) and a member under `str` stays an object before 3.11. Checked on the pass output on 3.10 and 3.14, not built. On 3.10 `str(k)` then stops the build, which hits four `raise ValueError("invalid timing: " + str(timing))` in EC_FR_S, and `f"{k}"` stops it too although `format` gives the integer there. Exempting the operand of `raise` would keep that model building: not built |

Checked and found correct: the fixes of the first review for `repr`/`%r`/`!r` of a member, `.name` of a non-member integer
(raises `AttributeError` as live), a stored dict key, `.name` of a child Space's Cell result, and the 64-bit guard of the
literal-tuple sum (`_in_arithmetic`, `reductions.py:67-74`). A negative literal index `sum(v()[i] for i in (-1, 1))` equals
live and an index out of range raises `IndexError` as live (built). Nothing found that depends on thread timing, call
frequency or the operating system.

## 5. Table split

Not a table area.

## 6. Bottom line

The decided design is sound in its core and the measured speed is real: substitution is worth 20x and the C loop over a
literal tuple 2x on micro models, in about 190 lines. The weak point is not the size but the boundary: the same value is an
integer or a member depending on the route it took, and ten shapes of ordinary Python cross that boundary unseen. Six of
them are closed by the patch (4.1 to 4.6, 4.10); 4.8 is two lines; 4.7 and 4.9 need the narrower substitution rule.

| | Version | Lines |
|---|---|---|
| Now | 15454b7 | about 395 |
| (a) keeps the measured speed | Now, plus the patch (+44), minus cut B | about 430 |
| (b) speed within 25-30% | (a) with cut A | about 405 |

No version is materially smaller: dropping row 1 costs 20x, dropping row 4 costs 2x on its shape, and rows 5 to 8 are what
keeps integer-held values equal to live.

Blocking: 4.1, 4.2, 4.4, 4.6, 4.8, 4.10 (silent); 4.3, 4.7, 4.9 (an unmodified upstream model that worked now raises).
