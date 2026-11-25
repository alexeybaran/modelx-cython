# Handover: enum members held by use (branch `review/r4-labels-a`)

Base: `r4/integration` 09cbd68 merged at 7cb9a14; 18 commits after it. Stack: CPython 3.14t, Cython 3.3.1, modelx 636877b.

## Design
- `enums.plan(...)` runs once for all modules, first (`cli.py`); per formula `_Uses` marks contexts, `_Rewrite` applies them.
- IntEnum (enum-safe): the integer is the member's value. Integer contexts: operand of a comparison (not `is`) or of
  arithmetic, truth test, subscript that is read, numeric builtins, argument of a Cell whose parameter is a number,
  `return` of a numeric Cell, a display iterated by a loop and a local, when the variable has integer uses only.
- StrEnum (text-safe, 3.11+): the integer is the position by declaration order. Integer contexts only: `==`, `!=`, `in`
  against members or text literals, argument of a parameter held as a position, loops and locals as above.
  A text literal there becomes the position (-2 when no member has it). Cell results stay objects.
- Every other use sees the member: a member written there stays; a held value is converted
  (`_mx_enum_of` by value, `_mx_enum_at` by position; a non-member integer stays that integer).
- Held values: a parameter traced as one class, a Cell result (IntEnum only), a Reference or item-Space parameter
  whose value is an IntEnum member. The class is reached as `self.<Ref>` or, when no Reference holds it, by import.
- Class certainty: a parameter or result that some formula converts back ("needy", closed over flows across modules)
  gets `_mx_enum_member` around any value whose class the text does not show; the Python call gate
  (`integer_boundaries.py`) does the same for Python callers and converts member or text to position (`_mx_enum_code`).
- Tracer: `enum_safe` and `text_safe` compare every special method; a class with its own operators is traced as an
  object (arguments, results, References); `enum_codes` marks parameters held as positions (typed `int`, 0..n-1).
## Done and passing
- `test_enum_literals.py` 101; unit suite 695 passed (1 needs the `lifelib` package); compiled enum and integer-boundary
  files 52 passed (`test_enum_uses.py` holds both reviews' cases and the Child_KR_S and VA_KR_S shapes;
  `test_enum_text_values.py` the StrEnum family).
- Micro models: formula code identical to 15454b7's; only the Python call gate text changed. 0.0134 s and 0.0276 s
  against 0.0126 s and 0.0256 s before (within noise, not a strict 3% proof). Pass time on 3,600 lines: 2.15 s (was 4.93 s).
## Not done
- Not run on any lifelib model, nor on Python 3.10 or 3.11 after the rewrite.
- StrEnum Cell results are objects (not positions): the gate cannot tell a Python caller from a Python-dispatched
  compiled call, so a result could not be converted back safely. Cost not measured.
- An integer given from Python to a position parameter is taken as a position. Mixed traces (members and plain integers, or two classes) are typed as before (`Integral`, or `str`): `.name` there
  raises as an integer does. A parameter rebound to another class and then read as a member is a build error.
- Size: `enums.py` 339 -> 842 lines (+748/-245); `tracer.py` +28/-8, `_mx_sys.pxd` +38, `integer_boundaries.py` +23/-4,
  `cli.py` +5/-3. The text-valued case is about 150 of these. No trimming was attempted.
## Not equal to live
- Raises `TypeError` where live gives a value: a member of another class, or a plain integer equal to a member, given
  to a needy IntEnum parameter or returned by a needy Cell.
- Raises `ValueError`: a text that is no member's value given to a position parameter.
- Gives a value where live raises `AttributeError`: a text given from Python to a Cell that reads `kind.name`.
  A Cell returning an IntEnum member returns its integer (as upstream).
