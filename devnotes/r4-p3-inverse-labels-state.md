# Label fixes on the inverse line: working state

Branch `r4/p3-inverse-fix-labels`, cut from `r4/p3-inverse` c342db9 (proposal line + the compiler tree of 633c9be).
Task: re-apply the label-area fixes as separate commits with their tests; touch only label/enum files and tests.

Plan (one commit each, full suite after each, push, then one line to the measuring session):
1. cb7cd2c: tracer `enum_safe` also requires IntEnum's own `__str__`/`__format__`/`__repr__`/`__hash__`, with a test
   of a class with its own `__str__` compared with live.
2. Object uses of enum values (the cases of 1868bc0 / bae54ee), done against this tree's `table_cells.RenameMembers`:
   `.name` of a child Space's enum Cell and of a local bound to one, `.name`/`str`/`repr` of an enum parameter,
   `.name` of an integer that is no member's (live AttributeError, not ValueError). Test: the cases of
   test_enum_object_uses, compared with live.
3. 4b96b21: test that a class redefining an enum definition is refused at load; its other cases if they pass here.

Done so far: (update as commits land)
