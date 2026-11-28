"""What counts as rebinding a formula's names, for the passes that need a formula without scope effects.

The table pass wraps a hold in ``try: ... except Exception: ...`` and evaluates a key part once with
``_mx_... := value``; neither rebinds a name of the formula, so neither stops the reduction, rounding or row passes.
A named handler (it deletes its name), a walrus of a formula name, and the other effects still do.
"""
import libcst as cst
import pytest

from modelx_cython.scopes import has_scope_effects


def _body(source):
    return cst.parse_module(source).body


@pytest.mark.parametrize('source', [
    'try:\n    h = 1\nexcept Exception:\n    pass\nx = max(a for a in v if a <= 2)\n',
    'y = (f(k) if g((_mx_okey_1 := k)) else 0)\n',
])
def test_generated_handlers_and_walruses_are_no_scope_effects(source):
    assert not has_scope_effects(_body(source))


@pytest.mark.parametrize('source', [
    'try:\n    h = 1\nexcept Exception as e:\n    pass\n',
    'y = (n := 3)\n',
    'global g\n',
    'del x\n',
    'f = lambda a: a\n',
])
def test_formula_bindings_are_scope_effects(source):
    assert has_scope_effects(_body(source))
