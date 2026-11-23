from types import SimpleNamespace

import libcst
import pytest

from modelx_cython.factory_reuse import scalar_factory_graph, unshadowed_builtins


@pytest.mark.parametrize(("formula", "admitted"), [
    ("return self.base() * self.point", True),
    ("return id(self)", False),
    ("return self.base", False),
    ("return self._space", False),
    ("return self.callback(self.point)", False),
    ("return getattr(self, 'point')", False),
    ("self.point = 2\n        return self.point", False),
    ("return self.data.loc[self.point]", False),
])
def test_receiver_isolation(formula, admitted):
    node = libcst.parse_module(f"""
class Product:
    def _mx_space_formula(self, key):
        return {{'refs': {{'point': key}}}}
    def _f_balance(self):
        {formula}
    def _f_base(self):
        return 1.0
""").body[0]
    numeric = SimpleNamespace(get_type_expr=lambda c_style: 'long long')
    cls = SimpleNamespace(cells={'balance', 'base'}, refs={'point': numeric})
    shape = SimpleNamespace(native=[SimpleNamespace(cell='balance')], extra=())
    assert scalar_factory_graph(node, cls, shape) is admitted


@pytest.mark.parametrize("signature", ["operand=callback", "abs=1", "*rest", "**options"])
def test_opaque_cell_signature(signature):
    node = libcst.parse_module(f"""
class Product:
    def _mx_space_formula(self, key):
        return {{'refs': {{'point': key}}}}
    def _f_balance(self):
        return self.helper()
    def _f_helper(self, {signature}):
        return 1.0
""").body[0]
    cls = SimpleNamespace(cells={'balance', 'helper'}, refs={})
    shape = SimpleNamespace(native=[SimpleNamespace(cell='balance')], extra=())
    assert not scalar_factory_graph(node, cls, shape)


@pytest.mark.parametrize("prefix", ["from other import *", "from other import abs", "round = callback"])
def test_builtin_bindings(prefix):
    assert not unshadowed_builtins(libcst.parse_module(prefix + '\n'))


def test_runtime_substituted_builtin(monkeypatch):
    import builtins
    from modelx_cython.runtime.factory_reuse import canonical_factory_builtins
    assert canonical_factory_builtins()
    monkeypatch.setattr(builtins, 'round', lambda value: value)
    assert not canonical_factory_builtins()
