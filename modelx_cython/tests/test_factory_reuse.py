from types import SimpleNamespace

import libcst
import pytest

from modelx_cython.factory_reuse import scalar_factory_graph


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
