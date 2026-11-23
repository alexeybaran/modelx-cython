import libcst as cst
import pytest

from modelx_cython.serial_collect import _factory_aliases


@pytest.mark.parametrize("body,eligible", [
    ("return {'refs': {'point': key}}", ['point']),
    ("if key < 0:\n            raise ValueError(key)\n        return {'refs': {'point': key}}", ['point']),
    ("return {'refs': {'point': self.source(key)}}", None),
    ("self.mark(key)\n        return {'refs': {'point': key}}", None),
    ("if key:\n            return None\n        return {'refs': {'point': key}}", None),
])
def test_static_reference_aliases(body, eligible):
    cls = cst.parse_module("class S:\n    def _mx_space_formula(self, key):\n        " + body + "\n").body[0]
    assert _factory_aliases(cls) == eligible
