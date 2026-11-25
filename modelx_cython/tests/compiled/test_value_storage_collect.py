"""S1 on the collect route: dense cache values of a reused root and of a shared Data filled by collect workers are
not zero-filled; the build and checks are those of ``test_value_storage``."""
from test_value_storage import _check


def test_unwritten_values_on_collect_and_direct_routes(tmp_path):
    _check(tmp_path, 'collect')
