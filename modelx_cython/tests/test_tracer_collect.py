"""The tracer's code filter and the exported collection adapter."""
from modelx_cython.tracer import MxCodeFilter

from test_tracer import compiled_codes


def test_code_filter_distinguishes_collection_adapter_and_user_cell():
    codes = compiled_codes("C:/model/Model_nomx/_mx_classes.py", """
class _c_Space1:
    def collect(self, cell):
        return collect_from_space(self._mx_collect_cache, cell)

class _c_Space2:
    def collect(self, x):
        return x
""")
    filt = MxCodeFilter()
    assert not filt(codes["_c_Space1.collect"])
    assert filt(codes["_c_Space2.collect"])
