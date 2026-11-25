"""Cache policy follows generic footprint and receiver evidence: the collect lifecycle."""
from types import SimpleNamespace
import libcst as cst
import pytest


@pytest.mark.parametrize('hook',['_mx_apply_formula','_mx_collect_attach','_mx_collect_release'])
def test_automatic_collect_keeps_custom_lifecycle_on_normal_path(hook):
    from modelx_cython.serial_collect import _match
    source='''class Example(_mx_sys.BaseSpace):
    def _mx_collect_contract_v2(self):
        return ('key', (), 'domain', False)
    def _f_collect_outputs(self):
        return {'sum': {'total': self.value()}}
'''
    cls=SimpleNamespace(spaces=(),is_locked=False,visitor=SimpleNamespace(dynamic_ref_classes=()),
        refs={'key':SimpleNamespace(get_type_expr=lambda **kwargs:'long long')},
        cells={'value':SimpleNamespace(has_args=lambda:False,get_rettype_expr=lambda **kwargs:'double',
            body_has_closure=False,called_with_kwargs=False)})
    assert _match(cst.parse_module(source).body[0],cls,False,False,False) is not None
    with_hook = source + f"    def {hook}(self, *args):\n        raise RuntimeError('custom lifecycle')\n"
    assert _match(cst.parse_module(with_hook).body[0],cls,False,False,False) is None
