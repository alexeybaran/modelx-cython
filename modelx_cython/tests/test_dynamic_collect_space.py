"""Exported collection ItemSpaces need fields even when tracing retains none."""

from modelx_cython.parser import ModuleVisitor


def test_collect_only_space_metadata_includes_parameter_and_dynamic_dict():
    source = '''_v_space_params_Projection = ['point_id']
class _c_Projection:
    __slots__ = ('_mx_itemspaces', '__dict__')
    def _mx_assign_params(self, point_id):
        self.point_id = point_id
'''
    visitor = ModuleVisitor('Example._mx_classes', source)
    assert visitor.space_params['_c_Projection'] == ('point_id',)
    assert '_c_Projection' in visitor.dict_classes
