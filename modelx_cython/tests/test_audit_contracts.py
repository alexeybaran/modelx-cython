"""Regression cases found by the architecture/genericity audit."""
import ast
from types import SimpleNamespace

import libcst as cst

from modelx_cython.enums import Constants
from modelx_cython.helpers import HelperCalls
from modelx_cython.reductions import NativeReductions
import pytest


def native_cell(result='double', params=()):
    return SimpleNamespace(params=params, has_formula_def=False,
        has_typeinfo=lambda: True, called_with_kwargs=False, body_has_closure=False,
        formula_allows_exact_receiver=True,
        get_rettype_expr=lambda **kw: result,
        get_argtype_expr=lambda *args, **kw: 'long long')


def test_enum_constants_do_not_capture_a_nested_receiver():
    source='''def formula(self):
    def nested(self):
        return self.Status.OK
    return nested(self.other)
'''
    cls=SimpleNamespace(name='_c_Values', cells={})
    pass_=Constants(cls,native_cell('long long'),
                    {('_c_Values','Status','OK'):('_mx_enum_const_0',200)})
    result=cst.parse_module(source).visit(pass_).code
    assert 'return self.Status.OK' in result
    assert pass_.count == 0


def test_helper_binding_does_not_capture_a_nested_receiver():
    source='''def formula(self):
    def nested(self):
        return self.lookup()
    return nested(self.other)
'''
    ref=SimpleNamespace(helper=('sample.provider','lookup'))
    result=cst.parse_module(source).visit(HelperCalls({'lookup':ref},set())).code
    assert result == source


def test_reduction_does_not_type_a_nested_receiver_from_outer_cells():
    source='''def formula(self):
    def nested(self):
        return self.sum_f64(self.term(k) for k in range(self.count()))
    return nested(self.other)
'''
    formula=cst.parse_module(source).body[0]
    cls=SimpleNamespace(name='_c_Values', refs={'sum_f64':SimpleNamespace(
        reduction='f64_sum',helper=('sample.provider','sum_f64'))},
        cells={'term':native_cell(params=('k',)),'count':native_cell('long long')})
    pass_=NativeReductions(cls,native_cell(),set(),formula,True)
    result=cst.Module([formula.visit(pass_)]).code
    assert result == source
    assert pass_.general_count == 0


@pytest.mark.parametrize('source', [
    'def formula(self):\n    self = self.other\n    return self.sum_f64(self.term(k) for k in range(self.count()))\n',
    'def formula(self):\n    return self.sum_f64(self.term(self) for self in range(self.count()))\n',
    'def formula(self):\n    self.snapshots.append(locals())\n    return self.sum_f64(self.term(k) for k in range(self.count()))\n',
])
def test_typed_reduction_preserves_receiver_binding_and_frame_observation(source):
    formula = cst.parse_module(source).body[0]
    cell = native_cell()
    cell.has_formula_def = True
    cls = SimpleNamespace(name='_c_Values', refs={'sum_f64': SimpleNamespace(
        reduction='f64_sum', helper=('sample.provider', 'sum_f64'))},
        cells={'term': native_cell(params=('k',)), 'count': native_cell('long long')})
    lowering = NativeReductions(cls, cell, set(), formula, True)
    assert cst.Module([formula.visit(lowering)]).code == source
    assert lowering.general_count == 0


def test_enum_constants_do_not_capture_a_comprehension_receiver():
    source='''def formula(self, value):
    return tuple(value == self.Status.OK for self in self.others)
'''
    cls=SimpleNamespace(name='_c_Values',cells={})
    pass_=Constants(cls,native_cell('object',('value',)),
                    {('_c_Values','Status','OK'):('_mx_enum_const_0',200)})
    result=cst.parse_module(source).visit(pass_).code
    assert 'self.Status.OK' in result
    assert pass_.count == 0


def test_declared_code_enum_uses_its_fixed_definition():
    source='def formula(self, value):\n    return value == self.Status.OK\n'
    pass_=Constants(SimpleNamespace(name='_c_Values',cells={}),native_cell('bint',('value',)),
                    {('_c_Values','Status','OK'):('_mx_enum_const_0',200)})
    namespace={'_mx_enum_const_0':200}
    transformed=cst.parse_module(source).visit(pass_).code
    expression=ast.parse(transformed).body[0].body[0].value
    assert isinstance(expression,ast.Compare)
    assert isinstance(expression.comparators[0],ast.Name)
    exec(transformed,namespace)
    assert namespace['formula'](object(),200) is True
    assert namespace['formula'](object(),201) is False


def test_enum_constants_decline_receiver_rebinding():
    source='def formula(self, value, other):\n    self = other\n    return value == self.Status.OK\n'
    pass_=Constants(SimpleNamespace(name='_c_Values',cells={}),
        native_cell('bint',('value','other')),
        {('_c_Values','Status','OK'):('_mx_enum_const_0',200)})
    assert cst.parse_module(source).visit(pass_).code==source
    assert pass_.count==0


def test_cache_hit_optimization_does_not_depend_on_ast_unparse(monkeypatch):
    from modelx_cython.cell_entries import EntryModule
    from modelx_cython.config import CompilerPolicy
    from modelx_cython.entry_pipeline import lower_entries
    monkeypatch.delattr(ast,'unparse')
    state = EntryModule(cst.parse_module(''))
    assert lower_entries({'m': state}, {}, {}, CompilerPolicy.resolve({}))['m'].syntax.code == ''




def test_provider_resources_use_the_legacy_resource_api(monkeypatch, tmp_path):
    from modelx_cython import helpers
    monkeypatch.delattr(helpers.resources, 'files')
    read = []
    def read_binary(package, resource):
        read.append((package, resource))
        return resource.encode()
    monkeypatch.setattr(helpers.resources, 'read_binary', read_binary)
    model = tmp_path/'compiled'
    model.mkdir()
    modules = helpers.install({'sample.provider'}, model)
    assert read == [('sample', 'provider.pyx'), ('sample', 'provider.pxd')]
    assert (model/(helpers.module_name('sample.provider')+'.pyx')).read_bytes()==b'provider.pyx'
    assert len(modules)==1


def test_enum_constants_follow_underlying_integer_not_overridden_conversion():
    from enum import IntEnum
    from modelx_cython.tracer import RuntimeValueInfo
    class Codes(IntEnum):
        VALUE=1
        def __int__(self):return 999
    info=RuntimeValueInfo(Codes)
    # Local enum declarations deliberately do not qualify for export metadata.
    Codes.__qualname__='Codes'
    info=RuntimeValueInfo(Codes)
    assert Codes.VALUE == 1 and int(Codes.VALUE)==999
    assert info.enum[2]['VALUE']==1


def test_a_provider_with_only_a_pxd_is_installed_as_a_header(monkeypatch, tmp_path):
    """No .pyx: the .pxd is copied next to the model and no extension module is built for it."""
    from modelx_cython import helpers
    def read_resource(package, resource):
        if resource.endswith('.pyx'):
            raise FileNotFoundError(resource)
        return resource.encode()
    monkeypatch.setattr(helpers, 'read_resource', read_resource)
    model = tmp_path/'compiled'
    model.mkdir()
    alias = helpers.module_name('sample.header')
    assert helpers.install({'sample.header'}, model) == []
    assert (model/(alias+'.pxd')).read_bytes() == b'header.pxd'
    assert not (model/(alias+'.pyx')).exists()
