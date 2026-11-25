"""Regression cases found by the architecture/genericity audit."""
import ast
from types import SimpleNamespace

import libcst as cst

from modelx_cython.enums import literals
from modelx_cython.helpers import HelperCalls
from modelx_cython.reductions import NativeReductions
import pytest


def native_cell(result='double', params=()):
    return SimpleNamespace(params=params, has_formula_def=False,
        has_typeinfo=lambda: True, called_with_kwargs=False, body_has_closure=False,
        formula_allows_exact_receiver=True,
        get_rettype_expr=lambda **kw: result,
        get_argtype_expr=lambda *args, **kw: 'long long')


STATUS = {'m._c_Values.Status': SimpleNamespace(enum=('m', 'Status', {'OK': 200}), enum_safe=True)}


def in_class(source):
    return 'class _c_Values:\n' + ''.join('    ' + line + '\n' for line in source.splitlines())


def test_enum_constants_do_not_capture_a_nested_receiver():
    source=in_class('''def formula(self):
    def nested(self):
        return self.Status.OK
    return nested(self.other)''')
    assert literals('m', source, STATUS) == source


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
    source=in_class('''def formula(self, value):
    return tuple(value == self.Status.OK for self in self.others)''')
    assert literals('m', source, STATUS) == source


def test_declared_code_enum_uses_its_fixed_definition():
    transformed=literals('m', in_class('def formula(self, value):\n    return value == self.Status.OK'), STATUS)
    expression=ast.parse(transformed).body[0].body[0].body[0].value
    assert isinstance(expression,ast.Compare)
    assert isinstance(expression.comparators[0],ast.Constant) and expression.comparators[0].value == 200
    namespace={}
    exec(transformed,namespace)
    assert namespace['_c_Values'].formula(object(),200) is True
    assert namespace['_c_Values'].formula(object(),201) is False


def test_enum_constants_decline_receiver_rebinding():
    source=in_class('def formula(self, value, other):\n    self = other\n    return value == self.Status.OK')
    assert literals('m', source, STATUS) == source


def test_cache_hit_optimization_does_not_depend_on_ast_unparse(monkeypatch):
    from modelx_cython.cell_entries import EntryModule
    from modelx_cython.config import CompilerPolicy
    from modelx_cython.entry_pipeline import lower_entries
    monkeypatch.delattr(ast,'unparse')
    state = EntryModule(cst.parse_module(''))
    assert lower_entries({'m': state}, {}, {}, CompilerPolicy.resolve({}))['m'].syntax.code == ''


def test_collection_admission_reuses_the_existing_syntax_without_ast(monkeypatch):
    from modelx_cython.serial_collect import analyze
    monkeypatch.delattr(ast, 'unparse')
    module = SimpleNamespace(classes={}, spec=SimpleNamespace(automatic_optimizations=True),
        visitor=SimpleNamespace(wrapper=SimpleNamespace(module=cst.parse_module('class Values: pass\n'))))
    assert analyze(module) == {}


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


def test_arithmetic_rewrite_preserves_global_assignment_effects():
    import cython
    from modelx_cython.checked_arithmetic import CheckedArithmetic
    cell=native_cell('double',('t',))
    cell.is_locked=False;cell.cls='_c_Values';cell._spec={}
    cls=SimpleNamespace(cells={},visitor=SimpleNamespace(dynamic_ref_classes=()))
    source='''def formula(self, t):
    global a, b
    a = t * t
    b = t + t
    return a / 12.0 + b
'''
    function=cst.parse_module(source).body[0]
    result=CheckedArithmetic(cls,cell).rewrite(function)
    namespace={'a':0,'b':0,'_mx_cy':cython}
    exec(cst.Module([result]).code,namespace)
    assert namespace['formula'](None,2)==4/12.0+4
    assert (namespace['a'],namespace['b'])==(4,4)


