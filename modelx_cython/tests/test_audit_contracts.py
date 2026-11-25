"""Regression cases found by the architecture/genericity audit."""
import ast
from types import SimpleNamespace

import libcst as cst

from modelx_cython.helpers import HelperCalls


def native_cell(result='double', params=()):
    return SimpleNamespace(params=params, has_formula_def=False,
        has_typeinfo=lambda: True, called_with_kwargs=False, body_has_closure=False,
        formula_allows_exact_receiver=True,
        get_rettype_expr=lambda **kw: result,
        get_argtype_expr=lambda *args, **kw: 'long long')


def test_helper_binding_does_not_capture_a_nested_receiver():
    source='''def formula(self):
    def nested(self):
        return self.lookup()
    return nested(self.other)
'''
    ref=SimpleNamespace(helper=('sample.provider','lookup'))
    result=cst.parse_module(source).visit(HelperCalls({'lookup':ref},set())).code
    assert result == source


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
