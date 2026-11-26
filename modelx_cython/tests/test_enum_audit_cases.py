"""Regression cases of enum constants: receivers that are not the formula's, and the value of a member."""
import ast
from types import SimpleNamespace

import libcst as cst

from modelx_cython.enums import Constants


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
