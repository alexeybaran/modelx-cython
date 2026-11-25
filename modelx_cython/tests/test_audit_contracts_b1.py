"""Regression cases found by the architecture/genericity audit: IntEnum members as integer literals (B1)."""
import ast
from types import SimpleNamespace

from modelx_cython.enums import literals


class Status(__import__('enum').IntEnum):
    OK = 0
    FAILED = 1


STATUS = {'m._c_Values.Status': SimpleNamespace(enum=('m', 'Status', {'OK': 0, 'FAILED': 1}), enum_safe=True,
                                                enum_class=Status)}


def in_class(source):
    return 'class _c_Values:\n' + ''.join('    ' + line + '\n' for line in source.splitlines())


def test_enum_constants_do_not_capture_a_nested_receiver():
    source=in_class('''def formula(self):
    def nested(self):
        return self.Status.OK
    return nested(self.other)''')
    assert literals('m', source, STATUS) == source


def test_enum_constants_do_not_capture_a_comprehension_receiver():
    source=in_class('''def formula(self, value):
    return tuple(value == self.Status.OK for self in self.others)''')
    assert literals('m', source, STATUS) == source


def test_declared_code_enum_uses_its_fixed_definition():
    held = {'m._c_Values.formula': SimpleNamespace(arg_types={'value': int}, enum_codes={'value': Status})}
    transformed=literals('m', in_class('def formula(self, value):\n    return value == self.Status.FAILED'), STATUS, held)
    expression=ast.parse(transformed).body[0].body[0].body[0].value
    assert isinstance(expression,ast.Compare)
    assert isinstance(expression.comparators[0],ast.Constant) and expression.comparators[0].value == 1
    namespace={}
    exec(transformed,namespace)
    assert namespace['_c_Values'].formula(object(),1) is True
    assert namespace['_c_Values'].formula(object(),0) is False


def test_enum_constants_decline_receiver_rebinding():
    source=in_class('def formula(self, value, other):\n    self = other\n    return value == self.Status.OK')
    assert literals('m', source, STATUS) == source


def test_enum_constants_follow_underlying_integer_not_overridden_conversion():
    from enum import IntEnum
    from modelx_cython.tracer import RuntimeValueInfo
    class Codes(IntEnum):
        OTHER=0
        VALUE=1
        def __int__(self):return 999
    info=RuntimeValueInfo(Codes)
    # Local enum declarations deliberately do not qualify for export metadata.
    Codes.__qualname__='Codes'
    info=RuntimeValueInfo(Codes)
    assert Codes.VALUE == 1 and int(Codes.VALUE)==999
    assert info.enum[2]['VALUE']==1
