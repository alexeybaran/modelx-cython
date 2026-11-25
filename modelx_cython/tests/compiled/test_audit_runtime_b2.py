"""Differential execution of defects found by the pre-coverage audit: scope overrides of enum members and
reductions (B2)."""
import functools
from http import HTTPStatus
import json
import math
import subprocess
import sys
from types import SimpleNamespace

import modelx as mx

from test_audit_runtime import compile_model


def test_scope_overrides_aliases_and_arithmetic(tmp_path):
    model=mx.new_model('AuditContracts')
    try:
        foreign=SimpleNamespace(Status=SimpleNamespace(OK=999),
            lookup=functools.partial(float,7), aggregate=sum,
            term=functools.partial(float), count=functools.partial(int,2))
        s=model.new_space('Values',refs={'Status':HTTPStatus,
            'other':foreign,
            'lookup':float,'aggregate':sum,'numbers':math,
            'downward':math.floor,'round_alias':round})
        s.new_cells('term','lambda k: 1.0 + k')
        s.new_cells('coded','lambda t, code: t + code')
        s.new_cells('call_code','lambda t: coded(t, Status.OK)').is_cached=False
        s.new_cells('flag','lambda t: t > 0')
        s.new_cells('compare_flag','lambda t: flag(t) == Status.OK').is_cached=False
        s.new_cells('count','lambda: 2')
        s.new_cells('nested_enum','''def nested_enum():
    def nested(self):
        return self.Status.OK
    return nested(other)
''')
        s.new_cells('nested_helper','''def nested_helper():
    def nested(self):
        return self.lookup()
    return nested(other)
''')
        s.new_cells('nested_sum','''def nested_sum():
    def nested(self):
        return self.aggregate(self.term(k) for k in range(self.count()))
    return nested(other)
''')
        s.new_cells('matches','lambda t: t == Status.OK').is_cached=False
        s.new_cells('condition','lambda t: 1.0 if t == Status.OK else 0.0').is_cached=False
        s.new_cells('compound','lambda t: 1.0 if t == Status.OK or t < 0 else 0.0').is_cached=False
        s.new_cells('comparison_value','lambda t: (t == Status.OK,)').is_cached=False
        s.new_cells('aliases','lambda x: numbers.floor(x) + downward(x) + round_alias(x)').is_cached=False
        s.new_cells('long_expression','''def long_expression(t):
    a = t + t + t + t + t + t + t + t
    b = t * (t - 1)
    return a / 12.0 + b
''').is_cached=False
        s.new_cells('hygienic','''def hygienic():
    _mx_reduce_0_function = 1.0
    return sum((term(k) + _mx_reduce_0_function for k in range(count()) if k != 1), 0.0)
''')
        s.new_cells('reassigned_tail','''def reassigned_tail(t):
    a = t * 2
    a = t * 3
    return a / 12.0 + a
''').is_cached=False
        # not an IntEnum class: an ordinary Reference (read per evaluation, rebindable)
        virtual=model.new_space('VirtualValues',refs={'Status':SimpleNamespace(OK=HTTPStatus.OK)})
        for name in ('matches','condition','compound','comparison_value','coded',
                     'call_code','flag','compare_flag'):
            cell=virtual.new_cells(name,s.cells[name].formula.source)
            cell.is_cached=s.cells[name].is_cached
        model.export(tmp_path/'exported')
    finally:model.close()
    compile_model(tmp_path,'from exported import mx_model\ns=mx_model.Values\n'
        's.term(0);s.count();s.nested_enum();s.nested_helper();s.nested_sum();s.matches(200);'
        's.condition(200);s.compound(200);s.comparison_value(200);'
        's.call_code(1);s.coded(1,201);'
        's.compare_flag(1);'
        's.aliases(1.25);s.long_expression(2);s.hygienic();s.reassigned_tail(2)\n'
        'v=mx_model.VirtualValues\nv.matches(200);v.condition(200);v.compound(200);'
        'v.comparison_value(200);v.call_code(1);v.coded(1,201);v.compare_flag(1)\n',
        integer_checks='checked')     # the arithmetic checks below are the checked mode's (owner's rule)
    admission=json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    info=lambda name:admission[f'exported_cy._mx_classes._c_Values._f_{name}']
    assert info('nested_sum')['native_reductions']==0
    assert info('hygienic')['native_reductions']==0      # a sum with a start argument is not lowered (R-6)
    script='''import math
from http import HTTPStatus
from types import SimpleNamespace
from PACKAGE import mx_model
s=mx_model.Values
assert s.nested_enum()==999
assert s.nested_helper()==7.0
assert s.nested_sum()==1.0
assert s.hygienic()==2.0
assert s.matches(200) is True and s.matches(201) is False
assert s.call_code(1)==201
class OverrideDependency(type(s)):
    def coded(self, t, code): return 1000+t+code
override=OverrideDependency(mx_model)
override._mx_copy_refs(s,s)
assert override.call_code(1)==1201
# HTTPStatus is no enum family (values not 0..n-1): none of its members is in the code, so its Reference is an
# ordinary one, rebound and read as in modelx (a family's Reference is fixed: test_enum_by_type)
s.Status=SimpleNamespace(OK=201)
assert s.matches(201) is True and s.matches(200) is False
s.Status=HTTPStatus
assert s.matches(200) is True
v=mx_model.VirtualValues
v.Status=SimpleNamespace(OK=201)
assert v.matches(200) is False and v.matches(201) is True
assert v.condition(200)==0.0 and v.condition(201)==1.0
assert v.call_code(1)==202
class BoolMember:
    def __eq__(self, value): return type(value) is bool and value
v.Status=SimpleNamespace(OK=BoolMember())
assert v.compare_flag(1) is True
class EqualityValue:
    def __bool__(self): raise AssertionError('unexpected comparison coercion')
equality_value=EqualityValue()
class ReturnsObject:
    def __eq__(self, value): return equality_value
v.Status=SimpleNamespace(OK=ReturnsObject())
assert v.comparison_value(200)[0] is equality_value
class CountTruth:
    calls=0
    def __bool__(self):
        self.calls+=1
        return self.answer
truth=CountTruth()
class CountEquality:
    def __eq__(self, value): return truth
v.Status=SimpleNamespace(OK=CountEquality())
for answer in (False,True):
    truth.answer=answer
    truth.calls=0
    assert v.compound(200)==float(answer)
    assert truth.calls==1
class EqualTo200:
    def __eq__(self, value): return value == 200
v.Status=SimpleNamespace(OK=EqualTo200())
assert v.matches(200) is True and v.matches(201) is False
failure=ValueError('enum comparison')
class BrokenMember:
    def __eq__(self, value): raise failure
v.Status=SimpleNamespace(OK=BrokenMember())
try:
    v.matches(200)
except ValueError as actual:
    assert actual is failure
else:
    raise AssertionError('subclass comparison exception lost')
for x in (-4.75,-0.0,1.25,17.5):
    assert s.aliases(x)==math.floor(x)*2+round(x)
s.downward=lambda x:10
assert s.aliases(1.25)==12
for t in (-(1<<32),-(1<<25),-1,0,1,(1<<25),(1<<32)):
    if abs(t)==(1<<32) and type(s).__module__.startswith('exported_cy'):
        # t*(t-1) leaves the signed range: the exact fallback is gone (checked mode), it raises
        try:
            s.long_expression(t)
        except OverflowError as error:
            assert 'Values.long_expression' in str(error),error
        else:
            raise AssertionError('long_expression: an intermediate beyond the signed range did not raise')
    else:
        assert s.long_expression(t)==(t+t+t+t+t+t+t+t)/12.0+t*(t-1)
    assert s.reassigned_tail(t)==(t*3)/12.0+t*3
'''
    for package in ('exported','exported_cy'):
        subprocess.run([sys.executable,'-B','-c',script.replace('PACKAGE',package)],cwd=tmp_path,check=True)
