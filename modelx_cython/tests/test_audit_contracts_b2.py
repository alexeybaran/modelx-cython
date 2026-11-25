"""Regression cases found by the architecture/genericity audit: lowered reductions (B2)."""
from types import SimpleNamespace

import libcst as cst
import pytest

from modelx_cython.reductions import NativeReductions

from test_audit_contracts import native_cell


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
    pass_=NativeReductions(cls,native_cell(),set(),formula,lambda:'_cold',True)
    result=cst.Module([formula.visit(pass_)]).code
    assert result == source
    assert pass_.count == 0


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
    lowering = NativeReductions(cls, cell, set(), formula, lambda: '_cold', True)
    assert cst.Module([formula.visit(lowering)]).code == source
    assert lowering.count == 0
