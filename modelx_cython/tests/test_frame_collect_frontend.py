"""Declared collect roots in the experimental frame frontend."""

from types import SimpleNamespace

import pytest

from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.modelx_frontend_native_mode import _collect_sum_plan


class _Domain:
    def __init__(self, values):
        self.values = values

    def value(self, space):
        return self.values


def _space(source, domain=(1, 2, 3)):
    return SimpleNamespace(
        _impl=SimpleNamespace(collect_over_spec={'point_id': _Domain(domain)}),
        cells={
            'collect_outputs': SimpleNamespace(formula=SimpleNamespace(source=source)),
            'value': object(),
        },
    )


def test_read_direct_cell_sum_from_modelx_declaration():
    table = SimpleNamespace(index=(1, 2, 3))
    assert _collect_sum_plan(
        _space('lambda: {"sum": {"value": value()}}'), 'point_id', table,
    ) == {'over': 'point_id', 'sum': ['value']}


def test_collect_function_docstring_keeps_direct_return_contract():
    source = 'def collect_outputs():\n    """Public result."""\n    return {"sum": {"value": value()}}'
    assert _collect_sum_plan(_space(source), 'point_id',
                             SimpleNamespace(index=(1, 2, 3))) == {
                                 'over': 'point_id', 'sum': ['value']}


@pytest.mark.parametrize('source', [
    'lambda: {"individual": {"value": value()}}',
    'lambda: {"sum": {"value": value() + 1}}',
    'lambda: {"sum": {"renamed": value()}}',
])
def test_non_scalar_collect_declarations_keep_author_formula(source):
    assert _collect_sum_plan(_space(source), 'point_id',
                             SimpleNamespace(index=(1, 2, 3))) == {
                                 'over': 'point_id', 'object_outputs': 'collect_outputs'}


def test_dv01_shaped_mixed_time_indexed_declaration_uses_object_traversal():
    source = '''lambda: {
        "individual": {"state": {t: state(t) for t in snapshot_months}},
        "sum": {"net_cf": {t: net_cf(t) for t in projection_months}},
    }'''
    assert _collect_sum_plan(_space(source), 'point_id',
                             SimpleNamespace(index=(1, 2, 3))) == {
                                 'over': 'point_id', 'object_outputs': 'collect_outputs'}


def test_changed_domain_order_cannot_use_full_table_traversal():
    with pytest.raises(ContractError, match='domain must match'):
        _collect_sum_plan(_space('lambda: {"sum": {"value": value()}}',
                                 domain=(3, 2, 1)),
                          'point_id', SimpleNamespace(index=(1, 2, 3)))
