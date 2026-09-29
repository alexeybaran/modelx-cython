from enum import IntEnum
from types import SimpleNamespace

import pytest

from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.trace_contracts import from_mx2cy_trace


class Category(IntEnum):
    ONE = 0


def test_frame_contracts_use_existing_trace_and_leave_unseen_cells_untyped():
    manifest = {'model': 'M', 'spaces': {'M.Projection': {
        'cells': {'a': {}, 'b': {}, 'unseen': {}}}}}
    logger = SimpleNamespace(cells_info={
        'export._mx_classes._c_Projection._f_a': SimpleNamespace(
            ret_type=SimpleNamespace(is_array=False, value_type=float),
            arg_types={'t': int}),
        'export._mx_classes._c_Projection._f_b': SimpleNamespace(
            ret_type=SimpleNamespace(is_array=False, value_type=Category),
            arg_types={}),
        'export._mx_classes._c_Data._f_a': SimpleNamespace(
            ret_type=SimpleNamespace(is_array=False, value_type=str),
            arg_types={}),
    })
    assert from_mx2cy_trace(logger, manifest) == {
        'a': {'return': float, 't': int},
        'b': {'return': Category},
    }


def test_frame_contracts_reject_ambiguous_trace():
    observed = SimpleNamespace(
        ret_type=SimpleNamespace(is_array=False, value_type=float),
        arg_types={})
    logger = SimpleNamespace(cells_info={
        'one._mx_classes._c_Projection._f_a': observed,
        'two._mx_classes._c_Projection._f_a': observed})
    manifest = {'model': 'M', 'spaces': {'M.Projection': {'cells': {'a': {}}}}}
    with pytest.raises(ContractError, match='Ambiguous'):
        from_mx2cy_trace(logger, manifest)
