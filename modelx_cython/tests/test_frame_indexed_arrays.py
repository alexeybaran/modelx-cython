"""Numeric Data-array reads preserve checked NumPy indexing and input ownership."""
import ast

import numpy as np
import pytest

from modelx_cython._frame_runtime.compiler import FrontendRewrite
from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.lookups import (
    IndexedFloat1, IndexedFloat2, IndexedInt1, IndexedInt2,
)
from modelx_cython._frame_runtime.runtime import load_inputs, save_inputs


@pytest.mark.parametrize('cls,values', [
    (IndexedInt1, np.array([0, -1, 2], dtype=np.int64)),
    (IndexedFloat1, np.array([0.0, np.nan, -1.0], dtype=np.float64)),
])
def test_one_axis_snapshot_and_index_bounds(cls, values):
    expected = values.copy()
    owned = cls(values)
    values[0] = 99
    assert owned.get(0) == expected[0]
    assert owned.get(-1) == expected[-1]
    with pytest.raises(IndexError): owned.get(len(values))
    with pytest.raises(IndexError): owned.get(-len(values) - 1)


@pytest.mark.parametrize('cls,dtype', [
    (IndexedInt2, np.int64), (IndexedFloat2, np.float64),
])
def test_two_axis_indices_and_snapshot(cls, dtype):
    source = np.arange(6, dtype=dtype).reshape(2, 3)
    owned = cls(source)
    source[0, 0] = 99
    assert owned.get(0, 0) == 0
    assert owned.get(-1, -1) == 5
    with pytest.raises(IndexError): owned.get(2, 0)
    with pytest.raises(IndexError): owned.get(0, -4)


def test_wrong_numeric_kind_and_rank_fail():
    with pytest.raises(TypeError): IndexedInt1(np.array([1.0]))
    with pytest.raises(TypeError): IndexedFloat2(np.array([1.0]))


def test_prepared_array_bundle_round_trip(tmp_path):
    arr = IndexedInt1(np.array([1, -1], dtype=np.int64))
    rows = [{'values': arr}, {'values': arr}]
    save_inputs(rows, tmp_path)
    loaded = load_inputs(tmp_path)
    assert loaded[0]['values'] is loaded[1]['values']
    assert loaded[0]['values'].get(-1) == -1


def test_frontend_lowers_only_typed_integral_indices():
    frontend = {'arrays': {'rates': 'input_rates'}}
    inputs = {'input_rates': IndexedFloat2}
    node = ast.parse('data.rates()[age(t), max(duration(t), 0)]', mode='eval')
    rewritten = FrontendRewrite(frontend, inputs, integer_names={'t'},
                                integer_cells={'age', 'duration'}).visit(node)
    assert ast.unparse(rewritten) == 'input_rates.get(age(t), max(duration(t), 0))'
    for expression in ('data.rates()[1.5, 0]', 'data.rates()[age(t) / 2, 0]',
                       'data.rates()[0:2, 0]'):
        with pytest.raises(ContractError, match='typed integral indices'):
            FrontendRewrite(frontend, inputs, integer_names={'t'},
                            integer_cells={'age'}).visit(ast.parse(expression, mode='eval'))
