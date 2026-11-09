"""Scalar sum identities must survive native result/cache type selection."""
import numbers

import numpy as np
import pytest

from modelx_cython.monkeytype_tracing import CallTrace
from modelx_cython.tracer import ReturnTypeInfo, RuntimeCellsInfo


def formula(self, n):
    pass


def merged(values):
    return RuntimeCellsInfo([CallTrace(formula, {'self': None, 'n': n}, ret_val=value)
                             for n, value in enumerate(values)]).ret_type


@pytest.mark.parametrize('values', [[0, 1.5], [1.5, 0], [0, np.float64(1.5)],
                                  [np.float64(1.5), 0], [0, 1.5, 0, 2.5]])
def test_scalar_integral_nonintegral_union_retains_object_result(values):
    assert merged(values) == ReturnTypeInfo(object)


def test_homogeneous_float_and_integral_scalar_contracts_are_unchanged():
    assert merged([0.0, 1.5]) == ReturnTypeInfo(float)
    assert merged([0, 1]) == ReturnTypeInfo(int)
    assert merged([0, np.int64(1)]) == ReturnTypeInfo(numbers.Integral)


def test_array_numeric_promotion_is_unchanged():
    result = merged([np.array([0], dtype=np.int64), np.array([1.5], dtype=float)])
    assert result == ReturnTypeInfo(numbers.Real, is_array=True, ndim=1)
