"""Public collect entry in the experimental frame-backed Space facade."""

import threading

import numpy as np
import pytest

from modelx_cython._frame_runtime.model import CompiledModel, _ProjectionSpace


class _Kernel:
    def __init__(self):
        self.scenario = -1
        self.calls = []
        self.fail = False

    def set_parameters(self, **values):
        self.scenario = values['scenario']

    def run_scalar(self, name):
        self.calls.append((name, self.scenario))
        if self.fail:
            raise ValueError('calculation failed')
        return np.asarray([1.0 + self.scenario, 2.0 + self.scenario])


def _model():
    model = object.__new__(CompiledModel)
    model._plan = {
        'collect': {'over': 'point_id', 'sum': ['total']},
        'point_parameter': 'point_id',
        'parameter_defaults': {'scenario': -1},
        'space_parameters': [('point_id', 'POSITIONAL_OR_KEYWORD'),
                             ('scenario', 'POSITIONAL_OR_KEYWORD')],
    }
    model.manifest = {'parameters': ['scenario']}
    model._lock = threading.RLock()
    model._closed = False
    model._active = 0
    model._collect_cache = {}
    model._kernel = _Kernel()
    model._projection = _ProjectionSpace(model)
    return model


def test_collect_uses_one_portfolio_entry_and_caches_aggregate():
    model = _model()
    result = model._projection.collect()
    assert result.sum.total == 1.0
    assert model._kernel.calls == [('total', -1)]
    assert model._projection.collect() is result
    assert model._kernel.calls == [('total', -1)]

    override = model._projection.collect(scenario=3)
    assert override.sum.total == 9.0
    assert model._kernel.calls == [('total', -1), ('total', 3)]
    assert model._kernel.scenario == -1
    assert model._projection.collect(3) is override


def test_collect_restores_inputs_and_does_not_cache_a_failure():
    model = _model()
    model._kernel.fail = True
    with pytest.raises(ValueError, match='calculation failed'):
        model._projection.collect(3)
    assert model._kernel.scenario == -1
    assert model._active == 0
    assert model._collect_cache == {}

    model._kernel.fail = False
    assert model._projection.collect(3).sum.total == 9.0


def test_collect_rejects_nested_call_without_clobbering_outer_context():
    model = _model()
    model._active = 1
    with pytest.raises(RuntimeError, match='active Cell evaluation'):
        model._projection.collect()
    assert model._kernel.calls == []
