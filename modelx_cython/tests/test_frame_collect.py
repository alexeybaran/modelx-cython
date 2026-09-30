"""Public collect entry in the experimental frame-backed Space facade."""

import threading

import numpy as np
import pandas as pd
import pytest

from modelx_cython._frame_runtime.model import (
    CompiledModel, _ProjectionSpace, _CollectAccumulator,
    _detach_collect_value)


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
    # The fixture bypasses CompiledModel.__init__; these hand-built bindings are ready.
    model._initialized = True
    model._active = 0
    model._collect_cache = {}
    model._space_bindings = None
    model._connected_context_key = None
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


class _ObjectKernel(_Kernel):
    def _run_object_outputs(self, name, detach, consume=None):
        assert name == 'collect_outputs'
        self.calls.append((name, self.scenario))
        scratch = {'individual': {'state': {0: np.array([0.0])}},
                   'sum': {'net_cf': {0: 0.0, 1: 0.0}}}
        result = []
        for policy in range(2):
            if self.fail and policy == 1:
                raise ValueError('calculation failed')
            scratch['individual']['state'][0][0] = policy + self.scenario
            scratch['sum']['net_cf'][0] = float(policy + 1)
            scratch['sum']['net_cf'][1] = float(policy + 2)
            detached = detach(scratch)
            if consume is None:
                result.append(detached)
            else:
                consume(policy, detached)
        scratch['individual']['state'][0][0] = -99.0
        return result if consume is None else None


def test_mixed_collect_detaches_each_policy_and_caches_whole_result():
    model = _model()
    model._plan['collect'] = {'over': 'point_id', 'object_outputs': 'collect_outputs'}
    model._plan['point_table'] = 'points'
    model._tables = {'points': pd.DataFrame(index=pd.Index([10, 20], name='point_id'))}
    model._kernel = _ObjectKernel()
    result = model._projection.collect()
    assert list(result.individual) == [10, 20]
    assert result.individual[10].state[0][0] == -1.0
    assert result.individual[20].state[0][0] == 0.0
    assert result.sum.net_cf == {0: 3.0, 1: 5.0}
    assert model._projection.collect() is result
    assert model._kernel.calls == [('collect_outputs', -1)]
    changed = model._projection.collect(3)
    assert changed.individual[10].state[0][0] == 3.0
    assert model._kernel.scenario == -1
    assert result.individual[10].state[0][0] == -1.0


def test_mixed_collect_failure_does_not_cache_and_allows_retry():
    model = _model()
    model._plan['collect'] = {'over': 'point_id', 'object_outputs': 'collect_outputs'}
    model._plan['point_table'] = 'points'
    model._tables = {'points': pd.DataFrame(index=pd.Index([10, 20], name='point_id'))}
    model._kernel = _ObjectKernel()
    model._kernel.fail = True
    with pytest.raises(ValueError, match='calculation failed'):
        model._projection.collect(3)
    assert model._kernel.scenario == -1
    assert model._collect_cache == {}
    model._kernel.fail = False
    assert model._projection.collect(3).sum.net_cf == {0: 3.0, 1: 5.0}


def test_collect_detach_rejects_retained_spaces():
    model = _model()
    with pytest.raises(TypeError, match='live Space'):
        _detach_collect_value({'bad': model._projection})


def test_collect_detach_reuses_exact_immutable_scalars_but_copies_subclasses():
    copied = []

    class TaggedFloat(float):
        def __deepcopy__(self, memo):
            copied.append(float(self))
            return TaggedFloat(self)

    value = {'sum': {'cash': {0: 1.25, 1: TaggedFloat(2.5)}},
             'individual': {'state': [3.0]}}
    detached = _detach_collect_value(value)
    assert detached is not value
    assert detached['sum']['cash'] is not value['sum']['cash']
    assert detached['sum']['cash'][0] is value['sum']['cash'][0]
    assert detached['sum']['cash'][1] is not value['sum']['cash'][1]
    assert detached['individual']['state'] is not value['individual']['state']
    assert copied == [2.5]


def test_collect_object_sections_check_shape_and_keep_order():
    accumulator = _CollectAccumulator()
    accumulator.add(5, {'individual': {'state': {0: 'first'}},
                        'sum': {'cash': {0: 1.0}}})
    accumulator.add(3, {'individual': {'state': {0: 'second'}},
                        'sum': {'cash': {0: 2.0}}})
    result = accumulator.finish()
    assert list(result.individual) == [5, 3]
    assert result.individual[3].state == {0: 'second'}
    assert result.sum.cash == {0: 3.0}

    with pytest.raises(ValueError, match='structure changed'):
        accumulator.add(7, {'individual': {'state': {1: 'changed'}},
                            'sum': {'cash': {0: 3.0}}})
    with pytest.raises(ValueError, match='structure changed'):
        another = _CollectAccumulator()
        another.add(5, {'sum': {'cash': {0: 1.0}}})
        another.add(3, {'sum': {'cash': {1: 2.0}}})
