"""Numeric map streaming preserves collect outputs and the checked fallback."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from modelx_cython.tests.reference_frame.compiler import build
from modelx_cython.tests.reference_frame.model import CompiledModel
from modelx_cython.tests.reference_frame.modelx_frontend_native_mode import from_modelx


def _source_cell(source):
    return SimpleNamespace(formula=SimpleNamespace(source=source),
                           is_cached=True, allow_none=False, parent=None)


def test_native_map_sum_keeps_public_shape_and_falls_back_for_duplicate_keys(tmp_path):
    def policy_formula(point_id):
        return None

    records = pd.DataFrame({
        'model_point': [{'rate': 1.0}, {'rate': 2.0}],
        'paths': [{'premium': np.array([0.5, 1.0], dtype=np.float64)},
                  {'premium': np.array([1.5, 2.0], dtype=np.float64)}],
    }, index=pd.Index([1, 2], name='point_id'))
    space = SimpleNamespace(
        parameters=('point_id',), spaces={},
        formula=SimpleNamespace(func=policy_formula,
                                source='lambda point_id: None'),
        refs={'model_point': object(), 'paths': object(), 'periods': (0, 1)},
        cells={
            'amount': _source_cell(
                "def amount(t):\n"
                "    return paths['premium'][t] + model_point['rate']\n"),
            'collect_outputs': _source_cell(
                "lambda: {'sum': {'amounts': {t: amount(t) for t in periods}}}"),
        },
        _impl=SimpleNamespace(collect_over_spec={
            'point_id': SimpleNamespace(value=lambda _: (1, 2))}))
    source = SimpleNamespace(name='NumericMap', Projection=space)
    spec = from_modelx(
        source, tmp_path / 'source', policy_records=records,
        row_references={'model_point': 'model_point', 'paths': 'paths'},
        object_bindings=('periods',),
        signatures={'amount': {'return': 'float', 'parameters': {'t': 'int'}}},
        typed_mapping_inputs={'model_point': {'rate': 'float'},
                              'paths': {'premium': 'float64_1d'}},
        native_typed_inputs=True)
    artifact = tmp_path / 'artifact'
    build(spec, artifact)
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['compiled_collect_map']['entries'] == [
        {'kind': 'map', 'name': 'amounts', 'cell': 'amount', 'source': 'periods'}]
    model = CompiledModel(artifact, spec.initial_tables, overrides={'periods': (0, 1)})
    try:
        expected = {0: 1.5 + 3.5, 1: 2.0 + 4.0}
        assert model._kernel._compiled_collect_keys() == {'periods': (0, 1)}
        fast = model.Projection.collect()
        assert fast.sum.amounts == pytest.approx(expected)
        assert model.Projection.collect() is fast
        assert model.Projection[1].amount(0) == pytest.approx(1.5)
        # The original-shell streaming hook uses this same accumulator after
        # its ordinary Space formula has selected each policy's native owner.
        kernel = model._kernel._kernel
        streamed = kernel.new_collect_accumulator({'periods': (0, 1)})
        assert streamed.matches({'periods': (0, 1)})
        assert not streamed.matches({'periods': (1, 0)})
        for policy in range(2):
            kernel.select_scope(policy)
            kernel.append_collect_accumulator(streamed)
        assert streamed.finish() == {'sum': {'amounts': expected}}
        detached = streamed.finish()
        detached['sum']['amounts'][0] = -1000.0
        assert streamed.finish()['sum']['amounts'][0] == expected[0]
        failed = kernel.new_collect_accumulator({'periods': (0, 2)})
        kernel.select_scope(0)
        with pytest.raises(IndexError):
            kernel.append_collect_accumulator(failed)
        with pytest.raises(ValueError, match='nonempty'):
            failed.finish()
        assert kernel.call('amount', (0,)) == pytest.approx(1.5)
        retry = kernel.new_collect_accumulator({'periods': (0, 1)})
        kernel.append_collect_accumulator(retry)
        assert retry.finish() == {'sum': {'amounts': {0: 1.5, 1: 2.0}}}

        # A tuple with duplicate keys has normal Python dict-comprehension
        # semantics and uses the original checked traversal.
        model.Projection.periods = (0, 0, 1)
        assert model._kernel._compiled_collect_keys() is None
        assert model.Projection.collect().sum.amounts == pytest.approx(expected)

        # The checked object fallback must also guard a failing native Cell.
        model.Projection.periods = (0, 0, 2)
        assert model._kernel._compiled_collect_keys() is None
        with pytest.raises(IndexError):
            model.Projection.collect()

        model.Projection.periods = (0, 1, 2)
        with pytest.raises(IndexError):
            model.Projection.collect()
        model.Projection.periods = (0, 1)
        assert model.Projection.collect().sum.amounts == pytest.approx(expected)
    finally:
        model.close()
