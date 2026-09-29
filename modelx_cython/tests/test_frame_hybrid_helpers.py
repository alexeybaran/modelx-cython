"""Object-valued Cells must not disable native numeric table helpers."""

from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from modelx_cython._frame_runtime.contracts import cell
from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.compiler import emit
from modelx_cython._frame_runtime.compiler import build
from modelx_cython._frame_runtime.frame_codegen import to_c_frame
from modelx_cython._frame_runtime.gil_codegen import augment
from modelx_cython._frame_runtime.model import CompiledModel
from modelx_cython._frame_runtime.model_plan import derive_spec
from modelx_cython._frame_runtime.modelx_frontend_native_mode import from_modelx


def _space_formula(point_id):
    return None


@cell
def rate() -> float:
    return lookup.field(data.rates(), point_id, 'r')


@cell
def report() -> object:
    return {'rate': rate()}


@cell
def numeric_total() -> float:
    return rate()


@cell
def report_total() -> float:
    return float(report()['rate'])


@cell
def point_value() -> float:
    return lookup.field(data.points(), point_id, 'value')


@cell
def point_object() -> object:
    return {'value': point_value()}


@cell
def point_object_value() -> float:
    return point_object()['value']


@cell
def row_amount() -> float:
    return model_point['amount']


@cell
def row_snapshot() -> object:
    return {'amount': model_point['amount']}


@cell
def collect_outputs() -> object:
    return {
        'individual': {'state': {t: model_point['amount'] + t for t in periods}},
        'sum': {'net_cf': {t: row_amount() + t for t in periods}},
    }


def test_numeric_lookup_stays_native_beside_object_cell():
    module = ModuleType('frame_hybrid_fixture')
    module.rate = rate
    module.report = report
    rates = pd.DataFrame({'r': [0.01, 0.02]},
                         index=pd.Index([1, 2], name='point_id'))
    spec = derive_spec(module, {'rates': rates}, references={},
                       space_formula=_space_formula, outputs={},
                       steps=3, length='0', cell_outputs=('rate', 'report'),
                       point_table='rates', cython_objects=True)
    _, manifest = emit(spec)
    assert spec.frontend['fields']['rates']['r'] in spec.inputs
    assert manifest['cells']['rate']['calculator'] == 'native'
    assert manifest['cells']['report']['calculator'] == 'python'


def test_native_scalar_traversal_skips_only_unneeded_object_binding():
    module = ModuleType('frame_mixed_scalar_fixture')
    module.rate = rate
    module.report = report
    module.numeric_total = numeric_total
    module.report_total = report_total
    rates = pd.DataFrame({'r': [0.01, 0.02]},
                         index=pd.Index([1, 2], name='point_id'))
    spec = derive_spec(module, {'rates': rates}, references={},
                       space_formula=_space_formula, outputs={},
                       steps=3, length='0',
                       scalar_outputs=('numeric_total', 'report_total'),
                       cell_outputs=('report',), point_table='rates',
                       cython_objects=True)
    source, manifest = emit(spec)
    assert manifest['native_scalar_outputs'] == ['numeric_total']
    source = augment(to_c_frame(source, manifest), manifest, spec)
    assert 'cdef void _select_native_policy' in source
    assert "cdef bint native_only = name == 'numeric_total'" in source
    assert 'if native_only:\n                self._select_native_policy(i)' in source


def test_native_scalar_selection_does_not_stale_object_receivers(tmp_path):
    module = ModuleType('frame_mixed_scalar_runtime_fixture')
    module.__file__ = __file__
    module.point_value = point_value
    module.point_object = point_object
    module.point_object_value = point_object_value
    table = pd.DataFrame({'value': [1., 2., 3.]},
                         index=pd.Index([0, 1, 2], name='point_id'))
    spec = derive_spec(module, {'points': table}, references={},
                       space_formula=_space_formula, outputs={}, steps=1,
                       length='0', scalar_outputs=('point_value', 'point_object_value'),
                       cell_outputs=('point_object',), point_table='points',
                       cython_objects=True)
    artifact = tmp_path / 'native_scalar_binding'
    build(spec, artifact)
    model = CompiledModel(artifact, {'points': table})
    try:
        assert list(model._kernel.run_scalar('point_value')) == [1., 2., 3.]
        assert list(model._kernel.run_scalar('point_object_value')) == [1., 2., 3.]
        assert list(model._kernel.run_scalar('point_value')) == [1., 2., 3.]
        detached = []
        def detach(value):
            detached.append(value['value'])
            return dict(value)
        assert model._kernel._run_object_outputs('point_object', detach) == [
            {'value': 1.}, {'value': 2.}, {'value': 3.}]
        assert detached == [1., 2., 3.]
        def fail_on_second(value):
            if value['value'] == 2.:
                raise ValueError('detachment failed')
            return dict(value)
        with pytest.raises(ValueError, match='detachment failed'):
            model._kernel._run_object_outputs('point_object', fail_on_second)
        assert model._kernel._run_object_outputs('point_object', dict) == [
            {'value': 1.}, {'value': 2.}, {'value': 3.}]
        assert list(model._kernel.run_scalar('point_value')) == [1., 2., 3.]
    finally:
        model.close()


def test_per_policy_object_reference_follows_binding_and_native_traversal(tmp_path):
    module = ModuleType('frame_per_policy_reference_fixture')
    module.__file__ = __file__
    module.row_amount = row_amount
    module.row_snapshot = row_snapshot
    points = pd.DataFrame({'model_point': [
        {'amount': 1.25}, {'amount': 2.5}, {'amount': 3.75}]},
        index=pd.Index([10, 11, 12], name='point_id'))
    spec = derive_spec(module, {'points': points}, references={},
                       space_formula=_space_formula, outputs={}, steps=1,
                       length='0', scalar_outputs=('row_amount',),
                       cell_outputs=('row_snapshot',), point_table='points',
                       row_references={'model_point': 'model_point'},
                       cython_objects=True)
    assert spec.model['bindings']['model_point'] == {
        'kind': 'row_reference', 'table': 'points', 'column': 'model_point'}
    artifact = tmp_path / 'per_policy_reference'
    build(spec, artifact)
    model = CompiledModel(artifact, {'points': points})
    try:
        assert model.Projection[10].model_point == {'amount': 1.25}
        assert list(model._kernel.run_scalar('row_amount')) == [1.25, 2.5, 3.75]
        assert model._kernel._run_object_outputs('row_snapshot', dict) == [
            {'amount': 1.25}, {'amount': 2.5}, {'amount': 3.75}]
        updated = points.copy(deep=True)
        updated.at[11, 'model_point'] = {'amount': 8.0}
        model._set_table('points', updated)
        assert model.Projection[11].model_point == {'amount': 8.0}
        assert list(model._kernel.run_scalar('row_amount')) == [1.25, 8.0, 3.75]
        assert model._kernel._run_object_outputs('row_snapshot', dict)[1] == {'amount': 8.0}
    finally:
        model.close()


def test_importer_accepts_prepared_policy_reference_context(tmp_path):
    def source_cell(source):
        return SimpleNamespace(formula=SimpleNamespace(source=source),
                               is_cached=True, allow_none=False, parent=None)
    space = SimpleNamespace(
        parameters=('point_id',), spaces={},
        formula=SimpleNamespace(func=_space_formula),
        refs={'model_point': {'amount': -1.0},
              'ScenarioInputs': object()},
        cells={'row_amount': source_cell(
            "def row_amount() -> float:\n    return model_point['amount']\n")},
        _impl=SimpleNamespace(collect_over_spec=None))
    source = SimpleNamespace(name='FrameRows', Projection=space)
    points = pd.DataFrame({'model_point': [{'amount': 4.0}, {'amount': 5.0}]},
                          index=pd.Index([0, 1], name='point_id'))
    spec = from_modelx(source, tmp_path / 'source', policy_records=points,
                       row_references={'model_point': 'model_point'})
    assert spec.model['point_table'] == 'mxcf_policy_records'
    assert spec.model['bindings']['model_point']['kind'] == 'row_reference'
    assert spec.initial_tables['mxcf_policy_records'].equals(points)
    assert 'ScenarioInputs' not in spec.inputs


def test_second_required_space_parameter_is_explicit_dv01_blocker(tmp_path):
    def valuation_formula(valuation_month, point_id, shock=-1):
        return None
    space = SimpleNamespace(
        parameters=('valuation_month', 'point_id', 'shock'), spaces={},
        formula=SimpleNamespace(func=valuation_formula), refs={}, cells={},
        _impl=SimpleNamespace(collect_over_spec={'point_id': object()}))
    source = SimpleNamespace(name='FrameValuation', Valuation=space)
    records = pd.DataFrame(index=pd.Index([1, 2], name='point_id'))
    with pytest.raises(ContractError, match='Additional Space parameters need literal defaults'):
        from_modelx(source, tmp_path / 'source', projection='Valuation',
                    policy_records=records)


def test_compiled_mixed_collect_uses_one_source_formula_and_detaches(tmp_path):
    module = ModuleType('frame_dv01_shaped_fixture')
    module.__file__ = __file__
    module.row_amount = row_amount
    module.collect_outputs = collect_outputs
    points = pd.DataFrame({
        'model_point': [{'amount': 1.0}, {'amount': 2.0}, {'amount': 3.0}],
        'periods': [(0, 1, 2)] * 3,
    }, index=pd.Index([10, 20, 30], name='point_id'))
    spec = derive_spec(module, {'points': points}, references={},
                       space_formula=_space_formula, outputs={}, steps=1,
                       length='0', cell_outputs=('collect_outputs',),
                       point_table='points',
                       row_references={'model_point': 'model_point',
                                       'periods': 'periods'},
                       cython_objects=True)
    spec.model['collect'] = {'over': 'point_id',
                             'object_outputs': 'collect_outputs'}
    artifact = tmp_path / 'mixed_collect'
    build(spec, artifact)
    model = CompiledModel(artifact, {'points': points})
    try:
        result = model.Projection.collect()
        assert result.sum.net_cf == {0: 6.0, 1: 9.0, 2: 12.0}
        assert list(result.individual) == [10, 20, 30]
        assert result.individual[10].state == {0: 1.0, 1: 2.0, 2: 3.0}
        assert result.individual[30].state == {0: 3.0, 1: 4.0, 2: 5.0}
        assert model.Projection.collect() is result
        streamed = []
        assert model._kernel._run_object_outputs(
            'collect_outputs', dict,
            lambda i, value: streamed.append((i, value['sum']['net_cf'][0]))) is None
        assert streamed == [(0, 1.0), (1, 2.0), (2, 3.0)]
        updated = points.copy(deep=True)
        updated.at[20, 'model_point'] = {'amount': 5.0}
        model._set_table('points', updated)
        again = model.Projection.collect()
        assert again.sum.net_cf == {0: 9.0, 1: 12.0, 2: 15.0}
        assert result.individual[20].state == {0: 2.0, 1: 3.0, 2: 4.0}
        assert again.individual[20].state == {0: 5.0, 1: 6.0, 2: 7.0}
    finally:
        model.close()
