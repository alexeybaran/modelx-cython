"""Object-valued Cells must not disable native numeric table helpers."""

from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from modelx_cython._frame_runtime.contracts import cell
from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.compiler import emit
from modelx_cython._frame_runtime.compiler import build
from modelx_cython._frame_runtime.frame_codegen import emit_context
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
    parts, manifest = emit(spec)
    assert manifest['native_scalar_outputs'] == ['numeric_total']
    source = augment(emit_context(parts, manifest), manifest, spec)
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
    import json
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert manifest['cells']['row_amount']['coverage']['body_backend'] == 'cython_gil'
    assert 'row_amount' in manifest['cython_numeric_fallback_cells']
    assert 'def row_amount' not in next(artifact.glob('mxk_*/_python.py')).read_text()
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
        formula=SimpleNamespace(func=_space_formula,
                                source='lambda point_id: None'),
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


def test_non_collect_space_uses_the_generated_direct_cell_entries(tmp_path):
    def source_cell(source):
        return SimpleNamespace(formula=SimpleNamespace(source=source),
                               is_cached=True, allow_none=False, parent=None)

    points = pd.DataFrame({'amount': [2.0, 5.0]},
                          index=pd.Index([1, 2], name='point_id'))
    space = SimpleNamespace(
        parameters=('point_id',), spaces={},
        formula=SimpleNamespace(func=_space_formula,
                                source='lambda point_id: None'),
        refs={}, cells={'amount': source_cell(
            'def amount() -> float:\n    return lookup.field(data.points(), point_id, "amount")\n')},
        _impl=SimpleNamespace(collect_over_spec=None))
    source = SimpleNamespace(name='NoCollect', Projection=space,
                             Data=SimpleNamespace(parameters=(), spaces={},
                                  refs={'points': points}, cells={}))
    spec = from_modelx(source, tmp_path / 'source',
                       point_table='points')
    assert spec.model.get('collect') is None
    artifact = tmp_path / 'artifact'
    build(spec, artifact)
    model = CompiledModel(artifact, spec.initial_tables)
    try:
        assert model.Projection[1].amount() == 2.0
        assert model.Projection[2].amount() == 5.0
        assert model.Projection[1].amount() == 2.0
        with pytest.raises(TypeError, match='collect'):
            model.Projection.collect()
    finally:
        model.close()


def test_uncached_object_cell_repeats_and_cached_parent_shares_its_result(tmp_path, monkeypatch):
    import sys
    import types

    calls = []
    tracker = types.ModuleType('mxcf_test_tracker')
    tracker.next_value = lambda: calls.append(1) or len(calls)
    monkeypatch.setitem(sys.modules, tracker.__name__, tracker)

    def source_cell(source, cached=True):
        return SimpleNamespace(formula=SimpleNamespace(source=source),
                               is_cached=cached, allow_none=False, parent=None)

    points = pd.DataFrame(index=pd.Index([1], name='point_id'))
    space = SimpleNamespace(
        parameters=('point_id',), spaces={},
        formula=SimpleNamespace(func=_space_formula,
                                source='lambda point_id: None'),
        refs={'tracker': tracker},
        cells={
            'transient': source_cell(
                'def transient() -> object:\n    return tracker.next_value()\n', False),
            'fixed': source_cell(
                'def fixed() -> object:\n    return transient()\n'),
            'transient_numeric': source_cell(
                'def transient_numeric() -> float:\n    return float(tracker.next_value())\n', False),
            'fixed_numeric': source_cell(
                'def fixed_numeric() -> float:\n    return transient_numeric()\n'),
        }, _impl=SimpleNamespace(collect_over_spec=None))
    source = SimpleNamespace(name='Uncached', Projection=space,
                             Data=SimpleNamespace(parameters=(), spaces={},
                                  refs={'points': points}, cells={}))
    spec = from_modelx(source, tmp_path / 'source', point_table='points')
    artifact = tmp_path / 'artifact'
    build(spec, artifact)
    assert spec.cells['transient'].__kernel_cached__ is False
    model = CompiledModel(artifact, spec.initial_tables)
    try:
        item = model.Projection[1]
        assert (item.transient(), item.transient()) == (1, 2)
        assert (item.fixed(), item.fixed()) == (3, 3)
        assert item.transient() == 4
        assert (item.transient_numeric(), item.transient_numeric()) == (5.0, 6.0)
        assert (item.fixed_numeric(), item.fixed_numeric()) == (7.0, 7.0)
        model.reset()
        assert item.fixed() == 8
    finally:
        model.close()


def test_declared_numeric_mapping_fields_keep_formula_source_and_rebind(tmp_path):
    def source_cell(source):
        return SimpleNamespace(formula=SimpleNamespace(source=source),
                               is_cached=True, allow_none=False, parent=None)

    def policy_formula(point_id):
        return None

    records = pd.DataFrame({
        'model_point': [{'amount': 2.0}, {'amount': 5.0}],
        'paths': [{'curve': np.array([0.25, 0.5], dtype=np.float64)},
                  {'curve': np.array([0.75, 1.0], dtype=np.float64)}],
    }, index=pd.Index([1, 2], name='point_id'))
    domain = SimpleNamespace(value=lambda _: (1, 2))
    space = SimpleNamespace(
        parameters=('point_id',), spaces={},
        formula=SimpleNamespace(func=policy_formula,
                                source='lambda point_id: None'),
        refs={'model_point': object(), 'paths': object()},
        cells={
            'amount': source_cell(
                "def amount(t):\n"
                "    return model_point['amount'] + paths['curve'][t]\n"),
            'collect_outputs': source_cell(
                "lambda: {'sum': {'amount': amount(0)}}"),
        },
        _impl=SimpleNamespace(collect_over_spec={'point_id': domain}))
    source = SimpleNamespace(name='TypedFields', Projection=space)
    spec = from_modelx(
        source, tmp_path / 'source', policy_records=records,
        row_references={'model_point': 'model_point', 'paths': 'paths'},
        signatures={'amount': {'return': 'float', 'parameters': {'t': 'int'}}},
        typed_mapping_inputs={
            'model_point': {'amount': 'float'},
            'paths': {'curve': 'float64_1d'},
        })
    artifact = tmp_path / 'artifact'
    build(spec, artifact)
    generated = next(artifact.glob('mxk_*/_kernel.pyx')).read_text()
    assert "self._gil_global_model_point['amount']" in generated  # one binding
    assert "self._gil_global_paths['curve']" in generated
    assert "self._gil_field_model_point_amount + self._gil_field_paths_curve[t]" in generated
    model = CompiledModel(artifact, spec.initial_tables)
    try:
        assert model.Projection.collect().sum.amount == 8.0
        assert model.Projection[1].amount(1) == 2.5
        changed = records.copy(deep=True)
        changed.at[2, 'model_point'] = {'amount': 7.0}
        model._set_table('mxcf_policy_records', changed)
        assert model.Projection.collect().sum.amount == 10.0
        invalid = changed.copy(deep=True)
        invalid.at[2, 'paths'] = {'curve': np.array([0.75, 1.0], dtype=np.float32)}
        model._set_table('mxcf_policy_records', invalid)
        with pytest.raises(ValueError):
            model.Projection.collect()
        model._set_table('mxcf_policy_records', changed)
        assert model.Projection.collect().sum.amount == 10.0
    finally:
        model.close()


def test_second_required_space_parameter_needs_initial_binding(tmp_path):
    def valuation_formula(valuation_month, point_id, shock=-1):
        return None
    space = SimpleNamespace(
        parameters=('valuation_month', 'point_id', 'shock'), spaces={},
        formula=SimpleNamespace(func=valuation_formula,
                                source='lambda valuation_month, point_id, shock=-1: None'),
        refs={}, cells={},
        _impl=SimpleNamespace(collect_over_spec={'point_id': object()}))
    source = SimpleNamespace(name='FrameValuation', Valuation=space)
    records = pd.DataFrame(index=pd.Index([1, 2], name='point_id'))
    with pytest.raises(ContractError, match='Required non-policy Space parameters need initial values'):
        from_modelx(source, tmp_path / 'source', projection='Valuation',
                    policy_records=records)


def test_collect_policy_selector_can_follow_required_space_parameter(tmp_path):
    def valuation_formula(valuation_month, point_id, shock=-1):
        return None

    def source_cell(source):
        return SimpleNamespace(formula=SimpleNamespace(source=source),
                               is_cached=True, allow_none=False, parent=None)

    records = pd.DataFrame({'model_point': [{'amount': 1.0}, {'amount': 2.0}]},
                           index=pd.Index([1, 2], name='point_id'))
    domain = SimpleNamespace(value=lambda _: (1, 2))
    space = SimpleNamespace(
        parameters=('valuation_month', 'point_id', 'shock'), spaces={},
        formula=SimpleNamespace(func=valuation_formula,
                                source='lambda valuation_month, point_id, shock=-1: None'),
        refs={'model_point': object()},
        cells={
            'row_amount': source_cell(
                "def row_amount() -> float:\n"
                "    return float(model_point['amount'] + valuation_month + shock)\n"),
            'collect_outputs': source_cell(
                "lambda: {'sum': {'row_amount': row_amount()}}"),
        },
        _impl=SimpleNamespace(collect_over_spec={'point_id': domain}))
    source = SimpleNamespace(name='FrameValuation', Valuation=space)
    spec = from_modelx(source, tmp_path / 'source', projection='Valuation',
                       policy_records=records,
                       row_references={'model_point': 'model_point'},
                       parameter_initial_values={'valuation_month': 0})
    assert spec.model['point_parameter'] == 'point_id'
    assert spec.model['required_parameters'] == ['valuation_month']
    artifact = tmp_path / 'valuation'
    build(spec, artifact)
    model = CompiledModel(artifact, spec.initial_tables)
    try:
        with pytest.raises(TypeError):
            model.Valuation.collect()
        assert model.Valuation[5, 1, -1].row_amount() == 5.0
        assert model.Valuation.collect(5).sum.row_amount == 11.0
        assert model.Valuation.collect(5, 0).sum.row_amount == 13.0
        assert model.Valuation.collect(5).sum.row_amount == 11.0
        changed = records.copy(deep=True)
        changed.at[2, 'model_point'] = {'amount': 4.0}
        model._set_table('mxcf_policy_records', changed)
        assert model.Valuation.collect(5).sum.row_amount == 13.0
    finally:
        model.close()


def test_compiled_source_space_formula_uses_explicit_connected_references(tmp_path):
    def valuation_formula(valuation_month, point_id, shock=-1):
        return None

    def source_cell(source):
        return SimpleNamespace(formula=SimpleNamespace(source=source),
                               is_cached=True, allow_none=False, parent=None)

    formula_source = (
        'def _formula(valuation_month, point_id, shock=-1):\n'
        '    curve = ScenarioInputs[valuation_month]\n'
        "    return {'refs': {'curve': curve, 'value': Actual[point_id] + shock}}\n")
    space = SimpleNamespace(
        parameters=('valuation_month', 'point_id', 'shock'), spaces={},
        formula=SimpleNamespace(func=valuation_formula, source=formula_source),
        refs={'ScenarioInputs': object(), 'Actual': object()},
        cells={
            'row_amount': source_cell('def row_amount() -> float:\n    return float(value)\n'),
            'collect_outputs': source_cell(
                "lambda: {'sum': {'row_amount': row_amount()}}"),
        },
        _impl=SimpleNamespace(collect_over_spec={'point_id':
            SimpleNamespace(value=lambda _: (1, 2))}))
    source = SimpleNamespace(name='FrameValuationSource', Valuation=space)
    records = pd.DataFrame({'value': [0.0, 0.0], 'curve': [None, None]},
                           index=pd.Index([1, 2], name='point_id'))
    spec = from_modelx(source, tmp_path / 'source', projection='Valuation',
                       policy_records=records,
                       row_references={'value': 'value', 'curve': 'curve'},
                       parameter_initial_values={'valuation_month': 0})
    assert spec.model['space_formula']['references'] == ['Actual', 'ScenarioInputs']
    artifact = tmp_path / 'valuation_space_formula'
    build(spec, artifact)
    connected = {'Actual': {1: 10, 2: 20},
                 'ScenarioInputs': {3: (0.1, 0.2)}}
    model = CompiledModel(artifact, spec.initial_tables,
                          space_bindings=connected)
    try:
        refs = model._space_formula_refs(3, 2, bindings=connected)
        assert refs == {'curve': (0.1, 0.2), 'value': 19}
        with pytest.raises(TypeError):
            model._space_formula_refs(2, bindings={
                'Actual': {}, 'ScenarioInputs': {}})
        with pytest.raises(ContractError, match='bindings differ'):
            model._space_formula_refs(3, 2, bindings={'Actual': {2: 20}})
        direct = model.Valuation[3, 1]
        assert direct.curve == (0.1, 0.2)
        assert direct.row_amount() == 9.0
        direct_context = direct._context
        assert model.Valuation[3, 1] is direct
        first = model.Valuation.collect(3)
        assert first.sum.row_amount == 28.0
        assert model.Valuation.collect(3) is first
        assert model.Valuation.collect(3, 0).sum.row_amount == 30.0
        assert model.Valuation.collect(3) is first
        assert direct._context is direct_context
        assert direct.row_amount() == 9.0
        assert model.Valuation[3, 1, 0].row_amount() == 10.0
        with pytest.raises(KeyError):
            model.Valuation.collect(99)
        assert model.Valuation.collect(3) is first

        class ReentrantActual:
            def __getitem__(self, point_id):
                model.Valuation.collect(3, 1)
                return 0

        model._space_bindings['Actual'] = ReentrantActual()
        with pytest.raises(RuntimeError, match='active Cell evaluation'):
            model.Valuation.collect(3, 1)
        model._space_bindings['Actual'] = connected['Actual']
        assert model.Valuation.collect(3, 1).sum.row_amount == 32.0
        assert model.Valuation.collect(3) is first
        connected['Actual'][1] = 11
        connected['ScenarioInputs'][3] = (0.3, 0.4)
        model.reset()
        assert model.Valuation[3, 1] is direct
        assert direct.curve == (0.3, 0.4)
        assert direct.row_amount() == 10.0
        assert model.Valuation.collect(3).sum.row_amount == 29.0
    finally:
        model.close()


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
