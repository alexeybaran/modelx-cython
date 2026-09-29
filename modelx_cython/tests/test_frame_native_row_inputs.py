"""Declared row-reference values can feed one native numeric Cell cache."""
import ast
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from modelx_cython._frame_runtime.compiler import FrontendRewrite, build, emit
from modelx_cython._frame_runtime.contracts import ContractError, KernelSpec, cell
from modelx_cython._frame_runtime.lookups import IndexedFloat1
from modelx_cython._frame_runtime.modelx_frontend_native_mode import from_modelx
from modelx_cython._frame_runtime.native_input_plan import expand
from modelx_cython._frame_runtime.model import CompiledModel


@cell
def _amount(t):
    j = t
    return paths['premium'][j] + model_point['rate'] + spread + state[j]


@cell
def _outputs():
    return {'sum': {'amount': _amount(1)}}


def _spec():
    module = ModuleType('test_native_row_inputs')
    module._amount = _amount
    module._outputs = _outputs
    names = ('paths', 'model_point', 'spread', 'state')
    return KernelSpec(
        module=module, inputs={name: object for name in names},
        outputs={}, steps=4, length='0', cell_outputs=('_outputs',),
        cython_objects=True,
        typed_mapping_inputs={
            'paths': {'premium': 'float64_1d'},
            'model_point': {'rate': 'float'},
        },
        model={'bindings': {
            name: {'kind': 'row_reference', 'table': 'points', 'column': name}
            for name in names}, 'point_table': 'points'},
    )


def test_declared_numeric_reads_use_one_native_closure():
    spec = expand(_spec(), {
        '_amount': {'parameters': {'t': 'int'}, 'return': 'float'},
    }, {'spread': 'float', 'state': 'float64_1d'})
    assert spec.inputs['mxinput_paths_premium'] is IndexedFloat1
    assert spec.inputs['mxinput_model_point_rate'] is float
    assert spec.model['bindings']['mxinput_state']['field'] is None
    assert spec.cells['_amount'].__annotations__ == {'t': int, 'return': float}
    assert _amount.__annotations__ == {}
    source, manifest = emit(spec)
    assert manifest['cells']['_amount']['calculator'] == 'native'
    assert manifest['cells']['_outputs']['calculator'] == 'python'
    assert 'mxinput_paths_premium.get(j)' in source
    assert 'mxinput_model_point_rate' in source
    assert 'mxinput_spread' in source
    assert 'mxinput_state.get(j)' in source


def test_undeclared_or_nonintegral_array_access_keeps_checked_path():
    frontend = {'native_mapping_fields': {'paths': {'premium': 'mxinput_paths_premium'}}}
    inputs = {'mxinput_paths_premium': IndexedFloat1}
    with pytest.raises(ContractError, match='integral index'):
        FrontendRewrite(frontend, inputs, integer_names={'t'},
                        global_names={'paths'}).visit(
            ast.parse("paths['premium'][t / 2]", mode='eval'))
    untouched = FrontendRewrite(frontend, inputs, integer_names={'t'},
                                global_names=()).visit(
        ast.parse("paths['premium'][t]", mode='eval'))
    assert ast.unparse(untouched) == "paths['premium'][t]"


def test_non_collect_space_can_select_nonfirst_policy_parameter(tmp_path):
    def policy_formula(period, point_id):
        return None

    def source_cell(source):
        return SimpleNamespace(formula=SimpleNamespace(source=source),
                               is_cached=True, allow_none=False, parent=None)

    records = pd.DataFrame({'rate': [0.1, 0.2]},
                           index=pd.Index([1, 2], name='point_id'))
    space = SimpleNamespace(
        parameters=('period', 'point_id'), spaces={},
        formula=SimpleNamespace(func=policy_formula,
                                source='def policy_formula(period, point_id): return None'),
        refs={'rate': object()},
        cells={'value': source_cell('def value(): return rate')},
        _impl=SimpleNamespace(collect_over_spec=None))
    source = SimpleNamespace(name='DirectTwoParameters', Projection=space)
    spec = from_modelx(
        source, tmp_path / 'source', policy_records=records,
        point_parameter='point_id', parameter_initial_values={'period': 0},
        row_references={'rate': 'rate'},
        signatures={'value': {'return': 'float', 'parameters': {}}})
    assert spec.model['point_parameter'] == 'point_id'
    assert spec.model.get('collect') is None
    assert spec.model['required_parameters'] == ['period']


def test_native_row_inputs_keep_numeric_results_and_checked_rebinding(tmp_path):
    """Exercise the built Cell, object output bridge, and atomic input replacement."""
    def source_cell(source):
        return SimpleNamespace(formula=SimpleNamespace(source=source),
                               is_cached=True, allow_none=False, parent=None)

    def policy_formula(point_id):
        return None

    records = pd.DataFrame({
        'model_point': [{'rate': 2.0}, {'rate': 5.0}],
        'paths': [{'premium': np.array([0.25, 0.5], dtype=np.float64)},
                  {'premium': np.array([0.75, 1.0], dtype=np.float64)}],
        'spread': [0.1, 0.2],
        'state': [np.array([0.3, 0.4], dtype=np.float64),
                  np.array([0.6, 0.7], dtype=np.float64)],
    }, index=pd.Index([1, 2], name='point_id'))
    space = SimpleNamespace(
        parameters=('point_id',), spaces={},
        formula=SimpleNamespace(func=policy_formula,
                                source='lambda point_id: None'),
        refs={name: object() for name in ('model_point', 'paths', 'spread', 'state')},
        cells={
            'amount': source_cell(
                "def amount(t):\n"
                "    j = t\n"
                "    return paths['premium'][j] + model_point['rate'] + spread + state[j]\n"),
            'collect_outputs': source_cell(
                "lambda: {'sum': {'amount': amount(1)}, "
                "'individual': {'amount': amount(1)}}"),
        },
        _impl=SimpleNamespace(collect_over_spec={
            'point_id': SimpleNamespace(value=lambda _: (1, 2))}))
    source = SimpleNamespace(name='NativeRowInputs', Projection=space)
    spec = from_modelx(
        source, tmp_path / 'source', policy_records=records,
        row_references={name: name for name in ('model_point', 'paths', 'spread', 'state')},
        signatures={'amount': {'return': 'float', 'parameters': {'t': 'int'}}},
        typed_mapping_inputs={'model_point': {'rate': 'float'},
                              'paths': {'premium': 'float64_1d'}},
        native_typed_inputs=True,
        typed_object_inputs={'spread': 'float', 'state': 'float64_1d'})
    artifact = tmp_path / 'artifact'
    build(spec, artifact)
    assert spec.cells['amount'].__annotations__ == {'t': int, 'return': float}
    assert spec.cells['collect_outputs'].__annotations__ == {}
    model = CompiledModel(artifact, spec.initial_tables)
    try:
        expected = (0.5 + 2.0 + 0.1 + 0.4, 1.0 + 5.0 + 0.2 + 0.7)
        assert model.Projection[1].amount(1) == pytest.approx(expected[0])
        result = model.Projection.collect()
        assert result.sum.amount == pytest.approx(sum(expected))
        assert [result.individual[point].amount for point in (1, 2)] == pytest.approx(expected)
        assert model.Projection[1].amount(1) == pytest.approx(expected[0])

        changed = records.copy(deep=True)
        changed.at[2, 'model_point'] = {'rate': 7.0}
        model._set_table('mxcf_policy_records', changed)
        assert model.Projection.collect().sum.amount == pytest.approx(sum(expected) + 2.0)

        invalid = changed.copy(deep=True)
        invalid.at[2, 'paths'] = {'premium': np.array([0.75, 1.0], dtype=np.float32)}
        with pytest.raises((TypeError, ValueError, ContractError)):
            model._set_table('mxcf_policy_records', invalid)
        # A rejected replacement never publishes a partial native binding.
        assert model.Projection.collect().sum.amount == pytest.approx(sum(expected) + 2.0)
    finally:
        model.close()
