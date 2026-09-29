"""Object-valued Cells must not disable native numeric table helpers."""

from types import ModuleType

import pandas as pd

from modelx_cython._frame_runtime.contracts import cell
from modelx_cython._frame_runtime.compiler import emit
from modelx_cython._frame_runtime.compiler import build
from modelx_cython._frame_runtime.frame_codegen import to_c_frame
from modelx_cython._frame_runtime.gil_codegen import augment
from modelx_cython._frame_runtime.model import CompiledModel
from modelx_cython._frame_runtime.model_plan import derive_spec


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
    finally:
        model.close()
