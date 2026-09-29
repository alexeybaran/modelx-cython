"""Object-valued Cells must not disable native numeric table helpers."""

from types import ModuleType

import pandas as pd

from modelx_cython._frame_runtime.contracts import cell
from modelx_cython._frame_runtime.compiler import emit
from modelx_cython._frame_runtime.frame_codegen import to_c_frame
from modelx_cython._frame_runtime.gil_codegen import augment
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
