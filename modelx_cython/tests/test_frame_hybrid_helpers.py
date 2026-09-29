"""Object-valued Cells must not disable native numeric table helpers."""

from types import ModuleType

import pandas as pd

from modelx_cython._frame_runtime.contracts import cell
from modelx_cython._frame_runtime.compiler import emit
from modelx_cython._frame_runtime.model_plan import derive_spec


def _space_formula(point_id):
    return None


@cell
def rate() -> float:
    return lookup.field(data.rates(), point_id, 'r')


@cell
def report() -> object:
    return {'rate': rate()}


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
