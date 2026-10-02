"""Original-shell resource preparation preserves binding and ownership contracts."""
from types import SimpleNamespace

import pandas as pd
import pytest

from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.lookups import Curve
from modelx_cython._frame_runtime.prepared_space_inputs import PreparedSpaceInputs


def manifest():
    return {
        'lookups': 'native',
        'inputs': {'premium': {'kind': 'float'}, 'discount': {'kind': 'curve'},
                   'rate': {'kind': 'float'}, 'scenario': {'kind': 'int'}},
        'model': {
            'tables': {
                'points': {'native': False},
                'discount': {'native': True, 'index_names': ['time'],
                             'columns': {'value': 'numeric'}}},
            'point_table': 'points',
            'reference_defaults': {'rate': 0.02},
            'parameter_defaults': {'scenario': -1},
            'bindings': {
                'premium': {'kind': 'field', 'table': 'points', 'column': 'premium'},
                'discount': {'kind': 'curve', 'table': 'discount', 'column': 'value'},
                'rate': {'kind': 'reference', 'name': 'rate'},
                'scenario': {'kind': 'parameter', 'name': 'scenario'}}}}


def tables():
    return {'points': pd.DataFrame({'premium': [1.5, -1.0]}, index=[11, 12]),
            'discount': pd.DataFrame({'value': [1.0, 0.95]},
                                     index=pd.Index([0, 1], name='time'))}


def test_original_owners_share_resources_but_keep_independent_records():
    inputs = PreparedSpaceInputs(manifest(), SimpleNamespace(Curve=Curve), tables())
    first = inputs.record(0, parameters={'scenario': 2})
    second = inputs.record(1, parameters={'scenario': 3})
    assert first[0] == 1.5 and second[0] == -1.0
    assert first[1] is second[1]
    assert first[3] == 2 and second[3] == 3
    assert inputs.record(0)[3] == -1


def test_replacement_is_atomic_and_does_not_change_existing_owner_record():
    source = tables()
    inputs = PreparedSpaceInputs(manifest(), SimpleNamespace(Curve=Curve), source)
    held = inputs.record(0)
    with pytest.raises(ContractError, match='finite real'):
        inputs.replace(source, references={'rate': float('nan')})
    assert inputs.record(0) is held


    replacement = tables()
    replacement['points'].iloc[0, 0] = 7.0
    inputs.replace(replacement, references={'rate': 0.03})
    assert inputs.record(0)[0] == 7.0 and inputs.record(0)[2] == 0.03
    assert held[0] == 1.5 and held[2] == 0.02
    assert held[1] is not inputs.record(0)[1]


@pytest.mark.parametrize('position', [-1, 2, True, 0.5])
def test_position_is_a_validated_native_selection_not_python_negative_index(position):
    inputs = PreparedSpaceInputs(manifest(), SimpleNamespace(Curve=Curve), tables())
    with pytest.raises(IndexError, match='Policy position'):
        inputs.record(position)


def test_prepared_snapshot_and_parameter_validation_match_existing_contract():
    source = tables()
    inputs = PreparedSpaceInputs(manifest(), SimpleNamespace(Curve=Curve), source)
    held = inputs.record(0)
    source['discount'].iloc[1, 0] = 0.1
    assert held[1].at(1) == 0.95
    with pytest.raises(ContractError, match='integral scalar'):
        inputs.record(0, parameters={'scenario': 0.5})
    with pytest.raises(ContractError, match='Unknown dynamic'):
        inputs.record(0, parameters={'unknown': 1})
    assert inputs.record(0) is held
