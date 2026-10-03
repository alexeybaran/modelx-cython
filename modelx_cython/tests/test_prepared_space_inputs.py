"""Original-shell resource preparation preserves binding and ownership contracts."""
from types import SimpleNamespace

import pandas as pd
import pytest

from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.lookups import Curve
from modelx_cython._frame_runtime.prepared_space_inputs import PreparedSpaceInputs
from modelx_cython._frame_runtime.prepared_space_inputs import PreparedReceiverInputs


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


def test_dynamic_receiver_rows_keep_owned_views_and_check_scalar_domains():
    import numpy as np
    class View:
        def __init__(self, values): self.values = values
    manifest = {'inputs':{'policy':{'kind':'int'},'amount':{'kind':'float'},
        'path':{'kind':'indexed_array','ctype':'IndexedFloat1'},
        'objects':{'kind':'object','python_type':'dict'}},
        'model':{'point_parameter':'policy_id','bindings':{
            'policy':{'kind':'point'},
            'amount':{'kind':'row_reference_numeric','column':'point','field':'amount'},
            'path':{'kind':'row_reference_numeric','column':'paths','field':'rate'},
            'objects':{'kind':'row_reference','column':'point'}}}}
    source = np.array([.01,.02])
    receiver = SimpleNamespace(policy_id=7,point={'amount':1.5},paths={'rate':source})
    resources = PreparedReceiverInputs(manifest,SimpleNamespace(IndexedFloat1=View))
    first = resources.record(receiver)
    assert first[0:2] == (7,1.5)
    assert first[2].values is source and first[3] is receiver.point
    receiver.point={'amount':-1.0}
    second = resources.record(receiver)
    assert second[1] == -1.0 and first[1] == 1.5
    receiver.point={'amount':float('nan')}
    with pytest.raises(ContractError,match='finite real'):
        resources.record(receiver)
    assert first[2].values is source


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


def test_typed_builtin_binding_keeps_value_checks_and_custom_conversions():
    import numpy as np
    from enum import IntEnum
    from modelx_cython._frame_runtime.input_codec import NativeInputCodec
    convert = NativeInputCodec._scalar_value
    calls = []
    class CustomInt(int):
        def __int__(self):
            calls.append('int')
            return 7
    class CustomFloat(float):
        def __float__(self):
            calls.append('float')
            return 2.5
    class Category(IntEnum):
        NEGATIVE = -1
    assert convert('n', {'kind': 'int'}, -(1 << 63)) == -(1 << 63)
    assert convert('n', {'kind': 'int'}, (1 << 63) - 1) == (1 << 63) - 1
    assert convert('n', {'kind': 'int'}, np.int64(4)) == 4
    assert convert('n', {'kind': 'int'}, CustomInt(3)) == 7
    assert convert('x', {'kind': 'float'}, CustomFloat(1.5)) == 2.5
    assert calls == ['int', 'float']
    category = {'kind': 'enum', 'members': {'NEGATIVE': -1}}
    assert convert('c', category, Category.NEGATIVE) == -1
    assert convert('c', category, -1) == -1
    with pytest.raises(ContractError, match='category domain'):
        convert('c', category, 0)
    for invalid in (True, np.bool_(False), 0.5):
        with pytest.raises(ContractError, match='integral scalar'):
            convert('n', {'kind': 'int'}, invalid)
    for invalid in (-(1 << 63) - 1, 1 << 63):
        with pytest.raises(ContractError, match='signed int64'):
            convert('n', {'kind': 'int'}, invalid)
    for invalid in (True, np.bool_(False), float('nan'), float('inf')):
        with pytest.raises(ContractError, match='finite real'):
            convert('x', {'kind': 'float'}, invalid)
