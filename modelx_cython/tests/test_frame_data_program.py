"""Source-backed Data preparation keeps arrays aligned with policy positions."""
import numpy as np
import pandas as pd
import pytest

from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.data_program import prepare_data


def _program():
    return {
        'references': ['points'],
        'modules': {'np': 'numpy'},
        'cells': {
            'ages': 'def ages():\n    return points["age"].to_numpy(dtype="int64")',
            'twice': 'def twice():\n    return np.asarray(ages() * 2)',
        },
        'source_policy_table': 'points',
        'domain_name': 'policy_ids',
        'domain_source': 'def policy_ids():\n    return range(len(data.ages()))',
        'policy_table': 'mxcf_policy_domain',
    }


def test_derived_arrays_and_domain_refresh_with_source_reference():
    source = pd.DataFrame({'age': [41, 42]}, index=pd.Index([10, 11], name='point_id'))
    first, domain = prepare_data(_program(), {'points': source})
    assert domain == (0, 1)
    assert first['twice'].tolist() == [82, 84]
    changed = pd.DataFrame({'age': [50, 51, 52]},
                           index=pd.Index([10, 11, 12], name='point_id'))
    second, domain = prepare_data(_program(), {'points': changed})
    assert domain == (0, 1, 2)
    assert second['twice'].tolist() == [100, 102, 104]
    assert first['twice'].tolist() == [82, 84]


def test_invalid_revision_does_not_publish_a_partial_derived_result():
    source = pd.DataFrame({'age': [41]}, index=pd.Index([10], name='point_id'))
    first, _ = prepare_data(_program(), {'points': source})
    with pytest.raises(KeyError):
        prepare_data(_program(), {'points': source.drop(columns='age')})
    assert first['twice'].tolist() == [82]


def test_recursive_data_cell_is_reported():
    program = _program()
    program['cells']['ages'] = 'def ages():\n    return twice()'
    with pytest.raises(ContractError, match='Recursive Data Cell'):
        prepare_data(program, {'points': pd.DataFrame({'age': [41]})})
