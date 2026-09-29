"""Column binding and repeated immutable input conversion in the frame runtime."""
import pandas as pd
import pytest
import threading
from types import SimpleNamespace

from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.data_binding import PreparedModelRows, prepare_inputs
from modelx_cython._frame_runtime.runtime import CompiledKernel
from modelx_cython._frame_runtime import runtime


def _manifest():
    return {
        'inputs': {
            'rate': {'kind': 'float'},
            'category': {'kind': 'enum', 'members': {'BASE': 0, 'STRESS': 1}},
            'point': {'kind': 'int'},
            'shared': {'kind': 'float'},
        },
        'model': {
            'point_table': 'points',
            'reference_defaults': {'shared': 0.5},
            'parameter_defaults': {},
            'bindings': {
                'rate': {'kind': 'field', 'table': 'points', 'column': 'rate'},
                'category': {'kind': 'field', 'table': 'points', 'column': 'category'},
                'point': {'kind': 'point'},
                'shared': {'kind': 'reference', 'name': 'shared'},
            },
        },
    }


def test_forked_item_updates_native_views_with_dynamic_row_references(monkeypatch):
    """A non-collect Space's typed fields follow its selected scenario refs."""
    original_paths = {'premium': [0.25, 0.5]}
    replacement_paths = {'premium': [0.75, 1.0]}
    kernel = object.__new__(CompiledKernel)
    kernel.artifact = 'fixture'
    kernel.manifest = {
        'inputs': {
            'paths': {'kind': 'object', 'python_type': 'dict'},
            'mxinput_paths_premium': {'kind': 'indexed_array', 'ctype': 'IndexedFloat1'},
            'spread': {'kind': 'float'},
            'mxinput_spread': {'kind': 'float'},
        },
        'model': {
            'bindings': {
                'paths': {'kind': 'row_reference'},
                'spread': {'kind': 'row_reference'},
            },
            'native_row_references': {
                'mxinput_paths_premium': {'column': 'paths', 'field': 'premium'},
                'mxinput_spread': {'column': 'spread', 'field': None},
            },
        },
    }
    kernel._records = ((original_paths, ('old',), 0.1, 0.1),)
    kernel._lock = threading.RLock()
    kernel._open_scope = None
    kernel._python_scope = None
    kernel._lookups = SimpleNamespace(IndexedFloat1=lambda array: tuple(array))
    kernel._module = SimpleNamespace(Kernel=lambda rows: SimpleNamespace(rows=rows))
    monkeypatch.setattr(runtime, 'ItemContext', lambda child: child._records[0])

    result = kernel.fork_item(0, input_overrides={
        'paths': replacement_paths, 'spread': 0.2})
    assert result == (replacement_paths, (0.75, 1.0), 0.2, 0.2)
    assert kernel._records[0][0] is original_paths
    with pytest.raises(ContractError, match='missing numeric row field'):
        kernel.fork_item(0, input_overrides={'paths': {'other': [1.0]}})
    assert kernel._records[0][1] == ('old',)


def test_aligned_columns_preserve_policy_values_and_category_codes():
    points = pd.DataFrame({'rate': [0.01, 0.02],
                           'category': ['base', 'stress']},
                          index=pd.Index([11, 12], name='point'))
    shared = 0.5
    rows = prepare_inputs(_manifest(), {'points': points},
                          references={'shared': shared}, packed={})
    assert rows == [
        {'rate': 0.01, 'category': 0, 'point': 11, 'shared': 0.5},
        {'rate': 0.02, 'category': 1, 'point': 12, 'shared': 0.5},
    ]
    assert rows[0]['shared'] is shared and rows[1]['shared'] is shared


def test_differently_ordered_table_keeps_label_lookup_and_missing_error():
    manifest = _manifest()
    manifest['model']['bindings']['rate']['table'] = 'other'
    points = pd.DataFrame({'category': ['base', 'stress']},
                          index=pd.Index([11, 12], name='point'))
    other = pd.DataFrame({'rate': [0.02, 0.01]},
                         index=pd.Index([12, 11], name='point'))
    rows = prepare_inputs(manifest, {'points': points, 'other': other}, packed={})
    assert [row['rate'] for row in rows] == [0.01, 0.02]
    with pytest.raises(KeyError):
        prepare_inputs(manifest, {'points': points, 'other': other.iloc[:1]}, packed={})


def test_repeated_builtin_input_is_validated_once_without_changing_record_keys():
    kernel = object.__new__(CompiledKernel)
    kernel.manifest = {'inputs': {'rate': {'kind': 'float'}}}
    original = kernel._scalar_value
    calls = []

    def counting(*args):
        calls.append(args)
        return original(*args)

    kernel._scalar_value = counting
    value = float('0.125')
    assert kernel._validate([{'rate': value}, {'rate': value}]) == ((0.125,), (0.125,))
    assert len(calls) == 1
    with pytest.raises(ContractError, match='Input names differ'):
        kernel._validate([{'rate': value, 'extra': 1}])


def test_custom_number_is_not_memoized_across_policy_records():
    class ChangingFloat(float):
        conversions = 0

        def __float__(self):
            type(self).conversions += 1
            return type(self).conversions * 0.1

    kernel = object.__new__(CompiledKernel)
    kernel.manifest = {'inputs': {'rate': {'kind': 'float'}}}
    value = ChangingFloat(0.5)
    rows = kernel._validate([{'rate': value}, {'rate': value}])
    assert rows[0] != rows[1]
    assert ChangingFloat.conversions >= 2


def test_model_prepared_rows_validate_fixed_builtins_once():
    points = pd.DataFrame({'rate': [0.01, 0.02],
                           'category': ['base', 'stress']},
                          index=pd.Index([11, 12], name='point'))
    manifest = _manifest()
    prepared = prepare_inputs(manifest, {'points': points}, packed={}, prepared=True)
    assert isinstance(prepared, PreparedModelRows)
    assert prepared.fixed_names == frozenset({'shared'})
    kernel = object.__new__(CompiledKernel)
    kernel.manifest = manifest
    expected = kernel._validate(list(prepared.rows))
    assert kernel._validate(prepared) == expected


def test_model_prepared_rows_still_convert_custom_reference_per_policy():
    class ChangingFloat(float):
        conversions = 0

        def __float__(self):
            type(self).conversions += 1
            return type(self).conversions * 0.1

    points = pd.DataFrame({'rate': [0.01, 0.02],
                           'category': ['base', 'stress']},
                          index=pd.Index([11, 12], name='point'))
    manifest = _manifest()
    value = ChangingFloat(0.5)
    prepared = prepare_inputs(manifest, {'points': points},
                              references={'shared': value}, packed={}, prepared=True)
    assert 'shared' not in prepared.fixed_names
    kernel = object.__new__(CompiledKernel)
    kernel.manifest = manifest
    converted = kernel._validate(prepared)
    assert converted[0][-1] != converted[1][-1]
