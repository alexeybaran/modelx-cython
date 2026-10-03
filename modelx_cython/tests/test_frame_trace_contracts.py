from enum import IntEnum
from types import SimpleNamespace

import pytest

from modelx_cython._frame_runtime.contracts import ContractError, TableKey
from modelx_cython.tests.reference_frame.trace_contracts import from_mx2cy_trace


class Category(IntEnum):
    ONE = 0


def test_frame_contracts_use_existing_trace_and_leave_unseen_cells_untyped():
    manifest = {'model': 'M', 'spaces': {'M.Projection': {
        'cells': {'a': {}, 'b': {}, 'unseen': {}}}}}
    logger = SimpleNamespace(cells_info={
        'export._mx_classes._c_Projection._f_a': SimpleNamespace(
            ret_type=SimpleNamespace(is_array=False, value_type=float),
            arg_types={'t': int}),
        'export._mx_classes._c_Projection._f_b': SimpleNamespace(
            ret_type=SimpleNamespace(is_array=False, value_type=Category),
            arg_types={}),
        'export._mx_classes._c_Data._f_a': SimpleNamespace(
            ret_type=SimpleNamespace(is_array=False, value_type=str),
            arg_types={}),
    })
    assert from_mx2cy_trace(logger, manifest) == {
        'a': {'return': float, 't': int},
        'b': {'return': Category},
    }


def test_frame_contracts_reject_ambiguous_trace():
    observed = SimpleNamespace(
        ret_type=SimpleNamespace(is_array=False, value_type=float),
        arg_types={})
    logger = SimpleNamespace(cells_info={
        'one._mx_classes._c_Projection._f_a': observed,
        'two._mx_classes._c_Projection._f_a': observed})
    manifest = {'model': 'M', 'spaces': {'M.Projection': {'cells': {'a': {}}}}}
    with pytest.raises(ContractError, match='Ambiguous'):
        from_mx2cy_trace(logger, manifest)


def test_nested_space_trace_uses_exported_module_and_class_identity():
    observed = SimpleNamespace(
        ret_type=SimpleNamespace(is_array=False, value_type=float),
        arg_types={'t': int})
    logger = SimpleNamespace(cells_info={
        'exported._m_Top._m_Child._mx_classes._c_Leaf._f_value': observed,
        'exported._m_Else._m_Child._mx_classes._c_Leaf._f_value': observed,
    })
    manifest = {'model': 'M', 'spaces': {'M.Top.Child.Leaf': {
        'cells': {'value': {}}}}}
    assert from_mx2cy_trace(logger, manifest, space='Top.Child.Leaf') == {
        'value': {'return': float, 't': int}}


def test_explicit_lookup_key_keeps_its_table_domain():
    observed = SimpleNamespace(
        ret_type=SimpleNamespace(is_array=False, value_type=str),
        arg_types={})
    logger = SimpleNamespace(cells_info={
        'export._mx_classes._c_Projection._f_key': observed,
        'export._mx_classes._c_Projection._f_other': observed})
    manifest = {'model': 'M', 'spaces': {'M.Projection': {'cells': {
        'key': {'source': "def key():\n    return lookup.key(data.rate_scenario(), scenario)"},
        'other': {'source': "def other():\n    return lookup.key(select_table(), scenario)"},
    }}}}
    assert from_mx2cy_trace(logger, manifest) == {
        'key': {'return': TableKey('rate_scenario')},
        'other': {'return': str},
    }


def test_conditional_key_uses_one_domain_only_when_every_return_agrees():
    observed = SimpleNamespace(
        ret_type=SimpleNamespace(is_array=False, value_type=str),
        arg_types={})
    logger = SimpleNamespace(cells_info={
        'export._mx_classes._c_Projection._f_scenario': observed,
        'export._mx_classes._c_Projection._f_mixed': observed})
    manifest = {'model': 'M', 'spaces': {'M.Projection': {'cells': {
        'scenario': {'source': "def scenario():\n"
                    "    if scenario_override == -1:\n"
                    "        return lookup.key(data.rates(), default_scenario)\n"
                    "    return lookup.key(data.rates(), scenario_override)"},
        'mixed': {'source': "def mixed():\n"
                 "    if flag:\n"
                 "        return lookup.key(data.rates(), key)\n"
                 "    return lookup.key(data.other(), key)"},
    }}}}
    assert from_mx2cy_trace(logger, manifest) == {
        'scenario': {'return': TableKey('rates')},
        'mixed': {'return': str},
    }
