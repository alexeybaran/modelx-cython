import pytest

from modelx_cython.config import TransSpec


testdata_nested = {"cells": {"sample": None}}
testdata_space = {
    "cells_params": {"t": {"size": 241}},
    "cells": {"disc_factors": {"return_type": "object"}},
    "spaces": {"Nested": testdata_nested},
}

testdata = {"spaces": {"Projection": testdata_space}}


class TestConf:
    @pytest.mark.parametrize(
        "obj_path",
        ["ModelName._mx_class_._c_Projection", "ModelName._mx_class_._m_Projection"],
    )
    def test_get_data(self, obj_path):
        data = TransSpec(testdata).get_spec(obj_path)
        assert data == testdata_space

    @pytest.mark.parametrize(
        "obj_path",
        [
            "ModelName._m_Projection._mx_classes._c_Nested",
            "ModelName._m_Projection._m_Nested",
        ],
    )
    def test_get_data2(self, obj_path):
        data = TransSpec(testdata).get_spec(obj_path)
        assert data == testdata_nested


@pytest.mark.parametrize('data',[
    {'automatic_optimizations':1}, {'native_cache_hits':True},
    {'dense_miss_binding':False,'native_cache_hits':'both'},
    {'dense_miss_binding':False,'propagate_exact_receiver':True},
])
def test_invalid_dependent_policy_is_rejected_before_emission(data):
    with pytest.raises(ValueError):
        TransSpec(data)


def test_explicit_policy_overrides_keep_their_existing_dependencies():
    policy=TransSpec({'automatic_optimizations':False,'dense_miss_binding':True,
        'propagate_exact_receiver':True,'native_cache_hits':'both'}).policy
    assert policy.propagate_exact_receiver and policy.native_cache_hits=='both'


def test_disabling_automatic_selection_disables_implicit_dense_entries():
    policy = TransSpec({'automatic_optimizations': False}).policy
    assert not policy.dense_miss_binding
    assert not policy.propagate_exact_receiver
    assert policy.native_cache_hits == 'none'


def test_integer_checks_defaults_to_auto_and_accepts_the_three_modes():
    from modelx_cython.config import CompilerPolicy
    assert CompilerPolicy.resolve({}).integer_checks == 'auto'
    for mode in ('auto', 'checked', 'unchecked'):
        assert CompilerPolicy.resolve({'integer_checks': mode}).integer_checks == mode


@pytest.mark.parametrize('value', ['Checked', 'on', True, None, 1])
def test_integer_checks_rejects_any_other_value(value):
    from modelx_cython.config import CompilerPolicy
    with pytest.raises(ValueError, match='integer_checks must be auto, checked or unchecked'):
        CompilerPolicy.resolve({'integer_checks': value})


def test_auto_resolves_from_the_largest_traced_integer():
    from modelx_cython.config import CompilerPolicy, UNCHECKED_LIMIT, resolve_integer_checks
    assert UNCHECKED_LIMIT == 2 ** 31
    auto = CompilerPolicy.resolve({})
    assert resolve_integer_checks(auto, 0) == 'unchecked'
    assert resolve_integer_checks(auto, 2 ** 31 - 1) == 'unchecked'
    assert resolve_integer_checks(auto, 2 ** 31) == 'checked'
    assert resolve_integer_checks(auto, 2 ** 70) == 'checked'
    for mode in ('checked', 'unchecked'):
        forced = CompilerPolicy.resolve({'integer_checks': mode})
        assert resolve_integer_checks(forced, 0) == resolve_integer_checks(forced, 2 ** 70) == mode
