"""Cache policy follows generic footprint and receiver evidence."""
from types import SimpleNamespace
import libcst as cst
import pytest
from modelx_cython.automatic import observed_axes, primary_capacity
from modelx_cython.cell_entries import DenseAxis


def cell(minimum,maximum,evaluations=1):
    return SimpleNamespace(_rt=SimpleNamespace(observed_min_args=minimum,
        observed_max_args=maximum,max_evaluations_per_receiver=evaluations),
        get_rettype_expr=lambda **kwargs:'double')


def test_near_full_window_keeps_native_range_contract():
    logical=(DenseAxis('x',0,100,'long long'),)
    assert observed_axes(cell({'x':0},{'x':98}),logical)==logical


def test_material_window_retains_separate_logical_domain():
    logical=(DenseAxis('x',0,100,'long long'),)
    assert observed_axes(cell({'x':20},{'x':39}),logical)==(DenseAxis('x',20,40,'long long'),)
    assert logical==(DenseAxis('x',0,100,'long long'),)


def test_multi_axis_savings_are_measured_for_whole_cache():
    logical=(DenseAxis('x',0,100,'long long'),DenseAxis('y',0,3,'long long'))
    assert observed_axes(cell({'x':0,'y':1},{'x':99,'y':2}),logical)==(
        DenseAxis('x',0,100,'long long'),DenseAxis('y',1,3,'long long'))


def test_primary_size_follows_receiver_work_and_preserves_common_cap():
    assert primary_capacity(cell({}, {},1))==2
    assert primary_capacity(cell({}, {},600))==2048
    assert primary_capacity(cell({}, {},1400))==4096
    assert primary_capacity(cell({}, {},100000))==4096


