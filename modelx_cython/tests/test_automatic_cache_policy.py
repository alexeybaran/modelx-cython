"""Cache policy follows generic footprint and the declared or sampled range."""
from types import SimpleNamespace
import libcst as cst
import pytest
from modelx_cython.automatic import observed_axes, primary_capacity
from modelx_cython.cell_entries import DenseAxis


def cell(minimum,maximum):
    return SimpleNamespace(_rt=SimpleNamespace(observed_min_args=minimum,
        observed_max_args=maximum),
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


def _ranged(*sizes):
    axes = tuple(DenseAxis(f'p{i}', 0, size, 'long long') for i, size in enumerate(sizes))
    return SimpleNamespace(params=tuple(axis.key for axis in axes), associative_logical_axes=axes,
                           has_typeinfo=lambda: True)


def test_primary_size_follows_the_declared_or_sampled_range_and_the_common_cap():
    assert primary_capacity(_ranged(1)) == 8
    assert primary_capacity(_ranged(600)) == 2048
    assert primary_capacity(_ranged(40, 35)) == 4096        # 1,400 entries
    assert primary_capacity(_ranged(100000)) == 4096
    # a float parameter has no countable range: the smallest table, which grows
    floats = SimpleNamespace(params=('x',), associative_logical_axes=(), has_typeinfo=lambda: True)
    assert primary_capacity(floats) == 8