"""cells_param_size at the model level and in enclosing Spaces: nearest declaration wins."""
from modelx_cython.config import TransSpec


def test_declared_sizes_inherit_by_space_path():
    spec = TransSpec({'cells_param_size': {'t': 100, 'x': 121, ('t', 'k'): (100, 3)},
                      'spaces': {'Outer': {'cells_param_size': {'t': 200},
                                           'spaces': {'Inner': {'cells_params': {'k': {'size': 5}}}}}}})
    assert spec.get_declared_sizes('Pkg._mx_classes._c_Other') == (
        {('t', 'k'): ((100, 3), 'model')}, {'t': (100, 'model'), 'x': (121, 'model')})
    assert spec.get_declared_sizes('Pkg._m_Outer._mx_classes._c_Inner') == (
        {('t', 'k'): ((100, 3), 'model')},
        {'t': (200, 'Outer'), 'x': (121, 'model'), 'k': (5, 'Outer.Inner')})


def test_space_level_only_is_upstream_form():
    spec = TransSpec({'spaces': {'Projection': {'cells_param_size': {'t': 241, ('i', 'j'): (3, 6)}}}})
    assert spec.get_declared_sizes('Pkg._mx_classes._c_Projection') == (
        {('i', 'j'): ((3, 6), 'Projection')}, {'t': (241, 'Projection')})
    assert spec.get_declared_sizes('Pkg._mx_classes._c_Other') == ({}, {})
