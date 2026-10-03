"""The structured build emitter preserves the generated frame program."""
import hashlib
from types import ModuleType

import pytest

from modelx_cython._frame_runtime.compiler import emit, _frame_cast_names
from modelx_cython._frame_runtime.contracts import KernelSpec, cell
from modelx_cython._frame_runtime.frame_codegen import to_c_frame


@cell
def value(t: int) -> float:
    return base + 0.5 * t


@cell
def total() -> float:
    return value(1) + value(2)


@cell
def report() -> object:
    return {'total': total()}


@pytest.mark.parametrize('mixed, expected', [
    (False, '9d940b287c2057d29532f9e9ec2602aaa1fcd75f87995d3027dc94516db5c6af'),
    # Includes the previously committed checked typed dependency bridge.
    (True, '78236203fa3baba5bd8f2495851f900e76b86268f4772415ee9e9e3abef44851'),
])
def test_structured_emission_preserves_generated_source(mixed, expected):
    module = ModuleType('frame_emission_fixture')
    module.value = value
    module.total = total
    if mixed:
        module.report = report
    spec = KernelSpec(module=module, inputs={'base': float}, outputs={},
                      steps=4, length='4',
                      cell_outputs=('report',) if mixed else ('total',),
                      scalar_outputs=('total',), cython_objects=mixed)
    parts, manifest = emit(spec)
    assert parts['header'][0].startswith('# cython:')
    for name in ('value', 'total'):
        entry = next(method for method in parts['methods']
                     if any(line.startswith('cdef inline double _cell_' + name + '(')
                            for line in method))
        assert any('(_Frame* state' in line for line in entry)
        assert all('self.' not in line for line in entry)
    direct = to_c_frame(parts, manifest)
    assert hashlib.sha256(direct.encode()).hexdigest() == expected


def test_frame_casts_do_not_change_formula_string_literals():
    marker = '__MX_FRAME_CAST_curve__'
    source = f"return {marker}.get(1) if flag else '{marker}'"
    assert _frame_cast_names(source, {marker: '(<Curve>state.curve)'}) == (
        f"return (<Curve>state.curve).get(1) if flag else '{marker}'")


def test_checked_python_formula_is_emitted_with_a_frame_receiver():
    module = ModuleType('frame_checked_formula_fixture')
    module.value = value
    module.total = total
    spec = KernelSpec(module=module, inputs={'base': float}, outputs={},
                      steps=4, length='4', cell_outputs=('total',),
                      scalar_outputs=('total',))
    parts, manifest = emit(spec, python_cells=('total',))
    assert all(any(line.startswith('cdef inline ') and '(_Frame* state' in line
                   for line in method) for method in parts['methods'])
    formula = next(method for method in parts['methods']
                   if method[0].startswith('cdef inline double _calc_total('))
    assert formula[0].startswith('cdef inline double _calc_total(_Frame* state')
    assert '(<object>state._python_scope).calculate' in '\n'.join(formula)
    assert 'self.' not in '\n'.join(formula)
    generated = to_c_frame(parts, manifest)
    assert '(<object>state._python_scope).calculate' in generated


def test_output_plan_is_bound_before_runtime_generation():
    module = ModuleType('frame_output_plan_fixture')
    module.value = value
    module.total = total
    spec = KernelSpec(module=module, inputs={'base': float},
                      outputs={'cash': {'period': 'value(t)', 'aggregate': 'total()'}},
                      steps=4, length='4')
    parts, manifest = emit(spec)
    assert parts['runtime'] == parts['runtime_prefix']
    assert not any('def evaluate(' in line or 'def run(' in line
                   or 'def run_scalar(' in line for line in parts['runtime'])
    assert parts['frame_outputs'] == {
        'length': '4',
        'tables': {'cash': ['_cell_value(state, t)', '_cell_total(state)']},
    }
    generated = to_c_frame(parts, manifest)
    assert 'command.output[t * 2 + 0] = _cell_value(state, t)' in generated
    assert 'command.output[t * 2 + 1] = _cell_total(state)' in generated
