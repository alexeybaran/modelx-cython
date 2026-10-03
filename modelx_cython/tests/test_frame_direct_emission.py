"""The structured build emitter preserves the generated frame program."""
import hashlib
from types import ModuleType

import pytest

from modelx_cython._frame_runtime.compiler import emit, _frame_cast_names
from modelx_cython._frame_runtime.contracts import KernelSpec, cell
from modelx_cython._frame_runtime.frame_codegen import emit_context


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
    # Includes checked typed dependencies and lazy object-entry tracking.
    (True, 'bf80aba88c5e8e73e9a64602944199da583d98e49022da41890625036c85a215'),
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
    direct = emit_context(parts, manifest)
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
    generated = emit_context(parts, manifest)
    assert '(<object>state._python_scope).calculate' in generated


def test_output_plan_is_bound_before_runtime_generation():
    module = ModuleType('frame_output_plan_fixture')
    module.value = value
    module.total = total
    spec = KernelSpec(module=module, inputs={'base': float},
                      outputs={'cash': {'period': 'value(t)', 'aggregate': 'total()'}},
                      steps=4, length='4')
    parts, manifest = emit(spec)
    assert 'runtime' not in parts and 'runtime_prefix' not in parts
    assert 'fields' not in parts  # No temporary class-shaped program.
    assert parts['field_types']['_epoch'] == 'unsigned int'
    assert parts['frame_outputs'] == {
        'length': '4',
        'tables': {'cash': ['_cell_value(state, t)', '_cell_total(state)']},
    }
    generated = emit_context(parts, manifest)
    assert 'command.output[t * 2 + 0] = _cell_value(state, t)' in generated
    assert 'command.output[t * 2 + 1] = _cell_total(state)' in generated


@pytest.mark.parametrize('steps, growing, mixed, expected', [
    (4, False, False, '1a3b1056b4ac295ccf7f39f3d42ba89e9de1a111dfb387b89fe17b5e9908a5a9'),
    (1200, False, False, '9f1003fef1e532b7513fb0f32a088012a8315e26fb5e4635eb56a3be28f39682'),
    (4, True, False, 'd3f500e8739263ec07faf1b3f6f715e02a6713ebac97e7909804fcdcd570806e'),
    (4, False, True, 'eb0aa55f1b69c7cc75d0230845ffe5f8e6fd02f103ad63c77191cbc7125b3851'),
    (1200, False, True, '05bcd4bcc80b785a6dd33af048d8fc578ce55bc47826301e6914c63bed604075'),
])
def test_direct_owner_and_guard_protocol_preserve_complete_program(steps, growing, mixed, expected):
    # Frozen at 068c270 before removing the mixed-runtime source rewrite. Covers
    # inline/heap allocation, dictionary growth, checked object boundaries and
    # native-only traversal selection, not just the owner declaration prefix.
    # Mixed snapshots additionally include lazy object bookkeeping; formula
    # and cache semantics remain covered by compiled behavioral fixtures.
    import ast
    from modelx_cython.formula_emitter import SourceFormula
    module = ModuleType('owner_protocol_probe')
    module.value = SourceFormula(ast.parse('def value(n):\n return base+0.5*n\n').body[0],
                                 {'n':int}, float, module_name=module.__name__, filename=__file__)
    module.total = SourceFormula(ast.parse('def total():\n return value(1)+value(2)\n').body[0],
                                 {}, float, module_name=module.__name__, filename=__file__)
    if mixed:
        module.report = SourceFormula(ast.parse("def report():\n return {'value': value(1)}\n").body[0],
                                      {}, object, module_name=module.__name__, filename=__file__)
    spec = KernelSpec(module, {'base':float}, {}, steps, '0',
                      cell_outputs=('report','total') if mixed else ('value','total'),
                      scalar_outputs=('total',), cython_objects=True,
                      original_cache_kinds={'value':'dictionary'} if growing else {})
    parts, manifest = emit(spec)
    assert hashlib.sha256(emit_context(parts, manifest).encode()).hexdigest() == expected
