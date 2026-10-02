"""Original standard-command metadata drives the frame formula handoff."""
import argparse
import io
import json
import sys
from enum import IntEnum

import pandas as pd
import pytest


class _PlanCategory(IntEnum):
    ZERO = 0
    ONE = 1


@pytest.fixture(autouse=True)
def isolated_export_imports():
    """Each fixture exports a different package under the same test name."""
    def clear():
        for name in tuple(sys.modules):
            if name in ('model', 'model_cy') or name.startswith(('model.', 'model_cy.')):
                sys.modules.pop(name, None)
    clear()
    yield
    clear()


def test_shared_plan_drives_helper_object_and_uncached_bodies(tmp_path, monkeypatch):
    import modelx as mx
    from modelx_cython import cli, compilation_plan
    from modelx_cython._frame_runtime import lookup
    from modelx_cython._frame_runtime.compiler import build
    from modelx_cython._frame_runtime.model import CompiledModel
    from modelx_cython._frame_runtime.compilation_plan_adapter import from_compilation_plan
    from modelx_cython.formula_emitter import SourceFormula

    m = mx.new_model('PlanHandoff')
    data = m.new_space('Data')
    data.points_data = pd.DataFrame({'value': [1.5, -1.0]},
                                   index=pd.Index([0, 1], name='policy_id'))
    data.new_cells('points', 'def points():\n return points_data\n')
    p = m.new_space('Projection', formula='def Projection(policy_id):\n return {"refs": {}}\n')
    p.data = data
    p.lookup = lookup
    p.flags = {'fail': False}
    p.new_cells('amount', 'def amount():\n return lookup.field(data.points(), policy_id, "value")\n')
    p.new_cells('report', 'def report():\n return {"amount": amount()}\n')
    p.new_cells('uncached', 'def uncached():\n return report()["amount"]\n')
    p.new_cells('receiver_value', 'def receiver_value():\n return _space\n')
    p.uncached.is_cached = False
    m.export(tmp_path / 'model')
    (tmp_path / 'model' / '_mx_formula_sources.json').unlink()
    m.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\n'
        'for i in (0,1):\n p=mx_model.Projection[i]\n p.amount(); p.report(); p.uncached()\n',
        encoding='utf-8')
    captured = []
    original = compilation_plan.build_package_plan
    def record(units):
        plan = original(units)
        captured.append(plan)
        return plan
    monkeypatch.setattr(compilation_plan, 'build_package_plan', record)
    monkeypatch.setenv('MX_CY_ORIGINAL_FRAME_SHELL', '0')
    # Execute the actual original command stages, not independently reconstructed
    # type metadata. No compiler processes are needed for this translate step.
    cli.main_handler(argparse.Namespace(model_path=str(tmp_path/'model'),
        sample=str(tmp_path/'sample.py'), no_spec=True, spec=None, setup=None,
        frame_collect=False, compile_only=False, translate_only=True), io.StringIO(), io.StringIO())
    space = captured[0].spaces['model_cy.Projection']
    assert space.cells['amount'].public_result == 'double'
    assert not space.cells['uncached'].cached
    assert space.cells['receiver_value'].receiver_requirements
    import modelx_cython._frame_runtime.modelx_frontend_exported as retired_discovery
    def forbidden(*args, **kwargs):
        raise AssertionError('Resolved handoff must not rediscover the export')
    monkeypatch.setattr(retired_discovery, 'open_exported_source', forbidden)
    spec = from_compilation_plan(captured[0], sys.modules['model'].mx_model,
        tmp_path/'source', projection='Projection', point_table='points',
        cython_objects=True, steps=8)
    assert all(isinstance(formula, SourceFormula) for formula in spec.cells.values())
    assert 'receiver_value' not in spec.cells
    assert 'receiver_value' in spec.model['original_receiver_cells']
    for formula in spec.cells.values():
        with pytest.raises(TypeError, match='metadata'):
            formula()
    artifact = tmp_path/'artifact'
    build(spec, artifact)
    manifest = json.loads((artifact/'manifest.json').read_text())
    assert all(not row['coverage']['detached_python'] for row in manifest['cells'].values())
    assert manifest['cells']['report']['coverage']['body_backend'] == 'cython_gil'
    assert manifest['cells']['uncached']['cached'] is False
    model = CompiledModel(artifact, spec.initial_tables)
    try:
        for policy, value in ((0, 1.5), (1, -1.0)):
            item = model.Projection[policy]
            assert item.amount() == value
            assert item.report() == {'amount': value}
            assert item.uncached() == value
    finally:
        model.close()


def test_source_metadata_preserves_literal_defaults():
    import ast
    from modelx_cython.formula_emitter import SourceFormula
    formula = SourceFormula(ast.parse('def value(t=2):\n return float(t)\n').body[0],
                            {'t': int}, float, filename=__file__, module_name=__name__)
    import inspect
    bound = inspect.signature(formula).bind()
    bound.apply_defaults()
    assert bound.arguments == {'t': 2}


@pytest.mark.parametrize('source', ['self.policies()', 'tuple(self.policies())'])
def test_resolved_export_domain_retains_order_and_repeated_selection(source):
    from types import SimpleNamespace
    from modelx_cython._frame_runtime.compilation_plan_adapter import _ResolvedDomain
    calls = []
    values = [4, 1, 4]
    def policies():
        calls.append(tuple(values))
        return iter(values)
    domain = _ResolvedDomain(SimpleNamespace(policies=policies), source)
    assert domain.ref_name == 'policies'
    assert tuple(domain.value(None)) == (4, 1, 4)
    values[:] = [8, 2]
    assert tuple(domain.value(None)) == (8, 2)
    assert calls == [(4, 1, 4), (8, 2)]


@pytest.mark.parametrize('source', ['self.policies(1)', 'tuple(self.policies(), self.other)',
                                  'self.policies() + self.other', 'list(self.policies())'])
def test_resolved_domain_does_not_execute_unsupported_expressions(source):
    from types import SimpleNamespace
    from modelx_cython._frame_runtime.compilation_plan_adapter import _ResolvedDomain
    from modelx_cython._frame_runtime.contracts import ContractError
    def forbidden(*args):
        raise AssertionError('Domain discovery must not execute expressions')
    with pytest.raises(ContractError):
        _ResolvedDomain(SimpleNamespace(policies=forbidden, other=forbidden), source)


def test_prepared_context_preserves_native_categories_behind_object_public_abi(tmp_path, monkeypatch):
    import modelx as mx
    from modelx_cython import cli, compilation_plan
    from modelx_cython._frame_runtime import lookup
    from modelx_cython._frame_runtime.compiler import build
    from modelx_cython._frame_runtime.compilation_plan_adapter import cell_bindings, from_compilation_plan, _resolve_cell_bindings
    from modelx_cython._frame_runtime.model import CompiledModel
    m = mx.new_model('CategoryHandoff')
    data = m.new_space('Data')
    data.points_data = pd.DataFrame({'category': ['ONE', 'ZERO']},
                                   index=pd.Index([0, 1], name='policy_id'))
    data.new_cells('points', 'def points():\n return points_data\n')
    p = m.new_space('Projection', formula='def Projection(policy_id):\n return {"refs": {}}\n')
    p.data, p.lookup, p.Kind = data, lookup, _PlanCategory
    p.new_cells('category', 'def category():\n return lookup.category(data.points(), policy_id, "category", Kind)\n')
    p.new_cells('amount', 'def amount(kind=0):\n return 3.5 if kind == Kind.ONE else -1.0\n')
    p.new_cells('report', 'def report():\n return {"category": category(), "amount": amount(category())}\n')
    m.export(tmp_path / 'model')
    m.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\n'
        'for i in (0,1):\n p=mx_model.Projection[i]\n'
        ' p.report();p.amount(p.Kind.ONE);p.amount(0)\n', encoding='utf-8')
    captured = []
    original = compilation_plan.build_package_plan
    def capture(units):
        result = original(units)
        captured.append(result)
        return result
    monkeypatch.setattr(compilation_plan, 'build_package_plan', capture)
    monkeypatch.setenv('MX_CY_ORIGINAL_FRAME_SHELL', '0')
    cli.main_handler(argparse.Namespace(model_path=str(tmp_path/'model'),
        sample=str(tmp_path/'sample.py'), no_spec=True, spec=None, setup=None,
        frame_collect=False, compile_only=False, translate_only=True), io.StringIO(), io.StringIO())
    space = captured[0].spaces['model_cy.Projection']
    assert space.cells['category'].public_result == 'object'
    assert dict(space.cells['amount'].public_arguments) == {'kind': 'object'}
    assert cell_bindings(space.cells['category'])['return'] is _PlanCategory
    assert cell_bindings(space.cells['amount'])['kind'] is _PlanCategory
    assert space.cells['category'].inventory()['physical_bindings']['return']['kind'] == 'enum'
    assert space.cells['category'].inventory()['public_result'] == 'object'
    category = space.cells['category'].original_info
    category._spec_ret_t = 'object'
    # A finalized plan is immutable; a new resolution honors a new explicit
    # specification, without changing the already recorded emission decision.
    assert cell_bindings(space.cells['category'])['return'] is _PlanCategory
    assert _resolve_cell_bindings(space.cells['category'])['return'] is object
    category._spec_ret_t = ''
    spec = from_compilation_plan(captured[0], sys.modules['model'].mx_model,
        tmp_path/'source', projection='Projection', point_table='points', steps=8)
    artifact = tmp_path/'artifact'
    build(spec, artifact)
    manifest = json.loads((artifact/'manifest.json').read_text())
    assert manifest['cells']['category']['result']['kind'] == 'enum'
    assert manifest['cells']['amount']['params'][0]['type']['kind'] == 'enum'
    assert manifest['cells']['amount']['calculator'] == 'native'
    model = CompiledModel(artifact, spec.initial_tables)
    try:
        for policy, category in ((0, _PlanCategory.ONE), (1, _PlanCategory.ZERO)):
            item = model.Projection[policy]
            assert item.category() is category
            assert item.amount(_PlanCategory.ONE) == item.amount(1) == 3.5
            assert item.amount(0) == item.amount(_PlanCategory.ZERO) == -1.0
            assert item.report() == {'category': category,
                                     'amount': 3.5 if category is _PlanCategory.ONE else -1.0}
    finally:
        model.close()
