"""Original standard-command metadata drives the frame formula handoff."""
import argparse
import io
import json
import sys
from enum import IntEnum
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings('error::pytest.PytestUnraisableExceptionWarning')


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
    from modelx_cython.tests.reference_frame.compiler import build
    from modelx_cython.tests.reference_frame.model import CompiledModel
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
    p.new_cells('numeric_key', 'def numeric_key(key):\n return float(key)\n')
    p.new_cells('numeric_pair', 'def numeric_pair(value, found):\n return value + found\n')
    p.new_cells('receiver_value', 'def receiver_value():\n return _space\n')
    p.uncached.is_cached = False
    m.export(tmp_path / 'model')
    (tmp_path / 'model' / '_mx_formula_sources.json').unlink()
    m.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\n'
        'for i in (0,1):\n p=mx_model.Projection[i]\n p.amount(); p.report(); p.uncached()\n'
        ' for key in (-1.0, 0.0, 0.5): p.numeric_key(key)\n'
        ' p.numeric_pair(0.5, 1)\n',
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
    import modelx_cython.tests.reference_frame.modelx_frontend_exported as retired_discovery
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
    assert manifest['cells']['numeric_key']['storage'] == 'growing_numeric'
    assert manifest['cells']['numeric_pair']['storage'] == 'growing_numeric'
    # Original public classes can consume the exact same prepared resources
    # without constructing a second model/Space facade. Exercise real generated
    # lookup classes and object-capable Cython, not a mocked formula evaluator.
    import importlib
    from modelx_cython._frame_runtime.prepared_space_inputs import PreparedSpaceInputs
    from modelx_cython._frame_runtime.python_calculator import PythonScope
    monkeypatch.syspath_prepend(str(artifact))
    native_lookups = importlib.import_module(manifest['package'] + '._lookups')
    native_kernel = importlib.import_module(manifest['package'] + '._kernel')
    prepared = PreparedSpaceInputs(manifest, native_lookups, spec.initial_tables)
    for position, expected in enumerate((1.5, -1.0)):
        owner = native_kernel.Kernel((prepared.record(position),))
        scope = PythonScope(artifact, manifest, owner)
        owner.set_python_scope(scope)
        owner.select_scope(0)
        assert owner.call('amount', ()) == expected
        assert owner.call('report', ()) == {'amount': expected}
        assert owner.call('uncached', ()) == expected
        for key in range(1024):
            assert owner.call('numeric_key', (key + 0.5,)) == key + 0.5
        # Old entries survive growth, signed zeros retain numeric key equality,
        # and a non-finite key can transition to the ordinary dictionary path.
        assert owner.call('numeric_key', (0.5,)) == 0.5
        assert owner.call('numeric_pair', (2.5, -1)) == 1.5
        assert owner.call('numeric_key', (-0.0,)) == 0.0
        import math
        assert math.isnan(owner.call('numeric_key', (float('nan'),)))
        assert owner.call('numeric_key', (0.5,)) == 0.5
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


def test_dynamic_row_resolution_preserves_scalar_reference_discovery(tmp_path,monkeypatch):
    import modelx as mx
    from modelx_cython import cli,compilation_plan
    from modelx_cython._frame_runtime.compilation_plan_adapter import from_compilation_plan
    from modelx_cython.tests.reference_frame.compiler import build
    model=mx.new_model('DynamicReference')
    p=model.new_space('Projection',formula='def Projection(policy_id):\n return {"refs":{"value":float(policy_id)+1.0}}\n')
    p.rate=0.5
    p.new_cells('amount','def amount():\n return value * rate\n')
    model.export(tmp_path/'model');model.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\nfor i in (0,1): mx_model.Projection[i].amount()\n')
    captured=[];original=compilation_plan.build_package_plan
    def capture(units):
        plan=original(units);captured.append(plan);return plan
    monkeypatch.setattr(compilation_plan,'build_package_plan',capture)
    monkeypatch.setenv('MX_CY_ORIGINAL_FRAME_SHELL','0')
    cli.main_handler(argparse.Namespace(model_path=str(tmp_path/'model'),sample=str(tmp_path/'sample.py'),
        no_spec=True,spec=None,setup=None,frame_collect=False,compile_only=False,translate_only=True),io.StringIO(),io.StringIO())
    assert captured[0].spaces['model_cy.Projection'].cells['amount'].receiver_requirements
    spec=from_compilation_plan(captured[0],sys.modules['model'].mx_model,tmp_path/'source',
        projection='Projection',policy_records=pd.DataFrame({'value':[1.0,2.0]},index=pd.Index([0,1],name='policy_id')),
        row_references={'value':'value'},point_parameter='policy_id',steps=4,cython_objects=True,
        typed_object_inputs={'value':'float'},native_typed_inputs=True,
        signatures={'amount':{'return':'float','parameters':{}}})
    build(spec,tmp_path/'artifact',compile_native=False)
    manifest=json.loads((tmp_path/'artifact/manifest.json').read_text())
    assert manifest['model']['reference_defaults']['rate']==0.5
    assert manifest['cells']['amount']['coverage']['python_free_proven'] is True


def test_closed_scalar_records_share_compiled_declaration_and_reset_on_failure(tmp_path,monkeypatch):
    import modelx as mx
    from dataclasses import replace
    from modelx_cython import cli,compilation_plan
    from modelx_cython._frame_runtime import lookup
    from modelx_cython.tests.reference_frame.compiler import build
    from modelx_cython._frame_runtime.compilation_plan_adapter import from_compilation_plan
    from modelx_cython._frame_runtime.prepared_space_inputs import PreparedSpaceInputs
    import importlib
    model=mx.new_model('ScalarRecords')
    data=model.new_space('Data')
    data.points_data=pd.DataFrame({'value':[1.5,-1.0]},index=pd.Index([0,1],name='policy_id'))
    data.new_cells('points','def points():\n return points_data\n')
    p=model.new_space('Projection',formula='def Projection(policy_id):\n return {"refs":{}}\n')
    p.data=data;p.lookup=lookup
    p.new_cells('amount','def amount():\n return lookup.field(data.points(),policy_id,"value")\n')
    p.new_cells('collect_outputs','def collect_outputs():\n return {"sum":{"total":amount()}}\n')
    model.export(tmp_path/'model');model.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\nfor i in (0,1): mx_model.Projection[i].collect_outputs()\n')
    captured=[];original=compilation_plan.build_package_plan
    def capture(units):
        result=original(units);captured.append(result);return result
    monkeypatch.setattr(compilation_plan,'build_package_plan',capture)
    monkeypatch.setenv('MX_CY_ORIGINAL_FRAME_SHELL','0')
    cli.main_handler(argparse.Namespace(model_path=str(tmp_path/'model'),sample=str(tmp_path/'sample.py'),
        no_spec=True,spec=None,setup=None,frame_collect=False,compile_only=False,translate_only=True),io.StringIO(),io.StringIO())
    plan=captured[0];space=plan.spaces['model_cy.Projection']
    spec=from_compilation_plan(plan,sys.modules['model'].mx_model,tmp_path/'source',
        projection='Projection',point_table='points',cython_objects=True,steps=8)
    spec=replace(spec,public_emission_plan=space)
    build(spec,tmp_path/'artifact')
    manifest=json.loads((tmp_path/'artifact/manifest.json').read_text())
    assert manifest['cells']['collect_outputs']['checked_namespace_required'] is False
    monkeypatch.syspath_prepend(str(tmp_path/'artifact'))
    lookups=importlib.import_module(manifest['package']+'._lookups')
    extension=importlib.import_module(manifest['package']+'._kernel')
    prepared=PreparedSpaceInputs(manifest,lookups,spec.initial_tables)
    records=tuple(prepared.record(i) for i in (0,1))
    owner=extension.Kernel(records)
    assert owner.collect_scalar_records(records,'total')==0.5
    assert owner.collect_scalar_records(records,'total')==0.5
    with pytest.raises(KeyError):owner.collect_scalar_records(records,'missing')
    assert owner.collect_scalar_records(records,'total')==0.5
    with pytest.raises(ValueError,match='nonempty'):owner.collect_scalar_records((),'total')


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
    from dataclasses import replace
    import modelx as mx
    from modelx_cython import cli, compilation_plan
    from modelx_cython._frame_runtime import lookup
    from modelx_cython.tests.reference_frame.compiler import build
    from modelx_cython._frame_runtime.compilation_plan_adapter import cell_bindings, from_compilation_plan, _resolve_cell_bindings
    from modelx_cython.tests.reference_frame.model import CompiledModel
    m = mx.new_model('CategoryHandoff')
    data = m.new_space('Data')
    data.points_data = pd.DataFrame({'category': ['ONE', 'ZERO']},
                                   index=pd.Index([0, 1], name='policy_id'))
    data.new_cells('points', 'def points():\n return points_data\n')
    domain = (0, 1)
    p = m.new_space('Projection', formula='def Projection(policy_id):\n return {"refs": {}}\n',
        refs={'policy_ids': domain}, collect_over={'policy_id': domain},
        collect_outputs='lambda: {"sum": {"total": amount(category())}, "individual": {"record": report()}}')
    p.data, p.lookup, p.Kind = data, lookup, _PlanCategory
    p.flags = {'fail': False, 'object_fail': False}
    p.new_cells('shallow', 'def shallow(n):\n return 0.5 * n\n')
    p.new_cells('native_report', 'def native_report():\n return {"value": shallow(2)}\n')
    p.new_cells('category', 'def category():\n return lookup.category(data.points(), policy_id, "category", Kind)\n')
    p.new_cells('amount', 'def amount(kind=0):\n return 3.5 if kind == Kind.ONE else -1.0\n')
    p.new_cells('report', 'def report():\n return {"category": category(), "amount": amount(category())}\n')
    p.new_cells('failed_parent', 'def failed_parent(n):\n child = amount(Kind.ONE)\n if flags["fail"]: raise ValueError("numeric retry")\n return child + n\n')
    p.new_cells('object_failure', 'def object_failure(n):\n if flags["object_fail"]: raise LookupError("object retry")\n return {"amount": amount(Kind.ONE)}\n')
    p.new_cells('numeric_from_object', 'def numeric_from_object(n):\n return float(object_failure(n)["amount"])\n')
    p.new_cells('caught_failure', 'def caught_failure(n):\n try:\n  return {"value": failed_parent(n)}\n except ValueError:\n  return {"value": -1.0}\n')
    m.export(tmp_path / 'model')
    m.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\n'
        'for i in (0,1):\n p=mx_model.Projection[i]\n'
        ' p.native_report();p.report();p.amount(p.Kind.ONE);p.amount(0)\n'
        ' for n in range(3): p.shallow(n);p.failed_parent(n);p.object_failure(n);p.numeric_from_object(n);p.caught_failure(n)\n', encoding='utf-8')
    captured = []
    original = compilation_plan.build_package_plan
    def capture(units):
        result = original(units)
        captured.append(result)
        return result
    monkeypatch.setattr(compilation_plan, 'build_package_plan', capture)
    (tmp_path/'spec.txt').write_text(repr({'spaces': {'Projection': {'native_context':
        {'point_table': 'points', 'steps': 8}}}}), encoding='utf-8')
    cli.main_handler(argparse.Namespace(model_path=str(tmp_path/'model'),
        sample=str(tmp_path/'sample.py'), no_spec=False, spec=str(tmp_path/'spec.txt'), setup=None,
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
    plain = tmp_path/'plain_artifact'
    build(spec, plain, compile_native=False)
    plain_manifest = json.loads((plain/'manifest.json').read_text())
    plain_source = (plain/plain_manifest['package']/'_kernel.pyx').read_text()
    spec = replace(spec, public_emission_plan=space)
    artifact = tmp_path/'artifact'
    build(spec, artifact)
    manifest = json.loads((artifact/'manifest.json').read_text())
    assert manifest['cells']['category']['result']['kind'] == 'enum'
    assert manifest['cells']['amount']['params'][0]['type']['kind'] == 'enum'
    assert manifest['cells']['amount']['calculator'] == 'native'
    assert space.cells['amount'].cache_kind == 'dictionary'
    assert manifest['cells']['amount']['storage'] == 'growing_numeric'
    assert manifest['original_cache_kinds']['amount'] == 'dictionary'
    assert manifest['public_entry_abi']['category']['public_return'] == 'object'
    assert manifest['public_entry_abi']['amount']['public_arguments'] == {'kind': 'object'}
    assert manifest['public_entry_abi']['amount']['physical_arguments']['kind']['kind'] == 'enum'
    adapted_source = (artifact/manifest['package']/'_kernel.pyx').read_text()
    # Prepared public entries additionally select the already-qualified typed
    # checked bridge for numeric bodies with object dependencies. Requiring the
    # old name/tuple Python adapter byte-for-byte would freeze an obsolete ABI.
    # Every cache contract and unaffected native formula must remain identical;
    # the runtime assertions below cover category conversion and failure/retry.
    import re
    bridges = set(manifest.get('original_checked_numeric_bridges', ()))
    for name, row in manifest['cells'].items():
        old = plain_manifest['cells'][name]
        for key in ('result', 'params', 'cached', 'storage', 'offset', 'slots'):
            assert row.get(key) == old.get(key), (name, key)
        if name not in manifest.get('object_cells', ()) and name not in bridges:
            prefix = '_cell_' if row.get('direct_input') is not None else '_calc_'
            pattern = r'^cdef inline [^\n]* ' + prefix + re.escape(name) + r'\([^\n]*:\n(?:[ \t]+[^\n]*\n|\n)*'
            before = re.search(pattern, plain_source, re.M)
            after = re.search(pattern, adapted_source, re.M)
            assert before and after and before.group() == after.group(), name
    assert 'failed_parent' in bridges and 'numeric_from_object' in bridges
    # A real compiled caller uses the public ABI, not a cast to the incompatible
    # private long-long signature. The calculation body/cache remains native.
    (artifact/'public_boundary.pyx').write_text(
        'from ' + manifest['package'] + '._kernel cimport _mx_public_category, _mx_public_amount, _mx_public_report, _mx_public_failed_parent, _mx_public_numeric_from_object, _mx_public_caught_failure\n'
        'def category(object owner):\n return _mx_public_category(owner)\n'
        'def amount(object owner, object kind):\n return _mx_public_amount(owner, kind)\n'
        'def report(object owner):\n return _mx_public_report(owner)\n'
        'def failed_parent(object owner, long long n):\n return _mx_public_failed_parent(owner, n)\n'
        'def numeric_from_object(object owner, long long n):\n return _mx_public_numeric_from_object(owner, n)\n'
        'def caught_failure(object owner, long long n):\n return _mx_public_caught_failure(owner, n)\n', encoding='utf-8')
    (artifact/'public_boundary_setup.py').write_text(
        "from setuptools import setup, Extension\nfrom Cython.Build import cythonize\n"
        "setup(packages=[], ext_modules=cythonize([Extension('public_boundary', ['public_boundary.pyx'])], compiler_directives={'language_level':3}))\n", encoding='utf-8')
    import subprocess
    built = subprocess.run([sys.executable, 'public_boundary_setup.py', 'build_ext', '--inplace'],
                           cwd=artifact, capture_output=True, text=True)
    (artifact/'public_boundary_build.log').write_text(built.stdout + built.stderr, encoding='utf-8')
    assert built.returncode == 0, built.stderr[-3000:]
    monkeypatch.syspath_prepend(str(artifact))
    import importlib
    boundary = importlib.import_module('public_boundary')
    from modelx_cython._frame_runtime.prepared_space_inputs import PreparedSpaceInputs
    from modelx_cython._frame_runtime.python_calculator import PythonScope
    native = importlib.import_module(manifest['package'] + '._kernel')
    native_lookups = importlib.import_module(manifest['package'] + '._lookups')
    resources = PreparedSpaceInputs(manifest, native_lookups, spec.initial_tables)
    for i, category in enumerate((_PlanCategory.ONE, _PlanCategory.ZERO)):
        owner = native.Kernel((resources.record(i),))
        scope = PythonScope(artifact, manifest, owner)
        owner.set_python_scope(scope)
        owner.select_scope(0)
        assert boundary.category(owner) is category
        assert boundary.amount(owner, _PlanCategory.ONE) == boundary.amount(owner, 1) == 3.5
        assert boundary.amount(owner, _PlanCategory.ZERO) == boundary.amount(owner, 0) == -1.0
        assert boundary.report(owner) == {'category': category,
            'amount': 3.5 if category is _PlanCategory.ONE else -1.0}
        # These completed native entries are the public entries' cache too.
        assert owner.call('amount', (1,)) == 3.5
        with pytest.raises(ValueError, match='category'):
            boundary.amount(owner, 9)
        assert boundary.amount(owner, 1) == 3.5
        live_flags = scope.namespace['flags']
        live_flags['fail'] = True
        for _ in range(2):
            with pytest.raises(ValueError, match='numeric retry'):
                boundary.failed_parent(owner, 1)
        assert boundary.caught_failure(owner, 2) == {'value': -1.0}
        assert boundary.amount(owner, 1) == 3.5  # Completed child survives.
        live_flags['fail'] = False
        assert boundary.failed_parent(owner, 1) == 4.5
        live_flags['object_fail'] = True
        for _ in range(2):
            with pytest.raises(LookupError, match='object retry'):
                boundary.numeric_from_object(owner, 1)
        live_flags['object_fail'] = False
        assert boundary.numeric_from_object(owner, 1) == 3.5
    # The standard command generated the original public classes and prepared
    # context together. Do not reapply shell transformations to completed source.
    package = tmp_path/'model_cy'
    source_file, declarations_file = package/'_mx_classes.py', package/'_mx_classes.pxd'
    public_source = source_file.read_text()
    # Test-only access to an internal owner; no extra product API is emitted.
    # Whole-model pickle with module-valued public refs is separately limited
    # by the original shell's automatic Cython serialization.
    import ast
    public_tree = ast.parse(public_source)
    for cls in public_tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == space.class_name:
            cls.body.extend(ast.parse('def _mx_test_owner(self):\n return self._mx_frame_owner\n'
                                      'def _mx_test_resources(self):\n return self._mx_prepared_resources\n').body)
    public_source = ast.unparse(ast.fix_missing_locations(public_tree)) + '\n'
    source_file.write_text(public_source, encoding='utf-8')
    original_build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
        cwd=tmp_path, capture_output=True, text=True)
    (tmp_path/'original_prepared_build.log').write_text(original_build.stdout+original_build.stderr, encoding='utf-8')
    assert original_build.returncode == 0, original_build.stderr[-5000:]
    # Both extensions are in their existing package locations. No alternative
    # CompiledModel or ConnectedModel participates in these original calls.
    import os
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join((str(artifact), str(tmp_path),
        str(Path(__file__).parent), environment.get('PYTHONPATH','')))
    verify_source = '''
import faulthandler
faulthandler.dump_traceback_later(15)
print('loading original package', flush=True)
from model_cy import mx_model
print('selecting original policy', flush=True)
p = mx_model.Projection[0]
assert p.shallow(2) == 1.0
assert p._mx_test_owner()._scope is None
assert p.native_report() == {'value': 1.0}
assert p.native_report() is p.native_report()
assert p._mx_test_owner()._scope is None
completed_before_objects = p._mx_test_owner().kernel.snapshot()
assert completed_before_objects
print('entering original category', flush=True)
assert p.category() is p.Kind.ONE
assert p._mx_test_owner()._scope is not None
assert all(entry in p._mx_test_owner().kernel.snapshot() for entry in completed_before_objects)
print('entering original amount', flush=True)
assert p.amount(p.Kind.ONE) == p.amount(1) == 3.5
assert p.report() == {'category': p.Kind.ONE, 'amount': 3.5}
q = mx_model.Projection[1]
assert q.category() is q.Kind.ZERO
assert q.report() == {'category': q.Kind.ZERO, 'amount': -1.0}
assert p.category() is p.Kind.ONE and p.report()['amount'] == 3.5
from model_cy._mx_frame_owner import SpaceFrameOwner
created = []
original_init = SpaceFrameOwner.__init__
def count_owner(self, *args):
    original_init(self, *args)
    created.append(self)
SpaceFrameOwner.__init__ = count_owner
collected = mx_model.Projection.collect()
assert collected.sum.total == 2.5
assert collected.individual[0].record == {'category': p.Kind.ONE, 'amount': 3.5}
assert collected.individual[1].record == {'category': p.Kind.ZERO, 'amount': -1.0}
assert len(created) == 1
assert mx_model.Projection.collect() is collected
assert p.report()['amount'] == 3.5 and q.report()['amount'] == -1.0
import pickle
restored_owner = pickle.loads(pickle.dumps(p._mx_test_owner()))
assert restored_owner.kernel.call('amount', (1,)) == 3.5
assert restored_owner.kernel.call('amount', (0,)) == -1.0
assert restored_owner.kernel.call('report', ()) == p.report()
assert restored_owner.kernel.call('shallow', (2,)) == 1.0
restored_resources = pickle.loads(pickle.dumps(mx_model.Projection._mx_test_resources()))
assert len(restored_resources) == 1
prepared, positions, data_view = next(iter(restored_resources.values()))
assert positions == {0: 0, 1: 1}
assert prepared.record(0) == p._mx_test_owner().inputs
assert data_view.points().equals(mx_model.Data.points())
print('checking derived receiver', flush=True)
class Derived(type(p)):
    def amount(self, kind=0): return 12.0
template = Derived(mx_model)
template._mx_copy_refs(mx_model.Projection, mx_model.Projection)
r = template[0]
assert r.report() == {'category': r.Kind.ONE, 'amount': 12.0}
print('original public classes enter prepared kernels; subclass dispatch preserved')
'''
    (tmp_path/'verify_original_prepared.py').write_text(verify_source, encoding='utf-8')
    verified = subprocess.run([sys.executable, 'verify_original_prepared.py'], cwd=tmp_path,
                              env=environment, capture_output=True, text=True, timeout=60)
    (tmp_path/'original_prepared_verify.log').write_text(verified.stdout+verified.stderr, encoding='utf-8')
    assert verified.returncode == 0, verified.stderr[-5000:]
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
