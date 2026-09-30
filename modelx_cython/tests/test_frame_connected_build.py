"""Collect-connected compilation is selected from source metadata, not names."""
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from modelx_cython._frame_runtime.connected_build import (
    build_collect_package, plan_collect_regions, plan_package, build_package)
from modelx_cython._frame_runtime.contracts import ContractError


def _graph():
    def node(refs=(), *, collect=False):
        return {'cells': {'value': 'source'}, 'space_references': {
            name: 'Example.' + name for name in refs},
            'collect': {'over': 'policy_id'} if collect else None}
    return {'schema': 'mxcf.exported_space_graph.v1', 'model': 'Example',
            'spaces': {'Example.Inputs': node(),
                       'Example.Policies': node(('Inputs',), collect=True),
                       'Example.Scenarios': node(('Inputs',), collect=True),
                       'Example.Report': node(('Policies', 'Scenarios')),
                       'Example.Unrelated': node()}}


def test_collect_regions_include_all_roots_and_consumers_in_dependency_order():
    plan = plan_collect_regions(_graph(), external_names=('Inputs',))
    assert plan['collect_roots'] == ['Policies', 'Scenarios']
    assert set(plan['compiled_spaces']) == {'Policies', 'Scenarios', 'Report'}
    assert plan['compiled_spaces'][-1] == 'Report'
    assert plan['external_spaces'] == ['Inputs']
    assert 'Unrelated' not in plan['compiled_spaces']


def test_collect_regions_reject_missing_input_and_plan_reference_cycles():
    with pytest.raises(ContractError, match='Unbound external Space'):
        plan_collect_regions(_graph())
    graph = _graph()
    graph['spaces']['Example.Policies']['space_references']['Report'] = 'Example.Report'
    plan = plan_collect_regions(graph, external_names=('Inputs',))
    assert set(plan['compiled_spaces']) == {'Policies', 'Scenarios', 'Report'}


def test_complete_package_uses_same_graph_without_collect_declarations():
    graph = _graph()
    for node in graph['spaces'].values():
        node['collect'] = None
    plan = plan_package(graph)
    assert plan['collect_roots'] == []
    assert set(plan['compiled_spaces']) == {
        'Inputs', 'Policies', 'Scenarios', 'Report', 'Unrelated'}
    assert plan['compiled_spaces'].index('Inputs') < plan['compiled_spaces'].index('Policies')
    assert plan['compiled_spaces'].index('Policies') < plan['compiled_spaces'].index('Report')
    assert plan['external_spaces'] == []


def test_one_build_invocation_uses_discovered_spaces_not_option_keys(tmp_path, monkeypatch):
    from modelx_cython._frame_runtime import compiler, connected_package
    from modelx_cython._frame_runtime import modelx_frontend_exported

    source = 'def value():\n    return 1\n'
    def space(references, collect=None):
        return {'parameters': [], 'references': list(references),
                'cells': {'value': {'source': source,
                                    'sha256': hashlib.sha256(source.encode()).hexdigest()}},
                'formula': None, 'collect': collect}
    exported = tmp_path / 'exported'
    exported.mkdir()
    (exported / '_mx_formula_sources.json').write_text(json.dumps({
        'schema': 'modelx.formula_sources.v2', 'model': 'OtherModel',
        'spaces': {
            'OtherModel.Inputs': space(()),
            'OtherModel.Policies': space(('Inputs',),
                {'over': 'policy_id', 'domain': {'kind': 'reference', 'name': 'ids'}}),
            'OtherModel.Total': space(('Policies',)),
            'OtherModel.Unrelated': space(()),
        }}))
    generated = []
    sessions = []
    def fake_frontend(package, destination, **options):
        generated.append(('frontend', options['projection'], options.get('steps')))
        return SimpleNamespace(projection=options['projection'])
    def fake_build(spec, artifact):
        generated.append(('build', spec.projection))
        artifact.mkdir(parents=True)
    def fake_assemble(package, graph, artifacts, **kwargs):
        generated.append(('assemble', sorted(artifacts)))
        return {'native': {}}
    @contextmanager
    def fake_source(package):
        sessions.append(package)
        def derive(destination, **options):
            return fake_frontend(package, destination, **options)
        from modelx_cython._frame_runtime.source_graph import describe_source_graph
        derive.source_graph = describe_source_graph(json.loads(
            (package / '_mx_formula_sources.json').read_text()))
        yield derive
    monkeypatch.setattr(modelx_frontend_exported, 'open_exported_source', fake_source)
    monkeypatch.setattr(compiler, 'build', fake_build)
    monkeypatch.setattr(connected_package, 'assemble_connected_package', fake_assemble)
    output = tmp_path / 'output'
    result = build_collect_package(exported, output,
        external_imports={'Inputs': 'provider:inputs'},
        space_options={'Policies': {'steps': 3}}, package_name='CompiledOther')
    assert result['plan']['compiled_spaces'] == ['Policies', 'Total']
    assert generated == [('frontend', 'Policies', 3), ('build', 'Policies'),
                         ('frontend', 'Total', None), ('build', 'Total'),
                         ('assemble', ['Policies', 'Total'])]
    assert result['package'] == output / 'CompiledOther'
    assert sessions == [exported]
    assert json.loads((output / 'build_plan.json').read_text())['collect_roots'] == ['Policies']


def test_complete_package_builds_single_space_without_collect(tmp_path, monkeypatch):
    from modelx_cython._frame_runtime import compiler, connected_package
    from modelx_cython._frame_runtime import modelx_frontend_exported

    source = 'def value():\n    return 1\n'
    exported = tmp_path / 'exported'
    exported.mkdir()
    (exported / '_mx_formula_sources.json').write_text(json.dumps({
        'schema': 'modelx.formula_sources.v2', 'model': 'Single',
        'spaces': {'Single.Space': {
            'parameters': [], 'references': [], 'formula': None,
            'cells': {'value': {'source': source,
                                'sha256': hashlib.sha256(source.encode()).hexdigest()}},
        }}}))
    called = []
    @contextmanager
    def fake_source(_package):
        def derive(destination, **options):
            return SimpleNamespace(projection=options['projection'])
        from modelx_cython._frame_runtime.source_graph import describe_source_graph
        derive.source_graph = describe_source_graph(json.loads(
            (_package / '_mx_formula_sources.json').read_text()))
        yield derive
    def fake_build(spec, artifact):
        called.append(spec.projection)
        artifact.mkdir(parents=True)
    monkeypatch.setattr(modelx_frontend_exported, 'open_exported_source', fake_source)
    monkeypatch.setattr(compiler, 'build', fake_build)
    monkeypatch.setattr(connected_package, 'assemble_connected_package',
                        lambda *args, **kwargs: {})
    result = build_package(exported, tmp_path / 'out', external_imports={})
    assert called == ['Space']
    assert result['plan']['compiled_spaces'] == ['Space']
    assert result['plan']['collect_roots'] == []


@pytest.mark.parametrize('preserve_metadata', [True, False])
def test_complete_package_compiles_ordinary_single_space(tmp_path, preserve_metadata):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_build import build_package
    from modelx_cython._frame_runtime.connected_model import load_connected

    model = mx.new_model('OrdinaryPackage')
    try:
        model.new_space('Projection').new_cells('value', 'lambda: 4.25')
        exported = tmp_path / 'exported'
        model.export(exported)
        if not preserve_metadata:
            (exported / '_mx_formula_sources.json').unlink()
    finally:
        model.close()
    result = build_package(exported, tmp_path / 'compiled', external_imports={})
    assert result['plan']['collect_roots'] == []
    connected = load_connected(
        tmp_path / 'compiled' / '_sources' / 'Projection' / 'source_graph.json',
        result['artifacts'], external_bindings={})
    try:
        assert connected.Projection.value() == 4.25
    finally:
        connected.close()


def test_complete_package_wires_real_cyclic_space_references(tmp_path):
    import modelx as mx
    from modelx_cython._frame_runtime.connected_build import build_package
    from modelx_cython._frame_runtime.connected_model import load_connected

    model = mx.new_model('CyclicSpaces')
    try:
        first = model.new_space('First')
        second = model.new_space('Second')
        first.Other = second
        second.Other = first
        first.new_cells('value', 'lambda: Other.value()')
        second.new_cells('value', 'lambda: 7')
        second.new_cells('via_first', 'lambda: Other.value()')
        exported = tmp_path / 'cyclic_export'
        model.export(exported)
    finally:
        model.close()
    result = build_package(
        exported, tmp_path / 'cyclic_compiled', external_imports={},
        space_options={'First': {'object_bindings': ('Other',)},
                       'Second': {'object_bindings': ('Other',)}})
    graph = tmp_path / 'cyclic_compiled' / '_sources' / 'First' / 'source_graph.json'
    connected = load_connected(graph, result['artifacts'], external_bindings={})
    try:
        assert connected.First.value() == 7
        assert connected.Second.via_first() == 7
        assert connected.First.Other is connected.Second
        assert connected.Second.Other is connected.First
    finally:
        connected.close()
