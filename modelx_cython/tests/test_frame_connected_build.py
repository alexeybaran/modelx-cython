"""Collect-connected compilation is selected from source metadata, not names."""
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from modelx_cython._frame_runtime.connected_build import (
    build_collect_package, plan_collect_regions)
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


def test_collect_regions_reject_missing_input_and_cycles():
    with pytest.raises(ContractError, match='Unbound external Space'):
        plan_collect_regions(_graph())
    graph = _graph()
    graph['spaces']['Example.Policies']['space_references']['Report'] = 'Example.Report'
    with pytest.raises(ContractError, match='Cyclic'):
        plan_collect_regions(graph, external_names=('Inputs',))


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
        yield lambda destination, **options: fake_frontend(package, destination, **options)
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
