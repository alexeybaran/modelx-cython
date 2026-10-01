"""The connected frame loader follows verified source edges, not name guesses."""
import json
from types import SimpleNamespace

import pytest

from modelx_cython._frame_runtime import connected_model
from modelx_cython._frame_runtime.contracts import ContractError


def _fixture(tmp_path, monkeypatch):
    graph = {
        'schema': 'mxcf.exported_space_graph.v1', 'model': 'Example',
        'spaces': {
            'Example.A': {'cells': {'a': 'sha-a'},
                          'space_references': {'B': 'Example.B', 'C': 'Example.C'}},
            'Example.B': {'cells': {'b': 'sha-b'},
                          'space_references': {'C': 'Example.C'}},
            'Example.C': {'cells': {}, 'space_references': {}},
        },
    }
    artifacts = {}
    for name, digest, formula, objects in (
            ('A', 'sha-a', ['B', 'C'], ['B', 'helper']),
            ('B', 'sha-b', ['C'], [])):
        artifact = tmp_path / name
        artifact.mkdir()
        (artifact / 'manifest.json').write_text(json.dumps({
            'model': {'projection': name,
                      'space_formula': {'references': formula},
                      'required_object_bindings': objects},
            'public_api': {'model_name': 'Example'},
            'cells': {name.lower(): {'coverage': {
                'original_source_sha256': digest}}},
        }))
        artifacts[name] = artifact
    created = []

    class FakeModel:
        def __init__(self, artifact, tables, *, _defer_bindings):
            assert _defer_bindings
            self.name = artifact.name
            self._plan = json.loads((artifact / 'manifest.json').read_text())['model']
            self.tables = tables
            self.overrides = self.bindings = None
            self.space = SimpleNamespace(name=self.name)
            self.closed = False
            created.append(self)

        def _finish_bindings(self, *, overrides, space_bindings,
                             relative_references=None):
            self.overrides = overrides
            self.bindings = space_bindings
            self.relative_references = relative_references

        def __getattr__(self, name):
            if name == self.name:
                return self.space
            raise AttributeError(name)

        def close(self):
            self.closed = True

    monkeypatch.setattr(connected_model, 'CompiledModel', FakeModel)
    return graph, artifacts, created


def test_connected_loader_wires_source_edges_and_closes_reverse_order(tmp_path, monkeypatch):
    graph, artifacts, created = _fixture(tmp_path, monkeypatch)
    outside = SimpleNamespace(name='C')
    helper = object()
    model = connected_model.load_connected(
        graph, artifacts, external_bindings={'C': outside, 'helper': helper},
        tables_by_space={'A': {'a': 1}, 'B': {'b': 2}})
    assert [item.name for item in created] == ['B', 'A']
    assert model.A is created[1].space
    assert model.B is created[0].space
    assert model.C is outside
    assert created[0].bindings == {'C': outside}
    assert created[1].bindings == {'B': model.B, 'C': outside}
    assert created[1].overrides == {'B': model.B, 'helper': helper}
    model.close()
    assert all(item.closed for item in created)


def test_connected_loader_wires_alias_to_exported_target(tmp_path, monkeypatch):
    graph, artifacts, created = _fixture(tmp_path, monkeypatch)
    graph['spaces']['Example.A']['space_references'] = {
        'Alias': 'Example.B', 'C': 'Example.C'}
    path = artifacts['A'] / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['model']['space_formula']['references'] = ['Alias', 'C']
    manifest['model']['required_object_bindings'] = ['Alias', 'helper']
    path.write_text(json.dumps(manifest))
    outside = SimpleNamespace(name='C')
    model = connected_model.load_connected(
        graph, artifacts, external_bindings={'C': outside, 'helper': object()},
        tables_by_space={'A': {}, 'B': {}})
    try:
        assert created[1].bindings['Alias'] is model.B
        assert created[1].overrides['Alias'] is model.B
    finally:
        model.close()


def test_connected_loader_rejects_stale_source_and_unbound_edges(tmp_path, monkeypatch):
    graph, artifacts, created = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match='Unbound source Space'):
        connected_model.load_connected(
            graph, artifacts, external_bindings={'helper': object()},
            tables_by_space={'A': {}, 'B': {}})
    assert not created
    graph['spaces']['Example.A']['cells']['a'] = 'different'
    with pytest.raises(ContractError, match='Compiled Cell source differs'):
        connected_model.load_connected(
            graph, artifacts,
            external_bindings={'C': object(), 'helper': object()},
            tables_by_space={'A': {}, 'B': {}})
    assert not created
    graph['spaces']['Example.A']['cells']['a'] = 'sha-a'
    graph['spaces']['Example.A']['space_references']['B'] = 'Other.B'
    with pytest.raises(ContractError, match='Invalid source Space edge'):
        connected_model.load_connected(
            graph, artifacts,
            external_bindings={'C': object(), 'helper': object()},
            tables_by_space={'A': {}, 'B': {}})
    assert not created


def test_connected_loader_cleans_up_after_late_missing_binding(tmp_path, monkeypatch):
    graph, artifacts, created = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ContractError, match='Unbound connected reference: A.helper'):
        connected_model.load_connected(
            graph, artifacts, external_bindings={'C': object()},
            tables_by_space={'A': {}, 'B': {}})
    assert [item.name for item in created] == ['B', 'A']
    assert all(item.closed for item in created)


def test_connected_loader_wires_cyclic_spaces_without_proxies(tmp_path, monkeypatch):
    graph, artifacts, created = _fixture(tmp_path, monkeypatch)
    graph['spaces']['Example.B']['space_references']['A'] = 'Example.A'
    path = artifacts['B'] / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['model']['space_formula']['references'] = ['C', 'A']
    path.write_text(json.dumps(manifest))
    model = connected_model.load_connected(
        graph, artifacts, external_bindings={'C': object(), 'helper': object()},
        tables_by_space={'A': {}, 'B': {}})
    try:
        a = next(item for item in created if item.name == 'A')
        b = next(item for item in created if item.name == 'B')
        assert a.bindings['B'] is b.space
        assert b.bindings['A'] is a.space
        assert model.A is a.space
        assert model.B is b.space
    finally:
        model.close()


def test_bound_item_parameter_shadows_base_space_reference():
    space = connected_model._SpaceView(None, {}, base_values={'i': 1},
                                       parameters=('i',))
    assert space.i == 1
    assert space[2].i == 2
    assert space[2] is space[2]
    assert space.i == 1
