"""An assembled connected package remains verifiable after relocation."""
import importlib
import json
import shutil
import sys

import pytest

from modelx_cython._frame_runtime import connected_package
from modelx_cython._frame_runtime.contracts import ContractError


def _fixture(tmp_path):
    graph = {
        'schema': 'mxcf.exported_space_graph.v1', 'model': 'Example',
        'spaces': {'Example.A': {'cells': {'a': 'original-a'},
                                 'space_references': {}}},
    }
    source = tmp_path / 'source'
    source.mkdir()
    artifact = source / 'A'
    artifact.mkdir()
    manifest = {
        'model': {'projection': 'A'},
        'public_api': {'model_name': 'Example', 'schema': 'mxcf.package.v1',
                       'tables_file': '_mx_tables.pkl'},
        'package': '_mx_example_a',
        'cells': {'a': {'coverage': {'original_source_sha256': 'original-a'}}},
    }
    (artifact / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (artifact / '_mx_tables.pkl').write_bytes(b'tables')
    private = artifact / '_mx_example_a'
    private.mkdir()
    (private / 'cell.pyd').write_bytes(b'native')
    return graph, artifact


def test_connected_package_imports_after_relocation_and_checks_native_hash(
        tmp_path, monkeypatch):
    graph, artifact = _fixture(tmp_path)
    destination = tmp_path / 'Build' / 'CompiledExample'
    connected_package.assemble_connected_package(destination, graph, {'A': artifact})
    relocated = tmp_path / 'Install' / destination.name
    relocated.parent.mkdir()
    shutil.move(str(destination), str(relocated))
    loaded = []
    sentinel = object()

    def fake_load(source_graph, artifacts, *, external_bindings, tables_by_space):
        loaded.append((source_graph, artifacts, external_bindings, tables_by_space))
        return sentinel

    monkeypatch.setattr(connected_package, 'load_connected', fake_load)
    monkeypatch.syspath_prepend(str(relocated.parent))
    package = importlib.import_module(relocated.name)
    try:
        assert package.load() is sentinel
        assert loaded[0][1] == {'A': relocated / 'A'}
        assert loaded[0][2] == {}
        (relocated / 'A' / '_mx_example_a' / 'cell.pyd').write_bytes(b'changed')
        with pytest.raises(ContractError, match='native hash mismatch'):
            package.load()
        assert len(loaded) == 1
    finally:
        sys.modules.pop(relocated.name, None)


def test_connected_package_rejects_stale_source_graph(tmp_path, monkeypatch):
    graph, artifact = _fixture(tmp_path)
    destination = tmp_path / 'CompiledExample'
    connected_package.assemble_connected_package(destination, graph, {'A': artifact})
    (destination / 'source_graph.json').write_text(json.dumps({**graph, 'model': 'Other'}))
    monkeypatch.setattr(connected_package, 'load_connected',
                        lambda *args, **kwargs: pytest.fail('Loaded stale graph'))
    with pytest.raises(ContractError, match='source graph hash mismatch'):
        connected_package.load_connected_package(destination / '__init__.py')


def test_connected_package_does_not_replace_existing_campaign(tmp_path):
    graph, artifact = _fixture(tmp_path)
    destination = tmp_path / 'CompiledExample'
    connected_package.assemble_connected_package(destination, graph, {'A': artifact})
    with pytest.raises(ContractError, match='fresh package directory'):
        connected_package.assemble_connected_package(destination, graph, {'A': artifact})
