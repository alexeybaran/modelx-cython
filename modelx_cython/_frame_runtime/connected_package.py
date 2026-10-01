"""Assemble and load one importable package from verified frame artifacts."""
from __future__ import annotations

import hashlib
import importlib
import json
import keyword
from pathlib import Path
import shutil

from .connected_model import load_connected
from .contracts import ContractError
from .source_graph import generated_space_name, is_empty_space_formula


_SCHEMA = 'mxcf.connected_package.v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provider_value(reference):
    if not isinstance(reference, str) or reference.count(':') != 1:
        raise ContractError('External provider must be module:attribute')
    module, attribute = reference.split(':')
    if (not module or not attribute
            or any(not part.isidentifier() for part in module.split('.'))
            or any(not part.isidentifier() for part in attribute.split('.'))):
        raise ContractError('Invalid external provider: ' + reference)
    value = importlib.import_module(module)
    for part in attribute.split('.'):
        value = getattr(value, part)
    return value


def assemble_connected_package(destination, graph, artifacts, *,
                               shells=(), external_imports=None, eager=False):
    """Copy compiled Space runtime files into a fresh importable package.

    External providers are explicit module:attribute references. ``eager``
    exposes a modelx-style ``mx_model`` at import time when those providers are
    available; ``load(external_bindings=...)`` is always available. The source
    graph and native binaries are hashed and checked again on every load.
    """
    destination = Path(destination)
    if (not destination.name.isidentifier() or keyword.iskeyword(destination.name)
            or destination.exists()):
        raise ContractError('Connected destination must be a fresh package directory')
    graph_bytes = (Path(graph).read_bytes() if isinstance(graph, (str, Path))
                   else (json.dumps(graph, indent=2) + '\n').encode('utf-8'))
    graph_data = json.loads(graph_bytes)
    if graph_data.get('schema') != 'mxcf.exported_space_graph.v1':
        raise ContractError('Expected a verified exported Space graph')
    model_name = graph_data.get('model')
    if (not isinstance(model_name, str) or not model_name.isidentifier()
            or keyword.iskeyword(model_name)):
        raise ContractError('Connected model name must be an identifier')
    artifacts = {str(name): Path(path) for name, path in artifacts.items()}
    shells = sorted(set(shells))
    if not artifacts:
        raise ContractError('Connected package needs Space artifacts')
    physical_names = {name: generated_space_name(name) for name in artifacts}
    if len(set(physical_names.values())) != len(physical_names):
        raise ContractError('Connected Space artifact names collide')
    external_imports = dict(external_imports or {})
    if any(not name.isidentifier() for name in external_imports):
        raise ContractError('External binding names must be identifiers')
    if eager and not external_imports:
        raise ContractError('Eager connected loading requires external providers')
    for reference in external_imports.values():
        if (not isinstance(reference, str) or reference.count(':') != 1
                or any(not part.isidentifier() for part in
                       reference.replace(':', '.').split('.'))):
            raise ContractError('Invalid external provider: ' + repr(reference))

    destination.mkdir(parents=True)
    (destination / 'source_graph.json').write_bytes(graph_bytes)
    entries = {}
    native = {}
    for name, original in sorted(artifacts.items()):
        if model_name + '.' + name not in graph_data['spaces']:
            raise ContractError('Artifact Space missing from source graph: ' + name)
        manifest_path = original / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        private = manifest.get('package')
        plan = manifest.get('model', {})
        if (not isinstance(private, str) or not private.isidentifier()
                or plan.get('projection') != physical_names[name]
                or plan.get('source_space', model_name + '.' + name)
                != model_name + '.' + name):
            raise ContractError('Artifact identity differs from connected Space: ' + name)
        current = destination / physical_names[name]
        current.mkdir()
        for filename in ('manifest.json', '_mx_tables.pkl'):
            shutil.copy2(original / filename, current / filename)
        shutil.copytree(original / private, current / private,
                        ignore=shutil.ignore_patterns('__pycache__'))
        entries[name] = {'path': physical_names[name],
                         'manifest_sha256': _sha(current / 'manifest.json'),
                         'tables_sha256': _sha(current / '_mx_tables.pkl')}
        for binary in sorted((current / private).rglob('*.pyd')):
            native[str(binary.relative_to(destination)).replace('\\', '/')] = _sha(binary)
    if not native:
        raise ContractError('Connected package contains no native extensions')
    for name in shells:
        node = graph_data['spaces'].get(model_name + '.' + name)
        if (node is None or name in artifacts or node['cells']
                or not is_empty_space_formula(node.get('formula'))):
            raise ContractError('Invalid Cell-free Space shell: ' + name)
    package_manifest = {
        'schema': _SCHEMA, 'model': model_name,
        'source_graph_sha256': _sha(destination / 'source_graph.json'),
        'artifacts': entries, 'shells': shells, 'native': native,
        'external_imports': external_imports, 'eager': bool(eager),
    }
    (destination / 'connected_manifest.json').write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    init = '''"""Generated connected modelx-cython package."""\nfrom modelx_cython._frame_runtime.connected_package import load_connected_package as _open\n\ndef load(*, external_bindings=None, tables_by_space=None):\n    return _open(__file__, external_bindings=external_bindings, tables_by_space=tables_by_space)\n'''
    if eager:
        init += f'\nmx_model = {model_name} = load()\n'
        init += f'__all__ = ["load", "mx_model", {model_name!r}]\n'
    else:
        init += '__all__ = ["load"]\n'
    (destination / '__init__.py').write_text(init, encoding='utf-8')
    return package_manifest


def load_connected_package(package_file, *, external_bindings=None,
                           tables_by_space=None):
    """Verify copied artifacts and resolve explicit providers before wiring."""
    directory = Path(package_file).resolve().parent
    package_manifest = json.loads((directory / 'connected_manifest.json').read_text(
        encoding='utf-8'))
    if package_manifest.get('schema') != _SCHEMA:
        raise ContractError('Unsupported connected package manifest')
    graph = directory / 'source_graph.json'
    if _sha(graph) != package_manifest['source_graph_sha256']:
        raise ContractError('Connected source graph hash mismatch')
    artifacts = {}
    for name, entry in package_manifest['artifacts'].items():
        artifact = directory / entry['path']
        if (artifact.resolve().parent != directory
                or _sha(artifact / 'manifest.json') != entry['manifest_sha256']
                or _sha(artifact / '_mx_tables.pkl') != entry['tables_sha256']):
            raise ContractError('Connected artifact hash mismatch: ' + name)
        artifacts[name] = artifact
    for relative, digest in package_manifest['native'].items():
        binary = directory / relative
        if not binary.resolve().is_relative_to(directory) or _sha(binary) != digest:
            raise ContractError('Connected native hash mismatch: ' + relative)
    supplied = dict(external_bindings or {})
    external = {name: _provider_value(provider) for name, provider in
                package_manifest['external_imports'].items()
                if name not in supplied}
    external.update(supplied)
    options = {'external_bindings': external, 'tables_by_space': tables_by_space}
    if package_manifest.get('shells'):
        options['shells'] = package_manifest['shells']
    return load_connected(graph, artifacts, **options)
