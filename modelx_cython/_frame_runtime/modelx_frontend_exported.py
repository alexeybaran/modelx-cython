"""Feed original modelx formula sources from an exported package to the frame emitter.

The exporter preserves the author sources before rewriting them as Python
methods. This adapter reads that metadata and the exported model's actual
references/Data values; it does not reverse-engineer rewritten methods.
"""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

from .contracts import ContractError
from .modelx_frontend_native_mode import from_modelx


class _SourceCell:
    def __init__(self, instance, name, info):
        self.formula = SimpleNamespace(source=info['source'])
        self.parameters = tuple(inspect.signature(getattr(instance, name)).parameters)
        self.is_cached = info['cached']
        self.allow_none = info['allow_none']
        self._method = getattr(instance, name)

    def __call__(self, *args, **kwargs):
        return self._method(*args, **kwargs)


class _CollectDomain:
    def __init__(self, instance, source):
        self._instance = instance
        self._source = source

    def value(self, _impl):
        value = getattr(self._instance, self._source['name'])
        return value() if self._source['kind'] == 'cell' else value

    @property
    def ref_name(self):
        return self._source['name'] if self._source['kind'] == 'cell' else None


class _SourceSpace:
    def __init__(self, instance, info):
        self.parameters = tuple(info['parameters'])
        self.spaces = getattr(instance, '_mx_spaces', {})
        self.refs = {name: getattr(instance, name) for name in info['references']}
        self.cells = {name: _SourceCell(instance, name, cell)
                      for name, cell in info['cells'].items()}
        self.formula = (SimpleNamespace(func=instance.__call__)
                        if info['formula'] is not None else None)
        collect = info.get('collect')
        self._impl = SimpleNamespace(collect_over_spec=(
            {collect['over']: _CollectDomain(instance, collect['domain'])}
            if collect is not None else None))


def _load_export(path):
    package = path.resolve()
    # A fresh name prevents Python's import cache from mixing two revisions of
    # the same export path during a development session.
    alias = '_mxcf_export_' + uuid4().hex
    spec = importlib.util.spec_from_file_location(
        alias, package / '__init__.py',
        submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ContractError('Exported model is not an importable package')
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in tuple(sys.modules):
            if name == alias or name.startswith(alias + '.'):
                del sys.modules[name]
        raise
    return alias, module.mx_model


def from_exported(package, destination, **options):
    """Derive one frame spec from the standard modelx export source handoff.

    The supported model subset remains the one declared by ``from_modelx``.
    Missing or mismatched source metadata is a build error, never permission to
    substitute the Python-export methods as different formula bodies.
    """
    path = Path(package)
    manifest_path = path / '_mx_formula_sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'modelx.formula_sources.v2':
        raise ContractError('Unsupported modelx formula-source manifest')
    for fullname, space in manifest['spaces'].items():
        for name, cell in space['cells'].items():
            digest = hashlib.sha256(cell['source'].encode('utf-8')).hexdigest()
            if digest != cell['sha256']:
                raise ContractError('Formula source hash mismatch: ' + fullname + '.' + name)
    alias, exported = _load_export(path)
    try:
        if exported._name != manifest['model']:
            raise ContractError('Exported model identity differs from formula sources')
        spaces = {}
        for fullname, info in manifest['spaces'].items():
            parts = fullname.split('.')
            if parts[0] != manifest['model']:
                raise ContractError('Formula source Space belongs to another model')
            instance = exported
            for name in parts[1:]:
                instance = getattr(instance, name)
            spaces[tuple(parts[1:])] = _SourceSpace(instance, info)
        proxy = SimpleNamespace(name=manifest['model'])
        projection = options.get('projection', 'Projection')
        data_space = options.get('data_space', 'Data')
        for name in (projection, data_space):
            if (name,) in spaces:
                setattr(proxy, name, spaces[(name,)])
        return from_modelx(proxy, destination, **options)
    finally:
        for name in tuple(sys.modules):
            if name == alias or name.startswith(alias + '.'):
                del sys.modules[name]
