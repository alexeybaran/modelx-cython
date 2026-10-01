"""Feed original modelx formula sources from an exported package to the frame emitter.

The exporter preserves the author sources before rewriting them as Python
methods. This adapter reads that metadata and the exported model's actual
references/Data values; it does not reverse-engineer rewritten methods.
"""
import importlib.util
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

from .contracts import ContractError
from .modelx_frontend_native_mode import from_modelx
from .source_graph import describe_source_graph


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


@contextmanager
def open_exported_source(package):
    """Load one verified export for one or several Space compilations."""
    path = Path(package)
    manifest_path = path / '_mx_formula_sources.json'
    manifest = (json.loads(manifest_path.read_text(encoding='utf-8'))
                if manifest_path.exists() else None)
    # Verify a supplied manifest before importing executable package code.
    graph = describe_source_graph(manifest) if manifest is not None else None
    alias, exported = _load_export(path)
    try:
        if manifest is None:
            from .exported_fallback import recover_source_manifest
            manifest = recover_source_manifest(exported)
            graph = describe_source_graph(manifest)
        if exported._name != manifest['model']:
            raise ContractError('Exported model identity differs from formula sources')
        spaces = {}
        for fullname, info in manifest['spaces'].items():
            parts = fullname.split('.')
            instance = exported
            for name in parts[1:]:
                instance = getattr(instance, name)
            source_space = _SourceSpace(instance, info)
            spaces[tuple(parts[1:])] = source_space
            # A base Space may define references with the same names as its
            # dynamic parameters. Its Cells can then be called without first
            # selecting an item (for example Foo.i = 1; Foo.foo()). Preserve
            # those values separately from the parameterized item signature.
            base_values = {}
            for name in set(info['parameters']) & set(info['references']):
                value = source_space.refs[name]
                if type(value) not in (type(None), bool, int, float, str):
                    raise ContractError('Unsupported base parameter reference: '
                                        + fullname + '.' + name)
                base_values[name] = value
            if base_values:
                graph['spaces'][fullname]['base_parameter_values'] = base_values
        def derive(destination, **options):
            dest = Path(destination)
            if dest.exists() and any(dest.iterdir()):
                raise FileExistsError('Frame source destination must be empty')
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'source_graph.json').write_bytes(
                (json.dumps(graph, indent=2) + '\n').encode('utf-8'))
            if manifest.get('source_provenance') == 'reconstructed_export_methods':
                (dest / 'recovered_formula_sources.json').write_text(
                    json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            proxy = SimpleNamespace(name=manifest['model'], source_graph=graph,
                                    spaces=spaces)
            projection = options.get('projection', 'Projection')
            data_space = options.get('data_space', 'Data')
            for name in (projection, data_space):
                if (name,) in spaces:
                    setattr(proxy, name, spaces[(name,)])
            return from_modelx(proxy, dest, **options)

        # Package planning and individual Space compilation must consume the
        # same verified graph. This also covers ordinary exports whose author
        # formula manifest must be reconstructed from exported methods.
        derive.source_graph = graph
        derive.source_manifest = manifest
        yield derive
    finally:
        for name in tuple(sys.modules):
            if name == alias or name.startswith(alias + '.'):
                del sys.modules[name]


def from_exported(package, destination, **options):
    """Derive one frame spec from the verified modelx formula sources."""
    with open_exported_source(package) as derive:
        return derive(destination, **options)
