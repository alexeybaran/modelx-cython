"""Wire separately compiled Spaces from their source-verified reference graph.

The frame frontend currently builds one flat Space per artifact. This loader
keeps those artifacts independent while giving their original formulas the
compiled Space references declared by the exported model's source graph.
External helpers and uncompiled Spaces are explicit inputs; no formula is
rewritten or interpreted here.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from .contracts import ContractError
from .model import CompiledModel
from .source_graph import generated_space_name


class _SpaceView:
    """Expose nested children and base parameter references on a compiled Space."""

    def __init__(self, space, children, base_values=None):
        object.__setattr__(self, '_space', space)
        object.__setattr__(self, '_children', children)
        object.__setattr__(self, '_base_values', dict(base_values or {}))

    def __getattr__(self, name):
        if name in self._children:
            return self._children[name]
        if name in self._base_values:
            return self._base_values[name]
        if (self._base_values and name in self._space._model.manifest['cells']
                and set(self._space.parameters) == set(self._base_values)):
            return getattr(self._space(**self._base_values), name)
        return getattr(self._space, name)

    def __setattr__(self, name, value):
        setattr(self._space, name, value)

    def __call__(self, *args, **kwargs):
        return self._space(*args, **kwargs)

    def __getitem__(self, key):
        return self._space[key]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ConnectedModel:
    """Own compiled Space artifacts and expose their ordinary Space entries."""

    def __init__(self, name, models, spaces, order):
        self.name = name
        self._models = models
        self._spaces = spaces
        self._order = order

    def __getattr__(self, name):
        try:
            return self._spaces[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def close(self):
        for name in reversed(self._order):
            self._models[name].close()


def load_connected(graph, artifacts, *, external_bindings, tables_by_space=None):
    """Load compiled Space artifacts and bind their declared dependencies.

    ``artifacts`` and ``tables_by_space`` use public Space names. An omitted
    Space in the source graph must be supplied through ``external_bindings``.
    Each artifact's original Cell hashes must match the graph before loading.
    The return value exposes the same Space names and Cell entry points as the
    exported model. It owns the compiled artifacts and must be closed.
    """
    if isinstance(graph, (str, Path)):
        graph = json.loads(Path(graph).read_text(encoding='utf-8'))
    if graph.get('schema') != 'mxcf.exported_space_graph.v1':
        raise ContractError('Expected a verified exported Space graph')
    model_name = graph.get('model')
    if not isinstance(model_name, str) or not model_name:
        raise ContractError('Missing graph model identity')
    artifacts = {str(name): Path(path) for name, path in artifacts.items()}
    external_bindings = dict(external_bindings)
    tables_by_space = dict(tables_by_space or {})
    if not artifacts or set(artifacts) & set(external_bindings):
        raise ContractError('Compiled and external Space names must be distinct')
    if set(tables_by_space) - set(artifacts):
        raise ContractError('Tables supplied for an uncompiled Space')
    graph_spaces = graph.get('spaces', {})
    if not isinstance(graph_spaces, dict):
        raise ContractError('Invalid exported Space graph')
    nodes = {}
    for name in artifacts:
        generated_space_name(name)
        fullname = model_name + '.' + name
        if fullname not in graph_spaces:
            raise ContractError('Compiled Space is absent from source graph: ' + name)
        nodes[name] = graph_spaces[fullname]
        base_values = nodes[name].get('base_parameter_values', {})
        if (not isinstance(base_values, dict)
                or set(base_values) - set(nodes[name].get('parameters', ()))
                or any(type(value) not in (type(None), bool, int, float, str)
                       for value in base_values.values())):
            raise ContractError('Invalid base parameter references: ' + name)

    def target_name(ref, target):
        # The exported identity, not the local alias, names the target Space.
        prefix = model_name + '.'
        if (not isinstance(target, str) or not target.startswith(prefix)
                or target not in graph_spaces):
            raise ContractError('Invalid source Space edge: ' + str(target))
        return target[len(prefix):]

    order = []
    visiting = set()
    visited = set()

    def visit(name):
        if name in visited:
            return
        if name in visiting:
            # Only reference wiring cycles are allowed here. Recursive Cell
            # evaluation remains guarded by each compiled cache entry.
            return
        visiting.add(name)
        for ref, target in nodes[name]['space_references'].items():
            dependency = target_name(ref, target)
            if dependency in artifacts:
                visit(dependency)
            elif dependency not in external_bindings:
                raise ContractError('Unbound source Space: ' + target)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in sorted(artifacts):
        visit(name)

    manifests = {}
    for name, artifact in artifacts.items():
        manifest = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
        plan = manifest.get('model', {})
        if (plan.get('projection') != generated_space_name(name)
                or plan.get('source_space', model_name + '.' + name)
                != model_name + '.' + name
                or manifest.get('public_api', {}).get('model_name') != model_name):
            raise ContractError('Compiled Space identity differs from source graph: ' + name)
        source_cells = nodes[name]['cells']
        cells = manifest.get('cells', {})
        if set(cells) != set(source_cells):
            raise ContractError('Compiled Cell inventory differs from source graph: ' + name)
        for cell_name, digest in source_cells.items():
            if cells[cell_name].get('coverage', {}).get('original_source_sha256') != digest:
                raise ContractError('Compiled Cell source differs from graph: '
                                    + name + '.' + cell_name)
        manifests[name] = manifest

    models = {}
    spaces = dict(external_bindings)
    try:
        # Construct every real Space before connecting any references. This
        # preserves Space identity even when A refers to B and B refers to A.
        for name in order:
            artifact = artifacts[name]
            manifest = manifests[name]
            plan = manifest['model']
            if name in tables_by_space:
                tables = tables_by_space[name]
            else:
                public = manifest.get('public_api', {})
                if public.get('schema') != 'mxcf.package.v1':
                    raise ContractError('Compiled Space has no embedded Data: ' + name)
                path = artifact / public['tables_file']
                if _sha(path) != public['tables_sha256']:
                    raise ContractError('Embedded Data hash mismatch: ' + name)
                tables = pickle.loads(path.read_bytes())
            model = CompiledModel(artifact, tables, _defer_bindings=True)
            models[name] = model
            spaces[name] = getattr(model, plan['projection'])
        children = {}
        for name in artifacts:
            if '.' in name:
                parent, leaf = name.rsplit('.', 1)
                if parent not in artifacts:
                    raise ContractError('Nested Space parent is absent: ' + parent)
                children.setdefault(parent, {})[leaf] = name
        for name in sorted(artifacts, key=lambda value: value.count('.'), reverse=True):
            base_values = nodes[name].get('base_parameter_values', {})
            if name in children or base_values:
                spaces[name] = _SpaceView(spaces[name], {
                    leaf: spaces[child] for leaf, child in children.get(name, {}).items()},
                    base_values)
        for name in order:
            plan = manifests[name]['model']
            refs = {ref: spaces[target_name(ref, target)]
                    for ref, target in nodes[name]['space_references'].items()}
            def resolve(ref):
                if ref in refs:
                    return refs[ref]
                try:
                    return external_bindings[ref]
                except KeyError as exc:
                    raise ContractError('Unbound connected reference: '
                                        + name + '.' + ref) from exc
            formula = plan.get('space_formula')
            formula_refs = (formula.get('references', ()) if formula is not None else ())
            space_bindings = {ref: resolve(ref) for ref in formula_refs}
            overrides = {ref: resolve(ref)
                         for ref in plan.get('required_object_bindings', ())}
            models[name]._finish_bindings(overrides=overrides,
                                          space_bindings=space_bindings or None)
        return ConnectedModel(model_name, models, spaces, order)
    except BaseException:
        for name in reversed(order):
            if name in models:
                models[name].close()
        raise
